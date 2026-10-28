"""Read a CGM export into a tidy (timestamp, mg/dL) series, as the game loaded it.

Sugar Sugar loads every source through ``cgm-format`` (``sugar_sugar.data.load_glucose_data``),
so the rows it showed a player are that library's glucose rows, sorted by time. Reading the files
any other way shifts row positions and units: D1NAMO keeps mmol/L in separate date and time
columns, and LOOP exports mix EGV, interpolated and calibration rows.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from eliot import start_action

_TS_ALIASES: tuple[str, ...] = (
    "Timestamp (YYYY-MM-DDThh:mm:ss)",
    "Timestamp",
    "timestamp",
    "time",
    "datetime",
    "date",
    "systemTime",
    "displayTime",
)
_GLUCOSE_ALIASES: tuple[str, ...] = (
    "Glucose Value (mg/dL)",
    "Glucose (mg/dL)",
    "glucose",
    "sgv",
    "Glucose Value",
    "historic_glucose_mmol_l",
    "scan_glucose_mmol_l",
    "glucoseValue",
)
_EVENT_ALIASES: tuple[str, ...] = ("Event Type", "event_type", "type")
_MMOL_HINTS: tuple[str, ...] = ("mmol", "mmol_l", "mmol/l")


def _pick(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for alias in aliases:
        if alias in columns:
            return alias
        found = lower.get(alias.lower())
        if found is not None:
            return found
    return None


def _parse_timestamps(series: pl.Series) -> pl.Series:
    text = series.cast(pl.Utf8)
    best: pl.Series | None = None
    best_n = 0
    for fmt in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S%.f",
        "%d-%m-%Y %H:%M",
        "%d-%m-%Y %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%Y-%m-%d",
    ):
        parsed = text.str.to_datetime(fmt, strict=False)
        n_ok = int(parsed.is_not_null().sum())
        if n_ok > best_n:
            best = parsed
            best_n = n_ok
    if best is not None and best_n > 0:
        return best
    return pl.Series(series.name, [None] * series.len(), dtype=pl.Datetime)


def read_glucose_series(path: Path | str) -> pl.DataFrame:
    """Load one CGM file as ``timestamp`` + ``glucose_mgdl``: the rows the game showed.

    Routed like ``sugar_sugar.data.load_glucose_data``: LOOP chronological exports keep
    their EGV and interpolated rows; D1NAMO and BIG IDEAs subject folders are parsed as
    a bundle; everything else goes through ``FormatParser.parse_file``. A file
    cgm-format cannot read falls back to the column-alias reader.
    """
    csv_path = Path(path)
    with start_action(action_type="ai.read_glucose_series", path=str(csv_path)) as action:
        try:
            out = _read_like_the_game(csv_path)
            action.log(message_type="info", reader="cgm-format", n_rows=out.height)
            return out
        except Exception as exc:  # noqa: BLE001 — fall back for formats the library rejects
            action.log(message_type="info", reader="aliases", reason=str(exc))
        return _read_by_aliases(csv_path, action)


def _read_like_the_game(path: Path) -> pl.DataFrame:
    from cgm_format import (
        CGM_SCHEMA_EXTENDED,
        ExtendedFormatProcessor,
        FormatParser,
        FormatProcessor,
    )

    if path.name.lower().endswith("_chronological.csv"):
        return _read_loop_chronological(path)
    if _is_subject_bundle(path):
        unified = FormatParser.parse_subject_directory(path.parent)
    else:
        unified = FormatParser.parse_file(path)
    extended = tuple(unified.columns) == tuple(CGM_SCHEMA_EXTENDED.get_column_names(data_only=False))
    processor = ExtendedFormatProcessor if extended else FormatProcessor
    glucose, _ = processor.split_glucose_events(unified)
    return (
        glucose.filter(pl.col("datetime").is_not_null() & pl.col("glucose").is_not_null())
        .select(
            pl.col("datetime").cast(pl.Datetime("us")).alias("timestamp"),
            pl.col("glucose").cast(pl.Float64).alias("glucose_mgdl"),
        )
        .sort("timestamp", maintain_order=True)
    )


def _is_subject_bundle(path: Path) -> bool:
    """D1NAMO / BIG IDEAs: a folder of files the library parses as one subject."""
    from cgm_format import FormatParser

    try:
        return FormatParser.detect_subject_format(path.parent) is not None
    except Exception:  # noqa: BLE001 — UnknownFormatError, OSError, ValueError
        return False


def _read_loop_chronological(path: Path) -> pl.DataFrame:
    """``sugar_sugar.data.load_loop_chronological_data``: EGV and interpolated rows."""
    raw = pl.read_csv(path, infer_schema_length=10_000)
    glucose_text = pl.col("Glucose (mg/dL)").cast(pl.Utf8, strict=False).str.strip_chars()
    return (
        raw.filter(pl.col("Event Type").is_in(["EGV", "Interpolated"]))
        .filter(pl.col("Glucose (mg/dL)").is_not_null() & (glucose_text != ""))
        .select(
            pl.col("Timestamp").str.to_datetime(strict=False).cast(pl.Datetime("us")).alias("timestamp"),
            glucose_text.cast(pl.Float64, strict=False).alias("glucose_mgdl"),
        )
        .filter(pl.col("timestamp").is_not_null() & pl.col("glucose_mgdl").is_not_null())
        .sort("timestamp", maintain_order=True)
    )


def _read_by_aliases(csv_path: Path, action: Any) -> pl.DataFrame:
    """Column-alias reader for files cgm-format rejects (EGV rows; mmol/L converted)."""
    empty = pl.DataFrame(schema={"timestamp": pl.Datetime, "glucose_mgdl": pl.Float64})
    try:
        raw = pl.read_csv(csv_path, infer_schema_length=20_000, ignore_errors=True)
    except Exception as exc:  # noqa: BLE001 — skip unreadable corpus files
        action.log(message_type="warning", reason=str(exc))
        return empty
    if raw.height == 0:
        return empty

    ts_col = _pick(raw.columns, _TS_ALIASES)
    glucose_col = _pick(raw.columns, _GLUCOSE_ALIASES)
    if ts_col is None and "date" in {c.lower() for c in raw.columns} and "time" in {
        c.lower() for c in raw.columns
    }:
        date_col = _pick(raw.columns, ("date",))
        time_col = _pick(raw.columns, ("time",))
        raw = raw.with_columns(
            (pl.col(date_col).cast(pl.Utf8) + " " + pl.col(time_col).cast(pl.Utf8)).alias(
                "_joined_ts"
            )
        )
        ts_col = "_joined_ts"
    if ts_col is None or glucose_col is None:
        action.log(message_type="warning", reason="missing_columns", columns=raw.columns)
        return empty

    event_col = _pick(raw.columns, _EVENT_ALIASES)
    frame = raw.select(
        [pl.col(ts_col).alias("timestamp_raw"), pl.col(glucose_col).alias("glucose_raw")]
        + ([pl.col(event_col).alias("event_type")] if event_col else [])
    )
    if "event_type" in frame.columns:
        events = frame["event_type"].cast(pl.Utf8).str.to_uppercase()
        keep = events.is_in(["EGV", "CGM", "GLUCOSE", ""]) | events.is_null()
        frame = frame.filter(keep)

    timestamps = _parse_timestamps(frame["timestamp_raw"])
    glucose = frame["glucose_raw"].cast(pl.Float64, strict=False)
    out = pl.DataFrame({"timestamp": timestamps, "glucose_mgdl": glucose}).drop_nulls()
    if out.height == 0:
        return out

    name_says_mmol = any(hint in glucose_col.lower() for hint in _MMOL_HINTS)
    median = float(out["glucose_mgdl"].median() or 0.0)
    if name_says_mmol or median < 30.0:
        out = out.with_columns((pl.col("glucose_mgdl") * 18.0).alias("glucose_mgdl"))
        action.log(message_type="info", unit="mmol_to_mgdl", median_before=median)

    return out.sort("timestamp").unique(subset=["timestamp"], keep="last")


def parse_timestamp(raw: Any) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw
    text = str(raw).strip()
    if not text:
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%dT%H:%M:%S%.f",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None
