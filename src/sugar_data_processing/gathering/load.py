"""Load sugar-sugar ``prediction_statistics.csv`` into typed Polars frames."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
from eliot import start_action

from sugar_data_processing.gathering.encoding import (
    as_strict_bool,
    cgm_duration_to_years,
    parse_duration_years,
    parse_optional_bool,
    parse_per_round_metrics,
    parse_round_context,
)
from sugar_data_processing.gathering.sources import redact_source_name


def _redact_literal_sources(cell: Any, parser: Any) -> str | None:
    try:
        items = parser(cell)
    except (SyntaxError, ValueError, TypeError):
        return None if cell is None else str(cell)
    if not items:
        return str(cell) if cell is not None else None
    cleaned: list[dict[str, Any]] = []
    for item in items:
        row = dict(item)
        if "data_source_name" in row:
            row["data_source_name"] = redact_source_name(row.get("data_source_name"))
        cleaned.append(row)
    return str(cleaned)

#: Games the server rotated out of the live export, kept as dated files beside it.
ARCHIVE_GLOB: str = "archived_*.csv"
#: Column ``read_statistics_csv`` adds when asked: the file each run was read from.
EXPORT_FILE_COLUMN: str = "export_file"


def statistics_csv_paths(path: Path | str) -> list[Path]:
    """The export plus every ``archived_*.csv`` beside it, the export first.

    The archives hold real study games that are not in the live export
    (``archived_2026-08-19.csv`` covers 17–19 August), so every reader of the
    export reads them too.
    """
    main = Path(path)
    archives = sorted(
        candidate
        for candidate in main.parent.glob(ARCHIVE_GLOB)
        if candidate.is_file() and candidate.resolve() != main.resolve()
    )
    return [main, *archives]


def read_statistics_csv(
    path: Path | str,
    *,
    with_origin: bool = False,
    **read_options: Any,
) -> pl.DataFrame:
    """The export and its archives as one frame; a run in both keeps the export's row."""
    frames: list[pl.DataFrame] = []
    for csv_path in statistics_csv_paths(path):
        frame = pl.read_csv(csv_path, **read_options)
        if with_origin:
            frame = frame.with_columns(pl.lit(csv_path.name).alias(EXPORT_FILE_COLUMN))
        frames.append(frame)
    combined = frames[0] if len(frames) == 1 else pl.concat(frames, how="diagonal_relaxed")
    if len(frames) > 1 and "run_id" in combined.columns:
        combined = combined.unique(subset="run_id", keep="first", maintain_order=True)
    return combined


def game_counter_view(path: Path | str) -> dict[str, int | None] | None:
    """What the game's own front-page counter sees, to reconcile it with person counts.

    The game counts distinct browser IDs (``study_id``) in its live export only.
    ``people`` is how many linked persons those browsers are (``None`` without a
    ``participant_key`` column); ``archived`` is the study IDs found only in the
    archives beside the export, which the game no longer counts.
    """
    paths = statistics_csv_paths(path)
    if not paths[0].is_file():
        return None
    live = pl.read_csv(paths[0], infer_schema_length=0)
    live_ids = set(live["study_id"].drop_nulls().to_list())
    archived_ids: set[str] = set()
    for archive in paths[1:]:
        archived_ids |= set(pl.read_csv(archive, infer_schema_length=0)["study_id"].drop_nulls().to_list())
    return {
        "browser_ids": len(live_ids),
        "people": live["participant_key"].drop_nulls().n_unique() if "participant_key" in live.columns else None,
        "archived": len(archived_ids - live_ids),
    }


REQUIRED_COLUMNS: list[str] = [
    "study_id",
    "run_id",
    "timestamp",
    "format",
    "is_example_data",
    "uses_cgm",
    "cgm_duration_years",
    "diabetic",
    "diabetes_duration",
    "rounds_played",
    "overall_mae_mgdl",
    "overall_rmse_mgdl",
    "overall_mape_pct",
    "per_round_metrics",
]

# Written by current sugar-sugar; absent on older exports.
OPTIONAL_COLUMNS: list[str] = [
    "generic_intervention",
    "challenge_unknown",
    "challenge_unknown_pct",
    "paper_mention",
    "paper_full_name",
    "round_context",
    "diabetic_type",
    "data_source_name",
    "gender",
    "age",
]


def load_prediction_statistics(
    path: Path | str, *, use_participant_keys: bool = True
) -> pl.DataFrame:
    """Read and coerce the sugar-sugar statistics export.

    Handles the current export encoding:

    * ``cgm_duration_years`` as ``value,unit`` (e.g. ``6,months``) or a bare year
    * optional ``round_context`` / challenge / paper / intervention columns
    * three-state bools (blank stays unknown, not False)

    Email / location / raw upload filenames are dropped so downstream artefacts
    stay pseudonymized (study design §5.2). Archived games beside the export
    (:func:`statistics_csv_paths`) are read with it; their study IDs were never
    linked by email, so each counts as its own person.
    """
    csv_path = Path(path)
    with start_action(action_type="gathering.load_prediction_statistics", path=str(csv_path)) as action:
        if not csv_path.exists():
            raise FileNotFoundError(f"Statistics CSV not found: {csv_path}")

        raw = read_statistics_csv(csv_path, with_origin=True, infer_schema_length=10_000)
        archived = raw[EXPORT_FILE_COLUMN] != csv_path.name
        if "participant_key" in raw.columns and bool(archived.any()):
            raw = raw.with_columns(
                pl.when(archived & pl.col("participant_key").is_null())
                .then(pl.col("study_id").cast(pl.Utf8))
                .otherwise(pl.col("participant_key"))
                .alias("participant_key")
            )
        n_archived = int(archived.sum())
        raw = raw.drop(EXPORT_FILE_COLUMN)
        missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        drop_cols = [c for c in ("email", "location") if c in raw.columns]
        df = raw.drop(drop_cols) if drop_cols else raw

        # Optional pseudonymous linkage prepared before contact fields are scrubbed.
        # Resetting localStorage creates a new study_id for an existing person.
        if use_participant_keys and "participant_key" in df.columns:
            if df["participant_key"].null_count() or df.filter(
                pl.col("participant_key").cast(pl.Utf8).str.strip_chars() == ""
            ).height:
                raise ValueError("participant_key must be present for every run")
            inconsistent = df.group_by("study_id").agg(
                pl.col("participant_key").n_unique().alias("n_keys")
            ).filter(pl.col("n_keys") != 1)
            if inconsistent.height:
                raise ValueError("Each original study_id must map to one participant_key")
            df = df.with_columns(
                pl.col("study_id").alias("original_study_id"),
                pl.col("participant_key").alias("study_id"),
            )

        if "data_source_name" in df.columns:
            redacted = [redact_source_name(v) for v in df["data_source_name"].to_list()]
            df = df.with_columns(pl.Series("data_source_name", redacted, dtype=pl.Utf8))

        if "per_round_metrics" in df.columns:
            df = df.with_columns(
                pl.Series(
                    "per_round_metrics",
                    [_redact_literal_sources(v, parse_per_round_metrics) for v in df["per_round_metrics"].to_list()],
                )
            )
        if "round_context" in df.columns:
            df = df.with_columns(
                pl.Series(
                    "round_context",
                    [_redact_literal_sources(v, parse_round_context) for v in df["round_context"].to_list()],
                )
            )

        optional_bool_cols = [
            c
            for c in ("is_example_data", "uses_cgm", "diabetic", "paper_mention")
            if c in df.columns
        ]
        for col in optional_bool_cols:
            parsed = [parse_optional_bool(v) for v in df[col].to_list()]
            df = df.with_columns(pl.Series(col, parsed, dtype=pl.Boolean))

        # Feature did not exist on older exports: missing / blank means they did not opt in.
        if "challenge_unknown" in df.columns:
            challenge = [as_strict_bool(v, default=False) for v in df["challenge_unknown"].to_list()]
            df = df.with_columns(pl.Series("challenge_unknown", challenge, dtype=pl.Boolean))
        else:
            df = df.with_columns(pl.lit(False).alias("challenge_unknown"))

        if "cgm_duration_years" in df.columns:
            years = [cgm_duration_to_years(v) for v in df["cgm_duration_years"].to_list()]
            df = df.with_columns(pl.Series("cgm_duration_years", years, dtype=pl.Float64))

        if "diabetes_duration" in df.columns:
            dm_years = [parse_duration_years(v) for v in df["diabetes_duration"].to_list()]
            df = df.with_columns(pl.Series("diabetes_duration", dm_years, dtype=pl.Float64))

        numeric_cols = [
            "age",
            "rounds_played",
            "overall_mae_mgdl",
            "overall_mse_mgdl",
            "overall_rmse_mgdl",
            "overall_mape_pct",
            "number",
            "challenge_unknown_pct",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False).alias(col))

        if "timestamp" in df.columns:
            df = df.with_columns(pl.col("timestamp").str.to_datetime(strict=False).alias("timestamp"))

        df = df.with_columns(
            pl.col("format").cast(pl.Utf8).str.to_uppercase().alias("format"),
            pl.col("study_id").cast(pl.Utf8).alias("study_id"),
            pl.col("run_id").cast(pl.Utf8).alias("run_id"),
        )

        action.log(
            message_type="info",
            n_rows=df.height,
            n_participants=df["study_id"].n_unique(),
            formats=df["format"].unique().to_list(),
            has_round_context="round_context" in df.columns,
            n_archived_runs=n_archived,
        )
        return df


# Re-exported so existing ``from sugar_data_processing.gathering.load import parse_per_round_metrics``
# callers keep working.
__all__ = [
    "ARCHIVE_GLOB",
    "REQUIRED_COLUMNS",
    "OPTIONAL_COLUMNS",
    "game_counter_view",
    "load_prediction_statistics",
    "read_statistics_csv",
    "statistics_csv_paths",
    "parse_per_round_metrics",
    "parse_round_context",
]
