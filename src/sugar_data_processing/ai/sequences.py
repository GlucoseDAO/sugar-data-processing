"""Explode run-level prediction series into one row per (player, game, point)."""

from __future__ import annotations

from typing import Any

import polars as pl
from eliot import start_action

from sugar_data_processing.config import EVALUATION_MODE_POST_FACTUM
from sugar_data_processing.gathering.encoding import parse_point_series, series_value
from sugar_data_processing.gathering.rounds import build_round_table


def _as_float(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    return float(text)


def _as_int(raw: Any) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def known_data_rounds(rounds: pl.DataFrame) -> pl.DataFrame:
    """Rounds we can actually replay: known source filename and trace class.

    Post-factum export drops the rest (empty ``data_source_name`` / ``data_class``).
    Human analysis still keeps those rows, with ``is_opposite_trait=False``.
    """
    source_ok = pl.col("data_source_name").fill_null("").str.strip_chars().str.len_chars() > 0
    class_ok = pl.col("data_class").fill_null("").str.strip_chars().str.len_chars() > 0
    return rounds.filter(source_ok & class_ok)


def _round_has_known_data(meta: dict[str, Any]) -> bool:
    if not meta:
        return False
    source = str(meta.get("data_source_name") or "").strip()
    data_class = str(meta.get("data_class") or "").strip()
    return bool(source) and bool(data_class)


def _round_lookup(rounds: pl.DataFrame) -> dict[tuple[str, int], dict[str, Any]]:
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rounds.iter_rows(named=True):
        key = (str(row["run_id"]), int(row["round_number"]))
        lookup[key] = row
    return lookup


def build_point_table(
    runs: pl.DataFrame,
    *,
    rounds: pl.DataFrame | None = None,
    evaluation_mode: str = EVALUATION_MODE_POST_FACTUM,
) -> pl.DataFrame:
    """One row per predicted glucose point, joined to round / opposite-trait context.

    ``location`` here is the window position in the CGM series
    (``window_start_index`` plus the point timestamp), not a geographic city.
    """
    with start_action(action_type="ai.build_point_table", n_runs=runs.height) as action:
        round_table = known_data_rounds(rounds if rounds is not None else build_round_table(runs))
        by_round = _round_lookup(round_table)
        rows: list[dict[str, Any]] = []

        for record in runs.iter_rows(named=True):
            predicted = parse_point_series(record.get("predicted_values"))
            actual = parse_point_series(record.get("real_values"))
            times = parse_point_series(record.get("prediction_times"))
            n_points = min(len(predicted), len(actual), len(times))
            if n_points == 0:
                continue
            run_id = str(record.get("run_id") or "")
            per_round_index: dict[int, int] = {}
            for i in range(n_points):
                round_number = _as_int(predicted[i].get("round")) or _as_int(actual[i].get("round"))
                if round_number is None:
                    round_number = 1
                point_index = per_round_index.get(round_number, 0)
                per_round_index[round_number] = point_index + 1
                meta = by_round.get((run_id, round_number), {})
                if not _round_has_known_data(meta):
                    continue
                rows.append(
                    {
                        "study_id": str(record["study_id"]),
                        "run_id": run_id,
                        "format": str(record.get("format") or ""),
                        "round_number": round_number,
                        "point_index": point_index,
                        "timestamp": str(series_value(times[i]) or ""),
                        "real_mgdl": _as_float(series_value(actual[i])),
                        "human_predicted_mgdl": _as_float(series_value(predicted[i])),
                        "window_start_index": meta.get("window_start_index"),
                        "window_start_time": meta.get("window_start_time") or "",
                        "prediction_start_time": meta.get("prediction_start_time") or "",
                        "window_end_time": meta.get("window_end_time") or "",
                        "data_source_name": meta.get("data_source_name") or "",
                        "source": meta.get("source") or "",
                        "data_class": meta.get("data_class") or "",
                        "player_trait": meta.get("player_trait") or "unknown",
                        "is_opposite_trait": bool(meta.get("is_opposite_trait") or False),
                        "challenge_unknown": bool(meta.get("challenge_unknown") or False),
                        "generic_slice_key": meta.get("generic_slice_key") or "",
                        "evaluation_mode": evaluation_mode,
                    }
                )

        schema = {
            "study_id": pl.Utf8,
            "run_id": pl.Utf8,
            "format": pl.Utf8,
            "round_number": pl.Int64,
            "point_index": pl.Int64,
            "timestamp": pl.Utf8,
            "real_mgdl": pl.Float64,
            "human_predicted_mgdl": pl.Float64,
            "window_start_index": pl.Int64,
            "window_start_time": pl.Utf8,
            "prediction_start_time": pl.Utf8,
            "window_end_time": pl.Utf8,
            "data_source_name": pl.Utf8,
            "source": pl.Utf8,
            "data_class": pl.Utf8,
            "player_trait": pl.Utf8,
            "is_opposite_trait": pl.Boolean,
            "challenge_unknown": pl.Boolean,
            "generic_slice_key": pl.Utf8,
            "evaluation_mode": pl.Utf8,
        }
        if not rows:
            action.log(message_type="info", n_points=0)
            return pl.DataFrame(schema=schema)

        points = pl.DataFrame(rows, schema=schema)
        action.log(message_type="info", n_points=points.height)
        return points
