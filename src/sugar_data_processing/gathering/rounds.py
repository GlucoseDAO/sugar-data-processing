"""Explode run-level statistics into one row per prediction round."""

from __future__ import annotations

from typing import Any

import polars as pl
from eliot import start_action

from sugar_data_processing.config import FORMAT_GENERIC, FORMAT_MIXED, FORMAT_OWN
from sugar_data_processing.gathering.load import parse_per_round_metrics


def classify_source(format_code: str, is_example_data: bool | None, round_number: int | None = None) -> str:
    """Map sugar-sugar format flags to generic / own / mixed segment labels.

    - Format A → always generic
    - Format B → always own
    - Format C → mixed: odd rounds generic, even rounds own (app convention)
    """
    fmt = (format_code or "").upper()
    if fmt == FORMAT_GENERIC:
        return "generic"
    if fmt == FORMAT_OWN:
        return "own"
    if fmt == FORMAT_MIXED:
        if round_number is None:
            return "mixed"
        return "generic" if int(round_number) % 2 == 1 else "own"
    if is_example_data is True:
        return "generic"
    if is_example_data is False:
        return "own"
    return "unknown"


def build_round_table(runs: pl.DataFrame) -> pl.DataFrame:
    """Expand ``per_round_metrics`` into a tidy round-level frame."""
    with start_action(action_type="gathering.build_round_table", n_runs=runs.height) as action:
        rows: list[dict[str, Any]] = []
        for record in runs.iter_rows(named=True):
            metrics = parse_per_round_metrics(record.get("per_round_metrics"))
            if not metrics:
                # Fall back to a single synthetic round from overall metrics
                rows.append(
                    {
                        "study_id": record["study_id"],
                        "run_id": record["run_id"],
                        "format": record["format"],
                        "round_number": 1,
                        "source": classify_source(record["format"], record.get("is_example_data"), 1),
                        "mae": record.get("overall_mae_mgdl"),
                        "rmse": record.get("overall_rmse_mgdl"),
                        "mape": record.get("overall_mape_pct"),
                        "mse": record.get("overall_mse_mgdl"),
                    }
                )
                continue
            for item in metrics:
                round_number = int(item.get("round_number") or item.get("round") or 0)
                rows.append(
                    {
                        "study_id": record["study_id"],
                        "run_id": record["run_id"],
                        "format": record["format"],
                        "round_number": round_number,
                        "source": classify_source(
                            record["format"],
                            record.get("is_example_data"),
                            round_number,
                        ),
                        "mae": float(item["mae"]) if item.get("mae") is not None else None,
                        "rmse": float(item["rmse"]) if item.get("rmse") is not None else None,
                        "mape": float(item["mape"]) if item.get("mape") is not None else None,
                        "mse": float(item["mse"]) if item.get("mse") is not None else None,
                    }
                )

        if not rows:
            empty = pl.DataFrame(
                schema={
                    "study_id": pl.Utf8,
                    "run_id": pl.Utf8,
                    "format": pl.Utf8,
                    "round_number": pl.Int64,
                    "source": pl.Utf8,
                    "mae": pl.Float64,
                    "rmse": pl.Float64,
                    "mape": pl.Float64,
                    "mse": pl.Float64,
                }
            )
            action.log(message_type="info", n_rounds=0)
            return empty

        rounds = pl.DataFrame(rows).with_columns(
            pl.col("round_number").cast(pl.Int64),
            pl.col("mae").cast(pl.Float64),
            pl.col("rmse").cast(pl.Float64),
            pl.col("mape").cast(pl.Float64),
            pl.col("mse").cast(pl.Float64),
        )
        action.log(message_type="info", n_rounds=rounds.height)
        return rounds
