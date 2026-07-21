"""Load sugar-sugar ``prediction_statistics.csv`` into typed Polars frames."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import polars as pl
from eliot import start_action


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


def _parse_literal(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    return ast.literal_eval(text)


def load_prediction_statistics(path: Path | str) -> pl.DataFrame:
    """Read and coerce the sugar-sugar statistics export.

    Email / location fields are dropped so downstream artefacts stay
    pseudonymized (study design §5.2).
    """
    csv_path = Path(path)
    with start_action(action_type="gathering.load_prediction_statistics", path=str(csv_path)) as action:
        if not csv_path.exists():
            raise FileNotFoundError(f"Statistics CSV not found: {csv_path}")

        raw = pl.read_csv(csv_path, infer_schema_length=10_000)
        missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        drop_cols = [c for c in ("email", "location", "data_source_name") if c in raw.columns]
        df = raw.drop(drop_cols) if drop_cols else raw

        bool_cols = [c for c in ("is_example_data", "uses_cgm", "diabetic") if c in df.columns]
        for col in bool_cols:
            df = df.with_columns(
                pl.col(col)
                .cast(pl.Utf8, strict=False)
                .str.to_lowercase()
                .is_in(["true", "1", "yes", "t"])
                .alias(col)
            )

        numeric_cols = [
            "age",
            "cgm_duration_years",
            "diabetes_duration",
            "rounds_played",
            "overall_mae_mgdl",
            "overall_mse_mgdl",
            "overall_rmse_mgdl",
            "overall_mape_pct",
            "number",
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
        )
        return df


def parse_per_round_metrics(cell: Any) -> list[dict[str, Any]]:
    """Parse the Python-literal list stored in ``per_round_metrics``."""
    parsed = _parse_literal(cell)
    if parsed is None:
        return []
    if not isinstance(parsed, list):
        raise TypeError(f"per_round_metrics must be a list, got {type(parsed)}")
    return [item for item in parsed if isinstance(item, dict)]
