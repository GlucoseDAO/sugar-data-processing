"""Ingest already-scored model predictions and join them onto exported points."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import polars as pl
from eliot import start_action

from sugar_data_processing.config import (
    EVALUATION_MODE_IN_PLACE,
    EVALUATION_MODE_POST_FACTUM,
)

REQUIRED_MODEL_COLUMNS: tuple[str, ...] = (
    "study_id",
    "run_id",
    "round_number",
    "point_index",
    "model_name",
    "model_predicted_mgdl",
)

JOIN_KEYS: tuple[str, ...] = ("study_id", "run_id", "round_number", "point_index")


@dataclass(frozen=True)
class ModelComparison:
    """Person- and mode-level summary of ingested model scores."""

    n_points: int
    n_people: int
    n_models: int
    models: list[str]
    evaluation_modes: list[str]
    human_mean_mae: float | None
    model_mean_mae: dict[str, float]
    by_mode: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_model_predictions(path: Path | str) -> pl.DataFrame:
    """Read a model-output CSV that follows the export join contract."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Model predictions CSV not found: {csv_path}")
    frame = pl.read_csv(csv_path, infer_schema_length=10_000)
    missing = [c for c in REQUIRED_MODEL_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Model CSV missing required columns: {missing}")
    if "evaluation_mode" not in frame.columns:
        frame = frame.with_columns(pl.lit(EVALUATION_MODE_POST_FACTUM).alias("evaluation_mode"))
    return frame.with_columns(
        pl.col("study_id").cast(pl.Utf8),
        pl.col("run_id").cast(pl.Utf8),
        pl.col("round_number").cast(pl.Int64),
        pl.col("point_index").cast(pl.Int64),
        pl.col("model_name").cast(pl.Utf8),
        pl.col("model_predicted_mgdl").cast(pl.Float64, strict=False),
        pl.col("evaluation_mode").cast(pl.Utf8),
    )


def ingest_model_predictions(
    points: pl.DataFrame,
    predictions: pl.DataFrame | Path | str,
) -> tuple[pl.DataFrame, ModelComparison]:
    """Join model predictions onto exported points and compute MAE.

    Returns the joined point frame plus a compact comparison summary.
    """
    with start_action(action_type="ai.ingest_model_predictions") as action:
        pred = (
            predictions
            if isinstance(predictions, pl.DataFrame)
            else load_model_predictions(predictions)
        )
        joined = points.join(pred, on=list(JOIN_KEYS), how="inner", suffix="_model")
        if "evaluation_mode_model" in joined.columns:
            joined = joined.with_columns(
                pl.coalesce(
                    [pl.col("evaluation_mode_model"), pl.col("evaluation_mode")]
                ).alias("evaluation_mode")
            ).drop("evaluation_mode_model")
        if joined.height == 0:
            comparison = ModelComparison(
                n_points=0,
                n_people=0,
                n_models=0,
                models=[],
                evaluation_modes=[],
                human_mean_mae=None,
                model_mean_mae={},
                by_mode={
                    EVALUATION_MODE_POST_FACTUM: {"n_points": 0},
                    EVALUATION_MODE_IN_PLACE: {"n_points": 0},
                },
            )
            action.log(message_type="warning", reason="no_join_matches")
            return joined, comparison

        scored = joined.with_columns(
            (pl.col("human_predicted_mgdl") - pl.col("real_mgdl")).abs().alias("human_abs_error"),
            (pl.col("model_predicted_mgdl") - pl.col("real_mgdl")).abs().alias("model_abs_error"),
        )
        human_mae = scored["human_abs_error"].mean()
        model_means = (
            scored.group_by("model_name")
            .agg(pl.col("model_abs_error").mean().alias("mae"))
            .to_dicts()
        )
        by_mode: dict[str, dict[str, Any]] = {}
        for mode in scored["evaluation_mode"].unique().to_list():
            sub = scored.filter(pl.col("evaluation_mode") == mode)
            by_mode[str(mode)] = {
                "n_points": sub.height,
                "n_people": sub["study_id"].n_unique(),
                "human_mean_mae": float(sub["human_abs_error"].mean() or 0.0),
                "model_mean_mae": {
                    str(row["model_name"]): float(row["mae"])
                    for row in sub.group_by("model_name")
                    .agg(pl.col("model_abs_error").mean().alias("mae"))
                    .to_dicts()
                },
            }
        comparison = ModelComparison(
            n_points=scored.height,
            n_people=int(scored["study_id"].n_unique()),
            n_models=int(scored["model_name"].n_unique()),
            models=sorted(str(m) for m in scored["model_name"].unique().to_list()),
            evaluation_modes=sorted(str(m) for m in scored["evaluation_mode"].unique().to_list()),
            human_mean_mae=float(human_mae) if human_mae is not None else None,
            model_mean_mae={str(row["model_name"]): float(row["mae"]) for row in model_means},
            by_mode=by_mode,
        )
        action.log(
            message_type="info",
            n_points=comparison.n_points,
            n_models=comparison.n_models,
            modes=comparison.evaluation_modes,
        )
        return scored, comparison
