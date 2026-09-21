"""Orchestrate reconstruct → convert → score → ingest for saved games."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
from eliot import start_action

from sugar_data_processing.ai.ingest import ModelComparison, ingest_model_predictions
from sugar_data_processing.ai.overlay import h6_human_vs_models, overlay_model_mae
from sugar_data_processing.ai.score import pick_primary_model, score_windows
from sugar_data_processing.ai.sequences import build_point_table
from sugar_data_processing.ai.traces import build_round_traces
from sugar_data_processing.ai.windows import PlayedWindow, reconstruct_played_windows
from sugar_data_processing.config import DEFAULT_SUGAR_SUGAR_ROOT
from sugar_data_processing.statistics.hypotheses import HypothesisSuite, run_all_hypotheses


@dataclass(frozen=True)
class ScoringResult:
    windows: list[PlayedWindow]
    predictions: pl.DataFrame
    scored_points: pl.DataFrame
    comparison: ModelComparison
    primary_model: str
    ai_participants: pl.DataFrame
    ai_suite: HypothesisSuite
    h6: dict[str, Any]
    traces: list[dict[str, Any]]


def run_post_factum_scoring(
    runs: pl.DataFrame,
    participants: pl.DataFrame,
    *,
    source_csv: Path | str,
    ai_dir: Path | str,
    sugar_root: Path | None = None,
) -> ScoringResult:
    """Rebuild 3-hour windows, score them, and attach AI person-level tables."""
    dest = Path(ai_dir)
    dest.mkdir(parents=True, exist_ok=True)
    root = Path(sugar_root) if sugar_root is not None else DEFAULT_SUGAR_SUGAR_ROOT
    with start_action(action_type="ai.run_post_factum_scoring") as action:
        points = build_point_table(runs)
        windows = reconstruct_played_windows(
            runs,
            raw_csv=source_csv,
            sugar_root=root,
            points=points,
        )
        predictions = score_windows(
            windows,
            ml_ready_path=dest / "ml_ready.csv",
        )
        predictions.write_csv(dest / "model_predictions.csv")
        if predictions.height:
            scored, comparison = ingest_model_predictions(points, predictions)
        else:
            scored, comparison = ingest_model_predictions(
                points,
                pl.DataFrame(
                    schema={
                        "study_id": pl.Utf8,
                        "run_id": pl.Utf8,
                        "round_number": pl.Int64,
                        "point_index": pl.Int64,
                        "model_name": pl.Utf8,
                        "model_predicted_mgdl": pl.Float64,
                        "evaluation_mode": pl.Utf8,
                    }
                ),
            )
        if scored.height:
            scored.write_csv(dest / "model_scores.csv")
        primary = pick_primary_model(predictions)
        ai_people = overlay_model_mae(participants, scored, primary) if scored.height else participants
        ai_suite = run_all_hypotheses(ai_people)
        h6 = h6_human_vs_models(participants, scored)
        traces = build_round_traces(windows, scored)
        action.log(
            message_type="info",
            n_windows=len(windows),
            n_pred_rows=predictions.height,
            primary=primary,
        )
        return ScoringResult(
            windows=windows,
            predictions=predictions,
            scored_points=scored,
            comparison=comparison,
            primary_model=primary,
            ai_participants=ai_people,
            ai_suite=ai_suite,
            h6=h6,
            traces=traces,
        )
