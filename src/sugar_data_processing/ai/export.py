"""Write tidy CSVs that an AI scoring pipeline can consume without re-parsing."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
from eliot import start_action

from sugar_data_processing.ai.sequences import build_point_table, known_data_rounds
from sugar_data_processing.config import EVALUATION_MODE_POST_FACTUM
from sugar_data_processing.gathering.rounds import build_round_table


def export_prediction_sequences(
    runs: pl.DataFrame,
    output_dir: Path | str,
    *,
    evaluation_mode: str = EVALUATION_MODE_POST_FACTUM,
) -> dict[str, Path]:
    """Export saved games as round + point CSVs for post-factum model scoring.

    Rounds with no source filename or trace class are omitted (human analysis
    still keeps them, with ``is_opposite_trait=False``).

    Files
    -----
    ``prediction_points.csv``
        One row per timestamped glucose point (the feed for models).
    ``prediction_rounds.csv``
        One row per round, with MAE and opposite-trait flags.
    ``prediction_runs.csv``
        One row per game, without the bulky literal lists.
    ``manifest.json``
        Column contract and evaluation-mode note.
    """
    output_dir = Path(output_dir)
    with start_action(action_type="ai.export_prediction_sequences") as action:
        output_dir.mkdir(parents=True, exist_ok=True)
        all_rounds = build_round_table(runs)
        rounds = known_data_rounds(all_rounds)
        n_dropped = all_rounds.height - rounds.height
        points = build_point_table(runs, rounds=rounds, evaluation_mode=evaluation_mode)

        run_cols = [
            c
            for c in (
                "study_id",
                "run_id",
                "timestamp",
                "format",
                "rounds_played",
                "overall_mae_mgdl",
                "challenge_unknown",
                "challenge_unknown_pct",
                "generic_intervention",
                "diabetic",
                "diabetic_type",
                "uses_cgm",
            )
            if c in runs.columns
        ]
        usable_ids = rounds["run_id"].unique().to_list() if rounds.height else []
        runs_kept = runs.filter(pl.col("run_id").is_in(usable_ids)) if usable_ids else runs.head(0)
        runs_slim = runs_kept.select(run_cols).with_columns(
            pl.lit(evaluation_mode).alias("evaluation_mode")
        )
        rounds_out = rounds.with_columns(pl.lit(evaluation_mode).alias("evaluation_mode"))

        paths = {
            "points": output_dir / "prediction_points.csv",
            "rounds": output_dir / "prediction_rounds.csv",
            "runs": output_dir / "prediction_runs.csv",
            "manifest": output_dir / "manifest.json",
        }
        points.write_csv(paths["points"])
        rounds_out.write_csv(paths["rounds"])
        runs_slim.write_csv(paths["runs"])

        manifest: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "evaluation_mode": evaluation_mode,
            "evaluation_mode_note": (
                "post_factum = score these saved sequences later. "
                "in_place = the model ran during the live game (not in this export)."
            ),
            "n_runs": runs_slim.height,
            "n_rounds": rounds.height,
            "n_rounds_dropped_unknown_data": n_dropped,
            "n_points": points.height,
            "files": {key: path.name for key, path in paths.items() if key != "manifest"},
            "point_columns": points.columns,
            "join_keys": ["study_id", "run_id", "round_number", "point_index"],
            "model_result_columns": [
                "study_id",
                "run_id",
                "round_number",
                "point_index",
                "model_name",
                "model_predicted_mgdl",
                "evaluation_mode",
            ],
        }
        paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        action.log(
            message_type="info",
            n_points=points.height,
            n_rounds=rounds.height,
            n_rounds_dropped=n_dropped,
            output=str(output_dir),
        )
        return paths
