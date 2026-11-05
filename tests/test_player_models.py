"""Per-player models: the latest day is Kronos, earlier players keep every scored model."""

from __future__ import annotations

import polars as pl

from sugar_data_processing.ai.player_models import (
    player_model_table,
    predictions_for_display,
)


def test_latest_day_is_kronos_only_and_earlier_players_keep_their_models() -> None:
    runs = pl.DataFrame(
        {
            "study_id": ["older", "older", "today"],
            "timestamp": [
                "2026-11-04 10:00:00",
                "2026-11-04 12:00:00",
                "2026-11-05 00:58:35",
            ],
        }
    )
    predictions = pl.DataFrame(
        {
            "study_id": ["older", "older", "today", "today"],
            "model_name": [
                "persistence",
                "linear",
                "persistence",
                "WHAT_IF_FORECAST-KronLSTM-R.80",
            ],
        }
    )
    shown = predictions_for_display(predictions, runs)
    assert set(shown.filter(pl.col("study_id") == "today")["model_name"].to_list()) == {"kronos"}
    assert set(shown.filter(pl.col("study_id") == "older")["model_name"].to_list()) == {
        "persistence",
        "linear",
    }
    table = {
        row["study_id"]: row["models"]
        for row in player_model_table(runs, shown).iter_rows(named=True)
    }
    assert table["today"] == "kronos"
    assert table["older"] == "linear,persistence"
