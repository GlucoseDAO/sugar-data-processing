"""Dashboard front-page summary built from a real (synthetic) participant table."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from sugar_data_processing.fixtures.synthetic import write_synthetic_csv
from sugar_data_processing.gathering import build_participant_table, load_prediction_statistics
from sugar_data_processing.statistics.hypotheses import run_all_hypotheses
from sugar_data_processing.statistics.progress import (
    MIN_CLASS_ROUNDS,
    REVISED_TARGET_ELIGIBLE,
    study_progress,
)


def _progress(tmp_path: Path, **kwargs: object) -> tuple[pl.DataFrame, dict]:
    csv_path = tmp_path / "prediction_statistics.csv"
    write_synthetic_csv(csv_path, n_participants=80, seed=11)
    people = build_participant_table(load_prediction_statistics(csv_path))
    return people, study_progress(people, run_all_hypotheses(people).to_dict(), **kwargs)


def test_recruitment_counts_eligible_people_against_quotas(tmp_path: Path) -> None:
    people, progress = _progress(tmp_path)
    assert progress["n_people"] == people.height
    assert progress["n_eligible"] == people.filter(pl.col("eligible_primary")).height
    assert sum(cell["target"] for cell in progress["cells"]) == REVISED_TARGET_ELIGIBLE
    deficit = sum(max(0, cell["target"] - cell["now"]) for cell in progress["cells"])
    assert progress["remaining_eligible"] == deficit
    assert {row["code"] for row in progress["power"]} == {"H1", "H2", "H3", "H4", "H5"}
    assert progress["curves"]["series"][-1]["code"] == "plan"


def test_trace_matching_and_design_use_per_trace_counts(tmp_path: Path) -> None:
    people, progress = _progress(tmp_path)
    assert {"mae_generic_diabetic_data", "n_rounds_generic_nondiabetic_data"} <= set(people.columns)
    design = progress["design"]
    assert design is not None and len(design["cells"]) == 4
    for cell in design["cells"]:
        assert cell["players_enough"] <= cell["players_any"] <= cell["players"]
    diabetic_traces = next(r for r in progress["trace_matched"]["by_class"] if r["key"] == "diabetic_data")
    enough = people.filter(pl.col("n_rounds_generic_diabetic_data") >= MIN_CLASS_ROUNDS)
    assert diabetic_traces["n_a"] == enough.filter(pl.col("diabetic").eq(True)).height
    assert diabetic_traces["n_b"] == enough.filter(pl.col("diabetic").eq(False)).height


def test_h6_card_says_when_only_baselines_were_scored(tmp_path: Path) -> None:
    h6 = {
        "comparisons": [
            {"model_name": "persistence", "n": 40, "human_mean_mae": 21.0, "model_mean_mae": 16.0,
             "p_value": 0.001, "significant": True},
            {"model_name": "linear", "n": 40, "human_mean_mae": 21.0, "model_mean_mae": 26.0,
             "p_value": 0.002, "significant": True},
        ]
    }
    _, progress = _progress(tmp_path, h6=h6)
    card = next(f for f in progress["findings"] if f["code"] == "H6")
    assert "Persistence predicted better than people" in card["headline"]
    assert "Only the simple baselines" in card["detail"]
    assert card["verdict"] == "significant"
