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
    assert progress["n_scored"] == people.filter(pl.col("n_rounds_total") > 0).height
    assert progress["n_eligible"] == people.filter(pl.col("eligible_primary")).height
    assert sum(cell["target"] for cell in progress["cells"]) == REVISED_TARGET_ELIGIBLE
    deficit = sum(max(0, cell["target"] - cell["now"]) for cell in progress["cells"])
    assert progress["remaining_eligible"] == deficit
    assert {row["code"] for row in progress["power"]} == {"H1", "H2", "H3", "H4", "H5"}
    assert progress["curves"]["series"][-1]["code"] == "plan"


def test_power_chart_draws_every_question_on_one_people_axis(tmp_path: Path) -> None:
    h6 = {"comparisons": [
        {"model_name": "persistence", "n": 40, "human_mean_mae": 21.0, "model_mean_mae": 16.0,
         "mean_diff": 5.0, "sd_diff": 9.0, "p_value": 0.001, "significant": True},
        {"model_name": "kronlstm", "n": 40, "human_mean_mae": 21.0, "model_mean_mae": 14.0,
         "mean_diff": 7.0, "sd_diff": 11.0, "p_value": 0.001, "significant": True},
    ]}
    _, progress = _progress(tmp_path, h6=h6)
    power = {row["code"]: row for row in progress["power"]}
    assert set(power) == {"H1", "H2", "H3", "H4", "H5", "H6"}
    h6_row = power["H6"]
    assert h6_row["model_label"] == "KronLSTM R.80"
    assert h6_row["difference_sd"] == 11.0
    assert h6_row["need_total"] > h6_row["n_now"] == 40
    curves = progress["curves"]
    drawn = {series["code"] for series in curves["series"]}
    assert {"H3", "H4", "H5", "H6", "plan"} <= drawn
    h4 = next(series for series in curves["series"] if series["code"] == "H4")
    assert h4["shared"] and "same line as H3" in h4["label"]
    for series in curves["series"]:
        assert len(series["values"]) == len(curves["people"])
        assert all(a <= b + 1e-9 for a, b in zip(series["values"], series["values"][1:]))
    dots = {pt["code"]: pt for pt in curves["now"]}
    assert dots["H6"]["people"] == 40
    assert dots["H1"]["people"] == 2 * power["H1"]["n_effective"]


def test_h6_power_without_a_measured_spread_uses_the_h5_planning_sd(tmp_path: Path) -> None:
    h6 = {"comparisons": [{"model_name": "persistence", "n": 40, "human_mean_mae": 21.0,
                           "model_mean_mae": 16.0, "p_value": 0.001, "significant": True}]}
    _, progress = _progress(tmp_path, h6=h6)
    h6_row = next(row for row in progress["power"] if row["code"] == "H6")
    h5_row = next(row for row in progress["power"] if row["code"] == "H5")
    assert h6_row["sd"] is None
    assert h6_row["difference_sd"] == h5_row["difference_sd"]
    label = next(s["label"] for s in progress["curves"]["series"] if s["code"] == "H6")
    assert "planning spread" in label


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
    assert card["tone"] == "good"


def test_h6_is_green_when_trained_models_beat_people_and_red_otherwise(tmp_path: Path) -> None:
    def card_for(model_mae: float) -> dict:
        h6 = {"comparisons": [{"model_name": "kronlstm", "n": 40, "human_mean_mae": 21.0,
                               "model_mean_mae": model_mae, "p_value": 0.001, "significant": True}]}
        _, progress = _progress(tmp_path, h6=h6)
        return next(f for f in progress["findings"] if f["code"] == "H6")

    assert card_for(14.0)["tone"] == "good"
    assert card_for(14.0)["verdict"] == "supported"
    assert card_for(28.0)["tone"] == "bad"


def test_every_finding_has_a_traffic_light_tone(tmp_path: Path) -> None:
    _, progress = _progress(tmp_path)
    assert progress["findings"]
    assert {f["tone"] for f in progress["findings"]} <= {"good", "warn", "bad"}
