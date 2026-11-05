"""Native AI export: explode, rebuild, and the milestone-AI cohort page."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from sugar_data_processing.ai.milestone import (
    recent_cohort,
    write_milestone_ai_bundle,
    write_population_with_recent_ai,
)
from sugar_data_processing.ai.native import (
    merge_model_predictions,
    model_name_from_repo,
    predictions_from_native_ai,
    write_native_ai_csv,
)
from sugar_data_processing.ai.score import pick_primary_model
from sugar_data_processing.ai.sequences import build_point_table
from sugar_data_processing.config import EVALUATION_MODE_IN_PLACE, EVALUATION_MODE_POST_FACTUM
from sugar_data_processing.fixtures.synthetic import write_synthetic_csv
from sugar_data_processing.gathering.load import load_prediction_statistics


def test_model_name_from_repo_uses_the_last_segment() -> None:
    assert model_name_from_repo("GlucoseDao/DUAL_MODE-CITRAS") == "DUAL_MODE-CITRAS"
    assert model_name_from_repo("glumind") == "glumind"
    assert model_name_from_repo("") == "native"
    assert (
        model_name_from_repo(
            "GlucoseDao/WHAT_IF_FORECAST-KronLSTM-R.80-10h-planned_bolus-physio_consistency"
        )
        == "kronos"
    )


def test_in_place_model_is_primary_over_glumind() -> None:
    frame = pl.DataFrame(
        {
            "model_name": ["glumind", "glumind", "live_model"],
            "evaluation_mode": [
                EVALUATION_MODE_POST_FACTUM,
                EVALUATION_MODE_POST_FACTUM,
                EVALUATION_MODE_IN_PLACE,
            ],
        }
    )
    assert pick_primary_model(frame) == "live_model"


def test_native_row_replaces_the_same_post_factum_key() -> None:
    post = pl.DataFrame(
        {
            "study_id": ["s"],
            "run_id": ["r"],
            "round_number": [1],
            "point_index": [0],
            "model_name": ["glumind"],
            "model_predicted_mgdl": [10.0],
            "evaluation_mode": [EVALUATION_MODE_POST_FACTUM],
        }
    )
    native = post.with_columns(
        pl.lit(12.0).alias("model_predicted_mgdl"),
        pl.lit(EVALUATION_MODE_IN_PLACE).alias("evaluation_mode"),
    )
    merged = merge_model_predictions(post, native)
    assert merged.height == 1
    assert merged["model_predicted_mgdl"][0] == 12.0
    assert merged["evaluation_mode"][0] == EVALUATION_MODE_IN_PLACE


def test_human_report_is_everyone_and_ai_is_the_recent_cohort(tmp_path: Path) -> None:
    human = write_synthetic_csv(tmp_path / "prediction_statistics.csv", n_participants=8, seed=4)
    runs = load_prediction_statistics(human)
    cohort = recent_cohort(runs, 4)
    points = build_point_table(runs)
    recent = points.filter(pl.col("study_id").is_in(cohort.study_ids))
    assert recent.height > 0
    older = next(
        study_id
        for study_id in points["study_id"].unique().to_list()
        if study_id not in set(cohort.study_ids)
    )
    predictions = recent.select(["study_id", "run_id", "round_number", "point_index"]).with_columns(
        pl.lit("kronos").alias("model_name"),
        pl.lit(100.0).alias("model_predicted_mgdl"),
        pl.lit(EVALUATION_MODE_IN_PLACE).alias("evaluation_mode"),
    )
    decoy = predictions.head(1).with_columns(pl.lit(str(older)).alias("study_id"))
    path = tmp_path / "model_predictions.csv"
    pl.concat([predictions, decoy]).write_csv(path)

    report = write_population_with_recent_ai(
        human,
        tmp_path / "milestone_ai.html",
        cohort_size=4,
        predictions_csv=path,
    )
    assert report.n_people == 8
    assert report.n_ai_cohort == 4
    assert set(report.ai_study_ids) == set(cohort.study_ids)
    assert older not in report.ai_study_ids
    text = report.html_path.read_text(encoding="utf-8")
    assert '"n_players": 8' in text
    assert "most recent" in text


def test_native_csv_roundtrip_and_milestone_page(tmp_path: Path) -> None:
    human = write_synthetic_csv(tmp_path / "prediction_statistics.csv", n_participants=8, seed=3)
    runs = load_prediction_statistics(human)
    cohort = recent_cohort(runs, 4)
    assert len(cohort.study_ids) == 4
    assert len(cohort.latest_two) == 2
    assert set(cohort.latest_two).issubset(cohort.study_ids)

    points = build_point_table(runs)
    scored = points.filter(pl.col("study_id").is_in(cohort.latest_two)).with_columns(
        pl.lit("kronos").alias("model_name"),
        (pl.col("real_mgdl") + 1.0).alias("model_predicted_mgdl"),
    )
    assert scored.height > 0
    ai_csv = tmp_path / "prediction_statistics_ai.csv"
    write_native_ai_csv(
        runs,
        scored,
        ai_csv,
        model_name="kronos",
        evaluation_mode=EVALUATION_MODE_IN_PLACE,
    )
    stored = pl.read_csv(ai_csv, infer_schema_length=0)
    assert set(stored["study_id"].to_list()) == set(cohort.latest_two)
    assert stored["email"].to_list() == [""] * stored.height
    assert stored["location"].to_list() == [""] * stored.height

    preds = predictions_from_native_ai(ai_csv)
    left = scored.select(
        ["study_id", "run_id", "round_number", "point_index", "model_predicted_mgdl"]
    ).sort(["study_id", "run_id", "round_number", "point_index"])
    right = preds.select(
        ["study_id", "run_id", "round_number", "point_index", "model_predicted_mgdl"]
    ).sort(["study_id", "run_id", "round_number", "point_index"])
    assert left.height == right.height
    assert left["point_index"].to_list() == right["point_index"].to_list()
    for got, expected in zip(
        right["model_predicted_mgdl"].to_list(), left["model_predicted_mgdl"].to_list(), strict=True
    ):
        assert abs(got - expected) < 1e-6

    bundle = write_milestone_ai_bundle(
        human,
        tmp_path / "milestone_ai",
        ai_csv=ai_csv,
        cohort_size=4,
    )
    assert bundle.n_people == 4
    assert bundle.n_with_ai == 2
    assert bundle.primary_model == "kronos"
    assert bundle.html_path.is_file()
    text = bundle.html_path.read_text(encoding="utf-8")
    assert "milestone AI" in text
    assert "milestone-ai" in text
    summary = json.loads(bundle.json_path.read_text(encoding="utf-8"))
    assert summary["n_people"] == 4
    assert summary["n_with_ai"] == 2
    assert summary["last_two_have_ai"] is True
    rebuilt_human = pl.read_csv(bundle.human_csv, infer_schema_length=0)
    assert rebuilt_human["study_id"].n_unique() == 4
