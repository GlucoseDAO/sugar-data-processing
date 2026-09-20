"""AI export / ingest path on real synthetic sequences (no mocks)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from sugar_data_processing.ai.export import export_prediction_sequences
from sugar_data_processing.ai.sequences import known_data_rounds
from sugar_data_processing.ai.ingest import ingest_model_predictions
from sugar_data_processing.ai.report import write_ai_report
from sugar_data_processing.comparison.benchmarks import benchmark_context
from sugar_data_processing.fixtures.synthetic import write_synthetic_csv
from sugar_data_processing.gathering import build_participant_table, load_prediction_statistics
from sugar_data_processing.statistics.hypotheses import run_all_hypotheses
from sugar_data_processing.verification.report import verify_dataset


def test_known_data_rounds_drops_empty_source() -> None:
    rounds = pl.DataFrame(
        {
            "run_id": ["keep", "drop-source", "drop-class"],
            "round_number": [1, 1, 1],
            "data_source_name": ["D1NAMO-001.csv", "", "BIGIDEAS-001.csv"],
            "data_class": ["diabetic", "nondiabetic", ""],
        }
    )
    kept = known_data_rounds(rounds)
    assert kept.height == 1
    assert kept["run_id"].to_list() == ["keep"]


def test_export_sequences_has_timestamp_value_location(tmp_path: Path) -> None:
    csv_path = tmp_path / "stats.csv"
    write_synthetic_csv(csv_path, n_participants=16, seed=9)
    runs = load_prediction_statistics(csv_path)
    paths = export_prediction_sequences(runs, tmp_path / "ai")
    points = pl.read_csv(paths["points"])
    assert points.height > 0
    for col in (
        "study_id",
        "run_id",
        "round_number",
        "point_index",
        "timestamp",
        "real_mgdl",
        "human_predicted_mgdl",
        "window_start_index",
        "evaluation_mode",
        "is_opposite_trait",
        "player_trait",
    ):
        assert col in points.columns
    assert points["evaluation_mode"].unique().to_list() == ["post_factum"]
    assert points["timestamp"].null_count() == 0
    assert points.filter(pl.col("data_source_name").str.len_chars() == 0).height == 0
    assert points.filter(pl.col("data_class").str.len_chars() == 0).height == 0
    assert (tmp_path / "ai" / "manifest.json").exists()


def test_ingest_model_scores_and_ai_report(tmp_path: Path) -> None:
    csv_path = tmp_path / "stats.csv"
    write_synthetic_csv(csv_path, n_participants=12, seed=8)
    runs = load_prediction_statistics(csv_path)
    people = build_participant_table(runs)
    exported = export_prediction_sequences(runs, tmp_path / "ai")
    points = pl.read_csv(exported["points"])
    sample = points.head(40).select(
        "study_id", "run_id", "round_number", "point_index", "real_mgdl"
    ).with_columns(
        pl.lit("toy-persistence").alias("model_name"),
        (pl.col("real_mgdl") + 2.0).alias("model_predicted_mgdl"),
        pl.lit("post_factum").alias("evaluation_mode"),
    ).drop("real_mgdl")
    pred_path = tmp_path / "model.csv"
    sample.write_csv(pred_path)

    scored, comparison = ingest_model_predictions(points, pred_path)
    assert scored.height == 40
    assert comparison.n_models == 1
    assert "toy-persistence" in comparison.model_mean_mae
    assert "post_factum" in comparison.evaluation_modes
    assert comparison.human_mean_mae is not None

    suite = run_all_hypotheses(people)
    verification = verify_dataset(runs, people)
    eligible = people.filter(pl.col("eligible_primary"))
    benchmarks = benchmark_context(eligible if eligible.height else people)
    report = write_ai_report(
        participants=people,
        runs=runs,
        suite=suite,
        benchmarks=benchmarks,
        verification=verification,
        output_dir=tmp_path / "out",
        source_csv=csv_path,
        comparison=comparison,
        scored_points=scored,
        sequences_dir=tmp_path / "ai",
    )
    text = report.read_text(encoding="utf-8")
    assert "AI Study Analysis Report" in text
    assert "toy-persistence" in text
    assert "post_factum" in text
    html = (tmp_path / "out" / "reports" / "ai_explorer.html").read_text(encoding="utf-8")
    assert "AI edition" in html or "ai" in html.lower()
