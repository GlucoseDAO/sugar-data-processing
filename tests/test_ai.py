"""AI export / ingest path on real synthetic sequences (no mocks)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from sugar_data_processing.ai.export import export_prediction_sequences
from sugar_data_processing.ai.score import MODEL_GLUMIND, score_windows
from sugar_data_processing.ai.sequences import known_data_rounds
from sugar_data_processing.ai.ingest import ingest_model_predictions
from sugar_data_processing.ai.report import write_ai_report
from sugar_data_processing.ai.traces import (
    LINE_STYLES,
    build_round_traces,
    forecast_minute_labels,
    percent_error_series,
)
from sugar_data_processing.ai.windows import PlayedWindow
from sugar_data_processing.config import find_forecasting_root
from sugar_data_processing.comparison.benchmarks import benchmark_context
from sugar_data_processing.fixtures.synthetic import write_synthetic_csv
from sugar_data_processing.gathering import build_participant_table, load_prediction_statistics
from sugar_data_processing.output.plots import task_comparison_summary
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
    assert "Sugar Sugar Study Analysis Report" in text
    assert "merged" in text.lower()
    html = (tmp_path / "out" / "reports" / "explorer.html").read_text(encoding="utf-8")
    assert 'data-tab="people"' in html
    assert "Actual CGM" in html
    assert "nextPerson" in html
    assert "line_styles" in html
    assert "aiTaskMae" in html
    assert "AI H1–H5" not in html
    assert "personDev-A" in html
    assert "aiDevHost" in html
    assert "pointRadius: 1" in html
    assert "All-round deviations" in html
    assert "chartLightbox" in html
    assert 'id="aiMix"' in html
    assert "aiMixMiddle" in html
    assert "bindEnlarge" in html
    assert "cloneDataset" in html
    assert "toDataURL" in html
    assert "aiCohortChips" in html
    assert "pushErrorBand" in html
    assert "legendFilter" in html
    assert 'fill: "+1"' in html
    assert "ds.pointRadius == null ? 0" not in html
    assert "AI_COLORS" in html
    assert "byName.human" in html
    assert "applyLightboxFocus" in html
    assert "lightboxFocusBar" in html
    assert "setLightboxFocus" in html
    assert "lightboxFocus" in html
    milestone = (tmp_path / "out" / "reports" / "milestone.html").read_text(encoding="utf-8")
    assert '"milestone": true' in milestone
    assert 'data-tab="ai"' not in milestone
    assert 'data-tab="people"' not in milestone


def test_round_traces_keep_actual_human_and_model_series() -> None:
    window = PlayedWindow(
        study_id="person-1",
        run_id="run-1",
        round_number=2,
        format="A",
        source="generic",
        data_source_name="example.csv",
        context_times=[f"t{i}" for i in range(24)],
        context_mgdl=[100.0 + i for i in range(24)],
        horizon_times=[f"h{i}" for i in range(12)],
        real_mgdl=[130.0 + i for i in range(12)],
        human_predicted_mgdl=[128.0 + i for i in range(12)],
        resolved_path="example.csv",
    )
    scored = pl.DataFrame(
        {
            "study_id": ["person-1"] * 12,
            "run_id": ["run-1"] * 12,
            "round_number": [2] * 12,
            "point_index": list(range(12)),
            "model_name": ["persistence"] * 12,
            "model_predicted_mgdl": [123.0] * 12,
        }
    )
    traces = build_round_traces([window], scored)
    assert len(traces) == 1
    trace = traces[0]
    assert len(trace["actual"]) == 36
    assert trace["human"][:24] == [None] * 24
    assert trace["human"][24:] == [128.0 + i for i in range(12)]
    assert trace["models"]["persistence"][24:] == [123.0] * 12
    assert LINE_STYLES["actual"]["color"] == "#0f172a"
    assert LINE_STYLES["human"]["dash"] == "dashed"
    assert LINE_STYLES["persistence"]["dash"] == "dotted"
    hidden = percent_error_series(trace["actual"], trace["human"], visible_end=trace["visible_end"])
    assert hidden[0] == pytest.approx((128.0 - 130.0) / 130.0 * 100.0)
    assert forecast_minute_labels(12) == [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]


def test_percent_error_falls_back_on_first_missing_prediction() -> None:
    actual = [100.0] * 24 + [200.0, 220.0]
    predicted = [None] * 24 + [None, 242.0]
    series = percent_error_series(actual, predicted, visible_end=23)
    assert series[0] == 0.0
    assert series[1] == pytest.approx(10.0)


def _sample_window() -> PlayedWindow:
    return PlayedWindow(
        study_id="person-1",
        run_id="run-1",
        round_number=2,
        format="A",
        source="generic",
        data_source_name="example.csv",
        context_times=[f"2026-01-01 08:{i:02d}:00" for i in range(24)],
        context_mgdl=[100.0 + i for i in range(24)],
        horizon_times=[f"2026-01-01 09:{i:02d}:00" for i in range(12)],
        real_mgdl=[130.0 + i for i in range(12)],
        human_predicted_mgdl=[128.0 + i for i in range(12)],
        resolved_path="example.csv",
    )


def test_glumind_scores_one_real_window() -> None:
    root = find_forecasting_root()
    assert root is not None
    assert (root / "test_model" / "best_model.pt").exists()
    scored = score_windows([_sample_window()], forecasting_root=root)
    names = set(scored["model_name"].to_list())
    assert {"persistence", "linear", MODEL_GLUMIND} <= names
    glu = scored.filter(pl.col("model_name") == MODEL_GLUMIND)
    assert glu.height == 12
    assert glu["model_predicted_mgdl"].null_count() == 0
    values = [float(v) for v in glu["model_predicted_mgdl"].to_list()]
    assert all(40.0 <= v <= 400.0 for v in values)


def test_task_comparison_pairs_only_complete_same_users() -> None:
    human = pl.DataFrame(
        {
            "study_id": ["u1", "u2", "u3"],
            "mae_format_a": [10.0, 20.0, 15.0],
            "mae_format_b": [8.0, None, 14.0],
            "mae_format_c": [12.0, 18.0, None],
        }
    )
    ai = pl.DataFrame(
        {
            "study_id": ["u1", "u2", "u3"],
            "mae_format_a": [11.0, 19.0, 16.0],
            "mae_format_b": [11.0, 21.0, 16.0],
            "mae_format_c": [12.5, 18.0, 17.0],
        }
    )
    summary = task_comparison_summary(human, ai)
    assert summary["n_joined"] == 3
    assert summary["tasks"]["Generic (A)"]["n"] == 3
    assert summary["tasks"]["Own (B)"]["n"] == 2
    assert summary["tasks"]["Mixed (C)"]["n"] == 2
    assert summary["cluster"]["n"] == 2
    assert summary["cluster"]["human_better_own"] == 2
    assert summary["tasks"]["Own (B)"]["human_better"] == 2
