"""Integration-style tests on synthetic statistics (no mocks)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from sugar_data_processing.fixtures.synthetic import write_synthetic_csv
from sugar_data_processing.pipeline import run_analysis


def test_full_pipeline_on_synthetic(tmp_path: Path) -> None:
    csv_path = tmp_path / "prediction_statistics.csv"
    write_synthetic_csv(csv_path, n_participants=80, seed=11)
    out = tmp_path / "output"
    result = run_analysis(csv_path, out)

    assert result.participants.height == 80
    assert result.report_path.exists()
    report = result.report_path.read_text(encoding="utf-8")
    assert "Human Study Analysis Report" in report
    assert "Diabetes status and prediction accuracy" in report
    assert "Own data vs generic example data" in report
    assert "How to read this report" in report
    assert "Quick glossary" in report
    assert "Challenge the unknown" in report
    assert "post factum" in report.lower() or "post_factum" in report
    assert "data:image/png;base64," in report
    assert 'style="width:100%;max-width:1200px' in report
    assert (out / "figures" / "h1_mae_by_diabetes.png").exists()
    assert (out / "figures" / "cohort_categories_pie.png").exists()
    assert (out / "figures" / "mae_by_format.png").exists()
    assert (out / "figures" / "players_vs_repeats.png").exists()
    assert (out / "figures" / "all_formats_own_vs_generic.png").exists()
    assert (out / "figures" / "people_clusters.png").exists()
    assert (out / "figures" / "opposite_trait.png").exists()
    assert (out / "reports" / "figures" / "h1_mae_by_diabetes.png").exists()
    assert (out / "reports" / "human_analysis_report.json").exists()
    explorer = out / "reports" / "human_explorer.html"
    assert explorer.exists()
    explorer_html = explorer.read_text(encoding="utf-8")
    assert "human study explorer" in explorer_html.lower()
    assert "chart.js" in explorer_html.lower()
    assert "Primary hypotheses" in explorer_html
    assert 'class="pair"' in explorer_html
    assert "Challenge the unknown" in explorer_html
    assert (out / "reports" / "ai_analysis_report.md").exists()
    assert not (out / "reports" / "study_analysis_report.md").exists()
    assert not (out / "reports" / "study_explorer.html").exists()
    assert not (out / "reports" / "study_analysis_report.json").exists()

    # Planted effects should generally be detectable with n=80
    assert result.suite.h1.overall is not None
    assert result.suite.h5 is not None
    assert result.suite.h1.mean_a < result.suite.h1.mean_b  # PwD better (lower MAE)
    assert result.suite.h1.generic is not None
    assert result.suite.h1.own is not None
    assert result.suite.h1.by_category
    assert result.suite.h3.overall is not None
    assert result.suite.h3.overall.predictor == "diabetes_duration_months"
    assert result.suite.h4.overall is not None
    assert result.suite.h4.overall.predictor == "cgm_duration_months"
    assert result.suite.h5.mean_diff > 0  # generic − own > 0 → own better
    json_text = (out / "reports" / "human_analysis_report.json").read_text(encoding="utf-8")
    assert '"by_category"' in json_text
    assert "diabetes_duration_months" in json_text
    assert "Months with diabetes" in explorer_html or "diabetes_duration_months" in explorer_html
    assert "pie-box" in explorer_html
    assert "aspectRatio: 1" in explorer_html
    assert "afterBuildTicks" in explorer_html
    assert "layeredCategoryScales(\"PwD\"" in explorer_html or 'layeredCategoryScales("PwD"' in explorer_html
    assert "months" in report.lower()


def test_gathering_eligibility(tmp_path: Path) -> None:
    csv_path = tmp_path / "stats.csv"
    write_synthetic_csv(csv_path, n_participants=40, seed=3)
    result = run_analysis(csv_path, tmp_path / "out")
    eligible = result.participants.filter(pl.col("eligible_primary"))
    assert eligible.height > 0
    assert "mae_primary" in result.participants.columns
    assert result.verification.schema_ok
    assert "verification" in (tmp_path / "out" / "reports" / "human_analysis_report.json").read_text(
        encoding="utf-8"
    )
