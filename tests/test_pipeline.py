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
    assert "Diabetes status and prediction accuracy" in report
    assert "Own data vs generic example data" in report
    assert "data:image/png;base64," in report
    assert (out / "figures" / "h1_mae_by_diabetes.png").exists()
    assert (out / "reports" / "figures" / "h1_mae_by_diabetes.png").exists()
    assert (out / "reports" / "study_analysis_report.json").exists()

    # Planted effects should generally be detectable with n=80
    assert result.suite.h1 is not None
    assert result.suite.h5 is not None
    assert result.suite.h1.mean_a < result.suite.h1.mean_b  # PwD better (lower MAE)
    assert result.suite.h5.mean_diff > 0  # generic − own > 0 → own better


def test_gathering_eligibility(tmp_path: Path) -> None:
    csv_path = tmp_path / "stats.csv"
    write_synthetic_csv(csv_path, n_participants=40, seed=3)
    result = run_analysis(csv_path, tmp_path / "out")
    eligible = result.participants.filter(pl.col("eligible_primary"))
    assert eligible.height > 0
    assert "mae_primary" in result.participants.columns
    assert result.verification.schema_ok
    assert "verification" in (tmp_path / "out" / "reports" / "study_analysis_report.json").read_text(
        encoding="utf-8"
    )
