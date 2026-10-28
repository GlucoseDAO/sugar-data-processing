"""Unit checks for the §7 test primitives."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

from sugar_data_processing.fixtures.synthetic import write_synthetic_csv
from sugar_data_processing.gathering import build_participant_table, load_prediction_statistics
from sugar_data_processing.output.narration import explain_hypothesis_result
from sugar_data_processing.statistics.hypotheses import _run_group_layer, run_all_hypotheses
from sugar_data_processing.statistics.tests import (
    correlation_analysis,
    independent_group_comparison,
    paired_comparison,
)


def test_independent_group_detects_shift() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(10, 2, size=40)
    b = rng.normal(14, 2, size=40)
    result = independent_group_comparison(a, b, hypothesis="H1", label_a="A", label_b="B")
    assert result.significant
    assert result.mean_a < result.mean_b


def test_paired_detects_own_advantage() -> None:
    rng = np.random.default_rng(1)
    own = rng.normal(12, 2, size=30)
    generic = own + rng.uniform(2, 4, size=30)
    result = paired_comparison(
        generic, own, hypothesis="H5", label_a="generic", label_b="own"
    )
    assert result.significant
    assert result.mean_diff > 0


def test_correlation_negative_duration() -> None:
    rng = np.random.default_rng(2)
    duration = rng.uniform(1, 20, size=50)
    mae = 25 - 0.5 * duration + rng.normal(0, 1.0, size=50)
    result = correlation_analysis(
        duration, mae, hypothesis="H3", predictor="duration"
    )
    assert result.coefficient < 0
    assert result.significant


def test_h1_h4_are_category_then_own_vs_generic(tmp_path: Path) -> None:
    csv_path = tmp_path / "prediction_statistics.csv"
    write_synthetic_csv(csv_path, n_participants=60, seed=7)
    people = build_participant_table(load_prediction_statistics(csv_path))
    suite = run_all_hypotheses(people)
    payload = suite.to_dict()

    assert "diabetes_duration_months" in people.columns
    pwd = people.filter(pl.col("diabetic") == True)  # noqa: E712
    if pwd.height:
        years = float(pwd["diabetes_duration"][0])
        months = float(pwd["diabetes_duration_months"][0])
        assert abs(months - years * 12.0) < 1e-6

    for key in ("h1", "h2", "h3", "h4"):
        layered = payload[key]
        assert isinstance(layered["by_category"], list)
        assert {row["category"] for row in layered["by_category"]}
        assert "overall" in layered
        assert "generic" in layered
        assert "own" in layered

    assert suite.h3.overall is not None
    assert suite.h3.overall.predictor == "diabetes_duration_months"
    assert suite.h4.overall is not None
    assert suite.h4.overall.predictor == "cgm_duration_months"
    assert suite.h1.overall is not None
    assert suite.h1.mean_a < suite.h1.mean_b
    assert suite.h1.overall.alpha == 0.025
    assert suite.h1.generic is not None and suite.h1.generic.alpha == 0.025
    assert suite.h2.overall is not None and suite.h2.overall.alpha == 0.025


def test_primary_p_between_point_025_and_point_05_is_not_significant() -> None:
    a = stats.norm.ppf((np.arange(40) + 0.5) / 40)
    shift = stats.t.isf(0.035 / 2, 78) * np.std(a, ddof=1) * np.sqrt(2 / 40)
    people = pl.DataFrame(
        {
            "diabetic": [True] * 40 + [False] * 40,
            "mae_generic": np.concatenate((a + 10, a + 10 + shift)),
        }
    )
    result = _run_group_layer(
        people,
        group_col="diabetic",
        mae_col="mae_generic",
        hypothesis="H1",
        label_a="PwD",
        label_b="non-PwD",
        notes=[],
        layer="generic",
    )
    assert result is not None
    assert 0.025 < result.p_value < 0.05
    assert not result.significant
    narration = explain_hypothesis_result(
        "h1",
        {
            "by_category": [],
            "overall": result.to_dict(),
            "generic": result.to_dict(),
            "own": None,
        },
    )
    assert "No — not statistically significant at α = 0.025" in narration


def test_missing_comparison_is_reported_as_not_run() -> None:
    narration = explain_hypothesis_result(
        "h1", {"by_category": [], "overall": None, "generic": None, "own": None}
    )
    assert "Category comparison (person MAE):** not run" in narration
    assert "Significant?: No" not in narration


def test_h5_does_not_silently_include_below_threshold_pairs(tmp_path: Path) -> None:
    csv_path = tmp_path / "stats.csv"
    write_synthetic_csv(csv_path, n_participants=40, seed=7)
    people = build_participant_table(load_prediction_statistics(csv_path)).with_columns(
        pl.lit(False).alias("eligible_h5")
    )
    assert people.filter(pl.col("mae_generic").is_not_null() & pl.col("mae_own").is_not_null()).height > 2
    suite = run_all_hypotheses(people)
    assert suite.h5 is None
    assert any("thresholds are not relaxed" in note for note in suite.notes)
