"""Unit checks for the §7 test primitives."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from sugar_data_processing.fixtures.synthetic import write_synthetic_csv
from sugar_data_processing.gathering import build_participant_table, load_prediction_statistics
from sugar_data_processing.statistics.hypotheses import run_all_hypotheses
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
