"""Unit checks for the §7 test primitives."""

from __future__ import annotations

import numpy as np

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
