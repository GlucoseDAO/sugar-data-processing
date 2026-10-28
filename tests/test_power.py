"""Reference values and boundaries for prospective recruitment calculations."""

from __future__ import annotations

import pytest

from sugar_data_processing.statistics.power import (
    enrollment_for_eligible,
    independent_t_power,
    required_correlation_n,
    required_independent_n,
    required_paired_n,
)


@pytest.mark.parametrize("sd, expected", [(8, 78), (10, 121), (12, 173), (15, 269)])
def test_primary_sample_sizes(sd: float, expected: int) -> None:
    # Independent statsmodels TTestIndPower reference values, two-sided alpha .025.
    assert required_independent_n(mean_difference=4, sd=sd) == (expected, expected)
    assert independent_t_power(expected, expected, mean_difference=4, sd=sd) >= 0.8
    assert independent_t_power(expected - 1, expected - 1, mean_difference=4, sd=sd) < 0.8


def test_original_target_does_not_deliver_eighty_percent_power() -> None:
    assert independent_t_power(100, 100, mean_difference=4, sd=10) == pytest.approx(
        0.7153425633
    )


def test_imbalance_needs_more_people() -> None:
    n_a, n_b = required_independent_n(mean_difference=4, sd=10, ratio=2)
    assert n_b == 2 * n_a
    assert n_a + n_b > 242
    assert independent_t_power(n_a, n_b, mean_difference=4, sd=10) >= 0.8


def test_secondary_targets_and_enrollment() -> None:
    assert required_paired_n(mean_difference=3, difference_sd=6) == 34
    assert required_correlation_n(0.25) == 124
    assert required_correlation_n(-0.20) == 194

    assert enrollment_for_eligible(350) == 389
    assert enrollment_for_eligible(350, unusable_fraction=0) == 350


def test_invalid_planning_inputs() -> None:
    with pytest.raises(ValueError):
        required_independent_n(mean_difference=4, sd=0)
    with pytest.raises(ValueError):
        required_independent_n(mean_difference=4, sd=10, ratio=-1)
    with pytest.raises(ValueError):
        required_independent_n(mean_difference=4, sd=10, power=1)
    with pytest.raises(ValueError):
        required_correlation_n(0)
    with pytest.raises(ValueError):
        enrollment_for_eligible(350, unusable_fraction=1)
