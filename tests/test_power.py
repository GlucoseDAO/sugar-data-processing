"""Reference values and boundaries for prospective recruitment calculations."""

from __future__ import annotations

import pytest

from sugar_data_processing.statistics.power import (
    correlation_power,
    enrollment_for_eligible,
    independent_t_power,
    paired_t_power,
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


def test_power_helpers_agree_with_required_sizes() -> None:
    assert paired_t_power(34, mean_difference=3, difference_sd=6) >= 0.8
    assert paired_t_power(33, mean_difference=3, difference_sd=6) < 0.8
    assert correlation_power(124, 0.25) == pytest.approx(0.8, abs=0.01)
    assert correlation_power(25, -0.25) < 0.3
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


def test_power_near_one_is_a_number_not_nan() -> None:
    # scipy's noncentral t gives NaN for the far lower tail at these sizes.
    assert paired_t_power(500, mean_difference=3.0, difference_sd=6.0) == pytest.approx(1.0)
    assert independent_t_power(400, 400, mean_difference=3.5, sd=4.0) == pytest.approx(1.0)
