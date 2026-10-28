"""Prospective sample sizes for independent people, never prediction rounds.

The t-test calculations use noncentral t distributions under equal-variance
normal planning assumptions. Correlation planning uses the Fisher-z approximation.
These are planning models, not guarantees for skewed outcomes or rank tests.
"""

from __future__ import annotations

from collections.abc import Callable
from math import atanh, ceil, isfinite, sqrt

from scipy import stats

from sugar_data_processing.config import ALPHA, PRIMARY_ALPHA


def _probability(value: float, name: str) -> None:
    if not isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be between 0 and 1, exclusively")


def _effect(mean_difference: float, sd: float) -> float:
    if not isfinite(sd) or sd <= 0:
        raise ValueError("sd must be finite and positive")
    if not isfinite(mean_difference) or mean_difference == 0:
        raise ValueError("mean_difference must be finite and nonzero")
    return abs(mean_difference) / sd


def _two_sided_power(df: float, noncentrality: float, alpha: float) -> float:
    critical = stats.t.isf(alpha / 2.0, df)
    return float(
        stats.nct.sf(critical, df, noncentrality)
        + stats.nct.cdf(-critical, df, noncentrality)
    )


def independent_t_power(
    n_a: int,
    n_b: int,
    *,
    mean_difference: float,
    sd: float,
    alpha: float = PRIMARY_ALPHA,
) -> float:
    """Two-sided independent t-test power for specified person counts."""
    _probability(alpha, "alpha")
    effect = _effect(mean_difference, sd)
    if min(n_a, n_b) < 2:
        raise ValueError("Each group needs at least two people")
    noncentrality = effect / sqrt(1.0 / n_a + 1.0 / n_b)
    return _two_sided_power(n_a + n_b - 2, noncentrality, alpha)


def _minimum_n(power_at_n: Callable[[int], float], target: float) -> int:
    _probability(target, "power")
    lower, upper = 1, 2
    while power_at_n(upper) < target:
        lower, upper = upper, 2 * upper
        if upper > 10_000_000:
            raise ValueError("Required sample exceeds the planning limit")
    while upper - lower > 1:
        middle = (lower + upper) // 2
        if power_at_n(middle) >= target:
            upper = middle
        else:
            lower = middle
    return upper


def required_independent_n(
    *,
    mean_difference: float,
    sd: float,
    power: float = 0.80,
    alpha: float = PRIMARY_ALPHA,
    ratio: float = 1.0,
) -> tuple[int, int]:
    """Smallest n_a and ceil(ratio*n_a) reaching target power.

    ``ratio`` is n_b/n_a; 1 means equally sized groups. Each group has at
    least two people. The supplied SD is the common between-person SD.
    """
    if not isfinite(ratio) or ratio <= 0:
        raise ValueError("ratio must be finite and positive")

    def second_group(n: int) -> int:
        return max(2, ceil(ratio * n))

    n_a = _minimum_n(
        lambda n: independent_t_power(
            n,
            second_group(n),
            mean_difference=mean_difference,
            sd=sd,
            alpha=alpha,
        ),
        power,
    )
    return n_a, second_group(n_a)


def required_paired_n(
    *,
    mean_difference: float,
    difference_sd: float,
    power: float = 0.80,
    alpha: float = ALPHA,
) -> int:
    """People with both conditions, using SD of within-person differences."""
    _probability(alpha, "alpha")
    effect = _effect(mean_difference, difference_sd)
    return _minimum_n(
        lambda n: _two_sided_power(n - 1, effect * sqrt(n), alpha), power
    )


def required_correlation_n(
    correlation: float, *, power: float = 0.80, alpha: float = ALPHA
) -> int:
    """Approximate complete pairs for a two-sided Pearson correlation test.

    This is not an exact Spearman or nonlinear learning-curve calculation.
    For H3/H4 the count applies inside the PwD/CGM subgroup, respectively.
    """
    _probability(alpha, "alpha")
    _probability(power, "power")
    if not isfinite(correlation) or not 0.0 < abs(correlation) < 1.0:
        raise ValueError("Absolute correlation must be between 0 and 1, exclusively")
    z = stats.norm.isf(alpha / 2.0) + stats.norm.ppf(power)
    return ceil(3.0 + (z / atanh(abs(correlation))) ** 2)


def enrollment_for_eligible(n_eligible: int, *, unusable_fraction: float = 0.10) -> int:
    """Inflate a completer target for an explicitly assumed unusable fraction."""
    if n_eligible < 1:
        raise ValueError("n_eligible must be positive")
    if not isfinite(unusable_fraction) or not 0.0 <= unusable_fraction < 1.0:
        raise ValueError("unusable_fraction must be between 0 inclusive and 1 exclusive")
    return ceil(n_eligible / (1.0 - unusable_fraction))
