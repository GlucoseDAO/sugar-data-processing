"""Low-level statistical primitives matching study design §7."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from scipy import stats

from sugar_data_processing.config import ALPHA, SHAPIRO_NORMAL_P
from sugar_data_processing.statistics.effect_sizes import (
    cohens_d,
    cohens_d_paired,
    rank_biserial_from_u,
)


@dataclass(frozen=True)
class NormalityResult:
    group: str
    n: int
    statistic: float
    p_value: float
    is_normal: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GroupComparisonResult:
    hypothesis: str
    comparison: str
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    mean_diff: float
    normality_a: NormalityResult
    normality_b: NormalityResult
    test_used: Literal["independent_t", "mann_whitney_u"]
    statistic: float
    p_value: float
    effect_size_name: str
    effect_size: float
    alpha: float
    significant: bool
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass(frozen=True)
class CorrelationResult:
    hypothesis: str
    predictor: str
    n: int
    normality_outcome: NormalityResult
    test_used: Literal["pearson", "spearman"]
    coefficient: float
    p_value: float
    log_model_r2: float | None
    linear_model_r2: float | None
    alpha: float
    significant: bool
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairedComparisonResult:
    hypothesis: str
    comparison: str
    n: int
    mean_a: float
    mean_b: float
    mean_diff: float
    normality_diff: NormalityResult
    test_used: Literal["paired_t", "wilcoxon"]
    statistic: float
    p_value: float
    effect_size_name: str
    effect_size: float
    alpha: float
    significant: bool
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def shapiro_wilk(values: np.ndarray, group: str = "group") -> NormalityResult:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n < 3:
        return NormalityResult(group=group, n=n, statistic=float("nan"), p_value=float("nan"), is_normal=False)
    # Shapiro-Wilk is limited to n <= 5000
    sample = values if n <= 5000 else np.random.default_rng(0).choice(values, size=5000, replace=False)
    statistic, p_value = stats.shapiro(sample)
    return NormalityResult(
        group=group,
        n=n,
        statistic=float(statistic),
        p_value=float(p_value),
        is_normal=bool(p_value > SHAPIRO_NORMAL_P),
    )


def independent_group_comparison(
    a: np.ndarray,
    b: np.ndarray,
    *,
    hypothesis: str,
    label_a: str,
    label_b: str,
    alpha: float = ALPHA,
) -> GroupComparisonResult:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    norm_a = shapiro_wilk(a, group=label_a)
    norm_b = shapiro_wilk(b, group=label_b)
    both_normal = norm_a.is_normal and norm_b.is_normal and a.size >= 3 and b.size >= 3

    if both_normal:
        statistic, p_value = stats.ttest_ind(a, b, equal_var=False)
        test_used: Literal["independent_t", "mann_whitney_u"] = "independent_t"
        effect_name = "cohen_d"
        effect = cohens_d(a, b)
    else:
        statistic, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
        test_used = "mann_whitney_u"
        effect_name = "rank_biserial"
        effect = rank_biserial_from_u(float(statistic), a.size, b.size)

    mean_a = float(np.mean(a)) if a.size else float("nan")
    mean_b = float(np.mean(b)) if b.size else float("nan")
    mean_diff = mean_a - mean_b
    significant = bool(np.isfinite(p_value) and p_value < alpha)
    direction = (
        f"{label_a} lower MAE (better)"
        if mean_a < mean_b
        else f"{label_b} lower MAE (better)"
        if mean_b < mean_a
        else "groups similar"
    )
    interpretation = (
        f"{'Significant' if significant else 'Non-significant'} difference "
        f"(p={p_value:.4g}, {test_used}). Mean MAE {label_a}={mean_a:.2f}, "
        f"{label_b}={mean_b:.2f} mg/dL → {direction}."
    )
    return GroupComparisonResult(
        hypothesis=hypothesis,
        comparison=f"{label_a} vs {label_b}",
        n_a=int(a.size),
        n_b=int(b.size),
        mean_a=mean_a,
        mean_b=mean_b,
        mean_diff=mean_diff,
        normality_a=norm_a,
        normality_b=norm_b,
        test_used=test_used,
        statistic=float(statistic),
        p_value=float(p_value),
        effect_size_name=effect_name,
        effect_size=float(effect),
        alpha=alpha,
        significant=significant,
        interpretation=interpretation,
    )


def _r_squared(y: np.ndarray, y_hat: np.ndarray) -> float:
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def correlation_analysis(
    x: np.ndarray,
    y: np.ndarray,
    *,
    hypothesis: str,
    predictor: str,
    alpha: float = ALPHA,
    force_spearman: bool = False,
) -> CorrelationResult:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    norm_y = shapiro_wilk(y, group="MAE")

    # Prefer Spearman when non-normal or when caller forces it (curved learning)
    use_pearson = (not force_spearman) and norm_y.is_normal and x.size >= 3
    if use_pearson:
        coefficient, p_value = stats.pearsonr(x, y)
        test_used: Literal["pearson", "spearman"] = "pearson"
    else:
        coefficient, p_value = stats.spearmanr(x, y)
        test_used = "spearman"

    linear_r2: float | None = None
    log_r2: float | None = None
    if x.size >= 3:
        slope, intercept = np.polyfit(x, y, 1)
        linear_r2 = _r_squared(y, slope * x + intercept)
        log_x = np.log(x + 1.0)
        log_slope, log_intercept = np.polyfit(log_x, y, 1)
        log_r2 = _r_squared(y, log_slope * log_x + log_intercept)

    significant = bool(np.isfinite(p_value) and p_value < alpha)
    sign = "negative (longer experience → lower MAE)" if coefficient < 0 else "positive"
    interpretation = (
        f"{'Significant' if significant else 'Non-significant'} {test_used} correlation "
        f"(ρ/r={coefficient:.3f}, p={p_value:.4g}, n={x.size}). Direction: {sign}."
    )
    return CorrelationResult(
        hypothesis=hypothesis,
        predictor=predictor,
        n=int(x.size),
        normality_outcome=norm_y,
        test_used=test_used,
        coefficient=float(coefficient),
        p_value=float(p_value),
        log_model_r2=None if log_r2 is None else float(log_r2),
        linear_model_r2=None if linear_r2 is None else float(linear_r2),
        alpha=alpha,
        significant=significant,
        interpretation=interpretation,
    )


def paired_comparison(
    a: np.ndarray,
    b: np.ndarray,
    *,
    hypothesis: str,
    label_a: str,
    label_b: str,
    alpha: float = ALPHA,
) -> PairedComparisonResult:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    diff = a - b
    norm_diff = shapiro_wilk(diff, group="difference")

    if norm_diff.is_normal and diff.size >= 3:
        statistic, p_value = stats.ttest_rel(a, b)
        test_used: Literal["paired_t", "wilcoxon"] = "paired_t"
    else:
        # zero_method='wilcox' drops exact-zero differences
        if np.allclose(diff, 0) or diff.size < 1:
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
        test_used = "wilcoxon"

    mean_a = float(np.mean(a)) if a.size else float("nan")
    mean_b = float(np.mean(b)) if b.size else float("nan")
    mean_diff = float(np.mean(diff)) if diff.size else float("nan")
    effect = cohens_d_paired(diff)
    significant = bool(np.isfinite(p_value) and p_value < alpha)
    better = label_b if mean_diff > 0 else label_a if mean_diff < 0 else "neither"
    interpretation = (
        f"{'Significant' if significant else 'Non-significant'} paired difference "
        f"(p={p_value:.4g}, {test_used}). Mean {label_a}={mean_a:.2f}, "
        f"{label_b}={mean_b:.2f} mg/dL (diff {label_a}−{label_b}={mean_diff:.2f}). "
        f"Lower MAE on: {better}."
    )
    return PairedComparisonResult(
        hypothesis=hypothesis,
        comparison=f"{label_a} vs {label_b}",
        n=int(a.size),
        mean_a=mean_a,
        mean_b=mean_b,
        mean_diff=mean_diff,
        normality_diff=norm_diff,
        test_used=test_used,
        statistic=float(statistic),
        p_value=float(p_value),
        effect_size_name="cohen_d_paired",
        effect_size=float(effect),
        alpha=alpha,
        significant=significant,
        interpretation=interpretation,
    )
