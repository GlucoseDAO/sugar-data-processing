"""Effect-size helpers for the study-design tests."""

from __future__ import annotations

import numpy as np


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Independent-samples Cohen's d (pooled SD)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    pooled = np.sqrt(((a.size - 1) * var_a + (b.size - 1) * var_b) / (a.size + b.size - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def cohens_d_paired(diff: np.ndarray) -> float:
    """Paired Cohen's d = mean(diff) / sd(diff)."""
    diff = np.asarray(diff, dtype=float)
    if diff.size < 2:
        return float("nan")
    sd = float(np.std(diff, ddof=1))
    if sd == 0:
        return 0.0
    return float(np.mean(diff) / sd)


def rank_biserial_from_u(u: float, n1: int, n2: int) -> float:
    """Mann–Whitney rank-biserial correlation from the U statistic."""
    if n1 == 0 or n2 == 0:
        return float("nan")
    return float(1.0 - (2.0 * u) / (n1 * n2))
