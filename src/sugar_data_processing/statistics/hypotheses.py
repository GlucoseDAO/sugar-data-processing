"""Run H1–H5 exactly as specified in the study design §7.3–7.4.

H6 (human vs baseline models) is deferred until computational baselines
are implemented in sugar-sugar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from eliot import start_action

from sugar_data_processing.gathering.participants import (
    h5_paired_population,
    primary_analysis_population,
)
from sugar_data_processing.statistics.tests import (
    CorrelationResult,
    GroupComparisonResult,
    PairedComparisonResult,
    correlation_analysis,
    independent_group_comparison,
    paired_comparison,
)


@dataclass(frozen=True)
class HypothesisSuite:
    h1: GroupComparisonResult | None
    h2: GroupComparisonResult | None
    h3: CorrelationResult | None
    h4: CorrelationResult | None
    h5: PairedComparisonResult | None
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "h1": None if self.h1 is None else self.h1.to_dict(),
            "h2": None if self.h2 is None else self.h2.to_dict(),
            "h3": None if self.h3 is None else self.h3.to_dict(),
            "h4": None if self.h4 is None else self.h4.to_dict(),
            "h5": None if self.h5 is None else self.h5.to_dict(),
            "h6": {
                "status": "deferred",
                "note": (
                    "H6 (human vs baseline models) is deferred per study design §7.4 "
                    "until persistence / linear-extrapolation / ARIMA baselines are available."
                ),
            },
            "notes": self.notes,
        }


def _series(df: pl.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns or df.height == 0:
        return np.asarray([], dtype=float)
    return df[col].to_numpy().astype(float)


def run_all_hypotheses(participants: pl.DataFrame) -> HypothesisSuite:
    """Execute the full §7 hypothesis battery on a participant table."""
    with start_action(action_type="statistics.run_all_hypotheses") as action:
        notes: list[str] = []
        primary = primary_analysis_population(participants)
        if primary.height == 0:
            notes.append(
                "No participants meet the primary analysis threshold "
                "(≥6 generic segments). H1–H4 skipped."
            )
            action.log(message_type="warning", reason="empty_primary")
            # Still attempt H5 if possible
            h5 = _run_h5(participants, notes)
            return HypothesisSuite(h1=None, h2=None, h3=None, h4=None, h5=h5, notes=notes)

        h1 = _run_h1(primary, notes)
        h2 = _run_h2(primary, notes)
        h3 = _run_h3(primary, notes)
        h4 = _run_h4(primary, notes)
        h5 = _run_h5(participants, notes)
        action.log(
            message_type="info",
            n_primary=primary.height,
            h1_sig=None if h1 is None else h1.significant,
            h2_sig=None if h2 is None else h2.significant,
            h5_sig=None if h5 is None else h5.significant,
        )
        return HypothesisSuite(h1=h1, h2=h2, h3=h3, h4=h4, h5=h5, notes=notes)


def _run_h1(primary: pl.DataFrame, notes: list[str]) -> GroupComparisonResult | None:
    pwd = primary.filter(pl.col("diabetic") == True)  # noqa: E712
    non = primary.filter(pl.col("diabetic") == False)  # noqa: E712
    if pwd.height < 2 or non.height < 2:
        notes.append(f"H1 skipped: need ≥2 per group (PwD={pwd.height}, non-PwD={non.height}).")
        return None
    return independent_group_comparison(
        _series(pwd, "mae_primary"),
        _series(non, "mae_primary"),
        hypothesis="H1",
        label_a="PwD",
        label_b="non-PwD",
    )


def _run_h2(primary: pl.DataFrame, notes: list[str]) -> GroupComparisonResult | None:
    cgm = primary.filter(pl.col("uses_cgm") == True)  # noqa: E712
    no_cgm = primary.filter(pl.col("uses_cgm") == False)  # noqa: E712
    if cgm.height < 2 or no_cgm.height < 2:
        notes.append(
            f"H2 skipped: need ≥2 per group (CGM={cgm.height}, non-CGM={no_cgm.height})."
        )
        return None
    return independent_group_comparison(
        _series(cgm, "mae_primary"),
        _series(no_cgm, "mae_primary"),
        hypothesis="H2",
        label_a="CGM users",
        label_b="non-CGM",
    )


def _run_h3(primary: pl.DataFrame, notes: list[str]) -> CorrelationResult | None:
    pwd = primary.filter(pl.col("diabetic") == True).filter(  # noqa: E712
        pl.col("diabetes_duration").is_not_null()
    )
    if pwd.height < 3:
        notes.append(f"H3 skipped: need ≥3 PwD with duration (n={pwd.height}).")
        return None
    return correlation_analysis(
        _series(pwd, "diabetes_duration"),
        _series(pwd, "mae_primary"),
        hypothesis="H3",
        predictor="diabetes_duration_years",
    )


def _run_h4(primary: pl.DataFrame, notes: list[str]) -> CorrelationResult | None:
    cgm = primary.filter(pl.col("uses_cgm") == True).filter(  # noqa: E712
        pl.col("cgm_duration_years").is_not_null()
    )
    if cgm.height < 3:
        notes.append(f"H4 skipped: need ≥3 CGM users with experience years (n={cgm.height}).")
        return None
    return correlation_analysis(
        _series(cgm, "cgm_duration_years"),
        _series(cgm, "mae_primary"),
        hypothesis="H4",
        predictor="cgm_duration_years",
    )


def _run_h5(participants: pl.DataFrame, notes: list[str]) -> PairedComparisonResult | None:
    paired = h5_paired_population(participants)
    if paired.height < 2:
        # Relaxed fallback: any participant with both MAE values (for early data)
        fallback = participants.filter(
            pl.col("mae_generic").is_not_null() & pl.col("mae_own").is_not_null()
        )
        if fallback.height < 2:
            notes.append(
                f"H5 skipped: need ≥2 participants with both own and generic MAE "
                f"(strict eligible={paired.height}, any both={fallback.height})."
            )
            return None
        notes.append(
            f"H5 used relaxed pairing (n={fallback.height}); "
            "strict §7.2 threshold (≥6 segments each side) not yet met."
        )
        paired = fallback
    return paired_comparison(
        _series(paired, "mae_generic"),
        _series(paired, "mae_own"),
        hypothesis="H5",
        label_a="generic data",
        label_b="own data",
    )
