"""Run H1–H5 exactly as specified in the study design §7.3–7.4.

H1–H4 are reported category-first (the four diabetes × CGM buckets), then
the same comparison is repeated on generic data and on own data. H3 uses
diabetes duration in months so short experience is not crushed on a year axis.

H6 (human vs baseline models) is deferred until computational baselines
are implemented in sugar-sugar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from eliot import start_action

from sugar_data_processing.config import (
    COHORT_LABELS,
    MAX_PLAUSIBLE_CGM_YEARS,
    MAX_PLAUSIBLE_DIABETES_YEARS,
)
from sugar_data_processing.gathering.participants import (
    h5_paired_population,
    own_analysis_population,
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

HypothesisLayer = GroupComparisonResult | CorrelationResult


@dataclass(frozen=True)
class LayeredHypothesisResult:
    """One hypothesis with a category snapshot, then overall / generic / own layers."""

    hypothesis: str
    by_category: list[dict[str, Any]]
    overall: HypothesisLayer | None
    generic: HypothesisLayer | None
    own: HypothesisLayer | None

    def to_dict(self) -> dict[str, Any]:
        overall_dict = None if self.overall is None else self.overall.to_dict()
        payload: dict[str, Any] = {
            "hypothesis": self.hypothesis,
            "by_category": self.by_category,
            "overall": overall_dict,
            "generic": None if self.generic is None else self.generic.to_dict(),
            "own": None if self.own is None else self.own.to_dict(),
        }
        if overall_dict is not None:
            for key, value in overall_dict.items():
                payload.setdefault(key, value)
        return payload

    def __getattr__(self, name: str) -> Any:
        if self.overall is not None and hasattr(self.overall, name):
            return getattr(self.overall, name)
        raise AttributeError(name)


@dataclass(frozen=True)
class HypothesisSuite:
    h1: LayeredHypothesisResult
    h2: LayeredHypothesisResult
    h3: LayeredHypothesisResult
    h4: LayeredHypothesisResult
    h5: PairedComparisonResult | None
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "h1": self.h1.to_dict(),
            "h2": self.h2.to_dict(),
            "h3": self.h3.to_dict(),
            "h4": self.h4.to_dict(),
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


def _mean_col(df: pl.DataFrame, col: str) -> float | None:
    if col not in df.columns or df.height == 0:
        return None
    values = df.filter(pl.col(col).is_not_null())[col]
    if values.len() == 0:
        return None
    value = values.mean()
    return None if value is None else float(value)


def _count_non_null(df: pl.DataFrame, col: str) -> int:
    if col not in df.columns or df.height == 0:
        return 0
    return int(df.filter(pl.col(col).is_not_null()).height)


def category_prediction_summary(participants: pl.DataFrame) -> list[dict[str, Any]]:
    """Per diabetes × CGM bucket: head-count and mean MAE on overall / generic / own."""
    rows: list[dict[str, Any]] = []
    for key, label in COHORT_LABELS.items():
        if "cohort_category" in participants.columns:
            sub = participants.filter(pl.col("cohort_category") == key)
        else:
            sub = participants.head(0)
        rows.append(
            {
                "category": key,
                "label": label,
                "n": sub.height,
                "n_generic": _count_non_null(sub, "mae_generic"),
                "n_own": _count_non_null(sub, "mae_own"),
                "mean_mae_primary": _mean_col(sub, "mae_primary"),
                "mean_mae_generic": _mean_col(sub, "mae_generic"),
                "mean_mae_own": _mean_col(sub, "mae_own"),
            }
        )
    return rows


def _empty_layer(hypothesis: str, participants: pl.DataFrame) -> LayeredHypothesisResult:
    return LayeredHypothesisResult(
        hypothesis=hypothesis,
        by_category=category_prediction_summary(participants),
        overall=None,
        generic=None,
        own=None,
    )


def _own_population(participants: pl.DataFrame, notes: list[str], hypothesis: str) -> pl.DataFrame:
    own = own_analysis_population(participants)
    if own.height >= 2:
        return own
    fallback = participants.filter(pl.col("mae_own").is_not_null())
    if fallback.height >= 2:
        notes.append(
            f"{hypothesis} own-data layer used relaxed pairing "
            f"(n={fallback.height}); strict ≥6 own segments not yet met."
        )
    return fallback


def run_all_hypotheses(participants: pl.DataFrame) -> HypothesisSuite:
    """Execute the full §7 hypothesis battery on a participant table."""
    with start_action(action_type="statistics.run_all_hypotheses") as action:
        notes: list[str] = []
        primary = primary_analysis_population(participants)
        if primary.height == 0:
            notes.append(
                "No participants meet the primary analysis threshold "
                "(≥6 generic segments). H1–H4 overall/generic layers skipped."
            )
            action.log(message_type="warning", reason="empty_primary")
            h5 = _run_h5(participants, notes)
            return HypothesisSuite(
                h1=_empty_layer("H1", participants),
                h2=_empty_layer("H2", participants),
                h3=_empty_layer("H3", participants),
                h4=_empty_layer("H4", participants),
                h5=h5,
                notes=notes,
            )

        h1 = _run_h1(participants, primary, notes)
        h2 = _run_h2(participants, primary, notes)
        h3 = _run_h3(participants, primary, notes)
        h4 = _run_h4(participants, primary, notes)
        h5 = _run_h5(participants, notes)
        action.log(
            message_type="info",
            n_primary=primary.height,
            h1_sig=None if h1.overall is None else h1.overall.significant,
            h2_sig=None if h2.overall is None else h2.overall.significant,
            h5_sig=None if h5 is None else h5.significant,
        )
        return HypothesisSuite(h1=h1, h2=h2, h3=h3, h4=h4, h5=h5, notes=notes)


def _run_group_layer(
    frame: pl.DataFrame,
    *,
    group_col: str,
    mae_col: str,
    hypothesis: str,
    label_a: str,
    label_b: str,
    notes: list[str],
    layer: str,
) -> GroupComparisonResult | None:
    if mae_col not in frame.columns:
        notes.append(f"{hypothesis} {layer} skipped: missing `{mae_col}`.")
        return None
    yes = frame.filter(pl.col(group_col) == True).filter(pl.col(mae_col).is_not_null())  # noqa: E712
    no = frame.filter(pl.col(group_col) == False).filter(pl.col(mae_col).is_not_null())  # noqa: E712
    if yes.height < 2 or no.height < 2:
        notes.append(
            f"{hypothesis} {layer} skipped: need ≥2 per group "
            f"({label_a}={yes.height}, {label_b}={no.height})."
        )
        return None
    return independent_group_comparison(
        _series(yes, mae_col),
        _series(no, mae_col),
        hypothesis=f"{hypothesis}:{layer}",
        label_a=label_a,
        label_b=label_b,
    )


def _run_h1(
    participants: pl.DataFrame, primary: pl.DataFrame, notes: list[str]
) -> LayeredHypothesisResult:
    own = _own_population(participants, notes, "H1")
    return LayeredHypothesisResult(
        hypothesis="H1",
        by_category=category_prediction_summary(participants),
        overall=_run_group_layer(
            primary,
            group_col="diabetic",
            mae_col="mae_primary",
            hypothesis="H1",
            label_a="PwD",
            label_b="non-PwD",
            notes=notes,
            layer="overall",
        ),
        generic=_run_group_layer(
            primary,
            group_col="diabetic",
            mae_col="mae_generic",
            hypothesis="H1",
            label_a="PwD",
            label_b="non-PwD",
            notes=notes,
            layer="generic",
        ),
        own=_run_group_layer(
            own,
            group_col="diabetic",
            mae_col="mae_own",
            hypothesis="H1",
            label_a="PwD",
            label_b="non-PwD",
            notes=notes,
            layer="own",
        ),
    )


def _run_h2(
    participants: pl.DataFrame, primary: pl.DataFrame, notes: list[str]
) -> LayeredHypothesisResult:
    own = _own_population(participants, notes, "H2")
    return LayeredHypothesisResult(
        hypothesis="H2",
        by_category=category_prediction_summary(participants),
        overall=_run_group_layer(
            primary,
            group_col="uses_cgm",
            mae_col="mae_primary",
            hypothesis="H2",
            label_a="CGM users",
            label_b="non-CGM",
            notes=notes,
            layer="overall",
        ),
        generic=_run_group_layer(
            primary,
            group_col="uses_cgm",
            mae_col="mae_generic",
            hypothesis="H2",
            label_a="CGM users",
            label_b="non-CGM",
            notes=notes,
            layer="generic",
        ),
        own=_run_group_layer(
            own,
            group_col="uses_cgm",
            mae_col="mae_own",
            hypothesis="H2",
            label_a="CGM users",
            label_b="non-CGM",
            notes=notes,
            layer="own",
        ),
    )


def _run_correlation_layer(
    frame: pl.DataFrame,
    *,
    filter_col: str,
    x_col: str,
    mae_col: str,
    hypothesis: str,
    predictor: str,
    notes: list[str],
    layer: str,
    min_n: int = 3,
) -> CorrelationResult | None:
    if x_col not in frame.columns or mae_col not in frame.columns:
        notes.append(f"{hypothesis} {layer} skipped: missing `{x_col}` or `{mae_col}`.")
        return None
    subset = frame.filter(pl.col(filter_col) == True).filter(  # noqa: E712
        pl.col(x_col).is_not_null() & pl.col(mae_col).is_not_null()
    )
    if x_col == "diabetes_duration_months":
        cap = MAX_PLAUSIBLE_DIABETES_YEARS * 12.0
        dropped = subset.filter(pl.col(x_col) > cap).height
        subset = subset.filter(pl.col(x_col) <= cap)
        if dropped:
            notes.append(
                f"{hypothesis} {layer}: dropped {dropped} row(s) with duration "
                f"> {MAX_PLAUSIBLE_DIABETES_YEARS:g} years so the months scale stays readable."
            )
    elif x_col == "cgm_duration_months":
        cap = MAX_PLAUSIBLE_CGM_YEARS * 12.0
        dropped = subset.filter(pl.col(x_col) > cap).height
        subset = subset.filter(pl.col(x_col) <= cap)
        if dropped:
            notes.append(
                f"{hypothesis} {layer}: dropped {dropped} row(s) with CGM experience "
                f"> {MAX_PLAUSIBLE_CGM_YEARS:g} years so the months scale stays readable."
            )
    if subset.height < min_n:
        notes.append(
            f"{hypothesis} {layer} skipped: need ≥{min_n} with {predictor} and {mae_col} "
            f"(n={subset.height})."
        )
        return None
    return correlation_analysis(
        _series(subset, x_col),
        _series(subset, mae_col),
        hypothesis=f"{hypothesis}:{layer}",
        predictor=predictor,
    )


def _with_duration_months(df: pl.DataFrame) -> pl.DataFrame:
    if "diabetes_duration_months" in df.columns:
        out = df
    elif "diabetes_duration" in df.columns:
        out = df.with_columns((pl.col("diabetes_duration") * 12.0).alias("diabetes_duration_months"))
    else:
        out = df
    if "cgm_duration_months" in out.columns:
        return out
    if "cgm_duration_years" not in out.columns:
        return out
    return out.with_columns((pl.col("cgm_duration_years") * 12.0).alias("cgm_duration_months"))


def _run_h3(
    participants: pl.DataFrame, primary: pl.DataFrame, notes: list[str]
) -> LayeredHypothesisResult:
    primary = _with_duration_months(primary)
    own = _with_duration_months(_own_population(participants, notes, "H3"))
    return LayeredHypothesisResult(
        hypothesis="H3",
        by_category=category_prediction_summary(participants),
        overall=_run_correlation_layer(
            primary,
            filter_col="diabetic",
            x_col="diabetes_duration_months",
            mae_col="mae_primary",
            hypothesis="H3",
            predictor="diabetes_duration_months",
            notes=notes,
            layer="overall",
        ),
        generic=_run_correlation_layer(
            primary,
            filter_col="diabetic",
            x_col="diabetes_duration_months",
            mae_col="mae_generic",
            hypothesis="H3",
            predictor="diabetes_duration_months",
            notes=notes,
            layer="generic",
        ),
        own=_run_correlation_layer(
            own,
            filter_col="diabetic",
            x_col="diabetes_duration_months",
            mae_col="mae_own",
            hypothesis="H3",
            predictor="diabetes_duration_months",
            notes=notes,
            layer="own",
        ),
    )


def _run_h4(
    participants: pl.DataFrame, primary: pl.DataFrame, notes: list[str]
) -> LayeredHypothesisResult:
    primary = _with_duration_months(primary)
    own = _with_duration_months(_own_population(participants, notes, "H4"))
    return LayeredHypothesisResult(
        hypothesis="H4",
        by_category=category_prediction_summary(participants),
        overall=_run_correlation_layer(
            primary,
            filter_col="uses_cgm",
            x_col="cgm_duration_months",
            mae_col="mae_primary",
            hypothesis="H4",
            predictor="cgm_duration_months",
            notes=notes,
            layer="overall",
        ),
        generic=_run_correlation_layer(
            primary,
            filter_col="uses_cgm",
            x_col="cgm_duration_months",
            mae_col="mae_generic",
            hypothesis="H4",
            predictor="cgm_duration_months",
            notes=notes,
            layer="generic",
        ),
        own=_run_correlation_layer(
            own,
            filter_col="uses_cgm",
            x_col="cgm_duration_months",
            mae_col="mae_own",
            hypothesis="H4",
            predictor="cgm_duration_months",
            notes=notes,
            layer="own",
        ),
    )


def _run_h5(participants: pl.DataFrame, notes: list[str]) -> PairedComparisonResult | None:
    paired = h5_paired_population(participants)
    if paired.height < 2:
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
