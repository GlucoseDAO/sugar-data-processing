"""Place human MAE against published GlucoBench / literature bands (§7.5)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import polars as pl

from sugar_data_processing.config import (
    DEEP_LEARNING_MAE_RANGE,
    PERSONALIZED_MAE_RANGE,
    SIMPLE_BASELINE_MAE_RANGE,
)


@dataclass(frozen=True)
class BenchmarkContext:
    n: int
    human_mean_mae: float
    human_median_mae: float
    human_sd_mae: float
    pct_below_simple_baseline_low: float
    pct_inside_simple_baseline_band: float
    pct_inside_deep_learning_band: float
    pct_inside_personalized_band: float
    simple_baseline_mae_range: tuple[float, float]
    deep_learning_mae_range: tuple[float, float]
    personalized_mae_range: tuple[float, float]
    narrative: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def benchmark_context(participants: pl.DataFrame, mae_col: str = "mae_primary") -> BenchmarkContext:
    values = (
        participants.filter(pl.col(mae_col).is_not_null())[mae_col]
        .to_numpy()
        .astype(float)
    )
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return BenchmarkContext(
            n=0,
            human_mean_mae=float("nan"),
            human_median_mae=float("nan"),
            human_sd_mae=float("nan"),
            pct_below_simple_baseline_low=float("nan"),
            pct_inside_simple_baseline_band=float("nan"),
            pct_inside_deep_learning_band=float("nan"),
            pct_inside_personalized_band=float("nan"),
            simple_baseline_mae_range=SIMPLE_BASELINE_MAE_RANGE,
            deep_learning_mae_range=DEEP_LEARNING_MAE_RANGE,
            personalized_mae_range=PERSONALIZED_MAE_RANGE,
            narrative="No human MAE values available for benchmark comparison.",
        )

    def _pct_in(lo: float, hi: float) -> float:
        return float(100.0 * np.mean((values >= lo) & (values <= hi)))

    mean_mae = float(np.mean(values))
    median_mae = float(np.median(values))
    sd_mae = float(np.std(values, ddof=1)) if n > 1 else 0.0
    simple_lo, simple_hi = SIMPLE_BASELINE_MAE_RANGE
    dl_lo, dl_hi = DEEP_LEARNING_MAE_RANGE
    pers_lo, pers_hi = PERSONALIZED_MAE_RANGE

    if mean_mae < simple_lo:
        band = "better than the published simple-baseline band"
    elif mean_mae <= simple_hi:
        band = "inside the published simple-baseline band"
    elif mean_mae <= dl_hi:
        band = "near / inside the deep-learning band"
    else:
        band = "worse than typical published model bands"

    narrative = (
        f"Human mean MAE is {mean_mae:.1f} mg/dL (median {median_mae:.1f}, SD {sd_mae:.1f}, n={n}), "
        f"which sits {band}. Reference 60-min bands — simple/ARIMA "
        f"{simple_lo:.0f}–{simple_hi:.0f}, deep learning {dl_lo:.0f}–{dl_hi:.0f}, "
        f"personalized {pers_lo:.0f}–{pers_hi:.0f} mg/dL (study design §7.5)."
    )
    return BenchmarkContext(
        n=n,
        human_mean_mae=mean_mae,
        human_median_mae=median_mae,
        human_sd_mae=sd_mae,
        pct_below_simple_baseline_low=float(100.0 * np.mean(values < simple_lo)),
        pct_inside_simple_baseline_band=_pct_in(simple_lo, simple_hi),
        pct_inside_deep_learning_band=_pct_in(dl_lo, dl_hi),
        pct_inside_personalized_band=_pct_in(pers_lo, pers_hi),
        simple_baseline_mae_range=SIMPLE_BASELINE_MAE_RANGE,
        deep_learning_mae_range=DEEP_LEARNING_MAE_RANGE,
        personalized_mae_range=PERSONALIZED_MAE_RANGE,
        narrative=narrative,
    )
