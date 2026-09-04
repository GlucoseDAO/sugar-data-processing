"""Flag demographic impossibilities, metric outliers, and session quirks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import polars as pl
from eliot import start_action

from sugar_data_processing.config import (
    MAE_IQR_OUTLIER_K,
    MAX_PLAUSIBLE_AGE,
    MAX_PLAUSIBLE_CGM_YEARS,
    MAX_PLAUSIBLE_DIABETES_YEARS,
    MIN_GENERIC_SEGMENTS,
)


@dataclass(frozen=True)
class Anomaly:
    study_id: str
    category: str
    severity: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _iqr_bounds(values: np.ndarray, k: float = MAE_IQR_OUTLIER_K) -> tuple[float, float]:
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def detect_anomalies(runs: pl.DataFrame, participants: pl.DataFrame) -> list[Anomaly]:
    """Return human-readable anomalies for the markdown report."""
    with start_action(action_type="verification.detect_anomalies") as action:
        anomalies: list[Anomaly] = []

        # Duplicate run_ids
        if "run_id" in runs.columns:
            dup_runs = (
                runs.group_by("run_id")
                .len()
                .filter(pl.col("len") > 1)
            )
            for row in dup_runs.iter_rows(named=True):
                anomalies.append(
                    Anomaly(
                        study_id="*",
                        category="duplicate_run",
                        severity="warning",
                        detail=f"run_id {row['run_id']} appears {row['len']} times.",
                    )
                )

        for row in participants.iter_rows(named=True):
            sid = str(row["study_id"])
            age = row.get("age")
            cgm_years = row.get("cgm_duration_years")
            dm_years = row.get("diabetes_duration")
            uses_cgm = row.get("uses_cgm")
            diabetic = row.get("diabetic")
            mae = row.get("mae_primary")

            if age is not None and np.isfinite(age):
                if age <= 0 or age > MAX_PLAUSIBLE_AGE:
                    anomalies.append(
                        Anomaly(sid, "implausible_age", "high", f"age={age}")
                    )
                if cgm_years is not None and np.isfinite(cgm_years) and cgm_years > age:
                    anomalies.append(
                        Anomaly(
                            sid,
                            "cgm_duration_gt_age",
                            "high",
                            f"cgm_duration_years={cgm_years} > age={age}",
                        )
                    )
                if dm_years is not None and np.isfinite(dm_years) and dm_years > age:
                    anomalies.append(
                        Anomaly(
                            sid,
                            "diabetes_duration_gt_age",
                            "high",
                            f"diabetes_duration={dm_years} > age={age}",
                        )
                    )

            if cgm_years is not None and np.isfinite(cgm_years):
                if cgm_years < 0 or cgm_years > MAX_PLAUSIBLE_CGM_YEARS:
                    anomalies.append(
                        Anomaly(
                            sid,
                            "implausible_cgm_duration",
                            "high",
                            f"cgm_duration_years={cgm_years} "
                            f"(plausible ≤ {MAX_PLAUSIBLE_CGM_YEARS})",
                        )
                    )
                if uses_cgm is False and cgm_years > 0:
                    anomalies.append(
                        Anomaly(
                            sid,
                            "cgm_flag_mismatch",
                            "medium",
                            f"uses_cgm=False but cgm_duration_years={cgm_years}",
                        )
                    )

            if dm_years is not None and np.isfinite(dm_years):
                if dm_years < 0 or dm_years > MAX_PLAUSIBLE_DIABETES_YEARS:
                    anomalies.append(
                        Anomaly(
                            sid,
                            "implausible_diabetes_duration",
                            "high",
                            f"diabetes_duration={dm_years}",
                        )
                    )
                if diabetic is False and dm_years > 0:
                    anomalies.append(
                        Anomaly(
                            sid,
                            "diabetes_flag_mismatch",
                            "medium",
                            f"diabetic=False but diabetes_duration={dm_years}",
                        )
                    )

            n_generic = row.get("n_rounds_generic") or 0
            n_total = row.get("n_rounds_total") or 0
            if n_total > 0 and n_generic < MIN_GENERIC_SEGMENTS:
                anomalies.append(
                    Anomaly(
                        sid,
                        "below_primary_threshold",
                        "info",
                        f"only {n_generic} generic rounds (need >={MIN_GENERIC_SEGMENTS})",
                    )
                )

            if mae is not None and np.isfinite(mae) and mae < 0:
                anomalies.append(
                    Anomaly(sid, "negative_mae", "high", f"mae_primary={mae}")
                )

        mae_vals = (
            participants.filter(pl.col("mae_primary").is_not_null())["mae_primary"]
            .to_numpy()
            .astype(float)
        )
        mae_vals = mae_vals[np.isfinite(mae_vals)]
        if mae_vals.size >= 4:
            lo, hi = _iqr_bounds(mae_vals)
            for row in participants.iter_rows(named=True):
                mae = row.get("mae_primary")
                if mae is None or not np.isfinite(mae):
                    continue
                if mae < lo or mae > hi:
                    anomalies.append(
                        Anomaly(
                            str(row["study_id"]),
                            "mae_iqr_outlier",
                            "medium",
                            f"mae_primary={mae:.2f} outside IQR fence [{lo:.2f}, {hi:.2f}]",
                        )
                    )

        # Session incompleteness at run level
        if "rounds_played" in runs.columns:
            short = runs.filter(pl.col("rounds_played") < MIN_GENERIC_SEGMENTS)
            for row in short.iter_rows(named=True):
                anomalies.append(
                    Anomaly(
                        str(row["study_id"]),
                        "short_session",
                        "info",
                        f"format {row.get('format')} run with rounds_played={row.get('rounds_played')}",
                    )
                )

        action.log(
            message_type="info",
            n_anomalies=len(anomalies),
            high=sum(1 for a in anomalies if a.severity == "high"),
        )
        return anomalies
