"""Structural / schema verification of gathered run data."""

from __future__ import annotations

from typing import Any

import polars as pl
from eliot import start_action

from sugar_data_processing.config import FORMAT_GENERIC, FORMAT_MIXED, FORMAT_OWN
from sugar_data_processing.gathering.load import REQUIRED_COLUMNS, parse_per_round_metrics
from sugar_data_processing.verification.anomalies import Anomaly

ALLOWED_FORMATS: frozenset[str] = frozenset(
    {FORMAT_GENERIC, FORMAT_OWN, FORMAT_MIXED}
)


def verify_schema(runs: pl.DataFrame) -> list[Anomaly]:
    """Check required columns, formats, IDs, and parseable round metrics.

    Returns a list of issues (empty means schema checks passed). High-severity
    issues indicate data that should not be trusted for hypothesis testing.
    """
    with start_action(action_type="verification.verify_schema", n_runs=runs.height) as action:
        issues: list[Anomaly] = []

        missing = [c for c in REQUIRED_COLUMNS if c not in runs.columns]
        if missing:
            issues.append(
                Anomaly(
                    study_id="*",
                    category="missing_columns",
                    severity="high",
                    detail=f"Missing required columns: {missing}",
                )
            )
            action.log(message_type="error", missing_columns=missing)
            return issues

        if runs.height == 0:
            issues.append(
                Anomaly(
                    study_id="*",
                    category="empty_dataset",
                    severity="high",
                    detail="CSV loaded but contains zero rows.",
                )
            )
            return issues

        null_study = runs.filter(pl.col("study_id").is_null() | (pl.col("study_id") == ""))
        for row in null_study.iter_rows(named=True):
            issues.append(
                Anomaly(
                    study_id="*",
                    category="null_study_id",
                    severity="high",
                    detail=f"run_id={row.get('run_id')} has empty study_id",
                )
            )

        null_run = runs.filter(pl.col("run_id").is_null() | (pl.col("run_id") == ""))
        for row in null_run.iter_rows(named=True):
            issues.append(
                Anomaly(
                    study_id=str(row.get("study_id") or "*"),
                    category="null_run_id",
                    severity="high",
                    detail="Empty run_id",
                )
            )

        bad_fmt = runs.filter(~pl.col("format").is_in(list(ALLOWED_FORMATS)))
        for row in bad_fmt.iter_rows(named=True):
            issues.append(
                Anomaly(
                    study_id=str(row["study_id"]),
                    category="invalid_format",
                    severity="high",
                    detail=f"format={row.get('format')!r} not in {sorted(ALLOWED_FORMATS)}",
                )
            )

        for row in runs.iter_rows(named=True):
            sid = str(row["study_id"])
            mae = row.get("overall_mae_mgdl")
            if mae is not None and isinstance(mae, (int, float)) and mae < 0:
                issues.append(
                    Anomaly(
                        study_id=sid,
                        category="negative_run_mae",
                        severity="high",
                        detail=f"overall_mae_mgdl={mae}",
                    )
                )

            cell: Any = row.get("per_round_metrics")
            try:
                metrics = parse_per_round_metrics(cell)
            except (SyntaxError, ValueError, TypeError) as exc:
                issues.append(
                    Anomaly(
                        study_id=sid,
                        category="unparseable_per_round_metrics",
                        severity="high",
                        detail=f"run_id={row.get('run_id')}: {exc}",
                    )
                )
                continue

            rounds_played = row.get("rounds_played")
            if (
                metrics
                and rounds_played is not None
                and isinstance(rounds_played, (int, float))
                and int(rounds_played) != len(metrics)
            ):
                issues.append(
                    Anomaly(
                        study_id=sid,
                        category="rounds_played_mismatch",
                        severity="medium",
                        detail=(
                            f"rounds_played={int(rounds_played)} but "
                            f"per_round_metrics has {len(metrics)} entries "
                            f"(run_id={row.get('run_id')})"
                        ),
                    )
                )

            for item in metrics:
                round_mae = item.get("mae")
                if round_mae is not None and float(round_mae) < 0:
                    issues.append(
                        Anomaly(
                            study_id=sid,
                            category="negative_round_mae",
                            severity="high",
                            detail=(
                                f"run_id={row.get('run_id')} "
                                f"round={item.get('round_number')} mae={round_mae}"
                            ),
                        )
                    )

        action.log(
            message_type="info",
            n_issues=len(issues),
            high=sum(1 for i in issues if i.severity == "high"),
        )
        return issues
