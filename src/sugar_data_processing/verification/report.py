"""Aggregate schema + quality verification into one report."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl
from eliot import start_action

from sugar_data_processing.verification.anomalies import Anomaly, detect_anomalies
from sugar_data_processing.verification.schema import verify_schema


@dataclass(frozen=True)
class VerificationReport:
    """Combined verification outcome for a gathered dataset."""

    schema_issues: list[Anomaly]
    quality_flags: list[Anomaly]
    n_runs: int
    n_participants: int

    @property
    def all_issues(self) -> list[Anomaly]:
        return [*self.schema_issues, *self.quality_flags]

    @property
    def n_high(self) -> int:
        return sum(1 for i in self.all_issues if i.severity == "high")

    @property
    def n_medium(self) -> int:
        return sum(1 for i in self.all_issues if i.severity == "medium")

    @property
    def schema_ok(self) -> bool:
        """True when no high-severity schema issues were found."""
        return not any(i.severity == "high" for i in self.schema_issues)

    @property
    def passed(self) -> bool:
        """True when schema is OK (quality flags alone do not fail verification)."""
        return self.schema_ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_ok": self.schema_ok,
            "passed": self.passed,
            "n_runs": self.n_runs,
            "n_participants": self.n_participants,
            "n_schema_issues": len(self.schema_issues),
            "n_quality_flags": len(self.quality_flags),
            "n_high": self.n_high,
            "n_medium": self.n_medium,
            "schema_issues": [a.to_dict() for a in self.schema_issues],
            "quality_flags": [a.to_dict() for a in self.quality_flags],
        }


def verify_dataset(runs: pl.DataFrame, participants: pl.DataFrame) -> VerificationReport:
    """Run schema checks then demographic / metric quality flags."""
    with start_action(action_type="verification.verify_dataset") as action:
        schema_issues = verify_schema(runs)
        quality_flags = detect_anomalies(runs, participants)
        report = VerificationReport(
            schema_issues=schema_issues,
            quality_flags=quality_flags,
            n_runs=runs.height,
            n_participants=participants.height,
        )
        action.log(
            message_type="info",
            schema_ok=report.schema_ok,
            n_schema_issues=len(schema_issues),
            n_quality_flags=len(quality_flags),
            n_high=report.n_high,
        )
        return report
