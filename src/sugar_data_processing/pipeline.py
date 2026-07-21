"""End-to-end orchestration of the five library stages.

1. gathering → 2. verification → 3. statistics → 4. comparison → 5. output
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
from eliot import start_action

from sugar_data_processing.comparison.benchmarks import BenchmarkContext, benchmark_context
from sugar_data_processing.gathering.load import load_prediction_statistics
from sugar_data_processing.gathering.participants import build_participant_table
from sugar_data_processing.output.report import write_report
from sugar_data_processing.statistics.hypotheses import HypothesisSuite, run_all_hypotheses
from sugar_data_processing.verification.anomalies import Anomaly
from sugar_data_processing.verification.report import VerificationReport, verify_dataset


@dataclass(frozen=True)
class AnalysisResult:
    """Artefacts produced by :func:`run_analysis`."""

    runs: pl.DataFrame
    participants: pl.DataFrame
    verification: VerificationReport
    suite: HypothesisSuite
    benchmarks: BenchmarkContext
    report_path: Path

    @property
    def anomalies(self) -> list[Anomaly]:
        """All verification issues (schema + quality). Kept for callers."""
        return self.verification.all_issues


def run_analysis(csv_path: Path | str, output_dir: Path | str) -> AnalysisResult:
    """Run the full study-design analysis pipeline.

    Stages
    ------
    gathering
        Load CSV and build the participant table.
    verification
        Schema checks and demographic / metric quality flags.
    statistics
        H1–H5 hypothesis tests on eligible populations.
    comparison
        Human MAE vs published GlucoBench / literature bands.
    output
        Write figures, markdown report, JSON, and processed tables.
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    with start_action(
        action_type="pipeline.run_analysis",
        csv=str(csv_path),
        output=str(output_dir),
    ) as action:
        # 1. Data gathering
        runs = load_prediction_statistics(csv_path)
        participants = build_participant_table(runs)

        # 2. Data verification
        verification = verify_dataset(runs, participants)

        # 3. Statistical tests
        suite = run_all_hypotheses(participants)

        # 4. Data comparison (literature / GlucoBench bands)
        eligible = participants.filter(pl.col("eligible_primary"))
        bench_frame = eligible if eligible.height > 0 else participants
        benchmarks = benchmark_context(bench_frame, mae_col="mae_primary")

        # 5. Output
        report_path = write_report(
            runs=runs,
            participants=participants,
            suite=suite,
            benchmarks=benchmarks,
            verification=verification,
            output_dir=output_dir,
            source_csv=csv_path,
        )
        action.log(
            message_type="info",
            report=str(report_path),
            n_participants=participants.height,
            verification_passed=verification.passed,
            n_verification_issues=len(verification.all_issues),
        )
        return AnalysisResult(
            runs=runs,
            participants=participants,
            verification=verification,
            suite=suite,
            benchmarks=benchmarks,
            report_path=report_path,
        )
