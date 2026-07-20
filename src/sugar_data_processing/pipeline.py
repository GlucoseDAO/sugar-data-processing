"""End-to-end orchestration: extract → test → compare → report."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
from eliot import start_action

from sugar_data_processing.comparison.anomalies import Anomaly, detect_anomalies
from sugar_data_processing.comparison.benchmarks import BenchmarkContext, benchmark_context
from sugar_data_processing.extraction.load import load_prediction_statistics
from sugar_data_processing.extraction.participants import build_participant_table
from sugar_data_processing.output.report import write_report
from sugar_data_processing.statistics.hypotheses import HypothesisSuite, run_all_hypotheses


@dataclass(frozen=True)
class AnalysisResult:
    runs: pl.DataFrame
    participants: pl.DataFrame
    suite: HypothesisSuite
    benchmarks: BenchmarkContext
    anomalies: list[Anomaly]
    report_path: Path


def run_analysis(csv_path: Path | str, output_dir: Path | str) -> AnalysisResult:
    """Run the full study-design analysis pipeline."""
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    with start_action(
        action_type="pipeline.run_analysis",
        csv=str(csv_path),
        output=str(output_dir),
    ) as action:
        runs = load_prediction_statistics(csv_path)
        participants = build_participant_table(runs)
        suite = run_all_hypotheses(participants)
        # Benchmarks use primary-eligible people when available
        eligible = participants.filter(pl.col("eligible_primary"))
        bench_frame = eligible if eligible.height > 0 else participants
        benchmarks = benchmark_context(bench_frame, mae_col="mae_primary")
        anomalies = detect_anomalies(runs, participants)
        report_path = write_report(
            runs=runs,
            participants=participants,
            suite=suite,
            benchmarks=benchmarks,
            anomalies=anomalies,
            output_dir=output_dir,
            source_csv=csv_path,
        )
        action.log(message_type="info", report=str(report_path), n_participants=participants.height)
        return AnalysisResult(
            runs=runs,
            participants=participants,
            suite=suite,
            benchmarks=benchmarks,
            anomalies=anomalies,
            report_path=report_path,
        )
