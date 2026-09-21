"""Compatibility wrapper: AI scoring now lives in the merged report."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from sugar_data_processing.ai.ingest import ModelComparison
from sugar_data_processing.comparison.benchmarks import BenchmarkContext
from sugar_data_processing.output.report import write_report
from sugar_data_processing.statistics.hypotheses import HypothesisSuite
from sugar_data_processing.verification.report import VerificationReport


def write_ai_report(
    *,
    participants: pl.DataFrame,
    runs: pl.DataFrame,
    suite: HypothesisSuite,
    benchmarks: BenchmarkContext,
    verification: VerificationReport,
    output_dir: Path,
    source_csv: Path,
    comparison: ModelComparison | None = None,
    scored_points: pl.DataFrame | None = None,
    sequences_dir: Path | None = None,
) -> Path:
    """Write the merged human+AI report (same path as ``write_report``)."""
    del comparison, scored_points, sequences_dir
    return write_report(
        runs=runs,
        participants=participants,
        suite=suite,
        benchmarks=benchmarks,
        verification=verification,
        output_dir=output_dir,
        source_csv=source_csv,
    )
