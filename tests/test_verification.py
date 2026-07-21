"""Verification stage checks on synthetic (and intentionally broken) data."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from sugar_data_processing.fixtures.synthetic import write_synthetic_csv
from sugar_data_processing.gathering import build_participant_table, load_prediction_statistics
from sugar_data_processing.verification import verify_dataset, verify_schema


def test_synthetic_schema_passes(tmp_path: Path) -> None:
    csv_path = tmp_path / "stats.csv"
    write_synthetic_csv(csv_path, n_participants=30, seed=5)
    runs = load_prediction_statistics(csv_path)
    participants = build_participant_table(runs)
    report = verify_dataset(runs, participants)

    assert report.schema_ok
    assert report.passed
    assert report.n_runs == runs.height
    assert report.n_participants == participants.height
    assert verify_schema(runs) == []


def test_invalid_format_flagged(tmp_path: Path) -> None:
    csv_path = tmp_path / "stats.csv"
    write_synthetic_csv(csv_path, n_participants=10, seed=2)
    runs = load_prediction_statistics(csv_path)
    runs = runs.with_columns(pl.lit("Z").alias("format"))
    issues = verify_schema(runs)
    assert any(i.category == "invalid_format" and i.severity == "high" for i in issues)
