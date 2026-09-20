"""Gathering-stage checks against the current sugar-sugar export encoding."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from sugar_data_processing.config import COHORT_DIABETIC_CGM
from sugar_data_processing.fixtures.synthetic import write_synthetic_csv
from sugar_data_processing.gathering import (
    build_participant_table,
    build_round_table,
    cgm_duration_to_years,
    classify_data_class,
    classify_source,
    is_opposite_trait,
    load_prediction_statistics,
    parse_cgm_duration,
    player_trait,
    redact_source_name,
    years_to_months,
)


def test_cgm_duration_value_unit() -> None:
    assert parse_cgm_duration("6,months") == (6.0, "months")
    assert cgm_duration_to_years("6,months") == 0.5
    assert cgm_duration_to_years("3,years") == 3.0
    assert cgm_duration_to_years(2) == 2.0
    assert cgm_duration_to_years("") is None
    assert cgm_duration_to_years(None) is None
    assert years_to_months(2.0) == 24.0
    assert years_to_months(None) is None


def test_classify_source_prefers_per_round_flag() -> None:
    assert classify_source("C", True, 2) == "generic"
    assert classify_source("C", False, 1) == "own"
    assert classify_source("C", None, 1) == "generic"
    assert classify_source("C", None, 2) == "own"
    assert classify_source("A", None, 1) == "generic"
    assert classify_source("B", None, 1) == "own"


def test_opposite_trait_field() -> None:
    assert player_trait(True) == "diabetic"
    assert player_trait(False) == "nondiabetic"
    assert player_trait(None) == "unknown"
    assert is_opposite_trait("diabetic", "nondiabetic") is True
    assert is_opposite_trait("diabetic", "diabetic") is False
    assert is_opposite_trait("unknown", "diabetic") is False


def test_redact_and_data_class() -> None:
    assert redact_source_name("D1NAMO-012.csv") == "D1NAMO-012.csv"
    assert redact_source_name("BIGIDEAS-003.csv") == "BIGIDEAS-003.csv"
    assert redact_source_name("Copy of Clarity_Export.csv") == "own_upload"
    assert classify_data_class("D1NAMO-001.csv", is_example=True, player_diabetic=False) == "diabetic"
    assert classify_data_class("BIGIDEAS-001.csv", is_example=True, player_diabetic=True) == "nondiabetic"
    assert classify_data_class("own_upload", is_example=False, player_diabetic=True) == "diabetic"


def test_latest_export_columns_round_trip(tmp_path: Path) -> None:
    csv_path = tmp_path / "prediction_statistics.csv"
    write_synthetic_csv(csv_path, n_participants=24, seed=4)
    raw = pl.read_csv(csv_path)
    assert "round_context" in raw.columns
    assert any("," in str(v) for v in raw["cgm_duration_years"].to_list() if v)

    runs = load_prediction_statistics(csv_path)
    assert "email" not in runs.columns
    assert runs["cgm_duration_years"].dtype == pl.Float64
    assert "own_upload" in runs["data_source_name"].to_list()

    rounds = build_round_table(runs)
    assert {"source", "data_source_name", "data_class", "is_example_data"} <= set(rounds.columns)
    mixed = rounds.filter(pl.col("format") == "C")
    if mixed.height:
        assert set(mixed["source"].unique()) <= {"generic", "own"}

    people = build_participant_table(runs)
    assert "cohort_category" in people.columns
    assert "played_all_formats" in people.columns
    assert "is_repeat_player" in people.columns
    assert "played_challenge_unknown" in people.columns
    assert "played_opposite_trait" in people.columns
    assert "mae_same_trait" in people.columns
    assert "mae_opposite_trait" in people.columns
    assert "diabetes_duration_months" in people.columns
    assert "cgm_duration_months" in people.columns
    assert people.filter(pl.col("cohort_category") == COHORT_DIABETIC_CGM).height > 0
    assert people.filter(pl.col("played_all_formats")).height > 0
    assert people.filter(pl.col("is_repeat_player")).height > 0
    assert people.filter(pl.col("played_challenge_unknown")).height > 0
    assert people.filter(pl.col("played_opposite_trait")).height > 0
    assert {"player_trait", "is_opposite_trait", "challenge_unknown"} <= set(rounds.columns)
    assert rounds["challenge_unknown"].null_count() == 0
    assert rounds["is_opposite_trait"].null_count() == 0


def test_blank_challenge_unknown_is_false(tmp_path: Path) -> None:
    csv_path = tmp_path / "stats.csv"
    write_synthetic_csv(csv_path, n_participants=8, seed=2)
    raw = pl.read_csv(csv_path)
    challenge = ["" if i == 0 else "false" if i == 1 else "true" for i in range(raw.height)]
    raw = raw.with_columns(pl.Series("challenge_unknown", challenge))
    raw.write_csv(csv_path)
    runs = load_prediction_statistics(csv_path)
    assert runs["challenge_unknown"].null_count() == 0
    assert runs["challenge_unknown"][0] == False
    assert runs["challenge_unknown"][1] == False
    assert runs["challenge_unknown"][2] == True
    rounds = build_round_table(runs)
    assert rounds["challenge_unknown"].null_count() == 0
    assert rounds["is_opposite_trait"].null_count() == 0
    assert rounds["is_example_data"].null_count() == 0


def test_blank_diabetic_stays_unknown(tmp_path: Path) -> None:
    csv_path = tmp_path / "stats.csv"
    write_synthetic_csv(csv_path, n_participants=8, seed=1)
    raw = pl.read_csv(csv_path).with_columns(pl.lit("").alias("diabetic"))
    raw.write_csv(csv_path)
    runs = load_prediction_statistics(csv_path)
    people = build_participant_table(runs)
    assert people["diabetic"].null_count() == people.height
    assert set(people["cohort_category"].to_list()) == {"unknown"}
