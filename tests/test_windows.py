"""Rebuilt game windows must be the two hours the player saw, and H6 must pair identical rounds."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sugar_data_processing.ai.glucose import read_glucose_series
from sugar_data_processing.ai.overlay import h6_human_vs_models
from sugar_data_processing.ai.source_files import public_corpus_files
from sugar_data_processing.ai.windows import (
    RECOVERED_FROM_OTHER_FILE,
    cut_played_window,
    load_series,
    locate_hidden_hour,
    played_round_key,
)
from sugar_data_processing.config import DEFAULT_SUGAR_SUGAR_ROOT

SUGAR_SUGAR_DATA = DEFAULT_SUGAR_SUGAR_ROOT / "data"


def _stamps(n: int) -> list[datetime]:
    start = datetime(2026, 1, 1, 0, 0)
    return [start + timedelta(minutes=5 * i) for i in range(n)]


def test_hidden_hour_is_found_by_recorded_values_not_by_time() -> None:
    values = np.arange(100.0, 300.0)
    stamps = _stamps(values.size)
    real = values[80:92].tolist()
    # The recorded timestamp points 50 rows away; the values still decide.
    assert locate_hidden_hour(values, stamps, real, first_forecast=stamps[130]) == 80


def test_hidden_hour_missing_from_the_file_is_dropped() -> None:
    values = np.arange(100.0, 300.0)
    stamps = _stamps(values.size)
    assert locate_hidden_hour(values, stamps, [500.0] * 12, first_forecast=stamps[80]) is None


def test_hidden_hour_needs_a_full_visible_context() -> None:
    values = np.arange(100.0, 300.0)
    assert locate_hidden_hour(values, _stamps(values.size), values[10:22].tolist()) is None


def test_repeated_flat_stretch_uses_the_time_hint_or_is_not_guessed() -> None:
    values = np.concatenate([np.arange(60.0, 100.0), np.full(20, 90.0), np.arange(100.0, 160.0), np.full(20, 90.0)])
    stamps = _stamps(values.size)
    flat = [90.0] * 12
    second = 40 + 20 + 60
    assert locate_hidden_hour(values, stamps, flat, first_forecast=stamps[second + 2]) in range(second, second + 9)
    assert locate_hidden_hour(values, stamps, flat) is None


def test_loop_export_keeps_egv_and_interpolated_rows(tmp_path: Path) -> None:
    path = tmp_path / "loop_1_chronological.csv"
    path.write_text(
        "Timestamp,Event Type,Glucose (mg/dL),Recommended Split\n"
        "2026-01-01 00:10:00,EGV,120,\n"
        "2026-01-01 00:00:00,EGV,110,\n"
        "2026-01-01 00:05:00,Interpolated,115.5,\n"
        "2026-01-01 00:07:00,Calibration,300,\n"
        "2026-01-01 00:15:00,EGV,,\n",
        encoding="utf-8",
    )
    series = read_glucose_series(path)
    assert series["glucose_mgdl"].to_list() == [110.0, 115.5, 120.0]


def test_d1namo_is_read_in_mgdl_like_the_game() -> None:
    glucose = SUGAR_SUGAR_DATA / "d1namo" / "diabetes_subset_pictures-glucose-food-insulin" / "001" / "glucose.csv"
    if not glucose.exists():
        pytest.skip("D1NAMO is not downloaded (uv run download --no-photos in sugar-sugar)")
    series = read_glucose_series(glucose)
    assert series.height > 1000
    first_mmol = float(pl.read_csv(glucose).filter(pl.col("type") == "cgm")["glucose"][0])
    assert series["glucose_mgdl"][0] == pytest.approx(first_mmol * 18.0, rel=0.01)


def test_h6_pairs_people_and_models_on_the_same_rounds() -> None:
    people = pl.DataFrame({"study_id": ["p1", "p2", "p3", "p4"], "mae_primary": [5.0, 5.0, 5.0, 5.0]})
    rows = []
    for person in ("p1", "p2", "p3", "p4"):
        for point in range(12):
            rows.append(
                {
                    "study_id": person,
                    "run_id": f"{person}-run",
                    "round_number": 1,
                    "point_index": point,
                    "source": "generic",
                    "format": "A",
                    "model_name": "kronlstm",
                    "human_abs_error": 30.0,
                    "model_abs_error": 20.0 + (0.5 if person == "p2" else 0.0),
                }
            )
    h6 = h6_human_vs_models(people, pl.DataFrame(rows))
    comparison = h6["comparisons"][0]
    # The people's 5 mg/dL came from rounds the model never saw; on the scored round they erred by 30.
    assert comparison["human_mean_mae"] == pytest.approx(30.0)
    assert comparison["model_mean_mae"] == pytest.approx(20.125)
    assert "identical" in h6["pairing"]


def _loop_file(root: Path, name: str, values: np.ndarray) -> Path:
    path = root / "data" / "subjects" / name / f"{name}_chronological.csv"
    path.parent.mkdir(parents=True)
    rows = [f"{stamp:%Y-%m-%d %H:%M:%S},EGV,{value}," for stamp, value in zip(_stamps(values.size), values)]
    path.write_text("Timestamp,Event Type,Glucose (mg/dL),Recommended Split\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def _played_round(values: np.ndarray, start: int, *, fmt: str = "A", study_id: str = "p1") -> tuple[dict, pl.DataFrame]:
    """A round whose recorded hidden hour is ``values[start:start + 12]``, saved under loop_1's name."""
    meta = {
        "study_id": study_id,
        "run_id": f"{study_id}-{fmt}",
        "round_number": 1,
        "format": fmt,
        "source": "generic",
        "data_source_name": "loop_1_chronological.csv",
        "window_start_index": None,
    }
    real = values[start : start + 12]
    points = pl.DataFrame(
        {
            "point_index": list(range(12)),
            "real_mgdl": real.tolist(),
            "human_predicted_mgdl": (real + 5.0).tolist(),
            "timestamp": [f"{stamp:%Y-%m-%d %H:%M:%S}" for stamp in _stamps(values.size)[start : start + 12]],
        }
    )
    return meta, points


def test_round_named_after_the_wrong_file_is_cut_from_the_one_that_holds_it(tmp_path: Path) -> None:
    named = _loop_file(tmp_path, "loop_1", np.arange(100.0, 300.0))
    holder = np.arange(400.0, 600.0)
    _loop_file(tmp_path, "loop_2", holder)
    meta, points = _played_round(holder, 80)
    window, reason = cut_played_window(
        meta, points, named, corpus=public_corpus_files(tmp_path), series_of=load_series
    )
    assert reason == RECOVERED_FROM_OTHER_FILE
    assert window is not None and window.data_source_name == "loop_2_chronological.csv"
    assert window.context_mgdl == holder[56:80].tolist()


def test_hour_held_by_two_files_is_not_guessed(tmp_path: Path) -> None:
    named = _loop_file(tmp_path, "loop_1", np.arange(100.0, 300.0))
    holder = np.arange(400.0, 600.0)
    _loop_file(tmp_path, "loop_2", holder)
    _loop_file(tmp_path, "loop_3", holder)
    meta, points = _played_round(holder, 80)
    corpus = public_corpus_files(tmp_path)
    assert set(corpus) == {"loop_1_chronological.csv", "loop_2_chronological.csv", "loop_3_chronological.csv"}
    assert cut_played_window(meta, points, named, corpus=corpus, series_of=load_series) == (
        None,
        "values_in_several_sources",
    )


def test_own_data_round_is_not_searched_in_the_public_corpus(tmp_path: Path) -> None:
    named = _loop_file(tmp_path, "loop_1", np.arange(100.0, 300.0))
    holder = np.arange(400.0, 600.0)
    _loop_file(tmp_path, "loop_2", holder)
    meta, points = _played_round(holder, 80, fmt="B")
    assert cut_played_window(meta, points, named, corpus={}, series_of=load_series) == (None, "values_not_in_source")


def test_a_round_carried_into_another_run_is_the_same_played_round() -> None:
    values = np.arange(100.0, 300.0)
    meta_a, points_a = _played_round(values, 80, fmt="A")
    meta_c, points_c = _played_round(values, 80, fmt="C")
    assert played_round_key(meta_a, points_a) == played_round_key(meta_c, points_c)
    redrawn = points_c.with_columns(pl.col("human_predicted_mgdl") + 1.0)
    assert played_round_key(meta_a, points_a) != played_round_key(meta_c, redrawn)
    other_person, _ = _played_round(values, 80, study_id="p2")
    assert played_round_key(meta_a, points_a) != played_round_key(other_person, points_a)
