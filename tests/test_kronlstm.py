"""KronLSTM R.80 (streamed ONNX bundle) on CGM-only game windows."""

from __future__ import annotations

import math

import polars as pl
import pytest

from sugar_data_processing.ai.kronlstm import (
    KRONLSTM_CHANNELS,
    featurize_game_history,
    find_kronlstm_bundle,
    kronlstm_bundle_summary,
)
from sugar_data_processing.ai.score import MODEL_KRONLSTM, score_windows
from sugar_data_processing.ai.traces import LINE_STYLES
from sugar_data_processing.ai.windows import PlayedWindow


def _window(round_number: int, context: list[float]) -> PlayedWindow:
    return PlayedWindow(
        study_id="person-1",
        run_id="run-1",
        round_number=round_number,
        format="A",
        source="generic",
        data_source_name="example.csv",
        context_times=[f"2026-01-01 08:{i:02d}:00" for i in range(len(context))],
        context_mgdl=context,
        horizon_times=[f"2026-01-01 09:{i:02d}:00" for i in range(12)],
        real_mgdl=[130.0 + i for i in range(12)],
        human_predicted_mgdl=[128.0 + i for i in range(12)],
        resolved_path="example.csv",
    )


def test_game_history_is_cgm_only_and_starts_at_first_reading() -> None:
    context = [float("nan"), 0.0] + [100.0 + i for i in range(22)]
    history, valid, tail = featurize_game_history(context, chunk_steps=12, tail_steps=12)
    assert history.shape == (24, len(KRONLSTM_CHANNELS))
    assert valid.tolist() == [True] * 22 + [False] * 2
    assert history[0, 0] == pytest.approx(100.0)
    assert history[21, 0] == pytest.approx(121.0)
    assert all(math.isnan(v) for v in history[:, 1:].ravel())
    assert all(math.isnan(v) for v in history[22:, 0])
    assert tail.tolist() == [110.0 + i for i in range(12)]


def test_short_game_history_pads_the_tail_in_front() -> None:
    history, valid, tail = featurize_game_history([150.0, 152.0, 154.0], chunk_steps=12, tail_steps=12)
    assert history.shape == (12, 4)
    assert int(valid.sum()) == 3
    assert all(math.isnan(v) for v in tail[:9])
    assert tail[9:].tolist() == [150.0, 152.0, 154.0]


def test_kronlstm_has_a_line_style() -> None:
    assert MODEL_KRONLSTM == "kronlstm"
    assert LINE_STYLES[MODEL_KRONLSTM]["label"] == "AI Kronsugar"


def test_kronlstm_scores_real_windows() -> None:
    bundle = find_kronlstm_bundle()
    if bundle is None:
        pytest.skip("KronLSTM ONNX bundle is not on disk")
    summary = kronlstm_bundle_summary(bundle)
    assert summary["serving"] == "stream"
    assert summary["horizon_steps"] >= 12
    rising = _window(1, [100.0 + 2 * i for i in range(24)])
    falling = _window(2, [200.0 - 2 * i for i in range(24)])
    scored = score_windows([rising, falling])
    kron = scored.filter(pl.col("model_name") == MODEL_KRONLSTM)
    assert kron.height == 24
    assert kron["model_predicted_mgdl"].null_count() == 0
    by_round = {
        int(key[0]): [float(v) for v in group.sort("point_index")["model_predicted_mgdl"].to_list()]
        for key, group in kron.group_by("round_number")
    }
    for values in by_round.values():
        assert all(40.0 <= v <= 400.0 for v in values)
    # The forecast continues from the last visible reading and keeps the trend's direction.
    assert abs(by_round[1][0] - 146.0) < 15.0
    assert abs(by_round[2][0] - 154.0) < 15.0
    assert by_round[1][-1] >= by_round[1][0]
    assert by_round[2][-1] < by_round[2][0]
