"""Timestral-28M R.81 (forecast-mode ONNX bundle) on CGM-only game windows."""

from __future__ import annotations

import math

import polars as pl
import pytest

from sugar_data_processing.ai.score import MODEL_TIMESTRAL, score_windows
from sugar_data_processing.ai.timestral import (
    TIMESTRAL_CHANNELS,
    featurize_game_history,
    find_timestral_bundle,
    timestral_bundle_summary,
)
from sugar_data_processing.ai.traces import LINE_STYLES
from sugar_data_processing.ai.windows import PlayedWindow
from sugar_data_processing.statistics.progress import MODEL_LABELS, TRAINED_MODELS


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


def test_game_history_ends_at_the_origin_and_is_cgm_only() -> None:
    context = [float("nan"), 0.0] + [100.0 + i for i in range(22)]
    history = featurize_game_history(context, input_steps=864)
    assert history.shape == (864, len(TIMESTRAL_CHANNELS))
    assert all(math.isnan(v) for v in history[:842, 0])
    assert history[842:, 0].tolist() == [100.0 + i for i in range(22)]
    assert all(math.isnan(v) for v in history[:, 1:].ravel())


def test_timestral_is_shown_as_sugarstral() -> None:
    assert MODEL_TIMESTRAL == "timestral"
    assert LINE_STYLES[MODEL_TIMESTRAL]["label"] == "AI Sugarstral"
    others = {name: style["color"] for name, style in LINE_STYLES.items() if name != MODEL_TIMESTRAL}
    assert LINE_STYLES[MODEL_TIMESTRAL]["color"] not in others.values()
    assert MODEL_LABELS[MODEL_TIMESTRAL] == "Sugarstral"
    assert MODEL_TIMESTRAL in TRAINED_MODELS


def test_timestral_scores_real_windows() -> None:
    bundle = find_timestral_bundle()
    if bundle is None:
        pytest.skip("Timestral ONNX bundle is not on disk")
    summary = timestral_bundle_summary(bundle)
    assert summary["task"] == "forecast"
    assert summary["horizon_steps"] >= 12
    rising = _window(1, [100.0 + 2 * i for i in range(24)])
    falling = _window(2, [200.0 - 2 * i for i in range(24)])
    scored = score_windows([rising, falling])
    stral = scored.filter(pl.col("model_name") == MODEL_TIMESTRAL)
    assert stral.height == 24
    assert stral["model_predicted_mgdl"].null_count() == 0
    by_round = {
        int(key[0]): [float(v) for v in group.sort("point_index")["model_predicted_mgdl"].to_list()]
        for key, group in stral.group_by("round_number")
    }
    for values in by_round.values():
        assert all(40.0 <= v <= 400.0 for v in values)
    # The forecast continues from the last visible reading and keeps the trend's direction.
    assert abs(by_round[1][0] - 146.0) < 15.0
    assert abs(by_round[2][0] - 154.0) < 15.0
    assert by_round[1][-1] >= by_round[1][0]
    assert by_round[2][-1] < by_round[2][0]
