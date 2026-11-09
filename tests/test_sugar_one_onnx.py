"""SugarOne W5.3 and R.28 ONNX bundles on CGM-only game windows."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from sugar_data_processing.ai.score import MODEL_SUGAR_ONE, MODEL_SUGAR_ONE_R28, score_windows
from sugar_data_processing.ai.sugar_one_forecast import (
    SUGAR_ONE_W53_CHANNELS,
    featurize_game_history,
    find_sugar_one_w53_bundle,
)
from sugar_data_processing.ai.sugar_one_inpaint import find_sugar_one_r28_bundle
from sugar_data_processing.ai.traces import LINE_STYLES
from sugar_data_processing.ai.windows import PlayedWindow
from sugar_data_processing.config import REPO_ROOT
from sugar_data_processing.statistics.progress import MODEL_LABELS, TRAINED_MODELS


def _window() -> PlayedWindow:
    return PlayedWindow(
        study_id="person-1",
        run_id="run-1",
        round_number=2,
        format="A",
        source="generic",
        data_source_name="example.csv",
        context_times=[f"2026-01-01 08:{i:02d}:00" for i in range(24)],
        context_mgdl=[100.0 + i for i in range(24)],
        horizon_times=[f"2026-01-01 09:{i:02d}:00" for i in range(12)],
        real_mgdl=[130.0 + i for i in range(12)],
        human_predicted_mgdl=[128.0 + i for i in range(12)],
        resolved_path="example.csv",
    )


def test_w53_history_pads_with_the_first_reading_and_zero_covariates() -> None:
    scale = np.ones(len(SUGAR_ONE_W53_CHANNELS), dtype=np.float32)
    offset = np.zeros(len(SUGAR_ONE_W53_CHANNELS), dtype=np.float32)
    history = featurize_game_history([110.0, 120.0], input_steps=4, scale=scale, offset=offset)
    assert history.shape == (4, 4)
    assert history[:, 0].tolist() == [110.0, 110.0, 110.0, 120.0]
    assert history[:, 1:].sum() == 0.0


def test_both_sugar_one_arms_are_in_the_publish_check() -> None:
    checkup = (REPO_ROOT / "deploy" / "publish-from-laptop.sh").read_text(encoding="utf-8")
    assert "sugar_one,sugar_one_r28,citras_w38_2,kronlstm,timestral" in checkup
    assert MODEL_SUGAR_ONE in TRAINED_MODELS
    assert MODEL_SUGAR_ONE_R28 in TRAINED_MODELS
    assert MODEL_LABELS[MODEL_SUGAR_ONE] == "SugarOne"
    assert MODEL_LABELS[MODEL_SUGAR_ONE_R28] == "SugarOne R.28"
    assert LINE_STYLES[MODEL_SUGAR_ONE_R28]["color"] not in {
        style["color"] for name, style in LINE_STYLES.items() if name != MODEL_SUGAR_ONE_R28
    }


def test_w53_and_r28_score_one_real_window() -> None:
    if find_sugar_one_w53_bundle() is None or find_sugar_one_r28_bundle() is None:
        pytest.skip("SugarOne ONNX bundles are not on disk")
    scored = score_windows([_window()])
    for name in (MODEL_SUGAR_ONE, MODEL_SUGAR_ONE_R28):
        rows = scored.filter(pl.col("model_name") == name).sort("point_index")
        assert rows.height == 12
        values = [float(v) for v in rows["model_predicted_mgdl"].to_list()]
        assert all(40.0 <= v <= 400.0 for v in values)
    w53 = [
        float(v)
        for v in scored.filter(pl.col("model_name") == MODEL_SUGAR_ONE)
        .sort("point_index")["model_predicted_mgdl"]
        .to_list()
    ]
    assert abs(w53[0] - 123.0) < 15.0
