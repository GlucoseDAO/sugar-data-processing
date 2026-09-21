"""Convert a 3-hour game window into the glucose-forecasting ML-ready layout."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
from eliot import start_action

from sugar_data_processing.ai.windows import PlayedWindow
from sugar_data_processing.config import (
    FORECAST_HORIZON_POINTS,
    MODEL_INPUT_SIZE,
    SAMPLE_MINUTES,
    VISIBLE_CONTEXT_POINTS,
)

ML_COLUMNS: tuple[str, ...] = (
    "sequence_id",
    "User ID",
    "Timestamp (YYYY-MM-DDThh:mm:ss)",
    "Recommended Split",
    "Study Group",
    "Event Type",
    "Glucose Value (mg/dL)",
    "Heart Rate",
    "Step Count",
    "Basal Rate (U/h)",
    "Bolus Insulin (U)",
    "Carbohydrates (g)",
)


def pad_context(context: list[float], input_size: int = MODEL_INPUT_SIZE) -> list[float]:
    """Left-pad the 24 visible points to the model's 128-step lookback.

    Only the last ``len(context)`` values are real game data. Earlier steps
    repeat the first visible glucose so we do not invent a longer history.
    """
    if not context:
        return [0.0] * input_size
    if len(context) >= input_size:
        return [float(v) for v in context[-input_size:]]
    pad = [float(context[0])] * (input_size - len(context))
    return pad + [float(v) for v in context]


def window_to_ml_rows(window: PlayedWindow) -> list[dict[str, object]]:
    """One ML-ready row per timestep: padded context, then the 12-step horizon."""
    history = pad_context(window.context_mgdl)
    first_ctx = datetime.strptime(window.context_times[0], "%Y-%m-%d %H:%M:%S")
    history_start = first_ctx - timedelta(minutes=SAMPLE_MINUTES * (MODEL_INPUT_SIZE - VISIBLE_CONTEXT_POINTS))
    rows: list[dict[str, object]] = []
    for i, value in enumerate(history):
        stamp = history_start + timedelta(minutes=SAMPLE_MINUTES * i)
        rows.append(_row(window, stamp, value, split="train"))
    for i in range(FORECAST_HORIZON_POINTS):
        stamp = datetime.strptime(window.horizon_times[i], "%Y-%m-%d %H:%M:%S")
        rows.append(_row(window, stamp, window.real_mgdl[i], split="test"))
    return rows


def _row(window: PlayedWindow, stamp: datetime, glucose: float, *, split: str) -> dict[str, object]:
    return {
        "sequence_id": window.sequence_id,
        "User ID": window.study_id,
        "Timestamp (YYYY-MM-DDThh:mm:ss)": stamp.strftime("%Y-%m-%dT%H:%M:%S"),
        "Recommended Split": split,
        "Study Group": "game_window",
        "Event Type": "EGV",
        "Glucose Value (mg/dL)": float(glucose),
        "Heart Rate": 0.0,
        "Step Count": 0.0,
        "Basal Rate (U/h)": 0.0,
        "Bolus Insulin (U)": 0.0,
        "Carbohydrates (g)": 0.0,
    }


def write_ml_ready_csv(windows: list[PlayedWindow], dest: Path | str) -> Path:
    """Write every reconstructed window as one glucose-forecasting CSV."""
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with start_action(action_type="ai.write_ml_ready_csv", n_windows=len(windows)) as action:
        rows: list[dict[str, object]] = []
        for window in windows:
            rows.extend(window_to_ml_rows(window))
        frame = pl.DataFrame(rows) if rows else pl.DataFrame(schema={c: pl.Utf8 for c in ML_COLUMNS})
        if rows:
            frame = frame.select(list(ML_COLUMNS))
        frame.write_csv(dest_path)
        action.log(message_type="info", n_rows=frame.height, path=str(dest_path))
        return dest_path
