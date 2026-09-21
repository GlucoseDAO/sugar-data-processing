"""Per-round line series for the explorer People tab."""

from __future__ import annotations

from typing import Any

import polars as pl

from sugar_data_processing.ai.windows import PlayedWindow

def percent_error_series(
    actual: list[float | None],
    predicted: list[float | None],
    *,
    visible_end: int,
) -> list[float | None]:
    """``(pred - actual) / actual * 100`` on the hidden hour, same as sugar-sugar.

    The first hidden slot may fall back to actual (0% error) when the prediction
    is missing, matching ``share._percent_error_series_for_round``.
    """
    start = int(visible_end) + 1
    out: list[float | None] = []
    actual_tail = actual[start:]
    predicted_tail = predicted[start:]
    for index, actual_raw in enumerate(actual_tail):
        predicted_raw = predicted_tail[index] if index < len(predicted_tail) else None
        actual_v = _as_float(actual_raw)
        predicted_v = _as_float(predicted_raw)
        if index == 0 and predicted_v is None and actual_v is not None and actual_v != 0.0:
            predicted_v = actual_v
        if actual_v is None or predicted_v is None or actual_v == 0.0:
            out.append(None)
        else:
            out.append((predicted_v - actual_v) / actual_v * 100.0)
    return out


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


LINE_STYLES: dict[str, dict[str, str]] = {
    "actual": {"color": "#0f172a", "dash": "solid", "label": "Actual CGM"},
    "human": {"color": "#2563eb", "dash": "dashed", "label": "Human"},
    "persistence": {"color": "#ea580c", "dash": "dotted", "label": "AI persistence"},
    "linear": {"color": "#16a34a", "dash": "dashdot", "label": "AI linear"},
    "glumind": {"color": "#9333ea", "dash": "longdash", "label": "AI GluMind"},
    "sugar_one": {"color": "#9333ea", "dash": "longdash", "label": "AI SugarOne / GluMind"},
}


def build_round_traces(
    windows: list[PlayedWindow],
    scored: pl.DataFrame,
) -> list[dict[str, Any]]:
    """One payload per played window: full 36-point actual + forecast lines."""
    by_key: dict[tuple[str, str, int], dict[str, list[float]]] = {}
    if scored.height:
        sort_cols = [c for c in ("study_id", "run_id", "round_number", "model_name", "point_index") if c in scored.columns]
        ordered = scored.sort(sort_cols) if sort_cols else scored
        for row in ordered.iter_rows(named=True):
            key = (str(row["study_id"]), str(row["run_id"]), int(row["round_number"]))
            by_key.setdefault(key, {}).setdefault(str(row["model_name"]), []).append(
                float(row["model_predicted_mgdl"])
            )

    traces: list[dict[str, Any]] = []
    for window in windows:
        key = (window.study_id, window.run_id, window.round_number)
        models = by_key.get(key, {})
        traces.append(
            {
                "study_id": window.study_id,
                "run_id": window.run_id,
                "round_number": window.round_number,
                "format": window.format,
                "source": window.source,
                "data_source_name": window.data_source_name,
                "times": window.context_times + window.horizon_times,
                "actual": window.context_mgdl + window.real_mgdl,
                "human": [None] * len(window.context_mgdl) + window.human_predicted_mgdl,
                "models": {name: [None] * len(window.context_mgdl) + values for name, values in models.items()},
                "visible_end": len(window.context_mgdl) - 1,
            }
        )
    return traces


def forecast_minute_labels(n_points: int) -> list[int]:
    """5-minute ticks for the hidden hour, matching sugar-sugar share charts."""
    return [(index + 1) * 5 for index in range(n_points)]
