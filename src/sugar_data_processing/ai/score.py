"""Score reconstructed 3-hour windows with local baselines and optional repo models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from eliot import start_action

from sugar_data_processing.ai.convert import pad_context, write_ml_ready_csv
from sugar_data_processing.ai.windows import PlayedWindow
from sugar_data_processing.config import (
    EVALUATION_MODE_POST_FACTUM,
    FORECAST_HORIZON_POINTS,
    PRIMARY_MODEL_NAME,
    find_forecasting_root,
)

MODEL_PERSISTENCE: str = "persistence"
MODEL_LINEAR: str = "linear"
MODEL_GLUMIND: str = "glumind"
MODEL_SUGAR_ONE: str = "sugar_one"


def score_windows(
    windows: list[PlayedWindow],
    *,
    forecasting_root: Path | None = None,
    ml_ready_path: Path | str | None = None,
) -> pl.DataFrame:
    """Return ingest-ready predictions for every window and every available model.

    Always scores ``persistence`` (last visible value) and ``linear``
    (slope of the last six visible points). ``glumind`` is added when a
    sibling glucose-forecasting checkout can load ``test_model`` and torch.
    """
    with start_action(action_type="ai.score_windows", n_windows=len(windows)) as action:
        if ml_ready_path is not None:
            write_ml_ready_csv(windows, ml_ready_path)
        rows: list[dict[str, object]] = []
        deep = _score_glumind_batch(windows, forecasting_root)
        for window in windows:
            preds = {
                MODEL_PERSISTENCE: _persistence(window.context_mgdl),
                MODEL_LINEAR: _linear(window.context_mgdl),
            }
            key = (window.study_id, window.run_id, window.round_number)
            if key in deep:
                preds[MODEL_GLUMIND] = deep[key]
            for model_name, values in preds.items():
                for index, value in enumerate(values):
                    rows.append(
                        {
                            "study_id": window.study_id,
                            "run_id": window.run_id,
                            "round_number": window.round_number,
                            "point_index": index,
                            "model_name": model_name,
                            "model_predicted_mgdl": float(value),
                            "evaluation_mode": EVALUATION_MODE_POST_FACTUM,
                        }
                    )
        schema = {
            "study_id": pl.Utf8,
            "run_id": pl.Utf8,
            "round_number": pl.Int64,
            "point_index": pl.Int64,
            "model_name": pl.Utf8,
            "model_predicted_mgdl": pl.Float64,
            "evaluation_mode": pl.Utf8,
        }
        frame = pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)
        action.log(
            message_type="info",
            n_rows=frame.height,
            models=sorted(frame["model_name"].unique().to_list()) if frame.height else [],
        )
        return frame


def _persistence(context: list[float]) -> list[float]:
    last = float(context[-1]) if context else 0.0
    return [last] * FORECAST_HORIZON_POINTS


def _linear(context: list[float], tail: int = 6) -> list[float]:
    series = [float(v) for v in context[-tail:]]
    if len(series) < 2:
        return _persistence(context)
    x = np.arange(len(series), dtype=float)
    slope, intercept = np.polyfit(x, np.asarray(series, dtype=float), 1)
    start = len(series)
    return [float(intercept + slope * (start + i)) for i in range(FORECAST_HORIZON_POINTS)]


def _score_glumind_batch(
    windows: list[PlayedWindow],
    forecasting_root: Path | None,
) -> dict[tuple[str, str, int], list[float]]:
    """Score every window with sibling GluMind (``test_model``), or {} if unavailable."""
    if not windows:
        return {}
    root = find_forecasting_root(forecasting_root)
    if root is None:
        return {}
    run_dir = root / "test_model"
    weights = run_dir / "best_model.pt"
    if not weights.exists():
        weights = run_dir / "last_model.pt"
    if not weights.exists():
        return {}
    try:
        return _glumind_predict_all(windows, root, run_dir, weights)
    except Exception as exc:  # noqa: BLE001 — optional sibling; never fail the pipeline
        with start_action(action_type="ai.glumind_optional") as action:
            action.log(message_type="warning", reason=str(exc))
        return {}


def _glumind_predict_all(
    windows: list[PlayedWindow],
    root: Path,
    run_dir: Path,
    weights: Path,
) -> dict[tuple[str, str, int], list[float]]:
    import json
    import sys

    import torch

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.glumind.glumind_model import GluMindModel

    meta_path = run_dir / "tuning_meta.json"
    if not meta_path.exists():
        meta_path = run_dir / "config.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    steps = int(meta.get("input_steps") or meta.get("n_time_steps") or 80)
    model = GluMindModel(
        n_time_steps=steps,
        n_features=3,
        d_model=int(meta.get("d_model", 32)),
        n_heads=int(meta.get("n_heads", 4)),
        ff_units=int(meta.get("ff_units", 128)),
        n_blocks=int(meta.get("n_blocks", 3)),
        prediction_horizon=int(meta.get("horizon") or meta.get("prediction_horizon") or FORECAST_HORIZON_POINTS),
        dropout=float(meta.get("dropout", 0.1)),
    )
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(_unwrap_state_dict(state))
    model.eval()

    glucose_lo, glucose_hi = 40.0, 400.0
    scalers_path = run_dir / "scalers.json"
    if scalers_path.exists():
        scalers = json.loads(scalers_path.read_text(encoding="utf-8"))
        glucose = scalers.get("glucose") if isinstance(scalers, dict) else None
        if isinstance(glucose, dict):
            glucose_lo = float(glucose.get("min", glucose_lo))
            glucose_hi = float(glucose.get("max", glucose_hi))
    span = glucose_hi - glucose_lo if glucose_hi > glucose_lo else 1.0

    out: dict[tuple[str, str, int], list[float]] = {}
    with torch.no_grad():
        for window in windows:
            hist = pad_context(window.context_mgdl, steps)
            scaled = [(float(v) - glucose_lo) / span for v in hist]
            zeros = [0.0] * len(scaled)
            features = np.stack(
                [
                    np.asarray(scaled, dtype=np.float32),
                    np.asarray(zeros, dtype=np.float32),
                    np.asarray(zeros, dtype=np.float32),
                ],
                axis=-1,
            )
            tensor = torch.from_numpy(features).unsqueeze(0)
            pred = model(tensor).squeeze(0).cpu().numpy().astype(float)
            pred = pred * span + glucose_lo
            out[(window.study_id, window.run_id, window.round_number)] = [
                float(v) for v in pred[:FORECAST_HORIZON_POINTS]
            ]
    return out


def _unwrap_state_dict(state: object) -> dict[str, object]:
    """Accept compiled (`_orig_mod.`) and wrapped Lightning-style checkpoints."""
    raw: object = state
    if isinstance(raw, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            inner = raw.get(key)
            if isinstance(inner, dict) and inner:
                raw = inner
                break
    if not isinstance(raw, dict):
        raise TypeError(f"Unexpected checkpoint type: {type(state)}")
    if any(str(key).startswith("_orig_mod.") for key in raw):
        return {str(key).removeprefix("_orig_mod."): value for key, value in raw.items()}
    return raw


def pick_primary_model(predictions: pl.DataFrame) -> str:
    if predictions.height == 0:
        return PRIMARY_MODEL_NAME
    names = set(predictions["model_name"].to_list())
    if MODEL_GLUMIND in names or MODEL_SUGAR_ONE in names:
        return MODEL_GLUMIND if MODEL_GLUMIND in names else MODEL_SUGAR_ONE
    if MODEL_LINEAR in names:
        return MODEL_LINEAR
    return MODEL_PERSISTENCE
