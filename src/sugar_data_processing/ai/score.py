"""Score reconstructed 3-hour windows with local baselines and optional repo models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from eliot import start_action

from sugar_data_processing.ai import citras, kronlstm, timestral
from sugar_data_processing.ai.cache import ForecastCache, Forecasts, bundle_fingerprint, files_fingerprint
from sugar_data_processing.ai.citras import ensure_citras_bundle, score_citras_windows
from sugar_data_processing.ai.convert import pad_context, write_ml_ready_csv
from sugar_data_processing.ai.kronlstm import KRONLSTM_MODEL_NAME, ensure_kronlstm_bundle, score_kronlstm_windows
from sugar_data_processing.ai.timestral import TIMESTRAL_MODEL_NAME, ensure_timestral_bundle, score_timestral_windows
from sugar_data_processing.ai.windows import PlayedWindow
from sugar_data_processing.config import (
    CITRAS_MODELS,
    EVALUATION_MODE_POST_FACTUM,
    FORECAST_HORIZON_POINTS,
    PRIMARY_MODEL_NAME,
    SUGAR_ONE_MODEL_SOURCE,
    SUGAR_ONE_RUN_DIR,
    find_forecasting_root,
)

MODEL_PERSISTENCE: str = "persistence"
MODEL_LINEAR: str = "linear"
MODEL_SUGAR_ONE: str = "sugar_one"
SUGAR_ONE_CHANNELS: tuple[str, ...] = ("glucose", "basal", "bolus", "carbs")
MODEL_CITRAS_W38_2: str = "citras_w38_2"
MODEL_KRONLSTM: str = KRONLSTM_MODEL_NAME
MODEL_TIMESTRAL: str = TIMESTRAL_MODEL_NAME


def score_windows(
    windows: list[PlayedWindow],
    *,
    forecasting_root: Path | None = None,
    ml_ready_path: Path | str | None = None,
    cache_path: Path | None = None,
) -> pl.DataFrame:
    """Return ingest-ready predictions for every window and every available model.

    Always scores ``persistence`` (last visible value) and ``linear``
    (slope of the last six visible points). ``sugar_one`` (the production
    SugarOne 1.0 checkpoint) is added when a sibling glucose-forecasting
    checkout has it.
    ``citras_w38_2`` is added when its dual-mode ONNX bundle is on disk or
    downloadable with ``HF_TOKEN``. ``kronlstm`` (streamed R.80, two ONNX
    graphs) and ``timestral`` (Timestral-28M R.81, one forecast-mode graph)
    are added on the same terms.

    With ``cache_path``, trained-model forecasts of windows scored before with the
    same model files are read back instead of recomputed.
    """
    with start_action(action_type="ai.score_windows", n_windows=len(windows)) as action:
        if ml_ready_path is not None:
            write_ml_ready_csv(windows, ml_ready_path)
        rows: list[dict[str, object]] = []
        cache = ForecastCache(cache_path)
        deep = _score_sugar_one_batch(windows, forecasting_root, cache)
        citras_by_model = _score_citras(windows, cache)
        kronlstm_preds = _score_kronlstm(windows, cache)
        timestral_preds = _score_timestral(windows, cache)
        cache.save()
        for window in windows:
            preds = {
                MODEL_PERSISTENCE: _persistence(window.context_mgdl),
                MODEL_LINEAR: _linear(window.context_mgdl),
            }
            key = (window.study_id, window.run_id, window.round_number)
            if key in deep:
                preds[MODEL_SUGAR_ONE] = deep[key]
            for model_name, by_key in citras_by_model.items():
                if key in by_key:
                    preds[model_name] = by_key[key]
            if key in kronlstm_preds:
                preds[MODEL_KRONLSTM] = kronlstm_preds[key]
            if key in timestral_preds:
                preds[MODEL_TIMESTRAL] = timestral_preds[key]
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
            reused=cache.hits,
            scored=cache.misses,
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


def _score_citras(windows: list[PlayedWindow], cache: ForecastCache) -> dict[str, Forecasts]:
    """Every registered CITRAS variant whose bundle is on disk or downloadable."""
    scored: dict[str, Forecasts] = {}
    for spec in CITRAS_MODELS:
        folder = ensure_citras_bundle(spec=spec) if windows else None
        if folder is None:
            continue
        preds = cache.score(
            spec.name,
            bundle_fingerprint(folder, Path(citras.__file__)),
            windows,
            lambda missing, spec=spec, folder=folder: score_citras_windows(missing, spec=spec, bundle_dir=folder),
        )
        if preds:
            scored[spec.name] = preds
    return scored


def _score_kronlstm(windows: list[PlayedWindow], cache: ForecastCache) -> Forecasts:
    folder = ensure_kronlstm_bundle() if windows else None
    if folder is None:
        return {}
    return cache.score(
        MODEL_KRONLSTM,
        bundle_fingerprint(folder, Path(kronlstm.__file__)),
        windows,
        lambda missing: score_kronlstm_windows(missing, bundle_dir=folder),
    )


def _score_timestral(windows: list[PlayedWindow], cache: ForecastCache) -> Forecasts:
    folder = ensure_timestral_bundle() if windows else None
    if folder is None:
        return {}
    return cache.score(
        MODEL_TIMESTRAL,
        bundle_fingerprint(folder, Path(timestral.__file__)),
        windows,
        lambda missing: score_timestral_windows(missing, bundle_dir=folder),
    )


def _score_sugar_one_batch(
    windows: list[PlayedWindow],
    forecasting_root: Path | None,
    cache: ForecastCache,
) -> Forecasts:
    """Score every window with the sibling checkout's SugarOne 1.0, or {} if unavailable."""
    if not windows:
        return {}
    root = find_forecasting_root(forecasting_root)
    if root is None:
        return {}
    run_dir = root / SUGAR_ONE_RUN_DIR
    weights = run_dir / "best_model.pt"
    if not weights.exists():
        return {}

    def predict(missing: list[PlayedWindow]) -> Forecasts:
        try:
            return _sugar_one_predict_all(missing, root, run_dir, weights)
        except Exception as exc:  # noqa: BLE001 — optional sibling; never fail the pipeline
            with start_action(action_type="ai.sugar_one_optional") as action:
                action.log(message_type="warning", reason=str(exc))
            return {}

    inputs = [weights, run_dir / "tuning_meta.json", run_dir / "config.json", run_dir / "scalers.json"]
    sources = [root / SUGAR_ONE_MODEL_SOURCE, Path(__file__)]
    return cache.score(MODEL_SUGAR_ONE, files_fingerprint([*inputs, *sources]), windows, predict)


def _sugar_one_predict_all(
    windows: list[PlayedWindow],
    root: Path,
    run_dir: Path,
    weights: Path,
    *,
    batch_size: int = 256,
) -> dict[tuple[str, str, int], list[float]]:
    """SugarOne reads 128 steps of glucose, basal, bolus and carbs, each min-max scaled with the
    run's ``scalers.json``, and returns scaled glucose for the next 12 steps.

    The game has only the 24 visible readings, left-padded with the first one (``pad_context``).
    Insulin and carbs were never logged, so those channels hold a raw zero, as
    ``glucose evaluate --zero-cov`` does.
    """
    import importlib.util
    import json

    import torch

    # Load the architecture file alone: it needs only torch, and the checkout's
    # other packages must not shadow ours on sys.path.
    spec = importlib.util.spec_from_file_location("sugar_one_model", root / SUGAR_ONE_MODEL_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {root / SUGAR_ONE_MODEL_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    meta_path = run_dir / "tuning_meta.json"
    if not meta_path.exists():
        meta_path = run_dir / "config.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    steps = int(meta["input_steps"])
    horizon = int(meta["horizon"])
    if horizon < FORECAST_HORIZON_POINTS:
        raise ValueError(f"SugarOne horizon {horizon} is shorter than the game's hour")
    model = module.SugarOneModel(
        n_time_steps=steps,
        n_features=len(SUGAR_ONE_CHANNELS),
        d_model=int(meta["d_model"]),
        n_heads=int(meta["n_heads"]),
        ff_units=int(meta["ff_units"]),
        n_blocks=int(meta["n_blocks"]),
        prediction_horizon=horizon,
        dropout=float(meta.get("dropout", 0.1)),
    )
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(_unwrap_state_dict(state))
    model.eval()

    features = json.loads((run_dir / "scalers.json").read_text(encoding="utf-8"))["features"]
    scale = np.asarray([float(features[name]["scale"][0]) for name in SUGAR_ONE_CHANNELS], dtype=np.float32)
    offset = np.asarray([float(features[name]["min"][0]) for name in SUGAR_ONE_CHANNELS], dtype=np.float32)

    out: dict[tuple[str, str, int], list[float]] = {}
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            chunk = windows[start : start + batch_size]
            raw = np.zeros((len(chunk), steps, len(SUGAR_ONE_CHANNELS)), dtype=np.float32)
            for row, window in enumerate(chunk):
                raw[row, :, 0] = pad_context(window.context_mgdl, steps)
            scaled = torch.from_numpy(raw * scale + offset)
            predicted = model(scaled).cpu().numpy().astype(float)[:, :FORECAST_HORIZON_POINTS]
            predicted = (predicted - float(offset[0])) / float(scale[0])
            if not np.isfinite(predicted).all():
                raise ValueError("SugarOne returned a non-finite forecast")
            for window, values in zip(chunk, predicted, strict=True):
                out[(window.study_id, window.run_id, window.round_number)] = [float(v) for v in values]
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
    if MODEL_SUGAR_ONE in names:
        return MODEL_SUGAR_ONE
    if MODEL_LINEAR in names:
        return MODEL_LINEAR
    return MODEL_PERSISTENCE
