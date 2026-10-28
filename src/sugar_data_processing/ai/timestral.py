"""Score game windows with the Timestral-28M R.81 ONNX forecaster (forecast mode).

R.81 (``GlucoseDao/WHAT_IF_FORECAST-Timestral_28M-R.81-...``, exported by arm R.118 of
``glucose_forecasting_metabonet``) is Timestral-28M fine-tuned to answer an announced meal.
``model.onnx`` reads 72 hours (864 five-minute steps) of glucose mg/dL, basal U/h, bolus U and
carbs g with NaN as "unknown", plus the announced carbs over the horizon, and returns the median
forecast in mg/dL. Timestral normalises every row itself, so the bundle's scalers are identity.

Sugar Sugar only has the 24 visible CGM points. Under the masked-history rule the history starts at
the first reading (NaN before it), and insulin and meals were never logged, so basal, bolus and
carbs stay NaN. The announcement is withheld (an all-NaN future block). The 12-step (1-hour) game
horizon is the first hour of the 2-hour forecast.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from eliot import start_action

from sugar_data_processing.ai.citras import _has_hf_token, _load_hf_token_from_dotenv
from sugar_data_processing.ai.windows import PlayedWindow
from sugar_data_processing.config import (
    FORECAST_HORIZON_POINTS,
    TIMESTRAL_CACHE_DIR,
    TIMESTRAL_HF_REPO,
)

TIMESTRAL_MODEL_NAME: str = "timestral"
TIMESTRAL_CHANNELS: tuple[str, ...] = ("glucose", "basal", "bolus", "carbs")
_BUNDLE_FILES: tuple[str, ...] = ("model.onnx", "onnx_meta.json")


def find_timestral_bundle(explicit: Path | None = None) -> Path | None:
    """Directory with ``model.onnx`` and ``onnx_meta.json``."""
    ordered: list[Path] = []
    if explicit is not None:
        ordered.append(Path(explicit))
    ordered.append(TIMESTRAL_CACHE_DIR)
    seen: set[Path] = set()
    for path in ordered:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if all((resolved / name).exists() for name in _BUNDLE_FILES):
            return resolved
    return None


def ensure_timestral_bundle(explicit: Path | None = None) -> Path | None:
    """Use a local ONNX folder, or download the private Hub repo when a token is set."""
    found = find_timestral_bundle(explicit)
    if found is not None:
        return found
    _load_hf_token_from_dotenv()
    if not _has_hf_token():
        return None
    TIMESTRAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=TIMESTRAL_HF_REPO,
        local_dir=str(TIMESTRAL_CACHE_DIR),
        allow_patterns=[*_BUNDLE_FILES, "ONNX.md", "README.md", "tuning_meta.json", "scalers.json", "best_info.json"],
    )
    return find_timestral_bundle(TIMESTRAL_CACHE_DIR)


def featurize_game_history(context_mgdl: list[float], *, input_steps: int) -> np.ndarray:
    """``(input_steps, 4)`` history for one window in physical units, ending at the origin.

    The visible points fill the last steps; everything before them is NaN, so the history starts
    at the first reading as the masked-history rule starts one. A missing glucose stays NaN, and
    basal, bolus and carbs were never logged, so they are NaN throughout.
    """
    glucose = np.asarray(context_mgdl, dtype=np.float64)[-input_steps:]
    glucose = np.where(np.isfinite(glucose) & (glucose > 0), glucose, np.nan)
    history = np.full((input_steps, len(TIMESTRAL_CHANNELS)), np.nan, dtype=np.float32)
    if glucose.size:
        history[-glucose.size :, 0] = glucose
    return history


def score_timestral_windows(
    windows: list[PlayedWindow],
    *,
    bundle_dir: Path | None = None,
    batch_size: int = 32,
) -> dict[tuple[str, str, int], list[float]]:
    """Return 12-step mg/dL forecasts, or ``{}`` when the ONNX bundle is unavailable."""
    if not windows:
        return {}
    folder = ensure_timestral_bundle(bundle_dir)
    if folder is None:
        return {}
    try:
        return _predict_all(windows, folder, batch_size=batch_size)
    except Exception as exc:  # noqa: BLE001 — optional model; never fail the pipeline
        with start_action(action_type="ai.timestral_optional") as action:
            action.log(message_type="warning", reason=str(exc), model=TIMESTRAL_MODEL_NAME)
        return {}


def _check_meta(meta: dict[str, Any]) -> None:
    channels = tuple(str(c) for c in meta.get("channels", []))
    if meta.get("family_kind") != "timestral" or meta.get("task") != "forecast":
        raise ValueError(f"not a forecast-mode Timestral bundle: {meta.get('family_kind')}/{meta.get('task')}")
    if channels != TIMESTRAL_CHANNELS:
        raise ValueError(f"unexpected Timestral channels {channels}")
    if meta.get("metadata_schema"):
        raise ValueError("this bundle needs per-person static inputs the game does not have")
    if int(meta["horizon_steps"]) < FORECAST_HORIZON_POINTS:
        raise ValueError(f"Timestral horizon {meta['horizon_steps']} is shorter than the game's hour")


@lru_cache(maxsize=2)
def _session(folder: Path) -> Any:
    import onnxruntime as ort

    return ort.InferenceSession(str(folder / "model.onnx"), providers=["CPUExecutionProvider"])


def _predict_all(
    windows: list[PlayedWindow],
    folder: Path,
    *,
    batch_size: int,
) -> dict[tuple[str, str, int], list[float]]:
    meta = json.loads((folder / "onnx_meta.json").read_text(encoding="utf-8"))
    _check_meta(meta)
    input_steps = int(meta["input_steps"])
    horizon = int(meta["horizon_steps"])
    names = [str(i["name"]) for i in meta["inputs"]]
    session = _session(folder)

    out: dict[tuple[str, str, int], list[float]] = {}
    for start in range(0, len(windows), batch_size):
        chunk = windows[start : start + batch_size]
        history = np.stack(
            [featurize_game_history(w.context_mgdl, input_steps=input_steps) for w in chunk], axis=0
        )
        future = np.full((len(chunk), horizon, len(TIMESTRAL_CHANNELS)), np.nan, dtype=np.float32)
        predicted = session.run(None, dict(zip(names, (history, future), strict=True)))[0]
        predicted = predicted[:, :FORECAST_HORIZON_POINTS]
        if not np.isfinite(predicted).all():
            raise ValueError("Timestral returned a non-finite forecast")
        for window, values in zip(chunk, predicted, strict=True):
            out[(window.study_id, window.run_id, window.round_number)] = [
                float(v) for v in np.asarray(values, dtype=float)
            ]
    return out


def timestral_bundle_summary(folder: Path | None = None) -> dict[str, Any]:
    """Small contract dump for logs (no weights, no token)."""
    found = find_timestral_bundle(folder)
    if found is None:
        return {"available": False, "name": TIMESTRAL_MODEL_NAME}
    meta = json.loads((found / "onnx_meta.json").read_text(encoding="utf-8"))
    return {
        "available": True,
        "name": TIMESTRAL_MODEL_NAME,
        "path": str(found),
        "family_kind": meta.get("family_kind"),
        "task": meta.get("task"),
        "input_steps": meta.get("input_steps"),
        "horizon_steps": meta.get("horizon_steps"),
        "channels": meta.get("channels"),
        "checkpoint_sha256": meta.get("checkpoint_sha256"),
    }
