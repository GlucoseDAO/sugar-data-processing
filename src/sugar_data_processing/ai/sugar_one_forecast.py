"""Score game windows with the SugarOne W5.3 forecast ONNX bundle.

W5.3 (``GlucoseDao/FORECAST-SugarOne-W5.3-10.7h-mard_loss-mv1-s42``) reads 128
five-minute steps of glucose, basal, bolus and carbs, each min-max scaled, and
returns a scaled 24-step (2-hour) glucose forecast. It is the same contract as
the local SugarOne 1.0 checkpoint: the 24 visible game points are left-padded
by repeating the first reading, and insulin and carbs stay raw zero because the
game never logged them. The 12-step game horizon is the first hour of that forecast.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from eliot import start_action

from sugar_data_processing.ai.citras import _has_hf_token, _load_hf_token_from_dotenv
from sugar_data_processing.ai.convert import pad_context
from sugar_data_processing.ai.windows import PlayedWindow
from sugar_data_processing.config import (
    FORECAST_HORIZON_POINTS,
    SUGAR_ONE_W53_CACHE_DIR,
    SUGAR_ONE_W53_HF_REPO,
)

SUGAR_ONE_W53_NAME: str = "sugar_one_w53"
SUGAR_ONE_W53_CHANNELS: tuple[str, ...] = ("glucose", "basal", "bolus", "carbs")
_BUNDLE_FILES: tuple[str, ...] = ("model.onnx", "onnx_meta.json", "scalers.json")


def find_sugar_one_w53_bundle(explicit: Path | None = None) -> Path | None:
    """Directory with ``model.onnx``, ``onnx_meta.json`` and ``scalers.json``."""
    ordered: list[Path] = []
    if explicit is not None:
        ordered.append(Path(explicit))
    ordered.append(SUGAR_ONE_W53_CACHE_DIR)
    seen: set[Path] = set()
    for path in ordered:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if all((resolved / name).exists() for name in _BUNDLE_FILES):
            return resolved
    return None


def ensure_sugar_one_w53_bundle(explicit: Path | None = None) -> Path | None:
    """Use a local ONNX folder, or download the Hub repo when a token is set."""
    found = find_sugar_one_w53_bundle(explicit)
    if found is not None:
        return found
    _load_hf_token_from_dotenv()
    if not _has_hf_token():
        return None
    SUGAR_ONE_W53_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=SUGAR_ONE_W53_HF_REPO,
        local_dir=str(SUGAR_ONE_W53_CACHE_DIR),
        allow_patterns=[*_BUNDLE_FILES, "ONNX.md", "README.md", "tuning_meta.json", "best_info.json"],
    )
    return find_sugar_one_w53_bundle(SUGAR_ONE_W53_CACHE_DIR)


def featurize_game_history(
    context_mgdl: list[float],
    *,
    input_steps: int,
    scale: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    """``(input_steps, 4)`` scaled history ending at the origin.

    Earlier steps repeat the first visible glucose. Basal, bolus and carbs are
    raw zeros, then every channel is min-max scaled (``raw * scale + min``).
    """
    raw = np.zeros((input_steps, len(SUGAR_ONE_W53_CHANNELS)), dtype=np.float32)
    raw[:, 0] = np.asarray(pad_context(context_mgdl, input_steps), dtype=np.float32)
    return raw * scale + offset


def score_sugar_one_w53_windows(
    windows: list[PlayedWindow],
    *,
    bundle_dir: Path | None = None,
    batch_size: int = 64,
) -> dict[tuple[str, str, int], list[float]]:
    """Return 12-step mg/dL forecasts, or ``{}`` when the ONNX bundle is unavailable."""
    if not windows:
        return {}
    folder = ensure_sugar_one_w53_bundle(bundle_dir)
    if folder is None:
        return {}
    try:
        return _predict_all(windows, folder, batch_size=batch_size)
    except Exception as exc:  # noqa: BLE001 — optional model; never fail the pipeline
        with start_action(action_type="ai.sugar_one_w53_optional") as action:
            action.log(message_type="warning", reason=str(exc), model=SUGAR_ONE_W53_NAME)
        return {}


def _scalers(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    features = json.loads((folder / "scalers.json").read_text(encoding="utf-8"))["features"]
    scale = np.asarray(
        [float(features[name]["scale"][0]) for name in SUGAR_ONE_W53_CHANNELS],
        dtype=np.float32,
    )
    offset = np.asarray(
        [float(features[name]["min"][0]) for name in SUGAR_ONE_W53_CHANNELS],
        dtype=np.float32,
    )
    return scale, offset


def _check_meta(meta: dict[str, Any]) -> tuple[int, int]:
    channels = tuple(str(c) for c in meta.get("channels", []))
    if meta.get("family_kind") != "sugar_one" or meta.get("task") != "forecast":
        raise ValueError(f"not a forecast-mode SugarOne bundle: {meta.get('family_kind')}/{meta.get('task')}")
    if channels != SUGAR_ONE_W53_CHANNELS:
        raise ValueError(f"unexpected SugarOne channels {channels}")
    horizon = int(meta["horizon_steps"])
    if horizon < FORECAST_HORIZON_POINTS:
        raise ValueError(f"SugarOne horizon {horizon} is shorter than the game's hour")
    return int(meta["input_steps"]), horizon


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
    input_steps, _horizon = _check_meta(meta)
    scale, offset = _scalers(folder)
    input_name = str(meta["inputs"][0]["name"])
    session = _session(folder)

    out: dict[tuple[str, str, int], list[float]] = {}
    glucose_scale = float(scale[0])
    glucose_offset = float(offset[0])
    for start in range(0, len(windows), batch_size):
        chunk = windows[start : start + batch_size]
        batch = np.stack(
            [
                featurize_game_history(
                    window.context_mgdl,
                    input_steps=input_steps,
                    scale=scale,
                    offset=offset,
                )
                for window in chunk
            ],
            axis=0,
        )
        predicted = np.asarray(session.run(None, {input_name: batch})[0], dtype=float)
        if predicted.ndim == 3:
            predicted = predicted[:, 0, :]
        predicted = predicted[:, :FORECAST_HORIZON_POINTS]
        predicted = (predicted - glucose_offset) / glucose_scale
        if not np.isfinite(predicted).all():
            raise ValueError("SugarOne W5.3 returned a non-finite forecast")
        for window, values in zip(chunk, predicted, strict=True):
            out[(window.study_id, window.run_id, window.round_number)] = [
                float(v) for v in values
            ]
    return out


def sugar_one_w53_bundle_summary(folder: Path | None = None) -> dict[str, Any]:
    """Small contract dump for logs (no weights, no token)."""
    found = find_sugar_one_w53_bundle(folder)
    if found is None:
        return {"available": False, "name": SUGAR_ONE_W53_NAME}
    meta = json.loads((found / "onnx_meta.json").read_text(encoding="utf-8"))
    return {
        "available": True,
        "name": SUGAR_ONE_W53_NAME,
        "path": str(found),
        "family_kind": meta.get("family_kind"),
        "task": meta.get("task"),
        "input_steps": meta.get("input_steps"),
        "horizon_steps": meta.get("horizon_steps"),
        "channels": meta.get("channels"),
    }
