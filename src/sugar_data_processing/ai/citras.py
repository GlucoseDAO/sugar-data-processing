"""Score game windows with dual-mode CITRAS-FM ONNX inpainters (W38.7, W38.2).

Each Hub bundle is a 2880-step gap filler (1440 past + 48 hidden + 1392 post),
not stock ``CitrasFM.from_pretrained``. Sugar Sugar only has the 24 visible
points the human saw, so we encode them at the end of the past block and run
the model's trained **forward** mode: withhold every channel after the origin
and mark those steps hidden, matching ``metabonet.inpaint_train.serve_windows``.
The 12-step (1-hour) game horizon is the first hour of the 48-step gap.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from eliot import start_action

from sugar_data_processing.ai.windows import PlayedWindow
from sugar_data_processing.config import (
    CITRAS_GAP_STEPS,
    CITRAS_MODELS,
    CITRAS_PAST_STEPS,
    CITRAS_POST_STEPS,
    CITRAS_SEQ_LEN,
    FORECAST_HORIZON_POINTS,
    REPO_ROOT,
    CitrasModelSpec,
    find_citras_bundle,
)

CITRAS_CHANNELS: tuple[str, ...] = (
    "glucose_div100",
    "glucose_observed",
    "log1p_basal",
    "basal_observed",
    "log1p_bolus",
    "bolus_observed",
    "hidden",
    "real_slot",
    "carbs_div100",
    "meal_known",
)
_HF_TOKEN_KEYS: tuple[str, ...] = (
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HF_API_KEY",
)


def featurize_forward_window(
    context_mgdl: list[float],
    *,
    past_steps: int = CITRAS_PAST_STEPS,
    gap_steps: int = CITRAS_GAP_STEPS,
    post_steps: int = CITRAS_POST_STEPS,
) -> np.ndarray:
    """Build one ``(L, 10)`` float32 window in the ONNX channel order.

    History the game did not show is unobserved (value 0, mask 0, ``real_slot`` 0).
    Insulin and carbs are unknown, so those channels stay unobserved. After the
    origin every channel is withheld and ``hidden`` is set (``after_origin``).
    """
    length = past_steps + gap_steps + post_steps
    glucose = np.full(length, np.nan, dtype=np.float64)
    basal = np.full(length, np.nan, dtype=np.float64)
    bolus = np.full(length, np.nan, dtype=np.float64)
    meal = np.full(length, np.nan, dtype=np.float64)
    real = np.zeros(length, dtype=bool)
    hidden = np.zeros(length, dtype=bool)

    context = np.asarray(context_mgdl, dtype=np.float64)
    kept = min(int(context.size), past_steps)
    if kept:
        start = past_steps - kept
        glucose[start:past_steps] = context[-kept:]
        real[start:past_steps] = True
    real[past_steps:] = True
    hidden[past_steps:] = True

    observed_glucose = np.isfinite(glucose) & (glucose > 0) & ~hidden & real
    observed_basal = np.isfinite(basal) & (basal >= 0) & real
    observed_bolus = np.isfinite(bolus) & (bolus >= 0) & real
    meal_flag = np.isfinite(meal) & (meal > 0) & real
    meal_known = np.isfinite(meal) & (meal >= 0) & real

    features = np.zeros((length, len(CITRAS_CHANNELS)), dtype=np.float32)
    features[observed_glucose, 0] = (glucose[observed_glucose] / 100.0).astype(np.float32)
    features[:, 1] = observed_glucose.astype(np.float32)
    features[observed_basal, 2] = np.log1p(np.clip(basal[observed_basal], 0.0, None)).astype(
        np.float32
    )
    features[:, 3] = observed_basal.astype(np.float32)
    features[observed_bolus, 4] = np.log1p(np.clip(bolus[observed_bolus], 0.0, None)).astype(
        np.float32
    )
    features[:, 5] = observed_bolus.astype(np.float32)
    features[:, 6] = hidden.astype(np.float32)
    features[:, 7] = real.astype(np.float32)
    features[meal_flag, 8] = (np.clip(meal[meal_flag], 0.0, None) / 100.0).astype(np.float32)
    features[:, 9] = meal_known.astype(np.float32)
    return features


def score_all_citras_windows(
    windows: list[PlayedWindow],
    *,
    batch_size: int = 8,
) -> dict[str, dict[tuple[str, str, int], list[float]]]:
    """Score every registered CITRAS variant that can be loaded."""
    scored: dict[str, dict[tuple[str, str, int], list[float]]] = {}
    for spec in CITRAS_MODELS:
        preds = score_citras_windows(windows, spec=spec, batch_size=batch_size)
        if preds:
            scored[spec.name] = preds
    return scored


def score_citras_windows(
    windows: list[PlayedWindow],
    *,
    spec: CitrasModelSpec | None = None,
    bundle_dir: Path | None = None,
    batch_size: int = 8,
) -> dict[tuple[str, str, int], list[float]]:
    """Return 12-step mg/dL forecasts, or ``{}`` when the ONNX bundle is unavailable."""
    if not windows:
        return {}
    chosen = spec if spec is not None else CITRAS_MODELS[0]
    folder = ensure_citras_bundle(bundle_dir, spec=chosen)
    if folder is None:
        return {}
    try:
        return _predict_all(windows, folder, batch_size=batch_size)
    except Exception as exc:  # noqa: BLE001 — optional model; never fail the pipeline
        with start_action(action_type="ai.citras_optional") as action:
            action.log(message_type="warning", reason=str(exc), model=chosen.name)
        return {}


def ensure_citras_bundle(
    explicit: Path | None = None,
    *,
    spec: CitrasModelSpec | None = None,
) -> Path | None:
    """Use a local ONNX folder, or download the private Hub repo when a token is set."""
    chosen = spec if spec is not None else CITRAS_MODELS[0]
    found = find_citras_bundle(explicit, spec=chosen)
    if found is not None:
        return found
    _load_hf_token_from_dotenv()
    if not any(os.environ.get(key) for key in _HF_TOKEN_KEYS):
        return None
    chosen.cache_dir.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=chosen.repo_id,
        local_dir=str(chosen.cache_dir),
        allow_patterns=["model.onnx", "onnx_meta.json", "tuning_meta.json", "ONNX.md"],
    )
    return find_citras_bundle(chosen.cache_dir, spec=chosen)


def _load_hf_token_from_dotenv() -> None:
    if any(os.environ.get(key) for key in _HF_TOKEN_KEYS):
        return
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key in _HF_TOKEN_KEYS and value.strip():
            os.environ[key] = value.strip().strip("'\"")
            return


def _predict_all(
    windows: list[PlayedWindow],
    folder: Path,
    *,
    batch_size: int,
) -> dict[tuple[str, str, int], list[float]]:
    import onnxruntime as ort

    meta = json.loads((folder / "onnx_meta.json").read_text(encoding="utf-8"))
    past = int(meta.get("gap_start") or CITRAS_PAST_STEPS)
    gap = int(meta.get("gap_steps") or CITRAS_GAP_STEPS)
    post = int(meta.get("post_steps") or CITRAS_POST_STEPS)
    if past + gap + post != CITRAS_SEQ_LEN:
        raise ValueError(f"unexpected CITRAS geometry {past}+{gap}+{post}")
    input_name = str(meta["inputs"][0]["name"]) if meta.get("inputs") else "x_features"
    session = ort.InferenceSession(
        str(folder / "model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    out: dict[tuple[str, str, int], list[float]] = {}
    for start in range(0, len(windows), batch_size):
        chunk = windows[start : start + batch_size]
        batch = np.stack(
            [
                featurize_forward_window(
                    window.context_mgdl,
                    past_steps=past,
                    gap_steps=gap,
                    post_steps=post,
                )
                for window in chunk
            ],
            axis=0,
        )
        predicted = session.run(None, {input_name: batch})[0]
        horizon = predicted[:, past : past + FORECAST_HORIZON_POINTS]
        for window, values in zip(chunk, horizon, strict=True):
            out[(window.study_id, window.run_id, window.round_number)] = [
                float(v) for v in np.asarray(values, dtype=float)
            ]
    return out


def citras_bundle_summary(
    folder: Path | None = None,
    *,
    spec: CitrasModelSpec | None = None,
) -> dict[str, Any]:
    """Small contract dump for logs (no weights, no token)."""
    chosen = spec if spec is not None else CITRAS_MODELS[0]
    found = find_citras_bundle(folder, spec=chosen)
    if found is None:
        return {"available": False, "name": chosen.name}
    meta = json.loads((found / "onnx_meta.json").read_text(encoding="utf-8"))
    return {
        "available": True,
        "name": chosen.name,
        "path": str(found),
        "family_kind": meta.get("family_kind"),
        "input_steps": meta.get("input_steps"),
        "gap_start": meta.get("gap_start"),
        "gap_steps": meta.get("gap_steps"),
        "channels": meta.get("channels"),
    }
