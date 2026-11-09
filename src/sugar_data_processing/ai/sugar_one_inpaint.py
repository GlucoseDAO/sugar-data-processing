"""Score game windows with the dual-mode SugarOne R.28 ONNX gap-filler.

R.28 (``GlucoseDao/DUAL_MODE-SugarOne-R.28-...``) is a 960-step inpainting
SugarOne: 864 steps of past (72 hours), a 48-step gap and 48 steps after it.
The game has no readings after the origin, so this runs the model's forward
mode, the same encoding as CitricSugar: the 24 visible points sit at the end
of the past block, and every channel after the origin is withheld. The output
is glucose in mg/dL for the whole window. The 12-step game horizon is the first
hour of the gap.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from eliot import start_action

from sugar_data_processing.ai.citras import (
    CITRAS_CHANNELS,
    _has_hf_token,
    _load_hf_token_from_dotenv,
    featurize_forward_window,
)
from sugar_data_processing.ai.windows import PlayedWindow
from sugar_data_processing.config import (
    FORECAST_HORIZON_POINTS,
    SUGAR_ONE_R28_CACHE_DIR,
    SUGAR_ONE_R28_HF_REPO,
)

SUGAR_ONE_R28_NAME: str = "sugar_one_r28"
_BUNDLE_FILES: tuple[str, ...] = ("model.onnx", "onnx_meta.json")


def find_sugar_one_r28_bundle(explicit: Path | None = None) -> Path | None:
    """Directory with ``model.onnx`` and ``onnx_meta.json``."""
    ordered: list[Path] = []
    if explicit is not None:
        ordered.append(Path(explicit))
    ordered.append(SUGAR_ONE_R28_CACHE_DIR)
    seen: set[Path] = set()
    for path in ordered:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if all((resolved / name).exists() for name in _BUNDLE_FILES):
            return resolved
    return None


def ensure_sugar_one_r28_bundle(explicit: Path | None = None) -> Path | None:
    """Use a local ONNX folder, or download the Hub repo when a token is set."""
    found = find_sugar_one_r28_bundle(explicit)
    if found is not None:
        return found
    _load_hf_token_from_dotenv()
    if not _has_hf_token():
        return None
    SUGAR_ONE_R28_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=SUGAR_ONE_R28_HF_REPO,
        local_dir=str(SUGAR_ONE_R28_CACHE_DIR),
        allow_patterns=[*_BUNDLE_FILES, "ONNX.md", "README.md", "tuning_meta.json"],
    )
    return find_sugar_one_r28_bundle(SUGAR_ONE_R28_CACHE_DIR)


def score_sugar_one_r28_windows(
    windows: list[PlayedWindow],
    *,
    bundle_dir: Path | None = None,
    batch_size: int = 8,
) -> dict[tuple[str, str, int], list[float]]:
    """Return 12-step mg/dL forecasts, or ``{}`` when the ONNX bundle is unavailable."""
    if not windows:
        return {}
    folder = ensure_sugar_one_r28_bundle(bundle_dir)
    if folder is None:
        return {}
    try:
        return _predict_all(windows, folder, batch_size=batch_size)
    except Exception as exc:  # noqa: BLE001 — optional model; never fail the pipeline
        with start_action(action_type="ai.sugar_one_r28_optional") as action:
            action.log(message_type="warning", reason=str(exc), model=SUGAR_ONE_R28_NAME)
        return {}


def _check_meta(meta: dict[str, Any]) -> tuple[int, int, int]:
    channels = tuple(str(c) for c in meta.get("channels", []))
    if meta.get("family_kind") != "inpaint_sugar_one" or meta.get("task") != "inpaint":
        raise ValueError(
            f"not a dual-mode SugarOne bundle: {meta.get('family_kind')}/{meta.get('task')}"
        )
    if channels != CITRAS_CHANNELS:
        raise ValueError(f"unexpected SugarOne R.28 channels {channels}")
    past = int(meta["gap_start"])
    gap = int(meta["gap_steps"])
    post = int(meta["post_steps"])
    if past + gap + post != int(meta["input_steps"]):
        raise ValueError(f"unexpected SugarOne R.28 geometry {past}+{gap}+{post}")
    if gap < FORECAST_HORIZON_POINTS:
        raise ValueError(f"SugarOne R.28 gap {gap} is shorter than the game's hour")
    return past, gap, post


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
    past, gap, post = _check_meta(meta)
    input_name = str(meta["inputs"][0]["name"])
    session = _session(folder)

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
        predicted = np.asarray(session.run(None, {input_name: batch})[0], dtype=float)
        horizon = predicted[:, past : past + FORECAST_HORIZON_POINTS]
        if not np.isfinite(horizon).all():
            raise ValueError("SugarOne R.28 returned a non-finite forecast")
        for window, values in zip(chunk, horizon, strict=True):
            out[(window.study_id, window.run_id, window.round_number)] = [
                float(v) for v in values
            ]
    return out


def sugar_one_r28_bundle_summary(folder: Path | None = None) -> dict[str, Any]:
    """Small contract dump for logs (no weights, no token)."""
    found = find_sugar_one_r28_bundle(folder)
    if found is None:
        return {"available": False, "name": SUGAR_ONE_R28_NAME}
    meta = json.loads((found / "onnx_meta.json").read_text(encoding="utf-8"))
    return {
        "available": True,
        "name": SUGAR_ONE_R28_NAME,
        "path": str(found),
        "family_kind": meta.get("family_kind"),
        "task": meta.get("task"),
        "input_steps": meta.get("input_steps"),
        "gap_start": meta.get("gap_start"),
        "gap_steps": meta.get("gap_steps"),
        "channels": meta.get("channels"),
    }
