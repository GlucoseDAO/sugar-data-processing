"""Score game windows with the streamed KronLSTM R.80 ONNX bundle (history + decode graphs).

R.80 (``GlucoseDao/WHAT_IF_FORECAST-KronLSTM-R.80-...``, exported by arm R.115 of
``glucose_forecasting_metabonet``) reads a record once, front to back, into a memory and decodes
every forecast from the memory at its origin. ``history.onnx`` reads 12 five-minute steps per call;
``decode.onnx`` forecasts 120 steps from the memory, the last 12 glucose readings and a known-future
block. Both take physical units (glucose mg/dL, basal U/h, bolus U, carbs g) with NaN as "unknown".

Sugar Sugar only has the 24 visible CGM points. Under the model's masked-history rule that is the
start of a stretch (the memory resets after a day-long gap) in which no insulin or meal was ever
logged, so basal, bolus and carbs stay NaN. The forecast is decoded with the carbs announcement
withheld and no planned dose (an all-NaN future block). The 12-step (1-hour) game horizon is the
first hour of the 10-hour forecast.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from eliot import start_action

from sugar_data_processing.ai.citras import _HF_TOKEN_KEYS, _load_hf_token_from_dotenv
from sugar_data_processing.ai.windows import PlayedWindow
from sugar_data_processing.config import FORECAST_HORIZON_POINTS, REPO_ROOT

KRONLSTM_MODEL_NAME: str = "kronlstm"
KRONLSTM_HF_REPO: str = (
    "GlucoseDao/WHAT_IF_FORECAST-KronLSTM-R.80-10h-planned_bolus-physio_consistency"
)
KRONLSTM_CACHE_DIR: Path = REPO_ROOT / "data" / "raw" / "hf_models" / "kronlstm_r80"
KRONLSTM_CHANNELS: tuple[str, ...] = ("glucose", "basal", "bolus", "carbs")
_BUNDLE_FILES: tuple[str, ...] = ("history.onnx", "decode.onnx", "onnx_meta.json")


def find_kronlstm_bundle(explicit: Path | None = None) -> Path | None:
    """Directory with ``history.onnx``, ``decode.onnx`` and ``onnx_meta.json``."""
    ordered: list[Path] = []
    if explicit is not None:
        ordered.append(Path(explicit))
    ordered.append(KRONLSTM_CACHE_DIR)
    seen: set[Path] = set()
    for path in ordered:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if all((resolved / name).exists() for name in _BUNDLE_FILES):
            return resolved
    return None


def ensure_kronlstm_bundle(explicit: Path | None = None) -> Path | None:
    """Use a local ONNX folder, or download the private Hub repo when a token is set."""
    found = find_kronlstm_bundle(explicit)
    if found is not None:
        return found
    _load_hf_token_from_dotenv()
    if not _has_hf_token():
        return None
    KRONLSTM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=KRONLSTM_HF_REPO,
        local_dir=str(KRONLSTM_CACHE_DIR),
        allow_patterns=[*_BUNDLE_FILES, "ONNX.md", "README.md", "tuning_meta.json", "best_info.json"],
    )
    return find_kronlstm_bundle(KRONLSTM_CACHE_DIR)


def _has_hf_token() -> bool:
    """An env/.env token, or a saved `hf auth login` on this machine."""
    if any(os.environ.get(key) for key in _HF_TOKEN_KEYS):
        return True
    from huggingface_hub import get_token

    return bool(get_token())


def featurize_game_history(
    context_mgdl: list[float],
    *,
    chunk_steps: int,
    tail_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(history (T, 4), valid (T,), glucose_tail (tail,))`` for one window, in physical units.

    ``T`` is the visible stretch rounded up to whole chunks; the padding is NaN and not valid, so
    the history graph leaves the memory as it was there. The stretch starts at the first reading,
    as the masked-history rule starts one. Basal, bolus and carbs were never logged, so they are NaN.
    """
    glucose = np.asarray(context_mgdl, dtype=np.float64)
    glucose = np.where(np.isfinite(glucose) & (glucose > 0), glucose, np.nan)
    readings = np.flatnonzero(np.isfinite(glucose))
    stretch = glucose[readings[0] :] if readings.size else glucose[:0]
    steps = max(chunk_steps, -(-stretch.size // chunk_steps) * chunk_steps)
    history = np.full((steps, len(KRONLSTM_CHANNELS)), np.nan, dtype=np.float32)
    history[: stretch.size, 0] = stretch
    valid = np.zeros(steps, dtype=bool)
    valid[: stretch.size] = True
    tail = np.full(tail_steps, np.nan, dtype=np.float32)
    kept = min(stretch.size, tail_steps)
    if kept:
        tail[-kept:] = stretch[-kept:]
    return history, valid, tail


def score_kronlstm_windows(
    windows: list[PlayedWindow],
    *,
    bundle_dir: Path | None = None,
    batch_size: int = 32,
) -> dict[tuple[str, str, int], list[float]]:
    """Return 12-step mg/dL forecasts, or ``{}`` when the ONNX bundle is unavailable."""
    if not windows:
        return {}
    folder = ensure_kronlstm_bundle(bundle_dir)
    if folder is None:
        return {}
    try:
        return _predict_all(windows, folder, batch_size=batch_size)
    except Exception as exc:  # noqa: BLE001 — optional model; never fail the pipeline
        with start_action(action_type="ai.kronlstm_optional") as action:
            action.log(message_type="warning", reason=str(exc), model=KRONLSTM_MODEL_NAME)
        return {}


def _check_meta(meta: dict[str, Any]) -> None:
    channels = tuple(str(c["name"]) for c in meta.get("channels", []))
    if meta.get("family_kind") != "kronlstm" or meta.get("serving") != "stream":
        raise ValueError(f"not a streamed KronLSTM bundle: {meta.get('family_kind')}/{meta.get('serving')}")
    if channels != KRONLSTM_CHANNELS:
        raise ValueError(f"unexpected KronLSTM channels {channels}")
    if int(meta.get("static_features") or 0):
        raise ValueError("this bundle needs per-person static inputs the game does not have")
    if int(meta["horizon_steps"]) < FORECAST_HORIZON_POINTS:
        raise ValueError(f"KronLSTM horizon {meta['horizon_steps']} is shorter than the game's hour")


@lru_cache(maxsize=2)
def _sessions(folder: Path) -> tuple[Any, Any]:
    """Both graphs, loaded once per process: the unrolled 120-step decode takes ~2 min to load."""
    import onnxruntime as ort

    return tuple(
        ort.InferenceSession(str(folder / name), providers=["CPUExecutionProvider"])
        for name in ("history.onnx", "decode.onnx")
    )


def _predict_all(
    windows: list[PlayedWindow],
    folder: Path,
    *,
    batch_size: int,
) -> dict[tuple[str, str, int], list[float]]:
    meta = json.loads((folder / "onnx_meta.json").read_text(encoding="utf-8"))
    _check_meta(meta)
    chunk = int(meta["chunk_steps"])
    tail_steps = int(meta["tail_steps"])
    horizon = int(meta["horizon_steps"])
    memory = [(str(m["name"]), tuple(int(d) for d in m["shape_after_batch"])) for m in meta["memory"]]
    names = [name for name, _ in memory]
    history_graph, decode_graph = _sessions(folder)

    out: dict[tuple[str, str, int], list[float]] = {}
    for start in range(0, len(windows), batch_size):
        chunk_windows = windows[start : start + batch_size]
        prepared = [
            featurize_game_history(w.context_mgdl, chunk_steps=chunk, tail_steps=tail_steps)
            for w in chunk_windows
        ]
        steps = max(h.shape[0] for h, _, _ in prepared)
        batch = len(prepared)
        history = np.full((batch, steps, len(KRONLSTM_CHANNELS)), np.nan, dtype=np.float32)
        valid = np.zeros((batch, steps), dtype=bool)
        for row, (h, v, _) in enumerate(prepared):
            history[row, : h.shape[0]] = h
            valid[row, : v.shape[0]] = v
        state = [np.zeros((batch, *shape), dtype=np.float32) for _, shape in memory]
        for offset in range(0, steps, chunk):
            feed = {
                "history": history[:, offset : offset + chunk],
                "valid": valid[:, offset : offset + chunk],
                **dict(zip(names, state, strict=True)),
            }
            state = history_graph.run(None, feed)
        future = np.full((batch, horizon, len(KRONLSTM_CHANNELS)), np.nan, dtype=np.float32)
        feed = {
            "glucose_tail": np.stack([tail for _, _, tail in prepared], axis=0),
            "future": future,
            **dict(zip(names, state, strict=True)),
        }
        predicted = decode_graph.run(None, feed)[0][:, :FORECAST_HORIZON_POINTS]
        if not np.isfinite(predicted).all():
            raise ValueError("KronLSTM returned a non-finite forecast")
        for window, values in zip(chunk_windows, predicted, strict=True):
            out[(window.study_id, window.run_id, window.round_number)] = [
                float(v) for v in np.asarray(values, dtype=float)
            ]
    return out


def kronlstm_bundle_summary(folder: Path | None = None) -> dict[str, Any]:
    """Small contract dump for logs (no weights, no token)."""
    found = find_kronlstm_bundle(folder)
    if found is None:
        return {"available": False, "name": KRONLSTM_MODEL_NAME}
    meta = json.loads((found / "onnx_meta.json").read_text(encoding="utf-8"))
    return {
        "available": True,
        "name": KRONLSTM_MODEL_NAME,
        "path": str(found),
        "family_kind": meta.get("family_kind"),
        "serving": meta.get("serving"),
        "horizon_steps": meta.get("horizon_steps"),
        "chunk_steps": meta.get("chunk_steps"),
        "channels": meta.get("channels"),
        "checkpoint_sha256": meta.get("checkpoint_sha256"),
    }
