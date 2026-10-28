"""Reuse model forecasts for windows whose visible hours and model files did not change.

A forecast depends only on the window's visible glucose values and on the model
(its weight files plus the code that feeds it), so the cache keys each forecast by
a digest of the context and a fingerprint of the model. A rebuild then scores only
new or changed windows. Set ``SDP_SCORE_CACHE=off`` to score everything again.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable
from pathlib import Path

import polars as pl
from eliot import start_action

from sugar_data_processing.ai.windows import PlayedWindow
from sugar_data_processing.config import DEFAULT_SCORE_CACHE

WindowKey = tuple[str, str, int]
Forecasts = dict[WindowKey, list[float]]

CACHE_SCHEMA: dict[str, pl.DataType] = {
    "model_name": pl.Utf8(),
    "model_fingerprint": pl.Utf8(),
    "input_digest": pl.Utf8(),
    "predicted_mgdl": pl.List(pl.Float64()),
}
_CACHE_OFF: frozenset[str] = frozenset({"0", "off", "false", "no", "none"})


def score_cache_path() -> Path | None:
    """``$SDP_SCORE_CACHE`` when set (``off`` disables the cache), else the repo default."""
    raw = os.environ.get("SDP_SCORE_CACHE")
    if raw is None:
        return DEFAULT_SCORE_CACHE
    if raw.strip().lower() in _CACHE_OFF or not raw.strip():
        return None
    return Path(raw).expanduser()


def window_digest(window: PlayedWindow) -> str:
    """Digest of the window and what every model reads: its visible glucose values.

    Players often see the same public trace slice; batched ONNX runs give those copies
    forecasts that differ in the last float32 digit, so each copy keeps its own entry and
    an unchanged rebuild reproduces the full run exactly.
    """
    payload = [window.study_id, window.run_id, window.round_number, [float(v) for v in window.context_mgdl]]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def files_fingerprint(paths: Iterable[Path]) -> str:
    """Digest of each file's name and bytes; missing files count as absent."""
    digest = hashlib.sha256()
    for path in sorted(Path(p) for p in paths):
        digest.update(path.name.encode())
        if not path.is_file():
            digest.update(b"<missing>")
            continue
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def bundle_fingerprint(folder: Path, *sources: Path) -> str:
    """Fingerprint of a model folder's files plus the source files that feed the model."""
    files = [path for path in Path(folder).iterdir() if path.is_file()]
    return files_fingerprint([*files, *sources])


def _key(window: PlayedWindow) -> WindowKey:
    return (window.study_id, window.run_id, window.round_number)


class ForecastCache:
    """Forecasts on disk, one row per model, model fingerprint and window digest."""

    def __init__(self, path: Path | None) -> None:
        self.path = Path(path) if path is not None else None
        self._rows: dict[tuple[str, str, str], list[float]] = {}
        self._used: dict[str, tuple[str, set[str]]] = {}
        self.hits: dict[str, int] = {}
        self.misses: dict[str, int] = {}
        if self.path is not None and self.path.exists():
            frame = pl.read_parquet(self.path)
            for row in frame.iter_rows(named=True):
                key = (row["model_name"], row["model_fingerprint"], row["input_digest"])
                self._rows[key] = list(row["predicted_mgdl"])

    def score(
        self,
        model_name: str,
        fingerprint: str,
        windows: list[PlayedWindow],
        scorer: Callable[[list[PlayedWindow]], Forecasts],
    ) -> Forecasts:
        """Cached forecasts for known windows; ``scorer`` runs only on the rest."""
        digests = {_key(w): window_digest(w) for w in windows}
        found: Forecasts = {}
        missing: list[PlayedWindow] = []
        for window in windows:
            cached = self._rows.get((model_name, fingerprint, digests[_key(window)]))
            if cached is None:
                missing.append(window)
            else:
                found[_key(window)] = list(cached)
        fresh = scorer(missing) if missing else {}
        for window in missing:
            values = fresh.get(_key(window))
            if values is not None:
                self._rows[(model_name, fingerprint, digests[_key(window)])] = [float(v) for v in values]
        self._used[model_name] = (fingerprint, set(digests.values()))
        self.hits[model_name] = len(found)
        self.misses[model_name] = len(missing)
        return {**found, **fresh}

    def save(self) -> None:
        """Keep this run's forecasts for the models it scored; other models stay as they were."""
        if self.path is None:
            return
        rows = []
        for (model, fingerprint, digest), values in self._rows.items():
            used = self._used.get(model)
            if used is not None and (fingerprint != used[0] or digest not in used[1]):
                continue
            rows.append(
                {"model_name": model, "model_fingerprint": fingerprint, "input_digest": digest, "predicted_mgdl": values}
            )
        with start_action(action_type="ai.forecast_cache.save", path=str(self.path)) as action:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            partial = self.path.with_suffix(".partial.parquet")
            pl.DataFrame(rows, schema=CACHE_SCHEMA).write_parquet(partial)
            partial.replace(self.path)
            action.log(message_type="info", n_rows=len(rows), reused=self.hits, scored=self.misses)
