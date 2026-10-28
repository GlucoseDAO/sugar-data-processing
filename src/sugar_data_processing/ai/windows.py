"""Rebuild the 3-hour (36-point) window the human actually played."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from eliot import start_action
from numpy.lib.stride_tricks import sliding_window_view

from sugar_data_processing.ai.glucose import parse_timestamp, read_glucose_series
from sugar_data_processing.ai.source_files import original_sources_by_run, resolve_source_path
from sugar_data_processing.ai.sequences import build_point_table, known_data_rounds
from sugar_data_processing.config import (
    DEFAULT_SUGAR_SUGAR_ROOT,
    FORECAST_HORIZON_POINTS,
    GAME_WINDOW_POINTS,
    VISIBLE_CONTEXT_POINTS,
)
from sugar_data_processing.gathering.rounds import build_round_table

#: A rebuilt hidden hour must equal the game's recorded actual values to within this (mg/dL).
MATCH_TOLERANCE_MGDL: float = 1.0


@dataclass(frozen=True)
class PlayedWindow:
    """One game window: 24 visible context points + 12 forecast points."""

    study_id: str
    run_id: str
    round_number: int
    format: str
    source: str
    data_source_name: str
    context_times: list[str]
    context_mgdl: list[float]
    horizon_times: list[str]
    real_mgdl: list[float]
    human_predicted_mgdl: list[float]
    resolved_path: str

    @property
    def sequence_id(self) -> str:
        return f"{self.run_id}:{self.round_number}"


def _fmt(stamp: datetime) -> str:
    return _naive(stamp).strftime("%Y-%m-%d %H:%M:%S")


def _naive(stamp: datetime) -> datetime:
    return stamp.replace(tzinfo=None) if stamp.tzinfo is not None else stamp


@dataclass(frozen=True)
class _Series:
    stamps: list[datetime]
    values: np.ndarray


def reconstruct_played_windows(
    runs: pl.DataFrame,
    *,
    raw_csv: Path | str | None = None,
    sugar_root: Path | None = None,
    points: pl.DataFrame | None = None,
) -> list[PlayedWindow]:
    """Locate each played 36-point window in the original CGM file.

    Own-data rounds use the upload saved under ``data/input/users``. Generic
    rounds use the published corpus file. A window is kept only where the
    file holds the 12 actual values the game recorded for that round, so the
    models see the two hours the player saw; anything else is skipped (logged
    by reason), never invented.
    """
    root = Path(sugar_root) if sugar_root is not None else DEFAULT_SUGAR_SUGAR_ROOT
    with start_action(action_type="ai.reconstruct_played_windows") as action:
        rounds = known_data_rounds(build_round_table(runs))
        point_table = points if points is not None else build_point_table(runs, rounds=rounds)
        originals = original_sources_by_run(raw_csv) if raw_csv is not None else {}
        series_cache: dict[str, _Series] = {}
        windows: list[PlayedWindow] = []
        skipped: Counter[str] = Counter()

        for meta in rounds.iter_rows(named=True):
            run_id = str(meta["run_id"])
            round_number = int(meta["round_number"])
            display_name = str(meta.get("data_source_name") or "")
            original = originals.get(run_id, {}).get(round_number) or originals.get(run_id, {}).get(0) or ""
            path = resolve_source_path(
                display_name,
                original_name=original,
                sugar_root=root,
            )
            if path is None:
                skipped["no_source_file"] += 1
                continue
            cache_key = str(path)
            if cache_key not in series_cache:
                frame = read_glucose_series(path)
                series_cache[cache_key] = _Series(
                    stamps=[_naive(s) for s in frame["timestamp"].to_list()],
                    values=frame["glucose_mgdl"].to_numpy().astype(float),
                )
            series = series_cache[cache_key]
            if series.values.size < GAME_WINDOW_POINTS:
                skipped["source_too_short"] += 1
                continue

            slice_points = point_table.filter(
                (pl.col("run_id") == run_id) & (pl.col("round_number") == round_number)
            ).sort("point_index")
            if slice_points.height == 0:
                skipped["no_points"] += 1
                continue

            window, reason = _cut_window(meta, slice_points, series, path)
            if window is None:
                skipped[reason] += 1
                continue
            windows.append(window)

        action.log(
            message_type="info",
            n_windows=len(windows),
            n_skipped=sum(skipped.values()),
            skipped=dict(skipped),
            n_sources=len(series_cache),
        )
        return windows


def locate_hidden_hour(
    values: np.ndarray,
    stamps: list[datetime],
    real: list[float],
    *,
    first_forecast: datetime | None = None,
    start_index: int | None = None,
) -> int | None:
    """Row where the round's recorded hidden hour starts, or None when this file lacks it.

    The game's recorded actual values decide. The forecast timestamp (or the
    saved window start) only picks between identical stretches, which a flat
    trace can repeat; without either hint such a tie is not guessed.
    """
    target = np.asarray(real, dtype=float)
    width = target.size
    if width == 0 or values.size < VISIBLE_CONTEXT_POINTS + width:
        return None
    gaps = np.abs(sliding_window_view(values, width) - target).max(axis=1)
    gaps[:VISIBLE_CONTEXT_POINTS] = np.inf
    best = float(gaps.min())
    if best > MATCH_TOLERANCE_MGDL:
        return None
    # Only the best fit counts: on a slow trace the stretch one row over is also within tolerance.
    hits = np.flatnonzero(gaps <= best + 1e-6)
    if hits.size <= 1:
        return int(hits[0]) if hits.size else None
    if first_forecast is not None:
        deltas = [abs((stamps[int(h)] - first_forecast).total_seconds()) for h in hits]
        return int(hits[int(np.argmin(deltas))])
    if start_index is not None and start_index >= 0:
        return int(hits[int(np.argmin(np.abs(hits - (start_index + VISIBLE_CONTEXT_POINTS))))])
    return None


def _cut_window(
    meta: dict[str, Any],
    slice_points: pl.DataFrame,
    series: _Series,
    path: Path,
) -> tuple[PlayedWindow | None, str]:
    real = [float(v) for v in slice_points["real_mgdl"].to_list() if v is not None]
    human = [float(v) for v in slice_points["human_predicted_mgdl"].to_list() if v is not None]
    if min(len(real), len(human)) < FORECAST_HORIZON_POINTS:
        return None, "short_round"

    first_forecast = parse_timestamp(slice_points["timestamp"][0])
    start_raw = meta.get("window_start_index")
    try:
        start_index = int(start_raw) if start_raw is not None and str(start_raw) != "" else None
    except (TypeError, ValueError):
        start_index = None
    forecast_i = locate_hidden_hour(
        series.values,
        series.stamps,
        real[:FORECAST_HORIZON_POINTS],
        first_forecast=_naive(first_forecast) if first_forecast is not None else None,
        start_index=start_index,
    )
    if forecast_i is None:
        return None, "values_not_in_source"

    origin = forecast_i - VISIBLE_CONTEXT_POINTS
    end = forecast_i + FORECAST_HORIZON_POINTS
    return (
        PlayedWindow(
            study_id=str(meta["study_id"]),
            run_id=str(meta["run_id"]),
            round_number=int(meta["round_number"]),
            format=str(meta.get("format") or ""),
            source=str(meta.get("source") or ""),
            data_source_name=str(meta.get("data_source_name") or ""),
            context_times=[_fmt(s) for s in series.stamps[origin:forecast_i]],
            context_mgdl=[float(v) for v in series.values[origin:forecast_i]],
            horizon_times=[_fmt(s) for s in series.stamps[forecast_i:end]],
            real_mgdl=real[:FORECAST_HORIZON_POINTS],
            human_predicted_mgdl=human[:FORECAST_HORIZON_POINTS],
            resolved_path=str(path),
        ),
        "ok",
    )
