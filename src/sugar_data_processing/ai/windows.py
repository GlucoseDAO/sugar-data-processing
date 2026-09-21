"""Rebuild the 3-hour (36-point) window the human actually played."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from eliot import start_action

from sugar_data_processing.ai.glucose import nearest_index, parse_timestamp, read_glucose_series
from sugar_data_processing.ai.source_files import original_sources_by_run, resolve_source_path
from sugar_data_processing.ai.sequences import build_point_table, known_data_rounds
from sugar_data_processing.config import (
    DEFAULT_SUGAR_SUGAR_ROOT,
    FORECAST_HORIZON_POINTS,
    GAME_WINDOW_POINTS,
    VISIBLE_CONTEXT_POINTS,
)
from sugar_data_processing.gathering.rounds import build_round_table


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


def reconstruct_played_windows(
    runs: pl.DataFrame,
    *,
    raw_csv: Path | str | None = None,
    sugar_root: Path | None = None,
    points: pl.DataFrame | None = None,
) -> list[PlayedWindow]:
    """Locate each played 36-point window in the original CGM file.

    Own-data rounds use the upload saved under ``data/input/users``. Generic
    rounds use the published corpus file. Windows that cannot be found are
    skipped (logged), never invented.
    """
    root = Path(sugar_root) if sugar_root is not None else DEFAULT_SUGAR_SUGAR_ROOT
    with start_action(action_type="ai.reconstruct_played_windows") as action:
        rounds = known_data_rounds(build_round_table(runs))
        point_table = points if points is not None else build_point_table(runs, rounds=rounds)
        originals = original_sources_by_run(raw_csv) if raw_csv is not None else {}
        series_cache: dict[str, pl.DataFrame] = {}
        windows: list[PlayedWindow] = []
        skipped = 0

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
                skipped += 1
                continue
            cache_key = str(path)
            if cache_key not in series_cache:
                series_cache[cache_key] = read_glucose_series(path)
            series = series_cache[cache_key]
            if series.height < GAME_WINDOW_POINTS:
                skipped += 1
                continue

            slice_points = point_table.filter(
                (pl.col("run_id") == run_id) & (pl.col("round_number") == round_number)
            ).sort("point_index")
            if slice_points.height == 0:
                skipped += 1
                continue

            window = _cut_window(meta, slice_points, series, path)
            if window is None:
                skipped += 1
                continue
            windows.append(window)

        action.log(
            message_type="info",
            n_windows=len(windows),
            n_skipped=skipped,
            n_sources=len(series_cache),
        )
        return windows


def _cut_window(
    meta: dict[str, Any],
    slice_points: pl.DataFrame,
    series: pl.DataFrame,
    path: Path,
) -> PlayedWindow | None:
    stamps = [_naive(s) for s in series["timestamp"].to_list() if isinstance(s, datetime)]
    values = [float(v) for v in series["glucose_mgdl"].to_list()]
    if len(stamps) != len(values):
        return None
    origin: int | None = None
    first_forecast = parse_timestamp(slice_points["timestamp"][0])
    if first_forecast is not None:
        first_forecast = _naive(first_forecast)
        forecast_i = nearest_index(stamps, first_forecast)
        if forecast_i is not None and forecast_i >= VISIBLE_CONTEXT_POINTS:
            origin = forecast_i - VISIBLE_CONTEXT_POINTS

    if origin is None:
        start_index = meta.get("window_start_index")
        if start_index is not None and str(start_index) != "":
            try:
                candidate = int(start_index)
            except (TypeError, ValueError):
                candidate = -1
            if 0 <= candidate <= len(values) - GAME_WINDOW_POINTS:
                forecast_stamp = stamps[candidate + VISIBLE_CONTEXT_POINTS]
                if first_forecast is None or abs(
                    (forecast_stamp - first_forecast).total_seconds()
                ) <= 180:
                    origin = candidate

    if origin is None:
        return None

    end = origin + GAME_WINDOW_POINTS
    if end > len(values):
        return None

    context = values[origin : origin + VISIBLE_CONTEXT_POINTS]
    horizon_src = values[origin + VISIBLE_CONTEXT_POINTS : end]
    context_times = [_fmt(s) for s in stamps[origin : origin + VISIBLE_CONTEXT_POINTS]]
    horizon_times = [
        _fmt(s) for s in stamps[origin + VISIBLE_CONTEXT_POINTS : end]
    ]

    real = [float(v) for v in slice_points["real_mgdl"].to_list() if v is not None]
    human = [float(v) for v in slice_points["human_predicted_mgdl"].to_list() if v is not None]
    n = min(len(real), len(human), FORECAST_HORIZON_POINTS, len(horizon_src))
    if n < FORECAST_HORIZON_POINTS:
        return None

    return PlayedWindow(
        study_id=str(meta["study_id"]),
        run_id=str(meta["run_id"]),
        round_number=int(meta["round_number"]),
        format=str(meta.get("format") or ""),
        source=str(meta.get("source") or ""),
        data_source_name=str(meta.get("data_source_name") or ""),
        context_times=context_times,
        context_mgdl=[float(v) for v in context],
        horizon_times=horizon_times[:n],
        real_mgdl=real[:n],
        human_predicted_mgdl=human[:n],
        resolved_path=str(path),
    )
