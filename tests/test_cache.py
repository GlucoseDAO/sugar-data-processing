"""Forecast cache: rebuilds score only new or changed windows."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from sugar_data_processing.ai.cache import ForecastCache, score_cache_path
from sugar_data_processing.ai.windows import PlayedWindow
from sugar_data_processing.config import DEFAULT_SCORE_CACHE


def _window(run_id: str, start: float) -> PlayedWindow:
    return PlayedWindow(
        study_id="person-1",
        run_id=run_id,
        round_number=1,
        format="A",
        source="generic",
        data_source_name="example.csv",
        context_times=[f"2026-01-01 08:{i:02d}:00" for i in range(24)],
        context_mgdl=[start + i for i in range(24)],
        horizon_times=[f"2026-01-01 09:{i:02d}:00" for i in range(12)],
        real_mgdl=[130.0] * 12,
        human_predicted_mgdl=[128.0] * 12,
        resolved_path="example.csv",
    )


class _Model:
    """Forecasts the last visible value and records which windows it was asked for."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def __call__(self, windows: list[PlayedWindow]) -> dict[tuple[str, str, int], list[float]]:
        self.asked.extend(w.run_id for w in windows)
        return {(w.study_id, w.run_id, w.round_number): [w.context_mgdl[-1]] * 12 for w in windows}


def test_rebuild_scores_only_new_and_changed_windows(tmp_path: Path) -> None:
    path = tmp_path / "forecasts.parquet"
    first = [_window("a", 100.0), _window("b", 150.0)]
    model = _Model()
    cache = ForecastCache(path)
    before = cache.score("kronlstm", "weights-1", first, model)
    cache.save()
    assert model.asked == ["a", "b"]

    changed = replace(first[1], context_mgdl=[90.0 + i for i in range(24)])
    second = [first[0], changed, _window("c", 120.0)]
    model.asked.clear()
    cache = ForecastCache(path)
    after = cache.score("kronlstm", "weights-1", second, model)
    assert model.asked == ["b", "c"]
    assert after[("person-1", "a", 1)] == before[("person-1", "a", 1)]
    assert after[("person-1", "b", 1)] == [113.0] * 12
    assert cache.hits == {"kronlstm": 1} and cache.misses == {"kronlstm": 2}


def test_new_model_files_score_everything_again(tmp_path: Path) -> None:
    path = tmp_path / "forecasts.parquet"
    windows = [_window("a", 100.0), _window("b", 150.0)]
    cache = ForecastCache(path)
    cache.score("citras", "weights-1", windows, _Model())
    cache.save()

    model = _Model()
    cache = ForecastCache(path)
    cache.score("citras", "weights-2", windows, model)
    cache.save()
    assert model.asked == ["a", "b"]
    assert {key[1] for key in ForecastCache(path)._rows} == {"weights-2"}


def test_models_not_scored_this_run_keep_their_forecasts(tmp_path: Path) -> None:
    path = tmp_path / "forecasts.parquet"
    windows = [_window("a", 100.0)]
    cache = ForecastCache(path)
    cache.score("glumind", "weights-1", windows, _Model())
    cache.score("kronlstm", "weights-1", windows, _Model())
    cache.save()

    cache = ForecastCache(path)
    cache.score("kronlstm", "weights-1", [_window("b", 150.0)], _Model())
    cache.save()
    kept = {(model, digest) for model, _, digest in ForecastCache(path)._rows}
    assert len(kept) == 2
    assert {model for model, _ in kept} == {"glumind", "kronlstm"}


def test_failed_forecasts_are_not_cached(tmp_path: Path) -> None:
    path = tmp_path / "forecasts.parquet"
    cache = ForecastCache(path)
    assert cache.score("glumind", "weights-1", [_window("a", 100.0)], lambda _: {}) == {}
    cache.save()
    model = _Model()
    ForecastCache(path).score("glumind", "weights-1", [_window("a", 100.0)], model)
    assert model.asked == ["a"]


def test_cache_path_follows_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SDP_SCORE_CACHE", str(tmp_path / "here.parquet"))
    assert score_cache_path() == tmp_path / "here.parquet"
    monkeypatch.setenv("SDP_SCORE_CACHE", "off")
    assert score_cache_path() is None
    monkeypatch.delenv("SDP_SCORE_CACHE")
    assert score_cache_path() == DEFAULT_SCORE_CACHE
