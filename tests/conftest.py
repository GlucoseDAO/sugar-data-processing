"""Shared test setup."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _private_score_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep pipeline runs in tests from writing into the repo's forecast cache."""
    monkeypatch.setenv("SDP_SCORE_CACHE", str(tmp_path / "score_cache" / "forecasts.parquet"))
