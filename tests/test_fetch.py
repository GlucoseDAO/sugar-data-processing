"""Fetch / scrub checks on real CSVs (no SSH unless SUGAR_REMOTE is set)."""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import pytest

from sugar_data_processing.fetch import (
    fetch_statistics,
    load_repo_dotenv,
    resolve_identity,
    resolve_remote_spec,
    scrub_statistics_csv,
    split_remote,
)
from sugar_data_processing.fixtures.synthetic import write_synthetic_csv


def _csv_with_contact(path: Path) -> Path:
    write_synthetic_csv(path, n_participants=8, seed=2)
    raw = pl.read_csv(path, infer_schema_length=0)
    raw = raw.with_columns(
        pl.lit("player@example.com").alias("email"),
        pl.lit("Rostock").alias("location"),
        pl.lit("Ada Lovelace").alias("paper_full_name"),
    )
    raw.write_csv(path)
    return path


def test_split_remote_appends_filename() -> None:
    host, path = split_remote("box:/remote/data/input")
    assert host == "box"
    assert path.endswith("prediction_statistics.csv")


def test_split_remote_account_from_env() -> None:
    load_repo_dotenv()
    spec = os.environ.get("SUGAR_REMOTE", "").strip()
    if not spec:
        account = os.environ.get("SUGAR_SSH_HOST", "").strip()
        remote_path = os.environ.get("SUGAR_SSH_STATS", "").strip()
        if not account or not remote_path:
            pytest.skip("SUGAR_REMOTE or SUGAR_SSH_HOST+SUGAR_SSH_STATS not set")
        spec = f"{account}:{remote_path}"
    parsed_host, parsed_path = split_remote(spec)
    assert parsed_host
    assert parsed_path.endswith(".csv")


def test_split_remote_rejects_windows_path() -> None:
    with pytest.raises(ValueError, match="Windows"):
        split_remote(r"C:\data\prediction_statistics.csv")


def test_resolve_identity_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    key = tmp_path / "ssh_identity"
    key.write_text("not-a-real-key\n", encoding="utf-8")
    monkeypatch.setenv("SUGAR_SSH_IDENTITY", str(key))
    assert resolve_identity(None) == key
    other = tmp_path / "other"
    other.write_text("x\n", encoding="utf-8")
    assert resolve_identity(other) == other


def test_resolve_remote_from_host_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUGAR_REMOTE", raising=False)
    monkeypatch.setenv("SUGAR_SSH_HOST", "box")
    monkeypatch.setenv("SUGAR_SSH_STATS", "/remote/data/input/prediction_statistics.csv")
    spec = resolve_remote_spec(None)
    assert spec == "box:/remote/data/input/prediction_statistics.csv"


def test_resolve_remote_host_without_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUGAR_REMOTE", raising=False)
    monkeypatch.setenv("SUGAR_SSH_HOST", "box")
    monkeypatch.delenv("SUGAR_SSH_STATS", raising=False)
    assert resolve_remote_spec(None) is None


def test_scrub_blanks_contact_columns(tmp_path: Path) -> None:
    src = _csv_with_contact(tmp_path / "in.csv")
    dest = tmp_path / "out.csv"
    blanked = scrub_statistics_csv(src, dest)
    assert set(blanked) == {"email", "location", "paper_full_name"}
    out = pl.read_csv(dest, infer_schema_length=0)
    assert set(out["email"].unique().to_list()) == {""}
    assert set(out["location"].unique().to_list()) == {""}
    assert set(out["paper_full_name"].unique().to_list()) == {""}
    assert out.height == pl.read_csv(src, infer_schema_length=0).height
    assert "study_id" in out.columns
    assert "overall_mae_mgdl" in out.columns


def test_fetch_from_local_source(tmp_path: Path) -> None:
    src = _csv_with_contact(tmp_path / "prediction_statistics.csv")
    dest = tmp_path / "raw" / "prediction_statistics.csv"
    result = fetch_statistics(source=src, dest=dest)
    assert result.n_rows > 0
    assert result.n_participants > 0
    assert "email" in result.scrubbed
    stored = pl.read_csv(dest, infer_schema_length=0)
    assert stored["email"].to_list() == [""] * stored.height


def test_fetch_from_live_remote(tmp_path: Path) -> None:
    load_repo_dotenv()
    if not os.environ.get("SUGAR_REMOTE", "").strip():
        pytest.skip("SUGAR_REMOTE not set; live SSH pull is opt-in")
    dest = tmp_path / "prediction_statistics.csv"
    result = fetch_statistics(dest=dest)
    assert result.n_rows > 0
    assert "study_id" in result.columns
    stored = pl.read_csv(dest, infer_schema_length=0)
    if "email" in stored.columns:
        assert all(v == "" for v in stored["email"].to_list())
