"""Locate the CGM file behind a played round (generic corpus or saved upload)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import polars as pl
from eliot import start_action

from sugar_data_processing.config import DEFAULT_SUGAR_SUGAR_ROOT
from sugar_data_processing.gathering.encoding import parse_per_round_metrics, parse_round_context
from sugar_data_processing.gathering.sources import is_generic_corpus_name, source_basename

_D1NAMO = re.compile(r"^D1NAMO-(\d{3})\.csv$", re.IGNORECASE)
_BIGIDEAS = re.compile(r"^BIGIDEAS-(\d{3})\.csv$", re.IGNORECASE)
_LOOP = re.compile(r"^(loop_\d+_chronological\.csv)$", re.IGNORECASE)


def sugar_sugar_root(explicit: Path | None = None) -> Path:
    return Path(explicit) if explicit is not None else DEFAULT_SUGAR_SUGAR_ROOT


def users_dir(root: Path) -> Path:
    return root / "data" / "input" / "users"


def safe_upload_filename(name: str) -> str:
    safe = (name or "uploaded").replace(" ", "_").replace("/", "_")
    if not safe.lower().endswith((".csv", ".json")):
        safe += ".csv"
    return safe


def original_sources_by_run(raw_csv: Path | str) -> dict[str, dict[int, str]]:
    """``run_id → {round_number → original filename}`` from the unredacted export.

    Used only to find saved uploads. Never written into public reports.
    """
    path = Path(raw_csv)
    if not path.exists():
        return {}
    raw = pl.read_csv(path, infer_schema_length=10_000)
    out: dict[str, dict[int, str]] = {}
    for record in raw.iter_rows(named=True):
        run_id = str(record.get("run_id") or "")
        if not run_id:
            continue
        per_round: dict[int, str] = {}
        run_source = source_basename(record.get("data_source_name"))
        try:
            metrics = parse_per_round_metrics(record.get("per_round_metrics"))
        except (SyntaxError, ValueError, TypeError):
            metrics = []
        try:
            contexts = parse_round_context(record.get("round_context"))
        except (SyntaxError, ValueError, TypeError):
            contexts = []
        context_map = {
            int(item.get("round_number") or item.get("round") or 0): item for item in contexts
        }
        for item in metrics:
            number = int(item.get("round_number") or item.get("round") or 0)
            if not number:
                continue
            name = source_basename(
                item.get("data_source_name")
                or context_map.get(number, {}).get("data_source_name")
                or run_source
            )
            if name:
                per_round[number] = name
        if not per_round and run_source:
            per_round[0] = run_source
        out[run_id] = per_round
    return out


def resolve_source_path(
    source_name: str,
    *,
    original_name: str = "",
    sugar_root: Path | None = None,
) -> Path | None:
    """Return the on-disk CGM file for a corpus name or a saved own upload."""
    root = sugar_sugar_root(sugar_root)
    key = source_basename(source_name)
    original = source_basename(original_name) or key

    if key.lower() == "example.csv":
        path = root / "data" / "example.csv"
        return path if path.exists() else None

    d1 = _D1NAMO.match(key)
    if d1:
        subject = d1.group(1)
        candidates = [
            root
            / "data"
            / "d1namo"
            / "diabetes_subset_pictures-glucose-food-insulin"
            / subject
            / "glucose.csv",
            root / "data" / "d1namo" / subject / "glucose.csv",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    big = _BIGIDEAS.match(key)
    if big:
        subject = big.group(1)
        candidates = [
            root / "data" / "bigideas" / subject / f"Dexcom_{subject}.csv",
            root / "data" / "bigideas" / f"BIGIDEAS-{subject}.csv",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    loop = _LOOP.match(key)
    if loop:
        stem = Path(loop.group(1)).stem.replace("_chronological", "")
        path = root / "data" / "subjects" / stem / loop.group(1)
        return path if path.exists() else None

    if is_generic_corpus_name(key):
        return None

    return _find_upload(root, original)


def _find_upload(root: Path, original_name: str) -> Path | None:
    folder = users_dir(root)
    if not folder.exists() or not original_name:
        return None
    safe = safe_upload_filename(original_name)
    stem = Path(safe).stem.lower()
    matches: list[Path] = []
    for path in folder.iterdir():
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.endswith(safe.lower()) or stem in name:
            matches.append(path)
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def describe_resolution(path: Path | None, source_name: str) -> dict[str, Any]:
    with start_action(action_type="ai.resolve_source_path") as action:
        action.log(
            message_type="info",
            source=source_name,
            found=path is not None,
            path=str(path) if path else "",
        )
    return {"source_name": source_name, "path": None if path is None else str(path)}
