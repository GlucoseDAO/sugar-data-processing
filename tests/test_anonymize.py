"""Anonymized statistics CSV: aliases stay linked, and H1–H5 do not move."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from sugar_data_processing.anonymize import anonymize_statistics_csv, initials_from_text
from sugar_data_processing.config import REPO_ROOT
from sugar_data_processing.fixtures.synthetic import write_synthetic_csv
from sugar_data_processing.gathering.load import load_prediction_statistics
from sugar_data_processing.gathering.participants import build_participant_table
from sugar_data_processing.gathering.rounds import build_round_table
from sugar_data_processing.gathering.sources import is_generic_corpus_name, source_basename
from sugar_data_processing.statistics.hypotheses import run_all_hypotheses

_EMAIL = re.compile(r"email(\d+)@gmail.com")
_REAL_CSV = REPO_ROOT / "data" / "raw" / "prediction_statistics.csv"


def test_initials_skip_file_words() -> None:
    assert initials_from_text("Ada Lovelace") == "AL"
    assert initials_from_text("Amy Liu") == "AL"
    assert initials_from_text("clarity_export.csv") == "P"


def test_same_person_keeps_one_alias_and_colliding_initials_are_numbered(tmp_path: Path) -> None:
    src = tmp_path / "in.csv"
    write_synthetic_csv(src, n_participants=4, seed=3)
    raw = pl.read_csv(src, infer_schema_length=0)
    study_ids = raw["study_id"].unique(maintain_order=True).to_list()
    ada_a, ada_b, amy, other = study_ids[:4]
    emails: list[str] = []
    papers: list[str] = []
    for study_id in raw["study_id"].to_list():
        if study_id in {ada_a, ada_b}:
            emails.append("Ada@Example.com" if study_id == ada_a else "ada@example.com")
            papers.append("Ada Lovelace")
        elif study_id == amy:
            emails.append("amy@example.com")
            papers.append("Amy Liu")
        else:
            emails.append("")
            papers.append("")
    raw = raw.with_columns(
        pl.Series("email", emails),
        pl.Series("paper_full_name", papers),
        pl.Series("location", ["Berlin"] * raw.height),
    )
    raw.write_csv(src)

    dest = tmp_path / "out.csv"
    anonymize_statistics_csv(src, dest)
    out = pl.read_csv(dest, infer_schema_length=0)
    text = dest.read_text(encoding="utf-8")

    ada_emails = set(out.filter(pl.col("study_id").is_in([ada_a, ada_b]))["email"].to_list())
    amy_emails = set(out.filter(pl.col("study_id") == amy)["email"].to_list())
    other_emails = set(out.filter(pl.col("study_id") == other)["email"].to_list())
    assert ada_emails == {"email1@gmail.com"}
    assert amy_emails == {"email2@gmail.com"}
    assert other_emails == {"email3@gmail.com"}

    ada_names = set(out.filter(pl.col("study_id").is_in([ada_a, ada_b]))["paper_full_name"].to_list())
    amy_names = set(out.filter(pl.col("study_id") == amy)["paper_full_name"].to_list())
    other_names = set(out.filter(pl.col("study_id") == other)["paper_full_name"].to_list())
    assert ada_names == {"AL"}
    assert amy_names == {"AL2"}
    assert other_names == {""}

    assert set(out["location"].to_list()) == {""}
    assert "Ada Lovelace" not in text
    assert "Amy Liu" not in text
    assert "ada@example.com" not in text.casefold()
    assert "Berlin" not in text
    assert "clarity_export.csv" not in text
    assert "D1NAMO-001.csv" in text
    assert "BIGIDEAS-001.csv" in text

    personal_labels = {
        source_basename(name)
        for name in out["data_source_name"].to_list()
        if name and not is_generic_corpus_name(name)
    }
    assert personal_labels <= {"AL.csv", "AL2.csv", "P.csv"}
    _assert_same_analysis(src, dest)


def test_real_export_round_trip_keeps_hypotheses(tmp_path: Path) -> None:
    if not _REAL_CSV.is_file():
        pytest.skip("data/raw/prediction_statistics.csv is not on this machine")

    original = pl.read_csv(_REAL_CSV, infer_schema_length=0)
    personal = _personal_filenames(original)
    assert personal, "expected personal upload filenames in the local export"

    dest = tmp_path / "prediction_statistics.csv"
    anonymize_statistics_csv(_REAL_CSV, dest)
    anon = pl.read_csv(dest, infer_schema_length=0)
    text = dest.read_text(encoding="utf-8")

    assert anon.height == original.height
    assert anon["study_id"].to_list() == original["study_id"].to_list()
    assert anon["run_id"].to_list() == original["run_id"].to_list()
    assert set(anon["location"].to_list()) <= {""}
    assert all(_EMAIL.fullmatch(value) for value in anon["email"].to_list())
    by_person = anon.select(["study_id", "email"]).unique()
    assert by_person.height == by_person["study_id"].n_unique()
    leaked = [name for name in personal if name in text]
    assert leaked == []
    assert "D1NAMO-" in text
    assert "BIGIDEAS-" in text
    _assert_same_analysis(_REAL_CSV, dest)


# Person-level fields that are means or flags, not "whichever run came last".
# Gender and free-text intervention can differ across a person's runs; group order
# is not part of the anonymized export.
_STABLE_PERSON = (
    "study_id",
    "diabetic",
    "uses_cgm",
    "cohort_category",
    "mae_overall",
    "mae_generic",
    "mae_own",
    "mae_primary",
    "mae_format_a",
    "mae_format_b",
    "mae_format_c",
    "n_rounds_generic",
    "n_rounds_own",
    "n_rounds_total",
    "n_runs",
    "eligible_primary",
    "eligible_own",
    "eligible_h5",
    "played_opposite_trait",
    "played_challenge_unknown",
)


def _analysis_payload(path: Path) -> dict[str, Any]:
    loaded = load_prediction_statistics(path)
    # Every saved round, so a renamed upload cannot hide a changed MAE or source class.
    rounds = build_round_table(loaded).sort(["study_id", "run_id", "round_number"])
    # One run per (person, format), ties broken by run_id. The pipeline's own
    # group_by().last() does not break timestamp ties, so this keeps the comparison
    # on the numbers rather than on row order.
    latest = _latest_run_per_format(loaded)
    participants = build_participant_table(latest).sort("study_id")
    keep = [column for column in _STABLE_PERSON if column in participants.columns]
    suite = run_all_hypotheses(participants).to_dict()
    return _plain(
        {
            "rounds": rounds.to_dicts(),
            "participants": participants.select(keep).to_dicts(),
            "hypotheses": suite,
        }
    )


def _latest_run_per_format(runs: pl.DataFrame) -> pl.DataFrame:
    winners = (
        runs.sort(["timestamp", "run_id"])
        .group_by(["study_id", "format"], maintain_order=True)
        .agg(pl.col("run_id").last())
    )
    return runs.join(winners.select("run_id"), on="run_id", how="semi")


def _assert_same_analysis(left: Path, right: Path) -> None:
    diffs = _diff(_analysis_payload(left), _analysis_payload(right), "analysis")
    assert not diffs, "renaming changed the analysis:\n" + "\n".join(diffs[:20])


def _diff(left: Any, right: Any, path: str) -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        found: list[str] = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                found.append(f"{path}.{key} missing on one side")
                continue
            found.extend(_diff(left[key], right[key], f"{path}.{key}"))
        return found
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [f"{path} length {len(left)} vs {len(right)}"]
        found = []
        for index, pair in enumerate(zip(left, right, strict=True)):
            found.extend(_diff(pair[0], pair[1], f"{path}[{index}]"))
        return found
    if isinstance(left, float) or isinstance(right, float):
        try:
            fa = float(left)
            fb = float(right)
        except (TypeError, ValueError):
            return [f"{path} {left!r} vs {right!r}"]
        if fa != fa and fb != fb:
            return []
        if abs(fa - fb) > 1e-8:
            return [f"{path} {fa} vs {fb}"]
        return []
    if left != right:
        return [f"{path} {left!r} vs {right!r}"]
    return []


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, float) and value != value:
        return None
    return value


def _personal_filenames(frame: pl.DataFrame) -> set[str]:
    """Exact source strings, including a URL whose last segment is the upload name."""
    found: set[str] = set()

    def remember(raw: object) -> None:
        text = str(raw or "").strip()
        base = source_basename(text)
        if text and base and not is_generic_corpus_name(base):
            found.add(text)

    for name in frame["data_source_name"].to_list():
        remember(name)
    for column in ("per_round_metrics", "round_context"):
        if column not in frame.columns:
            continue
        for cell in frame[column].to_list():
            text = str(cell or "").strip()
            if not text:
                continue
            parsed = ast.literal_eval(text)
            if not isinstance(parsed, list):
                continue
            for item in parsed:
                if isinstance(item, dict):
                    remember(item.get("data_source_name"))
    return found
