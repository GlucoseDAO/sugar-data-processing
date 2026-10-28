"""Build a verifier copy of ``prediction_statistics.csv``.

This is separate from fetch and from the analysis pipeline. Verifiers rebuild
the HTML themselves with ``sdp analyze --csv`` on the file this module writes.

What changes
------------
* Email — ``email1@gmail.com``, ``email2@gmail.com``, … One alias per person.
  The same address, or the same ``study_id``, always keeps the same alias.
  Two ``study_id`` values that shared an address stay on one alias.
* Names — initials (``Ada Lovelace`` → ``AL``). A later person with the same
  initials gets ``AL2``, ``AL3``, …
* Personal upload filenames — that person's initials plus ``.csv``.
  Published corpus names (``D1NAMO-001.csv``, ``BIGIDEAS-001.csv``, …) stay,
  because those names decide diabetic vs non-diabetic traces.
* Location — cleared. It is not used by the tests.

``study_id`` and ``run_id`` stay. They are the random ids the report already
groups on. Metrics, demographics, and round numbers stay so H1–H5 can be
recomputed.

No crosswalk of real address → alias is written. That file would undo this.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
from eliot import start_action

from sugar_data_processing.gathering.load import statistics_csv_paths
from sugar_data_processing.gathering.sources import is_generic_corpus_name, source_basename

_NAME_FIELDS: tuple[tuple[str, int], ...] = (
    ("paper_full_name", 0),
    ("nickname", 1),
    ("full_name", 1),
    ("name", 1),
)
_LITERAL_FIELDS: tuple[str, ...] = ("per_round_metrics", "round_context")
#: Which file of the export and its archives a row came from, while they are anonymized together.
_PART_COLUMN: str = "_anonymize_part"
_FILENAME_NOISE: frozenset[str] = frozenset(
    {
        "csv",
        "json",
        "txt",
        "xlsx",
        "cgm",
        "dexcom",
        "libre",
        "clarity",
        "export",
        "data",
        "glucose",
        "upload",
        "file",
        "user",
        "users",
        "bg",
        "nightscout",
        "patient",
        "report",
        "download",
        "copy",
        "final",
        "new",
        "old",
    }
)
_WORD = re.compile(r"[^\W\d_]+", flags=re.UNICODE)


@dataclass(frozen=True)
class AnonymizeResult:
    """Counts only. The real-to-alias map is not kept."""

    dest: Path
    n_rows: int
    n_people: int
    n_email_aliases: int
    n_name_cells: int
    n_personal_filenames: int


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def add(self, key: str) -> str:
        self._parent.setdefault(key, key)
        return key

    def find(self, key: str) -> str:
        self.add(key)
        while self._parent[key] != key:
            self._parent[key] = self._parent[self._parent[key]]
            key = self._parent[key]
        return key

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self._parent[root_right] = root_left


def initials_from_text(text: str) -> str:
    """First letters of a name or filename, skipping generic file words."""
    stem = Path(str(text or "")).stem
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", stem)
    tokens = _WORD.findall(spaced)
    kept = [token for token in tokens if token.casefold() not in _FILENAME_NOISE]
    if not kept:
        return "P"
    return "".join(token[0].upper() for token in kept)[:8]


def anonymize_statistics_frame(frame: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int]]:
    """Return a copy of ``frame`` with contact fields replaced.

    ``frame`` must be the all-text CSV read (``infer_schema_length=0``) so
    quoted round lists survive.
    """
    records = list(frame.iter_rows(named=True))
    people = _people_by_study(records)
    rewritten, counts = _rewrite_rows(records, people, list(frame.columns))
    out = pl.DataFrame(rewritten, schema={column: pl.Utf8 for column in frame.columns})
    return out, counts


def anonymize_statistics_csv(src: Path | str, dest: Path | str) -> AnonymizeResult:
    """Write the verifier CSV. Does not touch ``src``."""
    source = Path(src)
    target = Path(dest)
    with start_action(action_type="anonymize.statistics_csv", source=str(source)) as action:
        if not source.is_file():
            raise FileNotFoundError(f"Statistics CSV not found: {source}")
        # Archived games beside the export are anonymized together with it, so a
        # person keeps one alias across files, and written back as their own files.
        parts = [(path, pl.read_csv(path, infer_schema_length=0)) for path in statistics_csv_paths(source)]
        combined = pl.concat(
            [frame.with_columns(pl.lit(str(i)).alias(_PART_COLUMN)) for i, (_, frame) in enumerate(parts)],
            how="diagonal",
        )
        rewritten, counts = anonymize_statistics_frame(combined)
        target.parent.mkdir(parents=True, exist_ok=True)
        for i, (path, frame) in enumerate(parts):
            out = target if i == 0 else target.with_name(path.name)
            tmp = out.with_name(out.name + ".tmp")
            rewritten.filter(pl.col(_PART_COLUMN) == str(i)).select(frame.columns).write_csv(tmp)
            tmp.replace(out)
        result = AnonymizeResult(
            dest=target,
            n_rows=rewritten.height,
            n_people=counts["n_people"],
            n_email_aliases=counts["n_email_aliases"],
            n_name_cells=counts["n_name_cells"],
            n_personal_filenames=counts["n_personal_filenames"],
        )
        action.log(
            message_type="info",
            dest=str(target),
            n_rows=result.n_rows,
            n_people=result.n_people,
            n_email_aliases=result.n_email_aliases,
            n_name_cells=result.n_name_cells,
            n_personal_filenames=result.n_personal_filenames,
        )
        return result


def _people_by_study(records: list[dict[str, Any]]) -> dict[str, str]:
    """``study_id`` → person id.

    Rows that share a ``study_id`` are one person. Rows that share a non-empty
    email are one person too, even when the ``study_id`` differs.
    """
    groups = _UnionFind()
    for record in records:
        study_key = f"study:{record.get('study_id') or ''}"
        groups.add(study_key)
        email = _norm_email(record.get("email"))
        if email:
            groups.union(study_key, f"email:{email}")
    return {
        str(record.get("study_id") or ""): groups.find(f"study:{record.get('study_id') or ''}")
        for record in records
    }


def _best_names(
    records: list[dict[str, Any]],
    person_of_study: dict[str, str],
) -> dict[str, tuple[int, str]]:
    """Best label source per person. Lower rank wins (paper name, then filename)."""
    best: dict[str, tuple[int, str]] = {}
    for record in records:
        person = person_of_study[str(record.get("study_id") or "")]
        for column, rank in _NAME_FIELDS:
            text = str(record.get(column) or "").strip()
            if not text:
                continue
            current = best.get(person)
            if current is None or rank < current[0]:
                best[person] = (rank, text)
        if person in best:
            continue
        for filename in _personal_filenames(record):
            best[person] = (2, Path(source_basename(filename)).stem)
            break
    return best


def _rewrite_rows(
    records: list[dict[str, Any]],
    person_of_study: dict[str, str],
    columns: list[str],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    names = _best_names(records, person_of_study)
    alias_of: dict[str, str] = {}
    initials_of: dict[str, str] = {}
    used_tokens: dict[str, str] = {}
    next_email = 1
    n_name_cells = 0
    n_personal_filenames = 0
    rewritten: list[dict[str, str]] = []

    for record in records:
        person = person_of_study[str(record.get("study_id") or "")]
        if person not in alias_of:
            alias_of[person] = f"email{next_email}@gmail.com"
            next_email += 1
        row = {
            column: "" if record.get(column) is None else str(record.get(column))
            for column in columns
        }
        if "email" in row:
            row["email"] = alias_of[person]
        if "location" in row:
            row["location"] = ""

        personal = _personal_filenames(record)
        name_present = any(str(record.get(column) or "").strip() for column, _rank in _NAME_FIELDS)
        token = ""
        if personal or name_present:
            token = _initials_for(person, names, initials_of, used_tokens)
        label = _personal_label(token) if token else ""

        for column, _rank in _NAME_FIELDS:
            if column not in row or not str(record.get(column) or "").strip():
                continue
            row[column] = token
            n_name_cells += 1

        if "data_source_name" in row:
            raw_name = source_basename(row["data_source_name"])
            if raw_name and not is_generic_corpus_name(raw_name):
                row["data_source_name"] = label

        if personal:
            replacements = {filename: label for filename in personal}
            n_personal_filenames += len(personal)
            for column in _LITERAL_FIELDS:
                if column in row:
                    row[column] = _rewrite_literal(row[column], replacements)
        rewritten.append(row)

    counts = {
        "n_people": len(set(person_of_study.values())),
        "n_email_aliases": len(alias_of),
        "n_name_cells": n_name_cells,
        "n_personal_filenames": n_personal_filenames,
    }
    return rewritten, counts


def _token_taken(token: str, person: str, used_tokens: dict[str, str]) -> bool:
    owner = used_tokens.get(token)
    if owner not in (None, person):
        return True
    return is_generic_corpus_name(_personal_label(token))


def _initials_for(
    person: str,
    names: dict[str, tuple[int, str]],
    initials_of: dict[str, str],
    used_tokens: dict[str, str],
) -> str:
    existing = initials_of.get(person)
    if existing is not None:
        return existing
    base = initials_from_text(names.get(person, (2, "P"))[1])
    token = base
    number = 2
    while _token_taken(token, person, used_tokens):
        token = f"{base}{number}"
        number += 1
    used_tokens[token] = person
    initials_of[person] = token
    return token


def _personal_label(token: str) -> str:
    label = f"{token}.csv"
    if is_generic_corpus_name(label):
        return f"upload-{token}.csv"
    return label


def _personal_filenames(record: dict[str, Any]) -> list[str]:
    """Exact ``data_source_name`` strings that are not published corpus files."""
    found: list[str] = []
    _remember_personal(found, record.get("data_source_name"))
    for column in _LITERAL_FIELDS:
        text = str(record.get(column) or "").strip()
        if not text:
            continue
        parsed = ast.literal_eval(text)
        if not isinstance(parsed, list):
            continue
        for item in parsed:
            if isinstance(item, dict):
                _remember_personal(found, item.get("data_source_name"))
    return found


def _remember_personal(found: list[str], raw: object) -> None:
    text = str(raw or "").strip()
    if not text or text in found:
        return
    base = source_basename(text)
    if base and not is_generic_corpus_name(base):
        found.append(text)


def _rewrite_literal(cell: str, replacements: dict[str, str]) -> str:
    """Replace personal filenames inside one Python-literal cell.

    Only the ``data_source_name`` string tokens move. Numbers and corpus
    filenames stay as they were written.
    """
    text = "" if cell is None else str(cell)
    if not text.strip() or not replacements:
        return text
    tree = ast.parse(text, mode="eval")
    spans: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=True):
            if not isinstance(key, ast.Constant) or key.value != "data_source_name":
                continue
            if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                continue
            new_name = replacements.get(value.value)
            if new_name is None:
                continue
            if (
                value.lineno is None
                or value.col_offset is None
                or value.end_lineno is None
                or value.end_col_offset is None
            ):
                raise ValueError("A personal upload filename could not be located in a round cell.")
            start = _offset(text, value.lineno, value.col_offset)
            end = _offset(text, value.end_lineno, value.end_col_offset)
            spans.append((start, end, repr(new_name)))
    if not spans:
        return text
    updated = text
    for start, end, quoted in sorted(spans, reverse=True):
        updated = updated[:start] + quoted + updated[end:]
    parsed_after = ast.literal_eval(updated)
    if isinstance(parsed_after, list):
        for item in parsed_after:
            if not isinstance(item, dict):
                continue
            if str(item.get("data_source_name") or "") in replacements:
                raise ValueError("A personal upload filename was still present after anonymization.")
    return updated


def _offset(text: str, lineno: int, column: int) -> int:
    lines = text.splitlines(keepends=True)
    return sum(len(line) for line in lines[: lineno - 1]) + column


def _norm_email(value: object) -> str:
    return str(value or "").strip().casefold()
