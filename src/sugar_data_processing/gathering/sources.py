"""Classify predicted-trace sources the way current sugar-sugar scoreboards do."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from sugar_data_processing.config import DATA_CLASS_DIABETIC, DATA_CLASS_NONDIABETIC
from sugar_data_processing.gathering.encoding import parse_optional_bool

_D1NAMO_NAME = re.compile(r"^D1NAMO-(\d{3})\.csv$", re.IGNORECASE)
_BIGIDEAS_NAME = re.compile(r"^BIGIDEAS-(\d{3})\.csv$", re.IGNORECASE)
_DEXCOM_BIGIDEAS = re.compile(r"^dexcom_(\d{3})\.csv$", re.IGNORECASE)

# Bundled generic fallback is an insulin-treated Dexcom trace (declared diabetic).
_DECLARED_GENERIC_CLASS: dict[str, str] = {
    "example.csv": DATA_CLASS_DIABETIC,
}

OWN_UPLOAD_LABEL: str = "own_upload"


def source_basename(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    return Path(text).name


def is_generic_corpus_name(name: str) -> bool:
    """True when the filename is a published generic subject, not a personal upload."""
    key = source_basename(name)
    if not key:
        return False
    if key.lower() == "example.csv":
        return True
    if _D1NAMO_NAME.match(key) or _BIGIDEAS_NAME.match(key) or _DEXCOM_BIGIDEAS.match(key):
        return True
    if key.endswith("_chronological.csv"):
        return True
    return False


def redact_source_name(raw: Any) -> str:
    """Keep corpus names; collapse personal upload filenames to ``own_upload``."""
    key = source_basename(raw)
    if not key:
        return ""
    if is_generic_corpus_name(key):
        return key
    return OWN_UPLOAD_LABEL


def classify_data_class(
    source_name: str,
    *,
    is_example: bool | None,
    player_diabetic: bool | None,
) -> str | None:
    """Which kind of glucose trace a round predicted, or ``None`` if unknown.

    Mirrors sugar-sugar ``scoreboard.classify_round_source``:

    * D1NAMO / legacy LOOP chronological files → diabetic data
    * BIG IDEAs → non-diabetic data
    * declared generic metadata (``example.csv``) → diabetic data
    * own upload → the player's own diabetes status
    """
    name = source_basename(source_name)
    if not name:
        return None
    if _D1NAMO_NAME.match(name) or name.endswith("_chronological.csv"):
        return DATA_CLASS_DIABETIC
    if _BIGIDEAS_NAME.match(name) or _DEXCOM_BIGIDEAS.match(name):
        return DATA_CLASS_NONDIABETIC
    if name in _DECLARED_GENERIC_CLASS:
        return _DECLARED_GENERIC_CLASS[name]
    if is_example is True or name.lower() == "example.csv":
        return _DECLARED_GENERIC_CLASS.get("example.csv")
    if is_example is False or name == OWN_UPLOAD_LABEL:
        if player_diabetic is True:
            return DATA_CLASS_DIABETIC
        if player_diabetic is False:
            return DATA_CLASS_NONDIABETIC
        return None
    return None


def coerce_example_flag(raw: Any) -> bool | None:
    return parse_optional_bool(raw)
