"""Parse sugar-sugar CSV encodings (duration pairs, optional bools, literals)."""

from __future__ import annotations

import ast
from typing import Any

CGM_DURATION_UNITS: tuple[str, ...] = ("weeks", "months", "years")
_UNIT_TO_YEARS: dict[str, float] = {
    "weeks": 1.0 / 52.0,
    "months": 1.0 / 12.0,
    "years": 1.0,
}


def parse_literal(value: Any) -> Any:
    """Parse a Python-literal cell, or return the already-parsed object."""
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    return ast.literal_eval(text)


def parse_optional_bool(value: Any) -> bool | None:
    """Three-state bool: true / false / unknown (blank)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "null"}:
        return None
    if text in {"true", "1", "yes", "t", "y"}:
        return True
    if text in {"false", "0", "no", "f", "n"}:
        return False
    return None


def normalize_cgm_duration_unit(unit: str | None) -> str:
    text = str(unit or "years").strip().lower()
    if text in {"week", "weeks"}:
        return "weeks"
    if text in {"month", "months"}:
        return "months"
    if text in {"year", "years"}:
        return "years"
    return "years"


def parse_cgm_duration(raw: Any) -> tuple[float | None, str]:
    """Parse a stored CGM duration into ``(value, unit)``.

    Accepts a bare number (legacy years), a ``value,unit`` string such as
    ``6,months``, a ``(value, unit)`` / ``[value, unit]`` pair, or empty.
    """
    if raw is None:
        return None, "years"
    if isinstance(raw, bool):
        return None, "years"
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return _as_number(raw[0]), normalize_cgm_duration_unit(str(raw[1]))
    if isinstance(raw, (int, float)):
        return float(raw), "years"

    text = str(raw).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None, "years"
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    if "," in text:
        left, right = text.split(",", 1)
        unit_guess = normalize_cgm_duration_unit(right)
        if right.strip().lower() in {
            "week",
            "weeks",
            "month",
            "months",
            "year",
            "years",
        }:
            return _as_number(left), unit_guess
    return _as_number(text), "years"


def cgm_duration_to_years(raw: Any) -> float | None:
    """Convert a CSV CGM-duration cell to years (or ``None``)."""
    value, unit = parse_cgm_duration(raw)
    if value is None:
        return None
    return float(value) * _UNIT_TO_YEARS[normalize_cgm_duration_unit(unit)]


def parse_duration_years(raw: Any) -> float | None:
    """Parse a duration that is usually a bare year count, with the pair form allowed."""
    return cgm_duration_to_years(raw)


def parse_round_context(cell: Any) -> list[dict[str, Any]]:
    """Parse the ``round_context`` Python-literal list (empty if missing)."""
    parsed = parse_literal(cell)
    if parsed is None:
        return []
    if not isinstance(parsed, list):
        raise TypeError(f"round_context must be a list, got {type(parsed)}")
    return [item for item in parsed if isinstance(item, dict)]


def parse_per_round_metrics(cell: Any) -> list[dict[str, Any]]:
    """Parse the Python-literal list stored in ``per_round_metrics``."""
    parsed = parse_literal(cell)
    if parsed is None:
        return []
    if not isinstance(parsed, list):
        raise TypeError(f"per_round_metrics must be a list, got {type(parsed)}")
    return [item for item in parsed if isinstance(item, dict)]


def _as_number(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip()
    if text == "":
        return None
    return float(text)
