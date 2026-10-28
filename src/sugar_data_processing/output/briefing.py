"""A self-contained, contact-free briefing for readers who prefer plain language."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_livia_briefing(assessment: dict[str, Any], destination: Path) -> Path:
    """Render aggregate evidence and conditional recruitment targets without APIs."""
    public = {
        key: assessment[key]
        for key in (
            "assessment_date", "latest_record", "counts", "cohorts", "hypotheses",
            "primary_group_intervals", "quality_group_intervals", "exploratory",
            "quality_exploratory", "planning", "rank_correlation_sensitivity",
        )
    }
    template = Path(__file__).with_name("templates") / "livia_briefing.html"
    payload = json.dumps(public, allow_nan=True).replace("<", "\\u003c")
    payload = payload.replace(": NaN", ": null")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(template.read_text().replace("__ASSESSMENT_JSON__", payload))
    return destination
