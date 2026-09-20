"""AI edition of the study report — human traits plus model comparison."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
from eliot import start_action

from sugar_data_processing.ai.ingest import ModelComparison
from sugar_data_processing.comparison.benchmarks import BenchmarkContext
from sugar_data_processing.config import (
    AI_EXPLORER_HTML,
    AI_REPORT_JSON,
    AI_REPORT_MD,
    EVALUATION_MODE_IN_PLACE,
    EVALUATION_MODE_LABELS,
    EVALUATION_MODE_POST_FACTUM,
    HUMAN_EXPLORER_HTML,
    HUMAN_REPORT_MD,
)
from sugar_data_processing.output.explorer import write_explorer_html
from sugar_data_processing.statistics.hypotheses import HypothesisSuite
from sugar_data_processing.verification.report import VerificationReport


def write_ai_report(
    *,
    participants: pl.DataFrame,
    runs: pl.DataFrame,
    suite: HypothesisSuite,
    benchmarks: BenchmarkContext,
    verification: VerificationReport,
    output_dir: Path,
    source_csv: Path,
    comparison: ModelComparison | None,
    scored_points: pl.DataFrame | None = None,
    sequences_dir: Path | None = None,
) -> Path:
    """Write the AI edition markdown + HTML (empty comparison is still a valid stub)."""
    with start_action(action_type="ai.write_ai_report") as action:
        reports_dir = Path(output_dir) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        md_path = reports_dir / AI_REPORT_MD
        md_path.write_text(
            _render_ai_markdown(
                stamp=stamp,
                source_csv=source_csv,
                participants=participants,
                suite=suite,
                benchmarks=benchmarks,
                verification=verification,
                comparison=comparison,
                sequences_dir=sequences_dir,
            ),
            encoding="utf-8",
        )
        explorer_path = write_explorer_html(
            participants=participants,
            runs=runs,
            output_path=reports_dir / AI_EXPLORER_HTML,
            source_csv=source_csv,
            suite=suite,
            benchmarks=benchmarks,
            verification=verification,
            edition="ai",
            comparison=comparison.to_dict() if comparison is not None else None,
            sequences_dir=sequences_dir,
        )
        payload: dict[str, Any] = {
            "edition": "ai",
            "generated_at": stamp,
            "source_csv": str(source_csv),
            "n_runs": runs.height,
            "n_participants": participants.height,
            "hypotheses": suite.to_dict(),
            "benchmarks": benchmarks.to_dict(),
            "verification": verification.to_dict(),
            "model_comparison": None if comparison is None else comparison.to_dict(),
            "n_scored_points": 0 if scored_points is None else scored_points.height,
            "explorer": str(explorer_path),
            "human_report": HUMAN_REPORT_MD,
        }
        (reports_dir / AI_REPORT_JSON).write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        action.log(message_type="info", report=str(md_path))
        return md_path


def _render_ai_markdown(
    *,
    stamp: str,
    source_csv: Path,
    participants: pl.DataFrame,
    suite: HypothesisSuite,
    benchmarks: BenchmarkContext,
    verification: VerificationReport,
    comparison: ModelComparison | None,
    sequences_dir: Path | None,
) -> str:
    seq = str(sequences_dir) if sequences_dir is not None else "(not exported in this run)"
    notes = "\n".join(f"- {n}" for n in suite.notes) if suite.notes else "- None"
    if comparison is None or comparison.n_points == 0:
        model_section = _empty_model_section(seq)
    else:
        model_section = _filled_model_section(comparison, seq)

    return f"""# Sugar Sugar AI Study Analysis Report

Generated: **{stamp}**  
Source: `{source_csv}`  
Edition: **AI** (companion to the [human report]({HUMAN_REPORT_MD}))

This edition keeps every trait of the human analysis (cohort, H1–H5, Challenge
the unknown / opposite-trait, literature bands, verification) and adds the
comparison against models scored on the exported sequences.

Two scoring categories exist:

| Mode | Meaning | Status now |
| --- | --- | --- |
| `{EVALUATION_MODE_POST_FACTUM}` | {EVALUATION_MODE_LABELS[EVALUATION_MODE_POST_FACTUM]} | Available — this is how we reuse saved games |
| `{EVALUATION_MODE_IN_PLACE}` | {EVALUATION_MODE_LABELS[EVALUATION_MODE_IN_PLACE]} | Not collected yet |

{model_section}

## Human analysis (carried through)

The hypothesis battery and literature placement are the same numbers as the
human edition. Open [`{HUMAN_EXPLORER_HTML}`]({HUMAN_EXPLORER_HTML}) or
[`{AI_EXPLORER_HTML}`]({AI_EXPLORER_HTML}) for the interactive view.

- Participants: **{participants.height}**
- Human mean MAE: **{benchmarks.human_mean_mae:.2f} mg/dL**
- Verification schema: **{"PASSED" if verification.schema_ok else "FAILED"}**
- Pipeline notes:
{notes}

## How the two-stage AI path works

1. **Export** — `sdp export-ai` (also run by `sdp analyze`) writes
   `prediction_points.csv` / `prediction_rounds.csv` under `data/processed/ai/`.
   Each point has `timestamp`, `real_mgdl`, `human_predicted_mgdl`, and window
   **location** (`window_start_index` + window times).
2. **Score** — any model reads that CSV and writes predictions back with the
   join keys `study_id, run_id, round_number, point_index` plus `model_name`,
   `model_predicted_mgdl`, and `evaluation_mode`.
3. **Ingest** — `sdp ingest-ai --predictions path/to/model.csv` joins the
   scores and rewrites this AI report.
"""


def _empty_model_section(seq: str) -> str:
    return f"""## Model comparison

No ingested model scores yet. Sequences are ready at `{seq}`.

Feed `prediction_points.csv` to a model, write a CSV with the contract in
`manifest.json`, then run `sdp ingest-ai`. Until that happens this edition is
the scaffold: same human traits, empty comparison, evaluation mode =
`{EVALUATION_MODE_POST_FACTUM}` only.
"""


def _mode_model_cell(info: dict[str, Any]) -> str:
    models = info.get("model_mean_mae") or {}
    if not isinstance(models, dict) or not models:
        return "—"
    return ", ".join(f"{name}={mae:.2f}" for name, mae in models.items())


def _filled_model_section(comparison: ModelComparison, seq: str) -> str:
    model_rows = "\n".join(
        f"| `{name}` | {mae:.2f} |" for name, mae in sorted(comparison.model_mean_mae.items())
    )
    mode_rows = "\n".join(
        f"| `{mode}` | {info.get('n_points', 0)} | {info.get('n_people', 0)} | "
        f"{float(info.get('human_mean_mae') or 0.0):.2f} | {_mode_model_cell(info)} |"
        for mode, info in comparison.by_mode.items()
    )
    human = (
        f"{comparison.human_mean_mae:.2f}"
        if comparison.human_mean_mae is not None
        else "—"
    )
    return f"""## Model comparison

Sequences: `{seq}`  
Scored points: **{comparison.n_points}** across **{comparison.n_people}** people  
Models: {', '.join(f'`{m}`' for m in comparison.models) or '—'}  
Human point MAE on the scored subset: **{human} mg/dL**

| Model | Mean MAE (mg/dL) |
| --- | ---: |
{model_rows or "| — | — |"}

| Evaluation mode | Points | People | Human MAE | Model MAE |
| --- | ---: | ---: | ---: | --- |
{mode_rows or "| — | — | — | — | — |"}
"""
