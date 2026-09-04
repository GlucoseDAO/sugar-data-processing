"""Assemble a human-readable markdown report with embedded figures."""

from __future__ import annotations

import base64
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
from eliot import start_action

from sugar_data_processing.comparison.benchmarks import BenchmarkContext
from sugar_data_processing.config import COHORT_LABELS
from sugar_data_processing.output.explorer import write_explorer_html
from sugar_data_processing.output.narration import (
    explain_hypothesis_result,
    how_to_read_report,
)
from sugar_data_processing.output.plots import generate_all_figures
from sugar_data_processing.statistics.catalog import HYPOTHESES
from sugar_data_processing.statistics.hypotheses import HypothesisSuite
from sugar_data_processing.verification.anomalies import Anomaly
from sugar_data_processing.verification.report import VerificationReport


def write_report(
    *,
    runs: pl.DataFrame,
    participants: pl.DataFrame,
    suite: HypothesisSuite,
    benchmarks: BenchmarkContext,
    verification: VerificationReport,
    output_dir: Path,
    source_csv: Path,
) -> Path:
    """Write markdown + JSON artefacts under ``output_dir``.

    Figures are written under ``output_dir/figures``, copied beside the report
    under ``output_dir/reports/figures``, and embedded as base64 PNG data URIs
    inside the markdown so the report displays images without external paths.
    """
    with start_action(action_type="output.write_report") as action:
        output_dir = Path(output_dir)
        figures_dir = output_dir / "figures"
        reports_dir = output_dir / "reports"
        report_figures_dir = reports_dir / "figures"
        # Prefer repo-standard layout when output_dir is ./output
        if output_dir.name == "output":
            processed_dir = output_dir.parent / "data" / "processed"
        else:
            processed_dir = output_dir / "processed"

        figures_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        participants.write_parquet(processed_dir / "participants.parquet")
        runs.write_parquet(processed_dir / "runs.parquet")
        csv_ready = participants
        if "formats_played" in csv_ready.columns:
            csv_ready = csv_ready.with_columns(
                pl.col("formats_played").cast(pl.List(pl.Utf8)).list.join(",").alias("formats_played")
            )
        csv_ready.write_csv(processed_dir / "participants.csv")

        figure_paths = generate_all_figures(participants, suite, benchmarks, figures_dir)

        # Copy PNGs next to the markdown for viewers that prefer file links
        if report_figures_dir.exists():
            shutil.rmtree(report_figures_dir)
        report_figures_dir.mkdir(parents=True, exist_ok=True)
        report_figure_paths: dict[str, Path] = {}
        for key, src in figure_paths.items():
            dest = report_figures_dir / src.name
            shutil.copy2(src, dest)
            report_figure_paths[key] = dest

        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        md_path = reports_dir / "study_analysis_report.md"
        md = _render_markdown(
            stamp=stamp,
            source_csv=source_csv,
            runs=runs,
            participants=participants,
            suite=suite,
            benchmarks=benchmarks,
            verification=verification,
            figure_paths=report_figure_paths,
        )
        md_path.write_text(md, encoding="utf-8")

        explorer_path = write_explorer_html(
            participants=participants,
            runs=runs,
            output_path=reports_dir / "study_explorer.html",
            source_csv=source_csv,
        )

        payload: dict[str, Any] = {
            "generated_at": stamp,
            "source_csv": str(source_csv),
            "n_runs": runs.height,
            "n_participants": participants.height,
            "verification": verification.to_dict(),
            "hypotheses": suite.to_dict(),
            "hypothesis_catalog": HYPOTHESES,
            "benchmarks": benchmarks.to_dict(),
            "figures": {k: str(v) for k, v in report_figure_paths.items()},
            "explorer": str(explorer_path),
        }
        (reports_dir / "study_analysis_report.json").write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        action.log(message_type="info", report=str(md_path), n_figures=len(figure_paths))
        return md_path


def _embed_png(path: Path, caption: str) -> str:
    """Embed a PNG as a markdown image (base64 data URI + local relative path)."""
    if not path.exists():
        return f"_{caption}: figure missing._"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    rel = f"figures/{path.name}"
    # Data URI so the MD preview shows the plot without resolving paths;
    # relative link kept for exporters / GitHub-style viewers that prefer files.
    return (
        f"![{caption}](data:image/png;base64,{encoded})\n\n"
        f"*Figure: {caption}*  \n"
        f"[PNG file]({rel})"
    )


def _cohort_table(participants: pl.DataFrame) -> str:
    if "cohort_category" not in participants.columns:
        return ""
    lines = [
        "| Category | People | What it is |",
        "| --- | ---: | --- |",
    ]
    counts = (
        participants.group_by("cohort_category")
        .len()
        .to_dicts()
    )
    by_key = {str(row["cohort_category"]): int(row["len"]) for row in counts}
    for key, label in COHORT_LABELS.items():
        if key not in by_key:
            continue
        lines.append(f"| {label} | {by_key[key]} | `{key}` |")
    return "\n".join(lines)


def _issue_table(issues: list[Anomaly], limit: int = 80) -> str:
    lines: list[str] = []
    for a in issues[:limit]:
        lines.append(
            f"| `{a.study_id[:12]}` | {a.category} | {a.severity} | {a.detail} |"
        )
    if len(issues) > limit:
        lines.append(f"| … | … | … | ({len(issues) - limit} more omitted) |")
    return "\n".join(lines) if lines else "| — | — | — | none |"


def _hypothesis_section(
    key: str,
    result: dict[str, Any] | None,
    figure_blocks: list[str],
) -> str:
    figures = "\n\n".join(figure_blocks)
    narrative = explain_hypothesis_result(key, result)
    return f"""{narrative}

**Figure(s)**

{figures}
"""


def _render_markdown(
    *,
    stamp: str,
    source_csv: Path,
    runs: pl.DataFrame,
    participants: pl.DataFrame,
    suite: HypothesisSuite,
    benchmarks: BenchmarkContext,
    verification: VerificationReport,
    figure_paths: dict[str, Path],
) -> str:
    hyp = suite.to_dict()
    n_primary = int(participants.filter(pl.col("eligible_primary")).height)
    n_h5 = int(participants.filter(pl.col("eligible_h5")).height)
    n_repeat = (
        int(participants.filter(pl.col("is_repeat_player")).height)
        if "is_repeat_player" in participants.columns
        else 0
    )
    n_all_formats = (
        int(participants.filter(pl.col("played_all_formats")).height)
        if "played_all_formats" in participants.columns
        else 0
    )
    n_runs_total = (
        int(participants["n_runs"].sum()) if "n_runs" in participants.columns else runs.height
    )
    cohort_rows = _cohort_table(participants)
    all_issues = verification.all_issues
    schema_table = _issue_table(verification.schema_issues)
    quality_table = _issue_table(verification.quality_flags)

    def img(key: str, caption: str) -> str:
        path = figure_paths.get(key)
        if path is None:
            return f"_{caption}: figure missing._"
        return _embed_png(path, caption)

    notes_md = "\n".join(f"- {n}" for n in suite.notes) if suite.notes else "- None"
    schema_status = "PASSED" if verification.schema_ok else "FAILED"

    catalog_rows = "\n".join(
        f"| {info['code']} | {info['title']} | {info['question']} |"
        for info in HYPOTHESES.values()
    )
    reading_guide = how_to_read_report()

    return f"""# Sugar Sugar Study Analysis Report

Generated: **{stamp}**  
Source: `{source_csv}`

This report follows **Section 7 (Statistical Analysis Plan)** of
*Human Prediction of Next-Hour Glucose from Prior CGM Context*.

It is written so a student can follow the pipeline without reading the source code:
what was measured, who was included, what each test asked, and what the numbers mean.

{reading_guide}

## Hypothesis key (plain language)

| Code | Title | Question |
| --- | --- | --- |
{catalog_rows}

## 1. Cohort snapshot

**Goal of this section:** see how many sessions and people entered the analysis,
and what overall prediction accuracy looks like.

| Metric | Value | What it means |
| --- | ---: | --- |
| Raw runs (rows) | {runs.height} | Completed app sessions in the export |
| Unique participants | {participants.height} | Distinct people (`study_id`) |
| Repeat players | {n_repeat} | People with more than one saved run |
| Saved runs (all people) | {n_runs_total} | Sessions after counting replays |
| Played every variant (A+B+C) | {n_all_formats} | People who tried generic, own, and mixed |
| Eligible for primary analyses | {n_primary} | ≥6 generic segments (used for H1–H4) |
| Eligible for own-vs-generic test | {n_h5} | ≥6 generic **and** ≥6 own (used for H5) |
| Mean person MAE (mg/dL) | {benchmarks.human_mean_mae:.2f} | Average accuracy (lower is better) |
| Median person MAE (mg/dL) | {benchmarks.human_median_mae:.2f} | Typical person (robust to outliers) |
| SD person MAE (mg/dL) | {benchmarks.human_sd_mae:.2f} | Spread of accuracy across people |

{cohort_rows}

{img("mae_distribution", "Distribution of per-person MAE (mg/dL)")}

{img("cohort_pie", "Unique people in each diabetes × CGM category")}

{img("mae_by_format", "Person MAE on each task: A generic, B own data, C mixed")}

{img("players_vs_repeats", "How many people played once vs came back, and their MAE")}

{img("all_formats_own_vs_generic", "People who played every variant: own-data MAE vs generic MAE")}

Interactive filters for the same pictures live in [`study_explorer.html`](study_explorer.html).

## 2. Analysis population rules (§7.2)

**Why this matters:** statistical tests only include people with enough completed
segments. Otherwise short incomplete sessions would dominate the results.

- **Primary analyses (H1–H4):** ≥6 analyzable **generic** segments.
- **Own-data analyses:** ≥6 **own-data** segments.
- **Own-vs-generic paired analysis (H5):** both thresholds.
- **Person-level MAE:** mean of round MAEs so one busy participant cannot inflate
  the sample size by playing many rounds.

## 3. Primary hypotheses

**What these ask:** do two groups of people differ in prediction accuracy?

{_hypothesis_section(
    "h1",
    hyp["h1"],
    [img("mae_by_diabetes", "Person MAE by diabetes status (PwD vs non-PwD)")],
)}

{_hypothesis_section(
    "h2",
    hyp["h2"],
    [img("mae_by_cgm", "Person MAE by CGM use (users vs non-users)")],
)}

## 4. Secondary hypotheses

**What these ask:** does longer experience help, and is accuracy better on own data
than on generic example data?

{_hypothesis_section(
    "h3",
    hyp["h3"],
    [
        img(
            "diabetes_duration_scatter",
            "Diabetes duration (years) vs person MAE among PwD",
        )
    ],
)}

{_hypothesis_section(
    "h4",
    hyp["h4"],
    [
        img(
            "cgm_duration_scatter",
            "CGM experience (years) vs person MAE among CGM users",
        ),
        img(
            "duration_bins",
            "Exploratory MAE by diabetes-duration and CGM-experience bins",
        ),
    ],
)}

{_hypothesis_section(
    "h5",
    hyp["h5"],
    [
        img(
            "own_vs_generic",
            "Paired own-data MAE vs generic-data MAE (below diagonal = better on own data)",
        )
    ],
)}

{explain_hypothesis_result("h6", hyp.get("h6"))}

## 5. Literature / GlucoBench context (§7.5)

**Goal of this section:** place human MAE next to published 60-minute model bands.
This is contextual comparison, not the deferred formal H6 baseline test.

{benchmarks.narrative}

| Band | Range (mg/dL) | % of humans inside | How to read it |
| --- | --- | ---: | --- |
| Simple / ARIMA | {benchmarks.simple_baseline_mae_range[0]:.0f}–{benchmarks.simple_baseline_mae_range[1]:.0f} | {benchmarks.pct_inside_simple_baseline_band:.1f}% | Typical simple forecasting models |
| Deep learning | {benchmarks.deep_learning_mae_range[0]:.0f}–{benchmarks.deep_learning_mae_range[1]:.0f} | {benchmarks.pct_inside_deep_learning_band:.1f}% | Typical deep-learning reports |
| Personalized | {benchmarks.personalized_mae_range[0]:.0f}–{benchmarks.personalized_mae_range[1]:.0f} | {benchmarks.pct_inside_personalized_band:.1f}% | Personalized-model band |
| Below simple-band low | < {benchmarks.simple_baseline_mae_range[0]:.0f} | {benchmarks.pct_below_simple_baseline_low:.1f}% | Better than the simple-band floor |

{img("benchmark_bands", "Human MAE density vs published simple and deep-learning bands")}

## 6. Data verification

**Goal of this section:** show whether the input data looked structurally valid and
whether any demographic / metric oddities need review.

Schema checks: **{schema_status}**  
Total issues: **{len(all_issues)}** ({verification.n_high} high, {verification.n_medium} medium)  
— schema: {len(verification.schema_issues)}, quality flags: {len(verification.quality_flags)}.

### 6.1 Schema / structural

| study_id | category | severity | detail |
| --- | --- | --- | --- |
{schema_table}

### 6.2 Quality flags (demographics / metrics)

| study_id | category | severity | detail |
| --- | --- | --- | --- |
{quality_table}

## 7. Pipeline notes

{notes_md}

## 8. Machine-readable artefacts

- `study_analysis_report.json` — full hypothesis payloads + catalog
- `study_explorer.html` — filterable charts for cohort, tasks, repeats, and own vs generic
- `figures/` — PNG copies of every plot embedded above
- processed participant / run tables under `processed/` (or repo `data/processed/`)
"""