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
from sugar_data_processing.output.plots import generate_all_figures
from sugar_data_processing.statistics.catalog import HYPOTHESES, hypothesis_blurb, hypothesis_heading
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
        participants.write_csv(processed_dir / "participants.csv")

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


def _fmt_result(result: dict[str, Any] | None) -> str:
    if result is None:
        return "_Not run / insufficient sample._"
    lines = [
        f"- **Test used:** `{result.get('test_used')}`",
        f"- **n:** {_n_from_result(result)}",
        f"- **Statistic:** {result.get('statistic'):.4g}"
        if isinstance(result.get("statistic"), (int, float))
        else f"- **Statistic:** {result.get('statistic')}",
        f"- **p-value:** {result.get('p_value'):.4g}"
        if isinstance(result.get("p_value"), (int, float))
        else f"- **p-value:** {result.get('p_value')}",
        f"- **Effect size ({result.get('effect_size_name')}):** "
        f"{result.get('effect_size'):.3f}"
        if isinstance(result.get("effect_size"), (int, float))
        else "",
        f"- **Significant (α={result.get('alpha')}):** "
        f"{'yes' if result.get('significant') else 'no'}",
        f"- **Interpretation:** {result.get('interpretation')}",
    ]
    return "\n".join(line for line in lines if line)


def _n_from_result(result: dict[str, Any]) -> str:
    if "n" in result:
        return str(result["n"])
    if "n_a" in result:
        return f"{result['n_a']} vs {result['n_b']}"
    return "?"


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
    blurb = hypothesis_blurb(key)
    heading = hypothesis_heading(key)
    figures = "\n\n".join(figure_blocks)
    return f"""### {heading}

{blurb}

**Results**

{_fmt_result(result)}

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
    h6 = HYPOTHESES["h6"]

    catalog_rows = "\n".join(
        f"| {info['code']} | {info['title']} | {info['question']} |"
        for info in HYPOTHESES.values()
    )

    return f"""# Sugar Sugar Study Analysis Report

Generated: **{stamp}**  
Source: `{source_csv}`

This report follows **Section 7 (Statistical Analysis Plan)** of
*Human Prediction of Next-Hour Glucose from Prior CGM Context*.

**MAE** = mean absolute error of next-hour glucose predictions (mg/dL).  
**Person-level MAE** = mean of a participant's round MAEs (one score per person).

## Hypothesis key

| Code | Title | Question in plain language |
| --- | --- | --- |
{catalog_rows}

## 1. Cohort snapshot

How many sessions and people entered the analysis, and what is the overall
accuracy distribution?

| Metric | Value |
| --- | ---: |
| Raw runs (rows) | {runs.height} |
| Unique participants (`study_id`) | {participants.height} |
| Eligible for primary analyses (≥6 generic segments) | {n_primary} |
| Eligible for own-vs-generic paired test (≥6 generic **and** ≥6 own) | {n_h5} |
| Mean person MAE (mg/dL) | {benchmarks.human_mean_mae:.2f} |
| Median person MAE (mg/dL) | {benchmarks.human_median_mae:.2f} |
| SD person MAE (mg/dL) | {benchmarks.human_sd_mae:.2f} |

{img("mae_distribution", "Distribution of per-person MAE (mg/dL)")}

## 2. Analysis population rules (§7.2)

- **Primary analyses (diabetes status, CGM use, duration correlations):**
  participants with ≥6 analyzable **generic** segments.
- **Own-data analyses:** participants with ≥6 **own-data** segments.
- **Own-vs-generic paired analysis:** participants meeting both thresholds.
- **Person-level MAE:** mean of round MAEs so repeated rounds from the same
  participant do not inflate degrees of freedom.

## 3. Primary hypotheses

These compare independent groups on person-level MAE.

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

These test experience correlations and within-person own-vs-generic accuracy.

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

### {hypothesis_heading("h6")}

{hypothesis_blurb("h6")}

**Results**

_{h6['method']}_

## 5. Literature / GlucoBench context (§7.5)

How does the human cohort's MAE sit relative to published 60-minute model bands?

{benchmarks.narrative}

| Band | Range (mg/dL) | % of humans inside |
| --- | --- | ---: |
| Simple / ARIMA | {benchmarks.simple_baseline_mae_range[0]:.0f}–{benchmarks.simple_baseline_mae_range[1]:.0f} | {benchmarks.pct_inside_simple_baseline_band:.1f}% |
| Deep learning | {benchmarks.deep_learning_mae_range[0]:.0f}–{benchmarks.deep_learning_mae_range[1]:.0f} | {benchmarks.pct_inside_deep_learning_band:.1f}% |
| Personalized | {benchmarks.personalized_mae_range[0]:.0f}–{benchmarks.personalized_mae_range[1]:.0f} | {benchmarks.pct_inside_personalized_band:.1f}% |
| Below simple-band low | < {benchmarks.simple_baseline_mae_range[0]:.0f} | {benchmarks.pct_below_simple_baseline_low:.1f}% |

{img("benchmark_bands", "Human MAE density vs published simple and deep-learning bands")}

## 6. Data verification

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

- `study_analysis_report.json` — hypothesis payloads + catalog
- `figures/` — PNG copies of every plot embedded above
- processed participant / run tables under `processed/` (or repo `data/processed/`)
"""
