"""Assemble a human-readable markdown report with embedded figures."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
from eliot import start_action

from sugar_data_processing.comparison.anomalies import Anomaly
from sugar_data_processing.comparison.benchmarks import BenchmarkContext
from sugar_data_processing.output.plots import generate_all_figures
from sugar_data_processing.statistics.hypotheses import HypothesisSuite


def write_report(
    *,
    runs: pl.DataFrame,
    participants: pl.DataFrame,
    suite: HypothesisSuite,
    benchmarks: BenchmarkContext,
    anomalies: list[Anomaly],
    output_dir: Path,
    source_csv: Path,
) -> Path:
    """Write markdown + JSON artefacts under ``output_dir``."""
    with start_action(action_type="output.write_report") as action:
        output_dir = Path(output_dir)
        figures_dir = output_dir / "figures"
        reports_dir = output_dir / "reports"
        processed_dir = output_dir.parent / "data" / "processed"
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
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        md_path = reports_dir / "study_analysis_report.md"
        md = _render_markdown(
            stamp=stamp,
            source_csv=source_csv,
            runs=runs,
            participants=participants,
            suite=suite,
            benchmarks=benchmarks,
            anomalies=anomalies,
            figure_paths=figure_paths,
            reports_dir=reports_dir,
        )
        md_path.write_text(md, encoding="utf-8")

        payload: dict[str, Any] = {
            "generated_at": stamp,
            "source_csv": str(source_csv),
            "n_runs": runs.height,
            "n_participants": participants.height,
            "hypotheses": suite.to_dict(),
            "benchmarks": benchmarks.to_dict(),
            "anomalies": [a.to_dict() for a in anomalies],
            "figures": {k: str(v) for k, v in figure_paths.items()},
        }
        (reports_dir / "study_analysis_report.json").write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        action.log(message_type="info", report=str(md_path), n_figures=len(figure_paths))
        return md_path


def _rel(path: Path, base: Path) -> str:
    """Return a portable relative path from ``base`` (usually the reports dir)."""
    return Path(os.path.relpath(path.resolve(), start=base.resolve())).as_posix()


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


def _render_markdown(
    *,
    stamp: str,
    source_csv: Path,
    runs: pl.DataFrame,
    participants: pl.DataFrame,
    suite: HypothesisSuite,
    benchmarks: BenchmarkContext,
    anomalies: list[Anomaly],
    figure_paths: dict[str, Path],
    reports_dir: Path,
) -> str:
    hyp = suite.to_dict()
    n_primary = int(participants.filter(pl.col("eligible_primary")).height)
    n_h5 = int(participants.filter(pl.col("eligible_h5")).height)
    high = [a for a in anomalies if a.severity == "high"]
    medium = [a for a in anomalies if a.severity == "medium"]

    def img(key: str, caption: str) -> str:
        path = figure_paths.get(key)
        if path is None:
            return f"_{caption}: figure missing._"
        rel = _rel(path, reports_dir)
        return f"![{caption}]({rel})\n\n*{caption}*"

    notes_md = "\n".join(f"- {n}" for n in suite.notes) if suite.notes else "- None"

    anomaly_lines = []
    for a in anomalies[:80]:
        anomaly_lines.append(
            f"| `{a.study_id[:12]}` | {a.category} | {a.severity} | {a.detail} |"
        )
    if len(anomalies) > 80:
        anomaly_lines.append(f"| … | … | … | ({len(anomalies) - 80} more omitted) |")
    anomaly_table = "\n".join(anomaly_lines) if anomaly_lines else "| — | — | — | none |"

    return f"""# Sugar Sugar Study Analysis Report

Generated: **{stamp}**  
Source: `{source_csv}`

This report follows **Section 7 (Statistical Analysis Plan)** of
*Human Prediction of Next-Hour Glucose from Prior CGM Context*.

## 1. Cohort snapshot

| Metric | Value |
| --- | ---: |
| Raw runs (rows) | {runs.height} |
| Unique participants (`study_id`) | {participants.height} |
| Eligible primary (≥6 generic segments) | {n_primary} |
| Eligible H5 (≥6 generic **and** ≥6 own) | {n_h5} |
| Mean person MAE (mg/dL) | {benchmarks.human_mean_mae:.2f} |
| Median person MAE (mg/dL) | {benchmarks.human_median_mae:.2f} |
| SD person MAE (mg/dL) | {benchmarks.human_sd_mae:.2f} |

{img("mae_distribution", "Distribution of per-person MAE")}

## 2. Analysis population rules (§7.2)

- **Primary analyses (H1–H4):** participants with ≥6 analyzable **generic** segments.
- **Own-data analyses:** participants with ≥6 **own-data** segments.
- **Person-level MAE:** mean of round MAEs (one summary score per person) so
  repeated measures from the same participant do not inflate degrees of freedom.

## 3. Primary hypotheses

### H1 — PwD vs non-PwD (§7.3)

Shapiro–Wilk normality → independent t-test, else Mann–Whitney U. α = 0.05.

{_fmt_result(hyp["h1"])}

{img("mae_by_diabetes", "H1: MAE by diabetes status")}

### H2 — CGM users vs non-CGM (§7.3)

Same testing path as H1.

{_fmt_result(hyp["h2"])}

{img("mae_by_cgm", "H2: MAE by CGM use")}

## 4. Secondary hypotheses

### H3 — Diabetes duration vs MAE (§7.4)

Pearson if normal/linear, otherwise Spearman; plus linear vs log exploratory fits.

{_fmt_result(hyp["h3"])}

{img("diabetes_duration_scatter", "H3: diabetes duration scatter")}

### H4 — CGM experience vs MAE (§7.4)

Same approach as H3.

{_fmt_result(hyp["h4"])}

{img("cgm_duration_scatter", "H4: CGM experience scatter")}

{img("duration_bins", "Exploratory duration-bin MAE")}

### H5 — Own vs generic data (paired) (§7.4)

Difference scores → Shapiro–Wilk → paired t-test or Wilcoxon signed-rank.
Positive (generic − own) means better accuracy on own data.

{_fmt_result(hyp["h5"])}

{img("own_vs_generic", "H5: own vs generic MAE")}

### H6 — Human vs baseline models

{hyp["h6"]["note"]}

## 5. Literature / GlucoBench context (§7.5)

{benchmarks.narrative}

| Band | Range (mg/dL) | % of humans inside |
| --- | --- | ---: |
| Simple / ARIMA | {benchmarks.simple_baseline_mae_range[0]:.0f}–{benchmarks.simple_baseline_mae_range[1]:.0f} | {benchmarks.pct_inside_simple_baseline_band:.1f}% |
| Deep learning | {benchmarks.deep_learning_mae_range[0]:.0f}–{benchmarks.deep_learning_mae_range[1]:.0f} | {benchmarks.pct_inside_deep_learning_band:.1f}% |
| Personalized | {benchmarks.personalized_mae_range[0]:.0f}–{benchmarks.personalized_mae_range[1]:.0f} | {benchmarks.pct_inside_personalized_band:.1f}% |
| Below simple-band low | < {benchmarks.simple_baseline_mae_range[0]:.0f} | {benchmarks.pct_below_simple_baseline_low:.1f}% |

{img("benchmark_bands", "Human MAE vs published model bands")}

## 6. Anomalies & data-quality flags

Found **{len(anomalies)}** flags ({len(high)} high, {len(medium)} medium).

| study_id | category | severity | detail |
| --- | --- | --- | --- |
{anomaly_table}

## 7. Pipeline notes

{notes_md}

## 8. Machine-readable artefacts

- `study_analysis_report.json` — full hypothesis payloads
- `../figures/` — PNG graphics embedded above
- processed participant / run tables written beside the report under `data/processed/`
"""
