"""Assemble a human-readable markdown report with embedded figures."""

from __future__ import annotations

import base64
import json
import shutil
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

import polars as pl
from eliot import start_action

from sugar_data_processing.ai.export import export_prediction_sequences
from sugar_data_processing.ai.run import ScoringResult, run_post_factum_scoring
from sugar_data_processing.comparison.benchmarks import BenchmarkContext, benchmark_context
from sugar_data_processing.config import (
    COHORT_LABELS,
    EXPLORER_HTML,
    MILESTONE_HTML,
    REPORT_JSON,
    REPORT_MD,
    STALE_REPORT_FILES,
)
from sugar_data_processing.gathering.participants import opposite_trait_summary
from sugar_data_processing.output.explorer import markdown_fragment_to_html, write_explorer_html
from sugar_data_processing.output.narration import (
    explain_edition_and_ai_path,
    explain_hypothesis_result,
    explain_opposite_trait,
    how_to_read_report,
)
from sugar_data_processing.output.plots import (
    generate_ai_comparison_figures,
    generate_all_figures,
    task_comparison_summary,
)
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

        ai_dir = processed_dir / "ai"
        ai_paths = export_prediction_sequences(runs, ai_dir)

        scoring = run_post_factum_scoring(
            runs,
            participants,
            source_csv=source_csv,
            ai_dir=ai_dir,
        )
        ai_benchmarks = benchmark_context(
            scoring.ai_participants.filter(pl.col("eligible_primary"))
            if "eligible_primary" in scoring.ai_participants.columns
            and scoring.ai_participants.filter(pl.col("eligible_primary")).height > 0
            else scoring.ai_participants,
            mae_col="mae_primary",
        )

        human_figures = generate_all_figures(participants, suite, benchmarks, figures_dir)
        ai_figures = generate_ai_comparison_figures(
            participants,
            scoring.ai_participants,
            figures_dir,
        )
        ai_comparison = task_comparison_summary(participants, scoring.ai_participants)

        if report_figures_dir.exists():
            shutil.rmtree(report_figures_dir)
        report_figures_dir.mkdir(parents=True, exist_ok=True)
        report_figure_paths: dict[str, Path] = {}
        for key, src in human_figures.items():
            dest = report_figures_dir / src.name
            shutil.copy2(src, dest)
            report_figure_paths[key] = dest
        ai_report_figures: dict[str, Path] = {}
        for key, src in ai_figures.items():
            dest = report_figures_dir / src.name
            shutil.copy2(src, dest)
            ai_report_figures[key] = dest

        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        opposite = opposite_trait_summary(participants)
        md_path = reports_dir / REPORT_MD
        md = _render_markdown(
            stamp=stamp,
            source_csv=source_csv,
            runs=runs,
            participants=participants,
            suite=suite,
            benchmarks=benchmarks,
            verification=verification,
            figure_paths=report_figure_paths,
            opposite=opposite,
            ai_dir=ai_dir,
            scoring=scoring,
            ai_figure_paths=ai_report_figures,
            ai_task_comparison=ai_comparison,
        )
        md_path.write_text(md, encoding="utf-8")

        explorer_path = write_explorer_html(
            participants=participants,
            runs=runs,
            output_path=reports_dir / EXPLORER_HTML,
            source_csv=source_csv,
            suite=suite,
            benchmarks=benchmarks,
            verification=verification,
            edition="merged",
            comparison=scoring.comparison.to_dict(),
            sequences_dir=ai_dir,
            ai_participants=scoring.ai_participants,
            ai_suite=scoring.ai_suite,
            traces=scoring.traces,
            primary_model=scoring.primary_model,
            h6=scoring.h6,
        )
        write_explorer_html(
            participants=participants,
            runs=runs,
            output_path=reports_dir / MILESTONE_HTML,
            source_csv=source_csv,
            suite=suite,
            benchmarks=benchmarks,
            verification=verification,
            edition="milestone",
            comparison=None,
            sequences_dir=None,
            ai_participants=None,
            ai_suite=None,
            traces=[],
            primary_model="",
            h6=None,
            milestone=True,
        )
        _remove_stale_report_files(reports_dir)

        payload: dict[str, Any] = {
            "generated_at": stamp,
            "source_csv": str(source_csv),
            "n_runs": runs.height,
            "n_participants": participants.height,
            "verification": verification.to_dict(),
            "hypotheses": suite.to_dict(),
            "hypothesis_catalog": HYPOTHESES,
            "benchmarks": benchmarks.to_dict(),
            "edition": "merged",
            "human": {"figures": {k: str(v) for k, v in report_figure_paths.items()}},
            "ai": {
                "primary_model": scoring.primary_model,
                "n_windows": len(scoring.windows),
                "comparison": scoring.comparison.to_dict(),
                "hypotheses": scoring.ai_suite.to_dict(),
                "benchmarks": ai_benchmarks.to_dict(),
                "h6": scoring.h6,
                "task_comparison": ai_comparison,
                "figures": {k: str(v) for k, v in ai_report_figures.items()},
            },
            "explorer": str(explorer_path),
            "opposite_trait": opposite,
            "ai_export": {k: str(v) for k, v in ai_paths.items()},
        }
        json_text = json.dumps(payload, indent=2, default=str)
        (reports_dir / REPORT_JSON).write_text(json_text, encoding="utf-8")
        action.log(
            message_type="info",
            report=str(md_path),
            n_figures=len(human_figures) + len(ai_figures),
            n_windows=len(scoring.windows),
        )
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
        '<figure style="margin:0;">'
        f'<img src="data:image/png;base64,{encoded}" alt="{escape(caption)}" '
        f'style="width:100%;max-width:1200px;height:auto;" />'
        f'<figcaption style="color:#475569;font-size:0.92rem;margin-top:8px;">'
        f'{escape(caption)} · <a href="{rel}">PNG file</a></figcaption>'
        "</figure>"
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


def _remove_stale_report_files(reports_dir: Path) -> None:
    """Drop the old split human/AI / study_* copies. One report remains."""
    for name in STALE_REPORT_FILES:
        stale = reports_dir / name
        if stale.exists():
            stale.unlink()


def _pair(text_md: str, figure_html: str) -> str:
    """Put analysis copy on the left and its figure on the right."""
    text_html = markdown_fragment_to_html(text_md)
    return (
        '<div style="display:flex;flex-wrap:wrap;gap:28px;align-items:flex-start;'
        'margin:20px 0 36px;">'
        f'<div style="flex:1 1 340px;min-width:280px;">{text_html}</div>'
        f'<div style="flex:1.3 1 420px;min-width:340px;">{figure_html}</div>'
        "</div>\n"
    )


def _fmt_mae(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}"


def _render_ai_half(
    *,
    scoring: ScoringResult | None,
    img: Any,
    comparison: dict[str, Any] | None,
) -> str:
    if scoring is None:
        return "## 7b. AI results\n\nNo AI scoring ran in this write.\n"
    n_windows = len(scoring.windows)
    models = ", ".join(f"`{m}`" for m in scoring.comparison.models) or "none"
    tasks = (comparison or {}).get("tasks") or {}
    cluster = (comparison or {}).get("cluster") or {}
    task_rows = "\n".join(
        "| {label} | {n} | {human} | {ai} | {human_better} | {ai_better} |".format(
            label=label,
            n=int((tasks.get(label) or {}).get("n") or 0),
            human=_fmt_mae((tasks.get(label) or {}).get("human_mean")),
            ai=_fmt_mae((tasks.get(label) or {}).get("ai_mean")),
            human_better=int((tasks.get(label) or {}).get("human_better") or 0),
            ai_better=int((tasks.get(label) or {}).get("ai_better") or 0),
        )
        for label in ("Generic (A)", "Own (B)", "Mixed (C)")
    )
    return f"""
## 7b. AI results (same person, same task)

Age, diabetes type, and CGM experience do not move the model, so this half
does **not** repeat H1–H4. The question is whether a person is better on
generic or own data while the model stays flat — or whether own-data humans
beat the model, which then beats generic.

Primary model: **`{scoring.primary_model}`**.  
Reconstructed 3-hour windows: **{n_windows}**. Models: {models}.

A point is one person. Generic / Own / Mixed are formats A, B, and C.
**Mixed is format C**, not a blend of A and B. A person only appears on a
task if they have both a human MAE and an AI MAE for that task. The cluster
only keeps people who have Generic (A) and Own (B) on both sides.

| Task | n (same person both sides) | Human mean | AI mean | Human better | AI better |
| --- | ---: | ---: | ---: | ---: | ---: |
{task_rows}

Cluster (Generic vs Own, same users): **{int(cluster.get("n") or 0)}** people.
Human better on own: **{int(cluster.get("human_better_own") or 0)}**.
Human better on generic: **{int(cluster.get("human_better_generic") or 0)}**.
AI better on own: **{int(cluster.get("ai_better_own") or 0)}**.
AI better on generic: **{int(cluster.get("ai_better_generic") or 0)}**.
Mean |generic − own| gap: human **{_fmt_mae(cluster.get("human_mean_abs_gap"))}**,
AI **{_fmt_mae(cluster.get("ai_mean_abs_gap"))}** (smaller = more stable).

Models were fed only the 24 visible CGM points from the game (left-padded to 128).
Own-data windows come from the files players uploaded (saved under sugar-sugar `data/input/users`).

{_pair(
    "### Per-task MAE\\n\\n"
    "Generic (A), Own (B), Mixed (C). Blue is the human, purple is the model. "
    "Each tick is the same person on that task.",
    img("task_mae", "Same person, same task: human vs AI MAE", ai=True),
)}

{_pair(
    "### Same-user cluster\\n\\n"
    "X = Generic (A), Y = Own (B). Grey line joins that person's human point "
    "to their AI point. Below the diagonal = better on own data.",
    img("same_user_cluster", "Same user on generic vs own: human and AI", ai=True),
)}
"""


def _hypothesis_section(
    key: str,
    result: dict[str, Any] | None,
    figure_blocks: list[str],
) -> str:
    return _pair(explain_hypothesis_result(key, result), "\n".join(figure_blocks))


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
    opposite: dict[str, Any],
    ai_dir: Path,
    scoring: ScoringResult | None = None,
    ai_figure_paths: dict[str, Path] | None = None,
    ai_task_comparison: dict[str, Any] | None = None,
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

    def img(key: str, caption: str, *, ai: bool = False) -> str:
        store = ai_figure_paths if ai else figure_paths
        path = None if store is None else store.get(key)
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
    edition_note = explain_edition_and_ai_path()
    opposite_md = explain_opposite_trait(opposite)

    ai_block = _render_ai_half(
        scoring=scoring,
        img=img,
        comparison=ai_task_comparison,
    )

    return f"""# Sugar Sugar Study Analysis Report

Generated: **{stamp}**  
Source: `{source_csv}`  
Edition: **merged** (human H1–H5, then same-person human vs AI by task)

This report follows **Section 7 (Statistical Analysis Plan)** of
*Human Prediction of Next-Hour Glucose from Prior CGM Context*.

It is written so a student can follow the pipeline without reading the source code:
what was measured, who was included, what each test asked, and what the numbers mean.

{edition_note}

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
| Challenge the unknown | {opposite.get("n_challenge_unknown", 0)} | Opted into the opposite-corpus mix |
| Played opposite trait | {opposite.get("n_played_opposite_trait", 0)} | At least one round on the other diabetes class |
| Saved runs (all people) | {n_runs_total} | Sessions after counting replays |
| Played every variant (A+B+C) | {n_all_formats} | People who tried generic, own, and mixed |
| Eligible for primary analyses | {n_primary} | ≥6 generic segments (used for H1–H4) |
| Eligible for own-vs-generic test | {n_h5} | ≥6 generic **and** ≥6 own (used for H5) |
| Mean person MAE (mg/dL) | {benchmarks.human_mean_mae:.2f} | Average accuracy (lower is better) |
| Median person MAE (mg/dL) | {benchmarks.human_median_mae:.2f} | Typical person (robust to outliers) |
| SD person MAE (mg/dL) | {benchmarks.human_sd_mae:.2f} | Spread of accuracy across people |

{cohort_rows}

{_pair(
    "### Who is in the cohort\n\n"
    "Each slice is a unique person in one diabetes × CGM bucket. "
    "This is head-count, not accuracy.",
    img("cohort_pie", "Unique people in each diabetes × CGM category"),
)}

{_pair(
    "### How accurate people were\n\n"
    "Each tick is one person's MAE (mg/dL). Lower is better. "
    "Look at the centre, the spread, and any far-right outliers.",
    img("mae_distribution", "Distribution of per-person MAE (mg/dL)"),
)}

{_pair(
    "### Accuracy on each task\n\n"
    "A is generic traces, B is own data, C is mixed. "
    "One point per person who played that format.",
    img("mae_by_format", "Person MAE on each task: A generic, B own data, C mixed"),
)}

{_pair(
    "### First-timers vs people who came back\n\n"
    f"{n_repeat} of {participants.height} people saved more than one run. "
    "The swarm is person MAE, not a count bar.",
    img("players_vs_repeats", "How many people played once vs came back, and their MAE"),
)}

{_pair(
    "### People who tried every variant\n\n"
    "Only players who saved A, B, and C. Below the diagonal means "
    "better (lower MAE) on their own data than on generic traces.",
    img("all_formats_own_vs_generic", "People who played every variant: own-data MAE vs generic MAE"),
)}

{_pair(
    "### Everyone in own vs generic space\n\n"
    "Colour is the diabetes × CGM cohort. A dark ring means they opted "
    "into Challenge the unknown. Missing own-data MAE falls back to person MAE.",
    img("people_clusters", "People as points in own-vs-generic space (colour = cohort, ring = Challenge the unknown)"),
)}

Interactive filters live in [`{EXPLORER_HTML}`]({EXPLORER_HTML}). The milestone submission page (human participants only) is [`{MILESTONE_HTML}`]({MILESTONE_HTML}).

## 2. Analysis population rules (§7.2)

**Why this matters:** statistical tests only include people with enough completed
segments. Otherwise short incomplete sessions would dominate the results.

- **Primary analyses (H1–H4):** ≥6 analyzable **generic** segments for the
  overall / generic layers. The own-data layer uses ≥6 own segments (relaxed
  if that threshold is not yet met).
- **Own-data analyses:** ≥6 **own-data** segments.
- **Own-vs-generic paired analysis (H5):** both thresholds.
- **Reporting order for H1–H4:** results per diabetes × CGM category first,
  then that category split into generic data vs own data.
- **Person-level MAE:** mean of round MAEs so one busy participant cannot inflate
  the sample size by playing many rounds.

## 3. Primary hypotheses

**What these ask:** do two groups of people differ in prediction accuracy?
Each result starts with the four diabetes × CGM categories, then splits
generic data vs own data.

{_hypothesis_section(
    "h1",
    hyp["h1"],
    [img("mae_by_diabetes", "Person MAE by diabetes status, each group split into generic vs own data")],
)}

{_hypothesis_section(
    "h2",
    hyp["h2"],
    [img("mae_by_cgm", "Person MAE by CGM use, each group split into generic vs own data")],
)}

## 4. Secondary hypotheses

**What these ask:** does longer experience help, and is accuracy better on own data
than on generic example data? H3 and H4 use **months** so short experience is readable.
Both also split generic vs own MAE.

{_hypothesis_section(
    "h3",
    hyp["h3"],
    [
        img(
            "diabetes_duration_scatter",
            "Diabetes duration (months) vs MAE on generic and own data among PwD",
        )
    ],
)}

{_hypothesis_section(
    "h4",
    hyp["h4"],
    [
        img(
            "cgm_duration_scatter",
            "CGM experience (months) vs MAE on generic and own data among CGM users",
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

{explain_hypothesis_result("h6", (scoring.h6 if scoring is not None else hyp.get("h6")))}

## 4b. Challenge the unknown / opposite trait

{_pair(
    opposite_md,
    img("opposite_trait", "Same-trait vs opposite-trait person MAE (points, not bins)"),
)}

## 5. Literature / GlucoBench context (§7.5)

**Goal of this section:** place human MAE next to published 60-minute model bands.
This is contextual comparison, not the deferred formal H6 baseline test.

{_pair(
    (
        f"{benchmarks.narrative}\n\n"
        "| Band | Range (mg/dL) | % of humans inside | How to read it |\n"
        "| --- | --- | ---: | --- |\n"
        f"| Simple / ARIMA | {benchmarks.simple_baseline_mae_range[0]:.0f}–{benchmarks.simple_baseline_mae_range[1]:.0f} | {benchmarks.pct_inside_simple_baseline_band:.1f}% | Typical simple forecasting models |\n"
        f"| Deep learning | {benchmarks.deep_learning_mae_range[0]:.0f}–{benchmarks.deep_learning_mae_range[1]:.0f} | {benchmarks.pct_inside_deep_learning_band:.1f}% | Typical deep-learning reports |\n"
        f"| Personalized | {benchmarks.personalized_mae_range[0]:.0f}–{benchmarks.personalized_mae_range[1]:.0f} | {benchmarks.pct_inside_personalized_band:.1f}% | Personalized-model band |\n"
        f"| Below simple-band low | < {benchmarks.simple_baseline_mae_range[0]:.0f} | {benchmarks.pct_below_simple_baseline_low:.1f}% | Better than the simple-band floor |\n"
    ),
    img("benchmark_bands", "Human MAE density vs published simple and deep-learning bands"),
)}

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

{ai_block}

## 8. Machine-readable artefacts

- `{REPORT_JSON}` — human + AI payloads (hypotheses, H6, figures)
- `{EXPLORER_HTML}` — interactive report: Overview, Human, AI, People
- `{MILESTONE_HTML}` — human participants only (Overview + Human, no AI / People tabs)
- `figures/` — human PNGs plus `ai_task_mae.png` / `ai_same_user_cluster.png`
- processed participant / run tables under `processed/` (or repo `data/processed/`)
- AI sequences and `ml_ready.csv` under `{ai_dir}`
"""