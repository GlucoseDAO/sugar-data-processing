"""Student-facing plain-language summaries for notebook and report output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from sugar_data_processing.comparison.benchmarks import BenchmarkContext
from sugar_data_processing.statistics.catalog import HYPOTHESES
from sugar_data_processing.verification.report import VerificationReport

STATS_GLOSSARY: str = """
### Quick glossary (how to read the numbers)

| Term | Plain meaning |
| --- | --- |
| **MAE** | Mean absolute error (mg/dL). Average size of the prediction mistake. **Lower is better.** |
| **Person-level MAE** | One MAE per participant (mean of their round MAEs). Avoids counting the same person many times. |
| **p-value** | Chance of seeing a result at least this extreme if there were truly no effect. We call a result **significant** when p < 0.05. |
| **Significant** | Evidence is strong enough (at α = 0.05) to reject “no difference / no association”. Not the same as “large” or “clinically important”. |
| **Effect size** | How big the difference or association is (not only whether it is “significant”). |
| **t-test / Mann–Whitney** | Compare two independent groups. Mann–Whitney is used when data look non-normal. |
| **Pearson / Spearman** | Correlation: does one quantity rise/fall with another? Spearman is rank-based (more robust). |
| **Paired test** | Same person measured twice (own vs generic). Compares within-person differences. |
""".strip()


def explain_study_goal() -> str:
    return (
        "This analysis asks how well humans predict **next-hour glucose** from recent CGM "
        "context, and whether accuracy differs by diabetes status, CGM use, experience, "
        "and whether the data are the person’s own or a generic example."
    )


def explain_gathering(runs: pl.DataFrame, participants: pl.DataFrame) -> str:
    n_primary = int(participants.filter(pl.col("eligible_primary")).height)
    n_own = int(participants.filter(pl.col("eligible_own")).height)
    n_h5 = int(participants.filter(pl.col("eligible_h5")).height)
    mean_mae = participants.filter(pl.col("mae_primary").is_not_null())["mae_primary"].mean()
    mean_txt = f"{float(mean_mae):.2f}" if mean_mae is not None else "n/a"
    return (
        "### What just happened (data gathering)\n"
        "1. Loaded each **run** (one completed app session) from the statistics CSV.\n"
        "2. Expanded per-round prediction errors into tidy round rows.\n"
        "3. Collapsed rounds into **one row per person** with a person-level MAE.\n"
        "4. Marked who is eligible for primary analyses (≥6 generic rounds) and for the "
        "own-vs-generic paired test (≥6 generic and ≥6 own).\n\n"
        f"- Runs loaded: **{runs.height}**\n"
        f"- Unique participants: **{participants.height}**\n"
        f"- Eligible for primary analyses (H1–H4): **{n_primary}**\n"
        f"- Eligible for own-data analyses: **{n_own}**\n"
        f"- Eligible for own-vs-generic paired test (H5): **{n_h5}**\n"
        f"- Mean person MAE so far: **{mean_txt} mg/dL**\n\n"
        "Why person-level MAE? If one person plays many rounds, counting every round as an "
        "independent observation would overstate sample size. One summary score per person "
        "keeps the statistics honest."
    )


def explain_verification(verification: VerificationReport) -> str:
    status = "passed" if verification.schema_ok else "FAILED"
    lines = [
        "### What just happened (data verification)\n",
        "Before trusting hypothesis tests, we check two layers:\n",
        "1. **Schema / structural** — required columns, valid formats (A/B/C), parseable "
        "round metrics, non-empty IDs. High-severity failures mean the export is not trustworthy.\n",
        "2. **Quality flags** — implausible ages/durations, flag mismatches, MAE outliers, "
        "short sessions. These do not always stop the analysis, but they should be reviewed.\n\n",
        f"- Schema checks: **{status}**\n",
        f"- Schema issues: **{len(verification.schema_issues)}**\n",
        f"- Quality flags: **{len(verification.quality_flags)}** "
        f"({verification.n_high} high, {verification.n_medium} medium)\n",
    ]
    if verification.all_issues:
        lines.append("\nExamples (up to 5):\n")
        for issue in verification.all_issues[:5]:
            lines.append(
                f"- [{issue.severity}] `{issue.category}` — {issue.detail} "
                f"(study_id={issue.study_id})\n"
            )
    else:
        lines.append("\nNo verification issues were raised on this cohort.\n")
    return "".join(lines)


def _fmt_p(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4g}"
    return str(value)


def explain_hypothesis_result(key: str, result: dict[str, Any] | None) -> str:
    info = HYPOTHESES[key]
    header = (
        f"### {info['code']} — {info['title']}\n\n"
        f"**Research question:** {info['question']}\n\n"
        f"**Compared / predictor:** {info['groups_or_predictors']}\n\n"
        f"**Outcome:** {info['outcome']}\n\n"
        f"**Test plan:** {info['method']}\n\n"
    )
    if key == "h6":
        return header + (
            "**Status:** deferred. Computational baselines are not available yet, "
            "so this comparison is not run in the current pipeline.\n"
        )
    if result is None:
        return header + (
            "**Result:** not run (usually because too few eligible participants).\n"
        )

    significant = bool(result.get("significant"))
    sig_txt = (
        "Yes — there is statistically significant evidence of an effect at α = 0.05."
        if significant
        else "No — we do not have statistically significant evidence at α = 0.05."
    )
    body = (
        "**How to read this result**\n\n"
        f"- Test used: `{result.get('test_used')}`\n"
        f"- Sample size: {_n_label(result)}\n"
        f"- p-value: {_fmt_p(result.get('p_value'))}\n"
        f"- Significant?: {sig_txt}\n"
    )
    if isinstance(result.get("effect_size"), (int, float)):
        body += (
            f"- Effect size ({result.get('effect_size_name')}): "
            f"{result['effect_size']:.3f}\n"
        )
    interpretation = result.get("interpretation")
    if interpretation:
        body += f"\n**Plain-language takeaway:** {interpretation}\n"
    return header + body


def _n_label(result: dict[str, Any]) -> str:
    if "n" in result:
        return str(result["n"])
    if "n_a" in result:
        return f"{result['n_a']} vs {result['n_b']}"
    return "?"


def explain_all_hypotheses(payload: dict[str, Any]) -> str:
    parts = [
        "### What just happened (statistical tests)\n",
        "We ran the study-design hypothesis battery on **person-level MAE**.\n",
        "Primary tests compare groups; secondary tests look at experience correlations "
        "and own vs generic accuracy within the same person.\n\n",
        STATS_GLOSSARY,
        "\n\n",
    ]
    for key in ("h1", "h2", "h3", "h4", "h5", "h6"):
        parts.append(explain_hypothesis_result(key, payload.get(key)))
        parts.append("\n---\n\n")
    return "".join(parts)


def explain_benchmarks(benchmarks: BenchmarkContext) -> str:
    return (
        "### What just happened (literature comparison)\n"
        "Human accuracy is placed next to published 60-minute prediction bands "
        "(simple/ARIMA, deep learning, personalized). This is **context**, not a formal "
        "H6 model comparison on the same segments.\n\n"
        f"{benchmarks.narrative}\n\n"
        f"- People in simple/ARIMA band "
        f"({benchmarks.simple_baseline_mae_range[0]:.0f}–"
        f"{benchmarks.simple_baseline_mae_range[1]:.0f} mg/dL): "
        f"**{benchmarks.pct_inside_simple_baseline_band:.1f}%**\n"
        f"- People in deep-learning band "
        f"({benchmarks.deep_learning_mae_range[0]:.0f}–"
        f"{benchmarks.deep_learning_mae_range[1]:.0f} mg/dL): "
        f"**{benchmarks.pct_inside_deep_learning_band:.1f}%**\n"
        f"- People better than the simple-band floor "
        f"(<{benchmarks.simple_baseline_mae_range[0]:.0f} mg/dL): "
        f"**{benchmarks.pct_below_simple_baseline_low:.1f}%**\n"
    )


def explain_report_written(report_path: Path) -> str:
    reports_dir = Path(report_path).parent
    return (
        "### What just happened (output)\n"
        "Figures and a markdown report were written under the canonical "
        "`output/reports/` folder (same location for CLI and notebook).\n\n"
        f"- Report path: `{report_path}`\n"
        f"- PNG copies: `{reports_dir / 'figures'}`\n"
    )


def how_to_read_report() -> str:
    return (
        "## How to read this report\n\n"
        "Work top to bottom:\n"
        "1. **Cohort snapshot** — who is in the analysis and how accurate they were overall.\n"
        "2. **Population rules** — who counts for each test (eligibility thresholds).\n"
        "3. **Primary hypotheses** — group comparisons (diabetes status, CGM use).\n"
        "4. **Secondary hypotheses** — experience correlations and own vs generic data.\n"
        "5. **Literature context** — human MAE vs published model bands.\n"
        "6. **Verification** — data-quality checks you should not skip.\n\n"
        f"{STATS_GLOSSARY}\n"
    )
