"""Reproduce the production interim audit without model downloads or contact data.

Run: PYTHONPATH=src .venv/bin/python scripts/assess_production.py
Input must contain participant_key, linked on the server before emails were scrubbed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from eliot import start_action
from scipy import stats

from sugar_data_processing.gathering.load import load_prediction_statistics
from sugar_data_processing.gathering.participants import _latest_runs, build_participant_table
from sugar_data_processing.gathering.rounds import build_round_table
from sugar_data_processing.output.briefing import write_livia_briefing
from sugar_data_processing.statistics.hypotheses import run_all_hypotheses
from sugar_data_processing.statistics.power import (
    enrollment_for_eligible,
    required_correlation_n,
    required_independent_n,
    required_paired_n,
)
from sugar_data_processing.verification.report import verify_dataset

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261024


def mean_interval(a: np.ndarray, b: np.ndarray | None = None) -> list[float]:
    """Percentile bootstrap interval for a mean / independent mean difference."""
    rng = np.random.default_rng(SEED)
    draws = a[rng.integers(len(a), size=(20_000, len(a)))].mean(axis=1)
    if b is not None:
        draws -= b[rng.integers(len(b), size=(20_000, len(b)))].mean(axis=1)
    return np.quantile(draws, [0.025, 0.975]).tolist()


def group_summary(frame: pl.DataFrame, group: str, metric: str = "mae_generic") -> dict[str, Any]:
    frame = frame.sort("study_id")
    a = frame.filter(pl.col(group))[metric].to_numpy().astype(float)
    b = frame.filter(~pl.col(group))[metric].to_numpy().astype(float)
    pooled_sd = np.sqrt(
        ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
        / (len(a) + len(b) - 2)
    )
    mw = stats.mannwhitneyu(a, b, alternative="two-sided")
    welch = stats.ttest_ind(a, b, equal_var=False)
    return {
        "n_a": len(a), "n_b": len(b), "mean_a": float(a.mean()),
        "mean_b": float(b.mean()), "mean_difference": float(a.mean() - b.mean()),
        "mean_difference_ci95": mean_interval(a, b),
        "sd_a": float(a.std(ddof=1)), "sd_b": float(b.std(ddof=1)),
        "pooled_sd": float(pooled_sd), "mann_whitney_p": float(mw.pvalue),
        "welch_p": float(welch.pvalue),
        "probability_a_has_higher_error": float(mw.statistic / (len(a) * len(b))),
    }


def audit_flags(runs: pl.DataFrame, people: pl.DataFrame) -> pl.DataFrame:
    conflicts = runs.group_by("study_id").agg(
        (pl.col("diabetic").n_unique() > 1).alias("diabetes_conflict"),
        (pl.col("uses_cgm").n_unique() > 1).alias("cgm_conflict"),
    )
    return people.join(conflicts, on="study_id").with_columns(
        (
            pl.col("diabetes_conflict") | pl.col("cgm_conflict")
            | ~pl.col("age").is_between(18, 120)
            | (pl.col("cgm_duration_years") > 30).fill_null(False)
            | (pl.col("diabetes_duration") > 80).fill_null(False)
            | (pl.col("cgm_duration_years") > pl.col("age")).fill_null(False)
            | (pl.col("diabetes_duration") > pl.col("age")).fill_null(False)
        ).alias("audit_exclude")
    )


def exploratory_tests(rounds: pl.DataFrame, people: pl.DataFrame) -> dict[str, Any]:
    generic = rounds.filter(pl.col("source") == "generic").join(
        people.select("study_id", "diabetic", "uses_cgm", "eligible_primary"), on="study_id"
    ).filter(pl.col("eligible_primary"))
    by_source = generic.group_by("study_id", "diabetic", "uses_cgm", "data_class").agg(
        pl.col("mae").mean().alias("mae"), pl.len().alias("n_rounds")
    )
    same_corpus = by_source.filter(
        (pl.col("data_class") == "diabetic") & (pl.col("n_rounds") >= 6)
    )
    healthy = by_source.filter(
        (pl.col("data_class") == "nondiabetic")
        & (pl.col("n_rounds") >= 6) & ~pl.col("diabetic")
    )
    results: dict[str, Any] = {
        "diabetes_groups_on_diabetic_traces": group_summary(same_corpus, "diabetic", "mae"),
        "cgm_among_nondiabetic_on_nondiabetic_traces": group_summary(healthy, "uses_cgm", "mae"),
    }
    paired = by_source.filter(pl.col("n_rounds") >= 3).pivot(
        on="data_class", index="study_id", values="mae"
    ).drop_nulls(subset=["diabetic", "nondiabetic"]).sort("study_id")
    diff = (paired["diabetic"] - paired["nondiabetic"]).to_numpy()
    result = stats.wilcoxon(diff, alternative="two-sided")
    results["within_person_trace_difficulty"] = {
        "n": paired.height, "min_rounds_per_trace_class": 3,
        "mean_diabetic_trace_mae": float(paired["diabetic"].mean()),
        "mean_nondiabetic_trace_mae": float(paired["nondiabetic"].mean()),
        "mean_difference": float(diff.mean()), "mean_difference_ci95": mean_interval(diff),
        "n_harder_on_diabetic_traces": int((diff > 0).sum()),
        "wilcoxon_p": float(result.pvalue),
    }
    # Holm sensitivity across these three data-prompted exploratory comparisons.
    ordered = sorted(
        [(key, value.get("mann_whitney_p", value.get("wilcoxon_p"))) for key, value in results.items()],
        key=lambda item: item[1],
    )
    running = 0.0
    for index, (key, p_value) in enumerate(ordered):
        running = max(running, min(1.0, p_value * (len(ordered) - index)))
        results[key]["holm_p_within_three_exploratory_tests"] = running
    results["status"] = "post_hoc_exploratory; not independent confirmation"
    return results


def assessment() -> dict[str, Any]:
    source = ROOT / "data/raw/prediction_statistics.csv"
    runs = load_prediction_statistics(source)
    if "participant_key" not in runs.columns:
        raise ValueError("Production assessment requires pseudonymous participant linkage")
    raw_runs = load_prediction_statistics(source, use_participant_keys=False)
    raw_people = build_participant_table(raw_runs)
    people = audit_flags(runs, build_participant_table(runs))
    clean = people.filter(~pl.col("audit_exclude"))
    primary = people.filter(pl.col("eligible_primary"))
    quality_primary = clean.filter(pl.col("eligible_primary"))
    rounds = build_round_table(_latest_runs(runs)).filter(
        pl.col("mae").is_finite() & (pl.col("mae") >= 0)
    )
    suite = run_all_hypotheses(people)
    clean_suite = run_all_hypotheses(clean)
    verification = verify_dataset(runs, people)
    result: dict[str, Any] = {
        "assessment_date": "2026-10-24",
        "source": "ovh:/home/web/sugar-sugar/data/input/prediction_statistics.csv",
        "source_sha256_after_scrubbing_and_linkage": hashlib.sha256(source.read_bytes()).hexdigest(),
        "first_record": str(runs["timestamp"].min()), "latest_record": str(runs["timestamp"].max()),
        "counts": {
            "runs": runs.height, "study_ids": raw_people.height,
            "linked_identities": people.height, "primary_eligible": primary.height,
            "quality_sensitivity_eligible": quality_primary.height,
            "own_eligible": people.filter(pl.col("eligible_own")).height,
            "paired_eligible": people.filter(pl.col("eligible_h5")).height,
            "quality_sensitivity_own": clean.filter(pl.col("eligible_own")).height,
            "quality_sensitivity_paired": clean.filter(pl.col("eligible_h5")).height,
            "anonymous_study_ids": raw_runs.filter(pl.col("has_email") == "false")["study_id"].n_unique(),
        },
        "identity_caveat": "Normalized-email linkage; no-email IDs remain distinct and may still repeat humans.",
        "selection_rule": "Latest saved run per linked identity and format; finite nonnegative MAE rounds only.",
        "quality_sensitivity_rule": "Exclude conflicting diabetes/CGM status, age outside 18–120, CGM >30y, diabetes >80y, durations above age. Outcome-blind sensitivity, not protocol-prespecified exclusions.",
        "cohorts": primary.group_by("diabetic", "uses_cgm").agg(
            pl.len().alias("n"), pl.col("mae_generic").mean().alias("mean_mae"),
            pl.col("mae_generic").std().alias("sd_mae")
        ).sort("diabetic", "uses_cgm").to_dicts(),
        "hypotheses": suite.to_dict(), "quality_sensitivity_hypotheses": clean_suite.to_dict(),
        "raw_study_id_hypotheses": run_all_hypotheses(raw_people).to_dict(),
        "primary_group_intervals": {
            "h1": group_summary(primary, "diabetic"), "h2": group_summary(primary, "uses_cgm")
        },
        "quality_group_intervals": {
            "h1": group_summary(quality_primary, "diabetic"),
            "h2": group_summary(quality_primary, "uses_cgm"),
        },
        "exploratory": exploratory_tests(rounds, people),
        "quality_exploratory": exploratory_tests(rounds, clean),
        "verification_flag_counts": {
            category: sum(issue.category == category for issue in verification.all_issues)
            for category in sorted({issue.category for issue in verification.all_issues})
        },
        "bootstrap": {"seed": SEED, "replicates": 20_000, "interval": "95% percentile, descriptive mean parameter"},
    }
    plans = []
    for sd in [8.0, result["primary_group_intervals"]["h2"]["pooled_sd"], 10.0, 12.0]:
        for difference in [3.5, 4.0]:
            n_a, n_b = required_independent_n(mean_difference=difference, sd=sd)
            plans.append({"sd": sd, "difference": difference, "n_per_group": n_a,
                          "eligible_total": n_a + n_b})
    result["planning"] = {
        "scenarios": plans, "primary_alpha": 0.025, "power_per_primary_test": 0.8,
        "correlation_r_point25_subgroup_n": required_correlation_n(0.25),
        "paired_target_assumed_difference3_sd6": required_paired_n(mean_difference=3, difference_sd=6),
        "paired_sd_empirically_estimable": False,
        "operational_target_eligible": 320, "target_per_diabetes_and_cgm_margin": 160,
        "target_difference": 3.5, "planning_sd": 10.0,
        "recorded_participant_target_using_observed_eligibility": enrollment_for_eligible(
            320, unusable_fraction=1 - primary.height / people.height
        ),
        "recorded_participant_target_using_quality_sensitivity": enrollment_for_eligible(
            320, unusable_fraction=1 - quality_primary.height / people.height
        ),
        "alternative_eligible_for_difference4": 250,
    }
    for key, status, duration, cap in [
        ("h3", "diabetic", "diabetes_duration_months", 80 * 12),
        ("h4", "uses_cgm", "cgm_duration_months", 30 * 12),
    ]:
        sub = primary.filter(pl.col(status) & pl.col(duration).is_between(0, cap))
        rho, p_value = stats.spearmanr(sub[duration].to_numpy(), sub["mae_generic"].to_numpy())
        result.setdefault("rank_correlation_sensitivity", {})[key] = {
            "n": sub.height, "spearman_rho": float(rho), "p": float(p_value)
        }
    processed = ROOT / "data/processed"
    people.write_parquet(processed / "participants_audited.parquet")
    runs.write_parquet(processed / "runs.parquet")
    rounds.write_parquet(processed / "rounds_latest.parquet")
    return result


if __name__ == "__main__":
    with start_action(action_type="statistics.production_interim_audit"):
        payload = assessment()
        dest = ROOT / "output/reports/production_assessment_2026-10-24.json"
        dest.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
        write_livia_briefing(payload, dest.with_name("livia-study-briefing.html"))
        print(json.dumps({"counts": payload["counts"], "planning": payload["planning"],
                          "exploratory": payload["exploratory"]}, indent=2))
        print(f"Saved {dest}")
