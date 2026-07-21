"""Matplotlib / seaborn figures for the study-design scenarios."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from eliot import start_action

from sugar_data_processing.comparison.benchmarks import BenchmarkContext
from sugar_data_processing.config import (
    CGM_DURATION_BINS,
    DEEP_LEARNING_MAE_RANGE,
    DIABETES_DURATION_BINS,
    SIMPLE_BASELINE_MAE_RANGE,
)
from sugar_data_processing.statistics.hypotheses import HypothesisSuite

sns.set_theme(style="whitegrid", context="talk")


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _duration_bin(value: float | None, bins: list[tuple[str, float, float]]) -> str | None:
    if value is None or not np.isfinite(value):
        return None
    for label, lo, hi in bins:
        if lo <= value < hi:
            return label
    return bins[-1][0]


def generate_all_figures(
    participants: pl.DataFrame,
    suite: HypothesisSuite,
    benchmarks: BenchmarkContext,
    figures_dir: Path,
) -> dict[str, Path]:
    """Create the standard figure set and return relative name → path."""
    with start_action(action_type="output.generate_all_figures") as action:
        figures_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        paths["mae_by_diabetes"] = _plot_group_mae(
            participants,
            "diabetic",
            "Diabetes status vs person MAE (PwD vs non-PwD)",
            figures_dir / "h1_mae_by_diabetes.png",
        )
        paths["mae_by_cgm"] = _plot_group_mae(
            participants,
            "uses_cgm",
            "CGM use vs person MAE (users vs non-users)",
            figures_dir / "h2_mae_by_cgm.png",
        )
        paths["diabetes_duration_scatter"] = _plot_duration_scatter(
            participants,
            x_col="diabetes_duration",
            title="Diabetes duration vs person MAE (PwD only)",
            xlabel="Diabetes duration (years)",
            path=figures_dir / "h3_diabetes_duration_scatter.png",
            diabetic_only=True,
        )
        paths["cgm_duration_scatter"] = _plot_duration_scatter(
            participants,
            x_col="cgm_duration_years",
            title="CGM experience vs person MAE (CGM users only)",
            xlabel="CGM experience (years)",
            path=figures_dir / "h4_cgm_duration_scatter.png",
            cgm_only=True,
        )
        paths["duration_bins"] = _plot_duration_bins(participants, figures_dir / "h3_h4_duration_bins.png")
        paths["own_vs_generic"] = _plot_own_vs_generic(
            participants, figures_dir / "h5_own_vs_generic.png"
        )
        paths["benchmark_bands"] = _plot_benchmark_bands(
            participants, benchmarks, figures_dir / "benchmark_human_vs_literature.png"
        )
        paths["mae_distribution"] = _plot_mae_distribution(
            participants, figures_dir / "mae_distribution.png"
        )
        action.log(message_type="info", n_figures=len(paths), suite_h1=suite.h1 is not None)
        return paths


def _plot_group_mae(participants: pl.DataFrame, group_col: str, title: str, path: Path) -> Path:
    df = participants.filter(pl.col("mae_primary").is_not_null()).filter(pl.col(group_col).is_not_null())
    fig, ax = plt.subplots(figsize=(8, 5))
    if df.height == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return _save(fig, path)

    labels: list[str] = []
    values: list[float] = []
    for row in df.iter_rows(named=True):
        raw = row[group_col]
        if raw is True:
            labels.append("Yes")
        elif raw is False:
            labels.append("No")
        else:
            labels.append(str(raw))
        values.append(float(row["mae_primary"]))
    sns.boxplot(x=labels, y=values, ax=ax, color="#4C78A8")
    sns.stripplot(x=labels, y=values, ax=ax, color="#333333", alpha=0.55, size=5)
    ax.set_title(title)
    ax.set_xlabel(group_col.replace("_", " ").title())
    ax.set_ylabel("Person MAE (mg/dL)")
    return _save(fig, path)


def _plot_duration_scatter(
    participants: pl.DataFrame,
    *,
    x_col: str,
    title: str,
    xlabel: str,
    path: Path,
    diabetic_only: bool = False,
    cgm_only: bool = False,
) -> Path:
    df = participants.filter(pl.col("mae_primary").is_not_null()).filter(pl.col(x_col).is_not_null())
    if diabetic_only:
        df = df.filter(pl.col("diabetic") == True)  # noqa: E712
    if cgm_only:
        df = df.filter(pl.col("uses_cgm") == True)  # noqa: E712
    fig, ax = plt.subplots(figsize=(8, 5))
    if df.height == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return _save(fig, path)
    x = df[x_col].to_numpy().astype(float)
    y = df["mae_primary"].to_numpy().astype(float)
    ax.scatter(x, y, alpha=0.7, color="#4C78A8", edgecolor="white", s=60)
    if x.size >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        ax.plot(xs, slope * xs + intercept, color="#E45756", lw=2, label="linear fit")
        log_x = np.log(x + 1.0)
        log_slope, log_intercept = np.polyfit(log_x, y, 1)
        ax.plot(
            xs,
            log_slope * np.log(xs + 1.0) + log_intercept,
            color="#F58518",
            lw=2,
            ls="--",
            label="log fit",
        )
        ax.legend(fontsize=10)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Person MAE (mg/dL)")
    return _save(fig, path)


def _plot_duration_bins(participants: pl.DataFrame, path: Path) -> Path:
    rows: list[dict[str, object]] = []
    for row in participants.iter_rows(named=True):
        mae = row.get("mae_primary")
        if mae is None or not np.isfinite(mae):
            continue
        if row.get("diabetic"):
            label = _duration_bin(row.get("diabetes_duration"), DIABETES_DURATION_BINS)
            if label:
                rows.append({"facet": "Diabetes duration", "bin": label, "mae": mae})
        if row.get("uses_cgm"):
            label = _duration_bin(row.get("cgm_duration_years"), CGM_DURATION_BINS)
            if label:
                rows.append({"facet": "CGM experience", "bin": label, "mae": mae})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    if not rows:
        for ax in axes:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()
        return _save(fig, path)

    for ax, facet in zip(axes, ["Diabetes duration", "CGM experience"], strict=True):
        sub = [r for r in rows if r["facet"] == facet]
        if not sub:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()
            continue
        order = (
            [b[0] for b in DIABETES_DURATION_BINS]
            if facet.startswith("Diabetes")
            else [b[0] for b in CGM_DURATION_BINS]
        )
        sns.boxplot(
            x=[str(r["bin"]) for r in sub],
            y=[float(r["mae"]) for r in sub],  # type: ignore[arg-type]
            order=order,
            ax=ax,
            color="#72B7B2",
        )
        ax.set_title(facet)
        ax.set_xlabel("")
        ax.set_ylabel("Person MAE (mg/dL)")
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Exploratory duration bins (study design §7.4)", y=1.02)
    return _save(fig, path)


def _plot_own_vs_generic(participants: pl.DataFrame, path: Path) -> Path:
    df = participants.filter(
        pl.col("mae_generic").is_not_null() & pl.col("mae_own").is_not_null()
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    if df.height == 0:
        ax.text(0.5, 0.5, "No paired own/generic data", ha="center", va="center")
        ax.set_axis_off()
        return _save(fig, path)
    x = df["mae_generic"].to_numpy().astype(float)
    y = df["mae_own"].to_numpy().astype(float)
    ax.scatter(x, y, alpha=0.75, color="#54A24B", edgecolor="white", s=70)
    lim_max = float(max(np.max(x), np.max(y)) * 1.05)
    lim_min = float(min(np.min(x), np.min(y)) * 0.95)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color="#666666", ls="--", label="equal MAE")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal")
    ax.set_xlabel("Generic-data MAE (mg/dL)")
    ax.set_ylabel("Own-data MAE (mg/dL)")
    ax.set_title("Own vs generic MAE (below diagonal → better on own data)")
    ax.legend(fontsize=10)
    return _save(fig, path)


def _plot_benchmark_bands(
    participants: pl.DataFrame,
    benchmarks: BenchmarkContext,
    path: Path,
) -> Path:
    values = (
        participants.filter(pl.col("mae_primary").is_not_null())["mae_primary"]
        .to_numpy()
        .astype(float)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    if values.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return _save(fig, path)
    sns.kdeplot(values, ax=ax, fill=True, color="#4C78A8", alpha=0.35, label="Human MAE")
    ax.axvline(benchmarks.human_mean_mae, color="#4C78A8", lw=2, label="Human mean")
    s_lo, s_hi = SIMPLE_BASELINE_MAE_RANGE
    d_lo, d_hi = DEEP_LEARNING_MAE_RANGE
    ax.axvspan(s_lo, s_hi, color="#F58518", alpha=0.15, label="Simple / ARIMA band")
    ax.axvspan(d_lo, d_hi, color="#54A24B", alpha=0.15, label="Deep-learning band")
    ax.set_xlabel("Person MAE (mg/dL)")
    ax.set_ylabel("Density")
    ax.set_title("Human accuracy vs published 60-min model bands")
    ax.legend(fontsize=9)
    return _save(fig, path)


def _plot_mae_distribution(participants: pl.DataFrame, path: Path) -> Path:
    df = participants.filter(pl.col("mae_primary").is_not_null())
    fig, ax = plt.subplots(figsize=(8, 5))
    if df.height == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return _save(fig, path)
    sns.histplot(df["mae_primary"].to_numpy(), bins=20, ax=ax, color="#9D755D", edgecolor="white")
    ax.set_xlabel("Person MAE (mg/dL)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of per-person MAE")
    return _save(fig, path)
