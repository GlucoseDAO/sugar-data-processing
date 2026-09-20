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
    COHORT_DIABETIC_CGM,
    COHORT_DIABETIC_NON_CGM,
    COHORT_LABELS,
    COHORT_NONDIABETIC_CGM,
    COHORT_NONDIABETIC_NON_CGM,
    DEEP_LEARNING_MAE_RANGE,
    MAX_PLAUSIBLE_CGM_YEARS,
    MAX_PLAUSIBLE_DIABETES_YEARS,
    SIMPLE_BASELINE_MAE_RANGE,
)
from sugar_data_processing.statistics.hypotheses import HypothesisSuite

sns.set_theme(style="whitegrid", context="talk")


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


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
            "Diabetes status vs MAE, each group split into generic vs own",
            figures_dir / "h1_mae_by_diabetes.png",
            yes_label="PwD",
            no_label="non-PwD",
        )
        paths["mae_by_cgm"] = _plot_group_mae(
            participants,
            "uses_cgm",
            "CGM use vs MAE, each group split into generic vs own",
            figures_dir / "h2_mae_by_cgm.png",
            yes_label="CGM user",
            no_label="no CGM",
        )
        paths["diabetes_duration_scatter"] = _plot_duration_scatter(
            participants,
            x_col="diabetes_duration_months",
            title="Diabetes duration vs MAE on generic and own data (PwD only)",
            xlabel="Diabetes duration (months)",
            path=figures_dir / "h3_diabetes_duration_scatter.png",
            diabetic_only=True,
        )
        paths["cgm_duration_scatter"] = _plot_duration_scatter(
            participants,
            x_col="cgm_duration_months",
            title="CGM experience vs MAE on generic and own data (CGM users only)",
            xlabel="CGM experience (months)",
            path=figures_dir / "h4_cgm_duration_scatter.png",
            cgm_only=True,
        )
        paths["people_clusters"] = _plot_people_clusters(
            participants, figures_dir / "people_clusters.png"
        )
        paths["opposite_trait"] = _plot_opposite_trait(
            participants, figures_dir / "opposite_trait.png"
        )
        paths["own_vs_generic"] = _plot_own_vs_generic(
            participants, figures_dir / "h5_own_vs_generic.png"
        )
        paths["benchmark_bands"] = _plot_benchmark_bands(
            participants, benchmarks, figures_dir / "benchmark_human_vs_literature.png"
        )
        paths["mae_distribution"] = _plot_mae_distribution(
            participants, figures_dir / "mae_distribution.png"
        )
        paths["cohort_pie"] = _plot_cohort_pie(
            participants, figures_dir / "cohort_categories_pie.png"
        )
        paths["mae_by_format"] = _plot_mae_by_format(
            participants, figures_dir / "mae_by_format.png"
        )
        paths["players_vs_repeats"] = _plot_players_vs_repeats(
            participants, figures_dir / "players_vs_repeats.png"
        )
        paths["all_formats_own_vs_generic"] = _plot_all_formats_own_vs_generic(
            participants, figures_dir / "all_formats_own_vs_generic.png"
        )
        action.log(message_type="info", n_figures=len(paths), suite_h1=suite.h1 is not None)
        return paths


def _plot_group_mae(
    participants: pl.DataFrame,
    group_col: str,
    title: str,
    path: Path,
    *,
    yes_label: str,
    no_label: str,
) -> Path:
    """Category on the x-axis, each category split into generic vs own MAE."""
    fig, ax = plt.subplots(figsize=(12, 7.5))
    if group_col not in participants.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return _save(fig, path)

    labels: list[str] = []
    sources: list[str] = []
    values: list[float] = []
    for row in participants.iter_rows(named=True):
        raw = row.get(group_col)
        if raw is True:
            group = yes_label
        elif raw is False:
            group = no_label
        else:
            continue
        for source, col in (("generic", "mae_generic"), ("own", "mae_own")):
            mae = row.get(col)
            if mae is None or not np.isfinite(mae):
                continue
            labels.append(group)
            sources.append(source)
            values.append(float(mae))

    if not values:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return _save(fig, path)

    order = [yes_label, no_label]
    palette = {"generic": "#4C78A8", "own": "#54A24B"}
    sns.violinplot(
        x=labels,
        y=values,
        hue=sources,
        order=order,
        hue_order=["generic", "own"],
        palette=palette,
        ax=ax,
        inner=None,
        cut=0,
        dodge=True,
    )
    sns.swarmplot(
        x=labels,
        y=values,
        hue=sources,
        order=order,
        hue_order=["generic", "own"],
        palette={"generic": "#1e293b", "own": "#1e293b"},
        ax=ax,
        size=4,
        alpha=0.75,
        dodge=True,
        legend=False,
    )
    ax.set_title(title)
    ax.set_xlabel("Category")
    ax.set_ylabel("Person MAE (mg/dL)")
    ax.legend(title="Data source", fontsize=10)
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
    df = participants
    if x_col == "diabetes_duration_months" and x_col not in df.columns and "diabetes_duration" in df.columns:
        df = df.with_columns((pl.col("diabetes_duration") * 12.0).alias("diabetes_duration_months"))
    if x_col == "cgm_duration_months" and x_col not in df.columns and "cgm_duration_years" in df.columns:
        df = df.with_columns((pl.col("cgm_duration_years") * 12.0).alias("cgm_duration_months"))
    if diabetic_only:
        df = df.filter(pl.col("diabetic") == True)  # noqa: E712
    if cgm_only:
        df = df.filter(pl.col("uses_cgm") == True)  # noqa: E712
    if x_col == "diabetes_duration_months" and x_col in df.columns:
        df = df.filter(pl.col(x_col) <= MAX_PLAUSIBLE_DIABETES_YEARS * 12.0)
    if x_col == "cgm_duration_months" and x_col in df.columns:
        df = df.filter(pl.col(x_col) <= MAX_PLAUSIBLE_CGM_YEARS * 12.0)
    fig, ax = plt.subplots(figsize=(12, 7.5))
    plotted = 0
    series = (
        ("generic", "mae_generic", "#4C78A8"),
        ("own", "mae_own", "#54A24B"),
    )
    for label, y_col, color in series:
        if x_col not in df.columns or y_col not in df.columns:
            continue
        sub = df.filter(pl.col(x_col).is_not_null() & pl.col(y_col).is_not_null())
        if sub.height == 0:
            continue
        x = sub[x_col].to_numpy().astype(float)
        y = sub[y_col].to_numpy().astype(float)
        ax.scatter(x, y, alpha=0.75, color=color, edgecolor="white", s=60, label=label)
        plotted += 1
        if x.size >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            ax.plot(xs, slope * xs + intercept, color=color, lw=2)
    if plotted == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return _save(fig, path)
    ax.legend(fontsize=10, title="Data source")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Person MAE (mg/dL)")
    return _save(fig, path)


def _plot_people_clusters(participants: pl.DataFrame, path: Path) -> Path:
    """Each person as a point in own-vs-generic space, coloured by cohort."""
    df = participants.filter(pl.col("mae_primary").is_not_null())
    fig, ax = plt.subplots(figsize=(12.5, 9))
    if df.height == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return _save(fig, path)

    for key in _COHORT_ORDER:
        sub = df.filter(pl.col("cohort_category") == key) if "cohort_category" in df.columns else df.head(0)
        if sub.height == 0:
            continue
        x = (
            sub["mae_generic"].fill_null(sub["mae_primary"]).to_numpy().astype(float)
            if "mae_generic" in sub.columns
            else sub["mae_primary"].to_numpy().astype(float)
        )
        y = (
            sub["mae_own"].fill_null(sub["mae_primary"]).to_numpy().astype(float)
            if "mae_own" in sub.columns
            else sub["mae_primary"].to_numpy().astype(float)
        )
        challenge = (
            sub["played_challenge_unknown"].to_list()
            if "played_challenge_unknown" in sub.columns
            else [False] * sub.height
        )
        face = [_COHORT_COLORS[key]] * sub.height
        edge = ["#111827" if flag else "white" for flag in challenge]
        ax.scatter(
            x,
            y,
            s=70,
            alpha=0.8,
            c=face,
            edgecolors=edge,
            linewidths=1.2,
            label=COHORT_LABELS.get(key, key),
        )
    ax.set_xlabel("Generic MAE (mg/dL)")
    ax.set_ylabel("Own MAE (mg/dL) — falls back to person MAE if missing")
    ax.set_title("People as points (colour = cohort, dark ring = Challenge the unknown)")
    ax.legend(fontsize=8, loc="best")
    return _save(fig, path)


def _plot_opposite_trait(participants: pl.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    same_col = "mae_same_trait"
    opp_col = "mae_opposite_trait"
    labels: list[str] = []
    values: list[float] = []
    if same_col in participants.columns and opp_col in participants.columns:
        for row in participants.iter_rows(named=True):
            if row.get(same_col) is not None and np.isfinite(row[same_col]):
                labels.append("Same trait")
                values.append(float(row[same_col]))
            if row.get(opp_col) is not None and np.isfinite(row[opp_col]):
                labels.append("Opposite trait")
                values.append(float(row[opp_col]))
    if values:
        sns.violinplot(x=labels, y=values, ax=axes[0], inner=None, cut=0, color="#72B7B2")
        sns.swarmplot(x=labels, y=values, ax=axes[0], color="#1e293b", size=4, alpha=0.8)
        axes[0].set_ylabel("Person MAE (mg/dL)")
        axes[0].set_title("Same-trait vs opposite-trait traces")
    else:
        axes[0].text(0.5, 0.5, "No opposite-trait scores yet", ha="center", va="center")
        axes[0].set_axis_off()

    paired = participants
    if same_col in participants.columns and opp_col in participants.columns:
        paired = participants.filter(
            pl.col(same_col).is_not_null() & pl.col(opp_col).is_not_null()
        )
    else:
        paired = participants.head(0)
    if paired.height:
        x = paired[same_col].to_numpy().astype(float)
        y = paired[opp_col].to_numpy().astype(float)
        axes[1].scatter(x, y, alpha=0.8, color="#E45756", edgecolor="white", s=70)
        lim_max = float(max(np.max(x), np.max(y)) * 1.08)
        lim_min = float(min(np.min(x), np.min(y)) * 0.92)
        axes[1].plot([lim_min, lim_max], [lim_min, lim_max], color="#666666", ls="--")
        axes[1].set_xlim(lim_min, lim_max)
        axes[1].set_ylim(lim_min, lim_max)
        axes[1].set_aspect("equal")
        axes[1].set_xlabel("Same-trait MAE")
        axes[1].set_ylabel("Opposite-trait MAE")
        axes[1].set_title("Below diagonal = better on the opposite trait")
    else:
        axes[1].text(0.5, 0.5, "No one has both sides", ha="center", va="center")
        axes[1].set_axis_off()
    fig.suptitle("Challenge the unknown / opposite-trait play", y=1.02)
    return _save(fig, path)


def _plot_own_vs_generic(participants: pl.DataFrame, path: Path) -> Path:
    df = participants.filter(
        pl.col("mae_generic").is_not_null() & pl.col("mae_own").is_not_null()
    )
    fig, ax = plt.subplots(figsize=(10, 10))
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
    fig, ax = plt.subplots(figsize=(13, 7.5))
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
    fig, ax = plt.subplots(figsize=(12, 7.5))
    if df.height == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return _save(fig, path)
    values = df["mae_primary"].to_numpy().astype(float)
    sns.kdeplot(values, ax=ax, fill=True, color="#9D755D", alpha=0.35)
    sns.rugplot(values, ax=ax, color="#1e293b", height=0.06)
    ax.scatter(
        values,
        np.full_like(values, 0.02 * max(ax.get_ylim()[1], 1e-6)),
        alpha=0.45,
        s=18,
        color="#1e293b",
        zorder=3,
    )
    ax.set_xlabel("Person MAE (mg/dL)")
    ax.set_ylabel("Density")
    ax.set_title("Per-person MAE as a point cloud (each tick is one person)")
    return _save(fig, path)


_COHORT_ORDER: tuple[str, ...] = (
    COHORT_DIABETIC_CGM,
    COHORT_DIABETIC_NON_CGM,
    COHORT_NONDIABETIC_CGM,
    COHORT_NONDIABETIC_NON_CGM,
)
_COHORT_COLORS: dict[str, str] = {
    COHORT_DIABETIC_CGM: "#E45756",
    COHORT_DIABETIC_NON_CGM: "#F58518",
    COHORT_NONDIABETIC_CGM: "#4C78A8",
    COHORT_NONDIABETIC_NON_CGM: "#72B7B2",
}
_FORMAT_COLORS: dict[str, str] = {"A": "#4C78A8", "B": "#54A24B", "C": "#F58518"}


def _plot_cohort_pie(participants: pl.DataFrame, path: Path) -> Path:
    counts: dict[str, int] = {key: 0 for key in _COHORT_ORDER}
    if "cohort_category" in participants.columns:
        for row in participants.iter_rows(named=True):
            key = str(row.get("cohort_category") or "")
            if key in counts:
                counts[key] += 1

    fig, ax = plt.subplots(figsize=(8, 8))
    values = [counts[k] for k in _COHORT_ORDER]
    if sum(values) == 0:
        ax.text(0.5, 0.5, "No cohort labels", ha="center", va="center")
        ax.set_axis_off()
        return _save(fig, path)

    labels = [f"{COHORT_LABELS[k]} ({counts[k]})" for k in _COHORT_ORDER if counts[k] > 0]
    sizes = [counts[k] for k in _COHORT_ORDER if counts[k] > 0]
    colors = [_COHORT_COLORS[k] for k in _COHORT_ORDER if counts[k] > 0]
    wedges, _texts, autotexts = ax.pie(
        sizes,
        labels=None,
        colors=colors,
        autopct=lambda pct: f"{pct:.0f}%" if pct >= 4 else "",
        startangle=90,
        wedgeprops={"linewidth": 1.5, "edgecolor": "white"},
        textprops={"fontsize": 10},
    )
    for text in autotexts:
        text.set_color("white")
        text.set_fontweight("bold")
    ax.legend(
        wedges,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=10,
    )
    ax.set_aspect("equal")
    ax.set_title("Players by diabetes × CGM category")
    return _save(fig, path)


def _plot_mae_by_format(participants: pl.DataFrame, path: Path) -> Path:
    labels: list[str] = []
    values: list[float] = []
    for fmt, col in (("A", "mae_format_a"), ("B", "mae_format_b"), ("C", "mae_format_c")):
        if col not in participants.columns:
            continue
        for raw in participants[col].to_list():
            if raw is None:
                continue
            number = float(raw)
            if np.isfinite(number):
                labels.append(f"Format {fmt}")
                values.append(number)

    fig, ax = plt.subplots(figsize=(12, 8))
    if not values:
        ax.text(0.5, 0.5, "No per-format MAE", ha="center", va="center")
        ax.set_axis_off()
        return _save(fig, path)

    order = ["Format A", "Format B", "Format C"]
    palette = {"Format A": _FORMAT_COLORS["A"], "Format B": _FORMAT_COLORS["B"], "Format C": _FORMAT_COLORS["C"]}
    sns.violinplot(
        x=labels, y=values, order=order, hue=labels, palette=palette, legend=False, ax=ax, inner=None, cut=0
    )
    sns.swarmplot(x=labels, y=values, order=order, ax=ax, color="#1e293b", size=4, alpha=0.75)
    ax.set_title("How people performed on each task (one point per person)")
    ax.set_xlabel("Task (A = generic, B = own data, C = mixed)")
    ax.set_ylabel("Person MAE on that format (mg/dL)")
    return _save(fig, path)


def _plot_players_vs_repeats(participants: pl.DataFrame, path: Path) -> Path:
    n_people = participants.height
    n_repeat = (
        int(participants.filter(pl.col("is_repeat_player")).height)
        if "is_repeat_player" in participants.columns
        else 0
    )
    n_single = n_people - n_repeat
    n_runs = int(participants["n_runs"].sum()) if "n_runs" in participants.columns else n_people

    fig, ax = plt.subplots(figsize=(12, 8))
    labels: list[str] = []
    values: list[float] = []
    if "is_repeat_player" in participants.columns:
        for row in participants.iter_rows(named=True):
            mae = row.get("mae_primary")
            if mae is None or not np.isfinite(mae):
                continue
            labels.append("Repeat players" if row.get("is_repeat_player") else "Single-run")
            values.append(float(mae))
    if values:
        sns.violinplot(
            x=labels,
            y=values,
            order=["Single-run", "Repeat players"],
            ax=ax,
            color="#4C78A8",
            inner=None,
            cut=0,
        )
        sns.swarmplot(
            x=labels,
            y=values,
            order=["Single-run", "Repeat players"],
            ax=ax,
            color="#1e293b",
            alpha=0.8,
            size=5,
        )
        ax.set_ylabel("Person MAE (mg/dL)")
        ax.set_title(
            f"First-timers vs people who came back "
            f"(n={n_people} people, {n_single} single / {n_repeat} repeat, {n_runs} saved runs)"
        )
    else:
        ax.text(0.5, 0.5, "No MAE", ha="center", va="center")
        ax.set_axis_off()
    return _save(fig, path)


def _plot_all_formats_own_vs_generic(participants: pl.DataFrame, path: Path) -> Path:
    if "played_all_formats" not in participants.columns:
        fig, ax = plt.subplots(figsize=(11, 9))
        ax.text(0.5, 0.5, "No all-format flag", ha="center", va="center")
        ax.set_axis_off()
        return _save(fig, path)

    df = participants.filter(pl.col("played_all_formats")).filter(
        pl.col("mae_generic").is_not_null() & pl.col("mae_own").is_not_null()
    )
    fig, ax = plt.subplots(figsize=(11, 10))
    if df.height == 0:
        ax.text(0.5, 0.5, "No one played A, B and C with both scores", ha="center", va="center")
        ax.set_axis_off()
        return _save(fig, path)

    x = df["mae_generic"].to_numpy().astype(float)
    y = df["mae_own"].to_numpy().astype(float)
    better_own = int(np.sum(y < x))
    better_generic = int(np.sum(x < y))
    tied = int(df.height - better_own - better_generic)
    ax.scatter(x, y, alpha=0.8, color="#54A24B", edgecolor="white", s=80)
    lim_max = float(max(np.max(x), np.max(y)) * 1.08)
    lim_min = float(min(np.min(x), np.min(y)) * 0.92)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color="#666666", ls="--", label="equal MAE")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal")
    ax.set_xlabel("Generic-data MAE (mg/dL)")
    ax.set_ylabel("Own-data MAE (mg/dL)")
    ax.set_title(
        f"Players who tried every variant (n={df.height})\n"
        f"Better on own: {better_own}  |  Better on generic: {better_generic}  |  Tied: {tied}"
    )
    ax.legend(fontsize=10)
    return _save(fig, path)

