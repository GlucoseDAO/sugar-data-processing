"""Catalog of report figures with plain-language titles and reading tips."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ReportFigure:
    filename: str
    title: str
    tip: str


REPORT_FIGURES: tuple[ReportFigure, ...] = (
    ReportFigure(
        "mae_distribution.png",
        "Overall distribution of per-person MAE",
        "Spread of accuracy across people. Look for centre, spread, and outliers.",
    ),
    ReportFigure(
        "h1_mae_by_diabetes.png",
        "H1 - Diabetes status: PwD vs non-PwD, generic vs own",
        "Each category is split into generic-data MAE and own-data MAE. Lower is better.",
    ),
    ReportFigure(
        "h2_mae_by_cgm.png",
        "H2 - CGM use: users vs non-users, generic vs own",
        "Same idea as H1: category first, then generic vs own inside each group.",
    ),
    ReportFigure(
        "h3_diabetes_duration_scatter.png",
        "H3 - Diabetes duration (months) vs MAE on generic and own data",
        "Duration is in months so short experience is readable. Blue is generic, green is own.",
    ),
    ReportFigure(
        "h4_cgm_duration_scatter.png",
        "H4 - CGM experience (months) vs MAE on generic and own data",
        "Same layout as H3: months on x, blue generic / green own. A downward trend would mean more experience helps.",
    ),
    ReportFigure(
        "people_clusters.png",
        "People as points in own vs generic space",
        "Each dot is one person. Colour is the diabetes × CGM cohort. A dark ring means they opted into Challenge the unknown.",
    ),
    ReportFigure(
        "opposite_trait.png",
        "Same-trait vs opposite-trait traces",
        "Left: every person who has a score on that side. Right: people who played both. Below the diagonal is better on the opposite trait.",
    ),
    ReportFigure(
        "h5_own_vs_generic.png",
        "H5 - Own-data MAE vs generic-data MAE",
        "Points below the diagonal: better on own data than on generic data.",
    ),
    ReportFigure(
        "benchmark_human_vs_literature.png",
        "Human MAE vs published literature bands",
        "Where the human distribution sits relative to published model bands.",
    ),
    ReportFigure(
        "cohort_categories_pie.png",
        "Cohort mix: diabetes × CGM",
        "How many unique people sit in each of the four diabetes × CGM buckets.",
    ),
    ReportFigure(
        "mae_by_format.png",
        "Accuracy on each task (formats A, B, C)",
        "Lower boxes mean better accuracy. A is generic, B is own data, C is mixed.",
    ),
    ReportFigure(
        "players_vs_repeats.png",
        "Unique players vs people who played again",
        "Left: counts. Right: whether coming back is associated with better MAE.",
    ),
    ReportFigure(
        "all_formats_own_vs_generic.png",
        "People who played every variant: own vs generic",
        "Below the diagonal: better on their own data than on generic traces.",
    ),
    ReportFigure(
        "ai_task_mae.png",
        "Same person, same task: human vs AI MAE",
        "Generic (A), Own (B), Mixed (C). A person only appears if they have both scores on that task. Mixed is format C.",
    ),
    ReportFigure(
        "ai_same_user_cluster.png",
        "Same user on generic vs own: human and AI",
        "Only people with Generic (A) and Own (B) on both sides. Grey line joins that person's human point to their AI point.",
    ),
)


def report_figures_dir(output_dir: Path) -> Path:
    """Canonical figures directory next to the markdown report."""
    return Path(output_dir) / "reports" / "figures"


def show_report_figures(output_dir: Path) -> list[Path]:
    """Display every standard report figure with title and reading tip.

    Intended for notebooks; uses matplotlib so it does not depend on IPython widgets.
    Returns the paths that were shown (or would have been shown if present).
    """
    figures_dir = report_figures_dir(output_dir)
    shown: list[Path] = []
    for item in REPORT_FIGURES:
        path = figures_dir / item.filename
        print(f"\n=== {item.title} ===")
        print(item.tip)
        shown.append(path)
        if not path.exists():
            print("missing:", path)
            continue
        fig, ax = plt.subplots(figsize=(12, 7.5))
        ax.imshow(mpimg.imread(path))
        ax.set_axis_off()
        ax.set_title(item.title)
        plt.show()
        plt.close(fig)
    return shown
