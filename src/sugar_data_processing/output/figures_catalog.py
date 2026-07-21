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
        "H1 - Diabetes status: PwD vs non-PwD",
        "Lower boxes mean better accuracy. Compare the two groups visually.",
    ),
    ReportFigure(
        "h2_mae_by_cgm.png",
        "H2 - CGM use: users vs non-users",
        "Same idea as H1, now split by CGM use.",
    ),
    ReportFigure(
        "h3_diabetes_duration_scatter.png",
        "H3 - Diabetes duration vs MAE (PwD only)",
        "A downward trend would mean longer duration associates with better accuracy.",
    ),
    ReportFigure(
        "h4_cgm_duration_scatter.png",
        "H4 - CGM experience vs MAE (CGM users)",
        "A downward trend would mean more CGM experience associates with better accuracy.",
    ),
    ReportFigure(
        "h3_h4_duration_bins.png",
        "Exploratory duration / experience bins",
        "Same story as the scatters, summarised in duration bins.",
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
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.imshow(mpimg.imread(path))
        ax.set_axis_off()
        ax.set_title(item.title)
        plt.show()
        plt.close(fig)
    return shown
