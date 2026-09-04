"""Stage 5 — Output.

Figures (PNG) and markdown / JSON study analysis reports.
"""

from sugar_data_processing.output.explorer import write_explorer_html
from sugar_data_processing.output.figures_catalog import (
    REPORT_FIGURES,
    report_figures_dir,
    show_report_figures,
)
from sugar_data_processing.output.narration import (
    STATS_GLOSSARY,
    explain_all_hypotheses,
    explain_benchmarks,
    explain_gathering,
    explain_report_written,
    explain_study_goal,
    explain_verification,
    how_to_read_report,
)
from sugar_data_processing.output.plots import generate_all_figures
from sugar_data_processing.output.report import write_report

__all__ = [
    "REPORT_FIGURES",
    "STATS_GLOSSARY",
    "explain_all_hypotheses",
    "explain_benchmarks",
    "explain_gathering",
    "explain_report_written",
    "explain_study_goal",
    "explain_verification",
    "generate_all_figures",
    "how_to_read_report",
    "report_figures_dir",
    "show_report_figures",
    "write_explorer_html",
    "write_report",
]
