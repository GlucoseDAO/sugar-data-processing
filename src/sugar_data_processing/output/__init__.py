"""Stage 5 — Output.

Figures (PNG) and markdown / JSON study analysis reports.
"""

from sugar_data_processing.output.plots import generate_all_figures
from sugar_data_processing.output.report import write_report

__all__ = ["generate_all_figures", "write_report"]
