"""Stage 3 — Statistical tests (study design §7.3–7.4).

H1–H5 hypothesis battery on person-level MAE. H6 is deferred until
computational baselines exist in sugar-sugar.
"""

from sugar_data_processing.statistics.catalog import HYPOTHESES, hypothesis_blurb, hypothesis_heading
from sugar_data_processing.statistics.hypotheses import HypothesisSuite, run_all_hypotheses
from sugar_data_processing.statistics.tests import (
    correlation_analysis,
    independent_group_comparison,
    paired_comparison,
)

__all__ = [
    "HYPOTHESES",
    "hypothesis_blurb",
    "hypothesis_heading",
    "HypothesisSuite",
    "run_all_hypotheses",
    "independent_group_comparison",
    "paired_comparison",
    "correlation_analysis",
]
