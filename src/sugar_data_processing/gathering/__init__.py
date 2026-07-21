"""Stage 1 — Data gathering.

Load the sugar-sugar ``prediction_statistics.csv`` export, explode per-round
metrics, and build the person-level analysis table.
"""

from sugar_data_processing.gathering.load import (
    REQUIRED_COLUMNS,
    load_prediction_statistics,
    parse_per_round_metrics,
)
from sugar_data_processing.gathering.participants import (
    build_participant_table,
    h5_paired_population,
    own_analysis_population,
    primary_analysis_population,
)
from sugar_data_processing.gathering.rounds import build_round_table, classify_source

__all__ = [
    "REQUIRED_COLUMNS",
    "load_prediction_statistics",
    "parse_per_round_metrics",
    "build_round_table",
    "classify_source",
    "build_participant_table",
    "primary_analysis_population",
    "own_analysis_population",
    "h5_paired_population",
]
