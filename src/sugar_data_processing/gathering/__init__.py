"""Stage 1 — Data gathering.

Load the sugar-sugar ``prediction_statistics.csv`` export, explode per-round
metrics (and ``round_context``), and build the person-level analysis table.
"""

from sugar_data_processing.gathering.encoding import (
    cgm_duration_to_years,
    parse_cgm_duration,
    parse_optional_bool,
    parse_per_round_metrics,
    parse_round_context,
)
from sugar_data_processing.gathering.load import (
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    load_prediction_statistics,
)
from sugar_data_processing.gathering.participants import (
    all_format_population,
    build_participant_table,
    h5_paired_population,
    own_analysis_population,
    primary_analysis_population,
)
from sugar_data_processing.gathering.rounds import build_round_table, classify_source
from sugar_data_processing.gathering.sources import classify_data_class, redact_source_name

__all__ = [
    "REQUIRED_COLUMNS",
    "OPTIONAL_COLUMNS",
    "load_prediction_statistics",
    "parse_per_round_metrics",
    "parse_round_context",
    "parse_cgm_duration",
    "parse_optional_bool",
    "cgm_duration_to_years",
    "build_round_table",
    "classify_source",
    "classify_data_class",
    "redact_source_name",
    "build_participant_table",
    "primary_analysis_population",
    "own_analysis_population",
    "h5_paired_population",
    "all_format_population",
]
