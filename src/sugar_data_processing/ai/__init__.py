"""Post-factum scoring of saved Sugar Sugar games.

Rebuild the 3-hour window the human saw, convert it to the glucose-forecasting
CSV layout, score persistence / linear / SugarOne (when present), then fold
those MAE values into the merged report.
"""

from sugar_data_processing.ai.export import export_prediction_sequences
from sugar_data_processing.ai.ingest import ingest_model_predictions
from sugar_data_processing.ai.run import run_post_factum_scoring
from sugar_data_processing.ai.sequences import build_point_table, known_data_rounds
from sugar_data_processing.ai.traces import LINE_STYLES, build_round_traces

__all__ = [
    "LINE_STYLES",
    "build_point_table",
    "build_round_traces",
    "known_data_rounds",
    "export_prediction_sequences",
    "ingest_model_predictions",
    "run_post_factum_scoring",
]
