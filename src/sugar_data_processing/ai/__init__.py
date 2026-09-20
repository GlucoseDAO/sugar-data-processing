"""Split AI follow-on: export saved games to models, then ingest scored output.

Stage A — ``export_prediction_sequences``
    Reuse the human study export as a tidy point-level CSV (timestamp, glucose
    values, window location) so models can be scored **post factum**.

Stage B — ``ingest_model_predictions`` + ``write_ai_report``
    Read already-scored model files and render the AI edition of the report
    (same traits as the human edition, plus human-vs-model comparison).

Evaluation modes
    ``post_factum`` — models run later on saved sequences (what we have now).
    ``in_place`` — models scored during the game (not collected yet).
"""

from sugar_data_processing.ai.export import export_prediction_sequences
from sugar_data_processing.ai.ingest import ingest_model_predictions
from sugar_data_processing.ai.sequences import build_point_table, known_data_rounds

__all__ = [
    "build_point_table",
    "known_data_rounds",
    "export_prediction_sequences",
    "ingest_model_predictions",
]
