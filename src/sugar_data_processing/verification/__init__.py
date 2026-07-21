"""Stage 2 — Data verification.

Schema / structural checks plus demographic and metric quality flags.
"""

from sugar_data_processing.verification.anomalies import Anomaly, detect_anomalies
from sugar_data_processing.verification.report import VerificationReport, verify_dataset
from sugar_data_processing.verification.schema import verify_schema

__all__ = [
    "Anomaly",
    "VerificationReport",
    "detect_anomalies",
    "verify_dataset",
    "verify_schema",
]
