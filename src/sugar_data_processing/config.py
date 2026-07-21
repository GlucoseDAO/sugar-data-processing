"""Study-design constants used across the analysis pipeline."""

from __future__ import annotations

from pathlib import Path

# Primary analysis population (study design §7.2)
MIN_GENERIC_SEGMENTS: int = 6
MIN_OWN_SEGMENTS: int = 6

# All hypothesis tests (study design §7.3–7.4)
ALPHA: float = 0.05
SHAPIRO_NORMAL_P: float = 0.05

# Format → data-source mapping used by sugar-sugar
FORMAT_GENERIC: str = "A"
FORMAT_OWN: str = "B"
FORMAT_MIXED: str = "C"

# GlucoBench / literature MAE bands for 60-minute horizon (mg/dL)
SIMPLE_BASELINE_MAE_RANGE: tuple[float, float] = (12.0, 20.0)
DEEP_LEARNING_MAE_RANGE: tuple[float, float] = (11.0, 17.0)
PERSONALIZED_MAE_RANGE: tuple[float, float] = (15.0, 17.0)

# Clinically meaningful MAE difference (study design §8.2)
CLINICALLY_MEANINGFUL_MAE_DIFF: float = 3.5

# Duration bins for exploratory H3/H4 plots
DIABETES_DURATION_BINS: list[tuple[str, float, float]] = [
    ("<1 year", 0.0, 1.0),
    ("1–5 years", 1.0, 5.0),
    ("5–10 years", 5.0, 10.0),
    (">10 years", 10.0, float("inf")),
]
CGM_DURATION_BINS: list[tuple[str, float, float]] = [
    ("<1 year", 0.0, 1.0),
    ("1–2 years", 1.0, 2.0),
    ("2–5 years", 2.0, 5.0),
    (">5 years", 5.0, float("inf")),
]

# Soft anomaly thresholds (demographics / metrics)
MAX_PLAUSIBLE_CGM_YEARS: float = 30.0
MAX_PLAUSIBLE_DIABETES_YEARS: float = 80.0
MAX_PLAUSIBLE_AGE: float = 120.0
MAE_IQR_OUTLIER_K: float = 1.5

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_RAW_CSV: Path = REPO_ROOT / "data" / "raw" / "prediction_statistics.csv"
DEFAULT_FIXTURE_CSV: Path = REPO_ROOT / "data" / "fixtures" / "synthetic_prediction_statistics.csv"
# Single canonical output tree for CLI and notebook (reports always under output/reports)
DEFAULT_OUTPUT_DIR: Path = REPO_ROOT / "output"
DEFAULT_REPORTS_DIR: Path = DEFAULT_OUTPUT_DIR / "reports"
DEFAULT_FIGURES_DIR: Path = DEFAULT_OUTPUT_DIR / "figures"
