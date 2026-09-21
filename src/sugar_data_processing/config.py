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

# Player × CGM cohort labels (unique people, not runs)
COHORT_DIABETIC_CGM: str = "diabetic_cgm"
COHORT_DIABETIC_NON_CGM: str = "diabetic_non_cgm"
COHORT_NONDIABETIC_CGM: str = "nondiabetic_cgm"
COHORT_NONDIABETIC_NON_CGM: str = "nondiabetic_non_cgm"
COHORT_UNKNOWN: str = "unknown"

COHORT_LABELS: dict[str, str] = {
    COHORT_DIABETIC_CGM: "Diabetic with CGM",
    COHORT_DIABETIC_NON_CGM: "Diabetic without CGM",
    COHORT_NONDIABETIC_CGM: "Non-diabetic with CGM",
    COHORT_NONDIABETIC_NON_CGM: "Non-diabetic without CGM",
    COHORT_UNKNOWN: "Status unknown",
}

# Predicted-trace class (the data being guessed, not the player's own status)
DATA_CLASS_DIABETIC: str = "diabetic"
DATA_CLASS_NONDIABETIC: str = "nondiabetic"

# Player diabetes trait (same vocabulary as data_class so they can be compared)
PLAYER_TRAIT_DIABETIC: str = "diabetic"
PLAYER_TRAIT_NONDIABETIC: str = "nondiabetic"
PLAYER_TRAIT_UNKNOWN: str = "unknown"

TRAIT_LABELS: dict[str, str] = {
    PLAYER_TRAIT_DIABETIC: "Diabetic",
    PLAYER_TRAIT_NONDIABETIC: "Non-diabetic",
    PLAYER_TRAIT_UNKNOWN: "Unknown",
}

# Challenge the unknown: player opted into the opposite-corpus mix (formats A/C)
CHALLENGE_UNKNOWN_LABEL: str = "Challenge the unknown"

# AI evaluation timing. Current study games are scored after the fact.
EVALUATION_MODE_POST_FACTUM: str = "post_factum"
EVALUATION_MODE_IN_PLACE: str = "in_place"
EVALUATION_MODE_LABELS: dict[str, str] = {
    EVALUATION_MODE_POST_FACTUM: "Post factum (replay saved games)",
    EVALUATION_MODE_IN_PLACE: "In place (scored during the game)",
}

# One merged deliverable (human + AI). Old split filenames are deleted on write.
REPORT_MD: str = "analysis_report.md"
REPORT_JSON: str = "analysis_report.json"
EXPLORER_HTML: str = "explorer.html"
MILESTONE_HTML: str = "milestone.html"
STALE_REPORT_FILES: tuple[str, ...] = (
    "human_analysis_report.md",
    "human_analysis_report.json",
    "human_explorer.html",
    "ai_analysis_report.md",
    "ai_analysis_report.json",
    "ai_explorer.html",
    "study_analysis_report.md",
    "study_analysis_report.json",
    "study_explorer.html",
)

# Game window the human saw: 36 points = 3 hours at 5-minute sampling
# (24 visible context + 12 hidden forecast points).
GAME_WINDOW_POINTS: int = 36
VISIBLE_CONTEXT_POINTS: int = 24
FORECAST_HORIZON_POINTS: int = 12
SAMPLE_MINUTES: int = 5
MODEL_INPUT_SIZE: int = 128
PRIMARY_MODEL_NAME: str = "persistence"

# GlucoBench / literature MAE bands for 60-minute horizon (mg/dL)
SIMPLE_BASELINE_MAE_RANGE: tuple[float, float] = (12.0, 20.0)
DEEP_LEARNING_MAE_RANGE: tuple[float, float] = (11.0, 17.0)
PERSONALIZED_MAE_RANGE: tuple[float, float] = (15.0, 17.0)

# Clinically meaningful MAE difference (study design §8.2)
CLINICALLY_MEANINGFUL_MAE_DIFF: float = 3.5

# Duration bins for exploratory H3/H4 plots
DIABETES_DURATION_BINS: list[tuple[str, float, float]] = [
    ("<1 year", 0.0, 1.0),
    ("1-5 years", 1.0, 5.0),
    ("5-10 years", 5.0, 10.0),
    (">10 years", 10.0, float("inf")),
]
CGM_DURATION_BINS: list[tuple[str, float, float]] = [
    ("<1 year", 0.0, 1.0),
    ("1-2 years", 1.0, 2.0),
    ("2-5 years", 2.0, 5.0),
    (">5 years", 5.0, float("inf")),
]

# Soft anomaly thresholds (demographics / metrics)
MAX_PLAUSIBLE_CGM_YEARS: float = 30.0
MAX_PLAUSIBLE_DIABETES_YEARS: float = 80.0
MAX_PLAUSIBLE_AGE: float = 120.0
MAE_IQR_OUTLIER_K: float = 1.5

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_SUGAR_SUGAR_ROOT: Path = REPO_ROOT.parent / "sugar-sugar"
DEFAULT_FORECASTING_ROOT: Path = (
    REPO_ROOT.parent.parent / "glucose-forecasting" / "glucose-forecasting"
)
FORECASTING_ROOT_CANDIDATES: tuple[Path, ...] = (
    DEFAULT_FORECASTING_ROOT,
    REPO_ROOT.parent.parent / "g-forecasting2" / "glucose-forecasting",
    REPO_ROOT.parent.parent / "glucose-forecasting",
)


def find_forecasting_root(explicit: Path | None = None) -> Path | None:
    """First sibling checkout that contains the GluMind sources."""
    ordered: list[Path] = []
    if explicit is not None:
        ordered.append(Path(explicit))
    ordered.extend(FORECASTING_ROOT_CANDIDATES)
    seen: set[Path] = set()
    for path in ordered:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "scripts" / "glumind" / "glumind_model.py").exists():
            return resolved
    return None


DEFAULT_RAW_CSV: Path = REPO_ROOT / "data" / "raw" / "prediction_statistics.csv"
DEFAULT_FIXTURE_CSV: Path = REPO_ROOT / "data" / "fixtures" / "synthetic_prediction_statistics.csv"
DEFAULT_SIBLING_STATS: Path = (
    REPO_ROOT.parent / "sugar-sugar" / "data" / "input" / "prediction_statistics.csv"
)
# Single canonical output tree for CLI and notebook (reports always under output/reports)
DEFAULT_OUTPUT_DIR: Path = REPO_ROOT / "output"
DEFAULT_REPORTS_DIR: Path = DEFAULT_OUTPUT_DIR / "reports"
DEFAULT_FIGURES_DIR: Path = DEFAULT_OUTPUT_DIR / "figures"
DEFAULT_PROCESSED_DIR: Path = REPO_ROOT / "data" / "processed"
DEFAULT_AI_DIR: Path = DEFAULT_PROCESSED_DIR / "ai"
