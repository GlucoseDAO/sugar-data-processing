"""Plain-language descriptions of study-design hypotheses (H1–H6)."""

from __future__ import annotations

from typing import TypedDict


class HypothesisInfo(TypedDict):
    code: str
    title: str
    question: str
    groups_or_predictors: str
    outcome: str
    method: str
    section: str


HYPOTHESES: dict[str, HypothesisInfo] = {
    "h1": {
        "code": "H1",
        "title": "Diabetes status and prediction accuracy",
        "question": (
            "Do people with diabetes (PwD) predict next-hour glucose more accurately "
            "than people without diabetes?"
        ),
        "groups_or_predictors": "PwD vs non-PwD (`diabetic`)",
        "outcome": "Person-level MAE in mg/dL (lower is better)",
        "method": (
            "Shapiro–Wilk normality check, then independent t-test if normal, "
            "otherwise Mann–Whitney U. α = 0.05."
        ),
        "section": "§7.3 primary",
    },
    "h2": {
        "code": "H2",
        "title": "CGM use and prediction accuracy",
        "question": (
            "Do continuous glucose monitor (CGM) users predict next-hour glucose "
            "more accurately than people who do not use CGM?"
        ),
        "groups_or_predictors": "CGM users vs non-CGM (`uses_cgm`)",
        "outcome": "Person-level MAE in mg/dL (lower is better)",
        "method": (
            "Same path as H1: Shapiro–Wilk → independent t-test or Mann–Whitney U. "
            "α = 0.05."
        ),
        "section": "§7.3 primary",
    },
    "h3": {
        "code": "H3",
        "title": "Diabetes duration and prediction accuracy",
        "question": (
            "Among people with diabetes, is longer diabetes duration associated "
            "with better (lower) prediction MAE?"
        ),
        "groups_or_predictors": "Diabetes duration in years (PwD only)",
        "outcome": "Person-level MAE in mg/dL",
        "method": (
            "Pearson correlation if assumptions hold, otherwise Spearman; "
            "plus exploratory linear vs log fits. α = 0.05."
        ),
        "section": "§7.4 secondary",
    },
    "h4": {
        "code": "H4",
        "title": "CGM experience and prediction accuracy",
        "question": (
            "Among CGM users, is longer CGM experience associated with better "
            "(lower) prediction MAE?"
        ),
        "groups_or_predictors": "CGM experience in years (CGM users only)",
        "outcome": "Person-level MAE in mg/dL",
        "method": (
            "Same approach as H3: Pearson or Spearman plus exploratory linear/log fits. "
            "α = 0.05."
        ),
        "section": "§7.4 secondary",
    },
    "h5": {
        "code": "H5",
        "title": "Own data vs generic example data",
        "question": (
            "Within the same person, is prediction more accurate on their own CGM "
            "segments than on generic/example segments?"
        ),
        "groups_or_predictors": "Paired own-data MAE vs generic-data MAE",
        "outcome": (
            "Difference (generic MAE − own MAE); positive means better accuracy on own data"
        ),
        "method": (
            "Shapiro–Wilk on paired differences → paired t-test or Wilcoxon signed-rank. "
            "α = 0.05."
        ),
        "section": "§7.4 secondary",
    },
    "h6": {
        "code": "H6",
        "title": "Humans vs computational baseline models",
        "question": (
            "How does human next-hour MAE compare with persistence, linear-extrapolation, "
            "and ARIMA baselines on the same segments?"
        ),
        "groups_or_predictors": "Human MAE vs model MAE (same segments)",
        "outcome": "Person- or segment-level MAE in mg/dL",
        "method": "Deferred until baselines are implemented in sugar-sugar.",
        "section": "§7.4 deferred",
    },
}


def hypothesis_blurb(key: str) -> str:
    """Short markdown block explaining one hypothesis."""
    info = HYPOTHESES[key]
    return (
        f"**{info['code']} — {info['title']}** ({info['section']})\n\n"
        f"- **Question:** {info['question']}\n"
        f"- **Compared / predictor:** {info['groups_or_predictors']}\n"
        f"- **Outcome:** {info['outcome']}\n"
        f"- **Test plan:** {info['method']}"
    )


def hypothesis_heading(key: str) -> str:
    """Section heading with code + plain title."""
    info = HYPOTHESES[key]
    return f"{info['code']} — {info['title']}"
