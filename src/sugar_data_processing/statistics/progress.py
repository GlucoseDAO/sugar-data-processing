"""Live study progress for the dashboard front page.

Recruitment is counted against the corrected plan (docs/analysis-plan.md,
24 October 2026) and sample-size needs are recomputed from the observed
between-person SD on every build. The planned effect stays fixed: powering to
a favourable observed difference would only chase significance.
"""

from __future__ import annotations

from math import ceil, isfinite, sqrt
from typing import Any

import numpy as np
import polars as pl
from eliot import start_action

from sugar_data_processing.config import (
    ALPHA,
    CLINICALLY_MEANINGFUL_MAE_DIFF,
    COHORT_DIABETIC_CGM,
    COHORT_DIABETIC_NON_CGM,
    COHORT_LABELS,
    COHORT_NONDIABETIC_CGM,
    COHORT_NONDIABETIC_NON_CGM,
    MIN_GENERIC_SEGMENTS,
    PRIMARY_ALPHA,
)
from sugar_data_processing.gathering.participants import (
    h5_paired_population,
    own_analysis_population,
    primary_analysis_population,
)
from sugar_data_processing.statistics.power import (
    correlation_power,
    independent_t_power,
    paired_t_power,
    required_correlation_n,
    required_independent_n,
    required_paired_n,
)
from sugar_data_processing.statistics.tests import independent_group_comparison, paired_comparison

# Guidebook §8.1 and the corrected plan in docs/analysis-plan.md.
ORIGINAL_TARGET_ELIGIBLE: int = 200
MINIMUM_VIABLE_ELIGIBLE: int = 150
REVISED_TARGET_ELIGIBLE: int = 320
PLANNING_SD: float = 10.0
TARGET_POWER: float = 0.80
CORRELATION_TARGET_R: float = 0.25
PAIRED_EFFECT: float = 3.0
PAIRED_PLANNING_SD: float = 6.0
PAIRED_OPERATIONAL_TARGET: int = 70
CELL_TARGETS: dict[str, int] = {
    COHORT_DIABETIC_CGM: 100,
    COHORT_DIABETIC_NON_CGM: 60,
    COHORT_NONDIABETIC_CGM: 60,
    COHORT_NONDIABETIC_NON_CGM: 100,
}
CURVE_GROUP_SIZES: list[int] = list(range(5, 255, 5))
# The app gives players with diabetes mostly diabetic traces, which are harder,
# so H1 is also checked on one kind of trace at a time. The within-person
# difficulty check uses the 24 October audit's exploratory 3-round threshold.
TRACE_CLASSES: dict[str, str] = {
    "diabetic_data": "diabetic traces",
    "nondiabetic_data": "non-diabetic traces",
}
MIN_CLASS_ROUNDS: int = MIN_GENERIC_SEGMENTS
MIN_PAIRED_CLASS_ROUNDS: int = 3
MODEL_LABELS: dict[str, str] = {
    "persistence": "Persistence",
    "linear": "Linear trend",
    "glumind": "GluMind",
    "sugar_one": "SugarOne",
    "citras": "CITRAS-FM W38.7",
    "citras_w38_2": "CITRAS-FM W38.2",
    "kronlstm": "KronLSTM R.80",
}
TRAINED_MODELS: frozenset[str] = frozenset(
    {"glumind", "sugar_one", "citras", "citras_w38_2", "kronlstm"}
)
MIN_REPLAYED_PLAYERS: int = 30

VERDICT_LABELS: dict[str, str] = {
    "supported": "Supported",
    "exploratory": "Promising, exploratory",
    "opposite": "Against our assumptions",
    "significant": "Useful finding",
    "not_yet": "Not enough data yet",
    "too_few": "Too few people so far",
    "context": "Useful context",
}
# Dashboard colours: green = supported or useful, yellow = not enough data,
# red = a clear result against the study's assumptions.
VERDICT_TONES: dict[str, str] = {
    "supported": "good",
    "exploratory": "good",
    "significant": "good",
    "context": "good",
    "not_yet": "warn",
    "too_few": "warn",
    "opposite": "bad",
}


def study_progress(
    participants: pl.DataFrame,
    suite: dict[str, Any] | None,
    *,
    h6: dict[str, Any] | None = None,
    benchmarks: dict[str, Any] | None = None,
    linked_identities: bool = False,
) -> dict[str, Any]:
    """Recruitment, power at the planned effect, and plain-language findings.

    ``linked_identities`` says whether runs were merged by pseudonymous person
    key, so the page can say whether counts are people or browser IDs.
    """
    with start_action(action_type="statistics.study_progress") as action:
        suite = suite or {}
        primary = primary_analysis_population(participants)
        n_people = participants.height
        n_eligible = primary.height
        cells = [
            {
                "key": key,
                "label": COHORT_LABELS[key],
                "now": int(primary.filter(pl.col("cohort_category") == key).height)
                if "cohort_category" in primary.columns
                else 0,
                "target": target,
            }
            for key, target in CELL_TARGETS.items()
        ]
        remaining = sum(max(0, cell["target"] - cell["now"]) for cell in cells)
        rate = n_eligible / n_people if n_people else 0.0
        h1 = _group_power("H1", primary, "diabetic", ("with diabetes", "without diabetes"), suite.get("h1"))
        matched = _trace_matched(participants)
        h2 = _group_power("H2", primary, "uses_cgm", ("CGM users", "non-users"), suite.get("h2"))
        rows = [
            h1,
            h2,
            _correlation_power("H3", "Years with diabetes vs accuracy", suite.get("h3")),
            _correlation_power("H4", "Years using a CGM vs accuracy", suite.get("h4")),
            _paired_power(participants, suite.get("h5")),
        ]
        payload = {
            "linked_identities": linked_identities,
            "model_labels": MODEL_LABELS,
            "n_people": n_people,
            "n_eligible": n_eligible,
            "n_own": own_analysis_population(participants).height,
            "n_paired": h5_paired_population(participants).height,
            "eligible_rate": rate,
            "targets": {
                "original": ORIGINAL_TARGET_ELIGIBLE,
                "minimum": MINIMUM_VIABLE_ELIGIBLE,
                "revised": REVISED_TARGET_ELIGIBLE,
            },
            "remaining_eligible": remaining,
            "remaining_people": ceil(remaining / rate) if rate > 0 else None,
            "cells": cells,
            "planned_effect": CLINICALLY_MEANINGFUL_MAE_DIFF,
            "planning_sd": PLANNING_SD,
            "primary_alpha": PRIMARY_ALPHA,
            "target_power": TARGET_POWER,
            "power": rows,
            "curves": _power_curves([h1, h2]),
            "trace_matched": matched,
            "design": _design_check(participants, primary),
            "findings": _findings(suite, rows, matched, h6 or {}, benchmarks or {}),
        }
        action.log(message_type="info", n_eligible=n_eligible, remaining=remaining)
        return payload


def _values(df: pl.DataFrame, mask: pl.Expr, column: str) -> np.ndarray:
    if column not in df.columns or df.height == 0:
        return np.asarray([], dtype=float)
    values = df.filter(mask & pl.col(column).is_not_null())[column].to_numpy().astype(float)
    return values[np.isfinite(values)]


def _pooled_sd(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2 or b.size < 2:
        return None
    pooled = ((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2)
    return float(sqrt(pooled)) if pooled > 0 else None


def _layer(result: dict[str, Any] | None, *names: str) -> dict[str, Any] | None:
    for name in names:
        layer = (result or {}).get(name)
        if isinstance(layer, dict):
            return layer
    return None


def _safe(fn: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except ValueError:
        return None


def _group_power(
    code: str,
    primary: pl.DataFrame,
    column: str,
    labels: tuple[str, str],
    result: dict[str, Any] | None,
) -> dict[str, Any]:
    a = _values(primary, pl.col(column).eq(True), "mae_generic")
    b = _values(primary, pl.col(column).eq(False), "mae_generic")
    sd = _pooled_sd(a, b)
    layer = _layer(result, "generic", "overall")
    effect = CLINICALLY_MEANINGFUL_MAE_DIFF
    observed = float(a.mean() - b.mean()) if a.size and b.size else None
    sized = _safe(required_independent_n, mean_difference=effect, sd=sd) if sd else None
    need_observed = None if sized is None else sized[0]
    need_plan = required_independent_n(mean_difference=effect, sd=PLANNING_SD)[0]
    need_if_real = (
        _safe(required_independent_n, mean_difference=observed, sd=sd)
        if sd and observed
        else None
    )
    target = need_observed or need_plan
    return {
        "code": code,
        "kind": "groups",
        "labels": list(labels),
        "n_a": int(a.size),
        "n_b": int(b.size),
        "n_now": int(a.size + b.size),
        "n_effective": 2.0 / (1.0 / a.size + 1.0 / b.size) if a.size and b.size else 0.0,
        "sd": sd,
        "mean_a": float(a.mean()) if a.size else None,
        "mean_b": float(b.mean()) if b.size else None,
        "observed": observed,
        "p_value": None if layer is None else layer.get("p_value"),
        "significant": None if layer is None else layer.get("significant"),
        "alpha": PRIMARY_ALPHA,
        "power_now": _safe(independent_t_power, a.size, b.size, mean_difference=effect, sd=sd, alpha=PRIMARY_ALPHA)
        if sd
        else None,
        "need_per_group": need_observed,
        "need_per_group_plan": need_plan,
        "need_if_real_per_group": None if need_if_real is None else need_if_real[0],
        "still_needed": max(0, target - int(a.size)) + max(0, target - int(b.size)),
    }


def _by_trace_class(participants: pl.DataFrame, group: str, code: str) -> list[dict[str, Any]]:
    """Compare the two levels of ``group`` separately on each kind of public trace."""
    by_class: list[dict[str, Any]] = []
    for key, label in TRACE_CLASSES.items():
        column, count = f"mae_generic_{key}", f"n_rounds_generic_{key}"
        if column not in participants.columns or group not in participants.columns:
            continue
        enough = participants.filter(pl.col(count) >= MIN_CLASS_ROUNDS)
        a = _values(enough, pl.col(group).eq(True), column)
        b = _values(enough, pl.col(group).eq(False), column)
        row: dict[str, Any] = {
            "key": key,
            "label": label,
            "n_a": int(a.size),
            "n_b": int(b.size),
            "mean_a": float(a.mean()) if a.size else None,
            "mean_b": float(b.mean()) if b.size else None,
            "p_value": None,
            "significant": None,
        }
        if a.size >= 3 and b.size >= 3:
            result = independent_group_comparison(
                a, b, hypothesis=f"{code}:{key}", label_a="yes", label_b="no", alpha=PRIMARY_ALPHA
            )
            row.update(p_value=result.p_value, significant=result.significant, test_used=result.test_used)
        by_class.append(row)
    return by_class


def _trace_matched(participants: pl.DataFrame) -> dict[str, Any]:
    """H1 and H2 on one kind of trace at a time, plus the same players on both kinds."""
    by_class = _by_trace_class(participants, "diabetic", "H1")
    within: dict[str, Any] | None = None
    needed = {"mae_generic_diabetic_data", "mae_generic_nondiabetic_data"}
    if needed <= set(participants.columns):
        both = participants.filter(
            (pl.col("n_rounds_generic_diabetic_data") >= MIN_PAIRED_CLASS_ROUNDS)
            & (pl.col("n_rounds_generic_nondiabetic_data") >= MIN_PAIRED_CLASS_ROUNDS)
        )
        harder = both["mae_generic_diabetic_data"].to_numpy().astype(float)
        easier = both["mae_generic_nondiabetic_data"].to_numpy().astype(float)
        within = {
            "n": both.height,
            "mean_diabetic": float(harder.mean()) if both.height else None,
            "mean_nondiabetic": float(easier.mean()) if both.height else None,
            "n_harder_on_diabetic": int((harder > easier).sum()),
            "p_value": None,
            "significant": None,
        }
        if both.height >= 3:
            result = paired_comparison(
                harder,
                easier,
                hypothesis="trace difficulty",
                label_a="diabetic traces",
                label_b="non-diabetic traces",
                alpha=ALPHA,
            )
            within.update(p_value=result.p_value, significant=result.significant, test_used=result.test_used)
    return {
        "min_rounds": MIN_CLASS_ROUNDS,
        "min_paired_rounds": MIN_PAIRED_CLASS_ROUNDS,
        "by_class": by_class,
        "cgm_by_class": _by_trace_class(participants, "uses_cgm", "H2"),
        "within": within,
    }


def _design_check(participants: pl.DataFrame, primary: pl.DataFrame) -> dict[str, Any] | None:
    """Who saw which kind of public trace under the game's default assignment."""
    counts = ("n_rounds_generic_diabetic_data", "n_rounds_generic_nondiabetic_data")
    if not {*counts, "diabetic"} <= set(participants.columns):
        return None
    cells: list[dict[str, Any]] = []
    for flag, group in ((True, "Players with diabetes"), (False, "Players without diabetes")):
        people = participants.filter(pl.col("diabetic").eq(flag))
        for key, label in TRACE_CLASSES.items():
            rounds = people[f"n_rounds_generic_{key}"]
            cells.append(
                {
                    "group": group,
                    "trace": key,
                    "trace_label": label,
                    "is_default": (key == "diabetic_data") == flag,
                    "players": people.height,
                    "rounds": int(rounds.sum()),
                    "players_any": int((rounds > 0).sum()),
                    "players_enough": int((rounds >= MIN_CLASS_ROUNDS).sum()),
                }
            )
    saw_both = participants.filter((pl.col(counts[0]) > 0) & (pl.col(counts[1]) > 0))
    chose = (
        int(saw_both.filter(pl.col("played_challenge_unknown")).height)
        if "played_challenge_unknown" in saw_both.columns
        else None
    )
    off_default = primary.filter(
        (pl.col("diabetic").eq(True) & (pl.col(counts[1]) >= MIN_CLASS_ROUNDS))
        | (pl.col("diabetic").eq(False) & (pl.col(counts[0]) >= MIN_CLASS_ROUNDS))
    )
    return {
        "cells": cells,
        "min_rounds": MIN_CLASS_ROUNDS,
        "n_players": participants.height,
        "n_saw_both": saw_both.height,
        "n_saw_both_challenge": chose,
        "n_eligible": primary.height,
        "n_eligible_off_default": off_default.height,
    }


def _correlation_power(code: str, title: str, result: dict[str, Any] | None) -> dict[str, Any]:
    layer = _layer(result, "generic", "overall")
    n = int(layer.get("n") or 0) if layer else 0
    r = layer.get("coefficient") if layer else None
    need = required_correlation_n(CORRELATION_TARGET_R)
    real = abs(r) if r is not None and isfinite(r) else None
    return {
        "code": code,
        "kind": "correlation",
        "title": title,
        "n_now": n,
        "observed": r,
        "test_used": None if layer is None else layer.get("test_used"),
        "p_value": None if layer is None else layer.get("p_value"),
        "significant": None if layer is None else layer.get("significant"),
        "alpha": ALPHA,
        "power_now": _safe(correlation_power, n, CORRELATION_TARGET_R) if n >= 4 else None,
        "need_total": need,
        "need_if_real": _safe(required_correlation_n, real) if real else None,
        "still_needed": max(0, need - n),
    }


def _paired_power(participants: pl.DataFrame, result: dict[str, Any] | None) -> dict[str, Any]:
    paired = h5_paired_population(participants)
    n = paired.height
    diffs = (
        (paired["mae_generic"] - paired["mae_own"]).drop_nulls().to_numpy().astype(float)
        if n
        else np.asarray([], dtype=float)
    )
    sd = float(diffs.std(ddof=1)) if diffs.size >= 3 else None
    need = required_paired_n(mean_difference=PAIRED_EFFECT, difference_sd=PAIRED_PLANNING_SD)
    return {
        "code": "H5",
        "kind": "paired",
        "title": "Own data vs public data (same person)",
        "n_now": n,
        "sd": sd,
        "observed": None if not result else result.get("mean_diff"),
        "mean_generic": None if not result else result.get("mean_a"),
        "mean_own": None if not result else result.get("mean_b"),
        "p_value": None if not result else result.get("p_value"),
        "significant": None if not result else result.get("significant"),
        "alpha": ALPHA,
        "power_now": _safe(
            paired_t_power, n, mean_difference=PAIRED_EFFECT, difference_sd=PAIRED_PLANNING_SD
        )
        if n >= 2
        else None,
        "need_total": need,
        "need_operational": PAIRED_OPERATIONAL_TARGET,
        "still_needed": max(0, PAIRED_OPERATIONAL_TARGET - n),
    }


def _power_curves(rows: list[dict[str, Any]]) -> dict[str, Any]:
    effect = CLINICALLY_MEANINGFUL_MAE_DIFF
    series: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("sd"):
            continue
        series.append(
            {
                "label": f"{row['code']} · observed spread (SD {row['sd']:.1f})",
                "code": row["code"],
                "values": [
                    independent_t_power(n, n, mean_difference=effect, sd=row["sd"], alpha=PRIMARY_ALPHA)
                    for n in CURVE_GROUP_SIZES
                ],
            }
        )
    series.append(
        {
            "label": f"Planning assumption (SD {PLANNING_SD:.0f})",
            "code": "plan",
            "values": [
                independent_t_power(n, n, mean_difference=effect, sd=PLANNING_SD, alpha=PRIMARY_ALPHA)
                for n in CURVE_GROUP_SIZES
            ],
        }
    )
    now = [
        {"code": row["code"], "n_per_group": row["n_effective"], "power": row["power_now"]}
        for row in rows
        if row.get("power_now") is not None
    ]
    return {"group_sizes": CURVE_GROUP_SIZES, "series": series, "now": now}


def _p(p: float | None) -> str:
    if p is None or not isfinite(p):
        return "p not available"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.2f}" if p >= 0.1 else f"p = {p:.3f}"


def _mg(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f} mg/dL"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.0f}%"


def _names(comparisons: list[dict[str, Any]]) -> str:
    labels = [_model_label(c) for c in comparisons]
    return labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1]


def _model_label(comparison: dict[str, Any]) -> str:
    name = str(comparison.get("model_name") or "")
    return MODEL_LABELS.get(name, name)


def _verdict(significant: Any, expected_sign: float, observed: float | None) -> str:
    if significant is None or observed is None:
        return "too_few"
    if not significant:
        return "not_yet"
    return "supported" if observed * expected_sign > 0 else "opposite"


def _card(code: str, question: str, verdict: str, headline: str, detail: str) -> dict[str, Any]:
    return {
        "code": code,
        "question": question,
        "verdict": verdict,
        "verdict_label": VERDICT_LABELS[verdict],
        "tone": VERDICT_TONES[verdict],
        "headline": headline,
        "detail": detail,
    }


def _findings(
    suite: dict[str, Any],
    rows: list[dict[str, Any]],
    matched: dict[str, Any],
    h6: dict[str, Any],
    benchmarks: dict[str, Any],
) -> list[dict[str, Any]]:
    by_code = {row["code"]: row for row in rows}
    cards: list[dict[str, Any]] = []

    h1 = by_code["H1"]
    question = "Do people with diabetes predict better?"
    classes = {row["key"]: row for row in matched["by_class"]}
    fair = classes.get("diabetic_data")
    raw = (
        f"The raw comparison over all rounds ({_mg(h1['mean_a'])} vs {_mg(h1['mean_b'])}, "
        f"{_p(h1['p_value'])}) is misleading: players with diabetes were mostly given diabetic traces, "
        "which are harder for everyone."
        if h1["mean_a"] is not None
        else ""
    )
    if not fair or fair["p_value"] is None:
        cards.append(_card(
            "H1", question, "too_few",
            "Not enough players of both groups on the same kind of trace yet.",
            raw,
        ))
    else:
        verdict = _verdict(fair["significant"], -1.0, fair["mean_a"] - fair["mean_b"])
        headline = (
            f"On the same diabetic traces: {_mg(fair['mean_a'])} with diabetes vs "
            f"{_mg(fair['mean_b'])} without."
        )
        if verdict == "not_yet":
            headline = "No clear difference on the same data. " + headline
        other = classes.get("nondiabetic_data")
        other_text = (
            f" On non-diabetic traces: {_mg(other['mean_a'])} vs {_mg(other['mean_b'])} "
            f"({other['n_a']} vs {other['n_b']} players, {_p(other['p_value'])})."
            if other and other["p_value"] is not None
            else " Too few players with diabetes have played enough non-diabetic traces to compare them there."
        )
        cards.append(_card(
            "H1", question, verdict, headline,
            f"{fair['n_a']} vs {fair['n_b']} players with at least {matched['min_rounds']} rounds on "
            f"diabetic traces, {_p(fair['p_value'])} (threshold 0.025).{other_text} {raw}",
        ))

    within = matched.get("within") or {}
    if within.get("p_value") is not None:
        harder = (within["mean_diabetic"] or 0) > (within["mean_nondiabetic"] or 0)
        verdict = "significant" if within["significant"] else "not_yet"
        headline = (
            f"The same players erred more on diabetic traces: {_mg(within['mean_diabetic'])} vs "
            f"{_mg(within['mean_nondiabetic'])}."
            if harder
            else f"The same players: {_mg(within['mean_diabetic'])} on diabetic vs "
            f"{_mg(within['mean_nondiabetic'])} on non-diabetic traces."
        )
        cards.append(_card(
            "Traces", "Are diabetic glucose traces harder to predict?", verdict, headline,
            f"{within['n_harder_on_diabetic']} of {within['n']} players who predicted both kinds "
            f"(at least {matched['min_paired_rounds']} rounds each) did worse on diabetic traces, "
            f"{_p(within['p_value'])}. Exploratory: the 3-round threshold was set after the first look "
            "at the data. Any group comparison must therefore use the same kind of trace.",
        ))

    h2 = by_code["H2"]
    verdict = _verdict(h2["significant"], -1.0, h2["observed"])
    if verdict == "too_few":
        cards.append(_card("H2", "Do CGM users predict better?", verdict,
                           "Not enough eligible people in both groups yet.", ""))
    else:
        detail = f"{h2['n_a']} vs {h2['n_b']} people over all rounds, {_p(h2['p_value'])} (threshold 0.025)."
        same = [
            f"on {row['label']} {_mg(row['mean_a'])} vs {_mg(row['mean_b'])} "
            f"({row['n_a']} vs {row['n_b']}, {_p(row['p_value'])})"
            for row in matched.get("cgm_by_class", [])
            if row["p_value"] is not None
        ]
        if same:
            detail += (
                " On the same kind of trace (exploratory, CGM users vs non-users): " + "; ".join(same) + "."
            )
        if verdict == "not_yet" and h2["power_now"] is not None:
            detail += (
                f" With today's group sizes the main test would detect a real "
                f"{CLINICALLY_MEANINGFUL_MAE_DIFF:g} mg/dL difference only {_pct(h2['power_now'])} of the time."
            )
        headline = (
            f"No clear difference yet: {_mg(h2['mean_a'])} for CGM users vs {_mg(h2['mean_b'])} for non-users."
            if verdict == "not_yet"
            else f"CGM users {_mg(h2['mean_a'])} vs non-users {_mg(h2['mean_b'])}."
        )
        cards.append(_card("H2", "Do CGM users predict better?", verdict, headline, detail))
        for row in matched.get("cgm_by_class", []):
            if not row.get("significant"):
                continue
            fewer = row["mean_a"] < row["mean_b"]
            cards.append(_card(
                "H2 · same traces", "Do CGM users predict better on the same kind of trace?",
                "exploratory" if fewer else "opposite",
                f"On {row['label']}, CGM users erred {'less' if fewer else 'more'}: "
                f"{_mg(row['mean_a'])} vs {_mg(row['mean_b'])} for non-users.",
                f"{row['n_a']} vs {row['n_b']} players with at least {matched['min_rounds']} such rounds, "
                f"{_p(row['p_value'])} (threshold 0.025). Exploratory: this split was chosen after seeing "
                "the data, so it needs confirming in new players.",
            ))

    experience = (
        (by_code["H3"], "Do people with diabetes predict better the longer they have had it?", "years with diabetes"),
        (by_code["H4"], "Do CGM users predict better the longer they have used one?", "years of CGM use"),
    )
    for row, question, what in experience:
        if row["n_now"] < 3 or row["observed"] is None:
            cards.append(_card(
                row["code"], question, "too_few",
                f"Not enough people with a known duration yet ({row['n_now']}).", "",
            ))
            continue
        verdict = _verdict(row["significant"], -1.0, row["observed"])
        r = row["observed"]
        headline = {
            "not_yet": f"No clear link between {what} and accuracy (r = {r:+.2f}).",
            "supported": f"More {what} went with smaller errors (r = {r:+.2f}).",
            "opposite": f"More {what} went with larger errors (r = {r:+.2f}), the opposite of what we expected.",
        }[verdict]
        detail = (
            f"{row['n_now']} people, {_p(row['p_value'])} ({row['test_used'] or 'correlation'}). "
            f"A reliable answer needs about {row['need_total']} people in this group."
        )
        if verdict != "not_yet":
            detail += (
                " Fragile: far fewer people than planned, and secondary questions are not corrected "
                "for multiple testing."
            )
        cards.append(_card(row["code"], question, verdict, headline, detail))

    h5 = by_code["H5"]
    if h5["p_value"] is None:
        cards.append(_card(
            "H5", "Do people predict their own data better?", "too_few",
            f"Only {h5['n_now']} people have played enough rounds on both their own and public data.",
            f"Each person needs at least 6 rounds of each kind; the plan asks for about "
            f"{h5['need_operational']} such people.",
        ))
    else:
        verdict = _verdict(h5["significant"], 1.0, h5["observed"])
        cards.append(_card(
            "H5", "Do people predict their own data better?", verdict,
            f"Own data {_mg(h5['mean_own'])} vs public data {_mg(h5['mean_generic'])} "
            f"for the same {h5['n_now']} people.",
            f"{_p(h5['p_value'])}. Only people with at least 6 rounds of each kind count; "
            f"the plan asks for about {h5['need_operational']} of them.",
        ))

    comparisons = [c for c in (h6.get("comparisons") or []) if c.get("model_mean_mae") is not None]
    if comparisons:
        significant = [c for c in comparisons if c.get("significant")]
        human = comparisons[0].get("human_mean_mae")
        better = [c for c in comparisons if c["model_mean_mae"] < (c.get("human_mean_mae") or 0)]
        worse = [c for c in comparisons if c not in better]
        parts = []
        if better:
            parts.append(f"{_names(better)} predicted better than people")
        if worse:
            parts.append(f"people predicted better than {_names(worse)}")
        headline = "; ".join(parts) + f" (people: {_mg(human)})."
        detail = "; ".join(
            f"{_model_label(c)}: {_mg(c['model_mean_mae'])} ({_p(c.get('p_value'))})" for c in comparisons
        )
        replayed = max(int(c.get("n") or 0) for c in comparisons)
        detail += f". Same 3-hour windows, {replayed} players."
        if replayed < MIN_REPLAYED_PLAYERS:
            detail += (
                f" Preliminary: only {replayed} players could be replayed in this build, "
                "too few to generalise."
            )
        if not any(c["model_name"] in TRAINED_MODELS for c in comparisons):
            detail += (
                " Only the simple baselines were scored in this build; GlucoseDAO's trained models "
                "(CITRAS-FM, GluMind, KronLSTM) are not in this comparison yet."
            )
        # The study expects trained models to beat people: green when they do, red when
        # people beat them. Without trained scores, a clear baseline result is still useful.
        trained = [c for c in comparisons if c["model_name"] in TRAINED_MODELS]
        if replayed < MIN_REPLAYED_PLAYERS:
            verdict = "too_few"
        elif any(c.get("significant") and c not in better for c in trained):
            verdict = "opposite"
        elif any(c.get("significant") and c in better for c in trained):
            verdict = "supported"
        else:
            verdict = "significant" if significant else "not_yet"
        cards.append(_card(
            "H6", "How do people compare with algorithms?", verdict,
            headline[0].upper() + headline[1:],
            detail + " Persistence means 'glucose stays where it is'.",
        ))

    mean = benchmarks.get("human_mean_mae")
    simple = benchmarks.get("simple_baseline_mae_range")
    if mean is not None and simple:
        cards.append(_card(
            "Context", "How good are people at this?", "context",
            f"The average person is off by {_mg(mean)} over the next hour.",
            f"{benchmarks.get('pct_inside_simple_baseline_band', 0):.0f}% of people fall inside the "
            f"{simple[0]:g}–{simple[1]:g} mg/dL range of simple published models (GlucoBench).",
        ))
    return cards
