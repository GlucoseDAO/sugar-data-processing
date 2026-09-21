"""Rebuild the person-level MAE table using one model's round errors."""

from __future__ import annotations

import polars as pl
from eliot import start_action

from sugar_data_processing.config import (
    FORMAT_GENERIC,
    FORMAT_MIXED,
    FORMAT_OWN,
    MIN_GENERIC_SEGMENTS,
    MIN_OWN_SEGMENTS,
)
from sugar_data_processing.statistics.hypotheses import run_all_hypotheses
from sugar_data_processing.statistics.tests import paired_comparison


_MAE_COLS: tuple[str, ...] = (
    "mae_generic",
    "mae_own",
    "mae_overall",
    "mae_primary",
    "mae_format_a",
    "mae_format_b",
    "mae_format_c",
    "mae_same_trait",
    "mae_opposite_trait",
    "rmse_generic",
    "rmse_own",
    "rmse_overall",
    "rmse_primary",
    "mape_generic",
    "mape_own",
    "mape_overall",
    "mae_diff_generic_minus_own",
    "mae_diff_same_minus_opposite",
)


def overlay_model_mae(
    participants: pl.DataFrame,
    scored: pl.DataFrame,
    model_name: str,
) -> pl.DataFrame:
    """Copy the human person table, replacing MAE columns with one model's scores."""
    with start_action(action_type="ai.overlay_model_mae", model=model_name) as action:
        model_rows = scored.filter(pl.col("model_name") == model_name)
        if model_rows.height == 0:
            action.log(message_type="warning", reason="no_scores")
            return participants
        rounds = (
            model_rows.group_by(["study_id", "run_id", "round_number", "source", "format"])
            .agg(pl.col("model_abs_error").mean().alias("mae"))
        )
        if "is_opposite_trait" in model_rows.columns:
            trait = (
                model_rows.group_by(["study_id", "run_id", "round_number"])
                .agg(pl.col("is_opposite_trait").last())
            )
            rounds = rounds.join(trait, on=["study_id", "run_id", "round_number"], how="left")
        else:
            rounds = rounds.with_columns(pl.lit(False).alias("is_opposite_trait"))

        generic = _source_mae(rounds, "generic", "generic")
        own = _source_mae(rounds, "own", "own")
        overall = rounds.group_by("study_id").agg(pl.col("mae").mean().alias("mae_overall"))
        mae_a = _format_mae(rounds, FORMAT_GENERIC, "a")
        mae_b = _format_mae(rounds, FORMAT_OWN, "b")
        mae_c = _format_mae(rounds, FORMAT_MIXED, "c")
        same = _trait_mae(rounds, opposite=False, prefix="same_trait")
        opposite = _trait_mae(rounds, opposite=True, prefix="opposite_trait")

        keep = [c for c in participants.columns if c not in _MAE_COLS]
        overlaid = (
            participants.select(keep)
            .join(generic, on="study_id", how="left")
            .join(own, on="study_id", how="left")
            .join(overall, on="study_id", how="left")
            .join(mae_a, on="study_id", how="left")
            .join(mae_b, on="study_id", how="left")
            .join(mae_c, on="study_id", how="left")
            .join(same, on="study_id", how="left")
            .join(opposite, on="study_id", how="left")
            .with_columns(
                pl.coalesce([pl.col("mae_generic"), pl.col("mae_overall")]).alias("mae_primary"),
                (pl.col("mae_generic") - pl.col("mae_own")).alias("mae_diff_generic_minus_own"),
                (pl.col("mae_same_trait") - pl.col("mae_opposite_trait")).alias(
                    "mae_diff_same_minus_opposite"
                ),
            )
        )
        action.log(message_type="info", n_people=overlaid.height, n_rounds=rounds.height)
        return overlaid


def _source_mae(rounds: pl.DataFrame, source: str, prefix: str) -> pl.DataFrame:
    return (
        rounds.filter(pl.col("source") == source)
        .group_by("study_id")
        .agg(pl.col("mae").mean().alias(f"mae_{prefix}"))
    )


def _format_mae(rounds: pl.DataFrame, format_code: str, prefix: str) -> pl.DataFrame:
    return (
        rounds.filter(pl.col("format") == format_code)
        .group_by("study_id")
        .agg(pl.col("mae").mean().alias(f"mae_format_{prefix}"))
    )


def _trait_mae(rounds: pl.DataFrame, *, opposite: bool, prefix: str) -> pl.DataFrame:
    return (
        rounds.filter(pl.col("is_opposite_trait") == opposite)
        .group_by("study_id")
        .agg(pl.col("mae").mean().alias(f"mae_{prefix}"))
    )


def h6_human_vs_models(
    human: pl.DataFrame,
    scored: pl.DataFrame,
) -> dict[str, object]:
    """Paired person-level MAE: human vs each ingested model on the same people."""
    models = sorted(str(m) for m in scored["model_name"].unique().to_list()) if scored.height else []
    comparisons: list[dict[str, object]] = []
    notes: list[str] = []
    for name in models:
        ai = overlay_model_mae(human, scored, name)
        paired = human.join(
            ai.select(["study_id", "mae_primary"]).rename({"mae_primary": "mae_model"}),
            on="study_id",
            how="inner",
        ).filter(pl.col("mae_primary").is_not_null() & pl.col("mae_model").is_not_null())
        if paired.height < 2:
            notes.append(f"H6 skipped for {name}: need ≥2 people with both scores.")
            continue
        result = paired_comparison(
            paired["mae_primary"].to_numpy().astype(float),
            paired["mae_model"].to_numpy().astype(float),
            hypothesis="H6",
            label_a="human",
            label_b=name,
        )
        payload = result.to_dict()
        payload["model_name"] = name
        payload["human_mean_mae"] = float(paired["mae_primary"].mean() or 0.0)
        payload["model_mean_mae"] = float(paired["mae_model"].mean() or 0.0)
        comparisons.append(payload)
    return {
        "status": "scored" if comparisons else "empty",
        "n_models": len(comparisons),
        "comparisons": comparisons,
        "notes": notes,
        "lookback": (
            "Models were fed only the 3-hour game window (24 visible CGM points, "
            "left-padded to 128). They did not see extra pre-game history."
        ),
    }


def ai_suite_for_model(human: pl.DataFrame, scored: pl.DataFrame, model_name: str):
    """Rerun H1–H5 on the AI person table for one model."""
    overlaid = overlay_model_mae(human, scored, model_name)
    eligible = overlaid.filter(
        (pl.col("n_rounds_generic") >= MIN_GENERIC_SEGMENTS)
        | pl.col("mae_primary").is_not_null()
    )
    return overlaid, run_all_hypotheses(eligible if eligible.height else overlaid)
