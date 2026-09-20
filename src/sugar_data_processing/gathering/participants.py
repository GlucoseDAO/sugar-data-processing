"""Build per-participant analysis tables (one MAE per person per condition)."""

from __future__ import annotations

from typing import Any

import polars as pl
from eliot import start_action

from sugar_data_processing.config import (
    COHORT_DIABETIC_CGM,
    COHORT_DIABETIC_NON_CGM,
    COHORT_NONDIABETIC_CGM,
    COHORT_NONDIABETIC_NON_CGM,
    COHORT_UNKNOWN,
    FORMAT_GENERIC,
    FORMAT_MIXED,
    FORMAT_OWN,
    MIN_GENERIC_SEGMENTS,
    MIN_OWN_SEGMENTS,
)
from sugar_data_processing.gathering.rounds import build_round_table


def _latest_runs(runs: pl.DataFrame) -> pl.DataFrame:
    """Keep the latest run per (study_id, format)."""
    if "timestamp" in runs.columns:
        return runs.sort("timestamp").group_by(["study_id", "format"]).last()
    return runs.group_by(["study_id", "format"]).last()


def _source_mae(rounds: pl.DataFrame, source: str, prefix: str) -> pl.DataFrame:
    return (
        rounds.filter(pl.col("source") == source)
        .group_by("study_id")
        .agg(
            pl.col("mae").mean().alias(f"mae_{prefix}"),
            pl.col("rmse").mean().alias(f"rmse_{prefix}"),
            pl.col("mape").mean().alias(f"mape_{prefix}"),
            pl.len().alias(f"n_rounds_{prefix}"),
        )
    )


def _format_mae(rounds: pl.DataFrame, format_code: str, prefix: str) -> pl.DataFrame:
    return (
        rounds.filter(pl.col("format") == format_code)
        .group_by("study_id")
        .agg(
            pl.col("mae").mean().alias(f"mae_format_{prefix}"),
            pl.len().alias(f"n_rounds_format_{prefix}"),
        )
    )


def _trait_side_mae(rounds: pl.DataFrame, opposite: bool, prefix: str) -> pl.DataFrame:
    empty = pl.DataFrame(
        schema={
            "study_id": pl.Utf8,
            f"mae_{prefix}": pl.Float64,
            f"n_rounds_{prefix}": pl.Int64,
        }
    )
    if "is_opposite_trait" not in rounds.columns or rounds.height == 0:
        return empty
    return (
        rounds.filter(pl.col("is_opposite_trait") == opposite)
        .group_by("study_id")
        .agg(
            pl.col("mae").mean().alias(f"mae_{prefix}"),
            pl.len().alias(f"n_rounds_{prefix}"),
        )
    )


def _cohort_category(diabetic: bool | None, uses_cgm: bool | None) -> str:
    if diabetic is True and uses_cgm is True:
        return COHORT_DIABETIC_CGM
    if diabetic is True and uses_cgm is False:
        return COHORT_DIABETIC_NON_CGM
    if diabetic is False and uses_cgm is True:
        return COHORT_NONDIABETIC_CGM
    if diabetic is False and uses_cgm is False:
        return COHORT_NONDIABETIC_NON_CGM
    return COHORT_UNKNOWN


def _run_history(runs: pl.DataFrame) -> pl.DataFrame:
    """Counts over every saved run (not only the latest per format)."""
    return runs.group_by("study_id").agg(
        pl.len().alias("n_runs"),
        pl.col("format").n_unique().alias("n_formats_played"),
        pl.col("format").unique().sort().alias("formats_played"),
    )


def build_participant_table(runs: pl.DataFrame) -> pl.DataFrame:
    """Aggregate to one row per ``study_id`` with person-level MAE scores.

    Person MAE is the mean of round MAEs (study design §7.3 — one summary
    accuracy score per person). Separate scores are kept for generic vs own
    data so H5 can use a within-person paired design.

    Also records how many *runs* a person saved (repeats), which formats they
    played, and the four diabetes × CGM cohort buckets.
    """
    with start_action(action_type="gathering.build_participant_table", n_runs=runs.height) as action:
        latest = _latest_runs(runs)
        rounds = build_round_table(latest)
        history = _run_history(runs)

        demo_cols = [
            c
            for c in (
                "study_id",
                "age",
                "gender",
                "uses_cgm",
                "cgm_duration_years",
                "diabetic",
                "diabetic_type",
                "diabetes_duration",
                "challenge_unknown",
                "generic_intervention",
            )
            if c in latest.columns
        ]
        demographics = latest.select(demo_cols).group_by("study_id").last()

        generic = _source_mae(rounds, "generic", "generic")
        own = _source_mae(rounds, "own", "own")
        overall = rounds.group_by("study_id").agg(
            pl.col("mae").mean().alias("mae_overall"),
            pl.col("rmse").mean().alias("rmse_overall"),
            pl.col("mape").mean().alias("mape_overall"),
            pl.len().alias("n_rounds_total"),
        )
        mae_a = _format_mae(rounds, FORMAT_GENERIC, "a")
        mae_b = _format_mae(rounds, FORMAT_OWN, "b")
        mae_c = _format_mae(rounds, FORMAT_MIXED, "c")
        same_trait = _trait_side_mae(rounds, opposite=False, prefix="same_trait")
        opposite_trait = _trait_side_mae(rounds, opposite=True, prefix="opposite_trait")
        challenge_history = _challenge_history(runs)
        trait_flags = _trait_flags(rounds)

        participants = (
            demographics.join(generic, on="study_id", how="left")
            .join(own, on="study_id", how="left")
            .join(overall, on="study_id", how="left")
            .join(history, on="study_id", how="left")
            .join(mae_a, on="study_id", how="left")
            .join(mae_b, on="study_id", how="left")
            .join(mae_c, on="study_id", how="left")
            .join(same_trait, on="study_id", how="left")
            .join(opposite_trait, on="study_id", how="left")
            .join(challenge_history, on="study_id", how="left")
            .join(trait_flags, on="study_id", how="left")
            .with_columns(
                pl.col("n_rounds_generic").fill_null(0).cast(pl.Int64),
                pl.col("n_rounds_own").fill_null(0).cast(pl.Int64),
                pl.col("n_rounds_total").fill_null(0).cast(pl.Int64),
                pl.col("n_runs").fill_null(0).cast(pl.Int64),
                pl.col("n_formats_played").fill_null(0).cast(pl.Int64),
                pl.col("n_rounds_format_a").fill_null(0).cast(pl.Int64),
                pl.col("n_rounds_format_b").fill_null(0).cast(pl.Int64),
                pl.col("n_rounds_format_c").fill_null(0).cast(pl.Int64),
                pl.col("n_rounds_same_trait").fill_null(0).cast(pl.Int64),
                pl.col("n_rounds_opposite_trait").fill_null(0).cast(pl.Int64),
                pl.col("played_challenge_unknown").fill_null(False),
                pl.col("played_opposite_trait").fill_null(False),
                pl.coalesce([pl.col("mae_generic"), pl.col("mae_overall")]).alias("mae_primary"),
                pl.coalesce([pl.col("rmse_generic"), pl.col("rmse_overall")]).alias("rmse_primary"),
            )
            .with_columns(
                (pl.col("n_rounds_generic") >= MIN_GENERIC_SEGMENTS).alias("eligible_primary"),
                (pl.col("n_rounds_own") >= MIN_OWN_SEGMENTS).alias("eligible_own"),
                (
                    (pl.col("n_rounds_generic") >= MIN_GENERIC_SEGMENTS)
                    & (pl.col("n_rounds_own") >= MIN_OWN_SEGMENTS)
                ).alias("eligible_h5"),
                (pl.col("mae_generic") - pl.col("mae_own")).alias("mae_diff_generic_minus_own"),
                (pl.col("mae_same_trait") - pl.col("mae_opposite_trait")).alias(
                    "mae_diff_same_minus_opposite"
                ),
                (pl.col("n_runs") > 1).alias("is_repeat_player"),
                (pl.col("n_formats_played") >= 3).alias("played_all_formats"),
            )
        )

        categories = [
            _cohort_category(row.get("diabetic"), row.get("uses_cgm"))
            for row in participants.iter_rows(named=True)
        ]
        participants = participants.with_columns(pl.Series("cohort_category", categories))
        if "diabetes_duration" in participants.columns:
            participants = participants.with_columns(
                (pl.col("diabetes_duration") * 12.0).alias("diabetes_duration_months")
            )
        if "cgm_duration_years" in participants.columns:
            participants = participants.with_columns(
                (pl.col("cgm_duration_years") * 12.0).alias("cgm_duration_months")
            )

        action.log(
            message_type="info",
            n_participants=participants.height,
            n_eligible_primary=int(participants.filter(pl.col("eligible_primary")).height),
            n_eligible_h5=int(participants.filter(pl.col("eligible_h5")).height),
            n_repeat_players=int(participants.filter(pl.col("is_repeat_player")).height),
            n_all_formats=int(participants.filter(pl.col("played_all_formats")).height),
            n_challenge_unknown=int(participants.filter(pl.col("played_challenge_unknown")).height),
            n_opposite_trait=int(participants.filter(pl.col("played_opposite_trait")).height),
        )
        return participants


def primary_analysis_population(participants: pl.DataFrame) -> pl.DataFrame:
    """Participants completing ≥6 analyzable generic segments (§7.2)."""
    return participants.filter(pl.col("eligible_primary")).filter(pl.col("mae_primary").is_not_null())


def own_analysis_population(participants: pl.DataFrame) -> pl.DataFrame:
    """Participants completing ≥6 own-data segments (§7.2)."""
    return participants.filter(pl.col("eligible_own")).filter(pl.col("mae_own").is_not_null())


def h5_paired_population(participants: pl.DataFrame) -> pl.DataFrame:
    """Participants with enough segments in both own and generic conditions."""
    return participants.filter(pl.col("eligible_h5")).filter(
        pl.col("mae_generic").is_not_null() & pl.col("mae_own").is_not_null()
    )


def all_format_population(participants: pl.DataFrame) -> pl.DataFrame:
    """People who saved at least one run of every format (A, B, and C)."""
    return participants.filter(pl.col("played_all_formats"))


def _challenge_history(runs: pl.DataFrame) -> pl.DataFrame:
    if "challenge_unknown" not in runs.columns:
        return runs.select("study_id").unique().with_columns(
            pl.lit(False).alias("played_challenge_unknown")
        )
    return runs.group_by("study_id").agg(
        pl.col("challenge_unknown").fill_null(False).any().alias("played_challenge_unknown")
    )


def _trait_flags(rounds: pl.DataFrame) -> pl.DataFrame:
    if "is_opposite_trait" not in rounds.columns:
        return rounds.select("study_id").unique().with_columns(
            pl.lit(False).alias("played_opposite_trait"),
            pl.lit("unknown").alias("player_trait"),
        )
    return rounds.group_by("study_id").agg(
        pl.col("is_opposite_trait").fill_null(False).any().alias("played_opposite_trait"),
        pl.col("player_trait").last().alias("player_trait"),
    )


def opposite_trait_summary(participants: pl.DataFrame) -> dict[str, Any]:
    """Counts and MAE for same-trait vs opposite-trait (Challenge the unknown)."""
    n_challenge = (
        int(participants.filter(pl.col("played_challenge_unknown")).height)
        if "played_challenge_unknown" in participants.columns
        else 0
    )
    n_opposite = (
        int(participants.filter(pl.col("played_opposite_trait")).height)
        if "played_opposite_trait" in participants.columns
        else 0
    )
    same = (
        participants.filter(pl.col("mae_same_trait").is_not_null())["mae_same_trait"]
        if "mae_same_trait" in participants.columns
        else None
    )
    opp = (
        participants.filter(pl.col("mae_opposite_trait").is_not_null())["mae_opposite_trait"]
        if "mae_opposite_trait" in participants.columns
        else None
    )
    both = participants
    if "mae_same_trait" in participants.columns and "mae_opposite_trait" in participants.columns:
        both = participants.filter(
            pl.col("mae_same_trait").is_not_null() & pl.col("mae_opposite_trait").is_not_null()
        )
    else:
        both = participants.head(0)

    def _mean(frame: pl.Series | None) -> float | None:
        if frame is None or frame.len() == 0:
            return None
        value = frame.mean()
        return float(value) if value is not None else None

    return {
        "n_challenge_unknown": n_challenge,
        "n_played_opposite_trait": n_opposite,
        "n_with_same_trait_mae": int(same.len()) if same is not None else 0,
        "n_with_opposite_trait_mae": int(opp.len()) if opp is not None else 0,
        "n_with_both_sides": both.height,
        "mean_mae_same_trait": _mean(same),
        "mean_mae_opposite_trait": _mean(opp),
        "mean_mae_same_among_both": (
            float(both["mae_same_trait"].mean()) if both.height else None
        ),
        "mean_mae_opposite_among_both": (
            float(both["mae_opposite_trait"].mean()) if both.height else None
        ),
    }
