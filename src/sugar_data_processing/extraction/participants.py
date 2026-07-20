"""Build per-participant analysis tables (one MAE per person per condition)."""

from __future__ import annotations

import polars as pl
from eliot import start_action

from sugar_data_processing.config import MIN_GENERIC_SEGMENTS, MIN_OWN_SEGMENTS
from sugar_data_processing.extraction.rounds import build_round_table


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


def build_participant_table(runs: pl.DataFrame) -> pl.DataFrame:
    """Aggregate to one row per ``study_id`` with person-level MAE scores.

    Person MAE is the mean of round MAEs (study design §7.3 — one summary
    accuracy score per person). Separate scores are kept for generic vs own
    data so H5 can use a within-person paired design.
    """
    with start_action(action_type="extraction.build_participant_table", n_runs=runs.height) as action:
        latest = _latest_runs(runs)
        rounds = build_round_table(latest)

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
            )
            if c in latest.columns
        ]
        demographics = latest.select(demo_cols).group_by("study_id").last()

        generic = _source_mae(rounds, "generic", "generic")
        own = _source_mae(rounds, "own", "own")
        overall = (
            rounds.group_by("study_id")
            .agg(
                pl.col("mae").mean().alias("mae_overall"),
                pl.col("rmse").mean().alias("rmse_overall"),
                pl.col("mape").mean().alias("mape_overall"),
                pl.len().alias("n_rounds_total"),
            )
        )

        participants = (
            demographics.join(generic, on="study_id", how="left")
            .join(own, on="study_id", how="left")
            .join(overall, on="study_id", how="left")
            .with_columns(
                pl.col("n_rounds_generic").fill_null(0).cast(pl.Int64),
                pl.col("n_rounds_own").fill_null(0).cast(pl.Int64),
                pl.col("n_rounds_total").fill_null(0).cast(pl.Int64),
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
            )
        )

        action.log(
            message_type="info",
            n_participants=participants.height,
            n_eligible_primary=int(participants.filter(pl.col("eligible_primary")).height),
            n_eligible_h5=int(participants.filter(pl.col("eligible_h5")).height),
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
