"""Milestone AI page.

Human analysis covers every participant. AI MAE, H6, and People-tab forecast
lines cover only the most recent participants, including the two latest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from eliot import start_action

from sugar_data_processing.ai.ingest import ingest_model_predictions, load_model_predictions
from sugar_data_processing.ai.run import run_post_factum_scoring
from sugar_data_processing.ai.native import (
    blank_contact,
    predictions_from_native_ai,
    write_native_ai_csv,
)
from sugar_data_processing.ai.overlay import h6_human_vs_models, overlay_model_mae
from sugar_data_processing.ai.player_models import (
    attach_player_models,
    player_model_table,
    predictions_for_display,
    write_player_models,
)
from sugar_data_processing.ai.score import pick_primary_model
from sugar_data_processing.ai.sequences import build_point_table
from sugar_data_processing.ai.traces import build_round_traces, traces_from_forecast_points
from sugar_data_processing.ai.windows import reconstruct_played_windows
from sugar_data_processing.comparison.benchmarks import benchmark_context
from sugar_data_processing.config import (
    EVALUATION_MODE_IN_PLACE,
    EVALUATION_MODE_POST_FACTUM,
    MILESTONE_AI_COHORT_SIZE,
    MILESTONE_AI_HTML,
    MILESTONE_AI_JSON,
    NATIVE_AI_CSV_NAME,
)
from sugar_data_processing.gathering.load import load_prediction_statistics
from sugar_data_processing.gathering.participants import build_participant_table
from sugar_data_processing.output.explorer import write_explorer_html
from sugar_data_processing.output.narration import (
    explain_milestone_ai_page,
    explain_population_with_recent_ai,
)
from sugar_data_processing.statistics.hypotheses import run_all_hypotheses
from sugar_data_processing.verification.report import verify_dataset


@dataclass(frozen=True)
class CohortSlice:
    study_ids: list[str]
    latest_two: list[str]


@dataclass(frozen=True)
class SplitReport:
    """Full human analysis, AI restricted to the most recent participants."""

    html_path: Path
    n_people: int
    n_ai_cohort: int
    n_with_ai: int
    primary_model: str
    ai_study_ids: list[str]


@dataclass(frozen=True)
class MilestoneAiBundle:
    directory: Path
    human_csv: Path
    ai_csv: Path
    html_path: Path
    json_path: Path
    n_people: int
    n_with_ai: int
    primary_model: str
    evaluation_modes: list[str]


def write_population_with_recent_ai(
    human_csv: Path | str,
    html_path: Path | str,
    *,
    cohort_size: int = MILESTONE_AI_COHORT_SIZE,
    predictions_csv: Path | str | None = None,
) -> SplitReport:
    """Human analysis on every participant. AI scoring and traces on the most recent ones.

    ``predictions_csv`` skips model inference and reuses an already scored
    ``model_predictions.csv`` for that recent cohort.
    """
    source = Path(human_csv)
    html = Path(html_path)
    with start_action(
        action_type="ai.write_population_with_recent_ai", source=str(source)
    ) as action:
        runs = load_prediction_statistics(source)
        participants = build_participant_table(runs)
        verification = verify_dataset(runs, participants)
        suite = run_all_hypotheses(participants)
        eligible = participants.filter(pl.col("eligible_primary"))
        benchmarks = benchmark_context(
            eligible if eligible.height > 0 else participants,
            mae_col="mae_primary",
        )
        cohort = recent_cohort(runs, cohort_size)
        if len(cohort.latest_two) >= 2 and not set(cohort.latest_two).issubset(cohort.study_ids):
            raise ValueError("The two most recent participants are outside the AI cohort.")
        ai_runs = runs.filter(pl.col("study_id").is_in(cohort.study_ids))
        ai_participants = participants.filter(pl.col("study_id").is_in(cohort.study_ids))
        if predictions_csv is None:
            scoring = run_post_factum_scoring(
                ai_runs,
                ai_participants,
                source_csv=source,
                ai_dir=html.parent / "processed" / "ai",
            )
            ai_people = scoring.ai_participants
            ai_suite = scoring.ai_suite
            h6 = scoring.h6
            traces = scoring.traces
            comparison = scoring.comparison.to_dict()
            primary = scoring.primary_model
            model_table = scoring.player_models
        else:
            ai_people, ai_suite, h6, traces, comparison, primary, model_table = (
                _ai_from_saved_predictions(
                    ai_runs,
                    ai_participants,
                    source,
                    Path(predictions_csv),
                )
            )
        participants = attach_player_models(participants, model_table)
        ai_people = attach_player_models(ai_people, model_table)
        write_player_models(model_table, html.parent / "player_models.csv")
        scored_ids = [
            study_id
            for study_id in ai_people.filter(pl.col("mae_primary").is_not_null())[
                "study_id"
            ].to_list()
        ]
        note = explain_population_with_recent_ai(n_people=participants.height)
        sequences_dir = (
            Path(predictions_csv).parent
            if predictions_csv is not None
            else html.parent / "processed" / "ai"
        )
        write_explorer_html(
            participants=participants,
            runs=runs,
            output_path=html,
            source_csv=source,
            suite=suite,
            benchmarks=benchmarks,
            verification=verification,
            edition="merged",
            comparison=comparison,
            sequences_dir=sequences_dir,
            ai_participants=ai_people,
            ai_suite=ai_suite,
            traces=traces,
            primary_model=primary,
            h6=h6,
            title=f"Sugar Sugar — {participants.height} people",
            edition_note=note,
        )
        action.log(
            message_type="info",
            n_people=participants.height,
            n_ai_cohort=len(cohort.study_ids),
            n_with_ai=len(scored_ids),
            primary=primary,
        )
        return SplitReport(
            html_path=html,
            n_people=participants.height,
            n_ai_cohort=len(cohort.study_ids),
            n_with_ai=len(scored_ids),
            primary_model=primary,
            ai_study_ids=[str(study_id) for study_id in cohort.study_ids],
        )


def _ai_from_saved_predictions(
    ai_runs: pl.DataFrame,
    ai_participants: pl.DataFrame,
    human_csv: Path,
    predictions_csv: Path,
) -> tuple[
    pl.DataFrame,
    object,
    dict[str, object],
    list[dict[str, object]],
    dict[str, object],
    str,
    pl.DataFrame,
]:
    points = build_point_table(ai_runs)
    windows = reconstruct_played_windows(ai_runs, raw_csv=human_csv, points=points)
    predictions = predictions_for_display(
        load_model_predictions(predictions_csv).filter(
            pl.col("study_id").is_in(ai_runs["study_id"].to_list())
        ),
        ai_runs,
    )
    if predictions.height == 0:
        raise ValueError(f"No saved AI predictions for the recent cohort in {predictions_csv}")
    scored, comparison = ingest_model_predictions(points, predictions)
    primary = pick_primary_model(predictions)
    ai_people = (
        overlay_model_mae(ai_participants, scored, primary) if scored.height else ai_participants
    )
    ai_suite = run_all_hypotheses(ai_people)
    h6 = h6_human_vs_models(ai_participants, scored) if scored.height else {}
    traces = build_round_traces(windows, scored) if windows and scored.height else []
    if not traces and scored.height:
        traces = traces_from_forecast_points(scored)
    model_table = player_model_table(ai_runs, predictions)
    return ai_people, ai_suite, h6, traces, comparison.to_dict(), primary, model_table


def recent_cohort(runs: pl.DataFrame, n: int) -> CohortSlice:
    """Most recent ``n`` people by their latest session timestamp, plus the last two."""
    if runs.height == 0 or "study_id" not in runs.columns or "timestamp" not in runs.columns:
        return CohortSlice(study_ids=[], latest_two=[])
    people = (
        runs.select(
            pl.col("study_id").cast(pl.Utf8),
            pl.col("timestamp").cast(pl.Utf8).alias("timestamp"),
        )
        .group_by("study_id")
        .agg(pl.col("timestamp").max().alias("last_ts"))
        .sort(["last_ts", "study_id"], descending=[True, False])
    )
    ordered = [str(value) for value in people["study_id"].to_list()]
    latest_two = ordered[: min(2, len(ordered))]
    return CohortSlice(study_ids=ordered[:n], latest_two=latest_two)


def write_milestone_ai_bundle(
    human_csv: Path | str,
    dest_dir: Path | str,
    *,
    ai_csv: Path | str | None = None,
    scores_csv: Path | str | None = None,
    cohort_size: int = MILESTONE_AI_COHORT_SIZE,
    model_name: str = "glumind",
) -> MilestoneAiBundle:
    """Write the cohort CSVs, then the HTML those two files produce.

    Pass ``ai_csv`` when the native export already exists. Otherwise
    ``scores_csv`` (post-factum ``model_scores.csv``) is rewritten into that
    same schema for ``model_name`` only, and only for cohort members who have
    those scores. The two most recent people must be among them.
    """
    source = Path(human_csv)
    dest = Path(dest_dir)
    input_dir = dest / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    with start_action(action_type="ai.write_milestone_ai_bundle", source=str(source)) as action:
        raw = pl.read_csv(source, infer_schema_length=0)
        cohort = recent_cohort(raw, cohort_size)
        if len(cohort.latest_two) < 2 and raw["study_id"].n_unique() >= 2:
            raise ValueError("Could not identify the two most recent participants.")
        if not set(cohort.latest_two).issubset(cohort.study_ids):
            raise ValueError("The two most recent participants are outside the milestone cohort.")

        human_out = input_dir / "prediction_statistics.csv"
        kept = blank_contact(raw.filter(pl.col("study_id").is_in(cohort.study_ids)))
        kept.write_csv(human_out)

        ai_out = input_dir / NATIVE_AI_CSV_NAME
        if ai_csv is not None:
            _copy_native_slice(Path(ai_csv), ai_out, cohort.study_ids)
            origin = "native AI export"
        else:
            if scores_csv is None or not Path(scores_csv).is_file():
                raise FileNotFoundError(
                    "No native AI CSV and no model_scores.csv. "
                    "Pass --ai-csv, or run sdp analyze so post-factum scores exist."
                )
            _write_scores_as_native(
                human_out,
                Path(scores_csv),
                ai_out,
                study_ids=cohort.study_ids,
                model_name=model_name,
            )
            origin = (
                f"post-factum {model_name} scores written as {NATIVE_AI_CSV_NAME} "
                "(the live AI export is not on the server yet)"
            )

        bundle = render_milestone_ai(
            human_out,
            ai_out,
            dest / MILESTONE_AI_HTML,
            json_path=dest / MILESTONE_AI_JSON,
            origin=origin,
            latest_two=cohort.latest_two,
        )
        action.log(
            message_type="info",
            n_people=bundle.n_people,
            n_with_ai=bundle.n_with_ai,
            primary=bundle.primary_model,
        )
        return bundle


def render_milestone_ai(
    human_csv: Path | str,
    ai_csv: Path | str,
    html_path: Path | str,
    *,
    json_path: Path | str | None = None,
    origin: str = "",
    latest_two: list[str] | None = None,
) -> MilestoneAiBundle:
    """Build the milestone AI HTML from a human CSV and a native AI CSV."""
    human_path = Path(human_csv)
    ai_path = Path(ai_csv)
    html = Path(html_path)
    with start_action(action_type="ai.render_milestone_ai") as action:
        runs = load_prediction_statistics(human_path)
        participants = build_participant_table(runs)
        verification = verify_dataset(runs, participants)
        suite = run_all_hypotheses(participants)
        eligible = participants.filter(pl.col("eligible_primary"))
        benchmarks = benchmark_context(
            eligible if eligible.height > 0 else participants,
            mae_col="mae_primary",
        )
        points = build_point_table(runs)
        predictions = predictions_from_native_ai(ai_path)
        scored, comparison = ingest_model_predictions(points, predictions)
        if predictions.height == 0 or scored.height == 0:
            raise ValueError(f"Native AI CSV produced no joined forecast points: {ai_path}")
        primary = pick_primary_model(predictions)
        ai_people = overlay_model_mae(participants, scored, primary)
        ai_with = ai_people.filter(pl.col("mae_primary").is_not_null())
        required = (
            latest_two if latest_two is not None else recent_cohort(runs, runs.height).latest_two
        )
        have = set(ai_with["study_id"].to_list())
        missing = [study_id for study_id in required if study_id not in have]
        if missing:
            raise ValueError(
                "The most recent participants have no AI forecast, so they cannot "
                f"be on the milestone AI comparison ({len(missing)} missing)."
            )
        primary_scores = scored.filter(pl.col("model_name") == primary)
        traces = traces_from_forecast_points(primary_scores)
        h6 = h6_human_vs_models(participants, primary_scores)
        ai_suite = run_all_hypotheses(ai_with)
        modes = sorted(str(mode) for mode in predictions["evaluation_mode"].unique().to_list())
        note = explain_milestone_ai_page(
            n_people=participants.height,
            n_with_ai=ai_with.height,
            origin=origin or _origin_from_modes(modes),
        )
        write_explorer_html(
            participants=participants,
            runs=runs,
            output_path=html,
            source_csv=human_path,
            suite=suite,
            benchmarks=benchmarks,
            verification=verification,
            edition="milestone-ai",
            comparison=comparison.to_dict(),
            sequences_dir=ai_path.parent,
            ai_participants=ai_with,
            ai_suite=ai_suite,
            traces=traces,
            primary_model=primary,
            h6=h6,
            title="Sugar Sugar — milestone AI",
            edition_note=note,
        )
        payload = {
            "edition": "milestone-ai",
            "n_people": participants.height,
            "n_runs": runs.height,
            "n_with_ai": ai_with.height,
            "last_two_have_ai": True,
            "primary_model": primary,
            "evaluation_modes": modes,
            "ai_origin": origin or _origin_from_modes(modes),
            "inputs": {
                "human_csv": str(human_path),
                "ai_csv": str(ai_path),
            },
            "html": str(html),
            "rebuild": (
                "uv run sdp milestone-ai "
                f"--csv {human_path} --ai-csv {ai_path} --dest {html.parent}"
            ),
        }
        json_out = Path(json_path) if json_path is not None else html.with_name(MILESTONE_AI_JSON)
        json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        action.log(
            message_type="info",
            n_people=participants.height,
            n_with_ai=ai_with.height,
            html=str(html),
        )
        return MilestoneAiBundle(
            directory=html.parent,
            human_csv=human_path,
            ai_csv=ai_path,
            html_path=html,
            json_path=json_out,
            n_people=participants.height,
            n_with_ai=ai_with.height,
            primary_model=primary,
            evaluation_modes=modes,
        )


def _copy_native_slice(src: Path, dest: Path, study_ids: list[str]) -> None:
    frame = blank_contact(pl.read_csv(src, infer_schema_length=0))
    if "study_id" not in frame.columns:
        raise ValueError(f"Native AI CSV has no study_id column: {src}")
    kept = frame.filter(pl.col("study_id").is_in(study_ids))
    dest.parent.mkdir(parents=True, exist_ok=True)
    kept.write_csv(dest)


def _write_scores_as_native(
    human_csv: Path,
    scores_csv: Path,
    dest: Path,
    *,
    study_ids: list[str],
    model_name: str,
) -> None:
    runs = load_prediction_statistics(human_csv)
    scores = pl.read_csv(
        scores_csv,
        schema_overrides={
            "study_id": pl.Utf8,
            "run_id": pl.Utf8,
            "model_name": pl.Utf8,
            "format": pl.Utf8,
            "data_source_name": pl.Utf8,
            "source": pl.Utf8,
            "timestamp": pl.Utf8,
        },
    )
    needed = {
        "study_id",
        "run_id",
        "round_number",
        "point_index",
        "model_name",
        "model_predicted_mgdl",
        "real_mgdl",
    }
    missing = needed.difference(scores.columns)
    if missing:
        raise ValueError(f"model scores missing columns: {sorted(missing)}")
    subset = scores.filter(
        (pl.col("model_name") == model_name) & pl.col("study_id").is_in(study_ids)
    )
    write_native_ai_csv(
        runs,
        subset,
        dest,
        model_name=model_name,
        evaluation_mode=EVALUATION_MODE_POST_FACTUM,
    )
    written = pl.read_csv(dest, infer_schema_length=0)
    if written.height == 0 or "study_id" not in written.columns:
        raise ValueError(f"No {model_name} scores for the milestone cohort in {scores_csv}")


def _origin_from_modes(modes: list[str]) -> str:
    if modes == [EVALUATION_MODE_IN_PLACE]:
        return "in-game native AI export"
    if EVALUATION_MODE_IN_PLACE in modes:
        return "mix of in-game native AI and post-factum scores"
    return "post-factum scores stored in the native AI export schema"
