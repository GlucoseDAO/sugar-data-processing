"""Read and write the in-game AI export (``prediction_statistics_ai.csv``).

sugar-sugar writes one row per session and model. Forecast points live in
``predicted_values_AI`` as ``{version, round, value}`` literals, the same shape
as the human ``predicted_values`` column. This module turns that file into the
point-level table the rest of the pipeline already joins.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import polars as pl
from eliot import start_action

from sugar_data_processing.ai.player_models import canonical_model_name
from sugar_data_processing.config import (
    EVALUATION_MODE_IN_PLACE,
    EVALUATION_MODE_POST_FACTUM,
    NATIVE_AI_CSV_NAME,
)
from sugar_data_processing.fetch import SCRUB_COLUMNS
from sugar_data_processing.gathering.encoding import parse_point_series, series_value
from sugar_data_processing.gathering.sources import redact_source_name

# Matches sugar_sugar.ai_player.AI_CSV_COLUMNS. ``evaluation_mode`` is extra:
# the live app omits it (those rows are in-place). A file we write from a
# post-factum replay sets it so the page does not pretend the game scored it.
NATIVE_AI_COLUMNS: tuple[str, ...] = (
    "study_id",
    "run_id",
    "timestamp",
    "email",
    "format",
    "is_example_data",
    "data_source_name",
    "age",
    "gender",
    "uses_cgm",
    "diabetic",
    "diabetic_type",
    "diabetes_duration",
    "generic_intervention",
    "challenge_unknown",
    "challenge_unknown_pct",
    "location",
    "model_repo",
    "requested_revision",
    "resolved_revision",
    "rounds_played_AI",
    "rounds_failed_AI",
    "predicted_values_AI",
    "real_values_AI",
    "prediction_times_AI",
    "overall_mae_mgdl_AI",
    "overall_mse_mgdl_AI",
    "overall_rmse_mgdl_AI",
    "overall_mape_pct_AI",
    "per_round_metrics_AI",
    "round_context_AI",
    "timings_AI",
    "evaluation_mode",
)

PREDICTION_SCHEMA: dict[str, pl.DataType] = {
    "study_id": pl.Utf8,
    "run_id": pl.Utf8,
    "round_number": pl.Int64,
    "point_index": pl.Int64,
    "model_name": pl.Utf8,
    "model_predicted_mgdl": pl.Float64,
    "evaluation_mode": pl.Utf8,
}

_MODEL_TOKEN = re.compile(r"[^A-Za-z0-9_.-]+")
_MODES = frozenset({EVALUATION_MODE_IN_PLACE, EVALUATION_MODE_POST_FACTUM})


def native_ai_csv_beside(human_csv: Path | str) -> Path | None:
    """``prediction_statistics_ai.csv`` sitting next to a human export, if it exists."""
    candidate = Path(human_csv).with_name(NATIVE_AI_CSV_NAME)
    if candidate.is_file():
        return candidate
    return None


def model_name_from_repo(repo: str) -> str:
    """Short token from a Hugging Face repo id or a local model name.

    The in-game KronLSTM bundle is always ``kronos``, whatever the repo id is.
    """
    text = (repo or "").strip()
    if not text:
        return "native"
    short = text.rstrip("/").split("/")[-1]
    token = _MODEL_TOKEN.sub("_", short).strip("_.-")
    if not token:
        return "native"
    named = canonical_model_name(token)
    return named[:80] or "native"


def blank_contact(frame: pl.DataFrame) -> pl.DataFrame:
    """Empty email, location, and paper name when those columns are present."""
    updates = [pl.lit("").alias(col) for col in SCRUB_COLUMNS if col in frame.columns]
    if not updates:
        return frame
    return frame.with_columns(updates)


def predictions_from_native_ai(path: Path | str) -> pl.DataFrame:
    """Explode a native AI CSV into the model-prediction join table.

    Point order inside a round follows the file, which is the same order as
    the human ``predicted_values`` series. ``point_index`` therefore lines up
    with :func:`sugar_data_processing.ai.sequences.build_point_table`.
    """
    csv_path = Path(path)
    with start_action(action_type="ai.predictions_from_native", path=str(csv_path)) as action:
        if not csv_path.is_file():
            action.log(message_type="info", n_rows=0, reason="missing")
            return pl.DataFrame(schema=PREDICTION_SCHEMA)
        frame = blank_contact(pl.read_csv(csv_path, infer_schema_length=0))
        rows: list[dict[str, Any]] = []
        for record in frame.iter_rows(named=True):
            rows.extend(_prediction_rows(record))
        if not rows:
            action.log(message_type="info", n_rows=0)
            return pl.DataFrame(schema=PREDICTION_SCHEMA)
        predictions = pl.DataFrame(rows, schema=PREDICTION_SCHEMA)
        action.log(
            message_type="info",
            n_rows=predictions.height,
            n_people=predictions["study_id"].n_unique(),
            models=sorted(predictions["model_name"].unique().to_list()),
        )
        return predictions


def merge_model_predictions(post_factum: pl.DataFrame, native: pl.DataFrame) -> pl.DataFrame:
    """Keep both families. A native row replaces a post-factum row on the same key."""
    if native.height == 0:
        return post_factum
    if post_factum.height == 0:
        return native
    keys = ["study_id", "run_id", "round_number", "point_index", "model_name"]
    kept = post_factum.join(native.select(keys), on=keys, how="anti")
    return pl.concat([kept, native], how="vertical")


def write_native_ai_csv(
    runs: pl.DataFrame,
    scored: pl.DataFrame,
    dest: Path | str,
    *,
    model_name: str,
    evaluation_mode: str = EVALUATION_MODE_POST_FACTUM,
) -> Path:
    """Write one native-schema row per run that has ``model_name`` scores.

    Contact columns are blank. Personal upload filenames are ``own_upload``.
    """
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    model_rows = scored.filter(pl.col("model_name") == model_name) if scored.height else scored
    if model_rows.height == 0:
        pl.DataFrame({col: [] for col in NATIVE_AI_COLUMNS}).write_csv(dest_path)
        return dest_path

    run_lookup = {
        (str(row["study_id"]), str(row["run_id"])): row for row in runs.iter_rows(named=True)
    }
    written: list[dict[str, Any]] = []
    groups = model_rows.sort(["study_id", "run_id", "round_number", "point_index"]).partition_by(
        ["study_id", "run_id"],
        maintain_order=True,
    )
    for group in groups:
        study_id = str(group["study_id"][0])
        run_id = str(group["run_id"][0])
        written.append(
            _native_row(
                group,
                run_lookup.get((study_id, run_id), {}),
                model_name=model_name,
                evaluation_mode=evaluation_mode,
            )
        )
    pl.DataFrame(written, schema={col: pl.Utf8 for col in NATIVE_AI_COLUMNS}).write_csv(dest_path)
    return dest_path


def _prediction_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    predicted = parse_point_series(record.get("predicted_values_AI"))
    if not predicted:
        return []
    mode = _mode(record.get("evaluation_mode"))
    model_name = model_name_from_repo(str(record.get("model_repo") or ""))
    study_id = str(record.get("study_id") or "")
    run_id = str(record.get("run_id") or "")
    per_round: dict[int, int] = {}
    rows: list[dict[str, Any]] = []
    for item in predicted:
        value = _as_float(series_value(item))
        if value is None:
            continue
        round_number = _as_int(item.get("round"))
        if round_number is None:
            round_number = 1
        point_index = per_round.get(round_number, 0)
        per_round[round_number] = point_index + 1
        rows.append(
            {
                "study_id": study_id,
                "run_id": run_id,
                "round_number": round_number,
                "point_index": point_index,
                "model_name": model_name,
                "model_predicted_mgdl": value,
                "evaluation_mode": mode,
            }
        )
    return rows


def _native_row(
    group: pl.DataFrame,
    run: dict[str, Any],
    *,
    model_name: str,
    evaluation_mode: str,
) -> dict[str, Any]:
    predicted: list[dict[str, Any]] = []
    real: list[dict[str, Any]] = []
    times: list[dict[str, Any]] = []
    per_round: list[dict[str, Any]] = []
    contexts: list[dict[str, Any]] = []
    abs_errors: list[float] = []
    sq_errors: list[float] = []
    mape_terms: list[float] = []

    rounds = group.partition_by("round_number", maintain_order=True)
    for rnd in rounds:
        head = rnd.row(0, named=True)
        number = int(head["round_number"])
        fmt = str(head.get("format") or run.get("format") or "")
        source = redact_source_name(head.get("data_source_name") or run.get("data_source_name"))
        round_abs: list[float] = []
        for point in rnd.sort("point_index").iter_rows(named=True):
            pred = _as_float(point.get("model_predicted_mgdl"))
            actual = _as_float(point.get("real_mgdl"))
            if pred is None or actual is None:
                continue
            err = abs(pred - actual)
            round_abs.append(err)
            abs_errors.append(err)
            sq_errors.append(err * err)
            if actual != 0.0:
                mape_terms.append(err / abs(actual) * 100.0)
            stamp = str(point.get("timestamp") or "")
            predicted.append({"version": fmt, "round": number, "value": pred})
            real.append({"version": fmt, "round": number, "value": actual})
            times.append({"version": fmt, "round": number, "value": stamp})
        per_round.append(
            {
                "round_number": number,
                "mae": _mean(round_abs),
                "mse": _mean([e * e for e in round_abs]),
                "rmse": _rmse(round_abs),
                "mape": None,
                "status": "done",
                "data_source_name": source,
                "is_example_data": bool(run.get("is_example_data", True)),
                "generic_slice_key": "",
            }
        )
        contexts.append(
            {
                "round_number": number,
                "format": fmt,
                "data_source_name": source,
                "generic_slice_key": "",
                "job_key": "",
                "window_start_time": "",
                "window_end_time": "",
                "error": None,
            }
        )

    mae = _mean(abs_errors)
    mse = _mean(sq_errors)
    rmse = math.sqrt(mse) if mse is not None else None
    mape = _mean(mape_terms)
    return {
        "study_id": str(group["study_id"][0]),
        "run_id": str(group["run_id"][0]),
        "timestamp": str(run.get("timestamp") or ""),
        "email": "",
        "format": _format_cell(run, group),
        "is_example_data": str(bool(run.get("is_example_data", True))),
        "data_source_name": redact_source_name(run.get("data_source_name")),
        "age": _cell(run.get("age")),
        "gender": _cell(run.get("gender")),
        "uses_cgm": _cell(run.get("uses_cgm")),
        "diabetic": _cell(run.get("diabetic")),
        "diabetic_type": _cell(run.get("diabetic_type")),
        "diabetes_duration": _cell(run.get("diabetes_duration")),
        "generic_intervention": _cell(run.get("generic_intervention")),
        "challenge_unknown": _cell(run.get("challenge_unknown")),
        "challenge_unknown_pct": _cell(run.get("challenge_unknown_pct")),
        "location": "",
        "model_repo": model_name,
        "requested_revision": evaluation_mode,
        "resolved_revision": evaluation_mode,
        "rounds_played_AI": str(len(per_round)),
        "rounds_failed_AI": "0",
        "predicted_values_AI": str(predicted),
        "real_values_AI": str(real),
        "prediction_times_AI": str(times),
        "overall_mae_mgdl_AI": _cell(mae),
        "overall_mse_mgdl_AI": _cell(mse),
        "overall_rmse_mgdl_AI": _cell(rmse),
        "overall_mape_pct_AI": _cell(mape),
        "per_round_metrics_AI": str(per_round),
        "round_context_AI": str(contexts),
        "timings_AI": "[]",
        "evaluation_mode": evaluation_mode,
    }


def _mode(raw: Any) -> str:
    text = str(raw or "").strip()
    if text in _MODES:
        return text
    return EVALUATION_MODE_IN_PLACE


def _as_int(raw: Any) -> int | None:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _as_float(raw: Any) -> float | None:
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        number = float(raw)
    else:
        text = str(raw).strip()
        if text == "" or text in {"-", "nan", "None", "null"}:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
    if number != number:
        return None
    return number


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _rmse(abs_errors: list[float]) -> float | None:
    mse = _mean([err * err for err in abs_errors])
    if mse is None:
        return None
    return math.sqrt(mse)


def _format_cell(run: dict[str, Any], group: pl.DataFrame) -> str:
    if run.get("format"):
        return str(run["format"])
    if "format" in group.columns:
        return str(group["format"][0] or "")
    return ""


def _cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value)
