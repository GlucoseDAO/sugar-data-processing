"""Explode run-level statistics into one row per prediction round."""

from __future__ import annotations

from typing import Any

import polars as pl
from eliot import start_action

from sugar_data_processing.config import FORMAT_GENERIC, FORMAT_MIXED, FORMAT_OWN
from sugar_data_processing.gathering.encoding import (
    as_strict_bool,
    parse_optional_bool,
    parse_per_round_metrics,
    parse_round_context,
)
from sugar_data_processing.gathering.sources import (
    classify_data_class,
    is_opposite_trait,
    player_trait,
    redact_source_name,
)


def classify_source(
    format_code: str,
    is_example_data: bool | None,
    round_number: int | None = None,
) -> str:
    """Map sugar-sugar format flags to generic / own / mixed segment labels.

    Current exports write ``is_example_data`` on each round (and in
    ``round_context``). That flag wins. Legacy rows without it fall back to:

    - Format A → always generic
    - Format B → always own
    - Format C → mixed: odd rounds generic, even rounds own (app convention)
    """
    if is_example_data is True:
        return "generic"
    if is_example_data is False:
        return "own"

    fmt = (format_code or "").upper()
    if fmt == FORMAT_GENERIC:
        return "generic"
    if fmt == FORMAT_OWN:
        return "own"
    if fmt == FORMAT_MIXED:
        if round_number is None:
            return "mixed"
        return "generic" if int(round_number) % 2 == 1 else "own"
    return "unknown"


def _context_by_round(cell: Any) -> dict[int, dict[str, Any]]:
    try:
        entries = parse_round_context(cell)
    except (SyntaxError, ValueError, TypeError):
        return {}
    out: dict[int, dict[str, Any]] = {}
    for item in entries:
        try:
            number = int(item.get("round_number") or item.get("round") or 0)
        except (TypeError, ValueError):
            continue
        if number:
            out[number] = item
    return out


def build_round_table(runs: pl.DataFrame) -> pl.DataFrame:
    """Expand ``per_round_metrics`` (and ``round_context``) into a tidy frame."""
    with start_action(action_type="gathering.build_round_table", n_runs=runs.height) as action:
        rows: list[dict[str, Any]] = []
        for record in runs.iter_rows(named=True):
            context_map = _context_by_round(record.get("round_context"))
            metrics = parse_per_round_metrics(record.get("per_round_metrics"))
            player_diabetic = parse_optional_bool(record.get("diabetic"))
            run_is_example = parse_optional_bool(record.get("is_example_data"))
            run_source = redact_source_name(record.get("data_source_name"))
            if not metrics:
                rows.append(
                    _round_row(
                        record=record,
                        round_number=1,
                        mae=record.get("overall_mae_mgdl"),
                        rmse=record.get("overall_rmse_mgdl"),
                        mape=record.get("overall_mape_pct"),
                        mse=record.get("overall_mse_mgdl"),
                        item={},
                        context={},
                        run_is_example=run_is_example,
                        run_source=run_source,
                        player_diabetic=player_diabetic,
                    )
                )
                continue
            for item in metrics:
                round_number = int(item.get("round_number") or item.get("round") or 0)
                rows.append(
                    _round_row(
                        record=record,
                        round_number=round_number,
                        mae=item.get("mae"),
                        rmse=item.get("rmse"),
                        mape=item.get("mape"),
                        mse=item.get("mse"),
                        item=item,
                        context=context_map.get(round_number, {}),
                        run_is_example=run_is_example,
                        run_source=run_source,
                        player_diabetic=player_diabetic,
                    )
                )

        schema = {
            "study_id": pl.Utf8,
            "run_id": pl.Utf8,
            "format": pl.Utf8,
            "round_number": pl.Int64,
            "source": pl.Utf8,
            "data_source_name": pl.Utf8,
            "is_example_data": pl.Boolean,
            "data_class": pl.Utf8,
            "player_trait": pl.Utf8,
            "is_opposite_trait": pl.Boolean,
            "challenge_unknown": pl.Boolean,
            "generic_slice_key": pl.Utf8,
            "window_start_index": pl.Int64,
            "window_start_time": pl.Utf8,
            "prediction_start_time": pl.Utf8,
            "window_end_time": pl.Utf8,
            "mae": pl.Float64,
            "rmse": pl.Float64,
            "mape": pl.Float64,
            "mse": pl.Float64,
        }
        if not rows:
            action.log(message_type="info", n_rounds=0)
            return pl.DataFrame(schema=schema)

        rounds = pl.DataFrame(rows, schema=schema)
        action.log(message_type="info", n_rounds=rounds.height)
        return rounds


def _round_row(
    *,
    record: dict[str, Any],
    round_number: int,
    mae: Any,
    rmse: Any,
    mape: Any,
    mse: Any,
    item: dict[str, Any],
    context: dict[str, Any],
    run_is_example: bool | None,
    run_source: str,
    player_diabetic: bool | None,
) -> dict[str, Any]:
    is_example = parse_optional_bool(item.get("is_example_data"))
    if is_example is None:
        is_example = parse_optional_bool(context.get("is_example_data"))
    if is_example is None:
        is_example = run_is_example

    source_name = redact_source_name(
        item.get("data_source_name") or context.get("data_source_name") or run_source
    )
    source = classify_source(str(record.get("format") or ""), is_example, round_number)
    data_class = classify_data_class(
        source_name,
        is_example=is_example,
        player_diabetic=player_diabetic,
    )
    trait = player_trait(player_diabetic)
    window_index = (
        context.get("prediction_window_start_index")
        if context.get("prediction_window_start_index") is not None
        else item.get("prediction_window_start_index")
    )
    try:
        window_index_int = int(window_index) if window_index is not None and str(window_index) != "" else None
    except (TypeError, ValueError):
        window_index_int = None
    challenge = as_strict_bool(record.get("challenge_unknown"), default=False)
    slice_key = item.get("generic_slice_key") or context.get("generic_slice_key") or ""
    resolved_example = is_example if is_example is not None else source == "generic"
    return {
        "study_id": record["study_id"],
        "run_id": record["run_id"],
        "format": record["format"],
        "round_number": round_number,
        "source": source,
        "data_source_name": source_name,
        "is_example_data": resolved_example,
        "data_class": data_class or "",
        "player_trait": trait,
        "is_opposite_trait": is_opposite_trait(trait, data_class),
        "challenge_unknown": challenge,
        "generic_slice_key": str(slice_key) if slice_key else "",
        "window_start_index": window_index_int,
        "window_start_time": str(context.get("window_start_time") or ""),
        "prediction_start_time": str(context.get("prediction_start_time") or ""),
        "window_end_time": str(context.get("window_end_time") or ""),
        "mae": float(mae) if mae is not None else None,
        "rmse": float(rmse) if rmse is not None else None,
        "mape": float(mape) if mape is not None else None,
        "mse": float(mse) if mse is not None else None,
    }
