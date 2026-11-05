"""Which models a player actually has.

The latest day in the export is the in-game Kronos sessions. Those players are
not given the post-factum baselines. Everyone earlier keeps every model that
was scored on their played rounds.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

MODEL_KRONOS: str = "kronos"


def canonical_model_name(name: str) -> str:
    """Fold KronLSTM repo ids onto one model name. Other names stay as written."""
    text = str(name or "").strip()
    folded = text.casefold().replace("-", "").replace("_", "")
    if "kronlstm" in folded or "kronos" in folded:
        return MODEL_KRONOS
    return text or "native"


def kronos_only_study_ids(runs: pl.DataFrame) -> list[str]:
    """Players whose latest session falls on the newest day in ``runs``."""
    if runs.height == 0 or "study_id" not in runs.columns or "timestamp" not in runs.columns:
        return []
    stamped = runs.select(
        pl.col("study_id").cast(pl.Utf8),
        pl.col("timestamp").cast(pl.Utf8).str.slice(0, 10).alias("day"),
    ).filter(pl.col("day") != "")
    if stamped.height == 0:
        return []
    latest = stamped.select(pl.col("day").max()).item()
    if not latest:
        return []
    return (
        stamped.group_by("study_id")
        .agg(pl.col("day").max().alias("last_day"))
        .filter(pl.col("last_day") == latest)["study_id"]
        .cast(pl.Utf8)
        .to_list()
    )


def predictions_for_display(predictions: pl.DataFrame, runs: pl.DataFrame) -> pl.DataFrame:
    """Drop models a player did not run, and name Kronos consistently."""
    if predictions.height == 0:
        return predictions
    named = predictions.with_columns(
        pl.col("model_name")
        .cast(pl.Utf8)
        .map_elements(canonical_model_name, return_dtype=pl.Utf8)
        .alias("model_name")
    )
    only = kronos_only_study_ids(runs)
    if not only:
        return named
    return named.filter(
        ~pl.col("study_id").cast(pl.Utf8).is_in(only) | (pl.col("model_name") == MODEL_KRONOS)
    )


def player_model_table(runs: pl.DataFrame, predictions: pl.DataFrame) -> pl.DataFrame:
    """One row per player who has a model: ``study_id``, comma-separated ``models``."""
    only = set(kronos_only_study_ids(runs))
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    if (
        predictions.height
        and "study_id" in predictions.columns
        and "model_name" in predictions.columns
    ):
        grouped = predictions.group_by(pl.col("study_id").cast(pl.Utf8)).agg(
            pl.col("model_name").cast(pl.Utf8).unique().sort().alias("models")
        )
        for record in grouped.iter_rows(named=True):
            study_id = str(record["study_id"])
            models = (
                [MODEL_KRONOS] if study_id in only else [str(name) for name in record["models"]]
            )
            rows.append({"study_id": study_id, "models": ",".join(models)})
            seen.add(study_id)
    for study_id in sorted(only):
        if study_id not in seen:
            rows.append({"study_id": study_id, "models": MODEL_KRONOS})
    if not rows:
        return pl.DataFrame(schema={"study_id": pl.Utf8, "models": pl.Utf8})
    return pl.DataFrame(rows).sort("study_id")


def attach_player_models(participants: pl.DataFrame, table: pl.DataFrame) -> pl.DataFrame:
    """Add a ``models`` column. Players with no model get an empty string."""
    base = participants.drop("models") if "models" in participants.columns else participants
    if table.height == 0:
        return base.with_columns(pl.lit("").alias("models"))
    return base.join(
        table.select(
            pl.col("study_id").cast(pl.Utf8),
            pl.col("models").cast(pl.Utf8),
        ),
        on="study_id",
        how="left",
    ).with_columns(pl.col("models").fill_null(""))


def write_player_models(table: pl.DataFrame, path: Path | str) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(dest)
    return dest
