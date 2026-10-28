"""Archived games beside the export are read, anonymized and fetched with it."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from sugar_data_processing.anonymize import anonymize_statistics_csv
from sugar_data_processing.fetch import fetch_statistics
from sugar_data_processing.fixtures.synthetic import write_synthetic_csv
from sugar_data_processing.gathering.load import (
    game_counter_view,
    load_prediction_statistics,
    statistics_csv_paths,
)


def _export_with_archive(folder: Path, *, linked: bool) -> tuple[Path, pl.DataFrame, pl.DataFrame]:
    """A live export plus ``archived_2026-08-19.csv`` holding two new games and one duplicate."""
    folder.mkdir(parents=True, exist_ok=True)
    export = folder / "prediction_statistics.csv"
    write_synthetic_csv(export, n_participants=6, seed=11)
    frame = pl.read_csv(export, infer_schema_length=0)
    if linked:
        frame = frame.with_columns(pl.format("person-{}", pl.col("study_id")).alias("participant_key"))
        frame.write_csv(export)
    new_games = frame.head(2).with_columns(
        pl.format("archived-run-{}", pl.col("run_id")).alias("run_id"),
        pl.format("archived-id-{}", pl.col("study_id")).alias("study_id"),
    )
    duplicate = frame.slice(2, 1).with_columns(pl.lit("999.0").alias("overall_mae_mgdl"))
    archive = pl.concat([new_games, duplicate]).drop("participant_key", strict=False)
    archive.write_csv(folder / "archived_2026-08-19.csv")
    return export, frame, new_games


def test_archives_are_found_beside_the_export(tmp_path: Path) -> None:
    export, _, _ = _export_with_archive(tmp_path, linked=False)
    assert [p.name for p in statistics_csv_paths(export)] == [
        "prediction_statistics.csv",
        "archived_2026-08-19.csv",
    ]


def test_loader_adds_archived_games_and_keeps_the_export_row(tmp_path: Path) -> None:
    export, frame, new_games = _export_with_archive(tmp_path, linked=False)
    runs = load_prediction_statistics(export)
    assert runs.height == frame.height + 2
    assert set(new_games["run_id"]) <= set(runs["run_id"])
    duplicate_run = frame["run_id"][2]
    kept = runs.filter(pl.col("run_id") == duplicate_run)
    assert kept.height == 1
    assert kept["overall_mae_mgdl"][0] != 999.0


def test_unlinked_archived_ids_count_as_their_own_person(tmp_path: Path) -> None:
    export, frame, new_games = _export_with_archive(tmp_path, linked=True)
    runs = load_prediction_statistics(export)
    archived = runs.filter(pl.col("run_id").is_in(new_games["run_id"].to_list()))
    assert archived["study_id"].to_list() == archived["original_study_id"].to_list()
    linked = runs.filter(~pl.col("run_id").is_in(new_games["run_id"].to_list()))
    assert all(key.startswith("person-") for key in linked["study_id"].to_list())


def test_anonymized_copy_keeps_the_archive(tmp_path: Path) -> None:
    export, frame, _ = _export_with_archive(tmp_path / "raw", linked=False)
    dest = tmp_path / "anonymized" / "prediction_statistics.csv"
    result = anonymize_statistics_csv(export, dest)
    assert (dest.parent / "archived_2026-08-19.csv").is_file()
    assert result.n_rows == frame.height + 3
    assert load_prediction_statistics(dest).height == frame.height + 2


def test_fetch_copies_and_scrubs_archives(tmp_path: Path) -> None:
    export, _, _ = _export_with_archive(tmp_path / "server", linked=False)
    archive = export.with_name("archived_2026-08-19.csv")
    pl.read_csv(archive, infer_schema_length=0).with_columns(
        pl.lit("someone@example.org").alias("email")
    ).write_csv(archive)
    dest = tmp_path / "raw" / "prediction_statistics.csv"
    result = fetch_statistics(source=export, dest=dest)
    copied = dest.with_name("archived_2026-08-19.csv")
    assert result.archives == [copied]
    assert set(pl.read_csv(copied, infer_schema_length=0)["email"].fill_null("").to_list()) == {""}


def test_game_counter_view_counts_live_browsers_and_archived_ids(tmp_path: Path) -> None:
    export, frame, new_games = _export_with_archive(tmp_path, linked=True)
    view = game_counter_view(export)
    assert view == {
        "browser_ids": frame["study_id"].n_unique(),
        "people": frame["participant_key"].n_unique(),
        "archived": new_games["study_id"].n_unique(),
    }
    assert game_counter_view(tmp_path / "missing.csv") is None
