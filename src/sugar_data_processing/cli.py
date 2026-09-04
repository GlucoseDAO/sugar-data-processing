"""Typer CLI for the Sugar Sugar analysis pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from eliot import to_file
from pycomfort.logging import to_nice_file, to_nice_stdout
from rich.console import Console

from sugar_data_processing.config import (
    DEFAULT_FIXTURE_CSV,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RAW_CSV,
    REPO_ROOT,
)
from sugar_data_processing.fixtures.synthetic import write_synthetic_csv
from sugar_data_processing.pipeline import run_analysis

app = typer.Typer(
    name="sugar-data-processing",
    help="Analyze Sugar Sugar study results per the study-design statistical plan.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


def _configure_logging() -> None:
    logs = REPO_ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    to_nice_stdout()
    to_nice_file(logs / "eliot.jsonl", logs / "sugar_data_processing.log")
    to_file(open(logs / "eliot_raw.jsonl", "a", encoding="utf-8"))


@app.command("analyze")
def analyze(
    csv: Optional[Path] = typer.Option(
        None,
        "--csv",
        help="Path to sugar-sugar prediction_statistics.csv",
        exists=False,
        dir_okay=False,
        readable=True,
    ),
    output: Path = typer.Option(
        DEFAULT_OUTPUT_DIR,
        "--output",
        "-o",
        help="Output directory for reports and figures",
        file_okay=False,
    ),
    use_fixture: bool = typer.Option(
        False,
        "--fixture",
        help="Use bundled synthetic fixture instead of --csv / data/raw",
    ),
) -> None:
    """Gather → verify → test H1–H5 → compare → write markdown report and HTML explorer."""
    _configure_logging()
    if use_fixture:
        csv_path = DEFAULT_FIXTURE_CSV
        if not csv_path.exists():
            write_synthetic_csv(csv_path)
    elif csv is not None:
        csv_path = csv
    elif DEFAULT_RAW_CSV.exists():
        csv_path = DEFAULT_RAW_CSV
    elif DEFAULT_FIXTURE_CSV.exists():
        console.print(
            f"[yellow]No raw CSV at {DEFAULT_RAW_CSV}; falling back to fixture.[/yellow]"
        )
        csv_path = DEFAULT_FIXTURE_CSV
    else:
        console.print(
            "[red]No input CSV found.[/red] Pass --csv, place a file at "
            f"{DEFAULT_RAW_CSV}, or use --fixture."
        )
        raise typer.Exit(code=1)

    if not csv_path.exists():
        console.print(f"[red]CSV not found:[/red] {csv_path}")
        raise typer.Exit(code=1)

    console.print(f"[bold]Analyzing[/bold] {csv_path}")
    result = run_analysis(csv_path, output)
    v = result.verification
    schema_label = "passed" if v.schema_ok else "FAILED"
    console.print(f"[green]Report written:[/green] {result.report_path}")
    console.print(
        f"Participants: {result.participants.height} | "
        f"Verification: schema {schema_label}, "
        f"{len(v.all_issues)} issues ({v.n_high} high) | "
        f"Mean MAE: {result.benchmarks.human_mean_mae:.2f} mg/dL"
    )


@app.command("make-fixture")
def make_fixture(
    path: Path = typer.Option(
        DEFAULT_FIXTURE_CSV,
        "--path",
        help="Where to write the synthetic prediction_statistics.csv",
    ),
    n_participants: int = typer.Option(120, "--n", min=20, help="Number of synthetic participants"),
    seed: int = typer.Option(7, "--seed", help="RNG seed"),
) -> None:
    """Generate a synthetic statistics CSV that exercises H1–H5."""
    written = write_synthetic_csv(path, n_participants=n_participants, seed=seed)
    console.print(f"[green]Wrote fixture:[/green] {written}")


if __name__ == "__main__":
    app()
