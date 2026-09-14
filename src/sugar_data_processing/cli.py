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
    DEFAULT_SIBLING_STATS,
    REPO_ROOT,
)
from sugar_data_processing.fetch import fetch_statistics, load_repo_dotenv
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


def _run_analyze(csv_path: Path, output: Path) -> None:
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

    _run_analyze(csv_path, output)


@app.command("fetch")
def fetch(
    remote: Optional[str] = typer.Option(
        None,
        "--remote",
        help="SSH spec: host:/path or user@host:/path to the CSV or its data/input directory",
    ),
    sibling: bool = typer.Option(
        False,
        "--sibling",
        help=f"Copy from the local sugar-sugar checkout ({DEFAULT_SIBLING_STATS})",
    ),
    source: Optional[Path] = typer.Option(
        None,
        "--source",
        help="Local path to an already-downloaded prediction_statistics.csv",
        exists=False,
        dir_okay=False,
    ),
    dest: Path = typer.Option(
        DEFAULT_RAW_CSV,
        "--dest",
        help="Where to write the scrubbed CSV (analyze reads this by default)",
        dir_okay=False,
    ),
    identity: Optional[Path] = typer.Option(
        None,
        "--identity",
        "-i",
        help="SSH private key (passed to scp -i)",
        exists=True,
        dir_okay=False,
    ),
    keep_contact: bool = typer.Option(
        False,
        "--keep-contact",
        help="Keep email / location / paper name (only on a machine you trust with prod)",
    ),
    ask_pass: bool = typer.Option(
        False,
        "--ask-pass",
        help="Allow SSH password prompts (default is BatchMode, key-only)",
    ),
    then_analyze: bool = typer.Option(
        False,
        "--analyze",
        help="Run the analysis pipeline after a successful fetch",
    ),
    output: Path = typer.Option(
        DEFAULT_OUTPUT_DIR,
        "--output",
        "-o",
        help="Output directory used with --analyze",
        file_okay=False,
    ),
) -> None:
    """Copy the live (or sibling) statistics CSV here, with contact fields blanked.

    Configure the SSH target once in ``.env`` as ``SUGAR_REMOTE=user@host:/path``
    and ``SUGAR_SSH_IDENTITY`` if you pass ``-i`` to ssh. Then:

        uv run sdp fetch
        uv run sdp fetch --analyze
    """
    _configure_logging()
    load_repo_dotenv()
    try:
        pulled = fetch_statistics(
            dest=dest,
            remote=remote,
            sibling=sibling,
            source=source,
            identity=identity,
            batch=not ask_pass,
            keep_contact=keep_contact,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        if not remote and not sibling and source is None:
            console.print(
                "Set [bold]SUGAR_REMOTE[/bold] in .env to the same "
                "user@host:/path you already ssh to (do not commit .env)."
            )
            if DEFAULT_SIBLING_STATS.exists():
                console.print(
                    "Or pull the local checkout: [bold]uv run sdp fetch --sibling[/bold]"
                )
        raise typer.Exit(code=1) from exc

    console.print(f"[green]Wrote[/green] {pulled.dest}")
    console.print(f"Source: {pulled.source}")
    console.print(
        f"Rows: {pulled.n_rows} | Participants: {pulled.n_participants} | "
        f"Scrubbed: {', '.join(pulled.scrubbed) or 'none'}"
    )
    extra = [
        col
        for col in ("round_context", "generic_intervention", "challenge_unknown")
        if col in pulled.columns
    ]
    if extra:
        console.print(f"Current-app columns present: {', '.join(extra)}")
    if then_analyze:
        _run_analyze(pulled.dest, output)


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
