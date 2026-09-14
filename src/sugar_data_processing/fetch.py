"""Pull a sugar-sugar ``prediction_statistics.csv`` onto this machine."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from eliot import start_action

from sugar_data_processing.config import (
    DEFAULT_RAW_CSV,
    DEFAULT_SIBLING_STATS,
    REPO_ROOT,
)

SCRUB_COLUMNS: tuple[str, ...] = ("email", "location", "paper_full_name")


@dataclass(frozen=True)
class FetchResult:
    dest: Path
    source: str
    n_rows: int
    n_participants: int
    columns: list[str]
    scrubbed: list[str]


def load_repo_dotenv(path: Path | None = None) -> Path | None:
    """Load ``KEY=VALUE`` lines from the repo ``.env`` without overriding the real env."""
    env_path = path or (REPO_ROOT / ".env")
    if not env_path.is_file():
        return None
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value
    return env_path


def split_remote(spec: str) -> tuple[str, str]:
    """Split ``user@host:/abs/path`` (or ``host:/abs/path``) into host and file path."""
    text = spec.strip()
    if ":" not in text:
        raise ValueError(
            "Remote must be host:/path or user@host:/path "
            f"(got {spec!r})"
        )
    host, path = text.split(":", 1)
    host = host.strip()
    path = path.strip()
    if not host or not path:
        raise ValueError(f"Remote must be host:/path (got {spec!r})")
    if host.endswith("\\") or (len(host) == 1 and host.isalpha()):
        raise ValueError(
            f"Looks like a local Windows path, not an SSH remote: {spec!r}"
        )
    if not path.lower().endswith(".csv"):
        path = path.rstrip("/") + "/prediction_statistics.csv"
    return host, path


def resolve_remote_spec(explicit: str | None) -> str | None:
    if explicit and explicit.strip():
        return explicit.strip()
    composed_or_env = os.environ.get("SUGAR_REMOTE", "").strip()
    if composed_or_env:
        return composed_or_env
    host = os.environ.get("SUGAR_SSH_HOST", "").strip()
    path = os.environ.get("SUGAR_SSH_STATS", "").strip()
    if host and path:
        return f"{host}:{path}"
    return None


def resolve_identity(explicit: Path | None) -> Path | None:
    """SSH private key from ``--identity`` or ``SUGAR_SSH_IDENTITY``."""
    if explicit is not None:
        return explicit.expanduser()
    raw = os.environ.get("SUGAR_SSH_IDENTITY", "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def scrub_statistics_csv(src: Path, dest: Path, *, keep_contact: bool = False) -> list[str]:
    """Rewrite ``src`` to ``dest`` with contact columns blanked.

    Reads every column as text so quoted list fields survive the round-trip.
    """
    df = pl.read_csv(src, infer_schema_length=0)
    blanked: list[str] = []
    if not keep_contact:
        updates: list[pl.Expr] = []
        for col in SCRUB_COLUMNS:
            if col in df.columns:
                updates.append(pl.lit("").alias(col))
                blanked.append(col)
        if updates:
            df = df.with_columns(updates)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    df.write_csv(tmp)
    tmp.replace(dest)
    return blanked


def _scp_file(host: str, remote_path: str, dest: Path, *, identity: Path | None, batch: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    if partial.exists():
        partial.unlink()
    cmd: list[str] = ["scp"]
    if batch:
        cmd.extend(["-o", "BatchMode=yes"])
    if identity is not None:
        cmd.extend(["-i", str(identity)])
    cmd.extend([f"{host}:{remote_path}", str(partial)])
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0 or not partial.is_file():
        if partial.exists():
            partial.unlink()
        raise RuntimeError(
            f"scp failed ({completed.returncode}) for {host}:{remote_path}. "
            "Check SSH access, then retry. Use --ask-pass if this host needs a password."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial.replace(dest)


def fetch_statistics(
    *,
    dest: Path = DEFAULT_RAW_CSV,
    remote: str | None = None,
    sibling: bool = False,
    source: Path | None = None,
    identity: Path | None = None,
    batch: bool = True,
    keep_contact: bool = False,
) -> FetchResult:
    """Copy a statistics CSV from SSH, the sibling checkout, or a local path."""
    load_repo_dotenv()
    identity = resolve_identity(identity)
    with start_action(action_type="fetch.statistics", dest=str(dest)) as action:
        dest.parent.mkdir(parents=True, exist_ok=True)
        incoming = dest.with_name(dest.name + ".incoming")
        if incoming.exists():
            incoming.unlink()

        try:
            if source is not None:
                src_path = source.expanduser().resolve()
                if not src_path.is_file():
                    raise FileNotFoundError(f"Local source CSV not found: {src_path}")
                label = str(src_path)
                shutil.copy2(src_path, incoming)
            elif sibling:
                src_path = DEFAULT_SIBLING_STATS
                if not src_path.is_file():
                    raise FileNotFoundError(
                        f"Sibling export not found: {src_path}. "
                        "That path is only a local sugar-sugar checkout, not the live server."
                    )
                label = str(src_path)
                shutil.copy2(src_path, incoming)
            else:
                spec = resolve_remote_spec(remote)
                if spec is None:
                    raise ValueError(
                        "No remote configured. Pass --remote user@host:/path, "
                        "set SUGAR_REMOTE in .env, or use --sibling for the local checkout."
                    )
                host, remote_path = split_remote(spec)
                label = f"{host}:{remote_path}"
                _scp_file(host, remote_path, incoming, identity=identity, batch=batch)

            blanked = scrub_statistics_csv(incoming, dest, keep_contact=keep_contact)
        finally:
            incoming.unlink(missing_ok=True)

        stored = pl.read_csv(dest, infer_schema_length=0)
        n_participants = 0
        if "study_id" in stored.columns:
            n_participants = stored["study_id"].n_unique()
        action.log(
            message_type="info",
            source=label,
            n_rows=stored.height,
            n_participants=n_participants,
            scrubbed=blanked,
            has_round_context="round_context" in stored.columns,
        )
        return FetchResult(
            dest=dest,
            source=label,
            n_rows=stored.height,
            n_participants=n_participants,
            columns=list(stored.columns),
            scrubbed=blanked,
        )
