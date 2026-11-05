"""Pull a sugar-sugar ``prediction_statistics.csv`` onto this machine."""

from __future__ import annotations

import os
import posixpath
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
from eliot import start_action

from sugar_data_processing.config import (
    DEFAULT_RAW_CSV,
    DEFAULT_SIBLING_STATS,
    NATIVE_AI_CSV_NAME,
    REPO_ROOT,
)
from sugar_data_processing.gathering.load import ARCHIVE_GLOB

SCRUB_COLUMNS: tuple[str, ...] = ("email", "location", "paper_full_name")


@dataclass(frozen=True)
class FetchResult:
    dest: Path
    source: str
    n_rows: int
    n_participants: int
    columns: list[str]
    scrubbed: list[str]
    archives: list[Path] = field(default_factory=list)
    native_ai: Path | None = None
    native_ai_rows: int = 0


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
        raise ValueError(f"Remote must be host:/path or user@host:/path (got {spec!r})")
    host, path = text.split(":", 1)
    host = host.strip()
    path = path.strip()
    if not host or not path:
        raise ValueError(f"Remote must be host:/path (got {spec!r})")
    if host.endswith("\\") or (len(host) == 1 and host.isalpha()):
        raise ValueError(f"Looks like a local Windows path, not an SSH remote: {spec!r}")
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


def _scp_file(
    host: str,
    remote_path: str,
    dest: Path,
    *,
    identity: Path | None,
    batch: bool,
    missing_ok: bool = False,
) -> bool:
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
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0 or not partial.is_file():
        if partial.exists():
            partial.unlink()
        err = f"{completed.stderr or ''}{completed.stdout or ''}"
        if missing_ok and _remote_file_missing(err):
            return False
        raise RuntimeError(
            f"scp failed ({completed.returncode}) for {host}:{remote_path}. "
            "Check SSH access, then retry. Use --ask-pass if this host needs a password."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial.replace(dest)
    return True


def companion_remote_path(remote_csv: str) -> str:
    """Same directory as the human CSV, file name ``prediction_statistics_ai.csv``."""
    parent, sep, _name = remote_csv.rpartition("/")
    if not sep:
        parent, sep, _name = remote_csv.rpartition("\\")
    if not sep:
        return NATIVE_AI_CSV_NAME
    return f"{parent}{sep}{NATIVE_AI_CSV_NAME}"


def _remote_file_missing(err: str) -> bool:
    text = err.lower()
    return "no such file" in text or "not a regular file" in text


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

        companion_local: Path | None = None
        companion_remote: tuple[str, str] | None = None
        try:
            if source is not None:
                src_path = source.expanduser().resolve()
                if not src_path.is_file():
                    raise FileNotFoundError(f"Local source CSV not found: {src_path}")
                label = str(src_path)
                companion_local = src_path.with_name(NATIVE_AI_CSV_NAME)
                shutil.copy2(src_path, incoming)
                archive_sources = sorted(src_path.parent.glob(ARCHIVE_GLOB))
            elif sibling:
                src_path = DEFAULT_SIBLING_STATS
                if not src_path.is_file():
                    raise FileNotFoundError(
                        f"Sibling export not found: {src_path}. "
                        "That path is only a local sugar-sugar checkout, not the live server."
                    )
                label = str(src_path)
                companion_local = src_path.with_name(NATIVE_AI_CSV_NAME)
                shutil.copy2(src_path, incoming)
                archive_sources = sorted(src_path.parent.glob(ARCHIVE_GLOB))
            else:
                spec = resolve_remote_spec(remote)
                if spec is None:
                    raise ValueError(
                        "No remote configured. Pass --remote user@host:/path, "
                        "set SUGAR_REMOTE in .env, or use --sibling for the local checkout."
                    )
                host, remote_path = split_remote(spec)
                label = f"{host}:{remote_path}"
                companion_remote = (host, companion_remote_path(remote_path))
                _scp_file(host, remote_path, incoming, identity=identity, batch=batch)
                archive_sources = []

            blanked = scrub_statistics_csv(incoming, dest, keep_contact=keep_contact)
            if source is None and not sibling:
                archives = _fetch_remote_archives(
                    host, remote_path, dest, identity=identity, batch=batch, keep_contact=keep_contact
                )
            else:
                archives = [
                    _scrub_archive(path, dest, keep_contact=keep_contact) for path in archive_sources
                ]
        finally:
            incoming.unlink(missing_ok=True)

        native_ai, native_ai_rows = _fetch_native_companion(
            dest.with_name(NATIVE_AI_CSV_NAME),
            local=companion_local,
            remote=companion_remote,
            identity=identity,
            batch=batch,
            keep_contact=keep_contact,
        )
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
            archives=[str(path) for path in archives],
            native_ai=str(native_ai) if native_ai is not None else "",
            native_ai_rows=native_ai_rows,
        )
        return FetchResult(
            dest=dest,
            source=label,
            n_rows=stored.height,
            n_participants=n_participants,
            columns=list(stored.columns),
            scrubbed=blanked,
            archives=archives,
            native_ai=native_ai,
            native_ai_rows=native_ai_rows,
        )


def _scrub_archive(src: Path, dest: Path, *, keep_contact: bool) -> Path:
    """Scrub one ``archived_*.csv`` next to the fetched export, keeping its name."""
    out = dest.with_name(src.name)
    scrub_statistics_csv(src, out, keep_contact=keep_contact)
    return out


def _fetch_remote_archives(
    host: str,
    remote_path: str,
    dest: Path,
    *,
    identity: Path | None,
    batch: bool,
    keep_contact: bool,
) -> list[Path]:
    """Copy any ``archived_*.csv`` beside the remote export; none is not an error."""
    pattern = posixpath.join(posixpath.dirname(remote_path), ARCHIVE_GLOB)
    cmd: list[str] = ["scp"]
    if batch:
        cmd.extend(["-o", "BatchMode=yes"])
    if identity is not None:
        cmd.extend(["-i", str(identity)])
    with tempfile.TemporaryDirectory(dir=dest.parent) as staging:
        completed = subprocess.run([*cmd, f"{host}:{pattern}", staging], check=False, capture_output=True)
        if completed.returncode != 0:
            return []
        return [
            _scrub_archive(path, dest, keep_contact=keep_contact)
            for path in sorted(Path(staging).glob(ARCHIVE_GLOB))
        ]


def _fetch_native_companion(
    dest: Path,
    *,
    local: Path | None,
    remote: tuple[str, str] | None,
    identity: Path | None,
    batch: bool,
    keep_contact: bool,
) -> tuple[Path | None, int]:
    """Copy ``prediction_statistics_ai.csv`` when the source has one.

    A missing remote file is normal until the app has written in-game forecasts.
    Any other transfer failure is raised.
    """
    download = dest.with_name(dest.name + ".download")
    if download.exists():
        download.unlink()
    try:
        if remote is not None:
            host, remote_path = remote
            found = _scp_file(
                host,
                remote_path,
                download,
                identity=identity,
                batch=batch,
                missing_ok=True,
            )
            if not found:
                dest.unlink(missing_ok=True)
                return None, 0
        elif local is not None and local.is_file():
            shutil.copy2(local, download)
        else:
            dest.unlink(missing_ok=True)
            return None, 0
        scrub_statistics_csv(download, dest, keep_contact=keep_contact)
    finally:
        download.unlink(missing_ok=True)
    stored = pl.read_csv(dest, infer_schema_length=0)
    return dest, stored.height
