"""Runtime helpers for notebooks and long-lived kernels."""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path


def _looks_like_repo(path: Path) -> bool:
    return (path / "pyproject.toml").exists() and (
        path / "src" / "sugar_data_processing"
    ).is_dir()


def _notebook_path(extra_globals: dict[str, object] | None = None) -> Path | None:
    scope = extra_globals or {}
    for key in ("__vsc_ipynb_file__", "__notebook_path__"):
        raw = scope.get(key)
        if raw:
            path = Path(str(raw)).expanduser().resolve()
            if path.suffix == ".ipynb":
                return path
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is not None:
            for key in ("__vsc_ipynb_file__", "__session__"):
                raw = ip.user_ns.get(key)
                if raw and str(raw).endswith(".ipynb"):
                    return Path(str(raw)).expanduser().resolve()
    except Exception:
        pass
    return None


def _seed_dirs(extra_globals: dict[str, object] | None = None) -> list[Path]:
    seeds: list[Path] = []
    nb_path = _notebook_path(extra_globals)
    if nb_path is not None:
        seeds.append(nb_path.parent)
    for env_key in (
        "SUGAR_DATA_PROCESSING_ROOT",
        "CURSOR_WORKSPACE",
        "VSCODE_CWD",
        "PWD",
    ):
        raw = os.environ.get(env_key)
        if raw:
            seeds.append(Path(raw).expanduser().resolve())
    seeds.append(Path.cwd().resolve())
    return seeds


def ensure_package_importable(
    *,
    reload: bool = True,
    extra_globals: dict[str, object] | None = None,
) -> Path | None:
    """Ensure ``sugar_data_processing`` is importable; optionally clear module cache.

    Returns the repo root when ``src/`` was added to ``sys.path``, otherwise ``None``.
    """
    seen: set[Path] = set()
    repo: Path | None = None
    for seed in _seed_dirs(extra_globals):
        for candidate in [seed, *seed.parents]:
            if candidate in seen:
                continue
            seen.add(candidate)
            if not _looks_like_repo(candidate):
                continue
            src = candidate / "src"
            if str(src) not in sys.path:
                sys.path.insert(0, str(src))
            repo = candidate
            break
        if repo is not None:
            break

    if repo is None and importlib.util.find_spec("sugar_data_processing") is None:
        raise RuntimeError(
            "Could not import sugar_data_processing. "
            "Select the project kernel after: uv sync --group dev"
        )

    if reload:
        for name in list(sys.modules):
            if name == "sugar_data_processing" or name.startswith("sugar_data_processing."):
                del sys.modules[name]
    return repo


@dataclass(frozen=True)
class AnalysisSession:
    """Standard paths used by the CLI and the walkthrough notebook."""

    repo_root: Path
    csv_path: Path
    output_dir: Path
    reports_dir: Path


def prepare_session(
    extra_globals: dict[str, object] | None = None,
    *,
    ensure_fixture: bool = True,
) -> AnalysisSession:
    """Prepare a library session for notebook or interactive use.

    Uses the same default CSV and ``output/`` tree as the CLI. Reports are always
    written under ``output/reports/``.
    """
    # Reload first so a long-lived notebook kernel picks up src/ edits
    # (e.g. newly added DEFAULT_REPORTS_DIR) before importing symbols.
    ensure_package_importable(reload=True, extra_globals=extra_globals)

    from sugar_data_processing.config import (
        DEFAULT_FIXTURE_CSV,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_REPORTS_DIR,
        REPO_ROOT,
    )
    from sugar_data_processing.fixtures.synthetic import write_synthetic_csv

    csv_path = DEFAULT_FIXTURE_CSV
    if ensure_fixture and not csv_path.exists():
        write_synthetic_csv(csv_path)

    return AnalysisSession(
        repo_root=REPO_ROOT,
        csv_path=csv_path,
        output_dir=DEFAULT_OUTPUT_DIR,
        reports_dir=DEFAULT_REPORTS_DIR,
    )
