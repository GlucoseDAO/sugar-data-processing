"""Sugar Sugar study results analysis library.

Public stages (publishable package surface):

* :mod:`sugar_data_processing.gathering` — load and reshape exports
* :mod:`sugar_data_processing.verification` — schema + quality checks
* :mod:`sugar_data_processing.statistics` — H1–H5 tests
* :mod:`sugar_data_processing.comparison` — literature / GlucoBench bands
* :mod:`sugar_data_processing.output` — figures and reports

Orchestration: :func:`sugar_data_processing.pipeline.run_analysis`.
CLI entry points: ``sugar-data-processing`` / ``sdp``.
"""

from sugar_data_processing.cli import app
from sugar_data_processing.pipeline import AnalysisResult, run_analysis

__all__ = ["app", "run_analysis", "AnalysisResult"]


def main() -> None:
    app()
