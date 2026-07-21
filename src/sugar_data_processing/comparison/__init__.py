"""Stage 4 — Data comparison.

Place human person-level MAE against published GlucoBench / literature bands
(study design §7.5). H6 (human vs baselines in sugar-sugar) remains deferred.
"""

from sugar_data_processing.comparison.benchmarks import BenchmarkContext, benchmark_context

__all__ = ["BenchmarkContext", "benchmark_context"]
