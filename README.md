# sugar-data-processing

Statistical analysis library for **[Sugar Sugar](https://github.com/GlucoseDAO/sugar-sugar)** study results.

It implements **Section 7 (Statistical Analysis Plan)** of the study design
(*Human Prediction of Next-Hour Glucose from Prior CGM Context*).

The package is split into **five stages** so each part can be imported as a
library module, exercised in a Jupyter notebook, or run end-to-end via the CLI.

| Stage | Module | What it does |
| --- | --- | --- |
| **1. Data gathering** | `sugar_data_processing.gathering` | Load `prediction_statistics.csv`, explode rounds, build person-level table |
| **2. Data verification** | `sugar_data_processing.verification` | Schema / structural checks + demographic & metric quality flags |
| **3. Statistical tests** | `sugar_data_processing.statistics` | H1–H5 (§7.3–7.4) on person-level MAE |
| **4. Data comparison** | `sugar_data_processing.comparison` | Human MAE vs GlucoBench / literature bands (§7.5) |
| **5. Output** | `sugar_data_processing.output` | PNG figures + markdown / JSON report |

Orchestration: `sugar_data_processing.pipeline.run_analysis`.

## Hypotheses

| Hypothesis | Question | Test path |
| --- | --- | --- |
| **H1** | PwD vs non-PwD MAE | Shapiro → t-test / Mann–Whitney |
| **H2** | CGM vs non-CGM MAE | Shapiro → t-test / Mann–Whitney |
| **H3** | Diabetes duration vs MAE | Pearson / Spearman + log exploratory |
| **H4** | CGM experience vs MAE | Pearson / Spearman + log exploratory |
| **H5** | Own vs generic MAE (paired) | Shapiro on diffs → paired t / Wilcoxon |
| **H6** | Human vs baseline models | Deferred until baselines exist in sugar-sugar |

Person-level MAE (mean of round MAEs) is the analysis unit so repeated rounds
from the same participant do not inflate degrees of freedom.

## What data it uses

| Input | Path / source | Role |
| --- | --- | --- |
| **Real export** | `data/raw/prediction_statistics.csv` (gitignored) | sugar-sugar app statistics export |
| **Synthetic fixture** | `data/fixtures/synthetic_prediction_statistics.csv` | Committed demo cohort sized for H1–H5 |
| **CLI override** | `--csv /path/to/prediction_statistics.csv` | Explicit file, any location |

Required columns (enforced on load + re-checked in verification):

`study_id`, `run_id`, `timestamp`, `format`, `is_example_data`, `uses_cgm`,
`cgm_duration_years`, `diabetic`, `diabetes_duration`, `rounds_played`,
`overall_mae_mgdl`, `overall_rmse_mgdl`, `overall_mape_pct`, `per_round_metrics`

Email / location / raw upload filenames are dropped on load (pseudonymization).

### Analysis population (§7.2)

- **Primary (H1–H4):** ≥6 generic segments (format A, or odd rounds in mixed C)
- **Own-data:** ≥6 own segments (format B, or even rounds in mixed C)
- **H5:** participants meeting both thresholds

## Layout

```
src/sugar_data_processing/
  gathering/       # stage 1 — load CSV, explode rounds, participant table
  verification/    # stage 2 — schema checks + quality flags
  statistics/      # stage 3 — H1–H5 tests + effect sizes
  comparison/      # stage 4 — GlucoBench / literature bands
  output/          # stage 5 — plots + markdown/JSON report
  fixtures/        # synthetic CSV generator
  pipeline.py      # run_analysis() wires stages 1→5
  cli.py           # Typer entry point
notebooks/
  study_analysis.ipynb   # thin walkthrough over the library (same paths as CLI)
data/
  raw/             # drop real prediction_statistics.csv here (gitignored)
  fixtures/        # committed synthetic demo data
  processed/       # parquet/csv intermediates (written by output stage)
output/
  figures/         # PNGs
  reports/         # study_analysis_report.md + .json (+ reports/figures/)
tests/             # pytest (real synthetic data, no mocks)
docs/
  analysis-plan.md # study design §7 → module map
```

## Setup

```bash
uv sync
```

For the notebook (dev group includes Jupyter):

```bash
uv sync --group dev
```

## How to run the whole thing

End-to-end on the synthetic fixture (recommended first run):

```bash
uv sync
uv run sugar-data-processing make-fixture
uv run sugar-data-processing analyze --fixture
```

Then open:

- `output/reports/study_analysis_report.md`
- `output/reports/study_analysis_report.json`
- `output/figures/*.png`

Short alias:

```bash
uv run sdp analyze --fixture
```

### Real sugar-sugar export

```bash
cp /path/to/sugar-sugar/data/input/prediction_statistics.csv data/raw/
uv run sugar-data-processing analyze
```

Or pass an explicit path:

```bash
uv run sugar-data-processing analyze --csv /path/to/prediction_statistics.csv -o output
```

Resolution order for `analyze` without `--csv` / `--fixture`:

1. `data/raw/prediction_statistics.csv` if present
2. else fixture (with a yellow warning)

### Jupyter (same library modules and same report folder)

```bash
uv sync --group dev
uv run sugar-data-processing make-fixture
uv run jupyter notebook notebooks/study_analysis.ipynb
```

Select the `sugar-data-processing (.venv)` kernel. The notebook only calls library
APIs (`prepare_session`, gathering, verification, statistics, comparison, output).
It writes to the same place as the CLI: `output/reports/` (no notebook-specific
report directory).

### Library import

```python
from pathlib import Path
from sugar_data_processing import run_analysis

result = run_analysis(
    Path("data/fixtures/synthetic_prediction_statistics.csv"),
    Path("output"),
)
print(result.verification.passed, result.report_path)
```

Or stage-by-stage:

```python
from pathlib import Path

import polars as pl

from sugar_data_processing.gathering import load_prediction_statistics, build_participant_table
from sugar_data_processing.verification import verify_dataset
from sugar_data_processing.statistics import run_all_hypotheses
from sugar_data_processing.comparison import benchmark_context
from sugar_data_processing.output import write_report

csv = Path("data/fixtures/synthetic_prediction_statistics.csv")
runs = load_prediction_statistics(csv)
participants = build_participant_table(runs)
verification = verify_dataset(runs, participants)
suite = run_all_hypotheses(participants)
eligible = participants.filter(pl.col("eligible_primary"))
benchmarks = benchmark_context(eligible if eligible.height > 0 else participants)
write_report(
    runs=runs,
    participants=participants,
    suite=suite,
    benchmarks=benchmarks,
    verification=verification,
    output_dir=Path("output"),
    source_csv=csv,
)
```

## What each stage produces

| Stage | In-memory | On disk (via `analyze` / `write_report`) |
| --- | --- | --- |
| Gathering | `runs`, `participants` Polars frames | — |
| Verification | `VerificationReport` (schema + quality) | Embedded in report JSON / markdown §6 |
| Statistics | `HypothesisSuite` (H1–H5) | Report §3–4 + JSON `hypotheses` |
| Comparison | `BenchmarkContext` | Report §5 + JSON `benchmarks` |
| Output | report path | `output/reports/*`, `output/figures/*`, `data/processed/*` |

### Verification details

- **Schema:** required columns, non-empty IDs, format ∈ {A,B,C}, parseable
  `per_round_metrics`, non-negative MAE, `rounds_played` vs metrics length
- **Quality flags:** implausible age/durations, flag mismatches, MAE IQR
  outliers, short sessions, duplicate `run_id`, below-threshold participants

High-severity **schema** issues set `verification.schema_ok = False`. Quality
flags are reported but do not alone fail verification (they inform the report).

## Tests

```bash
uv run pytest
```

## Notes

- H6 is intentionally not implemented yet (study design defers baseline models).
- See `docs/analysis-plan.md` for the §7 → module mapping and format conventions.
