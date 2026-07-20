# sugar-data-processing

Statistical analysis pipeline for **[Sugar Sugar](https://github.com/GlucoseDAO/sugar-sugar)** study results.

It follows **Section 7 (Statistical Analysis Plan)** of the study design
(*Human Prediction of Next-Hour Glucose from Prior CGM Context*):

| Hypothesis | Question | Test path |
| --- | --- | --- |
| **H1** | PwD vs non-PwD MAE | Shapiro → t-test / Mann–Whitney |
| **H2** | CGM vs non-CGM MAE | Shapiro → t-test / Mann–Whitney |
| **H3** | Diabetes duration vs MAE | Pearson / Spearman + log exploratory |
| **H4** | CGM experience vs MAE | Pearson / Spearman + log exploratory |
| **H5** | Own vs generic MAE (paired) | Shapiro on diffs → paired t / Wilcoxon |
| **H6** | Human vs baseline models | Deferred until baselines exist in sugar-sugar |

Person-level MAE (mean of round MAEs) is the analysis unit so repeated
rounds from the same participant do not inflate degrees of freedom.

## Layout

```
src/sugar_data_processing/
  extraction/     # load CSV, explode rounds, build participant table
  statistics/     # H1–H5 tests + effect sizes
  comparison/     # GlucoBench bands + anomaly flags
  output/         # plots + markdown/JSON report
  fixtures/       # synthetic CSV generator
data/
  raw/            # drop real prediction_statistics.csv here (gitignored)
  fixtures/       # committed synthetic demo data
  processed/      # parquet/csv intermediates
output/
  figures/        # PNGs
  reports/        # study_analysis_report.md + .json
```

## Setup

```bash
uv sync
```

## Quick start

Generate a synthetic cohort (already sized for H1–H5) and analyze it:

```bash
uv run sugar-data-processing make-fixture
uv run sugar-data-processing analyze --fixture
```

Open `output/reports/study_analysis_report.md`.

### Real sugar-sugar export

Copy the app export into `data/raw/`:

```bash
cp /path/to/sugar-sugar/data/input/prediction_statistics.csv data/raw/
uv run sugar-data-processing analyze
```

Or pass an explicit path:

```bash
uv run sugar-data-processing analyze --csv /path/to/prediction_statistics.csv -o output
```

Email / location / raw upload filenames are dropped on load so reports stay
pseudonymized.

## Analysis population (§7.2)

- **Primary (H1–H4):** ≥6 generic segments (format A, or odd rounds in mixed C)
- **Own-data:** ≥6 own segments (format B, or even rounds in mixed C)
- **H5:** participants meeting both thresholds (relaxed pairing is noted in the report when early data is sparse)

## Tests

```bash
uv run pytest
```

## Notes

- H6 is intentionally not implemented yet (study design defers baseline models).
- Anomaly detection flags implausible durations (e.g. CGM years > age),
  MAE IQR outliers, short sessions, and flag mismatches.
- Graphics are written as PNG and embedded in the markdown report.
