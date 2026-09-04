# Mapping: study design §7 → this library

Source document: `sugar-sugar/data/input/study_design/The study - technical Guidebook.md`

| Pipeline stage | Module | Study design |
| --- | --- | --- |
| 1. Data gathering | `gathering/` | Load export; §7.2 populations (`eligible_primary`, `eligible_own`, `eligible_h5`) |
| 2. Data verification | `verification/` | Schema + demographic / metric quality flags |
| 3. Statistical tests | `statistics/` | §7.3 H1–H2; §7.4 H3–H5 |
| 4. Data comparison | `comparison/` | §7.5 literature / GlucoBench bands |
| 5. Output | `output/` | Markdown + JSON + figures + `study_explorer.html` |

| Study design section | Symbol |
| --- | --- |
| §7.2 Analysis populations | `gathering.participants` |
| §7.3 H1 / H2 | `statistics.hypotheses` + `statistics.tests.independent_group_comparison` |
| §7.4 H3 / H4 | `statistics.tests.correlation_analysis` |
| §7.4 H5 | `statistics.tests.paired_comparison` |
| §7.4 H6 | Deferred note in report only |
| §7.5 Literature bands | `comparison.benchmarks` |
| Data quality | `verification.schema` + `verification.anomalies` |

## Format conventions from current sugar-sugar

| Format | Meaning | Segment rule used here |
| --- | --- | --- |
| A | Generic / example | all rounds → `generic` |
| B | Own upload | all rounds → `own` |
| C | Mixed | per-round `is_example_data` (from `per_round_metrics` or `round_context`); legacy fallback is odd → generic, even → own |

The run-level `data_source_name` / `is_example_data` columns are only the **last** source of that session. Format C (and challenge-unknown mixes) must be reconstructed from the per-round fields.

### Export encodings this loader accepts

| Field | Current sugar-sugar form | How this library stores it |
| --- | --- | --- |
| `cgm_duration_years` | `value,unit` (e.g. `6,months`) or a bare year | float years |
| `per_round_metrics` | list of `{round_number, mae, …, data_source_name, is_example_data, generic_slice_key}` | exploded into the round table |
| `round_context` | list of window/source metadata per round | joined onto the same round rows |
| `generic_intervention` | pool policy (`d1namo`, `bigideas`, `mix:…`) | kept on the participant row |
| `challenge_unknown` / `paper_mention` | optional bools | three-state bools (blank stays unknown) |

Personal upload filenames are redacted to `own_upload`. Published corpus names (`D1NAMO-001.csv`, `BIGIDEAS-003.csv`, `example.csv`) are kept so diabetic vs non-diabetic **traces** can be classified the same way the app scoreboard does.

### Extra person-level columns

| Column | Meaning |
| --- | --- |
| `cohort_category` | one of the four diabetes × CGM buckets |
| `n_runs` / `is_repeat_player` | every saved session, not only the latest per format |
| `n_formats_played` / `played_all_formats` | whether the person saved A, B, and C |
| `mae_format_a` / `b` / `c` | person MAE on that task |
