# Mapping: study design §7 → this library

Source document: `sugar-sugar/data/input/study_design/The study - technical Guidebook.md`

| Pipeline stage | Module | Study design |
| --- | --- | --- |
| 1. Data gathering | `gathering/` | Load export; §7.2 populations (`eligible_primary`, `eligible_own`, `eligible_h5`) |
| 2. Data verification | `verification/` | Schema + demographic / metric quality flags |
| 3. Statistical tests | `statistics/` | §7.3 H1–H2; §7.4 H3–H5 |
| 4. Data comparison | `comparison/` | §7.5 literature / GlucoBench bands |
| 5. Output | `output/` | Human-readable markdown + JSON + figures |

| Study design section | Symbol |
| --- | --- |
| §7.2 Analysis populations | `gathering.participants` |
| §7.3 H1 / H2 | `statistics.hypotheses` + `statistics.tests.independent_group_comparison` |
| §7.4 H3 / H4 | `statistics.tests.correlation_analysis` |
| §7.4 H5 | `statistics.tests.paired_comparison` |
| §7.4 H6 | Deferred note in report only |
| §7.5 Literature bands | `comparison.benchmarks` |
| Data quality | `verification.schema` + `verification.anomalies` |

## Format conventions from sugar-sugar

| Format | Meaning | Segment rule used here |
| --- | --- | --- |
| A | Generic / example | all rounds → `generic` |
| B | Own upload | all rounds → `own` |
| C | Mixed | odd rounds → `generic`, even → `own` |
