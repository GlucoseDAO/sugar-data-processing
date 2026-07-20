# Mapping: study design §7 → this repo

Source document: `sugar-sugar/data/input/study_design/The study - technical Guidebook.md`

| Study design section | Module |
| --- | --- |
| §7.2 Analysis populations | `extraction/participants.py` (`eligible_primary`, `eligible_own`, `eligible_h5`) |
| §7.3 H1 / H2 | `statistics/hypotheses.py` + `statistics/tests.py` (`independent_group_comparison`) |
| §7.4 H3 / H4 | `correlation_analysis` (+ linear vs log R²) |
| §7.4 H5 | `paired_comparison` on generic − own MAE |
| §7.4 H6 | Deferred note in report only |
| §7.5 Literature bands | `comparison/benchmarks.py` |
| Data quality / anomalies | `comparison/anomalies.py` |
| Human-readable deliverable | `output/report.py` + `output/plots.py` |

## Format conventions from sugar-sugar

| Format | Meaning | Segment rule used here |
| --- | --- | --- |
| A | Generic / example | all rounds → `generic` |
| B | Own upload | all rounds → `own` |
| C | Mixed | odd rounds → `generic`, even → `own` |
