# Mapping: study design §7 → this library

Source document: `sugar-sugar/data/input/study_design/The study - technical Guidebook.md`

| Pipeline stage | Module | Study design |
| --- | --- | --- |
| 1. Data gathering | `gathering/` | Load export; §7.2 populations (`eligible_primary`, `eligible_own`, `eligible_h5`) |
| 2. Data verification | `verification/` | Schema + demographic / metric quality flags |
| 3. Statistical tests | `statistics/` | §7.3 H1–H2; §7.4 H3–H5 |
| 4. Data comparison | `comparison/` | §7.5 literature / GlucoBench bands |
| 5. Output | `output/` | Merged markdown + JSON + figures + `explorer.html` |
| 6. AI scoring | `ai/` | Rebuild 3-hour windows; score models; same report |

| Study design section | Symbol |
| --- | --- |
| §7.2 Analysis populations | `gathering.participants` |
| §7.3 H1 / H2 | `statistics.hypotheses` + `statistics.tests.independent_group_comparison` (category snapshot, then generic / own layers) |
| §7.4 H3 / H4 | `statistics.tests.correlation_analysis` (both durations in months; both split generic / own) |
| §7.4 H5 | `statistics.tests.paired_comparison` |
| §7.4 H6 | Paired human vs model MAE on reconstructed windows |
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
| `challenge_unknown` | opt-in flag | bool only; blank / missing / older exports → `false` |
| `paper_mention` | optional bool | three-state (blank stays unknown) |

Personal upload filenames are redacted to `own_upload`. Published corpus names (`D1NAMO-001.csv`, `BIGIDEAS-003.csv`, `example.csv`) are kept so diabetic vs non-diabetic **traces** can be classified the same way the app scoreboard does.

### Extra person-level columns

| Column | Meaning |
| --- | --- |
| `cohort_category` | one of the four diabetes × CGM buckets |
| `diabetes_duration_months` | `diabetes_duration` (years) × 12 — H3 axis |
| `cgm_duration_months` | `cgm_duration_years` × 12 — H4 axis |
| `n_runs` / `is_repeat_player` | every saved session, not only the latest per format |
| `n_formats_played` / `played_all_formats` | whether the person saved A, B, and C |
| `mae_format_a` / `b` / `c` | person MAE on that task |
| `player_trait` | player's diabetes class (`diabetic` / `nondiabetic` / `unknown`) |
| `played_challenge_unknown` | opted into Challenge the unknown (opposite-corpus mix) |
| `played_opposite_trait` | at least one round whose trace class ≠ player trait |
| `mae_same_trait` / `mae_opposite_trait` | person MAE split by same vs opposite diabetes class |

Reports: one merged file (`analysis_report.md` / `explorer.html`). The explorer
has Overview, Human, AI, and People tabs. People overlays actual CGM, the human
forecast, and each AI line. Models only see the 3-hour game window.

## Corrected recruitment planning (24 October 2026)

Production data are now available locally as a contact-scrubbed CSV with a
server-prepared pseudonymous identity map. The earlier assessment based only on
an August local export is superseded by the production assessment below.

**Recommended operational target: 320 primary-eligible linked participants**,
with approximately 160 in each diabetes margin and 160 in each CGM margin.
These are overlapping quotas. This retains the config's 3.5 mg/dL target
and plans conservatively for between-person SD 10 mg/dL. The calculated minimum
is 157 per group / 314 eligible, rounded to 320. If the study instead fixes the
meaningful target at the guidebook calculation's 4 mg/dL, use 121 per group /
242 eligible, rounded operationally to **250**. Choose the difference explicitly;
do not choose it to make observed results significant. Neither threshold has
been independently established as clinically important.

The production pilot has 207 saved runs, 184 study IDs, and 155 linked identity
groups. There are **109 primary-eligible identities**, or 104 in the stricter
quality sensitivity analysis. With 36 anonymous IDs not linkable by email,
these are estimates of independent people, not independently verified humans.
Observed pooled person-level SD is 7.95 mg/dL for H1 and 8.36 for H2; planning
with SD 10 provides a margin, rather than adopting the smallest pilot estimate.

| Target difference | Planning SD | Eligible per balanced group | Eligible total |
| --- | --- | ---: | ---: |
| 4 mg/dL | observed 8.36 | 85 | 170 |
| 3.5 mg/dL | observed 8.36 | 110 | 220 |
| 4 mg/dL | 10 | 121 | 242 |
| 3.5 mg/dL | 10 | 157 | 314 |
| 4 mg/dL | 12 | 173 | 346 |
| 3.5 mg/dL | 12 | 225 | 450 |

All rows target 80% power **per** two-sided primary comparison at alpha 0.025,
using an equal-variance normal t-test planning model. This does not guarantee
80% probability of detecting both primary effects, or power for rank tests,
interactions, nonlinear learning curves or a redesigned source-adjusted model.
Those need separate planning or simulation. Repeated rounds do not multiply
independent participants.

The original 100 per group with SD 10 and a 4 mg/dL effect has approximately
71.5% power, not 80%, under the stated two-sided corrected threshold. Its
150-per-group maximum cannot guarantee adequate power at higher SDs.

### Recruitment balance matters more than the total alone

Current primary counts: 26 with diabetes vs 83 without; 34 CGM vs 75 non-CGM.
An illustrative 320-person mix is:

| Diabetes | CGM | Now | Illustrative target | Additional eligible needed |
| --- | --- | ---: | ---: | ---: |
| Yes | Yes | 22 | 100 | 78 |
| Yes | No | 4 | 60 | 56 |
| No | Yes | 12 | 60 | 48 |
| No | No | 71 | 100 | 29 |

These quotas sum to 320 and provide crossover groups to help distinguish
CGM use from diabetes status; they do not establish interaction-model power.
The operational target needs **211 additional eligible identities**, or 216
if excluding the five flagged primary identities in the quality sensitivity.

The observed eligible fraction is 109/155 (70.3%), or 104/155 (67.1%) after
quality exclusions. At those rates, plan for **456–477 recorded participant
identities**, operationally about **460–480** total. This is not a website
visitor/conversion target: people who never saved a run are absent. Rates can
change with targeted recruitment and task design. For the 250-eligible / 4 mg/dL
alternative, the analogous range is 356–373 recorded identities.

### Secondary questions

- H3/H4: approximately **124 complete participants in the relevant subgroup**
  for |r|=0.25; 194 for |r|=0.20, at nominal two-sided alpha 0.05 and 80% power
  (Fisher-z approximation for Pearson). Current usable counts are 25 and 33.
- H5: **34 completed pairs** for a 3 mg/dL difference with within-person
  difference SD 6; 90 if SD is 10; 128 if SD is 12. The current two pairs cannot
  estimate that SD reliably. Keep the guidebook's approximately **70 completed
  pairs** as an operational cushion, subject to a proper variance review. Each
  pair needs at least six valid rounds of each kind. The software no longer
  silently substitutes below-threshold participants.
- H6: no new empirical power claim; matched human/model scoring remains pending
  for this production audit.

### Inference and study design

H1/H2 primary generic MAE now use alpha 0.025 in the code. Overall/generic are
repeated representations of the same primary endpoint, not independent
replications. H3–H5 and own-data layers remain exploratory at nominal 0.05;
report additional searched methods and analyses transparently.

The app assigns different source mixtures according to diabetes status. Raw H1
is therefore a comparison of assigned tasks as well as players. The recommended
next cohort should use balanced common windows or a prespecified model that
accounts for trace/window difficulty. More people alone cannot repair absent
source overlap. Do not claim the power table guarantees a redesigned estimand.

Use the existing production snapshot for discovery. Keep original hypotheses
visible; new data-prompted questions need a separately labeled plan and
independent future participants. Agree analysis milestones or valid sequential
inference before repeatedly testing accumulating data. Recalculate recruitment
from variance and completion rates with a fixed target effect, not from an
observed favorable effect or an arbitrary p-value crossing.

See [the full production assessment](interim-assessment-2026-10-24.md) and
[the draft hypothesis extension plan](hypothesis-registry.md). The English
source guidebook and translations have not been silently rewritten.
