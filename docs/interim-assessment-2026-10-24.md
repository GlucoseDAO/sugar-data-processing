# Sugar Sugar: production interim assessment, 24 October 2026

Production data support a raw diabetes-group accuracy difference and two
promising exploratory findings. They do **not** establish the proposed diabetes
or CGM expertise advantage, an experience-duration benefit, or an own-data
advantage. Different assigned glucose tasks and repeated identities materially
affect interpretation.

This production assessment supersedes the initial same-day inspection that
could find only an August local export. The production file was subsequently
retrieved with the user's authorized SSH access; no production files were changed.

## Data and independence

Source: `ovh:/home/web/sugar-sugar/data/input/prediction_statistics.csv`, records
from 23 March through **9 October 2026, 08:10:34** (timestamp as recorded).
The source checkout was at `570f85a`. The raw CSV was scrubbed of email,
location and paper-full-name fields before transfer. Matching normalized email
identities were linked on the server using opaque local person labels; no email
addresses were transferred in that linkage table.

| Count | Number |
| --- | ---: |
| Saved runs | 207 |
| Distinct browser study IDs | 184 |
| Linked identity groups | 155 |
| Primary eligible: at least six scored generic rounds | 109 |
| Primary eligible after stricter quality sensitivity | 104 |
| Own-data eligible: at least six scored own rounds | 3 |
| Paired eligible: at least six scored rounds of each kind | 2 |
| Own / paired after stricter quality sensitivity | 2 / 1 |

There are **36 anonymous study IDs** that cannot be linked by email and may
still repeat humans. Shared or mistyped emails can also cause linkage errors;
155 is an identity-based estimate, not verified unique-human enrollment.

The existing latest-run-per-person-and-format rule is retained. Demographics
now come deterministically from the latest saved run rather than whichever
format a Polars aggregation happened to return last. Only finite nonnegative
round MAEs count toward eligibility. A saved run with zero scored rounds does
not become a prediction round. Completing more rounds does not add independent
people.

The quality sensitivity excludes identities with conflicting diabetes or CGM
status, age outside 18–120, CGM duration over 30 years, diabetes duration over
80 years, or durations above age. These are outcome-blind sensitivity rules,
**not retrospective claims about protocol-prespecified exclusions**. Resolve
these records with the investigators before a final confirmatory analysis.
Do not remove high prediction errors simply to improve significance.

## Original hypotheses

MAE is mg/dL; lower is better. H1/H2 use person-level generic MAE and two-sided
alpha 0.025 (Bonferroni for the two primary comparisons). H3–H5 are exploratory.

| Hypothesis | Linked sample | Result | Meaning |
| --- | --- | --- | --- |
| H1: diabetes vs no diabetes | 26 vs 83 | Mean 25.88 vs 19.71; Mann–Whitney p=0.000247 | Significant raw distribution difference, opposite to the proposed advantage. Does not isolate ability because source assignment differs. |
| H2: CGM vs no CGM | 34 vs 75 | Mean 22.21 vs 20.72; Mann–Whitney p=0.386 | No overall evidence of a CGM advantage. Does not establish equivalence. |
| H3: diabetes duration | 25 | Pearson r=-0.246, p=0.235 | No clear duration association. Spearman rho=-0.259, p=0.211 agrees. |
| H4: CGM duration | 33 | Pearson r=+0.271, p=0.127 | No support for the proposed negative association. Spearman rho=+0.384, nominal p=0.0275 is an opposite, method-dependent signal. |
| H5: own vs generic | 2 pairs | Wilcoxon p=1.0 | Insufficient people for a useful conclusion. |
| H6: humans vs algorithms | Not scored in this audit | Pending verified matched windows | No human-versus-algorithm significance claim. |

H1 mean difference (diabetes minus no diabetes) is **+6.16 mg/dL**, descriptive
95% participant-bootstrap interval **2.8 to 9.4**. Welch's mean-test sensitivity
also finds a difference (p=0.000872). The Mann–Whitney result concerns ranks /
distributions; it is not automatically a test of the mean. The 95% mean interval
is not a Bonferroni-adjusted rank-test interval.

H2 mean difference is **+1.49 mg/dL**, descriptive 95% bootstrap interval
**-1.9 to +4.8**, Welch p=0.393. That interval permits differences in either
direction; a failed superiority test is not an equivalence test.

In the stricter quality sensitivity (104 eligible), H1 remains significant
(p=0.0000288), H2 remains inconclusive (p=0.198), and H3 remains inconclusive
(p=0.639). H4 becomes nominally positive with Pearson p=0.0363, again opposite
to the predicted benefit and sensitive to analysis choices. Own-data H3/H4 and
H5 cannot run under strict eligibility. This audit does not select whichever
method yields a favorable p-value or call the H4 benefit established.

### A misleading own-data result removed by identity linkage

The original study-ID analysis had eight own-data IDs and six paired IDs.
It reported H4 own-data Pearson r=-0.840, p=0.00909. Linking repeated identities
leaves **three** own-data people and **two** pairs; H4 own-data becomes
r=-0.780, **p=0.431**. The apparently significant own-data experience result
was not robust to using the correct independent unit.

## Exploratory findings suggested by the production snapshot

These comparisons are explicitly post hoc. Holm-adjusted values below cover
only these three named exploratory tests, not the complete history of interim
looks, all source thresholds, all dashboard layers, or all possible models.
Their adjustment is a robustness check, not retrospective confirmation.

| New comparison | Sample | Result | Holm p within these three tests |
| --- | --- | --- | ---: |
| Within-person diabetic vs non-diabetic traces, at least three rounds of each class | 16 pairs | Mean 25.71 vs 16.03; all 16 had higher error on diabetic traces; mean difference +9.68, descriptive 95% bootstrap interval about 6.8–12.6; Wilcoxon p=0.0000305 | 0.0000916 |
| CGM vs non-CGM among non-diabetic players on non-diabetic traces, at least six such rounds | 11 vs 58 | Mean 13.28 vs 20.30; mean difference -7.02, descriptive 95% interval about -9.9 to -4.2; Mann–Whitney p=0.000671 | 0.00134 |
| Diabetes groups on diabetic traces only, at least six such rounds | 25 vs 16 | Mean 26.73 vs 24.57; difference +2.16, 95% interval about -3.0 to +7.1; Mann–Whitney p=0.256 | 0.256 |

The CGM subgroup result persists in the quality sensitivity (11 vs 57;
Mann–Whitney p=0.000768, local Holm p=0.00154). The within-person difficulty
result also persists. However, source classes are not identical shared windows,
and people select some play modes themselves. These analyses do not establish
causal CGM training effects or a fully controlled diabetes comparison.

The three-round requirement for the difficulty exploration is not the
six-round-per-condition H5 rule. Requiring six per trace class leaves only three
pairs and a two-sided Wilcoxon p=0.25; this is a sample-size limitation and shows
why the exploratory threshold must be disclosed. At least one round per class
leaves 19 pairs, all in the same difficulty direction. No threshold was adopted
as a new confirmatory outcome.

## Recruitment revision

Observed pooled person-level SD is **7.95** for H1 and **8.36 mg/dL** for H2.
Keep the target effect fixed instead of powering to the favorable observed H1
difference. With SD 8.36, a 4 mg/dL effect needs about 85 per balanced group;
a 3.5 mg/dL effect needs about 110. These pilot estimates are uncertain and mix
different assigned tasks.

**Recommend 320 eligible participants**, operationally 160 in each diabetes
and CGM margin. This uses the config's 3.5 mg/dL threshold and a planning SD of
10 (calculated minimum 314 eligible). At observed eligibility rates, this means
about **460–480 recorded identities total**, not website visitors. We currently
have 109 eligible, or 104 in the quality sensitivity, so need about **211–216
additional eligible people** with deliberate group balance.

If the investigators instead choose 4 mg/dL as in the guidebook's calculation,
use **250 eligible**, about 360–375 recorded identities. At SD 12, the 3.5 mg/dL
minimum rises to 450 eligible. These are conditional t-test planning models,
not guarantees for rank tests, interactions or redesigned shared-trace models.

The current four primary cells are 22 diabetes+CGM, **4 diabetes without CGM**,
**12 non-diabetic CGM**, and 71 non-diabetic non-CGM. Prioritize the two sparse
crossover groups; simply adding more non-diabetic non-users will not answer the
CGM-versus-diabetes question.

For H3/H4, |r|=0.25 needs approximately 124 complete people **inside each
relevant subgroup**, not 124 total. For H5, retain roughly 70 completed pairs
as an operational cushion; the theoretical 34-pair minimum assumes paired SD 6
and a 3 mg/dL effect, which two current pairs cannot validate. Full scenarios,
assumptions and example recruitment quotas are in [the analysis plan](analysis-plan.md).

## Dashboard and hypothesis development

Pearson correlation is a proper test for a linear association, not an inherently
improper shortcut. It does not control trace difficulty, confounding or repeated
participants. The dashboard already has Mann–Whitney/Welch group comparisons
and paired t/Wilcoxon tests. Adding more unplanned tests is not the central fix.

The software now corrects primary multiplicity, honors pseudonymous identity
linkage, chooses the latest demographics deterministically, counts scored
rounds, avoids silently relaxing H5 eligibility, and distinguishes skipped
analyses from negative results. The new briefing adds descriptive bootstrap
intervals, rank-correlation sensitivity, source-aware exploratory comparisons,
clear population counts and conditional recruitment scenarios.

A next-cohort plan should use common/counterbalanced windows and, where data
support it, a model with participant and trace/window effects, prespecified
covariates and a diabetes × CGM interaction. H3/H4's proposed early learning
curves need nonlinear/segmented or spline models planned separately; one global
correlation does not test an early plateau. More model complexity cannot rescue
absent source overlap or only four people in a subgroup.

Keep H1–H6 visible and label new questions separately. Freeze this snapshot as
discovery data, and test new hypotheses in genuinely new independent people.
Do not repeatedly stop/restart collection when ordinary p-values cross 0.05.
Agree analysis milestones, a stopping rule or valid sequential methods before
further inferential looks. [Draft extension registry](hypothesis-registry.md).

## Reproducibility and deliverables

- `scripts/assess_production.py`: reproduce aggregate JSON and the self-contained briefing.
- `output/reports/production_assessment_2026-10-24.json`: complete audit (gitignored).
- `output/reports/livia-study-briefing.html`: interactive plain-language briefing.
- [Published briefing](https://sugar-sugar-study-briefing.just-dna-seq-0763.chatgpt.site): public to anyone with the link; aggregate data only.
- `data/raw/prediction_statistics.csv`: scrubbed, linked production input (gitignored).
- `data/raw/production_identity_map.csv`: opaque identity map (gitignored).
- `data/processed/participants_audited.parquet`: linked person-level data and quality flags.

Run from this repository:

```bash
PYTHONPATH=src .venv/bin/python scripts/assess_production.py
```

Bootstrap mean intervals use 20,000 person-resamples with seed 20261024 and
stable identity sorting. Mean intervals and rank-test p-values answer different
statistical questions; both are labeled. The prior August-only JSON is retained
as an archival local-inspection record and is not current production evidence.

Methods references: [SciPy Mann–Whitney definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html),
[SciPy Pearson assumptions](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html),
[statsmodels two-sided independent-test power specification](https://www.statsmodels.org/stable/generated/statsmodels.stats.power.TTestIndPower.solve_power.html).
Spearman p-values in this sensitivity check use the standard asymptotic
calculation; [SciPy recommends permutation inference for smaller samples](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html).
Treat the nominal opposite H4 signal as exploratory; it is not confirmation of
a duration effect. A next-cohort repeated-measures model can follow the
[mixed-effects framework](https://www.statsmodels.org/stable/mixed_linear.html)
when the participant/trace overlap supports it.
