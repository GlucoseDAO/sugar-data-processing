# Hypothesis registry and next-cohort draft

Status: analysis-planning draft, 24 October 2026. This is not a preregistration
claim, a final protocol amendment, or independent confirmation of new findings.

## Preserve the original questions

| Hypothesis | Outcome / population | Current status |
| --- | --- | --- |
| H1 diabetes groups | Person-level generic MAE, at least six valid generic rounds | Primary, two-sided alpha 0.025; raw group difference is confounded by assigned trace source |
| H2 CGM groups | Same endpoint / eligibility | Primary, two-sided alpha 0.025; no overall evidence of benefit |
| H3 diabetes duration | Generic MAE among PwD with usable duration | Exploratory; a global correlation does not test the proposed early-years curve |
| H4 CGM duration | Generic MAE among CGM users with usable duration | Exploratory; no established negative association; opposite method-dependent signal |
| H5 own familiarity | Paired person MAE, at least six valid rounds of each kind | Exploratory, only two linked pairs; do not substitute below-threshold players |
| H6 humans vs algorithms | Same verified windows and forecast horizons, paired within person | Pending matched reconstruction/scoring for this production audit |

The currently published pilot findings should retain these labels even if future
priorities change. Do not rename a favorable subgroup result as confirmation of
H2, or silently replace a failed original endpoint.

## Candidate extensions, suggested by the pilot

| Candidate question | Why consider it | Next-cohort design / analysis to specify |
| --- | --- | --- |
| E1: Does CGM benefit differ by player diabetes status? | Non-diabetic CGM users show a favorable exploratory signal on non-diabetic traces; the overall comparison does not | Recruit the two sparse crossover groups; balance shared windows; estimate a diabetes × CGM interaction with uncertainty. Separate interaction power from marginal H1/H2 power. |
| E2: How much does trace difficulty explain the group difference? | All 16 mixed-source players with at least three rounds of each class had larger errors on diabetic traces | Randomize / counterbalance a common trace bank across groups; prespecify trace/window effects and a difficulty measure such as persistence error or observed variability. A source-class label alone is incomplete adjustment. |
| E3: Does practice during the game improve accuracy? | Plausible new question; no learning test was claimed in this assessment | Counterbalance task order, define the within-person round/time contrast, account for trace difficulty and fatigue; plan a repeated-measures model with participant effects. |

Before collecting a confirmatory next cohort, fix: the exact outcome, relevant
population, target effect, exclusions, source balancing, model, covariates,
missing-data handling, multiplicity family, number/timing of interim looks and
stopping rule. Pilot data may inform variance and feasibility; do not size a new
study solely to the largest exploratory effect.

## Keep discovery and confirmation separate

Freeze the current production export and audit hash as discovery data. A
confirmatory new-cohort participant must be independent of the linked identities
in this snapshot; a new browser study ID or another game from an old player does
not create independent validation. Resolve anonymous and uncertain linkage where
possible. If reusing the existing data for model development, call the results
exploratory and use participant-disjoint validation with a clearly stated scope.

A model with participant and trace/window effects is appropriate only when the
design and sample contain enough crossing/overlap to identify them. Shared
windows, balanced recruitment, and sufficient people come before model complexity.
Keep original primary results and all planned sensitivity results visible,
including inconvenient directions and non-significant comparisons.
