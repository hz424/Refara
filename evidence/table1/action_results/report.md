# Reference changes: replay results

The replay executed 95 declared benchmark changes using saved, real predictions. Each action recovered the expected canonical prediction, observed target, residual and available scores; the largest component discrepancy was 2.22e-16.

The example covers six Norman pipelines, CPA, scGen, CellOT and five native-effect pipelines in PBMC, and ridge plus the five saved CellFlow fits in Kang. The original protocol specifies 44 Norman/Kang cases; a separate coverage addendum specifies the 51 PBMC cases. These are organizer operations on cached data, not errors found in a published benchmark.

## What each operation required

A new-input selection loads an already saved prediction for the changed input. Reuse preserves the model-native output; effect construction and scoring can still change. The final column compares full canonical residual vectors at the fixed numerical tolerance.

| Operation | Cases | Reuse prediction | Select new-input prediction | Residual changed |
|---|---:|---:|---:|---:|
| Replace model input | 14 | 10 | 4 | 4 |
| Replace model input and paired effect baseline | 14 | 10 | 4 | 7 |
| Replace prediction baseline | 14 | 14 | 0 | 7 |
| Replace observation baseline | 14 | 14 | 0 | 14 |
| Move both scoring references together | 14 | 14 | 0 | 7 |
| Re-encode the same delivered prediction | 14 | 14 | 0 | 0 |
| Declare a state task in place of an effect task | 9 | 9 | 0 | 9 |
| Change unused references in Kang state scoring | 2 | 2 | 0 | 0 |

For a state-output pipeline, moving the same scoring reference on both sides preserves the residual. Native effects retain their emitted meaning and do not acquire a prediction-reference subtraction. Changing an observation reference creates a new realized effect target; it does not require a new predictor output. Changes of task type are recorded separately from changes within a reference-effect task.

All 16 storage-conversion or unused-role cases preserved their full canonical components and scores. Norman uses all four matched Systema metric formulas. PBMC and Kang report standardized MSE and raw RMSE because this capsule does not supply their Systema training-centroid and candidate artifacts. CellFlow scores are computed for each fit before averaging.

## Inspect an individual change

Open [action_summary.tsv](action_summary.tsv) for one row per resource, pipeline and operation. [actions.json](actions.json) contains the executable plan, before/after scores, declared input and storage provenance, and component discrepancies. [cases.json](cases.json) records both contracts; [components.npz](components.npz) retains the canonical vectors and fit-level scores.

The planner covers the listed reference, storage and native-state-lift dependencies. Feature coordinates, scales, metric centre and retrieval collection are held fixed. Storage exports are decoded before metrics are calculated. An organizer changing metric artifacts must specify that change separately.

## Missing declarations and injected checks

Deleting each role ID from each case produced 156 explicit missing-information results and 129 valid omissions of unused roles. See [role_omission_checks.json](role_omission_checks.json). These statuses test the declared dependencies; they are not a diagnostic-accuracy score for Systema.

Four injected inconsistencies were rejected: an incompatible storage baseline, a missing active storage baseline, a stale model-input declaration and unsupported metric coordinates. A native effect deliberately centred twice passed its self-consistent metadata check, but failed comparison with the trusted canonical output. That case shows why consistent declarations alone cannot verify arbitrary prediction values. See [injected_stress_tests.json](injected_stress_tests.json).

This is a cached-array replay: it runs no model inference or training and does not measure analyst time. Repeated evaluation supplied with the same role definitions and arrays can recover the same numerical results. The demonstrated value is an explicit, executable route from a benchmark change to the operations it requires.
