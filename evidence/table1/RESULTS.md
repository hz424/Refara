# What explicit reference roles add to the evaluation workflow

The three-role workflow turns reference changes into explicit operations on predictions, targets and scores. In this replay, all 95 operations on real cached outputs recovered the intended numerical components. The same Systema primary scores were retained. The demonstrated value is a tested route from a declared benchmark change to its correct execution and interpretation.

## A concrete organizer decision

The example contains seven state-output pipelines and seven fixed native-effect pipelines from Norman and PBMC, plus two treated-state pipelines from Kang. These operations use the first task and allocation in each resource, at depth 8 for Norman/PBMC and depth 4 for Kang; the complete Norman metric panel is reported separately below. The prediction-reference replacement operation gives different instructions to the two output types. The seven state-output pipelines require construction of a new effect from their unchanged state prediction. The seven native effects retain their emitted values. Both outcomes were checked against the original arrays.

Moving the prediction and observation references together preserves the residual of the seven state-output pipelines. For the seven native effects, the observation reference still changes the observed response; adding another prediction-centering subtraction would change the supplied prediction. This illustrates why a shared-reference cancellation statement needs the output and conversion rules to be explicit.

Replacing a model input likewise depends on the pipeline contract. In the single-input replacement example, Norman CPA and the three PBMC state pipelines require predictions corresponding to the new input. The ten fixed outputs reuse their predictions. The paired-input/effect operation repeats the same dependency check while updating the declared prediction baseline. Across the complete suite, eight operations select a different cached input prediction and 87 reuse the existing prediction. These counts describe the chosen operations; no model inference was timed.

All 16 storage-conversion or unused-role controls preserve their canonical components. The Kang cases explicitly retain the treated-state target and check both the ridge pipeline and CellFlow, with its five fits scored individually before averaging. PBMC contributes a supplemental, metadata-selected extension covering all eight pipelines; the extension was recorded before its operation outputs in [PBMC_COVERAGE_ADDENDUM.json](PBMC_COVERAGE_ADDENDUM.json).

## Role interventions under unchanged Systema formulas

The Norman metric panel covers all 55 tasks, six pipelines, 30 allocations and six depths. At each allocation it evaluates all 27 assignments of observation reference, prediction reference and model input. Single-role contrasts change one assignment while holding the other two fixed. The table gives selected CPA's mean absolute change in the 55-task mean standardized MSE, averaging all recorded allocations and directed single-role changes.

| Cells per capture per block | Cmodel | Cpred | Cobs |
|---|---:|---:|---:|
| 8 | 0.00853903 | 0.0234604 | 0.0234291 |
| 16 | 0.00604529 | 0.0124451 | 0.0123111 |
| 32 | 0.00408010 | 0.00678078 | 0.00674123 |
| 64 | 0.00255375 | 0.00376000 | 0.00375867 |
| 128 | 0.00176744 | 0.00229539 | 0.00228361 |
| 135 | 0.00175504 | 0.00222744 | 0.00222418 |

The three columns answer distinct intervention questions. In particular, changing Cmodel with Cpred fixed isolates the input response; changing both to preserve an input-paired effect is a different operation. The complete [role contrasts](results/role_contrasts.tsv) retain every model, depth and metric, including zero changes for unused roles.

The joint cube also separates model-input interactions from scoring geometry. For selected CPA, the mean absolute task-mean standardized-MSE interaction between Cobs and Cmodel is 0.00442030 at depth 8 and 0.000251055 at depth 135. All fixed-output pipelines have zero interactions involving Cmodel in this replay. Cobs–Cpred interactions also occur with fixed state predictions, so that interaction alone does not establish a model-input response. The complete [interaction table](results/role_interactions.tsv) retains all three role pairs and all four metrics.

These are finite factorial score differences. The reverse of every operation is included, which makes positive and negative contrast counts symmetric by construction. Magnitudes describe the supplied reference pool; the signs are not ranking reversals or estimates of systematic bias. Changing an effect baseline can change the declared prediction or observed realization, so these differences do not select a more accurate target.

## The Systema comparison

The single-realization workflow reports one prespecified realization at each depth. The repeated workflow retains 90 direct-state or 180 correctly paired effect realizations at each depth. The full-role workflow uses exactly the same primary scores and adds the intervention and operation records. All 288 model–depth–metric–view rows in [workflow_comparison.tsv](results/workflow_comparison.tsv) retain this equality.

This equality is expected from the matched metric inputs. The additional output is the explicit account of which prediction or target changed and how the requested operation should be executed. A repeated Systema evaluation with the same role metadata and explicit manual reasoning can recover these results. A diagnostic absent from its usual report is therefore unreported, rather than incorrect. The study measures neither analyst time nor a diagnostic-accuracy advantage over that manual comparison.

## Checks and limits of the evidence

An independent implementation reconstructed 6,480 score rows from the original arrays, including all four metrics, and checked the full panel against the earlier direct-state and paired-effect calculations. All four summary tables and their 10,368 descriptive threshold counts were independently reproduced. The largest raw metric discrepancy was 1.78 × 10⁻¹⁵.

The independent operation check matched all 29 small-example input arrays to their source arrays, reconstructed the 95 before/after cases and checked 1,520 saved components. The largest component discrepancy was 9.99 × 10⁻¹⁶. It also reproduced all 285 role-omission outcomes: 156 omit required information and 129 remove unused declarations. These are API and operation checks, not estimates of how often analysts make such errors.

Five injected software cases remain separate from the valid operations. Four inconsistent provenance or coordinate declarations are rejected. The fifth supplies an incorrectly centered native prediction with internally consistent metadata; identifying that error requires a trusted expected prediction. The test retains this limitation rather than treating hashes as proof of how a prediction was generated.

The evidence supports correct execution of explicit role declarations across the supplied cached pipelines. Independent organizer adoption, improved analyst decisions and reduced working time remain unmeasured. This is a retrospective demonstration on previously examined data; original training quality, biological generalization and new-model performance are separate questions. The [QA report](qa/QA_REPORT.md), [reader operation report](action_results/report.md) and full arrays retain the evidence needed to inspect these conclusions.
