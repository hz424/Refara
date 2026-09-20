# Changing references in a perturbation benchmark

This companion checks what must change when a benchmark organizer replaces a model input, changes an effect baseline or exports a prediction in another representation. It records the affected prediction, observation and scoring operations, then checks their numerical consequences on cached perturbation predictions.

The study uses Systema metric formulas throughout the Norman comparison. Its purpose is to evaluate the use of explicit reference roles in a benchmark workflow. The conventional single-realization, repeated-evaluation and full-role workflows share source data, available metadata, declared primary prediction targets and control budgets. Full-role diagnostics add interventions to the repeated primary evaluation; the primary scores remain the same. See [the results](RESULTS.md) for the empirical comparisons.

## Run the organizer example

The included example contains real cached predictions from PBMC, Norman and Kang, including CPA, scGen, CellOT and five CellFlow fits. It uses the first task and allocation of each resource: depth 8 for Norman/PBMC and depth 4 for Kang. This small operation example is separate from the complete 55-task Norman metric panel. It runs without the original large caches. Python 3.10.17 and NumPy 1.26.4 were used for verification. From the extracted bundle:

```bash
python -m pip install -r requirements-replay.txt
python code/role_actions.py demo --out ../role-demo-run
```

Choose an output directory that does not already exist. Open `../role-demo-run/report.md` to read the operation results. In that output directory, `cases.json` contains the full before/after declarations and `actions.json` contains each proposed action and its checked result. `components.npz` preserves the prediction, target, residual and metric vectors, including individual CellFlow fits.

To verify the bundle and reproduce both the metric summaries and the organizer example:

```bash
python code/replay_bundle.py --out ../role-value-replay
```

This checks the manifest, rebuilds the four metric tables, compares the action reports and arrays, and verifies that the bundle remains unchanged. Input files remain byte-authenticated. Recomputed Pearson scores allow an absolute float64 roundoff difference of at most 1e-14 across BLAS kernels; identifiers, decisions, counts, other metrics and prediction/residual arrays remain exact. The replay receipt records the observed numerical differences. Full metric generation and raw-source QA additionally require the upstream files named in the execution receipts. The compact example and saved-score replay are self-contained.

## The three roles determine the required operation

| Change requested by the organizer | Relevant role | Required operation |
|---|---|---|
| Supply another control block to an input-conditioned model | Model input, Cmodel | Use the prediction corresponding to the new input, then update the affected scores |
| Define a state model's effect against another baseline | Prediction reference, Cpred | Reuse the raw state prediction, form the newly declared effect and rescore |
| Measure the observed effect against another control block | Observation reference, Cobs | Rebuild the observed effect and identify the new target realization |
| Change an export format while keeping the delivered prediction fixed | Storage encoding | Decode using the recorded storage baseline and recover the same canonical prediction |
| Reassign scoring references in a direct treated-state task | Cpred and Cobs are unused | Keep the prediction and primary score |

Output type and model-input dependence are separate declarations. A native effect already represents a response and has no prediction-centering subtraction. Rendering that effect as a treated state requires its own declared state baseline. A fixed state output can ignore Cmodel while still requiring Cpred for an effect task.

The model's input reference, the effect baselines, an export's encoding baseline and the metric's own centering vector have different jobs. The software keeps them separate. In particular, reference-centred Pearson requires a consistent metric centre, and centroid retrieval requires a consistent candidate collection. Recovering the original residual guarantees MSE and RMSE; checking the complete metric inputs also protects the other scores.

## What is compared with Systema

| Workflow | Primary evaluation | Additional recorded information |
|---|---|---|
| Systema single realization | One specified input or input/observation pair | Its scores and source metadata |
| Systema with repeated evaluation | All recorded direct-state inputs or correctly paired held-out-effect realizations | Score ranges and repetition support |
| Systema with explicit reference roles | The same repeated primary evaluation | All role interventions, dependency declarations and checked operations |

Systema's existing custom-reference metrics and model control inputs are part of the comparison. A diagnostic absent from a usual score report is recorded as unreported. A repeated-evaluation workflow with explicit manual role reasoning receives the same information and can reconstruct the full-role results. This study therefore tests the correctness and empirical consequences of an executable workflow, with analyst effort and independent user benefit left for a user study.

The Norman panel retains all six models, 55 tasks, 30 allocations, six depths and 27 role assignments. Metrics are standardized MSE, raw RMSE, perturbation-centred Pearson and centroid accuracy. Only standardized MSE uses the saved training gene scales. Pearson uses the saved training-perturbation centroid; centroid accuracy is the fraction of the 54 incorrect targets farther away than the correct target. All figures are descriptive for this pooled screen.

The full role cube is a diagnostic collection. Changing Cpred can redefine the predicted effect; changing Cobs changes the realized observed effect. Results remain named by their role assignments. Their differences establish the consequences of the declared operations, rather than identifying a universally correct reference policy.

## Evidence and interpretation

The metric calculation and independent checker use separate score implementations. The checker reconstructs 6,480 rows from the original arrays, covering 25,920 metric values, and compares the complete direct-state and paired-effect panels with the previous study. It also checks all 297,000 Norman balanced-pattern MSE rows, role-label permutations and fixed-output negative controls. The largest discrepancy is 1.78 × 10⁻¹⁵.

All 95 operation cases passed independent reconstruction, with a maximum component discrepancy of 9.99 × 10⁻¹⁶. The operation examples use real cached predictions and controls. They rehearse changes a benchmark organizer could make; they are not reports of mistakes found in an external benchmark. Injected faults are recorded separately as software checks. The arrays retain prediction and target identity so that equal scores cannot hide a change in the quantity being compared.

The planner conservatively marks affected scores for recomputation. It uses declared input dependence to decide whether a prediction can be reused or a matching-input prediction is required. Its API holds the metric centre, scales and candidate collection fixed. Changes to those metric artifacts require a separately specified evaluation. These rules have been checked for correctness; a globally minimal recomputation schedule or a measured reduction in inference cost is outside the present test.

This is a retrospective study: these datasets and earlier results were already known. Its protocol was fixed before the new metric cube and operation outputs. Model training and prediction generation were completed upstream. Runtime measurements concern cached-array calculations. Reused allocations, model pairs and fitted outputs are not additional biological replicates.

The prior complete coupling analysis is documented in `reference_coupling_validation_v1`. The metric definitions and their original biological motivation are described in [Systema](https://www.nature.com/articles/s41587-025-02777-8) and its [official implementation](https://github.com/mlbio-epfl/systema). This companion leaves the manuscript and the released reference-design software unchanged.
