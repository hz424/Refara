# Comparing models fairly

Start with the same prediction task and the same available information for all
models. Decide which response to score and whether each model predicts treated
expression or an effect. Models can use different subsets of the information
available to them.

## Distinguish the four operations

`treated_state` scores treated expression directly. The two-block
`heldout_effect` comparison uses B1 for any model inputs and prediction centring, and
B2 to measure the observed effect. `shared_effect` also uses B1 for the observed
response. Choose the target that matches the application and specify the fitted
parameters, evaluated tasks, gene scales, averaging weights, and training and model-selection
rules before comparing scores.

| Change | What stays fixed | What to do |
|---|---|---|
| Representation of a prediction | The underlying prediction, measured target and control assignment | Keep the baseline needed to reconstruct the prediction. If `E = Q - c`, evaluate `Q - b` as `E + c - b`. Dropping `c` changes the prediction being scored. |
| Model-input controls | Scoring references, fitted parameters and scoring procedure | Supply the prediction for the new input, `F(C')`. Generate it with the fitted model or reconstruct it from sufficient saved pointwise outputs; a prediction for `F(C)` alone is insufficient. |
| Prediction baseline | The model's output and the measured target | Subtract the new baseline from the predicted state and rescore. A model that directly predicts an effect needs no additional subtraction. |
| Observation controls | The predictor, prediction baseline and target population | Recalculate the observed response from the new controls and rescore. Sampling new controls from the same population changes the measured response, while the underlying population effect can stay the same. |

Repeated allocations measure sensitivity to sampling controls from the supplied
pool. The primary two-block comparison shares controls between model input and
prediction centring; the three-block D pattern separates the roles to examine
each one. Changing the biological condition, target population or requested
effect changes the scientific question.

Related approaches address different parts of this process.
[Nicol et al., v2](https://www.biorxiv.org/content/10.64898/2026.05.07.723486v2.full.pdf)
evaluate a supplied prediction using both split-reference orientations, and
also examine control-derived predictions and independent DEG selection.
[scPertEval](https://github.com/Virtual-Cell-Research-Community/scPertEval/blob/4685f11927e887745737600170da7a655b727553/src/scperteval/api.py#L398)
organizes prediction representations, metrics, transformations and reporting.
Changing the model's input controls additionally requires the prediction for
that input.

## Inspect the shared-model input comparison

Figure 3 evaluates the Figure 2 scGen, CPA and CellOT fits across four training
groups, with eight roots per group. The paired training regimes reuse those
roots. Changing model-input controls changed predicted expression even with
both scoring references held fixed. The mean model order stayed the same in all
48 pair–depth comparisons for each of the two input changes. Each seed was scored
before averaging; seeds and allocations are computational repeats.

The [current Figure 3 replay](../evidence/current_submission/README.md) reconstructs
all 576 plotted values and 48 pair–depth comparisons for each input change from
seed/root summaries. A separate representation check covered 15,552 cases;
equivalent state and effect representations agreed to within about `2.2e-16`
in MSE. That recorded check, which distinguishes representation changes from
native float32 centring differences, is described in the
[earlier Figure 3 guide](../evidence/figure3/README.md).
The [reproduction guide](REPRODUCING.md) also covers the Kang ridge/CellFlow
comparison.

## Inspect the real Norman role audit

The [`evidence/table1`](../evidence/table1/README.md) example contains the
six-pipeline, 55-task Norman analysis in Table 1. Its
`results/role_contrasts.tsv` and `results/role_interactions.tsv` include every
model, depth and metric. `mean_abs_task_mean_change` first averages signed
changes across tasks, then takes absolute values and averages over allocations
and directed reassignments.

For input-conditioned CPA at depth 8, changing the input controls changes
task-mean standardized MSE by about 0.0085 on average in absolute value. Changing
either the prediction baseline or the observation controls gives about 0.023.
All five fixed-output pipelines have zero input contrasts because their
predictions are held fixed. The mean absolute observation–input interaction
falls from about 0.0044 at depth 8 to 0.00025 at depth 135. Depth counts controls per
capture per block. The fixed-output CPA uses the original fit; input-conditioned
CPA uses the validation-selected refit. Parameters stay fixed during the analysis.

After installation, run from the repository root:

```bash
.venv/bin/python evidence/replay_table1.py --output outputs/current_table1
```

Use a new output directory. This reproduces the summary tables and the small
example that changes predictions and scoring references. Rebuilding the full
score cube requires the upstream prediction files listed in the companion's
source records. The rows and statistic used in Table 1 are specified in
[`TABLE1_CURRENT_SELECTION.json`](../evidence/TABLE1_CURRENT_SELECTION.json).
The primary scores match ordinary repeated evaluation; the additional
comparisons change one reference role at a time.

## Run the existing equal-effect example

Run this small example after installation:

```bash
.venv/bin/python examples/reference_protocol/make_example.py --output outputs/fairness_example
.venv/bin/reference-design run outputs/fairness_example/plan.json --output outputs/fairness_results
```

`native_effect` predicts `[2,2]`; `conditioned_state` predicts its input block
plus `[2,2]`. They give the same effect under the heldout-effect target, so their
residuals and MSE agree. `primary_allocation_scores.tsv` shows this for every
task/allocation. A third, fixed-state predictor gives a different prediction.
The equivalent pair both have aggregate MSE 2.5, as confirmed by independently
calculating their residuals. `receipt.json` lists the target, control assignments
and membership checks.

For your own AnnData inputs, follow the [scoring guide](REFERENCE_DESIGN.md) and
[preparation example](../examples/anndata_prepare/README.md):

```bash
.venv/bin/reference-design prepare manifest.json --output outputs/prepared
.venv/bin/reference-design run outputs/prepared/plan.json --output outputs/results
```

Review the included tasks and samples in `preview.md` before scoring.
`receipt.json` lists the inputs and target. These file checks establish which
inputs were scored; verifying an external model's training and prediction
history requires its execution records.

## Keep the official VCC scorer

Set up the manifest and environments in the [VCC guide](VCC_AUDIT.md), then run:

```bash
/path/to/new-vcc-runtime/py310/bin/reference-design vcc plan audit.json --output outputs/vcc_plan
/path/to/new-vcc-runtime/py310/bin/reference-design vcc run audit.json --python /path/to/new-vcc-runtime/official/bin/python --output outputs/vcc_run
/path/to/new-vcc-runtime/py310/bin/reference-design vcc summarize outputs/vcc_run --output outputs/vcc_summary
```

The adapter uses the official metrics and reports calibration status for the
full set of supplied comparisons. It lets you examine sensitivity to reference
choices. Whether a submission used only permitted training and prediction
information must be checked from the submission's records.
