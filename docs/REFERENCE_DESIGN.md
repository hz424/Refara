# Running a reference comparison

`reference-design run` scores supplied predictions under a declared target and
writes task, acquisition-unit and overall summaries. Read
[choosing references](choosing-references.md) first if you are deciding which
controls belong in the prediction or observation.

To choose among control budgets, use the [organizer workflow](organizer.md).
It uses development scores to decide which model orderings to report, then
checks those decisions on assessment cases.

For raw-cell official VCC metrics and calibration, use the separate
[`reference-design vcc` workflow](VCC_AUDIT.md). The MSE targets and aggregation
described here apply to `prepare`, `run` and `score`.

Commands below run from the repository root after the
[installation steps](../README.md#install).

## Start with AnnData or cell tables

Use `prepare` to select treated cells and matched controls, allocate reference
blocks, and export an existing prediction matrix into a runnable plan:

```bash
.venv/bin/python examples/anndata_prepare/make_example.py --output outputs/anndata_inputs
.venv/bin/reference-design prepare outputs/anndata_inputs/manifest.json --output outputs/anndata_prepared
.venv/bin/reference-design run outputs/anndata_prepared/plan.json --output outputs/anndata_results
```

Inspect `outputs/anndata_prepared/preview.md` for the selected expression layer,
task groups and cell counts. Use the manifest to map cell annotations to
prediction rows. Predictions may be task-by-gene AnnData matrices or TSV tables. Baselines for restored state predictions use the same formats. The
[preparation guide](../examples/anndata_prepare/README.md) explains each field,
gene-panel selection and the matching rules.

`prepare` uses your preprocessed data and predictions. Its output contains
everything `run` needs, even if the original AnnData files have moved.

## Run the two-block example

```bash
.venv/bin/python examples/reference_protocol/make_example.py --output outputs/protocol_example
.venv/bin/reference-design run outputs/protocol_example/plan.json --output outputs/protocol_results
```

The example supplies synthetic controls, treated means and three predictors for
three tasks in two acquisition units. Open `outputs/protocol_results/report.md`
for the target and results. The output directory must be absent or empty.

## Write a plan for your data

Apply the normalization used by your predictors before preparing the files. Gene
names must be unique and agree across inputs; the reader aligns their order and
requires finite values. Keep the gene panel and optional gene scales fixed.

A plan is a JSON file with paths relative to its own directory:

```json
{
  "target": "heldout_effect",
  "prediction_controls": "available",
  "diagnostics": false,
  "tasks": [
    {
      "id": "donor1_B_cells",
      "unit": "donor1",
      "treated": "treated.tsv",
      "references": "references.tsv",
      "controls": "controls.tsv",
      "membership": "membership.tsv",
      "models": [
        {"name": "effect_model", "kind": "effect", "prediction": "effect.tsv"},
        {"name": "state_model", "kind": "state", "prediction": "state.tsv"}
      ]
    }
  ]
}
```

Run it with:

```bash
.venv/bin/reference-design run plan.json --output results
```

| Field | Values and meaning |
|---|---|
| `target` | `treated_state` compares treated states directly; `heldout_effect` measures a response against B2; `shared_effect` measures it against B1 |
| `prediction_controls` | `available` or `unavailable` at prediction time |
| `diagnostics` | Optional boolean, default `false`; requests the balanced five-pattern calculation |
| `scales` | Optional path to positive gene divisors in a `gene`, `value` TSV |
| Task `id`, `unit` | Unique task label and acquisition-unit label for aggregation |
| Task `treated` | Path to the treated-mean vector |
| Task `references` | Required for effect targets and block-conditioned state predictions |
| Task `controls`, `membership` | Supply both to verify the reference means from source cells |
| Task `models` | At least two models with distinct names; include the complete family for this task |

All tasks must share the same model names, kinds, conditioning and representation,
and the same depths. They can have different numbers of allocations. The program
averages allocations within each task, tasks equally within each unit, and units
equally overall, separately at each depth. Choose unit labels and this weighting
to match the intended comparison.

## Prepare the TSV files

Use tab-separated files with a header row.

| Input | Columns |
|---|---|
| Treated mean, fixed prediction, or scales | `gene`, `value` |
| Untreated cells | `cell_id`, optional `stratum`, then one column per gene |
| Reference means or block-conditioned predictions | `allocation`, `depth`, `block`, optional `cells_per_block`, then one column per gene |
| Membership | `allocation`, `depth`, `cell_id`, `stratum`, `block`, `within_block_index` |

Use a consistent block set across allocations and depths. The accepted sets are
B1, B2, B1/B2 and B1/B2/B3. A native-effect-only fresh-reference comparison can
use B2 alone; a block-conditioned treated-state comparison can use B1 alone.
A fresh-reference comparison including state predictions needs B1 and B2.
With reference means alone, the scorer relies on your block definitions.
Supply controls and membership to check depth, strata, disjoint cell assignments
and reconstructed means. The generated example includes all three files.

To allocate two balanced blocks from untreated cells:

```bash
.venv/bin/reference-design allocate controls.tsv --blocks 2 --depth 20 --allocations 30 --seed 0 --output refs
```

Depth is cells **per stratum per block**. This command needs at least twice
the requested depth in every stratum and weights strata equally in each mean.
Use `--blocks 3` for the full five-pattern diagnostic; three blocks is also
the default.
Repeat `--depth` to request several depths; smaller blocks are nested within larger
ones for each allocation. The command writes `refs/references.tsv`,
`refs/membership.tsv` and `refs/allocation.json`.

## Describe each prediction

Use `kind: "state"` for predicted treated expression and `kind: "effect"` for a
native effect already on the response scale. Output kind and input dependence
are separate: either kind can be fixed or block-conditioned. Fixed predictions
use `gene`, `value` files. Predictions generated separately from each input
block use `"conditioning": "block"` and the block-table format. Supply a prediction
for every named allocation, depth and block; each must come from that input block.
Block conditioning requires `prediction_controls: "available"`.

A state saved after baseline subtraction keeps `kind: "state"`. Add
`"representation": "effect"` and a `"baseline"` file with the same layout; the
reader restores the state before scoring. Default representation is `native`
and default conditioning is `fixed`.

An optional model `provenance` path points to JSON with a `files` list of
`{"path": "...", "sha256": "..."}` entries and an optional `description`.
Those paths resolve relative to the provenance file. The program verifies the
file hashes and records them; retain the associated training and selection history
when sharing the results.

An optional `generation_record` links the prediction to its model declaration,
checkpoint and code. For block-conditioned predictions, it also records cell
membership and all input-block keys. The checks appear in `receipt.json` under
each task's `prediction_evidence`: `declaration` without a record, or
`generation_record_binding` with a valid record. Both leave `verified_input_use`
false because checking the record does not reproduce the predictor. See the
[generation-record guide](REFERENCE_AUDIT.md#optional-generation-records-consumed-by-the-scorer)
for validation requirements.

For `treated_state`, all models are states. Fixed state predictions can be scored
without reference files, in which case depth and allocation are reported as `none`.
With block-conditioned states, the primary prediction comes from B1.

With `prediction_controls: "unavailable"`, use `heldout_effect` for fixed native
effects or `treated_state` for fixed states. `shared_effect` requires controls
available at prediction time; evaluator-only references cannot enter the
delivered prediction.

For effect targets, let Y be the treated mean and E a native effect, selecting
`E(B1)` when it is block-conditioned. The primary residual is
`E - (Y - B2)` for `heldout_effect` and `E - (Y - B1)` for
`shared_effect`. A state prediction uses `M(B1) - B1` in place of E; a fixed state
uses its fixed vector in place of `M(B1)`. Thus the fresh-reference primary
comparison uses two blocks even though the three reference roles are recorded.
Each residual is divided by the fixed gene scales, when supplied, before MSE.
Native effects use no prediction-centring subtraction, including when their
predictions depend on B1.

## Read the output

`report.md` describes the target, aggregation and results. `receipt.json` records
the plan, role assignments, input hashes, membership checks, software version and
availability of the optional diagnostic.

When the diagnostic is available, the report also shows pairwise margins under
S, M, P, O and D, flags S-to-D reversals or ties, and states whether the
allocation-level identity checks apply and pass. Links beside the table open
the complete score and pairwise TSVs.

- `primary_allocation_scores.tsv` contains each model's MSE for every task,
  allocation and depth.
- `primary_task_scores.tsv`, `primary_unit_scores.tsv` and
  `primary_summary_scores.tsv` contain successive averages.
- `primary_task_pairs.tsv`, `primary_unit_pairs.tsv` and
  `primary_summary_pairs.tsv` contain all unordered model pairs at those levels.

Pairwise margins are `MSE(model_b) - MSE(model_a)`: positive values favour
`model_a`. The balanced diagnostic uses the same utility orientation. Absolute
margins at or below `1e-12` are numerical ties in the original diagnostic.

Repeated allocations measure sensitivity within the supplied cell pool.
Population inference also requires a justified replication structure and a
statistical procedure for the intended claim. See
[choosing references](choosing-references.md#interpret-the-comparison) for the
measurement and replication assumptions.

## Add the balanced sensitivity calculation

Set `"diagnostics": true` in a plan with an effect target and three reference
blocks. If the plan has two blocks or a treated-state target, the primary result
still runs and the report explains why the five-pattern calculation is unavailable.

| Pattern | Shared roles |
|---|---|
| S | All three |
| M | Observation and prediction centring |
| P | Observation and model input |
| O | Prediction centring and model input |
| D | Each role uses a different block |

Each of the 27 assignments is scored before averaging within a pattern.
D uses three distinct blocks; the primary fresh-reference comparison uses its
own target label.

Eligible runs add `diagnostic_allocation_scores.tsv` and
`diagnostic_allocation_pairs.tsv`, plus task, unit and summary score/pair tables
with the same `diagnostic_` prefix. The allocation pairs include the shared margin
`d_S`, reference-distance term `V`, separated margin `d_D`, reversal flag and
numerical identity checks. Pairs containing a block-conditioned native effect
still receive all 27 scores, actual margins and observed reversal flags, but the
fixed-effect identity is marked `identity_applicable: false`. Its prediction,
residual, verification, tolerance and expected-crossing fields are unavailable
(`None` in Python, empty in TSV), rather than reported as a failed identity.
`identity_failures` counts only applicable identities; applicable and inapplicable
pair counts are recorded separately. `all_identities_verified` requires every pair
to be applicable and verified. Descriptive reversals outside theorem scope use
`ranking_reversal`, without an expected/unexpected theorem classification.
The exact squared-error condition and its interpretation
are given in [choosing references](choosing-references.md#interpret-the-comparison).

The original diagnostic can also run directly:

```bash
.venv/bin/python examples/reference_design/run.py --output outputs/reference_example
.venv/bin/reference-design score outputs/reference_example/input/config.json --output outputs/reference_rescored
```

Its configuration contains `tasks` and optional `scales`, using the same model and
TSV formats, with three blocks required. It writes `scores.tsv`, `pairs.tsv`,
`checks.json` and `sensitivity.pdf`.

## Call the Python API

```python
from reference_design.protocol import run_plan

run_plan("plan.json", "results")
```

For arrays already in a common gene order, the balanced diagnostic also exposes:

```python
from reference_design import Prediction, allocate_controls, score_references

allocation = allocate_controls(control_matrix, cell_ids, depth=20, seed=0,
                               strata=capture_ids)
result = score_references(
    treated_mean, allocation.means,
    [Prediction("effect_model", effect_prediction, kind="effect"),
     Prediction("state_model", state_prediction, kind="state")],
    scales=gene_scales,
)
print(result.pairwise)
```

Either output kind has shape `[gene]` for a fixed prediction or `[3, gene]` for
predictions in control-block order. Use the TSV interface when gene alignment is
needed. A two-block primary plan uses two-block prediction tables; the array
diagnostic always enumerates three blocks.
