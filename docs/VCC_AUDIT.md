# Reference-design audit with cell-eval2

VCC 2026 already separates model-input controls from held-out scoring controls
and uses a common measured control reference for predicted and observed effects.
This workflow provides supplementary diagnostics for supplied predictions.
It scores complete context panels with the official
cell-eval2 implementation and records the cells, predictions and calibration
artifacts behind each result.

This workflow accepts raw single-cell counts and uses the pinned `vcc2026`
metric suite. Use the [MSE workflow](REFERENCE_DESIGN.md) for treated means and
effect or state predictions. The public example demonstrates reference
sensitivity on supplied data; evaluating competition submissions requires
access to the held-out scoring panel.

## Who supplies what

An organizer or investigator with access to the scoring cells supplies the
context panel, eligible scoring controls and calibration inputs. A model author
supplies predictions and describes how they were trained and generated. Public
controls supplied to a model are registered separately from held-out scoring
controls. A participant without access to the held-out panel can document model inputs and run the public example; the organizer executes the held-out audit.

Every context contains a complete perturbation panel. H5AD inputs carry unique
cell identifiers, unique gene names, a perturbation-label column and the declared
control label. Prediction artifacts contain all non-control perturbations and
omit control rows; the adapter supplies the assigned scoring controls. This
payload convention is distinct from the complete AnnData pair passed to the
upstream scorer. Gene support and non-control perturbation support must agree
across the reference and predictions. Supply counts that meet the upstream
input requirements; mean expression, effect vectors and DE tables are
insufficient for this scorer.

The manifest records:

| Record | Purpose |
|---|---|
| Reference artifact and source pool | Bind the scoring matrix, eligible control IDs, dataset identity and data split |
| Public input pools | Bind the controls actually available to each model at inference |
| Models and predictions | Identify frozen count predictions, fitted configuration or checkpoint, named input pool, whether it was actually used, and prediction provenance |
| Context, panel and strata | Keep the complete comparison panel and the intended allocation composition explicit |
| Allocations | Distinguish the original reference from declared equal-depth shared or separated control assignments |
| Unit declaration | Record experimental-unit identities, the basis for independence and the responsible investigator |
| Runtime and upstream revision | Record the scoring implementation, backend and numerical environment |

Cell identities are interpreted within their declared source datasets. Distinct
barcodes do not establish independent donors or acquisitions. Conversely, a
barcode string reused by two unrelated datasets is not evidence of cell reuse.
The audit preserves source identities so that investigators can assess both.

## Reference roles and official scoring

`C_obs` supplies the observed-response control. `C_pred` supplies the reference
for the predicted response. `C_model` denotes controls supplied when predictions
were generated. These controls affect prediction generation; changing a scoring
reference does not regenerate a model prediction.

The upstream configuration field is `control_source`, with values `real` and
`pred`. Pool/source records in the audit manifest describe physical data; they
are not a separate upstream `reference_source` option.

| Audit construction | Official setting | Interpretation |
|---|---|---|
| Shared scored reference | `control_source="real"` | The real-side controls supply both scored references, as in the pinned competition convention |
| Separated scored references | `control_source="pred"` | Real-side and prediction-side controls supply their own references; this is a reference-design diagnostic |
| Conditioning-input intervention | Separately supplied predictions | A prediction generated with another named input pool is a separate intervention arm |

For a separated construction, the worker may assemble a scoring view containing
the supplied non-control predictions and the assigned `C_pred` control cells.
This view is an explicit scoring input. Original submission files remain bound
and unchanged, and the non-control prediction payload must remain identical.
The receipt distinguishes that transformation from a prediction generated under
a different `C_model` input.

The complete context panel is passed to upstream scoring together. For example,
perturbation discrimination uses other perturbations as competitors, and the
normalized expression-error metric uses a ratio of panel sums with a correction
that also depends on the prediction panel. Averaging independent task-level MSE
callbacks would compute a different quantity. See the pinned
[metric specification](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/docs/vcc2026_metrics/vcc2026-metrics.md).

Each scored reference view has its own authenticated baseline and replicate
calibration. For a separated-reference arm, the baseline prediction uses the
same assigned `C_pred` controls as the model scoring views. The upstream
replicate anchor splits the real cells and uses each
half's own controls; its internal `control_source="pred"` remains in force even
when submissions use shared controls. An anchor is bound to the real data and
scoring semantics, so replacing the observation controls requires a compatible
calibration artifact. The adapter retains upstream estimator, gate, seed and
clipping conventions. See
[anchor implementation](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/anchor.py)
and [competition rule](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/competition.py).

## Setup and commands

Use the [installation guide](VCC_INSTALLATION.md) to create separate runtimes
for this package (Python 3.10) and the official scorer (Python 3.12). From the
package directory, supply the two installed Python interpreters and a new
runtime directory:

```bash
bash scripts/setup_vcc_runtime.sh \
  --python310 /path/to/python3.10 \
  --python312 /path/to/python3.12 \
  --output /path/to/new-vcc-runtime
```

The setup creates `py310/`, `official/` and `official_source/` under that runtime
directory. The official checkout is pinned to
`5e64833518a6603a0301cbe28185d49c30f4a986`. Set the manifest's `upstream.path` to
`/path/to/new-vcc-runtime/official_source`, retain that pinned commit, and record
the selected backend and environment with each run. The public example uses
`device="cpu"` and `de_backend="scanpy"` and requires no model fitting.

From the package directory, plan the audit without scoring:

```bash
/path/to/new-vcc-runtime/py310/bin/reference-design vcc plan audit.json --output outputs/vcc_plan
```

Run preparation, official scoring and the audit summary with the separate
scorer interpreter:

```bash
/path/to/new-vcc-runtime/py310/bin/reference-design vcc run audit.json \
  --python /path/to/new-vcc-runtime/official/bin/python \
  --output outputs/vcc_run
```

Rebuild a summary from a completed run:

```bash
/path/to/new-vcc-runtime/py310/bin/reference-design vcc summarize outputs/vcc_run \
  --output outputs/vcc_summary
```

Planning specifies the requested analyses. Preparation then validates the H5AD
inputs and writes scoring requests tied to those files; scoring follows separately. Use a fresh output directory
for a changed manifest, input artifact or upstream revision.

Run numerical commands inside a scheduled job on a cluster. Resource requests
belong to the batch script; the audit does not allocate GPUs or submit training.

## Read the output

Retain the manifest, membership records, worker requests and receipts, official
metric tables, baseline and anchor artifacts, and the summary together. Together, they
identify the original files, the scoring views and the parameters used.

Metric tables preserve upstream raw aggregates and both calibration columns
where available. The calibrated comparison uses `from_replicate`, including its
`avg_score` row. Diagnostic bundles can also retain a different
`from_baseline` average; that column must not be substituted for the calibrated
comparison. The upstream bundle's `rule_digest` and diagnostic reasons identify
whether its scoring settings match the pinned competition rule.
See [upstream scoring](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/score.py).

Read each model comparison within its context, panel, allocation and metric or
calibrated aggregate. Ties and unavailable results remain explicit. A changed
control pool can change metric gates and calibration as well as a raw score;
the retained component tables show which quantities moved. The ordinary MSE
identity `d_D = d_S + V` belongs to the assumptions in the MSE guide and is not
an identity for this metric suite.

The summary writes `METRIC_SCORES.tsv`, `COMPLETE_MODEL_PAIRS.tsv`,
`CONTROL_MEMBERSHIP.json`, `CALIBRATION_BINDINGS.json`, `UNIT_DECLARATIONS.json`
and `WORKER_FAILURES.json`, bound by `SUMMARY.json`.
`COMPLETE_DESCRIPTIVE_VCC_AUDIT` means every declared calibrated result is
available. A recognized official baseline-scale rejection instead produces
`UNAVAILABLE_CALIBRATION` for that preparation and its model responses. The
prediction inputs are still validated and bound, while every one of the six
metric slots and the aggregate remains `NA`, with the original upstream reason.
These artifacts are labelled `validated_scoring_inputs_only` because scoring
could not proceed. All model-pair slots remain present with explicit
unavailable comparisons.

When every declared slot has been handled and some calibrations are unavailable,
execution reports `COMPLETE_WITH_UNAVAILABLE_ANALYSES`, the summary reports
`UNAVAILABLE_OFFICIAL_SCORES`, and the run/summarize CLI exits with code 2.
This identifies a completed audit with unavailable analyses. Missing workers or
unexpected execution/input errors remain failures; they are not converted into
calibration unavailability. `INCOMPLETE_WORKERS` identifies incomplete worker
evidence. No reduced-metric average replaces an unavailable aggregate. Each
context retains its own aggregate; the workflow does not combine context
averages into a new leaderboard score.

Repeated allocations describe sensitivity within the supplied cells. They do
not add independent biological replicates. For inferential comparisons, first
define independent units and a scientifically justified way to construct one
utility per unit. The existing
[unit/design audit](DESIGN_AUDIT_CONTRACT.md) checks the resulting declared
support and decision rule. It does not automatically receive allocation-level
scores as independent observations. The Sound Life matching replay remains a
study-specific analysis.

## Public example and verification

The default entry point, `examples/vcc_audit/run.py`, uses the public scPerturb
version of **Adamson et al.'s K562 10X005 dataset**. The source contains 15,006
cells and 32,738 genes and is distributed under CC BY 4.0 in
[Zenodo record 13350497](https://zenodo.org/records/13350497). The downloaded
`AdamsonWeissman2016_GSM2406677_10X005.h5ad` must match SHA256
`6c6eca0f53f8887b86597e2a4ff512ff2b2d3d9c78ee7deec9a6e7d6ae859d01`.
Counts are taken from `X` after a complete finite, nonnegative, integer-count
check. The original full gene axis is retained. The original study is
[Adamson et al. (2016)](https://doi.org/10.1016/j.cell.2016.11.048).

Target and cell selection is specified before scoring. The three named
single-target constructs with at least 200 cells are mapped as follows:

| Source construct | Gene on the scoring axis |
|---|---|
| `ATF6_only_pMJ145` | `ATF6` |
| `IRE1_only_pMJ148` | `ERN1` |
| `PERK_only_pMJ146` | `EIF2AK3` |

HGNC identifies [IRE1 as an ERN1 alias](https://rest.genenames.org/fetch/symbol/ERN1)
and [PERK as an EIF2AK3 alias](https://rest.genenames.org/fetch/symbol/EIF2AK3).
The source mappings and annotation-snapshot hashes are retained in `SOURCE.json`.
Combination and unassigned constructs are excluded by these declared metadata
rules. The source `nperts` field counts underscore-separated label segments;
it is not used as a biological target count. The first 200 original cell IDs
per canonical target are selected in sorted order. `target_selection.tsv` and
`SOURCE.json` record source construct counts and selection. Selection uses no
expression effect, DE result or model score.

The two negative-control constructs, `3x_neg_ctrl_pMJ144-1` and
`3x_neg_ctrl_pMJ144-2`, supply one declared non-targeting control pool. Their
original construct labels remain in the metadata.

Metadata-valid control IDs are sorted once. The first 200 supply the primary
public input pool, the next 200 supply its alternative, and the next 400 supply
the scoring pool. Every actual source cell receives the same
`scperturb-adamson2016-k562-10x005:` prefix before splitting. These disjoint
partitions are within one public dataset; their distinct cell identities do not
establish biological independence. The audit includes the original shared pool
and shared/separated allocations at depths 100 and 200 with seeds 0 and 1.
Every allocation is retained, including an official calibration rejection. For
these frozen Adamson inputs, the original shared view calibrated, while the
first 100-cell shared view reached a baseline direction-reach value of 1 and
was rejected. A different control assignment can therefore change whether a
calibrated comparison is defined, as well as change an available score.

Three frozen teaching predictions are generated:

- `input_mean` resamples only the primary 200 input controls, with 200 predicted
  cells per target. Its expected expression is the input pool's mean.
- `input_mean_alternative` applies the same algorithm and random seed to the
  second, non-overlapping 200-cell input pool. It uses the same generation-rule
  hash as `input_mean` and is declared as that arm's input intervention.
- `truth_informed` draws **Poisson** counts from each selected observed
  perturbation mean and deliberately uses the observed responses. It has no
  claim to deployable predictive performance.

These arms are generated examples, not trained models. Their input membership,
resampling records and generation rules identify how each prediction was made.
A separate calibration oracle uses the official generic response profile of
the selected observed panel. Its dispersed emission uses all metadata-valid
source controls as its template, including controls in the public input and
scoring partitions. This transductive calibration input is recorded separately
from candidate model inputs. Its official fractional-count emission is kept
unchanged; the worker then replaces its control rows with each view's assigned
`C_pred` controls.

`SOURCE.json` records the source license and hashes, selected targets, complete
cell membership, raw-count checks, rules and seeds. Both input arms share the
same rule file: the checkpoint-hash field identifies an algorithm configuration,
and the actual input pool is recorded separately. Fixture preparation checks the inputs; scoring and parity verification
are separate steps.

```bash
/path/to/new-vcc-runtime/py310/bin/python examples/vcc_audit/run.py \
  --official-python /path/to/new-vcc-runtime/official/bin/python \
  --upstream /path/to/new-vcc-runtime/official_source \
  --output outputs/vcc_example
```

The runner downloads the pinned source and checks its SHA256. To reuse an
existing copy, add
`--source /absolute/path/to/AdamsonWeissman2016_GSM2406677_10X005.h5ad`.
Add `--prepare-only` to write the fixture and manifest without scoring.

To prepare the Adamson fixture from a downloaded source without scoring, use the
official Python environment inside a CPU job:

```bash
/path/to/new-vcc-runtime/official/bin/python \
  examples/vcc_audit/prepare_public_example.py --dataset adamson10x005 \
  --source /absolute/path/to/AdamsonWeissman2016_GSM2406677_10X005.h5ad \
  --output outputs/vcc_fixture
```

The output contains `real.h5ad`, `input_controls.h5ad`,
`input_controls_alternative.h5ad`, the three candidate prediction files,
`calibration_baseline.h5ad`, membership and generation-rule records, and
`SOURCE.json`. `control_ids.txt` lists scoring controls;
`input_control_ids.txt` and `input_alt_control_ids.txt` list the two public input
pools. `baseline_template_control_ids.txt` identifies the larger oracle template.

The Dixit K562 13-day profile remains available as `run.py --dataset dixit`
or `prepare_public_example.py --dataset dixit`. It retains the complete
21,713-gene axis and selects ten metadata-eligible targets. With 200 observed
cells per target and 400 scoring controls, all ten real-side DE gates were
empty under the pinned configuration. An independent diagnostic using every
eligible observed cell and 3,091 scoring controls found only four targets with
significant genes (7, 14, 8 and 7 genes); the other six remained empty. These
results show where calibration fails under the unchanged official gates;
the source-alignment checks are retained.

The former 600-cell, 1,000-gene H1 tutorial example remains available with
`run.py --dataset h1-tutorial` or `prepare_example.py`. Its source is the pinned
[upstream tutorial](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/docs/tutorial.md)
and public [VCC 2025 training data](https://huggingface.co/datasets/arcinstitute/VCC_train).
The first original-view preparation on that fixture is an expected failure
example: the official generic baseline's direction-reach metric equals 1,
leaving a zero calibration denominator. The upstream builder rejects it before
computing the replicate anchor. This failure is retained; the adapter neither
changes the official gates nor omits the unavailable metric to create an average.

Before distributing a run, follow the independent
[review checklist](VCC_AUDIT_REVIEW_CHECKLIST.md) and identify the code and inputs
used for numerical validation.
For the [installation test commands](VCC_INSTALLATION.md), point the official
worker tests to the installed, pinned source checkout:

```bash
export REFERENCE_DESIGN_OFFICIAL_SOURCE=/path/to/new-vcc-runtime/official_source
```

The independent example verifier reconstructs the original shared-reference
view and the first separated-reference view directly from the example inputs.
It calls the official APIs to build new calibration bundles and recompute all
six metrics for every model in the manifest. It compares raw aggregates,
calibrated scores and their average, plus baseline and replicate-anchor values,
against the adapter outputs:

```bash
/path/to/new-vcc-runtime/official/bin/python \
  examples/vcc_audit/verify_parity.py \
  --example outputs/vcc_example \
  --output outputs/vcc_example_parity
```

The three-arm example has 252 comparison slots: 248 numerical comparisons and
four exact structural-null comparisons. The official normalized expression-error
anchor is a ratio of sums with no per-perturbation tidy row, so its two cohort-count
extrema are not applicable in each view. Those four slots require literal nulls
on both sides, finite split values, null split cohort counts and the pinned split
seeds. All scored values and replicate statistics still require finite numbers;
the other five metrics' cohort counts require equal positive integers.

The verifier writes `PARITY.json`, `COMPARISONS.tsv`, `INPUT_BINDINGS.json` and
its independent official results. `FULL_ROSTER_VERIFICATION.json` authenticates
all declared worker, metric and model-pair slots, including every unavailable
row. `UNAVAILABILITY_PARITY.json` records an additional direct official replay
of the first equal-depth shared allocation when the example contains unavailable
calibrations. That allocation is selected from manifest order before reading
scores. Its official baseline rejection must match the adapter's recorded
reason; another successful allocation is not substituted. Other unavailable
allocations receive complete source/response/NA-row checks. Only the first
rejection is replayed independently. The verifier computes expected values
without the worker's numerical helpers or calibration bundle. Tolerance is
`1e-10 + 1e-10 * abs(expected)`. The worker's separate
`official_zero_endpoint_check` verifies that a baseline scores zero against its
own calibration; the independent verifier checks the full scoring path.

`PANEL_AGGREGATION.json` retains the expression-error numerator and denominator
for each perturbation. A separate Python sum verifies their ratio against both
official and adapter aggregates. It also reports the range of denominators and
the mean of per-perturbation ratios, identifying whether the observed panel
actually demonstrates a difference between these two aggregation rules. This
check uses the existing official metric components.
