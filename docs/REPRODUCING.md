# Reproduce the numerical results

Use the Python 3.10 environment from the [installation instructions](../README.md#install).
Run commands from the repository root and choose a new output directory.
For Figures 2 and 3 and Extended Data Figure 9, run:

```bash
.venv/bin/python evidence/current_submission/replay.py --output outputs/current_submission
```

This [capsule](../evidence/current_submission/README.md) uses the completed
2,560-epoch Influenza CPA fits and reconstructs model comparisons, input changes
and metric concordance from saved scores and per-seed/root summaries. Its
manifest binds the inputs to the matching Source Data.

## Organizer reporting rules

```bash
.venv/bin/python evidence/organizer/replay.py --output outputs/norman_organizer
```

The Norman replay uses bundled scores for six models and 55 tasks, with a fixed
development/assessment split, four reporting gaps and six control budgets.
At 192 controls and gap 0.01, single-split and repeated-primary rules report
323 and 165 comparisons, with 65 and zero unsupported directions. Outputs include
all thresholds, held comparisons and budget results.

For fixed-threshold and equal-coverage comparisons, including transfer to
39 additional Influenza PBMC donors with disjoint donor and cell sets within
the same study, run:

```bash
.venv/bin/python evidence/organizer/evaluate.py --output outputs/reporting_comparison
```

Each dataset's decisions are fixed before its full assessment scores are
examined. The [organizer guide](organizer.md) describes the commands and inputs;
its [comparison section](organizer.md#compare-reporting-rules) gives the protocol,
donor/cell-overlap checks and risk–coverage outputs.

## Five reporting rules on Parse/GSE181897

```bash
.venv/bin/python -m reference_design.reporting_validation replay --input evidence/reporting_reliability --output outputs/reporting_reliability
```

The [capsule](../evidence/reporting_reliability/USAGE.txt) supplies compressed
task-by-allocation losses, C/P/A roles and expected results for frozen models.
It compares margin, repeated splitting, cross-fitting, bootstrap and Refara,
reconstructing assessment intervals, natural/common coverage, pool sensitivity
and inference costs, including a first-partition cross-fitting cost companion.

All rules report only supported directions at every common count 1–36 of 63 possible
primary comparisons and 1–294 of 504 secondary comparisons. Natural coverage
differs, so fewer unresolved reports alone do not establish a Refara advantage.
These data were examined before this comparison; donor intervals assume working
independence despite shared control pools.

The [trained-model example](../examples/real_model_reference/USAGE.txt) runs
normalization, inference and equivalent effect/state scoring on a CPU using
bundled counts and weights, without downloads or retraining.

## External fixed-budget comparison

```bash
.venv/bin/python evidence/gse181897_reference_design/replay.py --output outputs/external_design
```

Saved GSE181897 task losses cover 16 model-selection donors and 46 disjoint
assessment donors. Primary selection compares eight input plus eight observation
controls with all 16 controls shared. Both choices use the same eight deployment
controls and a common effect target measured from separate controls. Selection and assessment donors share
acquisition pools.

| Output | Contents |
|---|---|
| `RESULTS.json` | Selected families, both prespecified confidence intervals and the joint 5% benefit decision |
| `policy_summary.tsv` | All reference designs and secondary budgets |
| `primary_donor_losses.tsv` | Paired assessment-donor losses |
| `mechanism_diagnostics.tsv` | Designs with identical model inputs and different observation references |
| `all_model_losses.tsv` | Assessment loss for every admitted family |

All designs selected PCA; the primary strategies had identical loss and did not
meet the 5% benefit criterion. The seven families use frozen Parse-trained
weights: 32 unavailable or ambiguous inputs are filled with training-control
means, and 1,968 unambiguous common genes are scored. Results concern these fixed
adapters and finite-cell measured effects. Treated and control cells come from
different physical pools; donor intervals condition on the observed pools and
selection split. The protocol was frozen locally before expression access for
this analysis; its timestamp and hashes accompany the inputs.

## Earlier six-component suite

For mechanism examples, program responses, donor matching and reference-role
operations, run:

```bash
.venv/bin/python evidence/reproduce.py --output outputs/conclusions
```

The suite's `model_comparisons` and `figure3` inputs use the earlier 1,280-epoch
Influenza CPA fits. Use the current PBMC capsule above for those manuscript
values. The change to 2,560 epochs leaves largest-depth reversal counts,
input-change pair orderings and program-response inputs unchanged.

The suite writes tables, logs and `RESULTS.json`. To list or select components:

```bash
.venv/bin/python evidence/reproduce.py --list
.venv/bin/python evidence/reproduce.py --claim model_comparisons --output outputs/comparisons
```

## Inputs and calculations

| Component | Starting inputs | Calculations |
|---|---|---|
| Reference-role mechanism | Constructed prediction/reference arrays | Residuals, scoped S/D identity, reversals, stable cases and an input-dependent-effect counterexample |
| Model comparisons | 15,360 seed/unit utilities, already averaged over allocations | Equal-seed then equal-unit means, ranks, all pair margins and reversal identities across four training groups and every depth |
| Model-input changes | Per-seed/root displacement and score summaries | Seed-pooled displacement, equal-root score changes and 96 pair/depth comparisons for two input changes |
| Program responses | Original-eight task/program predictions and observations; additional-39 donor errors | Direction disagreements, squared errors and balanced additional-donor summaries |
| Matching sensitivity | Sign counts for ten witness matchings; utility sums for 144 matchings | Exact sign tests, Holm adjustment over 56 directions, decisions and leaders |
| Role sensitivity | Norman task-score cube and small prediction/reference arrays | Single-role contrasts, 95 operation cases, invariance controls, CPA sensitivity ratios and depth effects |

Scores support reaggregation; prediction/reference arrays support residual
recalculation. Preprocessing, fitting and reconstruction of the restricted
matching graph require separate upstream inputs.

## Read the results

- The [current PBMC replay](../evidence/current_submission/README.md) lists all
  reversed pairs, rankings, depths and exact ties. At the largest depth,
  separating prediction or observation references reverses 10 of 28 pairs in
  each group. Figure 3 averages per-seed results and weights eight biological roots equally;
  [representation checks](../evidence/figure3/README.md) are documented separately.
- The [program replay](../evidence/program_responses/README.md) reproduces
  75/1,480 and 65/1,480 direction disagreements. It scores seeds separately before
  averaging and recovers observation-dependent error ordering at fixed predictions.
- The [matching replay](../evidence/matching_sensitivity/README.md) reconstructs
  three matching-dependent directional decisions and 67 NC versus 77 RBF leaders.
- [Role sensitivity](../evidence/README.md#role-sensitivity) averages signed changes
  over 55 tasks before taking absolute values and averaging reassignments and
  allocations. For conditioned CPA, each scoring-role/input sensitivity ratio is
  about 2.7 at depth 8 and 1.3 at depth 135. These compare the aggregate summaries;
  depths count untreated cells per capture per block. All three sensitivities decline with depth; fixed-output input zeros are
  structural controls. The statistic measures score sensitivity, not accuracy
  gain or a bound on ranking changes.

## Further reconstruction and tests

The [Jerber reference audit](../evidence/jerber_reference_audit/USAGE.txt)
replays the fixed-budget S/O comparison.

The [upstream guide](../reconstruction/current_upstream/README.md) supplies asset
manifests, local path configuration and commands for checking/copying inputs,
CPA continuation and prediction in its native environment. Training requires
prepared inputs; prediction requires the fitted state and control counts.
The [ridge–scGen capsule](../capsules/gse162632_scgen/README.md) recalculates scores
from bundled treated, prediction and reference arrays. See
[additional analyses](ADDITIONAL_ANALYSES.md) for simulations and training reviews,
and [data access](DATA_ACCESS.md) for external inputs and confidential attachments.

```bash
.venv/bin/python -m pytest -q
```

Tests cover calculations, averaging order, ties, counterexamples and altered
inputs. External-data tests skip when inputs are absent. To supply matching
Source Data, run:

```bash
.venv/bin/python -m pytest -c pyproject.toml -q --source-data-root=/path/to/05_SOURCE_DATA
```

The validator checks the package inventory and file hashes before these tests run.
