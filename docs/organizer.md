# Auditing model comparisons and reference budgets

Compare reference designs at the same control-cell budget and check which model
comparisons retain their direction. Define roles, budgets and sampling strata
before inspecting results. The [operation guide](REFERENCE_AUDIT.md) distinguishes
changes that require new predictions from changes that only require rescoring.

`reference-design design` calibrates reporting rules on development scores, then
freezes which comparisons to report. It compares single-split and repeated-score
rules, with an optional control-geometry rule. `check-design` checks those
decisions against recorded assessment realizations. The design also selects
the smallest tested budget where the unique anchor winner has reported
comparisons against every competitor.

Choose the prediction target, metric, control roles, candidate budgets and reporting gap before running the design. Calibration describes variation across the declared models and reference realizations; its empirical ranges are not confidence bounds or guarantees for unseen models.

Each budget is assessed against its own anchor winner. Consistent A wins at 20 cells and consistent B wins at 40 cells can therefore yield a 20-cell recommendation for A. To preserve the larger-budget winner, compare winners across budgets separately. Role and cell-count fields are user declarations; the [reference audit](REFERENCE_AUDIT.md) and membership checks examine how the underlying predictions and scores were produced.

## Summarize independently assessed directions

`reference_design.direction_report` summarizes declared model comparisons using
intervals supplied by an independent analysis. Define each task, its model pair
and its reported direction before examining the validation intervals. Every
candidate needs an explicit A, B or hold decision, including comparisons that
are not published.

```python
from reference_design.direction_report import Comparison, ReportedDirection, summarize_directions

pair = Comparison("cell_type/perturbation/time", "model_A", "model_B")
report = ReportedDirection(pair, "a", (0.1, 0.3))
summary = summarize_directions([pair], [report])
```

Supply interval bounds on the `loss(B) - loss(A)` scale: positive values favor A.
The function orients the interval toward the published direction and assigns:

| Interval relative to the published direction | Status |
|---|---|
| Entirely positive | Supported |
| Entirely negative | Reversed |
| Touches or crosses zero | Unresolved |
| Explicit hold decision | Hold |

Coverage is the number published divided by all candidates. Reversal and
unresolved fractions both use the number published as their denominator;
unresolved comparisons stay in it. If every comparison is held, coverage is
zero and both conditional fractions are `None` (`null` in JSON).

Run the four-comparison synthetic example to export JSON and TSV:

```sh
.venv/bin/python examples/direction_report.py --output outputs/direction_report
```

The example has one comparison in each category. The API classifies supplied
intervals; the study must establish their estimation, multiplicity correction
and experimental independence. For population mean loss differences, declare
the population and weights and estimate uncertainty from independent
experimental units. Model pairs and reference allocations are not additional
biological replicates. Compare rules on the same roster at common achievable
coverage, retaining reversal and unresolved fractions. The checks below instead
measure stability across the supplied reference realizations.

## Compare a fixed total control budget

To compare reference sharing, separation and cross-fitting using the same
physical control cells, run:

```sh
.venv/bin/python examples/fixed_budget.py --output outputs/fixed_budget
```

The example scores six designs: all controls shared; 25%, 50% or 75% assigned
to model input with the rest reserved for the observed effect; and two-fold or
four-fold cross-fitting. Prediction centring uses the input controls. Each
cross-fitting direction is scored before its squared error is averaged.
The output records physical cell counts and inference work for every design.

For your own predictions, use
`reference_design.fixed_budget.score_reference_design`. Supply a control
matrix, one predicted treated state per input cell (or a direct-effect vector),
a treated mean and fixed positive gene scales. Row order determines the
allocation, so fix that order before inspecting scores. The reusable prediction
bank assumes each cell's prediction is unchanged when other input cells are
added or removed. Models that encode an entire input set require new predictions
for each set; the [operation audit](REFERENCE_AUDIT.md) checks that distinction.

A low design-specific score selects a model for that design. To assess whether
the choice improves prediction, compare the selected models on disjoint donors
with a common deployment input and common measured target. The independent
GSE181897 example uses this sequence:

```sh
.venv/bin/python evidence/gse181897_reference_design/replay.py --output outputs/external_design
.venv/bin/python evidence/gse181897_reference_design/plot.py --results outputs/external_design/RESULTS.json --output outputs/external_design_figures
```

Its primary analysis gives each donor equal weight, then each eligible context
and condition equal weight within donor. The comparison uses 16 controls in
total and requires at least 5% lower held-out loss, supported by both paired
donor t and studentized-bootstrap intervals, within a fourfold inference limit.
Other budgets, deployment depths and treated-state losses are reported
separately. The replay starts from saved task losses; it reconstructs model
choices and inference without retraining or downloading expression matrices.

All 24 budget/design choices selected PCA, including both overlap diagnostics.
Shared and separated designs had identical held-out loss: 0.049466 standardized
MSE across 46 donors and 777 tasks, or 0% reduction. The prespecified 5% benefit
criterion was not met; secondary budgets and deployment depth also selected
the same model. The [replay protocol](../evidence/gse181897_reference_design/protocol.json)
and loss tables distinguish scoring changes from changes in the deployed model.

## Reproduce the Norman comparison

Run from the repository root after installation:

```sh
.venv/bin/python evidence/organizer/replay.py --output outputs/norman_organizer
```

The bundled scores cover six models, 55 Norman tasks, six control budgets and
90 realizations per task and budget. Each realization combines one of 30
allocations with one of three model-input blocks. The other two blocks define
the observation reference. `evidence/organizer/study.json` records the model
names, control counts, reference roles and fixed split into 27 development and
28 assessment tasks.

At 192 controls and gap 0.01, single-split reporting releases 323 of 420
comparisons; 65 fail the assessment rule. Repeated-primary reporting releases
165, all supported across the recorded realizations. It recommends budgets
for two assessment tasks and holds the other 26. The replay freezes each
design before checking assessment scores and retains all four gaps, six
budgets, and reported and held comparisons.

Read `results/policy_summary.tsv` for coverage and unsupported directions,
`results/budget_summary.tsv` for budget recommendations, and
`results/geometry_comparison.tsv` for the geometry comparison. Geometry scaling
does not improve on repeated primary in this example. These tasks share the
same experimental control pool; the checks measure stability within its
recorded allocations.

To inspect the inputs and run the primary gap step by step:

```sh
.venv/bin/python evidence/organizer/export.py --output outputs/norman_inputs --stage design
.venv/bin/reference-design design outputs/norman_inputs/manifest.json --output outputs/norman_design
.venv/bin/python evidence/organizer/export.py --output outputs/norman_inputs --stage assessment
.venv/bin/reference-design check-design outputs/norman_design/design.json outputs/norman_inputs/assessment_scores.tsv --output outputs/norman_check
```

## Compare reporting rules

```sh
.venv/bin/python evidence/organizer/evaluate.py --output outputs/reporting_comparison
.venv/bin/python evidence/organizer/plot_comparison.py --results outputs/reporting_comparison --output outputs/reporting_figures
```

Compare the rules with a development-fixed margin threshold, and compare
absolute anchor margins with development-adjusted margins at the same coverage.
All four gaps and six Norman budgets are retained. Thresholds and selected
comparisons are frozen before full assessment scores are exported.

Read `results/comparison_rules.tsv` for the fixed-threshold results and
`results/comparison_aurc.tsv` for the area under each risk–coverage curve (lower
is better). The full curves, per-model-pair composition, donor summaries and
individual comparison assessments are under `datasets/*/check/`.

At gap 0.01 and 192 controls, repeated-primary reporting retains 165 comparisons
and the development-fixed margin threshold retains 137, both with zero
unsupported directions. Across the full coverage range, the simple anchor-margin
ordering has lower observed area under the risk–coverage curve at each of the six
budgets. Pair-specific calibration therefore does not improve overall ranking
of comparisons in this analysis.

The same command checks transfer from eight development donors to 39 additional
Influenza PBMC donors, using B cells and combined monocytes. The donor groups
and their selected control cells are disjoint; they share the study and eight
acquisition batches. Each training group compares its two previously selected
models. At gap 0.01, single-split reporting retains all 78 task comparisons with
zero unsupported directions in both groups. Repeated-primary reporting retains
73 and 78, also with zero unsupported directions. These data test transfer of
the frozen thresholds but show no benefit over single-split reporting.

`comparison_protocol.json` records the retrospective analysis design.
`transfer/membership.json` records the donor and cell-overlap checks. Control
budgets describe one allocation; repeated allocations are not independent
biological samples. The tables report finite-allocation outcomes and do not
assign confidence intervals to model-pair counts.

To compare rules on your own exported organizer inputs:

```sh
.venv/bin/python evidence/organizer/compare.py freeze --manifest INPUTS/manifest.json --protocol evidence/organizer/comparison_protocol.json --output outputs/comparison_plan
.venv/bin/python evidence/organizer/compare.py check --plan outputs/comparison_plan/plan.json --assessment INPUTS/assessment_scores.tsv --output outputs/comparison_check
```

Choose reporting gaps in your metric's units before examining assessment
outcomes. A retained comparison must keep its anchor direction and exceed the
declared gap in every recorded assessment realization. Held comparisons remain
in the output. For top-k curves, an anchor tie has no direction and counts as
unsupported; priority ties use the recorded case and model-label order.

## Try a small example

From the extracted repository root after installation:

```sh
.venv/bin/python examples/organizer/make_example.py --output outputs/organizer_inputs
.venv/bin/reference-design design outputs/organizer_inputs/manifest.json --output outputs/organizer_design
.venv/bin/reference-design check-design outputs/organizer_design/design.json outputs/organizer_inputs/assessment.tsv --output outputs/organizer_check
```

Use absent or empty output directories. Inspect `organizer_design/report.md` and `budget_recommendations.tsv` before running the check. The example has three models, two budgets, two development cases and three assessment cases. It includes a strict reversal and a positive margin below the reporting gap. At the smaller budget, geometry releases more comparisons and makes more unsupported releases than repeated primary; the check reports `no_gain`. At the larger budget, the policies have equal counts and the check reports `parity`.

## Use protocol scores

The converter accepts `primary_allocation_scores.tsv` from `reference-design run`.
For a small example that starts with prediction and reference files:

```sh
.venv/bin/python examples/reference_protocol/make_example.py --output outputs/protocol_inputs
.venv/bin/reference-design run outputs/protocol_inputs/plan.json --output outputs/protocol_scores
.venv/bin/python examples/organizer/from_scores.py --scores outputs/protocol_scores/primary_allocation_scores.tsv --config examples/organizer/protocol_config.json --output outputs/protocol_organizer_inputs
.venv/bin/reference-design design outputs/protocol_organizer_inputs/manifest.json --output outputs/protocol_design
.venv/bin/reference-design check-design outputs/protocol_design/design.json outputs/protocol_organizer_inputs/assessment_scores.tsv --output outputs/protocol_check
```

For your own scores, copy `examples/organizer/protocol_config.json` and set the
models, reference roles, budgets and reporting gap. List every task in either
`development_cases` or `assessment_cases`, and fix this split before comparing
the policies. `budget_by_depth` maps each depth label to a declared budget ID;
enter the physical cell counts for your strata and blocks explicitly.

The converter maps `task` to `case`, `allocation` to `realization`, and `MSE` to
`score`, retaining unit and model labels. It checks complete model and allocation
support and records the mapping and input hashes. Its exports separate
development scores and assessment anchors, read by `design`, from the full
assessment scores read by `check-design`. This route runs the two policies
that require only repeated scores.

## Declare the inputs

The example generator writes a `manifest.json` template. Paths resolve relative to the manifest. Omit `geometry` and `geometry_tolerance` to use single-split and repeated-primary reporting alone.

| Field | Meaning |
| --- | --- |
| `schema_version` | `1` |
| `target` | `heldout_effect`, `treated_state` or `shared_effect` |
| `metric` | Object with a descriptive `name` and `direction`: `lower` or `higher` is better |
| `models` | Ordered list of at least two distinct model names; this order defines pair orientation |
| `budgets` | List of objects containing `id`, `control_cells_available`, `input_cells`, `observation_cells` and `description` |
| `roles` | Text for `model_input`, `prediction_centring`, `observation_centring`, `participant_information` and `evaluator_information` |
| `development_scores` | Complete development realization scores in TSV format |
| `assessment_anchors` | Assessment scores for the declared anchor realization only |
| `geometry` | Optional. One control geometry value for every development and assessment case/budget; enables `geometry_scaled` |
| `anchor_realization` | Realization label used to choose the provisional ordering |
| `minimum_gap` | Nonnegative, user-selected reporting tolerance in the metric's score units |
| `numerical_tolerance` | Positive tolerance for numerical ties |
| `geometry_tolerance` | Required with `geometry`. Positive cutoff for treating geometry as zero |
| `evaluation_scope` | Description of the supplied cases, model family and reference realizations |

Budget counts are nonnegative integers. Input and observation blocks are disjoint in this budget accounting, so their counts must sum to no more than the available control count. Unused cells can support other declared reference realizations. Choose a reporting gap that represents a meaningful score difference for your task.

The score TSV has exactly these columns:

```text
case    unit    budget    realization    model    score
```

Each development case supplies every budget, at least two realization IDs including the anchor, and every model at each realization. Use the same realization IDs across budgets within a case. Assessment anchors supply every assessment case, budget and model with only `anchor_realization`. Development and assessment case IDs must be disjoint. Cases can share a unit label; every row for a case must retain the same unit. The complete assessment TSV used later by `check-design` follows the development score format.

When using geometry scaling, the geometry TSV has exactly these columns:

```text
case    unit    budget    geometry
```

Compute a finite, nonnegative geometry value from controls using the same definition and preprocessing for every case and budget. The program reads these values separately from the scores. Include exactly the development and assessment cases, with matching unit labels. Scores must be finite and tables complete; duplicate rows, missing entries and undeclared fields are rejected before output is written.

## Calibrate and freeze decisions

For ordered models A and B, the pair margin is utility(A) minus utility(B): `score(B) - score(A)` for a lower-is-better metric, and `score(A) - score(B)` otherwise. A positive margin favors A.

For each budget and model pair, the program computes each development case's largest absolute margin change from its anchor across recorded realizations. `B` is the largest such case shift. `K` is the largest case shift divided by that case's geometry among cases above `geometry_tolerance`.

| Policy | Empirical range at an assessment case |
| --- | --- |
| `single_split` | `0` |
| `repeated_primary` | `B` |
| `geometry_scaled` | `K × assessment geometry` |

The geometry policy is unavailable for a pair if any development case has geometry at or below the cutoff and a shift above numerical tolerance, or if no development geometry exceeds the cutoff. Assessment geometry at or below the cutoff also produces hold.

A pair is released in its anchor direction only when:

```text
absolute anchor margin > empirical range + minimum_gap + numerical_tolerance
```

For each assessment case and policy, `budget_recommendations.tsv` selects the least available control budget where the unique anchor-best model has released comparisons against every other declared model. Equal available budgets follow manifest order. If no declared budget meets the rule, the recommendation is hold.

Outputs are `design.json`, pairwise `recommendations.tsv`, `calibration.tsv`,
`budget_recommendations.tsv`, `report.md` and a SHA256 `receipt.json`.
`design.json` contains the configuration, input digests, calibration, assessment
anchors, geometry and decisions, so checking remains possible after development
inputs move. The complete assessment score table is reserved for `check-design`.

## Check the frozen recommendations

`check-design` loads `design.json`, verifies its stored decisions against its calibration and anchors, and then reads complete assessment scores. Assessment cases, units, budgets and models must match the frozen design exactly. Anchor scores must retain exactly their original numeric values. Each assessment case/budget requires at least two realizations, with the same realization IDs across budgets within a case.

A released pair is supported when its margin, oriented toward the released winner, exceeds `minimum_gap + numerical_tolerance` in **every** recorded assessment realization. The check flags any strict reversal below negative numerical tolerance, numerical tie within that tolerance, or positive margin at or below the gap plus tolerance. Flags can overlap across a pair's realizations.

`policy_summary.tsv` reports each budget separately. Coverage divides released comparisons by all declared assessment case/model-pair comparisons. The unsupported fraction divides unsupported releases by releases; it is `null` in JSON and blank in TSV when there are no releases. The report labels this value undefined. These are counts of recorded comparisons; shared units or reference pools are not treated as independent replicates.

Geometry achieves `strict_gain` over repeated primary at a budget only when geometry makes at least one release, has no more unsupported releases, has no lower coverage, and improves at least one of those counts. Equal counts produce `parity`; a tradeoff produces `no_gain`. An all-hold geometry policy produces `insufficient_releases`. A supported geometry release can improve a repeated-primary policy that holds every comparison.

`budget_assessments.tsv` separately checks each frozen minimum-budget recommendation: its selected winner must beat every opponent by the declared gap in every recorded realization at the selected budget. Held recommendations retain `decision: hold`, undefined support and `assessment_reason: held_by_design`.

The check also writes pairwise `assessments.tsv`, `check.json`, `report.md` and a SHA256 receipt. It leaves the frozen calibration and recommendations unchanged. Run a separately declared sensitivity manifest in a separate directory when evaluating another reporting gap.
