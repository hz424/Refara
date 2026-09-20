# Statistical interpretation

Control allocations and repeated fits measure sensitivity within the supplied
data. Biological inference also requires justified experimental units and a
procedure suited to the comparison. Record both the control assignments and
the units used for statistical testing.

## Current program-response comparisons

Figure 4a,b evaluates eight donors across five cell types, giving 40 tasks.
Model-family selection leaves out the evaluation donor. Cell-type tasks within
a donor and the 37 overlapping Hallmark programs share information, so counts
of changed directions or priorities describe this task panel.

Figure 4c evaluates 78 B-cell and monocyte tasks in 39 additional donors from
the same study, using models selected from the original eight-donor cohort.
Predictions and treated observations stay fixed while the observation controls
change between B1 and B2. This changes the sign of the D-selected minus
S-selected mean program-MSE difference after balancing batches and capture
orientations. The additional cohort tests within-study sensitivity to the
observation reference; it is not an independent study.

## Low replication and genetic perturbations

For the 12-direction exact-sign/Holm family at 5%, fewer than eight non-tied independent roots cannot support any rejection. Extended Data Fig. 10c starts at eight roots; the Methods explain why rejection is impossible below that count. The hierarchical-null extension shows that preserving whole roots in a bootstrap does not by itself provide finite-sample error control; the Gaussian root-t comparison relies on its distributional assumptions.

The Norman analysis evaluates 55 held-out two-gene combinations within one pooled experiment. Its eight capture partitions balance acquisition composition; they are not eight independent experiments. Control allocations and shared-gene tasks are dependent, so their ranges and crossing counts are descriptive. The three original fitted predictors return expression states; their shared-versus-fully-separated margins are invariant under balanced allocation. Prediction- or observation-only separation can change CPA comparisons through its conditioning controls. Zero-effect comparisons remain diagnostic.

The added direct-effect ridge predictor was selected using validation within the original training combinations. Its emitted effects remain fixed during reference reassignment. Every effect/state comparison reports the shared margin `d_S`, reference distance `V` and separated margin `d_D`, where positive margins favour the effect model. The identity `d_D = d_S + V` implies a state-to-effect reversal when `-V < d_S < 0`. Conditions are checked at the same aggregation level as the margins, with absolute margins at or below 10⁻¹² treated as numerical ties. This added analysis followed the original state-model results; it is a within-screen test of the comparison mechanism.

## Earlier analysis records

In the original attribution tables, `primary_support` and the claim-gate
Booleans indicate that a diagnostic zero-threshold rule was met. They remain
for historical replay, together with their identifiers and the legacy
`claim_mapping` object. They do not provide calibrated confidence statements
or population evidence. For the current
interpretation, see the Source Data schema and revised manuscript.

Both the 56-member targeted calibration and the 198-scenario generic
qualification failed their all-scenario criteria. Replaying their calculations
does not establish inferential calibration. Exact-sign/Holm tests, finite-root
means and repeated-root diagnostic bounds answer different questions and
cannot be substituted for one another.

Under the stated six-mapping average, Stage A E1/E2 both measure the same
model-independent reference distance. Their equality and nonnegative values
therefore follow from the construction. Stage B depends on predictions from
different conditioning inputs. The successor B-condition comparison remains a
near-tie: five of eight individual roots cross, although all eight
shifts have the same sign.

The separate historical interferon-γ B-cell–monocyte follow-up examined one
fixed observed-response contrast, without applying the models selected for the
current Figure 4 analysis. It failed its prespecified stability criterion.
Its allocation and batch-omission ranges describe sensitivity; a sign change
between capture orientations prevented a stable cell-type ordering. This
analysis is reported in Supplementary Table 5.

## E3 label correction

Only `endpoint_label` is corrected in the two attribution TSVs. E3 is the raw
contrast `Q(A)-Q(C)` with the historical orientation `raw_direction=-1`.
All numeric fields, raw/oriented signs and legacy decisions are unchanged.
The projection manifest records the label correction and updated hashes.

## Reproduction levels

Each replay records its starting inputs and calculations: prediction scoring,
allocation-utility aggregation, root summaries, graph-certificate verification
where inputs are supplied, or figure rendering. Check that starting point when
interpreting a successful replay; upstream inputs are only checked if supplied.
