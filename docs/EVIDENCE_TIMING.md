# Evidence timing

This record distinguishes comparisons specified before their outcomes were
calculated from analyses developed after related results had been examined.
The specifications were recorded by the study team; they were not public
preregistrations.

## Original comparisons and later sensitivity analyses

- The study team fixed the held-out GSE162632 evaluation donors, tasks,
  configurations, reference designs, utility and complete 84-comparison family
  before inspecting that family's results.
- The focal contrast was selected after the complete 84-member family had been
  inspected.
- The separate 56-member family of paired shared-to-split and
  shared-to-rotation contrast changes was specified after the results were
  inspected. Its diagnostic bounds and targeted calibration were reported
  separately from the original 84-member family. The calibration failed its
  all-scenario criterion. The bounds therefore remain diagnostics rather than
  calibrated confidence intervals; the two families do not carry joint
  140-comparison error control.
- The broader PBMC allocation and alternative-metric analyses are post hoc
  sensitivity analyses that retain each complete comparison family.
- The expanded eight-family training comparison was specified before its new
  fits, with earlier evaluation results available. The later extension of
  Influenza CPA training from 1,280 to 2,560 epochs was also planned after
  earlier test results had been seen. The amendment preceded the new results,
  and configuration selection used training-side validation only.

## Program analyses and the additional 39 donors

- The earlier fixed-model program analysis was specified after the ranking
  results were known and before its program readouts were calculated.
- That exploratory screen nominated an interferon-γ program contrast between
  B cells and monocytes. The measured-response contrast, donor and cell
  memberships, allocations, aggregation and stability criterion were fixed
  before expression was extracted from 39 additional donors. The test failed
  its prespecified stability criterion (Supplementary Table 5).
- The later selected-model program analysis (Figure 4) was specified after
  earlier analyses and updated gene-level rankings, before calculating its new
  program outcomes. It reused expression data and cell memberships already
  examined in the measured-response follow-up. Figure 4c compares selected
  models' errors in those 39 donors; the earlier test assessed the measured
  cell-type contrast without running these selected models.

The 39 donors entered neither model fitting nor family selection. They came
from the same study and shared acquisition batches with the discovery cohort,
so these results assess within-study sensitivity.

## External model selection and acquisition analyses

- The GSE181897 protocol and adapters were developed using earlier Parse
  results and recorded locally before expression extraction or scoring for
  this analysis. The fixed split used 16 selection donors and 46 assessment
  donors, with all 12 acquisition pools crossing that split. Both shared and
  observation-separated selection chose PCA and produced identical assessment
  predictions; the prespecified 5% benefit criterion was not met. The
  [protocol](../evidence/gse181897_reference_design/protocol.json) records the
  timestamp, scope and hashes.
- The Sound Life analysis was developed after earlier results were examined
  and conditions on the observed donor–batch graph.
- Metadata or protocol cases that did not meet the stated requirements were
  not analysed directionally.

The manuscript's Supplementary Methods provide the full sequence of training
amendments, selection rules and follow-up analyses.
