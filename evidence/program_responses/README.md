# Reproduce program-response comparisons

Run from the repository root in the main toolkit environment:

```bash
.venv/bin/python evidence/program_responses/replay.py --output outputs/program_responses
```

Use a new output directory. The command recalculates the program-response comparisons in Figure 4 and writes numerical results.

For the eight original donors, the supplied table contains the observed and S-selected/D-selected predicted scores for every donor, five cell types, 37 Hallmark programs, three separately scored seeds, and both observation references. The replay checks that all combinations are present, predictions stay fixed between observation references, and seed means match the records. It calculates squared residuals for each seed before averaging seeds, programs, cell types and equally weighted donors.

Mean predicted program scores disagree in direction, with both absolute scores exceeding 0.025, for **75/1,480** task–program combinations under original8 training and **65/1,480** under cap48 training. Predictions remain fixed when the observation reference changes. The resulting mean program-error difference, D-selected minus S-selected, changes sign. Values below are in units of 10⁻³; the output tables retain full precision.

| Training regime | Original eight, B1 → B2 | Additional 39, B1 → B2 |
|---|---:|---:|
| original8 | +2.24 → −3.99 | +4.24 → −3.35 |
| cap48 | +3.50 → −2.36 | +3.23 → −3.51 |

Positive values favour S-selected models; negative values favour D-selected models. These comparisons describe mean program errors under the specified reference choices within the same study.

For the additional 39 same-study donors, computation starts from saved donor-level errors. It averages donors within each batch/capture orientation, the two orientations within each batch, and eight batches equally, then compares the results with the published means. For the original eight donors, it also checks the observation-shift identity at each seed/task/program and compares recalculated donor errors with the supplied table.

`MANIFEST.json` records hashes and submission-relative source paths; `expected.json` contains published numerical targets. Inputs use pseudonymous root/donor labels. Computation starts after model selection, inference and gene-to-program aggregation. The inputs are locally authored GSE162632-derived numerical summaries under CC BY 4.0; that grant does not cover the underlying GEO data or third-party model implementations. Only program identifiers, not gene-set membership definitions, are included.

Run `.venv/bin/python evidence/program_responses/selection_replay.py --output outputs/program_selection` to recompute all 54 S/O/D choices, tie sets and margins from 384 bundled candidate-utility rows. This checks the 18 paired selection cases against the verified roster; the program-score replay above retains the historical S/D numerical record, and neither replay establishes generalization to independent experiments.
