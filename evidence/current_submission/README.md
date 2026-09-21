# Current PBMC results

Reproduce the current numerical summaries for Figure 2, Figure 3 and Extended
Data Figure 1 from the bundled scores and summaries for each fitted seed and
evaluation donor. From the repository root, use the installed Python 3.10 environment:

```bash
.venv/bin/python evidence/current_submission/replay.py --output outputs/current_submission
```

Choose a new output directory. The replay runs on a CPU with Python and NumPy
using about 4.6 MB of bundled inputs and code. It writes `RESULTS.json`, a log
and numerical tables for each component. Successful completion prints
`PASS_CURRENT_SUBMISSION_NUMERICAL_REPLAY`.

Metric-concordance inputs retain the historical `ed9/` directory and output
keys; they correspond to Extended Data Figure 1 in the current manuscript.

The inputs include the completed 2,560-epoch CPA fits for both Influenza PBMC
training groups. Other models and both Parse groups retain their original fits.
The earlier 1,280-epoch CPA replays remain in `../model_comparisons/` and
`../figure3/`.

| Component | Starting inputs | Calculations and checks |
|---|---|---|
| Figure 2 | 15,360 negative-MSE utilities, one per seed, unit, depth, reference design and model, already averaged over allocations | Equal-seed then equal-unit means, all 640 model scores and 2,240 pair comparisons; numerical agreement with current Source Data and the identity of each reversed or tied pair |
| Figure 3 | Displacement and score summaries for each fitted seed and evaluation donor | 576 plotted donor-level values, 24 primary-depth comparisons and 96 comparisons over all depths, checked against current Source Data |
| Extended Data Figure 1 | 5,760 metric-score rows after allocation and seed averaging | Eight-model ranks, leader sets and Kendall concordance for 640 donor-level and 80 aggregate designs, checked against the supplied concordance table |

The Figure 2 replay retains 10 of 28 S-to-D pair reversals in each of the four
groups at its largest depth. All 96 Figure 3 pair/depth comparisons retain their
ordering. Exact values, ties and depth-dependent results remain in the output
tables. Allocations and training seeds do not add biological replication.

Inputs and expected values come from the separately delivered
`05_SOURCE_DATA/CPA2560_selected/` and `CPA2560_figure_sources/` components;
unchanged Parse and non-CPA summaries retain their earlier source records.
Each component manifest records the source paths, hashes and filtering steps.
The [model-comparison](model_comparisons/SOURCE_MANIFEST.json) and
[Figure 3](figure3/SOURCE_MANIFEST.json) source manifests record the numerical
implementations' historical base revision,
`af59f2cccba607fe2c9823d082e35cb537d8bac5`.

The Figure 2 and metric-rank inputs already average the 1,000 control
allocations. Repeating cell selection, training, inference or allocation-level
scoring requires the data and model environments listed in
`reconstruction/current_upstream/README.md` in the source checkout
(`06_CODE_RELEASE/upstream_reconstruction/README.md` in the submission package).
The Figure 3 replay checks the saved summaries; its earlier representation
checks are documented separately in the source receipt.

The Figure 4 program inputs are unchanged by the CPA extension: all 36 selected
model identities remained the same. Their existing replay remains
`../program_responses/replay.py`.

Author-written code is BSD-3-Clause. Influenza PBMC-derived summaries retain
CC BY 4.0; Parse-derived rows and files retain CC BY-NC 4.0. See
[LICENSE_SCOPE.md](LICENSE_SCOPE.md).
