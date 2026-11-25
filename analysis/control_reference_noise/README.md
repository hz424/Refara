# Control-reference noise and prediction references

These scripts produce the reference-noise, disjoint-cell, cross-individual and
model-generated-reference results of the current manuscript (Fig. 4 panels a and b,
Fig. 5 panels b and c, and Supplementary Tables 11–17). They rescore predictions that
the study had already produced, or ask the fitted models for their output on
control cells. Nothing is refitted and no treated value is used to form a
prediction.

## What this directory contains

- Code only. No expression data, prediction arrays, fitted models or result
  tables are bundled.
- The inputs are study outputs that are not distributed in this repository:
  per-cell prediction arrays are several gigabytes per fit, and Parse-derived
  values are subject to the provider's CC BY-NC 4.0 terms
  ([licence scope](../../LICENSE_SCOPE.md), [data access](../../docs/DATA_ACCESS.md)).
  The result tables are provided as Source Data with the manuscript.
- The scripts are not part of the installed `reference-design` package and are
  not exercised by the test suite. They were checked by reproducing released
  utilities (see "Built-in checks").

## Inputs

Set these environment variables; a script stops with a message naming the one
it needs ([`data_locations.py`](data_locations.py)).

| Variable | Content | Needed by |
|---|---|---|
| `REFARA_EVALUATION_ROOT` | Directory with `<dataset>/<regime>/` evaluation outputs (layout below) | all scripts |
| `REFARA_OUTPUT_ROOT` | Directory for outputs; created if absent | all scripts except `control_bank.py` |
| `REFARA_TRAINING_CODE` | The study's training and evaluation code (model loaders) | `control_bank.py` |
| `REFARA_PARSE_TASKS` | Evaluation task table of the Parse analysis (TSV) | `disjoint_cells.py` |
| `REFARA_PARSE_CONTROL_POOL` | Directory with `pool_expr_log1p.npy` and `groups.json`: 768 hash-ranked control cells per donor and cell type | `disjoint_cells.py` |

Layout under `REFARA_EVALUATION_ROOT`, with datasets `parse` (regimes
`common48`, `common128`) and `gse162632` (regimes `original8`, `cap48`):

```text
<dataset>/<regime>/EVALUATION_BINDING.json
<dataset>/<regime>/PANEL_SEED_UTILITIES.npy     (panel, seed, allocation, depth, design, donor, family)
<dataset>/<regime>/banks/<Family>__primary__<seed>/points.npy    state outputs: task x control cell x gene
<dataset>/<regime>/banks/<Family>__primary__deterministic/effects.npy    direct effects: task x gene
<dataset>/<regime>/score_cache_v1/{treated,canonical_controls,group_index,root_index,task_index,
                                    max_depth_allocations,scales,weights}.npy
```

## Scripts

Run them from this directory, in this order. Output paths below are relative
to `REFARA_OUTPUT_ROOT`; for steps 1, 5 and 6 pass the full path to `--out`.

| Step | Script | Writes | Manuscript item |
|---|---|---|---|
| 1 | `cross_individual_score.py --regime R --fit F --out cross_individual/per_fit/R__F.npz`, once per released Parse fit (28) | per-fit utilities for own-donor and cross-individual assignments | Fig. 5c |
| 2 | `cross_individual_aggregate.py` | `cross_individual/XDONOR_UTILITIES.tsv`, `XDONOR_PAIRS.tsv`, `XDONOR_SUMMARY.json` | Fig. 5c, Supplementary Table 13 |
| 3 | `reference_noise_decomposition.py` | `reference_noise/REFERENCE_NOISE_*.tsv` | Fig. 5b, Supplementary Tables 11 and 12 |
| 4 | `disjoint_cells.py` | `reference_noise/DISJOINT_CELL_TEST.tsv` | Supplementary Table 14 |
| 5 | `control_bank.py --dataset D --regime R --fit-key F --out own_control/control_points/D/R__F.npy`, once per state-output fit (36) | the model's output for each control cell under the control condition | Fig. 4a,b |
| 6 | `own_control_score.py --regime R --fit F --out own_control/per_fit/R__F.npz`, once per Parse state-output fit (18) | per-fit utilities with the raw control mean or the model's own control output as the prediction reference | Fig. 4a,b |
| 7 | `own_control_aggregate.py` | `own_control/OWN_CONTROL_UTILITIES.tsv`, `OWN_CONTROL_PAIRS.tsv` | Fig. 4a,b, Supplementary Tables 15 and 16 |
| 8 | `own_control_influenza.py` (about 25 minutes on one core) | `own_control/OWN_CONTROL_*_GSE162632.tsv` | Fig. 4b, Supplementary Tables 15 and 16 |
| 9 | `control_offset.py` | `own_control/CONTROL_OFFSET_NOISE_CORRECTED.tsv`: distance between each model's control output and the control mean, with and without the sampling variance of the measured mean | Supplementary Table 17 |

A fit name is `<Family>__primary__<seed>` for scGen, CPA and CellOT (seeds 17,
29 and 43) and `<Family>__primary__deterministic` for NC, CM, TW, PCA and RBF.
Step 1 takes one to fifteen minutes per fit and reads one prediction array at a
time; steps 2 to 4, 7 and 9 take seconds to a few minutes.

## Definitions

Utility is negative standardized mean squared error, with tasks weighted
equally within donor and donors equally. A role is a block of control cells;
tuples are ordered (observation reference, prediction reference, model input).

| Code | Assignment |
|---|---|
| `S`, `M`, `P`, `O`, `D` | Own-donor designs: shared; model input, prediction reference or observation reference separated; all three separated |
| `OX` | Model input and prediction reference are the same block of another donor; observation reference is the donor's own |
| `PX` | Own model input; prediction reference from another donor |
| `MX` | Model input from another donor; own prediction reference |
| `DX` | Model input and prediction reference from two different other donors |
| `raw`, `own` | Predicted effect = state output minus the raw mean of the input control cells, or minus the model's own output for those cells under the control condition |
| suffix `n16`, `n48`, `full` | 16 or 48 control cells per reference (Parse), or the whole canonical pool as one block |

## Built-in checks

- `cross_individual_aggregate.py` compares the own-donor designs with the
  released utilities for every allocation, seed and donor and prints the
  largest difference (4.5e-10 for Parse).
- `own_control_influenza.py` does the same for designs S, O and D (5e-9).
- `control_bank.py` reproduces three tasks of the released predictions with the
  loaded model before writing any control output, and stops if they differ by
  more than 5e-3 (observed: below 3e-5).
- Control outputs are reproducible to about 1e-5 per cell and 1e-7 for stratum
  means across thread counts and hardware, not bit for bit.

The control condition is the decoded latent without a perturbation shift for
scGen, the control category of the perturbation encoder for CPA, and the
autoencoder reconstruction (identity transport) for CellOT.
