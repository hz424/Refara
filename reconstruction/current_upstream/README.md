# Reconstruct the current PBMC analyses

Use these tools to locate and check the files needed for the current PBMC
analyses, continue CPA training to 2,560 epochs, or regenerate CPA predictions.
For a quick check of the reported values, use
`evidence/current_submission/replay.py` in the source checkout or
`06_CODE_RELEASE/current_submission_replay/replay.py` in the submission package.

Training and cell-level prediction require additional model and data files.
`CURRENT_RECONSTRUCTION_MANIFEST.json` lists their relative paths, hashes, roles
and access status, including whether each file is supplied for review. The tools
work with files on your machine; obtain the required inputs through the access
routes below.

## What can be reconstructed

Paths beginning with `05_SOURCE_DATA/` or `06_CODE_RELEASE/` refer to the
separate submission package.

| Starting inputs | Where to start | Reproduced quantities |
|---|---|---|
| Current seed/root score and displacement summaries | `06_CODE_RELEASE/current_submission_replay/replay.py` | Current Figures 2–3 and Extended Data Figure 9 summaries, including CPA2560. Starts after allocation scoring. |
| Current allocation-level utility arrays | `05_SOURCE_DATA/CPA2560_selected/selected_2560/mse/` | Model means, pair margins, depth-dependent reversals and seed/root aggregation. Model fitting and cell-level scoring are upstream. |
| Validation donor risks and evaluation utility arrays | `05_SOURCE_DATA/Validation_reference_selection/replay.py` | All 128 configuration choices and six complete external comparison tables. |
| Derived CellFlow prediction, treated and control means | `05_SOURCE_DATA/Supplementary_Figure_7_Kang_selected/replay_selected.py` | 960 input-change rows, five utility tensors and 32 donor endpoints. Training and generation are upstream. |
| Cached real predictions and Norman score cube | `06_CODE_RELEASE/reference-role-value-v1/code/replay_bundle.py` | Four metric summaries and 95 role-operation cases. |
| Confidential Sound Life anonymous graph and utilities | `Sound_Life_Graph_Utilities_for_Review.zip` | Matching geometry, complete directional classifications and 144 G12 matchings, from a separate confidential attachment of derived inputs. |
| Authenticated CPA1280 checkpoint and prepared full-training bundle | `native_cpa.py resume` | A new CPA2560 continuation using the original trainer and optimizer/scheduler/RNG checkpoint. Requires separately obtained assets and the native CUDA environment. |
| Authenticated CPA2560 fitted state and supplied control counts | `native_cpa.py predict` | Per-cell predicted states on the scored gene panel, using the fitted model and supplied controls. Normalization uses the complete gene axis before panel selection. |

## Bind your asset directories

Copy `ROOTS.example.json` to a local `ROOTS.json` and enter the directories
containing your inputs. Relative paths resolve beside that JSON file. The
`submission` directory must contain sibling `05_SOURCE_DATA/` and
`06_CODE_RELEASE/` directories. Set the roots needed for your chosen operation;
the checker reports any missing directory or file.

Across four PBMC training groups, the manifest lists 81 selected or retained
candidate models and their CellOT component weights, 26 prepared training and
validation bundles, and 81 saved prediction banks. The upstream archive contains
about 180 GB of prediction arrays and 49 GB of prepared arrays; these large
files require a separate transfer.

The confidential attachment `06_CODE_RELEASE/current_native_sources.zip`
supplies the 26 source-code, protocol and allocation records selected by
`--profile code`. Its `ROOTS.json` locates the files after extraction, and
`additional_artifacts` in the manifest gives the archive hash. Model weights,
prepared expression and prediction banks are separate inputs. The original
protocol records include historical locations and donor/capture identifiers,
so this attachment is supplied through confidential review rather than GitHub.

The manifest's `root_roles` entries describe access to each input group. Raw
data are available from GSE162632 and the Parse PBMC dataset. The exact prepared
bundles, cell memberships, fitted models and prediction banks require a separate
author-supplied archive under the original provider terms. A public checkpoint
archive or DOI is not yet available.

## Check assets without reading every large bank

The tools need Python 3.10 or later. Asset checks, plans and exports use only the standard library.

```bash
python reconstruct.py check --roots ROOTS.json --profile delivered --output delivered-check.json
python reconstruct.py check --roots ROOTS.json --profile all --hash-limit-mib 1 --output archive-inventory.json
```

The second command hashes files up to 1 MiB and checks only existence and size
for larger files. It reports `COMPLETE_PARTIAL_HASH_INVENTORY` when these checks
pass. Each file receives `HASH_MATCH`, `SIZE_ONLY_HASH_NOT_CHECKED`, or a missing
input or mismatch status. Omit the limit to verify every selected file's hash.

Use `--profile models`, `prepared`, `banks`, `evaluation-inputs` or `code` to
select an input group. `--group gse162632/original8` restricts the relevant study
records. `--recipe original8:17` selects the inputs for one CPA continuation,
including its prepared training bundle.

```bash
python native_cpa.py check --roots ROOTS.json --regime original8 --seed 17 --output cpa-input-check.json
```

This checks the selected file hashes, prepared training bundle, donor split and
original training settings. Add `--check-runtime` in the native CPA environment
to check the six recorded package versions. This input check leaves the
optimizer checkpoint unopened; training is a separate command.

## Generate and execute native commands

```bash
python reconstruct.py plan --roots ROOTS.json --recipe original8:17 \
  --native-python /path/to/native-cpa-python --output cpa-plan.json
```

The plan saves commands for input checks, training continuation and prediction.
To run the recorded CPA continuation, execute:

```bash
/path/to/native-cpa-python native_cpa.py resume --roots ROOTS.json \
  --regime original8 --seed 17 --device cuda --output new-cpa-fit
```

Use the recorded native environment: cpa-tools 0.8.8, scvi-tools 0.20.3,
PyTorch 2.0.0, pytorch-lightning 1.9.5, NumPy 1.23.5 and AnnData 0.9.2. Keep it
separate from the toolkit's Python 3.10/NumPy 2.2.6 environment. The script checks
these versions and the original trainer, loader and scoring-code hashes before
calling the same `fit_bundle` operation used for the completed continuation.
Choose a new output directory. Release checks validated the inputs and existing
continuation records; they did not repeat training.

For CPA inference:

```bash
/path/to/native-cpa-python native_cpa.py predict --roots ROOTS.json \
  --regime original8 --seed 17 --device cpu \
  --counts native-control-counts.npz --features source-features.tsv \
  --labels prediction-labels.tsv --output new-cpa-prediction
```

Supply counts as a SciPy sparse NPZ, with one row per control cell and columns
in the complete fitted gene order. The feature TSV must list `feature_id` in
that order; the label TSV must give `context` and `condition` for each requested
prediction. The command verifies the fitted state and source-code hashes, then
writes `predicted_states.npy` and `RECONSTRUCTION_RECEIPT.json`. Prediction uses
the completed fit and supplied counts, so the prepared training matrix and
earlier optimizer checkpoint are unnecessary.

To transfer the files needed for this prediction:

```bash
python reconstruct.py export --roots ROOTS.json --profile cpa-predict \
  --recipe original8:17 --destination cpa-inference-assets --output cpa-export.json
python reconstruct.py check --roots cpa-inference-assets/ROOTS.json --profile cpa-predict \
  --recipe original8:17 --output cpa-export-check.json
```

This profile copies the selected CPA2560 state and four native Python sources:
`train_cpa.py`, `common.py`, `prepare_data.py` and `predict_cpa_points.py`.
For `original8:17`, these five files total 80,944,044 bytes. Move the export as
a whole and pass its `ROOTS.json` to `native_cpa.py predict`, together with your
counts, features and labels. A recipe supplied without `--profile cpa-predict`
keeps the continuation selection, including prepared training inputs.

The manifest lists the other families' scripts and model files under
`native_custom_code` and `fitted_model`. Their original commands include
`train_baselines.py --bundle --candidate --output`,
`train_scgen.py --bundle --protocol --protocol-sha256 --lr --epochs --checkpoints --seed --output`,
and `train_cellot.py refit --bundle --protocol --representation --checkpoint --seed --output`.
Read the scripts' arguments and the complete candidate grids before execution.

The CPA commands accept relocated inputs. The full four-group evaluator also
checks training-completion records and nested file references that retain the
original locations. Running that remaining pipeline on another machine requires
a relocation check after its external inputs have been obtained.

## Copy selected inputs

To inspect a transfer without copying large arrays:

```bash
python reconstruct.py export --roots ROOTS.json --profile banks --dry-run \
  --destination selected-assets --output transfer-plan.json
```

To create a new local directory containing a selected set:

```bash
python reconstruct.py export --roots ROOTS.json --profile code \
  --destination code-assets --output code-export.json
```

Export copies the selected files into a new directory, verifies their hashes
and sizes, and writes relative paths in `ROOTS.json` plus the file list in
`SELECTED_ASSETS.json`. It also accepts `--profile models`, `--profile prepared`
or `--recipe original8:17`. Check the dry-run size before a large transfer.

Original metadata can contain source identifiers and historical locations;
preserve its access conditions when sharing a confidential review copy. Export
creates a local copy, and redistribution remains subject to the source terms.
If copying fails, the partial directory and failure receipt remain available
for inspection.

## Verification

`VERIFICATION.json` records the completed checks. Using one CPA2560 fit, we
regenerated predictions for 24 real control cells on CPU. All 48,000 values
agreed with the saved predictions within the declared absolute tolerance of
10⁻⁵; the largest difference was 3.3 × 10⁻⁶. This check covers inference for
those cells from the existing fit.

The same 24-cell check also passed after exporting the five `cpa-predict`
assets and moving the code, fitted state and input fixture to a new directory.
The relocated run reproduced all 48,000 values within the same tolerance.

Run the standard-library tests with:

```bash
python -m unittest discover -s tests -v
```

These tests use synthetic files to check relative paths, rejection of changed
or missing inputs, and export to a relocated directory. The private review run
also reconstructed the Sound Life analysis and passed its graph integration
test using the three inputs from the separate confidential attachment.
