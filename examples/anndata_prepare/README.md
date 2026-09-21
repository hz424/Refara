# Prepare an analysis from AnnData

Prepare and score three synthetic tasks in two units using processed cell expression and two prediction matrices in AnnData files. The example matches controls by unit and cell type and allocates two disjoint blocks within each batch.

From the extracted repository root with Python 3.10 available:

```sh
python3.10 -m venv .venv
.venv/bin/python -m pip install --no-deps -r requirements/vcc-build-py310.txt
.venv/bin/python -m pip install --no-deps -r requirements/vcc-py310-linux.txt -r requirements/unified-prepare-py310.txt
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
.venv/bin/python -m pip check
.venv/bin/python examples/anndata_prepare/make_example.py --output /tmp/reference-anndata-inputs
.venv/bin/reference-design prepare /tmp/reference-anndata-inputs/manifest.json --output /tmp/reference-anndata-prepared
.venv/bin/reference-design run /tmp/reference-anndata-prepared/plan.json --output /tmp/reference-anndata-results
```

The VCC setup script creates the same preparation-capable Python 3.10 environment
under `py310/`; use those executable paths when it is already installed. The
historical `prepare-locked.txt` remains available for the separately recorded
protocol.3 environment.

Use new or empty output directories. Inspect `preview.md` in the prepared directory before interpreting the results. The preview identifies each task's unit, target condition, cell counts and strata. `prepare_receipt.json` contains the detailed selections/counts, declared preprocessing description and input/output file hashes. The prepared directory is portable; move it as a whole and run its plan without the original AnnData files.

The example contains 18 cells, three genes and three tasks. Cell expression comes from the `normalized` layer of cells.h5ad (`X` is deliberately zero-filled); each prediction AnnData uses `X`. The two-block plan runs the primary heldout-effect comparison; requested five-pattern diagnostics are reported unavailable because they require three blocks. Units receive equal weight even though one has two tasks and the other has one.

## Use your own files

Copy the generated manifest and update its paths and metadata mappings. Supply processed cell expression and existing predictions on the same declared scale. Complete normalization, gene selection, fitting and prediction before preparation.

- Cells: choose `format: "h5ad"`, `path`, and explicit `layer` (`"X"` or a named layer). Cell IDs come from obs_names and genes from var_names. Set `unit_column`, `condition_column`, `strata_columns`, and `context_columns` to actual obs column names. Use empty lists when there is no stratification or additional context matching. Metadata must contain meaningful strings; explicitly convert numeric categories before saving AnnData. Duplicate IDs or missing annotations are rejected.
- Tasks: provide a TSV with `task`, `unit`, `condition`, plus exactly the context columns. Each row selects treated cells matching that unit, condition and context. Controls match the same unit/context and the declared `control_value`. Do not make a task out of a control condition. Unlisted cell groups are ignored and counted.
- Gene panel: the default uses all cell genes. An optional top-level `gene_panel` list explicitly selects a subset from cell and prediction AnnData. Missing selected genes cause an error. Gene order is aligned by labels. No automatic intersection is performed.
- Models: declare `name`, `kind` (`state` or `effect`), `conditioning` (`fixed` for a supplied task-level prediction), and `representation` (`native`, or `effect` when a state prediction was stored after baseline subtraction). Either output kind can be fixed or block-conditioned. Native effects use `representation: "native"` and receive no prediction-baseline subtraction. A state stored as an effect requires its original `baseline`, with the same matrix input schema as predictions.
- Predictions: use a wide TSV path with `task` plus the selected gene columns, or an object such as `{"format":"h5ad","path":"my_predictions.h5ad","layer":"X","task_column":"__index__"}`. Use a named obs column instead of `__index__` when task IDs live there. Every declared task must occur exactly once for every model and baseline. Predictive uncertainty samples must first be reduced according to your chosen prediction definition.
- Target: `heldout_effect` compares native effects, or B1-centred state predictions, with treated expression minus held-out B2. `shared_effect` uses B1 for both observation reference and state centring. `treated_state` compares supplied states directly with treated expression. Declare prediction controls as `available` or `unavailable`; state-to-effect centring and all block-conditioned predictions require available controls. Fixed native effects can be evaluated against held-out controls even when the predictor receives none.
- Aggregation: explicitly choose `treated_aggregation: "cell_mean"` or `"equal_stratum_mean"`. The latter averages each represented treated stratum equally. Reference blocks always contain equal depth from every matched control stratum. Choose strata and units according to the experimental design; equal weighting alone does not make them biologically interchangeable.
- Allocation: effect targets and block-conditioned state models require `blocks` (2 or 3), `depths`, `allocations`, and `seed`. Each control stratum needs at least blocks × maximum depth cells. Fixed-only treated-state evaluation omits `allocation` and needs no control cells. Do not interpret different allocations or random seeds as new biological units.

For TSV cell inputs, use `format: "tsv"`, `path`, `cell_id_column`, and an explicit `gene_columns` list in place of the AnnData layer. Keep the same metadata mappings and task table. This route does not import AnnData:

```sh
.venv/bin/python examples/anndata_prepare/make_example.py --format tsv --output /tmp/reference-tsv-inputs
.venv/bin/reference-design prepare /tmp/reference-tsv-inputs/manifest.json --output /tmp/reference-tsv-prepared
.venv/bin/reference-design run /tmp/reference-tsv-prepared/plan.json --output /tmp/reference-tsv-results
```

## Generate predictions from allocated cells

Run the complete cell-to-prediction workflow in the same environment:

```sh
.venv/bin/python examples/anndata_prepare/block_conditioned.py --output /tmp/reference-block-example
```

Add `--format h5ad` to use AnnData inputs. The default uses TSV and needs no AnnData dependency. This example uses the same synthetic cells and a small affine state predictor with declared coefficients.

The script first prepares the fixed baselines into `allocated/`, freezing the controls and their membership. It then reads each allocation/depth/block's cell IDs, passes those cells to `predict_cells`, and saves a separate prediction for each block. The predictor applies gene-specific gains and condition-specific offsets to the cell values before averaging; its predictions vary with the selected cells. Replace `predict_cells` with your model's inference and record the model parameters used.

`inputs/block_manifest.json` binds the predictions to the membership hashes. Each provenance file also records hashes of the predictions, source cells and task table, controls, membership, reference means, coefficients and prediction code. Preparation verifies these files, then writes a portable plan into `prepared/`. Scoring writes `results/primary_allocation_scores.tsv` and `results/primary_summary_scores.tsv`. The primary comparison uses B1 for model input and state centring, and B2 for the observed effect.

Changing the allocation seed requires generating predictions again: preparation rejects the old membership hashes. Editing a recorded prediction or input file also fails its provenance check. These checks establish file consistency; the executable predictor shows how the recorded cells enter inference.

## Supply existing block-conditioned predictions

The basic example uses fixed supplied predictions. If predictions already exist for each block, declare `conditioning: "block"` and use `prediction_files` mapping task IDs to the existing allocation/depth/block TSVs. Also supply `membership_sha256` mapping task IDs to the SHA256 of each corresponding prepared membership table. A mismatch is rejected. For states stored as effects, provide `baseline_files` with the same task mapping. No prediction is recomputed under a new control block.

The digest check verifies the supplied allocation files. Optional `provenance_files` document which cells the predictor consumed by mapping tasks to manifests containing `files: [{path, sha256}]`. Their files are verified and copied into the prepared directory. Record the original fitting and prediction choices; a new allocation does not change the inputs used to generate a saved prediction.

The H5AD reader loads the complete selected AnnData file into memory before selecting rows and genes. Budget memory accordingly, or prepare a smaller input file. Hashing streams the file bytes; AnnData loading is not backed or out-of-core.
