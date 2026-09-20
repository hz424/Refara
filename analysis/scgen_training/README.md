# scGen training reconstruction

These programs reconstruct checkpoint prediction risks and saved reference-sensitivity summaries from `SCGEN_TRAINING_ROBUSTNESS_20260910_V1.zip`. This archive is
supplied separately for confidential review and is not part of the public
repository. The programs verify its fixed SHA-256 and all member hashes before
using its prepared folds, saved weights or completed utility arrays.
`review_archive.json` identifies each accepted review distribution by its exact
archive and member-manifest hashes, including the attachment delivered on
12 September 2026. The original archive is also accepted. Scientific inputs
are checked against the same original manifest. Unlisted repackings are rejected;
the existing `--companion-dir` route verifies the consumed scientific inputs
in an extracted attachment.

## Checkpoint risks

Use Python 3.10 with the packages in `requirements-inference.txt`:

```bash
python analysis/scgen_training/reconstruct_checkpoint_risks.py \
  --archive /path/to/SCGEN_TRAINING_ROBUSTNESS_20260910_V1.zip \
  --out-dir outputs/checkpoint-risks --device cpu --threads 2
```

The command reconstructs all 42 states from seven prepared folds, checks 1,200
donor/task risks, 240 donor risks and six candidate means, and recomputes the
archived selection. It uses posterior means, equal-training-donor task shifts
and fold-specific scales. The selected settings are learning rate 0.001 and
80 epochs. The fixed risk-comparison tolerances are `abs_tol=2e-7` and
`rel_tol=2e-6` under `math.isclose`.

CPU inference uses saved tensors through `torch.load(weights_only=True)` and
validates their architecture, dimensions and values. It needs about 8 GB RAM
and several GB of temporary storage. Use a new output directory. This command starts from prepared folds and
checkpoints. Raw-count
preprocessing and external cell-level scoring require separate source inputs.

`validate_upstream_parity.py` optionally compares the implementation with the
original scGen runtime supplied through `--scgen-source PATH`.

## Saved-table aggregation

For saved-table aggregation only:

```bash
python analysis/scgen_training/review_companion.py \
  --archive /path/to/SCGEN_TRAINING_ROBUSTNESS_20260910_V1.zip \
  --out-dir outputs/training-summary --mode all
```

The review attachment contains an edited protocol description and an explicit
record of that editorial projection. For this attachment, saved-risk aggregation
checks the unchanged fold manifests, fit receipts and donor/task diagnostics
against the original scientific manifest, then recomputes the six candidate
risks and selection. The original chronology records are retained as provenance;
this replay creates no new protocol seal.

The three selected-configuration refits reproduce the first three original
scGen states and are not additional independent fitted-state samples. The added analysis evaluates
training-side prediction-based selection procedure.

Software terms are in [LICENSE_SCOPE.md](LICENSE_SCOPE.md). The
confidential archive retains its separate data and checkpoint terms.
