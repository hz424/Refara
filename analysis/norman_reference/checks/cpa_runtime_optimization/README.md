# CPA training and prediction

This component fits CPA 0.8.8 for held-out two-gene combinations and predicts expression from separate non-targeting control cells. The fixed run uses seed 17 and 200 epochs. Training requires a CUDA GPU; inference also supports CPU.

Create a Python 3.10.17 environment and install `requirements-cpa-lock.txt`. The lock contains the exact packages used for training. Input preparation and score aggregation use their separately documented environments.

```bash
python -m pip install -r requirements-cpa-lock.txt
python norman_cpa.py train --input train.h5ad --protocol model_protocol.json --output cpa_state
python norman_cpa.py predict --state cpa_state --controls eval_controls.h5ad --tasks test_tasks.tsv --output cpa_predictions
```

The prepared H5AD files share a 2,000-feature Ensembl axis on the whole-transcriptome CP10K-log1p scale. Their `condition` column uses `ctrl`, a gene symbol, or `GENE1+GENE2`. Training and inference cell IDs must be disjoint. Each query pair must be absent from training, with both constituents present as single-gene conditions. Predictions are native expression states; the scorer performs the specified baseline conversion.

`predictions.tsv` maps each target to a float32 matrix of control cells × genes. The companion cell and gene tables define its axes. Regenerate these arrays from the saved state and control cells; measured test responses are not model inputs.

The production driver omits per-batch R² logging, which occurs after optimizer updates and is unused by fitting. Synthetic CPU and GPU checks found identical backward gradients, losses, optimizer states, random-number states, epoch states and final models. Training history includes reconstruction and adversarial diagnostics. See `verification/` for the checks and exact code change.

To rerun the implementation check on synthetic data:

```bash
python verification/verify_unused_metrics.py --driver verification/reference_driver.py --output verification_run
```

Add `--gpu` to repeat it on CUDA. The check compares enabled and omitted logging in the unchanged reference driver using synthetic data.

`verification/audit_scored_tuples.py` independently reconstructs all 27 reference-role assignments for the first and last frozen tasks, allocations 0 and 29, and depths 8 and 135. It reads per-cell predictions and controls directly and compares both atomic and pattern-averaged scores without importing the main scorer.
