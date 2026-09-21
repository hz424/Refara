# Norman results and selected CPA checkpoint

Recompute the selected-CPA numerical results from the included arrays:

```bash
python -m pip install -r analysis/norman_training_review/requirements.txt
python analysis/norman_training_review/rebuild.py --output-dir outputs/norman-training
```

Run from the repository root and choose a new output directory. The command needs no expression dataset, GPU or model prediction. It reconstructs the 12-candidate training-side selection, all 275 common-target scores, 220 paired differences, the reference-margin tables and the figure-source summaries.

`model/checkpoint_0400.pt` is the single full-training fit used for the current CPA results: per-epoch scheduling, adversarial weight 10 and 400 epochs. The configuration, fixed validation grid and selection rule are recorded in `training_configuration.json`. Selection used 11 held-out training combinations; the 44-condition cell holdout was secondary. The cell-membership table records the internal split of the 68,002 training cells, all of which entered the final refit.

The `CPA_0.8.8` label in the current reference tables denotes this selected checkpoint. The original 200-epoch CPA remains in `../norman_reference/` and `../norman_direct_effect/` for historical reconstruction. Those earlier training commands do not produce the selected checkpoint. `model/provenance.json` and `executed_code_index.tsv` identify the saved fit and executed research code by hash.

This route reconstructs numerical results from saved arrays. The selected checkpoint and training records are included; a portable rerun of the full training search is not included. The expression matrix and per-control model predictions remain in the separate reconstruction route.
