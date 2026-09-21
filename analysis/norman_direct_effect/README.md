# Direct-effect comparisons in the Norman screen

This directory retains the original 200-epoch CPA analysis. For the current Extended Data Figure 3 and Supplementary Table 8, use [the selected-CPA replay](../norman_training_review/README.md). The training and scoring commands below reproduce the earlier model and results.

This analysis compares a fixed direct-effect ridge predictor with three state-output predictors. Five-fold validation within the original training combinations selected its penalty. All 55 test combinations, 30 reference allocations and six control depths enter each effect/state comparison.

From this directory, use a separate Python 3.10 environment with `requirements-replay.txt`:

```bash
python code/replay_effect_summary.py --output effect-summary-replay
```

The command reconstructs all four pair tables from the included effect utilities and the retained `../norman_reference/data/utilities.npz`. It checks the margins, reference distance, reversal condition and table values. The replay uses saved utilities. The training inputs, fitted coefficients and fixed gene-level predictions are supplied in reviewer Source Data.

`results/task_allocation_pairs.tsv.gz` contains every task/allocation/depth/state-model comparison. The other three tables average over allocations, tasks, or both before testing the reversal condition. Every row reports `d_S`, `V` and `d_D` with the identity `d_D = d_S + V`. Positive margins favour the effect predictor; numerical ties use a tolerance of 10⁻¹².

`model_selection/` records all candidate penalties, fold membership and validation errors. `protocol/` records the fixed model, selection and evaluation rules. This analysis followed inspection of the original state-model results. Its model and selection rules were fixed before fitting and evaluation.

The source workbook contains both state-model and effect/state comparisons. All results describe one pooled screen; control-allocation ranges measure sensitivity within that screen.

## Full reconstruction

Install `requirements-score.txt` and obtain the original prepared Norman inputs from reviewer Source Data or the preparation route in `../norman_reference/REPRODUCING.md`. From this directory:

```bash
norman_source=/path/to/REVIEWER_FILES/Source_Data/Norman_reference_v1814
OPENBLAS_NUM_THREADS=1 python code/fit_direct_effect.py \
  --prepared-dir "$norman_source/inputs" \
  --protocol protocol/DIRECT_EFFECT_PROTOCOL_V1.json \
  --output-dir outputs/effect-fit

python code/score_direct_effect.py \
  --protocol protocol/DIRECT_EFFECT_PROTOCOL_V1.json \
  --fit outputs/effect-fit --original "$norman_source" \
  --output outputs/effect-scores

OPENBLAS_NUM_THREADS=1 python code/audit_direct_effect.py \
  --protocol protocol/DIRECT_EFFECT_PROTOCOL_V1.json \
  --original "$norman_source" --fit outputs/effect-fit \
  --results outputs/effect-scores --output outputs/effect-audit.json
```

The fitting command repeats the fixed validation rule and refit; the scoring command uses the original evaluation inputs and retained state utilities. The independent audit recomputes fitting and scoring without importing their production functions. These commands require the separate expression-derived inputs; the compact public replay above does not.
