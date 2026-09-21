# Genetic-perturbation reference sensitivity

This directory retains the original 200-epoch CPA analysis. For the current Extended Data Figure 3 and Supplementary Table 8, use [the selected-CPA replay](../norman_training_review/README.md). The training and scoring commands below reproduce the earlier model and results.

This analysis predicts 55 held-out two-gene combinations in the Norman K562 CRISPRa screen. Additive, compositional ridge and CPA predictors return expression states; zero effect is a separate diagnostic baseline. Thirty fixed control allocations cover six reference depths. All model pairs and tasks are retained.

Use the utilities in `data/` to rebuild the aggregate summaries in `figure_source/`. The reviewer Source Data component, `Norman_reference_v1814/`, contains complete scores, normalized evaluation inputs and the fixed CPA state. The reproduction guide also explains how to download the raw counts and regenerate per-control predictions.

| Route | Environment | Guide |
|---|---|---|
| Recompute summaries from supplied utilities | `requirements-score.txt` | [Reproduction](REPRODUCING.md) |
| Recreate CPA predictions from the fixed checkpoint | `requirements-cpa-lock.txt` | [Reproduction](REPRODUCING.md) |
| Recreate normalized inputs and repeat fitting/scoring | Separate preparation, CPA and scoring environments | [Reproduction](REPRODUCING.md) |

The eight capture partitions belong to one pooled experiment. Allocation ranges describe sensitivity within that screen. Source citations and terms are in [LICENSE_SCOPE.md](LICENSE_SCOPE.md).

Quick replay after installing `requirements-score.txt`:

```bash
python code/replay_norman_summary.py --input data/utilities.npz --output reproduced_summary
```
