# Reproduction

This directory retains the original 200-epoch CPA analysis. For the current Extended Data Figure 3 and Supplementary Table 8, use [the selected-CPA replay](../norman_training_review/README.md). The training and scoring commands below reproduce the earlier model and results.

Run these commands from this component directory. Set `NORMAN_DATA` to the accompanying Source Data component. Each route uses Python 3.10 and its listed requirements in a separate environment.

## Aggregate summaries

Install `requirements-score.txt`, then run:

```bash
python code/replay_norman_summary.py --input data/utilities.npz --output reproduced_summary
```

This rebuilds the summary tables and checks the reference-role identities from stored utilities. Full-precision tables include every task, model, pattern and assignment.

## Fixed-model predictions

Set `NORMAN_DATA` to the reviewer `Norman_reference_v1814/` Source Data component. Install `requirements-cpa-lock.txt`, then run:

```bash
python code/norman_cpa.py predict --state "$NORMAN_DATA/model/cpa" --controls "$NORMAN_DATA/inputs/eval_controls.h5ad" --tasks "$NORMAN_DATA/inputs/test_tasks.tsv" --output cpa_predictions
```

Inference runs on CPU or GPU and writes one control-cell-by-gene state matrix per test combination, with cell/gene labels and checksums.

## Inputs, fitting and complete scoring

Download the [public count matrix](https://exampledata.scverse.org/pertpy/norman_2019_raw.h5ad), identified by GEO [GSE133344](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344). The preparation command verifies its checksum. In the `requirements-preparation.txt` environment:

```bash
python code/replay_preparation.py --source-h5ad norman_2019_raw.h5ad --metadata-dir metadata --frozen-protocol "$NORMAN_DATA/protocol/FROZEN_TASK_PROTOCOL.json" --reference-dir "$NORMAN_DATA/inputs" --output-dir prepared
```

This also regenerates `train.h5ad`, which is omitted from the distributed component. The full scorer checks that all prepared inputs passed the preparation audit. To repeat CPA fitting, use the CPA environment and a CUDA GPU:

```bash
python code/norman_cpa.py train --input prepared/train.h5ad --protocol "$NORMAN_DATA/protocol/FROZEN_MODEL_PROTOCOL_V2.json" --output cpa_refit
```

In the scoring environment, fit the deterministic baselines and score the fixed-model predictions:

```bash
python code/norman_reference.py fit --data prepared --protocol "$NORMAN_DATA/protocol/SCORING_PROTOCOL_V2.json" --quality prepared/INPUT_AUDIT_PASS.json --output baselines
python code/norman_reference.py score --data prepared --protocol "$NORMAN_DATA/protocol/SCORING_PROTOCOL_V2.json" --quality prepared/INPUT_AUDIT_PASS.json --baselines baselines --cpa cpa_predictions --output rescored
```

Semantic tests run with `python -m unittest discover -s code -p 'test_*.py'`. The CPA runtime-optimization checks and their CPU/GPU equivalence reports are in `checks/cpa_runtime_optimization/`.

## Numerical source tables

To rebuild those tables, set `NORMAN_DATA` to the accompanying reviewer `Norman_reference_v1814/` component, which supplies the complete scored results and their checksums:

```bash
python build_source_data.py --results-dir "$NORMAN_DATA/results" --output-dir rebuilt_figure_source
```
