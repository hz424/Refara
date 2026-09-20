# Hierarchical-null simulations

Regenerate the four original C02 hierarchical-null cells (CMP002, CMP006, CMP010 and CMP014) and all 20 procedure-by-root-count event counts from simulated data. The wrapper calls the original `run_cell` in `frozen_original/` with the same numerical defaults and random streams, using local execution paths. These simulations appear in Extended Data Fig. 10b; the output filenames retain the earlier Figure 5b numbering.

Use CPython 3.10 and NumPy 1.26.4. The completed exact replay used CPython 3.10.20. No SciPy, model training or biological data are needed.

```bash
python -m pip install -r simulations/hierarchical_null_v189/requirements.txt
python simulations/hierarchical_null_v189/replay.py \
  --output-dir new-null-replay \
  --verify-against SOURCE_DATA/data/derived/figure2/F2D_PREEXISTING_NEGATIVE_CONTROLS_V1.tsv
```

For independent scheduler jobs, pass `--root-count 8`, `12`, `20` or `40` into the same new output directory. Once all four finish, pass `--aggregate-only --verify-against ...` to produce the complete comparison and receipt. Existing count files cause a failure instead of being overwritten. Failed outer replicates are counted without replacement, and any failure prevents a PASS receipt.

Each root count has 2,500 outer replicates, four models and 12 directional hypotheses. The exact root-sign and CR1-t procedures plus root/task/cell centred unstudentized bootstraps are evaluated on the same generated sample. Each bootstrap uses 499 resamples. Holm adjustment is applied within each procedure to all 12 directions.

The literal seed namespace is `GENERAL_DIRECTIONAL_FAMILY_V3_SOURCE_FREE_VALIDATION_V1`. Each PCG64 seed is the unsigned big-endian integer from SHA256(namespace + NUL + cell_id + NUL + `OUTER` + NUL + six-digit index + NUL + stream purpose). Outer indices are 000000 through 002499; stream purposes are `DATA`, `BOOT_ROOT`, `BOOT_TASK` and `BOOT_CELL`. The full seed ledger and original formulas are included.

`FIGURE_5B_REGENERATED_EVENT_COUNTS.tsv` reports all complete keys, newly generated event counts and archived counts. The September 2026 complete replay reproduced all 20 archived event counts exactly with no failed replicates; its results are supplied in the `Hierarchical_null_replay_v189` Source Data component. Task and cell resampling intentionally treat dependent descendants as independent. These results assess calibration only under the specified simulation.
