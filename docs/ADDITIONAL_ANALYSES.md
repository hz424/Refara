# Additional numerical analyses

Use this guide for simulations, training reviews and earlier benchmark
analyses. For the current manuscript results, start with the
[reproduction guide](REPRODUCING.md). Historical version labels below refer to
input versions, so their figure numbers may differ from the current paper.

## Starting inputs

| Component | Starting input | Guide |
|---|---|---|
| Focal ridge–scGen comparison | Bundled treated means, references, PCA effects and scGen predictions | [Focal capsule](../capsules/gse162632_scgen/README.md) |
| Influenza PBMC allocation | Bundled label-level utilities and pairwise interactions | [Allocation capsule](../capsules/gse162632_allocation/README.md) |
| Held-out families and conditioning changes | Manifest-bound root/task utilities in separate Source Data | Aggregate command below |
| Biological consequences | Cached predictions, controls, treated means and fixed program mapping in Source Data | [Component guide](../analysis/biological_consequence/README.md) |
| Selected Norman CPA | Bundled checkpoint, saved arrays, configuration and membership records | [Selected-CPA replay](../analysis/norman_training_review/README.md) |
| Replication and test choice | Author-generated simulations with specified random streams | [Low-replication analysis](../analysis/low_replication/README.md) |
| Donor–batch matching | Aggregate summaries; graph-level verification needs authorized individual inputs | [Data access](DATA_ACCESS.md) |

## Aggregate replay with Source Data

Use the `05_SOURCE_DATA` directory with the 1.10.0 binding from the matching
submission package. It contains the corrected historical Jerber scGen
results: applying cell-type shifts to all 20 donors changes its shared
reference rank from 1 to 5; its independent-split and cross-fit ranks remain 6.
The [source records](../provenance/README.md) distinguish this correction from
the preserved 1.8.3 hashes. The validator rejects earlier Source Data packages.

The directory itself can be renamed; keep its internal paths and values intact
so the manifest checks succeed. The historical numerical runtime uses Python
3.10.17 and the pins in `requirements/locked.txt`, separately from the main
toolkit environment:

```bash
/path/to/python3.10.17 -m venv .venv-replay
.venv-replay/bin/python -m pip install -r requirements/locked.txt
mkdir -p outputs/analysis
.venv-replay/bin/python scripts/replay_all.py \
  --source-data-root /path/to/05_SOURCE_DATA --analysis-only \
  --receipt outputs/analysis/REPLAY_RECEIPT.json
```

This reconstructs the 84-member held-out family, 56 paired construction changes,
structural-zero correction, multi-realization sensitivity, conditioning changes
and factor-isolation summaries. It starts from supplied utilities and summary
tables. Per-component guides describe lower-level inputs when available.

For the selected Norman CPA, use the main toolkit environment:

```bash
.venv/bin/python analysis/norman_training_review/rebuild.py \
  --output-dir outputs/norman_training
```

This recalculates scores from saved arrays and compares them with the expected
results. It starts after the original training search. For earlier Norman model
configurations, see `analysis/norman_reference/` and
`analysis/norman_direct_effect/`.

## External-input tests

```bash
REFERENCE_CELL_SOURCE_DATA_ROOT=/path/to/05_SOURCE_DATA \
  .venv/bin/python -m pytest -q
```

Tests declare when they need Source Data, prepared training inputs or authorized
graph records. Missing inputs produce explicit skips. Detailed sources and
redistribution terms are in [data access](DATA_ACCESS.md); the evidence timing
and inferential scope are in [evidence timing](EVIDENCE_TIMING.md) and
[statistical interpretation](STATISTICAL_INTERPRETATION.md).
