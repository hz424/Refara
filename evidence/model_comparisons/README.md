# Reference sharing and model comparisons

This replay preserves the earlier 1,280-epoch Influenza CPA results. For the
current 2,560-epoch results, use the [current PBMC replay](../current_submission/README.md).
The other model fits and both Parse groups are unchanged.

Recompute model rankings, pairwise margins and reversal identities for eight
prediction configurations across four training groups. From the repository
root, run:

```bash
.venv/bin/python evidence/model_comparisons/replay.py --output outputs/model_comparisons
```

The command uses the Python standard library and bundled TSV inputs to produce
numerical tables.

## What is recomputed

The input contains 15,360 scored utilities: negative MSE for each
seed, biological unit, reference depth, design and model. Each supplied value
has already been averaged over 1,000 balanced equal-depth allocations.
The command averages the three scored seeds within each unit, then averages
the eight units equally. It reconstructs all 28 model-pair margins, complete
rankings and reversal identities at every registered depth. Higher utility
means lower MSE; a positive A-minus-B margin favours A.

The configurations are NC, CM, TW, PCA, RBF, scGen, CPA and CellOT. The groups
are Influenza PBMC with 8 cells or a 48-cell cap per training stratum,
and Parse with 48 cells or a 128-cell cap per training stratum. A stratum
combines donor, cell type and condition, including controls. The
evaluation reference depths are 4/6/8 for Influenza PBMC and
16/24/32/40/48 for Parse. Training-cell counts and evaluation depths are
different quantities. Seeds are 17/29/43; deterministic baselines repeat
their score over this axis. Training regimes within a resource reuse the
same evaluation units.

| Design | Shared roles |
|---|---|
| S | Model input, prediction reference and observation reference |
| M | Prediction and observation references; model input separate |
| P | Model input and observation reference; prediction reference separate |
| O | Model input and prediction reference; observation reference separate |
| D | All three roles use distinct blocks |

`MODEL_SCORES.tsv` gives means and ranks; `SEED_MODEL_SCORES.tsv` and
`UNIT_MODEL_SCORES.tsv` expose the aggregation. `PAIR_MARGINS.tsv` names every
pair and its S-to-design status. `POLICY_SUMMARY.tsv` reports reversal counts,
ties and RMS changes in pair margins. Exact ties are retained, with midranks;
small nonzero values are not forced to zero. These are descriptive comparisons
of the supplied units, not significance tests or independent allocation replicates.

## Provenance and checks

`SOURCE_MANIFEST.json` records input hashes and source records for the four
evaluations. After recalculation, the replay compares the results with the
mean-utility and pair-margin tables in `expected/`, checking values, model
orders and reversal identities.

The replay starts after prediction scoring and allocation averaging.
To recreate its inputs from the matching manuscript Source Data distribution:

```bash
.venv/bin/python evidence/model_comparisons/prepare_inputs.py \
  --source-data-root /path/to/05_SOURCE_DATA \
  --output /path/to/new/model_comparison_inputs \
  --origin-release NAME_OF_SOURCE_RELEASE
```

The exporter accepts the source version pinned in its QA hash. Tests cover
score-before-averaging, equal-unit weights, exact ties and missing or altered
inputs.
