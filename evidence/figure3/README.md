# Figure 3: changing model inputs

This replay preserves the earlier 1,280-epoch Influenza CPA results. For the
current 2,560-epoch results, use the [current PBMC replay](../current_submission/README.md).
The other model fits and both Parse groups are unchanged.

From the repository root:

```bash
.venv/bin/python evidence/figure3/replay.py --output outputs/figure3
```

The replay calculates 576 plotted values from saved summaries for four training groups, scGen/CPA/CellOT, eight roots per group, three seeds and 1,000 allocations. It checks 24 pair comparisons at the displayed depths and 48 pair–depth comparisons for each input change. Mean model order is unchanged in all comparisons. Squared seed-level RMS values are averaged before taking the final RMS; utility changes average separately scored seeds and then roots equally.

`source/` contains 46 numerical files, checked against `SOURCE_MANIFEST.json`; `expected/` contains the values used to check the results.

Computation starts from per-seed/root summaries, after inference and cell-level scoring. The 15,552 coordinate-invariance checks belong to an earlier run and are retained as historical receipts. The replay applies the manuscript's three-family and display-depth selection; the source tables also retain the other recorded models and depths.

Ranges show variation across roots; paired training regimes reuse the same roots. These are descriptive comparisons without confidence intervals or hypothesis tests.

The `gse162632` key denotes Influenza PBMC; `parse` denotes Parse PBMC. Inputs contain pseudonymized root indices and derived summaries. Raw expression, private donor mappings, checkpoints and large prediction arrays are excluded. Parse-derived summaries retain their provider terms; see the repository's [data licence scope](../../LICENSE_SCOPE.md).
