# Figure 3 metric summaries

The accompanying Source Data include the original Figure 3 tables and additional individual-model Pearson and effect-cosine summaries. To check them, run:

```bash
python analysis/metric_summaries/validate.py --source-data-root /path/to/Source_Data
```

The check uses only the Python standard library and verifies summary hashes, values, missing-score definitions and equal-root aggregation. It reads the delivered summaries; the original scalar score arrays are not bundled. See `Source_Data/Figure_3_metric_summaries_v1818/README.md` for column definitions and reconstruction details.
