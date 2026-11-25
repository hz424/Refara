# Reanalysis of published CRISPR perturbation benchmarks

`paired_shifts_and_reversals.py` computes the benchmark means, unique-perturbation paired shifts,
95% Student-*t* confidence intervals, two-sided paired *t*-test *P*-values and pairwise ranking reversals
reported in Fig. 3 and Supplementary Tables 21 and 27 of the current manuscript.

## Inputs

The input tables (`double_metrics_by_perturbation.tsv` and `single_metrics_by_perturbation.tsv`) are produced
from the predictions deposited by Ahlmann-Eltze et al. ([Zenodo record 16092690](https://zenodo.org/records/16092690))
and are also supplied in `Source_Data.zip` (`Supplementary_Table_21/`) with the manuscript.

Set two environment variables before running:

```bash
export REFARA_CRISPR_METRICS_DIR=/path/to/Supplementary_Table_21_dir
export REFARA_OUTPUT_ROOT=outputs/crispr_reanalysis
python analysis/published_crispr_benchmarks/paired_shifts_and_reversals.py
```

The script writes `published_crispr/AHLMANN_PAIRED_SHIFTS.tsv` and
`published_crispr/AHLMANN_RANKING_REVERSALS.tsv` under `REFARA_OUTPUT_ROOT`.
