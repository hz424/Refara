# Few independent acquisition units

This analysis extends the hierarchical Gaussian null in panel b to R=2, 3, 4 and 6. It uses the existing generator, all five procedures, 2,500 simulations per root count and 499 bootstrap draws. The original 20 rows at R=8, 12, 20 and 40 are unchanged. Panel c shows recovery at R=8, 12, 20 and 40. The analytic zeros below eight roots are derived in the manuscript Methods and retained in Source Data.

`source_data/Figure_5_Source_Data.xlsx` contains every plotted value, and the TSV files provide the detailed fields. `results/` contains event counts and failures. `LOW_N_FROZEN_SPEC.json` specifies the design, random streams and code fixed before these simulations.

Install `requirements.txt` in Python 3.10. To regenerate the extension in a new directory, run:

```bash
python run_low_n.py --root-count 2 --output-dir reproduced
python run_low_n.py --root-count 3 --output-dir reproduced
python run_low_n.py --root-count 4 --output-dir reproduced
python run_low_n.py --root-count 6 --output-dir reproduced
python aggregate_low_n.py --results-dir reproduced
python build_source_workbook.py
```

Run `validate_numerics.py` to compare Student-t tails with SciPy and `validate_extension.py` to check the unchanged original grid and the two-root bootstrap against its analytic behavior. The only change to the original numerical module is support for degrees of freedom 1, 2, 3 and 5. The original generator, sign-test implementation and numerical algorithm remain fixed.

All intervals in panels a–c describe Monte Carlo uncertainty. Low-root exact-sign zeros arise because no possible p-value passes the first of 12 Holm thresholds. The root-t comparison assumes Gaussian data. Resampling whole roots does not by itself calibrate the bootstrap with very few roots.
