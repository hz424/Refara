#!/usr/bin/env python3
"""Validate added low-df tails and unchanged old-df behavior before simulation."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import stats
from run_low_n import module

HERE = Path(__file__).resolve().parent


def main():
    old = module('old_comparator_numerics', HERE / 'frozen_original/comparator_numerics_v1.py')
    new = module('new_comparator_numerics', HERE / 'comparator_numerics_low_n.py')
    grid = np.r_[np.linspace(-20, 20, 801), np.logspace(-8, 6, 401), -np.logspace(-8, 6, 401)]
    checks = []
    for df in sorted(new.SUPPORTED_T_DF):
        got = np.array([new.student_t_survival(float(t), df) for t in grid])
        expected = stats.t.sf(grid, df)
        # Near zero, forming df/(df+t²) rounds away t², with at most O(3e-8)
        # absolute tail error. Near rejection thresholds use a tighter bound.
        assert np.allclose(got, expected, rtol=2e-12, atol=3e-8)
        threshold_t = stats.t.isf(np.array([.05 / h for h in range(1, 13)]), df)
        threshold_error = max(abs(new.student_t_survival(float(t), df) - stats.t.sf(t, df)) for t in threshold_t)
        assert threshold_error < 5e-14
        assert new.student_t_survival(0., df) == .5
        assert new.student_t_survival(float('inf'), df) == 0.
        assert new.student_t_survival(float('-inf'), df) == 1.
        if df in old.SUPPORTED_T_DF:
            assert np.array_equal(got, [old.student_t_survival(float(t), df) for t in grid])
        # Independent analytic Cauchy survival at df=1.
        if df == 1:
            stable_cauchy = np.where(grid > 0, np.arctan(1 / np.where(grid == 0, 1, grid)) / np.pi, .5 - np.arctan(grid) / np.pi)
            assert np.allclose(got, stable_cauchy, rtol=2e-12, atol=3e-8)
        result = new.one_sample_root_t_positive(list(np.linspace(-.4, 1.7, df + 1)))
        ref = stats.ttest_1samp(np.linspace(-.4, 1.7, df + 1), 0., alternative='greater')
        assert abs(result.one_sided_p_value - ref.pvalue) < 2e-14
        checks.append(dict(df=df, grid_points=len(grid), maximum_absolute_error=float(np.max(abs(got - expected))), maximum_threshold_error=float(threshold_error), old_bitwise_preserved=df in old.SUPPORTED_T_DF))
    receipt = dict(status='PASS', reference='scipy.stats.t.sf and scipy.stats.ttest_1samp; analytic Cauchy df=1', degrees_of_freedom=checks)
    (HERE / 'qa/NUMERICAL_VALIDATION.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
