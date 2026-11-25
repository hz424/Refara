#!/usr/bin/env python3
"""Paired shifts, confidence intervals and pairwise ranking reversals across four published CRISPR benchmarks.

Reads the per-perturbation metric tables produced from the deposited predictions of Ahlmann-Eltze et al.
(Zenodo record 16092690):
  - double_metrics_by_perturbation.tsv  (Norman double perturbations: 5 splits x 62 pairs, n = 310 rows, N = 122 unique)
  - single_metrics_by_perturbation.tsv  (Adamson n = 48 / N = 40; Replogle K562 n = 262 / N = 216; Replogle RPE1 n = 413 / N = 353)

Published means and pairwise ranking reversals are evaluated on the benchmark's perturbation-split tables.
Paired shifts, 95% Student-t confidence intervals and two-sided paired t-test P-values are computed across unique
held-out genetic perturbations (averaging repeated splits of the same perturbation first) so split reuse does not
inflate degrees of freedom. Log-space survival evaluation avoids IEEE-754 float64 underflow on extreme t statistics.

Environment variables:
  REFARA_CRISPR_METRICS_DIR  directory containing double_metrics_by_perturbation.tsv and single_metrics_by_perturbation.tsv
  REFARA_OUTPUT_ROOT         directory for outputs (writes published_crispr/AHLMANN_PAIRED_SHIFTS.tsv and AHLMANN_RANKING_REVERSALS.tsv)
"""
from __future__ import annotations

import csv
import itertools
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import betaln

PRIMARY_DOUBLE = ['additive_model', 'geneformer', 'gears', 'uce', 'uce33', 'scbert', 'scfoundation', 'scgpt', 'cpa']
PRIMARY_SINGLE = ['lpm_selftrained', 'lpm_scFoundationGeneEmb', 'lpm_scgptGeneEmb', 'lpm_gearsPertEmb',
                  'geneformer', 'gears', 'uce', 'uce33', 'scbert', 'scgpt', 'cpa']


def _env_dir(name: str, desc: str, create: bool = False) -> Path:
    val = os.environ.get(name)
    if not val:
        raise SystemExit(f"Set {name} to {desc}. See analysis/published_crispr_benchmarks/README.md.")
    p = Path(val)
    if create:
        p.mkdir(parents=True, exist_ok=True)
    elif not p.exists():
        raise SystemExit(f"{name}={val} does not exist.")
    return p


def fmt_p_exact(t_stat: float, df: int) -> str:
    if not np.isfinite(t_stat) or df < 1:
        return 'nan'
    if t_stat == 0.0:
        return '1.00e+00'
    log_sf = float(stats.t.logsf(abs(t_stat), df))
    if np.isfinite(log_sf):
        log_p = log_sf + math.log(2.0)
    else:
        x = df / (df + t_stat * t_stat)
        a = 0.5 * df
        corr = 1.0 + (a / (2.0 * (a + 1.0))) * x + (3.0 * a / (8.0 * (a + 2.0))) * x * x
        log_p = a * math.log(x) - math.log(a) - float(betaln(a, 0.5)) + math.log(corr)
    log10_p = log_p / math.log(10.0)
    if log10_p >= 0.0:
        return '1.00e+00'
    exp10 = math.floor(log10_p)
    mant = 10.0 ** (log10_p - exp10)
    if mant >= 9.995:
        mant /= 10.0
        exp10 += 1
    return f'{mant:.2f}e{exp10:+03d}'


def paired_ci_p_unique(s_a: pd.Series, s_b: pd.Series):
    diff = (s_a - s_b).dropna().values.astype(np.float64)
    n = len(diff)
    if n < 2:
        return np.nan, np.nan, np.nan, 'nan', n
    m = float(diff.mean())
    sd = float(diff.std(ddof=1))
    if sd == 0.0:
        return m, m, m, ('1.00e+00' if m == 0.0 else '1.00e-300'), n
    se = sd / math.sqrt(n)
    tc = float(stats.t.ppf(0.975, n - 1))
    return m, m - tc * se, m + tc * se, fmt_p_exact(m / se, n - 1), n


def main() -> None:
    in_dir = _env_dir('REFARA_CRISPR_METRICS_DIR', 'the directory holding double_metrics_by_perturbation.tsv and single_metrics_by_perturbation.tsv')
    out_dir = _env_dir('REFARA_OUTPUT_ROOT', 'a directory for outputs', create=True) / 'published_crispr'
    out_dir.mkdir(parents=True, exist_ok=True)

    shift_rows, rev_rows = [], []
    for mode, methods in (('double', PRIMARY_DOUBLE), ('single', PRIMARY_SINGLE)):
        df = pd.read_csv(in_dir / f'{mode}_metrics_by_perturbation.tsv', sep='\t')
        df = df[df['train'].isin(['test', 'val']) & (df['perturbation'] != 'ctrl')].copy()
        for ds, g_ds in df.groupby('dataset_name'):
            avail = g_ds[g_ds['method'].isin(methods)]
            if mode == 'single':
                counts = avail.groupby(['seed', 'perturbation'])['method'].nunique()
                common_keys = set(counts[counts == len(methods)].index)
                avail = avail[avail.set_index(['seed', 'perturbation']).index.isin(common_keys)].copy()

            piv_p_meas = avail.pivot_table(index=['seed', 'perturbation'], columns='method', values='pearson_delta_measured')
            piv_p_own = avail.pivot_table(index=['seed', 'perturbation'], columns='method', values='pearson_delta_own')
            piv_l_meas = avail.pivot_table(index=['seed', 'perturbation'], columns='method', values='l2_measured')
            piv_l_own = avail.pivot_table(index=['seed', 'perturbation'], columns='method', values='l2_own')

            up_p_meas = avail.pivot_table(index='perturbation', columns='method', values='pearson_delta_measured', aggfunc='mean')
            up_p_own = avail.pivot_table(index='perturbation', columns='method', values='pearson_delta_own', aggfunc='mean')
            up_l_meas = avail.pivot_table(index='perturbation', columns='method', values='l2_measured', aggfunc='mean')
            up_l_own = avail.pivot_table(index='perturbation', columns='method', values='l2_own', aggfunc='mean')

            n_split_rows = len(piv_l_meas)
            n_uniq_total = len(up_l_meas)

            for m in methods:
                if m not in piv_p_meas.columns:
                    continue
                mp, plo, phi, pp_str, n_up_p = paired_ci_p_unique(up_p_own[m], up_p_meas[m])
                ml, llo, lhi, lp_str, _ = paired_ci_p_unique(up_l_own[m], up_l_meas[m])
                n_label = f'{n_split_rows} ({n_uniq_total})' if n_up_p == n_uniq_total else f'{n_split_rows} ({n_uniq_total}; {n_up_p} non-const)'
                shift_rows.append([
                    ds, m, n_split_rows, n_uniq_total, n_up_p, n_label,
                    f'{piv_p_meas[m].mean():.4f}', f'{piv_p_own[m].mean():.4f}',
                    f'{mp:+.4f}', f'{plo:+.4f}', f'{phi:+.4f}', pp_str,
                    f'{piv_l_meas[m].mean():.4f}', f'{piv_l_own[m].mean():.4f}',
                    f'{ml:+.4f}', f'{llo:+.4f}', f'{lhi:+.4f}', lp_str,
                ])

            present = [m for m in methods if m in piv_p_meas.columns]
            for a, b in itertools.combinations(present, 2):
                pm_a, pm_b = piv_p_meas[a].mean(), piv_p_meas[b].mean()
                po_a, po_b = piv_p_own[a].mean(), piv_p_own[b].mean()
                rev_p = (pm_a - pm_b) * (po_a - po_b) < 0
                lm_a, lm_b = piv_l_meas[a].mean(), piv_l_meas[b].mean()
                lo_a, lo_b = piv_l_own[a].mean(), piv_l_own[b].mean()
                rev_l = (lm_a - lm_b) * (lo_a - lo_b) < 0
                _, _, _, m_p_str, _ = paired_ci_p_unique(up_p_meas[a], up_p_meas[b])
                _, _, _, o_p_str, _ = paired_ci_p_unique(up_p_own[a], up_p_own[b])
                if rev_p or rev_l:
                    rev_rows.append([
                        ds, f'{a} vs {b}', int(rev_p), int(rev_l),
                        f'{pm_a:.4f} vs {pm_b:.4f}', f'{po_a:.4f} vs {po_b:.4f}',
                        m_p_str, o_p_str,
                        f'{lm_a:.4f} vs {lm_b:.4f}', f'{lo_a:.4f} vs {lo_b:.4f}',
                    ])

    with open(out_dir / 'AHLMANN_PAIRED_SHIFTS.tsv', 'w', newline='') as h:
        w = csv.writer(h, delimiter='\t')
        w.writerow(['dataset', 'method', 'n_eval_rows', 'n_unique_perts', 'n_unique_pearson', 'units_label',
                    'pearson_measured', 'pearson_own', 'pearson_shift_unique', 'pearson_ci95_low', 'pearson_ci95_high', 'pearson_p_ttest',
                    'l2_measured', 'l2_own', 'l2_shift_unique', 'l2_ci95_low', 'l2_ci95_high', 'l2_p_ttest'])
        w.writerows(shift_rows)

    with open(out_dir / 'AHLMANN_RANKING_REVERSALS.tsv', 'w', newline='') as h:
        w = csv.writer(h, delimiter='\t')
        w.writerow(['dataset', 'pair', 'reversed_pearson_delta', 'reversed_l2',
                    'pearson_measured_pair', 'pearson_own_pair', 'pearson_diff_measured_p', 'pearson_diff_own_p',
                    'l2_measured_pair', 'l2_own_pair'])
        w.writerows(rev_rows)


if __name__ == '__main__':
    main()
