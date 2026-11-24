#!/usr/bin/env python3
"""Genuine out-of-sample test of the reference-noise correction, using disjoint control cells.

The agreement of the fitted lines with the full 144-cell pool follows algebraically from sampling without
replacement and does not test extrapolation beyond the pool; this script does, with cells that were not used.

Data: the 768 hash-ranked PBS control cells per donor x cell type for the seven
Parse cell types in which every evaluation donor has at least 768 controls (56 strata); the released
treated means, direct-effect predictions, gene scales and weights. No state-output predictions exist for
these cells, so the test covers the five direct-effect families (the term being corrected is theirs).

Fit set A: pool cells 0-143 of each stratum.  Test set B: cells 144-767 (624 cells, disjoint from A).
From A alone, for every task t in stratum g:
    U_A(t)  = -|| e_t - (Y_t - mean(A_g)) ||_s^2        utility with all of A as the reference
    S2_A(g) = (1/143) sum_c || x_c - mean(A_g) ||_s^2   standardized dispersion of A
    predicted utility for an independent reference of m cells:  U_A + S2_A/144 - S2_A/m
Measured: the same score with the reference set to the mean of m cells of B, averaged over the disjoint
m-cell blocks that B provides.  Tasks are weighted equally within donor and donors equally.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_locations import evaluation_root, output_root, parse_control_pool, parse_tasks

POOL = parse_control_pool()
PUB = evaluation_root() / 'parse'
MANIFEST = parse_tasks()
OUT = output_root() / 'reference_noise'
DIRECT = ('NC', 'CM', 'TW', 'PCA', 'RBF')
NA = 144
DEPTHS = (16, 48, 144, 312, 624)


def main():
    pool = np.load(POOL / 'pool_expr_log1p.npy')                      # (56, 768, 2000) float32
    groups = json.load(open(POOL / 'groups.json'))
    gkey = {(g['donor'], g['cell_type']): g['group_index'] for g in groups}
    man = pd.read_csv(MANIFEST, sep='\t')
    rows, summary = [], {}
    for regime, label in (('common48', '48 cells'), ('common128', '<=128 cells')):
        cache = PUB / regime / 'score_cache_v1'
        treated = np.load(cache / 'treated.npy')
        root_index = np.load(cache / 'root_index.npy')
        task_index = np.load(cache / 'task_index.npy')
        scales = np.load(cache / 'scales.npy')
        weights = np.load(cache / 'weights.npy')
        canon = np.load(cache / 'canonical_controls.npy', mmap_mode='r')
        group_index = np.load(cache / 'group_index.npy')
        assert (man.task_index.values == task_index).all() and (man.analysis_root_index.values == root_index).all()
        std = np.sqrt(weights) / scales
        pg = np.array([gkey.get((d, c), -1) for d, c in zip(man.donor, man.cell_type)])
        keep = np.flatnonzero(pg >= 0)
        # consistency of the pool with the released canonical controls (same transform and gene panel):
        # squared standardized distance between stratum means, against its sampling expectation
        chk = []
        for g in np.unique(pg[keep]):
            t0 = keep[pg[keep] == g][0]
            a = np.asarray(pool[g], dtype=np.float64) * std
            c = np.asarray(canon[group_index[t0]], dtype=np.float64) * std
            s2 = ((a - a.mean(0)) ** 2).sum() / (len(a) - 1)
            chk.append((float(((a.mean(0) - c.mean(0)) ** 2).sum()), s2 * (1 / len(a) + 1 / len(c))))
        chk = np.array(chk)
        xs = np.asarray(pool, dtype=np.float64) * std                 # standardized cells
        mean_a = xs[:, :NA].mean(axis=1)
        s2_a = ((xs[:, :NA] - mean_a[:, None]) ** 2).sum(axis=(1, 2)) / (NA - 1)   # (56,)
        s2_b = ((xs[:, NA:] - xs[:, NA:].mean(axis=1)[:, None]) ** 2).sum(axis=(1, 2)) / (768 - NA - 1)
        y = treated[keep] * std
        g_of = pg[keep]
        r_of = root_index[keep]
        counts = np.bincount(r_of, minlength=8)

        def donor_mean(v):
            return float(np.mean([v[r_of == r].mean() for r in range(8)]))

        res = {}
        for fam in DIRECT:
            e = np.load(PUB / regime / 'banks' / f'{fam}__primary__deterministic' / 'effects.npy')[keep] * std
            base = y - e                                              # residual = ref - (Y - e)
            u_a = -((mean_a[g_of] - base) ** 2).sum(axis=1)
            for m in DEPTHS:
                nb = (768 - NA) // m
                meas = np.zeros(len(keep))
                for b in range(nb):
                    ref = xs[:, NA + b * m: NA + (b + 1) * m].mean(axis=1)
                    meas -= ((ref[g_of] - base) ** 2).sum(axis=1)
                meas /= nb
                pred = u_a + s2_a[g_of] / NA - s2_a[g_of] / m
                naive = u_a                                           # no correction: score with A as reference
                p, q, nv = donor_mean(pred), donor_mean(meas), donor_mean(naive)
                res[(fam, m)] = (p, q)
                rows.append(['Parse', label, fam, m, nb, f'{p:.6f}', f'{q:.6f}', f'{p - q:.6f}', f'{nv:.6f}'])
        err = np.array([p - q for p, q in res.values()])
        summary[label] = {
            'tasks': int(len(keep)), 'tasks_per_donor': counts.tolist(), 'strata': int(len(np.unique(g_of))),
            'max_abs_error': float(np.abs(err).max()), 'mean_error': float(err.mean()),
            'dispersion_A_mean': float(donor_mean(s2_a[g_of])), 'dispersion_B_mean': float(donor_mean(s2_b[g_of])),
            'pool_vs_canonical_mean_sq_distance': float(chk[:, 0].mean()), 'expected_if_same_population': float(chk[:, 1].mean()),
            'correction_at_48_cells': float(donor_mean(s2_a[g_of]) / 48)}
        print(f'\n=== Parse | training {label}: {len(keep)} tasks in {len(np.unique(g_of))} strata ===')
        print(f"pool vs released canonical controls, mean squared distance {chk[:, 0].mean():.5f} (expected {chk[:, 1].mean():.5f} for two samples of one population)")
        print(f"dispersion S2: fit cells {summary[label]['dispersion_A_mean']:.4f}, disjoint test cells {summary[label]['dispersion_B_mean']:.4f}")
        print(f"{'family':6s} " + ' '.join(f'{"m=%d pred/meas" % m:>22s}' for m in DEPTHS))
        for fam in DIRECT:
            print(f'{fam:6s} ' + ' '.join(f'{res[(fam, m)][0]:10.5f}/{res[(fam, m)][1]:9.5f} ' for m in DEPTHS))
        print(f"max |predicted - measured| = {np.abs(err).max():.2e}; size of the correction at 48 cells = {summary[label]['correction_at_48_cells']:.4f}")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'DISJOINT_CELL_TEST.tsv', 'w', newline='') as h:
        w = csv.writer(h, delimiter='\t')
        w.writerow(['dataset', 'training', 'family', 'reference_cells_m', 'disjoint_blocks_averaged', 'predicted_from_fit_cells',
                    'measured_with_disjoint_cells', 'error', 'uncorrected_score_with_144_fit_cells'])
        w.writerows(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'DISJOINT_CELL_TEST_SUMMARY.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
