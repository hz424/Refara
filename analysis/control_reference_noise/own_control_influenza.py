#!/usr/bin/env python3
"""Influenza PBMC (GSE162632): raw-control versus model's-own-control prediction reference.

Small enough to score directly (40 tasks, 24 canonical control cells, 1,000 released allocations, depths 4/6/8 and
the full 24-cell pool).  Before any new quantity is computed the scorer is checked against the released per-seed
utilities for designs S, O and D.
"""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np

from data_locations import evaluation_root, output_root

EVAL = evaluation_root() / 'gse162632'
OWN = output_root() / 'own_control'
FAMILIES = ('NC', 'CM', 'TW', 'PCA', 'RBF', 'scGen', 'CPA', 'CellOT')
DIRECT, STATE_OUTPUT, SEEDS = FAMILIES[:5], FAMILIES[5:], (17, 29, 43)
DEPTHS = (4, 6, 8)
PERM = list(itertools.permutations(range(3), 3))
PAIR = [(i, j) for i in range(3) for j in range(3) if i != j]


def sq(x):
    return (x * x).sum(axis=-1)


def block_means(arr, ranks):
    """arr (40, 24, G); ranks (A, 3, 40, n) -> (A, 3, 40, G)."""
    a, b, g, n = ranks.shape
    w = np.zeros((a, b, g, arr.shape[1]))
    ia, ib, ig = np.meshgrid(np.arange(a), np.arange(b), np.arange(g), indexing='ij')
    for k in range(n):                                     # ranks within a block are distinct cells
        w[ia, ib, ig, ranks[..., k]] += 1.0 / n
    return np.einsum('abgc,gcf->abgf', w, arr, optimize=True)


def fit(x, y):
    x = np.asarray(x, float)
    b = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
    return y.mean() - b * x.mean(), b


def main():
    rows_u, rows_p, summary = [], [], {}
    for regime, label in (('original8', '8 cells'), ('cap48', '<=48 cells')):
        base = EVAL / regime
        c = base / 'score_cache_v1'
        std = np.sqrt(np.load(c / 'weights.npy')) / np.load(c / 'scales.npy')
        y = np.load(c / 'treated.npy') * std                                  # (40, G)
        ctl = np.load(c / 'canonical_controls.npy').astype(np.float64) * std   # (40, 24, G)
        root = np.load(c / 'root_index.npy')
        alloc = np.load(c / 'max_depth_allocations.npy').astype(np.int64)      # (1000, 3, 40, 8)
        donors = lambda v: np.array([v[..., root == r].mean(axis=-1) for r in range(8)])   # -> (8, ...)
        pub = np.load(base / 'PANEL_SEED_UTILITIES.npy', mmap_mode='r')        # (panel, seed, label, depth, pattern, root, family)
        res = {}            # (family, kind, design, depth) -> per-donor mean utility
        worst = 0.0
        fits = {f: [np.load(base / 'banks' / f'{f}__primary__deterministic' / 'effects.npy') * std] for f in DIRECT}
        for f in STATE_OUTPUT:
            fits[f] = [(np.load(base / 'banks' / f'{f}__primary__{s}' / 'points.npy').astype(np.float64) * std,
                        np.load(OWN / 'control_points' / 'gse162632' / f'{regime}__{f}__primary__{s}.npy').astype(np.float64) * std) for s in SEEDS]
        for di, n in enumerate(DEPTHS):
            acc = {}
            for lo in range(0, 1000, 100):
                ranks = alloc[lo:lo + 100, :, :, :n]
                cb = block_means(ctl, ranks)                                   # (A, 3, 40, G)
                obs = y[None, None] - cb                                       # observed effect per block
                for f in DIRECT:
                    e = fits[f][0]
                    u = -sq(e[None, None] - obs)                               # (A, 3, 40)
                    acc.setdefault((f, 'direct', 'S', 0), []).append(u.mean(axis=1))
                for f in STATE_OUTPUT:
                    for si, (pts, qp) in enumerate(fits[f]):
                        pb, qb = block_means(pts, ranks), block_means(qp, ranks)
                        raw = pb - cb                                          # effect with raw reference of the same block
                        own = pb - qb
                        s_raw = -sq(raw - obs).mean(axis=1)
                        o_raw = np.mean([-sq(raw[:, j] - obs[:, i]) for i, j in PAIR], axis=0)
                        d_raw = np.mean([-sq(pb[:, m] - cb[:, p] - obs[:, o]) for o, p, m in PERM], axis=0)
                        s_own = -sq(own - obs).mean(axis=1)
                        o_own = np.mean([-sq(own[:, j] - obs[:, i]) for i, j in PAIR], axis=0)
                        for name, val in (('S', s_raw), ('O', o_raw), ('D', d_raw)):
                            acc.setdefault((f, 'raw', name, si), []).append(val)
                        for name, val in (('S', s_own), ('O', o_own)):
                            acc.setdefault((f, 'own', name, si), []).append(val)
            for (f, kind, design, si), chunks in acc.items():
                per_label = donors(np.concatenate(chunks, axis=0))             # (8, 1000)
                if kind in ('direct', 'raw'):
                    pi = {'S': 0, 'O': 3, 'D': 4}[design]
                    ref = np.asarray(pub[0, si, :, di, pi, :, FAMILIES.index(f)]).T      # (8, 1000)
                    worst = max(worst, float(np.abs(per_label - ref).max()))
                res.setdefault((f, kind, design, n), []).append(per_label.mean(axis=1))
        # full 24-cell pool as one block
        cb = ctl.mean(axis=1)
        for f in DIRECT:
            res.setdefault((f, 'direct', 'S', 24), []).append(donors(-sq(fits[f][0] - (y - cb))))
        for f in STATE_OUTPUT:
            for pts, qp in fits[f]:
                pb, qb = pts.mean(axis=1), qp.mean(axis=1)
                res.setdefault((f, 'raw', 'S', 24), []).append(donors(-sq(pb - y)))
                res.setdefault((f, 'own', 'S', 24), []).append(donors(-sq(pb - qb - (y - cb))))
        res = {k: np.mean(v, axis=0) for k, v in res.items()}                  # seed average -> (8,)
        v = {n: np.mean([res[(f, 'raw', 'S', n)] - res[(f, 'raw', 'D', n)] for f in STATE_OUTPUT], axis=0) for n in DEPTHS}
        cc = np.mean([v[n] * n for n in DEPTHS], axis=0)
        v[24] = cc / 24
        allv = [float(v[n].mean()) for n in (*DEPTHS, 24)]
        inter, slope = {}, {}
        keys = [(f, 'direct') for f in DIRECT] + [(f, k) for f in STATE_OUTPUT for k in ('raw', 'own')]
        for key in keys:
            us = {n: res[(*key, 'S', n)] for n in (*DEPTHS, 24)}
            a_m, b_m = fit(allv, np.array([us[n].mean() for n in (*DEPTHS, 24)]))
            a_d = np.array([fit([v[n][r] for n in (*DEPTHS, 24)], np.array([us[n][r] for n in (*DEPTHS, 24)]))[0] for r in range(8)])
            inter[key], slope[key] = (a_m, a_d), b_m
        print(f'\n=== Influenza PBMC | training {label} ===  (check against released S/O/D utilities per allocation, seed and donor: {worst:.1e}; V at 4/6/8/24 = {np.round(allv, 4)})')
        print(f"{'model':8s} {'reference':8s} {'S n8':>9s} {'O n8':>9s} {'S 24':>9s} {'slope':>7s} {'corrected':>10s}")
        for key in keys:
            f, kind = key
            o8 = res[(f, 'direct', 'S', 8)].mean() if kind == 'direct' else res[(f, kind, 'O', 8)].mean()
            print(f"{f:8s} {kind:8s} {res[(*key, 'S', 8)].mean():9.4f} {o8:9.4f} {res[(*key, 'S', 24)].mean():9.4f} {slope[key]:7.3f} {inter[key][0]:10.4f}")
            rows_u.append(['Influenza PBMC', label, f, kind, f"{res[(*key, 'S', 4)].mean():.6f}", f"{res[(*key, 'S', 8)].mean():.6f}", f'{o8:.6f}',
                           f"{res[(*key, 'S', 24)].mean():.6f}", f'{slope[key]:.4f}', f'{inter[key][0]:.6f}'])
        for f in STATE_OUTPUT:
            summary[f'{label}|{f}'] = {'O_minus_S_own_n8': float((res[(f, 'own', 'O', 8)] - res[(f, 'own', 'S', 8)]).mean()),
                                       'O_minus_S_raw_n8': float((res[(f, 'raw', 'O', 8)] - res[(f, 'raw', 'S', 8)]).mean()),
                                       'corrected_own': float(inter[(f, 'own')][0]), 'corrected_raw': float(inter[(f, 'raw')][0])}
            print(f"{f}: O-S at 8 cells, own control {summary[f'{label}|{f}']['O_minus_S_own_n8']:+.5f}; raw control {summary[f'{label}|{f}']['O_minus_S_raw_n8']:+.5f}")
        print('direct-minus-state margins:   S n8 raw | S n8 own | corrected raw (donors>0) | corrected own (donors>0)')
        for d in DIRECT:
            for f in STATE_OUTPUT:
                m_raw = float((res[(d, 'direct', 'S', 8)] - res[(f, 'raw', 'S', 8)]).mean())
                m_own = float((res[(d, 'direct', 'S', 8)] - res[(f, 'own', 'S', 8)]).mean())
                c_raw, c_own = inter[(d, 'direct')][0] - inter[(f, 'raw')][0], inter[(d, 'direct')][0] - inter[(f, 'own')][0]
                dr, do = inter[(d, 'direct')][1] - inter[(f, 'raw')][1], inter[(d, 'direct')][1] - inter[(f, 'own')][1]
                print(f'  {d:4s}-{f:7s} {m_raw:+.4f} | {m_own:+.4f} | {c_raw:+.4f} ({int((dr > 0).sum())}) | {c_own:+.4f} ({int((do > 0).sum())})')
                rows_p.append(['Influenza PBMC', label, f'{d}-{f}', f'{m_raw:.6f}', f'{m_own:.6f}', f'{c_raw:.6f}', int((dr > 0).sum()), f'{c_own:.6f}',
                               int((do > 0).sum()), f'{do.min():.6f}', f'{do.max():.6f}'])
        summary[f'{label}|scorer_check'] = worst
    with open(OWN / 'OWN_CONTROL_UTILITIES_GSE162632.tsv', 'w', newline='') as h:
        w = csv.writer(h, delimiter='\t')
        w.writerow(['dataset', 'training', 'family', 'prediction_reference', 'U_S_4', 'U_S_8', 'U_O_8', 'U_S_24', 'slope_dU_S_dV', 'corrected_intercept'])
        w.writerows(rows_u)
    with open(OWN / 'OWN_CONTROL_PAIRS_GSE162632.tsv', 'w', newline='') as h:
        w = csv.writer(h, delimiter='\t')
        w.writerow(['dataset', 'training', 'pair_direct_minus_state', 'd_S_8_raw_reference', 'd_S_8_own_control', 'corrected_raw_reference', 'donors_positive_raw',
                    'corrected_own_control', 'donors_positive_own', 'corrected_own_donor_min', 'corrected_own_donor_max'])
        w.writerows(rows_p)
    (OWN / 'OWN_CONTROL_SUMMARY_GSE162632.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
