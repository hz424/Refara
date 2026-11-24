#!/usr/bin/env python3
"""Reference-sampling noise in shared-reference scores: decomposition and depth extrapolation.

Uses only released per-seed utilities (no refit, no rescoring):
  PANEL_SEED_UTILITIES.npy  (panel, seed, allocation label, depth, pattern, donor, family); panel 0 = primary.

Claim tested. For a fixed direct-effect prediction, the shared-reference (S) residual
  e - (Y - b_i) contains the sampling error of the control-block mean b_i, so
  E[U_S] = U_inf - V/2, with V = E||b_i - b_j||_s^2 = 2 E||b_i - mu||_s^2.
For a state-output prediction, the shared block cancels: U_S = -||M(b_i) - Y||_s^2, which depends on
depth only through the model's own response to its input block, E[U_S] = U_inf - kappa V.
Under fully disjoint references (D), U_D = U_S - V for state outputs (manuscript identity) and
U_D = U_S for direct effects.  Hence, with d = U_direct - U_state,
  d_inf = d_S + (1/2 - kappa) V = d_D - (1/2 + kappa) V.

V is read off the published scores as U_S - U_D of the state-output families at each depth.
U_inf is the intercept of U_S regressed on V across the published depths (per donor and for the
equal-donor mean).  The Parse full-pool score (144 control cells per role, one block) from cross_individual_score.py is a
numerical check: with blocks drawn without replacement the fitted line passes through it in expectation.
"""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np

from data_locations import evaluation_root, output_root

PUB = evaluation_root()
OUT = output_root() / 'reference_noise'
XD = output_root() / 'cross_individual' / 'XDONOR_UTILITIES.tsv'
FAMILIES = ('NC', 'CM', 'TW', 'PCA', 'RBF', 'scGen', 'CPA', 'CellOT')
DIRECT, STATE_OUTPUT = FAMILIES[:5], FAMILIES[5:]
PATTERNS = ('S', 'M', 'P', 'O', 'D')
SETTINGS = [('Influenza PBMC', 'gse162632', 'original8', '8 cells', (4, 6, 8)),
            ('Influenza PBMC', 'gse162632', 'cap48', '<=48 cells', (4, 6, 8)),
            ('Parse', 'parse', 'common48', '48 cells', (16, 24, 32, 40, 48)),
            ('Parse', 'parse', 'common128', '<=128 cells', (16, 24, 32, 40, 48))]


def fit_line(x, y):
    """Least squares y = a + b x along axis 0 of y; returns a, b, max |residual|."""
    x = np.asarray(x, dtype=float)
    xm = x.mean()
    b = ((x - xm).reshape(-1, *[1] * (y.ndim - 1)) * (y - y.mean(axis=0))).sum(axis=0) / ((x - xm) ** 2).sum()
    a = y.mean(axis=0) - b * xm
    res = y - (a + b * x.reshape(-1, *[1] * (y.ndim - 1)))
    return a, b, np.abs(res).max(axis=0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    rows_fam, rows_pair, rows_depth = [], [], []
    xd = {}
    if XD.exists():          # optional: produced by cross_individual_aggregate.py
        with open(XD) as h:
            for r in csv.DictReader(h, delimiter='\t'):
                xd[(r['regime'], r['design'], r['family'])] = float(r['mean_utility'])
    for dataset, key, regime, train_label, depths in SETTINGS:
        arr = np.load(PUB / key / regime / 'PANEL_SEED_UTILITIES.npy', mmap_mode='r')
        u = np.asarray(arr[0]).mean(axis=(0, 1))                 # (depth, pattern, donor, family)
        nd = len(depths)
        assert u.shape == (nd, 5, 8, 8), u.shape
        s, d = u[:, 0], u[:, 4]                                   # (depth, donor, family)
        v_state = s[:, :, 5:] - d[:, :, 5:]                       # (depth, donor, 3)
        v_spread = float(np.abs(v_state - v_state[:, :, :1]).max())
        direct_pattern_spread = float(np.abs(u[:, :, :, :5] - u[:, :1, :, :5]).max())
        v = v_state.mean(axis=2)                                  # (depth, donor)
        v_mean = v.mean(axis=1)                                   # (depth,)
        c_scale = float(np.mean(v_mean * np.asarray(depths)))     # V ~ c / n
        # equal-donor mean fit and per-donor fits
        a_m, b_m, r_m = fit_line(v_mean, s.mean(axis=1))          # (family,)
        a_r = np.zeros((8, 8)); b_r = np.zeros((8, 8))
        for r in range(8):
            a_r[r], b_r[r], _ = fit_line(v[:, r], s[:, r])
        tag = f'{dataset} | training {train_label}'
        nmax = depths[-1]
        s_max, d_max, o_max = s[-1].mean(axis=0), d[-1].mean(axis=0), u[-1, 3].mean(axis=0)
        rank = lambda x: [FAMILIES[i] for i in np.argsort(-x)]
        summary[tag] = {
            'depths': list(depths), 'V_by_depth': v_mean.tolist(), 'V_times_n': (v_mean * np.asarray(depths)).tolist(),
            'state_family_V_max_spread': v_spread, 'direct_utility_max_spread_across_patterns': direct_pattern_spread,
            'slope_dU_S_dV': dict(zip(FAMILIES, b_m.tolist())), 'intercept_U_inf': dict(zip(FAMILIES, a_m.tolist())),
            'max_abs_fit_residual': dict(zip(FAMILIES, r_m.tolist())),
            'ranking_S_at_max_depth': rank(s_max), 'ranking_O_at_max_depth': rank(o_max),
            'ranking_D_at_max_depth': rank(d_max), 'ranking_reference_noise_free': rank(a_m)}
        for fi, f in enumerate(FAMILIES):
            rows_fam.append([dataset, train_label, f, 'direct' if f in DIRECT else 'state', nmax,
                             f'{s_max[fi]:.6f}', f'{o_max[fi]:.6f}', f'{d_max[fi]:.6f}', f'{b_m[fi]:.4f}', f'{a_m[fi]:.6f}',
                             f'{r_m[fi]:.2e}', f'{a_r[:, fi].min():.6f}', f'{a_r[:, fi].max():.6f}'])
            for di, n in enumerate(depths):
                rows_depth.append([dataset, train_label, f, n, f'{v_mean[di]:.6f}', f'{s[di].mean(axis=0)[fi]:.6f}',
                                   f'{u[di, 3].mean(axis=0)[fi]:.6f}', f'{d[di].mean(axis=0)[fi]:.6f}'])
        n_state_lead_s = n_rev_inf = n_rev_d = 0
        for f, g in itertools.product(DIRECT, STATE_OUTPUT):
            i, j = FAMILIES.index(f), FAMILIES.index(g)
            ds, dd, dinf = s_max[i] - s_max[j], d_max[i] - d_max[j], a_m[i] - a_m[j]
            slope = b_m[i] - b_m[j]
            n_star = (-slope * c_scale / dinf) if (dinf > 0 and slope < 0) else float('nan')
            donors_pos = int(np.sum(a_r[:, i] - a_r[:, j] > 0))
            dr = a_r[:, i] - a_r[:, j]
            n_state_lead_s += ds < 0
            n_rev_inf += (ds < 0 < dinf)
            n_rev_d += (ds < 0 < dd)
            rows_pair.append([dataset, train_label, f'{f}-{g}', nmax, f'{ds:.6f}', f'{dd:.6f}', f'{(ds + dd) / 2:.6f}',
                              f'{dinf:.6f}', donors_pos, '' if np.isnan(n_star) else f'{n_star:.0f}', f'{dr.min():.6f}', f'{dr.max():.6f}'])
        summary[tag].update({'direct_state_pairs_state_leads_under_S': int(n_state_lead_s),
                             'of_which_direct_leads_under_D': int(n_rev_d),
                             'of_which_direct_leads_reference_noise_free': int(n_rev_inf)})
        if key == 'parse' and xd:   # full-pool check at 144 cells per role (not used in the fit)
            v144 = c_scale / 144.0
            chk = {}
            for fi, f in enumerate(FAMILIES):
                pred = a_m[fi] + b_m[fi] * v144
                meas = xd[(regime, 'S_full', f)]
                chk[f] = {'predicted': float(pred), 'measured': meas, 'error': float(pred - meas)}
            summary[tag]['out_of_sample_S_at_144_cells'] = chk
            summary[tag]['out_of_sample_max_abs_error'] = float(max(abs(x['error']) for x in chk.values()))
        print(f'\n=== {tag} ===')
        print(f"V by depth {dict(zip(depths, np.round(v_mean, 5)))}; n*V = {np.round(v_mean * np.asarray(depths), 3)}")
        print(f'state-family spread of (U_S-U_D): {v_spread:.1e}; direct spread across patterns: {direct_pattern_spread:.1e}')
        print(f"{'family':7s} {'U_S(n=%d)' % nmax:>11s} {'U_D':>9s} {'slope':>8s} {'U_inf':>9s} {'fit res':>8s}")
        for fi, f in enumerate(FAMILIES):
            print(f'{f:7s} {s_max[fi]:11.4f} {d_max[fi]:9.4f} {b_m[fi]:8.3f} {a_m[fi]:9.4f} {r_m[fi]:8.1e}')
        print('rank S   :', ' > '.join(rank(s_max)))
        print('rank D   :', ' > '.join(rank(d_max)))
        print('rank inf :', ' > '.join(rank(a_m)))
        print(f'direct-state pairs: state leads under S {n_state_lead_s}/15; of these direct leads under D {n_rev_d}, noise-free {n_rev_inf}')
        if key == 'parse' and xd:
            print('full-pool S at 144 cells (pred vs measured): ' +
                  ', '.join(f"{f} {c['predicted']:.4f}/{c['measured']:.4f}" for f, c in chk.items()))
    with open(OUT / 'REFERENCE_NOISE_FAMILIES.tsv', 'w', newline='') as h:
        w = csv.writer(h, delimiter='\t')
        w.writerow(['dataset', 'training', 'family', 'class', 'max_depth', 'U_S', 'U_O', 'U_D', 'slope_dU_S_dV', 'U_inf_intercept',
                    'max_abs_fit_residual', 'U_inf_min_over_donors', 'U_inf_max_over_donors'])
        w.writerows(rows_fam)
    with open(OUT / 'REFERENCE_NOISE_PAIRS.tsv', 'w', newline='') as h:
        w = csv.writer(h, delimiter='\t')
        w.writerow(['dataset', 'training', 'pair_direct_minus_state', 'max_depth', 'd_S', 'd_D', 'midpoint', 'd_inf',
                    'donors_with_d_inf_positive_of_8', 'shared_reference_crossing_depth', 'd_inf_donor_min', 'd_inf_donor_max'])
        w.writerows(rows_pair)
    with open(OUT / 'REFERENCE_NOISE_BY_DEPTH.tsv', 'w', newline='') as h:
        w = csv.writer(h, delimiter='\t')
        w.writerow(['dataset', 'training', 'family', 'depth', 'V', 'U_S', 'U_O', 'U_D'])
        w.writerows(rows_depth)
    (OUT / 'REFERENCE_NOISE_SUMMARY.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
