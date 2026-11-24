#!/usr/bin/env python3
"""Compare the raw-control and model's-own-control prediction references for scGen, CPA and CellOT (Parse)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from data_locations import output_root

OWN = output_root() / 'own_control'
XD = output_root() / 'cross_individual' / 'per_fit'
DIRECT, STATE_OUTPUT, SEEDS = ('NC', 'CM', 'TW', 'PCA', 'RBF'), ('scGen', 'CPA', 'CellOT'), (17, 29, 43)
DEPTH = {'n16': 16, 'n48': 48, 'full': 144}


def load(path):
    z = np.load(path)
    return {str(n): u.mean(axis=0) for n, u in zip(z['names'], z['utility'])}        # design -> (donor,)


def fit(x, y):
    x = np.asarray(x, float)
    b = ((x - x.mean())[:, None] * (y - y.mean(axis=0))).sum(axis=0) / ((x - x.mean()) ** 2).sum()
    return y.mean(axis=0) - b * x.mean(), b


def main():
    families = [f for f in STATE_OUTPUT if all((OWN / 'per_fit' / f'{r}__{f}__primary__{s}.npz').exists() for r in ('common48', 'common128') for s in SEEDS)]
    print('state families available:', families)
    rows_u, rows_p, summary = [], [], {}
    for regime, label in (('common48', '48 cells'), ('common128', '<=128 cells')):
        direct = {f: load(XD / f'{regime}__{f}__primary__deterministic.npz') for f in DIRECT}
        xstate = {f: [load(XD / f'{regime}__{f}__primary__{s}.npz') for s in SEEDS] for f in families}
        own = {f: [load(OWN / 'per_fit' / f'{regime}__{f}__primary__{s}.npz') for s in SEEDS] for f in families}
        avg = lambda lst, k: np.mean([d[k] for d in lst], axis=0)
        # reference separation per donor from the released identity U_S - U_D = V (state families, raw reference)
        v = {t: np.mean([avg(xstate[f], f'S_{t}') - avg(xstate[f], f'D_{t}') for f in families], axis=0) for t in ('n16', 'n48')}
        c = np.mean([v['n16'] * 16, v['n48'] * 48], axis=0)
        v['full'] = c / 144
        vmean = [float(v[t].mean()) for t in DEPTH]
        # consistency of the two scorers on the raw reference
        chk = max(float(np.abs(avg(own[f], f'S_raw_{t}') - avg(xstate[f], f'S_{t}')).max()) for f in families for t in DEPTH)
        util = {}
        for f in DIRECT:
            util[(f, 'direct')] = {t: direct[f][f'S_{t}'] for t in DEPTH}
        for f in families:
            util[(f, 'raw')] = {t: avg(own[f], f'S_raw_{t}') for t in DEPTH}
            util[(f, 'own')] = {t: avg(own[f], f'S_own_{t}') for t in DEPTH}
        inter, slope = {}, {}
        for key, u in util.items():
            a_d, b_d = [], []
            for r in range(8):
                a, b = fit([v[t][r] for t in DEPTH], np.array([[u[t][r]] for t in DEPTH]))
                a_d.append(a[0]); b_d.append(b[0])
            a_m, b_m = fit(vmean, np.array([[u[t].mean()] for t in DEPTH]))
            inter[key], slope[key] = (float(a_m[0]), np.array(a_d)), float(b_m[0])
        print(f'\n=== Parse | training {label} ===   (scorer consistency on raw reference: {chk:.1e}; V at 16/48/144 = {np.round(vmean, 4)})')
        print(f"{'model':8s} {'reference':8s} {'S n16':>9s} {'S n48':>9s} {'O n48':>9s} {'S 144':>9s} {'slope':>7s} {'corrected':>10s}")
        for key in util:
            f, kind = key
            o48 = direct[f]['S_n48'].mean() if kind == 'direct' else avg(own[f], f'O_{kind}_n48').mean()
            print(f"{f:8s} {kind:8s} {util[key]['n16'].mean():9.4f} {util[key]['n48'].mean():9.4f} {o48:9.4f} {util[key]['full'].mean():9.4f} {slope[key]:7.3f} {inter[key][0]:10.4f}")
            rows_u.append(['Parse', label, f, kind, f"{util[key]['n16'].mean():.6f}", f"{util[key]['n48'].mean():.6f}", f'{o48:.6f}',
                           f"{util[key]['full'].mean():.6f}", f'{slope[key]:.4f}', f'{inter[key][0]:.6f}'])
        for f in families:
            ox = {t: float(avg(own[f], f'OX_own_{t}').mean() - util[(f, 'own')][t].mean()) for t in DEPTH}
            so = {t: float(avg(own[f], f'O_own_{t}').mean() - util[(f, 'own')][t].mean()) for t in ('n16', 'n48')}
            sr = {t: float(avg(own[f], f'O_raw_{t}').mean() - util[(f, 'raw')][t].mean()) for t in ('n16', 'n48')}
            print(f'{f}: O-S with own control {so}; O-S with raw control {sr}; another individual as input (own control) {ox}')
            summary[f'{label}|{f}'] = {'O_minus_S_own': so, 'O_minus_S_raw': sr, 'other_individual_input_own': ox,
                                       'corrected_own': inter[(f, 'own')][0], 'corrected_raw': inter[(f, 'raw')][0], 'slope_own': slope[(f, 'own')], 'slope_raw': slope[(f, 'raw')]}
        print('direct-minus-state margins:   S n48 raw | S n48 own | corrected raw | corrected own (donors > 0 of 8)')
        for d in DIRECT:
            for f in families:
                m_raw = float((util[(d, 'direct')]['n48'] - util[(f, 'raw')]['n48']).mean())
                m_own = float((util[(d, 'direct')]['n48'] - util[(f, 'own')]['n48']).mean())
                c_raw = inter[(d, 'direct')][0] - inter[(f, 'raw')][0]
                c_own = inter[(d, 'direct')][0] - inter[(f, 'own')][0]
                dn_own = inter[(d, 'direct')][1] - inter[(f, 'own')][1]
                dn_raw = inter[(d, 'direct')][1] - inter[(f, 'raw')][1]
                print(f'  {d:4s}-{f:7s} {m_raw:+.4f} | {m_own:+.4f} | {c_raw:+.4f} ({int((dn_raw > 0).sum())}) | {c_own:+.4f} ({int((dn_own > 0).sum())})')
                rows_p.append(['Parse', label, f'{d}-{f}', f'{m_raw:.6f}', f'{m_own:.6f}', f'{c_raw:.6f}', int((dn_raw > 0).sum()), f'{c_own:.6f}', int((dn_own > 0).sum()),
                               f'{dn_own.min():.6f}', f'{dn_own.max():.6f}'])
    with open(OWN / 'OWN_CONTROL_UTILITIES.tsv', 'w', newline='') as h:
        w = csv.writer(h, delimiter='\t')
        w.writerow(['dataset', 'training', 'family', 'prediction_reference', 'U_S_16', 'U_S_48', 'U_O_48', 'U_S_144', 'slope_dU_S_dV', 'corrected_intercept'])
        w.writerows(rows_u)
    with open(OWN / 'OWN_CONTROL_PAIRS.tsv', 'w', newline='') as h:
        w = csv.writer(h, delimiter='\t')
        w.writerow(['dataset', 'training', 'pair_direct_minus_state', 'd_S_48_raw_reference', 'd_S_48_own_control', 'corrected_raw_reference', 'donors_positive_raw',
                    'corrected_own_control', 'donors_positive_own', 'corrected_own_donor_min', 'corrected_own_donor_max'])
        w.writerows(rows_p)
    (OWN / 'OWN_CONTROL_SUMMARY.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
