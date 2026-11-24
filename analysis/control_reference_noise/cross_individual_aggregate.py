#!/usr/bin/env python3
"""Aggregate per-fit cross-donor utilities, validate against the published scores, count reversals."""
from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from data_locations import evaluation_root, output_root

W = output_root() / 'cross_individual'
PUBLISHED = evaluation_root() / 'parse'
# order of the design axis in the released utility arrays
RELEASED_PATTERNS = ('ALL_SHARED', 'OBS_PRED_SHARED_MODEL_SEPARATE', 'OBS_MODEL_SHARED_PRED_SEPARATE',
                     'PRED_MODEL_SHARED_OBS_SEPARATE', 'ALL_DISJOINT')
FAMILIES = ('NC', 'CM', 'TW', 'PCA', 'RBF', 'scGen', 'CPA', 'CellOT')
DIRECT, STATE_OUTPUT, SEEDS = FAMILIES[:5], FAMILIES[5:], (17, 29, 43)
PUB_DEPTH_INDEX = {'n16': 0, 'n48': 4}
PAIRS = list(itertools.combinations(range(8), 2))


def load_patterns():
    return RELEASED_PATTERNS


def main():
    patterns = load_patterns()
    short = {'ALL_SHARED': 'S'}
    for p in patterns:
        if p == 'ALL_SHARED':
            continue
        short[p] = {'MODEL': 'M', 'PRED': 'P', 'OBS': 'O'}.get(p.split('_SEPARATE')[0].split('_')[-1], 'D')
    print('published pattern order:', [(p, short[p]) for p in patterns])
    summary = {'validation': {}, 'reversals': {}, 'labels': None}
    util_rows, pair_rows = [], []
    for regime in ('common48', 'common128'):
        fam = {}
        names = None
        seed_arrays = {}
        for f in FAMILIES:
            fits = [f'{f}__primary__deterministic'] if f in DIRECT else [f'{f}__primary__{s}' for s in SEEDS]
            arrs = []
            for fit in fits:
                z = np.load(W / 'per_fit' / f'{regime}__{fit}.npz')
                names = [str(x) for x in z['names']] if names is None else names
                assert [str(x) for x in z['names']] == names and int(z['task_keys']) == 722
                arrs.append(z['utility'])                       # (design, A, R)
                summary['labels'] = int(len(z['labels']))
            seed_arrays[f] = arrs
            fam[f] = np.mean(arrs, axis=0)
        # --- validation against the published per-seed utilities (panel 0 = primary) ---
        pub = np.load(PUBLISHED / regime / 'PANEL_SEED_UTILITIES.npy', mmap_mode='r')  # (panel, seed, label, depth, pattern, root, family)
        a = summary['labels']
        worst = 0.0
        for tag, di in PUB_DEPTH_INDEX.items():
            for pi, p in enumerate(patterns):
                d = names.index(f'{short[p]}_{tag}')
                for fi, f in enumerate(FAMILIES):
                    for si in range(3):
                        mine = seed_arrays[f][si if f in STATE_OUTPUT else 0][d]            # (A, R)
                        ref = np.asarray(pub[0, si, :a, di, pi, :, fi])
                        worst = max(worst, float(np.abs(mine - ref).max()))
        summary['validation'][regime] = {'max_abs_diff_vs_published_per_label_seed_root': worst,
                                         'compared': 'S/M/P/O/D at depths 16 and 48, 8 families, 3 seeds, 8 donors'}
        print(f'[{regime}] validation vs published PANEL_SEED_UTILITIES: max |diff| = {worst:.3e}')
        # --- label-averaged tables ---
        root_util = {f: fam[f].mean(axis=1) for f in FAMILIES}                        # (design, R)
        mean_util = {f: root_util[f].mean(axis=1) for f in FAMILIES}                  # (design,)
        for di_, name in enumerate(names):
            for f in FAMILIES:
                util_rows.append([regime, name, f, f'{mean_util[f][di_]:.8f}'] + [f'{v:.8f}' for v in root_util[f][di_]])
        summary['reversals'][regime] = {}
        for tag in ('n16', 'n48', 'full'):
            base = names.index(f'S_{tag}')
            designs = [n for n in names if n.endswith('_' + tag) and not n.startswith('S_')]
            for name in designs:
                d = names.index(name)
                rev_all, rev_ds = [], []
                for i, j in PAIRS:
                    fi, fj = FAMILIES[i], FAMILIES[j]
                    m0 = mean_util[fi][base] - mean_util[fj][base]
                    m1 = mean_util[fi][d] - mean_util[fj][d]
                    r0 = root_util[fi][base] - root_util[fj][base]
                    r1 = root_util[fi][d] - root_util[fj][d]
                    rev = bool(m0 * m1 < 0)
                    crossing = int(np.sum(r0 * r1 < 0))
                    kind = 'direct-state' if (fi in DIRECT) != (fj in DIRECT) else ('direct-direct' if fi in DIRECT else 'state-state')
                    pair_rows.append([regime, tag, name, f'{fi}-{fj}', kind, f'{m0:.8f}', f'{m1:.8f}', int(rev), crossing])
                    if rev:
                        rev_all.append(f'{fi}-{fj}')
                        if kind == 'direct-state':
                            rev_ds.append(f'{fi}-{fj}')
                order0 = [FAMILIES[i] for i in np.argsort([-mean_util[f][base] for f in FAMILIES])]
                order1 = [FAMILIES[i] for i in np.argsort([-mean_util[f][d] for f in FAMILIES])]
                summary['reversals'][regime][name] = {
                    'reversed_of_28': len(rev_all), 'reversed_direct_state_of_15': len(rev_ds), 'pairs': rev_all,
                    'ranking_under_S': order0, 'ranking_under_design': order1,
                    'state_shift_from_S': {f: float(mean_util[f][d] - mean_util[f][base]) for f in STATE_OUTPUT},
                    'direct_shift_from_S': {f: float(mean_util[f][d] - mean_util[f][base]) for f in DIRECT}}
    out = W
    with open(out / 'XDONOR_UTILITIES.tsv', 'w', newline='') as h:
        w = csv.writer(h, delimiter='\t')
        w.writerow(['regime', 'design', 'family', 'mean_utility'] + [f'donor_root_{r}' for r in range(8)])
        w.writerows(util_rows)
    with open(out / 'XDONOR_PAIRS.tsv', 'w', newline='') as h:
        w = csv.writer(h, delimiter='\t')
        w.writerow(['regime', 'blockset', 'design', 'pair', 'kind', 'margin_S', 'margin_design', 'reversed', 'donors_with_sign_change_of_8'])
        w.writerows(pair_rows)
    (out / 'XDONOR_SUMMARY.json').write_text(json.dumps(summary, indent=1))
    for regime in summary['reversals']:
        print(f'\n== {regime}: reversals relative to own-donor shared (S) at the same block set ==')
        print(f"{'design':10s} {'of 28':>6s} {'dir-state/15':>13s}   state shifts (scGen, CPA, CellOT)        top model S -> design")
        for name, r in summary['reversals'][regime].items():
            s = r['state_shift_from_S']
            print(f"{name:10s} {r['reversed_of_28']:6d} {r['reversed_direct_state_of_15']:13d}   "
                  f"{s['scGen']:+.4f} {s['CPA']:+.4f} {s['CellOT']:+.4f}        {r['ranking_under_S'][0]} -> {r['ranking_under_design'][0]}")


if __name__ == '__main__':
    main()
