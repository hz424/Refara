#!/usr/bin/env python3
"""Measured cross-donor reference designs from the published Parse prediction banks.

Nothing is refitted: the released per-cell state predictions (points.npy, task x 144 canonical control
cells x 2000 genes) and direct-effect banks are rescored.

Tuple order follows the manuscript: (observation, prediction reference, model input).
A role is a (donor, block) pair. The observation role is always the target
donor's own block, so the measured response is always properly paired.

  within donor (published patterns): S (i,i,i)  M (i,i,j)  P (i,j,i)  O (i,j,j)  D (i,j,l)
  cross donor, d' != d:
    OX  obs (d,i)  pred (d',j)  model (d',j)   conditioned and centred on another individual
    PX  obs (d,i)  pred (d',j)  model (d,i)    own input, baseline from another individual
    MX  obs (d,i)  pred (d,i)   model (d',j)   input from another individual, own baseline
    DX  obs (d,i)  pred (d',i)  model (d'',i)  two other individuals, d' != d''

Block sets: depth 48 (three disjoint published streams), depth 16 (their nested
prefixes) and the full 144-cell canonical pool (one block; S/OX/PX/MX/DX only).

Utility = -sum_g w_g ((residual_g)/s_g)^2, tasks equal within donor.
All squared norms are evaluated from Gram matrices in float64.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from data_locations import evaluation_root

PUB = evaluation_root() / 'parse'
DIRECT = ('NC', 'CM', 'TW', 'PCA', 'RBF')
STATE_OUTPUT = ('scGen', 'CPA', 'CellOT')
SEEDS = (17, 29, 43)
R = 8          # donors (roots)
NB = 7         # blocks per donor: 0-2 depth 48, 3-5 depth 16, 6 full pool
BLOCKSETS = {'n48': (0, 1, 2), 'n16': (3, 4, 5)}
FULL = 6


def block_weight_matrix(alloc, labels):
    """(A, NB, 88, 144) averaging weights for every allocation label and block."""
    a = len(labels)
    w = np.zeros((a, NB, 88, 144), dtype=np.float64)
    g = np.arange(88)[:, None]
    for ai, label in enumerate(labels):
        for b in range(3):
            ranks = alloc[label, b].astype(np.int64)          # (88, 48)
            assert len(np.unique(ranks[0])) == 48
            np.add.at(w[ai, b], (g, ranks), 1.0 / 48)
            np.add.at(w[ai, 3 + b], (g, ranks[:, :16]), 1.0 / 16)
        w[ai, FULL] = 1.0 / 144
    assert np.allclose(w.sum(-1), 1.0)
    return w


def i_y(r):
    return r


def i_c(r, b):
    return R + r * NB + b


def i_x(r, b, state):
    return R + R * NB + (r * NB + b if state else r)


def tuples_for(state):
    """design name -> (R, T, 4) index array and (4,) coefficients.

    residual = Y_d - C_obs - X_model + C_pred  (state)
    residual = Y_d - C_obs - E_model           (direct; fourth slot repeats with coef 0)
    """
    designs = {}

    def add(name, rows_per_root):
        designs[name] = np.asarray(rows_per_root, dtype=np.int64)

    def row(d, obs, pred, model):
        # obs/pred/model are (donor, block)
        if state:
            return [i_y(d), i_c(*obs), i_x(model[0], model[1], True), i_c(*pred)]
        return [i_y(d), i_c(*obs), i_x(model[0], 0, False), i_c(*obs)]

    for tag, blocks in BLOCKSETS.items():
        pats = {'S': [], 'M': [], 'P': [], 'O': [], 'D': []}
        for i in blocks:
            pats['S'].append((i, i, i))
            for j in blocks:
                if j == i:
                    continue
                pats['M'].append((i, i, j))
                pats['P'].append((i, j, i))
                pats['O'].append((i, j, j))
                for l in blocks:
                    if l not in (i, j):
                        pats['D'].append((i, j, l))
        for p, tri in pats.items():
            add(f'{p}_{tag}', [[row(d, (d, o), (d, q), (d, m)) for (o, q, m) in tri] for d in range(R)])
        ox, px, mx, dx = [], [], [], []
        for d in range(R):
            others = [e for e in range(R) if e != d]
            ox.append([row(d, (d, i), (e, j), (e, j)) for e in others for i in blocks for j in blocks])
            px.append([row(d, (d, i), (e, j), (d, i)) for e in others for i in blocks for j in blocks])
            mx.append([row(d, (d, i), (d, i), (e, j)) for e in others for i in blocks for j in blocks])
            dx.append([row(d, (d, i), (e, i), (f, i)) for e in others for f in others if f != e for i in blocks])
        add(f'OX_{tag}', ox)
        add(f'PX_{tag}', px)
        add(f'MX_{tag}', mx)
        add(f'DX_{tag}', dx)
    s, ox, px, mx, dx = [], [], [], [], []
    for d in range(R):
        others = [e for e in range(R) if e != d]
        s.append([row(d, (d, FULL), (d, FULL), (d, FULL))])
        ox.append([row(d, (d, FULL), (e, FULL), (e, FULL)) for e in others])
        px.append([row(d, (d, FULL), (e, FULL), (d, FULL)) for e in others])
        mx.append([row(d, (d, FULL), (d, FULL), (e, FULL)) for e in others])
        dx.append([row(d, (d, FULL), (e, FULL), (f, FULL)) for e in others for f in others if f != e])
    add('S_full', s)
    add('OX_full', ox)
    add('PX_full', px)
    add('MX_full', mx)
    add('DX_full', dx)
    coef = np.array([1.0, -1.0, -1.0, 1.0 if state else 0.0])
    return designs, coef


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regime', required=True, choices=('common48', 'common128'))
    ap.add_argument('--fit', required=True, help='e.g. scGen__primary__17 or TW__primary__deterministic')
    ap.add_argument('--labels', type=int, default=30, help='number of allocation labels (0..n-1)')
    ap.add_argument('--max-keys', type=int, default=0, help='debug: only the first k task keys')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    t0 = time.time()
    base = PUB / args.regime
    cache = base / 'score_cache_v1'
    family = args.fit.split('__')[0]
    state = family in STATE_OUTPUT
    treated = np.load(cache / 'treated.npy')
    controls = np.load(cache / 'canonical_controls.npy')           # (88, 144, 2000) f4
    group_index = np.load(cache / 'group_index.npy')
    root_index = np.load(cache / 'root_index.npy')
    task_index = np.load(cache / 'task_index.npy')
    alloc = np.load(cache / 'max_depth_allocations.npy')           # (1000, 3, 88, 48)
    scales = np.load(cache / 'scales.npy')
    weights = np.load(cache / 'weights.npy')
    std = np.sqrt(weights) / scales                                # (2000,)

    n_keys = int(task_index.max()) + 1
    rowmap = np.full((n_keys, R), -1, dtype=np.int64)
    rowmap[task_index, root_index] = np.arange(len(task_index))
    assert (rowmap >= 0).all()
    labels = list(range(args.labels))
    a = len(labels)
    wmat = block_weight_matrix(alloc, labels)                      # (A, NB, 88, 144)
    cmean = np.einsum('abgc,gcf->abgf', wmat, controls.astype(np.float64), optimize=True) * std  # (A,NB,88,2000)

    bank = np.load(base / 'banks' / args.fit / ('points.npy' if state else 'effects.npy'), mmap_mode='r')
    designs, coef = tuples_for(state)
    cc = np.outer(coef, coef)                                      # (4, 4)
    n_vec = R + R * NB + (R * NB if state else R)
    util = {name: np.zeros((a, R)) for name in designs}            # sum over task keys
    keys = range(n_keys if not args.max_keys else args.max_keys)
    for k in keys:
        rows = rowmap[k]                                           # (R,) bank rows, root order
        groups = group_index[rows]
        vec = np.empty((a, n_vec, 2000), dtype=np.float64)
        vec[:, :R] = (treated[rows] * std)[None]
        vec[:, R:R + R * NB] = cmean[:, :, groups].transpose(0, 2, 1, 3).reshape(a, R * NB, 2000)
        if state:
            pts = np.asarray(bank[rows], dtype=np.float64)         # (R, 144, 2000)
            # (A, NB, R, 144) x (R, 144, 2000) -> (A, R, NB, 2000)
            pm = np.einsum('abrc,rcf->arbf', wmat[:, :, groups], pts, optimize=True) * std
            vec[:, R + R * NB:] = pm.reshape(a, R * NB, 2000)
        else:
            vec[:, R + R * NB:] = (np.asarray(bank[rows], dtype=np.float64) * std)[None]
        gram = vec @ vec.transpose(0, 2, 1)                        # (A, n_vec, n_vec)
        for name, idx in designs.items():                         # idx (R, T, 4)
            sub = gram[:, idx[:, :, :, None], idx[:, :, None, :]]  # (A, R, T, 4, 4)
            q = np.tensordot(sub, cc, axes=([3, 4], [0, 1]))       # (A, R, T)
            util[name] -= q.mean(axis=2)
        if k % 100 == 0:
            print(f'key {k}/{len(keys)} elapsed {time.time() - t0:.0f}s', flush=True)
    n = len(keys)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, names=np.array(list(util)), utility=np.stack([util[x] / n for x in util]),
             labels=np.array(labels), fit=args.fit, regime=args.regime, task_keys=n)
    print(json.dumps({'fit': args.fit, 'regime': args.regime, 'labels': a, 'task_keys': n,
                      'seconds': round(time.time() - t0, 1), 'out': str(out)}))


if __name__ == '__main__':
    main()
