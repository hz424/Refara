#!/usr/bin/env python3
"""Score state-output fits with the model's own control prediction as the prediction reference.

Predicted effect for an input block m:  mean_c F(cell c, perturbation) - mean_c F(cell c, no perturbation)
(the second term from control_bank.py), compared with the observed effect Y - C_obs.
Designs (own donor unless stated; utility = -standardized MSE; tasks equal within donor):
  S_raw / O_raw   released convention: predicted state minus the raw mean of the input block
  S_own / O_own   predicted state minus the model's control prediction for the same input block;
                  S: observation block = input block, O: observation block != input block
  OX_own          input block (and its control prediction) from another donor, observation block own
Block sets: n16, n48 (three disjoint released streams and their prefixes) and the full 144-cell pool.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from data_locations import output_root
from cross_individual_score import PUB, R, NB, BLOCKSETS, FULL, block_weight_matrix

OWN = output_root() / 'own_control'

N_VEC = R + 3 * R * NB


def i_y(r): return r
def i_c(r, b): return R + r * NB + b
def i_p(r, b): return R + R * NB + r * NB + b
def i_q(r, b): return R + 2 * R * NB + r * NB + b


def designs():
    out = {}
    for tag, blocks in list(BLOCKSETS.items()) + [('full', (FULL,))]:
        s_raw, o_raw, s_own, o_own, ox_own = [], [], [], [], []
        for d in range(R):
            others = [e for e in range(R) if e != d]
            s_raw.append([[i_y(d), i_c(d, i), i_p(d, i), i_c(d, i)] for i in blocks])
            s_own.append([[i_y(d), i_c(d, i), i_p(d, i), i_q(d, i)] for i in blocks])
            ox_own.append([[i_y(d), i_c(d, i), i_p(e, j), i_q(e, j)] for e in others for i in blocks for j in blocks])
            if len(blocks) > 1:
                o_raw.append([[i_y(d), i_c(d, i), i_p(d, j), i_c(d, j)] for i in blocks for j in blocks if j != i])
                o_own.append([[i_y(d), i_c(d, i), i_p(d, j), i_q(d, j)] for i in blocks for j in blocks if j != i])
        out[f'S_raw_{tag}'] = np.asarray(s_raw)
        out[f'S_own_{tag}'] = np.asarray(s_own)
        out[f'OX_own_{tag}'] = np.asarray(ox_own)
        if o_raw:
            out[f'O_raw_{tag}'] = np.asarray(o_raw)
            out[f'O_own_{tag}'] = np.asarray(o_own)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regime', required=True)
    ap.add_argument('--fit', required=True)
    ap.add_argument('--labels', type=int, default=30)
    ap.add_argument('--max-keys', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    t0 = time.time()
    base = PUB / args.regime
    cache = base / 'score_cache_v1'
    treated = np.load(cache / 'treated.npy')
    controls = np.load(cache / 'canonical_controls.npy')
    group_index = np.load(cache / 'group_index.npy')
    root_index = np.load(cache / 'root_index.npy')
    task_index = np.load(cache / 'task_index.npy')
    alloc = np.load(cache / 'max_depth_allocations.npy')
    std = np.sqrt(np.load(cache / 'weights.npy')) / np.load(cache / 'scales.npy')
    ctrl_pred = np.load(OWN / 'control_points' / 'parse' / f'{args.regime}__{args.fit}.npy')
    assert ctrl_pred.shape == controls.shape
    n_keys = int(task_index.max()) + 1
    rowmap = np.full((n_keys, R), -1, dtype=np.int64)
    rowmap[task_index, root_index] = np.arange(len(task_index))
    labels = list(range(args.labels))
    a = len(labels)
    wmat = block_weight_matrix(alloc, labels)
    cmean = np.einsum('abgc,gcf->abgf', wmat, controls.astype(np.float64), optimize=True) * std
    qmean = np.einsum('abgc,gcf->abgf', wmat, ctrl_pred.astype(np.float64), optimize=True) * std
    bank = np.load(base / 'banks' / args.fit / 'points.npy', mmap_mode='r')
    des = designs()
    coef = np.array([1.0, -1.0, -1.0, 1.0])
    cc = np.outer(coef, coef)
    util = {k: np.zeros((a, R)) for k in des}
    keys = range(n_keys if not args.max_keys else args.max_keys)
    for k in keys:
        rows = rowmap[k]
        groups = group_index[rows]
        vec = np.empty((a, N_VEC, 2000))
        vec[:, :R] = (treated[rows] * std)[None]
        vec[:, R:R + R * NB] = cmean[:, :, groups].transpose(0, 2, 1, 3).reshape(a, R * NB, 2000)
        pts = np.asarray(bank[rows], dtype=np.float64)
        vec[:, R + R * NB:R + 2 * R * NB] = (np.einsum('abrc,rcf->arbf', wmat[:, :, groups], pts, optimize=True) * std).reshape(a, R * NB, 2000)
        vec[:, R + 2 * R * NB:] = qmean[:, :, groups].transpose(0, 2, 1, 3).reshape(a, R * NB, 2000)
        gram = vec @ vec.transpose(0, 2, 1)
        for name, idx in des.items():
            sub = gram[:, idx[:, :, :, None], idx[:, :, None, :]]
            util[name] -= np.tensordot(sub, cc, axes=([3, 4], [0, 1])).mean(axis=2)
        if k % 100 == 0:
            print(f'key {k}/{len(keys)} elapsed {time.time() - t0:.0f}s', flush=True)
    n = len(keys)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, names=np.array(list(util)), utility=np.stack([util[x] / n for x in util]), labels=np.array(labels),
             fit=args.fit, regime=args.regime, task_keys=n)
    print(json.dumps({'fit': args.fit, 'regime': args.regime, 'task_keys': n, 'seconds': round(time.time() - t0, 1)}))


if __name__ == '__main__':
    main()
