#!/usr/bin/env python3
"""Offset between a model's control-condition output and the control mean, with and without sampling noise.

For each stratum the model's output for the 144 (Parse) or 24 (Influenza PBMC) canonical control cells under the control
condition is compared with the measured cells, in the standardized units of the utility.

Same-cell distance: ||mean f - mean x||_s^2 over the same n cells. It contains the sampling variance of the measured
mean wherever the model does not reproduce the sampled cells.

Squared offset, noise removed: a U-statistic over pairs of different cells,

    sum_{i != j} <f_i - x_i, f_j - x_j>_s / (n (n - 1)),

where x_i is a measured control cell and f_i the model's output for it. Products of the same cell are excluded, so
cell-level noise does not contribute; the estimate is unbiased if the cells are a random sample of the stratum and each
output depends only on its own input cell (true of scGen, CPA and CellOT).

The last column is the rise in noise-corrected utility when the prediction reference changes from the raw control mean
to the model's own control output (from own_control_aggregate.py and own_control_influenza.py), which equals the
squared offset when the offset is common to the model's treated and control outputs.
Strata are weighted by their number of tasks within donor, donors equally, and seeds are averaged.
"""
import csv

import numpy as np

from data_locations import evaluation_root, output_root

EV = evaluation_root()
OWN = output_root() / 'own_control'


def corrected_utilities(path):
    with open(path) as handle:
        return {(r['training'], r['family'], r['prediction_reference']): float(r['corrected_intercept'])
                for r in csv.DictReader(handle, delimiter='\t')}


UTILITIES = {'gse162632': corrected_utilities(OWN / 'OWN_CONTROL_UTILITIES_GSE162632.tsv'),
             'parse': corrected_utilities(OWN / 'OWN_CONTROL_UTILITIES.tsv')}
rows = []
for dataset, sub, regimes in (('Influenza PBMC', 'gse162632', (('original8', '8 cells'), ('cap48', '<=48 cells'))),
                              ('Parse', 'parse', (('common48', '48 cells'), ('common128', '<=128 cells')))):
    for regime, label in regimes:
        c = EV / sub / regime / 'score_cache_v1'
        std = np.sqrt(np.load(c / 'weights.npy')) / np.load(c / 'scales.npy')
        x = np.load(c / 'canonical_controls.npy', mmap_mode='r')                      # (strata, cells, genes)
        gi, ri = np.load(c / 'group_index.npy'), np.load(c / 'root_index.npy')
        n = x.shape[1]

        def donor_mean(per_stratum):
            return float(np.mean([per_stratum[gi][ri == r].mean() for r in range(8)]))

        sx, xx = np.empty((x.shape[0], x.shape[2])), np.empty(x.shape[0])
        for g in range(x.shape[0]):
            xg = np.asarray(x[g], dtype=np.float64) * std
            sx[g], xx[g] = xg.sum(0), (xg ** 2).sum()
        noise = (xx - (sx ** 2).sum(1) / n) / (n * (n - 1))                          # sampling variance of the measured mean
        for fam in ('scGen', 'CPA', 'CellOT'):
            same_cell, offset = [], []
            for seed in (17, 29, 43):
                f = np.load(OWN / 'control_points' / sub / f'{regime}__{fam}__primary__{seed}.npy', mmap_mode='r')
                if f.shape != x.shape:
                    raise SystemExit(f'control output and control cells differ in shape for {regime} {fam} {seed}')
                e_same, e_off = np.empty(x.shape[0]), np.empty(x.shape[0])
                for g in range(x.shape[0]):
                    fg = np.asarray(f[g], dtype=np.float64) * std
                    xg = np.asarray(x[g], dtype=np.float64) * std
                    sf = fg.sum(0)
                    e_same[g] = ((sf - sx[g]) ** 2).sum() / n ** 2
                    e_off[g] = (((sf ** 2).sum() - (fg ** 2).sum()) - 2 * ((sf * sx[g]).sum() - (fg * xg).sum())
                                + ((sx[g] ** 2).sum() - xx[g])) / (n * (n - 1))
                same_cell.append(donor_mean(e_same))
                offset.append(donor_mean(e_off))
            rise = UTILITIES[sub][(label, fam, 'own')] - UTILITIES[sub][(label, fam, 'raw')]
            rows.append([dataset, label, fam, n, f'{np.mean(same_cell):.6f}', f'{donor_mean(noise):.6f}', f'{np.mean(offset):.6f}',
                         f'{min(offset):.6f}', f'{max(offset):.6f}', f'{rise:.6f}'])
            print(f'{dataset:15s} {label:12s} {fam:7s} n={n:3d} same-cell {np.mean(same_cell):.4f}  noise of measured mean '
                  f'{donor_mean(noise):.4f}  offset {np.mean(offset):.4f}  utility rise {rise:.4f}')
with open(OWN / 'CONTROL_OFFSET_NOISE_CORRECTED.tsv', 'w', newline='') as h:
    w = csv.writer(h, delimiter='\t')
    w.writerow(['dataset', 'training', 'family', 'control_cells_per_stratum', 'reconstruction_error_same_cells',
                'sampling_variance_of_measured_mean', 'squared_offset_noise_corrected', 'seed_min', 'seed_max',
                'noise_corrected_utility_change_raw_to_own'])
    w.writerows(rows)
