#!/usr/bin/env python3
"""Mean-level reconstruction error of each model's control-condition output against the measured control cells.

For each stratum the model's output for the 144 (Parse) or 24 (Influenza PBMC) canonical control cells under the control
condition is averaged and compared with the mean of the measured cells: ||mean F(x, control) - mean x||_s^2, the same
standardized squared error as the utility.  Strata are weighted by their number of tasks within donor, donors equally.
This is the quantity that enters a raw-control-mean score and cancels in an own-control score when it is common to the
treated and control outputs.
"""
import csv
from pathlib import Path
import numpy as np

from data_locations import evaluation_root, output_root

EV = evaluation_root()
OWN = output_root() / 'own_control'
rows = []
for dataset, sub, regimes in (('Influenza PBMC', 'gse162632', (('original8', '8 cells'), ('cap48', '<=48 cells'))),
                              ('Parse', 'parse', (('common48', '48 cells'), ('common128', '<=128 cells')))):
    for regime, label in regimes:
        c = EV / sub / regime / 'score_cache_v1'
        std = np.sqrt(np.load(c / 'weights.npy')) / np.load(c / 'scales.npy')
        ctl = np.load(c / 'canonical_controls.npy').astype(np.float64).mean(axis=1) * std     # (groups, G)
        gi, ri = np.load(c / 'group_index.npy'), np.load(c / 'root_index.npy')
        for fam in ('scGen', 'CPA', 'CellOT'):
            vals = []
            for seed in (17, 29, 43):
                q = np.load(OWN / 'control_points' / sub / f'{regime}__{fam}__primary__{seed}.npy').astype(np.float64).mean(axis=1) * std
                err = ((q - ctl) ** 2).sum(axis=1)                                            # per stratum
                per_task = err[gi]
                vals.append(np.mean([per_task[ri == r].mean() for r in range(8)]))
            rows.append([dataset, label, fam, f'{np.mean(vals):.6f}', f'{min(vals):.6f}', f'{max(vals):.6f}'])
            print(f'{dataset:15s} {label:12s} {fam:7s} control reconstruction error {np.mean(vals):.4f} (seeds {min(vals):.4f}-{max(vals):.4f})')
with open(OWN / 'CONTROL_RECONSTRUCTION_ERROR.tsv', 'w', newline='') as h:
    w = csv.writer(h, delimiter='\t')
    w.writerow(['dataset', 'training', 'family', 'control_reconstruction_error', 'seed_min', 'seed_max'])
    w.writerows(rows)
