#!/usr/bin/env python3
"""Control-condition predictions of the released state-output fits for the canonical Parse control cells.

For each released fit (scGen, CPA, CellOT; seeds 17/29/43; two training regimes) the model is
loaded exactly as for the released prediction bank (evaluate_reference.StateProvider) and asked for its output for
the same 144 canonical control cells per stratum WITHOUT a perturbation:
  scGen   decode(posterior-mean latent) with zero latent shift
  CPA     predict_points(..., conditions='ctrl')  (the control category of the fitted perturbation encoder)
  CellOT  autoencoder decode(encode(x)), i.e. the identity transport
Nothing is refitted and no treated value is read. Before writing, the loaded provider is checked against the
released treated-condition bank on three tasks.
Output: (88 strata, 144 cells, 2000 genes) float32, in the order of score_cache_v1/canonical_controls.npy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from data_locations import evaluation_root, training_code

EVAL_ROOT = evaluation_root()
sys.path.insert(0, str(training_code()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='parse', choices=('parse', 'gse162632'))
    ap.add_argument('--regime', required=True)
    ap.add_argument('--fit-key', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    import evaluate_reference as er
    import reference_inputs as inputs
    t0 = time.time()
    EVAL = EVAL_ROOT / args.dataset
    binding = json.loads((EVAL / args.regime / 'EVALUATION_BINDING.json').read_text())
    entry = next(e for e in er.all_fits(binding) if e['key'] == args.fit_key)
    family = entry['family']
    assert family in ('scGen', 'CPA', 'CellOT')
    query = inputs.load_queries(binding['dataset'], with_counts=family == 'CPA')
    provider = er.StateProvider(entry, query, args.device)
    n_groups, k = len(query.canonical_offsets), query.controls_per_group
    context = {}
    for row, task in enumerate(query.tasks):
        context.setdefault(int(query.group_index[row]), task['context'])
    assert len(context) == n_groups and (n_groups, k) == ((88, 144) if args.dataset == 'parse' else (40, 24))

    # 1. provider identity: reproduce released treated predictions for three tasks
    bank = np.load(EVAL / args.regime / 'banks' / args.fit_key / 'points.npy', mmap_mode='r')
    worst = 0.0
    for row in ((0, 2500, 5775) if args.dataset == 'parse' else (0, 17, 39)):
        task = query.tasks[row]
        offsets = query.canonical_offsets[int(query.group_index[row])]
        blocks = query.anchors[task['control_group_id']] if (family == 'CellOT' or args.dataset == 'gse162632') else [list(range(k))]
        pred = np.zeros((k, 2000), dtype=np.float32)
        for ranks in blocks:
            pred[ranks] = provider.predict(task, offsets[ranks])
        worst = max(worst, float(np.abs(pred - np.asarray(bank[row])).max()))
    print(json.dumps({'fit': args.fit_key, 'released_bank_max_abs_difference': worst}), flush=True)
    assert worst < 5e-3, 'provider does not reproduce the released bank'

    # 2. control-condition output for every stratum
    out = np.zeros((n_groups, k, 2000), dtype=np.float32)
    for g in range(n_groups):
        offsets = query.canonical_offsets[g]
        if family == 'scGen':
            with provider.api.evaluation(provider.module, provider.torch):
                o = provider.api.decode(provider.module, np.asarray(provider.latent[offsets], dtype=np.float32), provider.torch)
        elif family == 'CPA':
            o = provider.api.predict_points(provider.model, query.counts[offsets], contexts=context[g], conditions='ctrl',
                                            feature_ids=query.feature_ids, batch_size=256)
        else:
            import torch
            ae = provider.model.autoencoder
            ae.eval()
            x = torch.as_tensor(np.asarray(query.expression[offsets], dtype=np.float32), device=next(ae.parameters()).device)
            with torch.no_grad():
                o = ae.decode(ae.encode(x)).cpu().numpy()
        o = np.asarray(o, dtype=np.float32)
        assert o.shape == (k, 2000) and np.isfinite(o).all()
        out[g] = o
    canon = np.asarray(query.expression[query.canonical_offsets], dtype=np.float32)       # the raw control cells
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, out, allow_pickle=False)
    receipt = {'dataset': args.dataset, 'regime': args.regime, 'fit_key': args.fit_key, 'family': family, 'device': args.device, 'shape': list(out.shape),
               'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'released_bank_max_abs_difference': worst,
               'control_definition': {'scGen': 'decode(latent), zero shift', 'CPA': "condition 'ctrl'", 'CellOT': 'autoencoder reconstruction'}[family],
               'mean_abs_reconstruction_error': float(np.abs(out - canon).mean()), 'seconds': round(time.time() - t0, 1)}
    path.with_suffix('.json').write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
