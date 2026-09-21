#!/usr/bin/env python3
"""Reconstruct all 42 CV checkpoint risks and the frozen training selection.

Author implementation of the fixed evaluation equations using PyTorch operators.
Requires NumPy and PyTorch >= 2.0. Does not import scGen or scvi-tools.
Only weights_only=True is used, after checking the fixed companion manifest.
BSD-3-Clause; see LICENSE and LICENSE_SCOPE.md.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from review_companion import (ARCHIVE_SHA, MANIFEST_SHA, PREFIX, extract_one,
                              file_sha, member_path, module_from, require,
                              scientific_manifest, verify)

RATES = (0.001, 0.0003)
EPOCHS = (80, 160, 320)
# Fixed before replay. CPU/GPU float32 GEMM differences affect decoded means;
# aggregation remains float64. These limits are far below candidate gaps.
ATOL = 2e-7
RTOL = 2e-6


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


class BoundCompanion:
    def __init__(self, root):
        self.root = root
        p = scientific_manifest(root)
        self.manifest = {}
        for line in p.read_text().splitlines():
            digest, name = line.split('  ', 1)
            member_path(PREFIX + name)
            require(name not in self.manifest, 'Duplicate manifest member')
            self.manifest[name] = digest
        self.bindings = {}

    def path(self, name):
        member_path(PREFIX + name)
        path = self.root / name
        require(name in self.manifest, 'Unbound input: ' + name)
        require(path.is_file() and not path.is_symlink(), 'Regular input required: ' + name)
        require(path.resolve().is_relative_to(self.root.resolve()), 'Input leaves companion root')
        if name not in self.bindings:
            digest = file_sha(path)
            require(digest == self.manifest[name], 'Input hash differs: ' + name)
            self.bindings[name] = {'sha256': digest, 'bytes': path.stat().st_size}
        return path

    def read(self, name):
        return json.loads(self.path(name).read_text())


@contextlib.contextmanager
def companion_inputs(args):
    if args.companion_dir is not None:
        yield BoundCompanion(args.companion_dir), None
    else:
        verification = verify(args.archive)
        with tempfile.TemporaryDirectory(prefix='bound-training-', dir=args.out_dir) as tmp:
            root = Path(tmp)
            with zipfile.ZipFile(args.archive) as z:
                for name in z.namelist():
                    relative = member_path(name)
                    if (relative in ('MANIFEST.sha256', 'ORIGINAL_MANIFEST.sha256')
                        or relative.startswith('05_TRAINING_RECORDS/prepared/')
                        or relative.startswith('05_TRAINING_RECORDS/runs/cv/')
                        or relative.startswith('06_AUDIT_RECORDS/')
                        or relative == '04_CODE/select_training_configuration.py'):
                        extract_one(z, relative, root / relative)
            yield BoundCompanion(root), verification


def state_schema():
    expected = {}
    for family, dimensions in (('z_encoder.encoder', (2000, 512, 512)),
                               ('decoder.decoder', (100, 512, 512))):
        for i, (din, dout) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            prefix = f'{family}.fc_layers.Layer {i}'
            expected[prefix + '.0.weight'] = (dout, din)
            expected[prefix + '.0.bias'] = (dout,)
            for suffix in ('weight', 'bias', 'running_mean', 'running_var'):
                expected[prefix + '.1.' + suffix] = (dout,)
            expected[prefix + '.1.num_batches_tracked'] = ()
    for prefix, dout, din in (('z_encoder.mean_encoder', 100, 512),
                               ('z_encoder.var_encoder', 100, 512),
                               ('decoder.linear_out', 2000, 512)):
        expected[prefix + '.weight'] = (dout, din)
        expected[prefix + '.bias'] = (dout,)
    return expected


class InferenceState:
    """Fixed 2000→512→512→100 / 100→512→512→2000 evaluation graph.

    Each hidden layer: Linear, evaluation BatchNorm(eps=.001), LeakyReLU(.01).
    Dropout is identity in evaluation. The stored mean head supplies the latent;
    the posterior variance head is validated but unused by mean prediction.
    No model classes or executable objects are loaded from checkpoint files.
    """
    def __init__(self, checkpoint, device):
        loaded = torch.load(checkpoint, map_location='cpu', weights_only=True)
        require(isinstance(loaded, dict) and isinstance(loaded.get('state_dict'), dict),
                'Checkpoint state dictionary required')
        raw = loaded['state_dict']
        require(all(k.startswith('module.') for k in raw), 'Unexpected checkpoint namespace')
        state = {k[len('module.'):]: v for k, v in raw.items()}
        expected = state_schema()
        require(set(state) == set(expected), 'The 34-tensor architecture differs')
        for name, value in state.items():
            require(isinstance(value, torch.Tensor) and tuple(value.shape) == expected[name],
                    'Tensor geometry differs: ' + name)
            dtype = torch.int64 if name.endswith('num_batches_tracked') else torch.float32
            require(value.dtype == dtype and bool(torch.isfinite(value).all()),
                    'Invalid tensor: ' + name)
        self.state = {k: v.to(device) for k, v in state.items()}
        self.device = device
        self.epoch = int(loaded['epoch']) + 1
        self.global_step = int(loaded['global_step'])

    def linear(self, x, prefix):
        return F.linear(x, self.state[prefix + '.weight'], self.state[prefix + '.bias'])

    def hidden(self, x, family):
        for i in range(2):
            prefix = f'{family}.fc_layers.Layer {i}'
            x = self.linear(x, prefix + '.0')
            bn = prefix + '.1'
            x = F.batch_norm(x, self.state[bn + '.running_mean'], self.state[bn + '.running_var'],
                             self.state[bn + '.weight'], self.state[bn + '.bias'],
                             training=False, momentum=.01, eps=.001)
            x = F.leaky_relu(x, negative_slope=.01)
        return x

    def encode_tensor(self, x):
        return self.linear(self.hidden(x, 'z_encoder.encoder'), 'z_encoder.mean_encoder')

    def decode_tensor(self, x):
        return self.linear(self.hidden(x, 'decoder.decoder'), 'decoder.linear_out')

    @torch.inference_mode()
    def batches(self, x, operation):
        outputs = []
        for start in range(0, len(x), 512):
            tensor = torch.as_tensor(np.ascontiguousarray(x[start:start + 512], dtype=np.float32),
                                     device=self.device)
            outputs.append(operation(tensor).cpu().numpy())
        result = np.concatenate(outputs)
        require(np.isfinite(result).all(), 'Nonfinite inference')
        return result


def load_fold(bound, fold):
    prefix = f'05_TRAINING_RECORDS/prepared/{fold}/'
    manifest = bound.read(prefix + 'manifest.json')
    require(manifest['fold_id'] == fold, 'Fold identity differs')
    arrays = {}
    for name, record in manifest['files'].items():
        p = bound.path(prefix + name)
        require(file_sha(p) == record['sha256'], 'Prepared manifest binding differs')
        value = np.load(p, allow_pickle=False)
        require(list(value.shape) == record['shape'] and np.isfinite(value).all(), 'Invalid array')
        arrays[name.removesuffix('.npy')] = value
    require(arrays['scales'].shape == (2000,) and (arrays['scales'] >= .1).all(), 'Invalid scales')
    sets = []
    for part in ('train', 'val'):
        x = arrays['X_' + part]
        require(x.dtype == np.float32 and x.ndim == 2 and x.shape[1] == 2000, 'Invalid expression')
        donor, task, arm = (arrays[f'{part}_{name}'] for name in ('donor', 'task', 'arm'))
        require(all(a.shape == (len(x),) and a.dtype == np.int64 for a in (donor, task, arm)), 'Invalid axis')
        ids = set(donor.tolist()); sets.append(ids)
        for d in ids:
            for t in range(5):
                for a in (0, 1):
                    require(np.sum((donor == d) & (task == t) & (arm == a)) == 8,
                            'Expected eight cells per donor/task/arm')
    require(not sets[0].intersection(sets[1]) and sets[0] | sets[1] == set(range(40)), 'Donor split differs')
    return arrays


def predict_risks(model, arrays):
    latent = model.batches(arrays['X_train'], model.encode_tensor)
    donor, task, arm = (arrays['train_' + n] for n in ('donor', 'task', 'arm'))
    shifts = []
    for t in range(5):
        differences = []
        for d in np.unique(donor):
            means = [latent[(donor == d) & (task == t) & (arm == a)].mean(axis=0, dtype=np.float64)
                     for a in (0, 1)]
            differences.append(means[1] - means[0])
        shifts.append(np.asarray(differences).mean(axis=0, dtype=np.float64))
    shifts = np.asarray(shifts)
    x = arrays['X_val']
    val_latent = model.batches(x, model.encode_tensor)
    donor, task, arm = (arrays['val_' + n] for n in ('donor', 'task', 'arm'))
    controls = arm == 0
    shifted = np.asarray(val_latent[controls] + shifts[task[controls]].astype(np.float32), dtype=np.float32)
    decoded = model.batches(shifted, model.decode_tensor)
    task_rows, donor_rows = [], []
    for d in np.unique(donor):
        risks = []
        for t in range(5):
            predicted = decoded[(donor[controls] == d) & (task[controls] == t)].mean(axis=0, dtype=np.float64)
            observed = x[(donor == d) & (task == t) & (arm == 1)].mean(axis=0, dtype=np.float64)
            risk = float(np.mean(((predicted - observed) / arrays['scales']) ** 2, dtype=np.float64))
            require(math.isfinite(risk), 'Nonfinite risk')
            task_rows.append({'donor_id': int(d), 'task_id': t, 'risk': risk}); risks.append(risk)
        donor_rows.append({'donor_id': int(d), 'risk': math.fsum(risks) / 5})
    return {'per_donor_task': task_rows, 'per_donor': donor_rows,
            'mean_donor_risk': math.fsum(r['risk'] for r in donor_rows) / len(donor_rows)}


def compare_rows(actual, expected, axes):
    a = {tuple(row[k] for k in axes): row['risk'] for row in actual}
    e = {tuple(row[k] for k in axes): row['risk'] for row in expected}
    require(len(a) == len(actual) and len(e) == len(expected) and a.keys() == e.keys(), 'Risk row identity differs')
    differences = [abs(a[k] - e[k]) for k in a]
    require(all(math.isclose(a[k], e[k], abs_tol=ATOL, rel_tol=RTOL) for k in a), 'Archived risk tolerance exceeded')
    return max(differences)


def replay(bound, args):
    started = time.time()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if args.device == 'cuda':
        require(torch.cuda.is_available(), 'CUDA device unavailable')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    selector = module_from(bound.path('04_CODE/select_training_configuration.py'), 'frozen_training_selector')
    expected_selection = bound.read('06_AUDIT_RECORDS/TRAINING_SELECTION.json')
    seal = bound.read('06_AUDIT_RECORDS/TRAINING_SELECTION_SEAL.json')
    require(seal['selection_sha256'] == file_sha(bound.path('06_AUDIT_RECORDS/TRAINING_SELECTION.json')),
            'Selection seal differs')
    pooled = {(lr, epoch): [] for lr in RATES for epoch in EPOCHS}
    checks, task_rows = [], []
    for fold in (f'fold{i:02}' for i in range(1, 8)):
        arrays = load_fold(bound, fold)
        for lr in RATES:
            prefix = f'05_TRAINING_RECORDS/runs/cv/{fold}_lr{lr}/'
            fit = bound.read(prefix + 'FIT_RECEIPT.json')
            require(fit['status'] == 'PASS_TRAINING_ONLY_CV' and fit['fold_id'] == fold
                    and fit['learning_rate'] == lr, 'Fit identity differs')
            for epoch in EPOCHS:
                checkpoint = bound.path(prefix + f'checkpoints/epoch{epoch:03}.ckpt')
                expected = bound.read(prefix + f'checkpoints/epoch{epoch:03}_validation.json')
                model = InferenceState(checkpoint, args.device)
                require(model.epoch == epoch and model.global_step == epoch * math.ceil(len(arrays['X_train']) / 512),
                        'Checkpoint epoch or step differs')
                actual = predict_risks(model, arrays)
                pooled[lr, epoch].extend(actual['per_donor'])
                record = {'fold': fold, 'learning_rate': lr, 'epoch': epoch,
                          'donor_task_rows': len(actual['per_donor_task']),
                          'max_abs_task_risk_error': compare_rows(actual['per_donor_task'], expected['per_donor_task'], ('donor_id', 'task_id')),
                          'max_abs_donor_risk_error': compare_rows(actual['per_donor'], expected['per_donor'], ('donor_id',)),
                          'abs_fold_risk_error': abs(actual['mean_donor_risk'] - expected['mean_donor_risk'])}
                require(math.isclose(actual['mean_donor_risk'], expected['mean_donor_risk'], abs_tol=ATOL, rel_tol=RTOL), 'Fold risk differs')
                checks.append(record)
                for row in actual['per_donor_task']:
                    task_rows.append({'fold': fold, 'learning_rate': lr, 'epoch': epoch, **row})
                save_json(args.out_dir / f'{fold}_lr{lr}_epoch{epoch:03}_reconstructed.json', actual)
                print(json.dumps({'checkpoint': len(checks), **record}), flush=True)
    candidates, selected = selector.rank_complete_candidates([
        {'learning_rate': lr, 'epoch': epoch, 'per_donor': pooled[lr, epoch]} for lr, epoch in pooled])
    require((selected['learning_rate'], selected['epoch']) ==
            (expected_selection['selected_learning_rate'], expected_selection['selected_epoch']), 'Selected candidate differs')
    candidate_errors = []
    for actual, expected in zip(candidates, expected_selection['primary_risk_candidate_grid']):
        require((actual['learning_rate'], actual['epoch']) == (expected['learning_rate'], expected['epoch']), 'Candidate order differs')
        compare_rows(actual['per_donor'], expected['per_donor'], ('donor_id',))
        delta = abs(actual['equal_donor_prediction_risk'] - expected['equal_donor_prediction_risk'])
        require(math.isclose(actual['equal_donor_prediction_risk'], expected['equal_donor_prediction_risk'], abs_tol=ATOL, rel_tol=RTOL), 'Candidate mean differs')
        candidate_errors.append(delta)
    with (args.out_dir / 'RECONSTRUCTED_DONOR_TASK_RISKS.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(task_rows[0])); writer.writeheader(); writer.writerows(task_rows)
    save_json(args.out_dir / 'RECONSTRUCTED_CANDIDATE_GRID.json', candidates)
    return {'status': 'PASS_ALL_42_CHECKPOINT_RISKS_AND_FROZEN_SELECTION',
            'checkpoint_count': len(checks), 'donor_task_risks_checked': len(task_rows),
            'donor_risks_checked': sum(len(v) for v in pooled.values()),
            'candidate_count': len(candidates), 'selected_learning_rate': selected['learning_rate'],
            'selected_epoch': selected['epoch'], 'selected_equal_donor_prediction_risk': selected['equal_donor_prediction_risk'],
            'max_abs_task_risk_error': max(r['max_abs_task_risk_error'] for r in checks),
            'max_abs_donor_risk_error': max(r['max_abs_donor_risk_error'] for r in checks),
            'max_abs_candidate_risk_error': max(candidate_errors),
            'comparison_tolerance': {'absolute': ATOL, 'relative': RTOL, 'rule': 'math.isclose; fixed before replay'},
            'checkpoint_checks': checks, 'input_bindings': bound.bindings,
            'checkpoint_load_mode': 'torch.load(weights_only=True), fixed-manifest hash before deserialization',
            'prediction_batch_size': 512, 'numpy': np.__version__, 'torch': torch.__version__,
            'python': platform.python_version(), 'device': args.device, 'threads': args.threads,
            'elapsed_seconds': time.time() - started, 'script_sha256': file_sha(Path(__file__)),
            'historical_selection_seal_replaced': False, 'new_training': False,
            'scope': 'Prepared fold matrices plus 42 saved checkpoints to latent shifts, held-out donor/task risks and frozen candidate ranking.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--archive', type=Path)
    source.add_argument('--companion-dir', type=Path)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    require(args.threads > 0, 'Positive thread count required')
    os.umask(0o077)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    with companion_inputs(args) as (bound, verification):
        result = replay(bound, args)
        result['archive_verification'] = verification
        result['original_companion_archive_sha256'] = ARCHIVE_SHA
        save_json(args.out_dir / 'CHECKPOINT_RISK_REPLAY_RECEIPT.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'checkpoint_count', 'max_abs_task_risk_error', 'selected_learning_rate', 'selected_epoch')}))


if __name__ == '__main__':
    main()
