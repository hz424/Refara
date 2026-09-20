"""Allocate controls, predict from each recorded cell block, then prepare and score."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from make_example import make_example as make_inputs
from reference_design.io import read_cells
from reference_design.prepare import prepare_plan
from reference_design.protocol import run_plan


def read_rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict_cells(cells, gain, offset):
    """Small deterministic state predictor; replace with your model's inference."""
    return np.mean(cells * np.asarray(gain) + np.asarray(offset), axis=0)


def run_example(output, format='tsv'):
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Example directory must be absent or empty')
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = make_inputs(output/'inputs', format)
    manifest = json.loads(manifest_path.read_text())
    inputs, allocated = manifest_path.parent, output/'allocated'

    # The first preparation freezes the cell selections using the fixed baselines.
    prepare_plan(manifest_path, allocated)
    allocation_plan = json.loads((allocated/'plan.json').read_text())
    conditions = {row['task']: row['condition'] for row in read_rows(inputs/'tasks.tsv')}
    parameters = {
        'genes': ['g1', 'g2', 'g3'],
        'gain': [1.25, 0.75, 1.10],
        'offset_by_condition': {'pert_a': [0.8, 1.2, 0.7], 'pert_b': [1.8, 2.2, 1.7]},
    }
    # These are declared toy coefficients; no treated evaluation cells enter inference.
    checkpoint = inputs/'affine_parameters.json'
    write_json(checkpoint, parameters)
    code = inputs/'predictor_source.py'
    shutil.copyfile(Path(__file__).resolve(), code)
    model = dict(name='affine_state', kind='state', conditioning='block',
                 representation='native', prediction_files={}, membership_sha256={},
                 provenance_files={})

    for index, task in enumerate(allocation_plan['tasks']):
        controls_path = allocated/task['controls']
        membership_path = allocated/task['membership']
        controls = read_cells(controls_path, tuple(parameters['genes']))
        by_id = dict(zip(controls.cell_ids, controls.values))
        blocks = defaultdict(list)
        for row in read_rows(membership_path):
            key = (row['allocation'], int(row['depth']), row['block'])
            blocks[key].append(row['cell_id'])

        prediction_path = inputs/f't{index}_affine_prediction.tsv'
        with prediction_path.open('w', newline='') as stream:
            writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
            writer.writerow(['allocation', 'depth', 'block', *controls.genes])
            for key, cell_ids in sorted(blocks.items()):
                # Pass only the exact member cells for this allocation/depth/block.
                selected = np.stack([by_id[cell_id] for cell_id in cell_ids])
                prediction = predict_cells(selected, parameters['gain'],
                                           parameters['offset_by_condition'][conditions[task['id']]])
                writer.writerow([*key, *prediction])

        provenance_path = inputs/f't{index}_affine_provenance.json'
        artifacts = [prediction_path, inputs/manifest['cells']['path'], inputs/manifest['tasks'],
                     controls_path, membership_path,
                     allocated/task['references'], checkpoint, code]
        write_json(provenance_path, {
            'description': 'Affine state predictions computed from each recorded input-cell block.',
            'files': [{'path': os.path.relpath(path, inputs), 'sha256': digest(path)}
                      for path in artifacts],
        })
        model['prediction_files'][task['id']] = prediction_path.name
        model['membership_sha256'][task['id']] = digest(membership_path)
        model['provenance_files'][task['id']] = provenance_path.name

    manifest['models'] = [manifest['models'][0], model]
    block_manifest = inputs/'block_manifest.json'
    write_json(block_manifest, manifest)
    prepare_plan(block_manifest, output/'prepared')
    run_plan(output/'prepared/plan.json', output/'results')
    return block_manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--format', choices=['h5ad', 'tsv'], default='tsv')
    args = parser.parse_args()
    manifest = run_example(args.output, args.format)
    print(f'Block manifest: {manifest}')
    print(f'Prepared plan: {manifest.parents[1]/"prepared/plan.json"}')
    print(f'Scores: {manifest.parents[1]/"results/primary_summary_scores.tsv"}')
