#!/usr/bin/env python3
"""Export the donor-disjoint influenza organizer comparison."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import itertools
import json
from pathlib import Path, PurePosixPath

import numpy as np

ROOT = Path(__file__).resolve().parent / 'transfer'
COLUMNS = ['case', 'unit', 'budget', 'realization', 'model', 'score']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def score_arrays(treated, controls, effects, scales, weights):
    """Score one task: effects are seed × input block × gene, already centered.

    Each input block uses the equal mean of the other two control blocks for
    observation centering. Squared errors precede the mean over fitted seeds.
    """
    treated = np.asarray(treated, dtype=np.float64)
    controls = np.asarray(controls, dtype=np.float64)
    effects = np.asarray(effects, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    require(treated.ndim == 1 and treated.size > 0, 'Expected one treated gene profile')
    genes = treated.size
    require(controls.shape == (3, genes), 'Expected three control block means')
    require(effects.ndim == 3 and effects.shape[0] > 0 and effects.shape[1:] == (3, genes),
            'Expected seed by input block by gene effects')
    require(scales.shape == weights.shape == (genes,), 'Gene scale or weight axis differs')
    require(all(np.isfinite(a).all() for a in (treated, controls, effects, scales, weights)),
            'Scoring arrays must be finite')
    require(np.all(scales > 0) and np.all(weights >= 0) and abs(weights.sum() - 1) < 1e-10,
            'Expected positive scales and normalized nonnegative weights')
    output = []
    for block in range(3):
        observation = treated - controls[[j for j in range(3) if j != block]].mean(axis=0)
        residual = (effects[:, block] - observation) / scales
        output.append(float(np.sum(residual * residual * weights, axis=-1).mean()))
    return np.asarray(output)


def authenticate(source_dir=ROOT):
    root = Path(source_dir).resolve()
    manifest = json.loads((root / 'SOURCE_MANIFEST.json').read_text())
    require(manifest['schema_version'] == 1, 'Unsupported source manifest')
    for name, record in manifest['files'].items():
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe source path')
        path = root / relative
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root),
                'Source is missing or outside the bundle: ' + name)
        require(path.stat().st_size == record['bytes'] and digest(path) == record['sha256'],
                'Source binding differs: ' + name)
    return manifest


def _scores(payload, study, partition, anchor_only=False):
    cases = study[partition + '_cases']
    units = study['units']
    realizations = [study['configuration']['anchor_realization']] if anchor_only else study['realizations']
    models = study['configuration']['models']
    budgets = [row['id'] for row in study['configuration']['budgets']]
    expected = set(itertools.product(cases, budgets, realizations, models))
    reader = csv.DictReader(io.StringIO(payload.decode()), delimiter='\t')
    require(reader.fieldnames == COLUMNS, 'Unexpected score columns')
    values = {}
    for row in reader:
        key = tuple(row[k] for k in ('case', 'budget', 'realization', 'model'))
        require(key in expected and key not in values, 'Duplicate or undeclared score row')
        require(row['unit'] == units[row['case']], 'Score unit differs')
        value = float(row['score'])
        require(np.isfinite(value) and value >= 0, 'Score must be finite and nonnegative')
        values[key] = value
    require(set(values) == expected, 'Incomplete score support')
    return values


def export_inputs(output, group, stage='all', source_dir=ROOT):
    """Export design inputs before making complete assessment tables available."""
    require(stage in ('design', 'assessment', 'all'), 'Unsupported export stage')
    require(group in ('original8', 'cap48'), 'Unknown training group')
    root, output = Path(source_dir).resolve(), Path(output)
    manifest = authenticate(root)
    study = json.loads((root / group / 'study.json').read_text())
    require(not set(study['development_cases']) & set(study['assessment_cases']), 'Cases overlap')
    require(not {study['units'][c] for c in study['development_cases']} &
            {study['units'][c] for c in study['assessment_cases']}, 'Donor units overlap')
    tables = {}
    for name, partition, anchors in [('development_scores.tsv', 'development', False),
                                     ('assessment_anchors.tsv', 'assessment', True),
                                     ('assessment_scores.tsv', 'assessment', False)]:
        # The design stage never reads full assessment score values.
        if stage == 'design' and name == 'assessment_scores.tsv':
            continue
        source_name = group + '/' + name + '.gz'
        record = manifest['files'][source_name]
        payload = gzip.decompress((root / source_name).read_bytes())
        require(hashlib.sha256(payload).hexdigest() == record['export_sha256'] and
                len(payload) == record['export_bytes'], 'Expanded source binding differs: ' + name)
        tables[name] = payload
    config = (json.dumps(study['configuration'], indent=2, sort_keys=True) + '\n').encode()
    development = _scores(tables['development_scores.tsv'], study, 'development')
    anchors = _scores(tables['assessment_anchors.tsv'], study, 'assessment', True)
    if stage != 'design':
        assessment = _scores(tables['assessment_scores.tsv'], study, 'assessment')
        require(all(assessment[key] == value for key, value in anchors.items()), 'Anchor scores differ')
    design_names = ('development_scores.tsv', 'assessment_anchors.tsv')
    if stage == 'assessment':
        require(output.is_dir(), 'Export design inputs first')
        require((output / 'manifest.json').read_bytes() == config, 'Previously exported manifest changed')
        for name in design_names:
            require((output / name).read_bytes() == tables[name], 'Previously exported design input changed')
        require(not (output / 'assessment_scores.tsv').exists(), 'Assessment scores already exported')
        names = ('assessment_scores.tsv',)
    else:
        require(not output.exists(), 'Output already exists; choose a fresh directory')
        names = design_names if stage == 'design' else (*design_names, 'assessment_scores.tsv')
        output.mkdir(parents=True)
        (output / 'manifest.json').write_bytes(config)
    for name in names:
        with (output / name).open('xb') as stream:
            stream.write(tables[name])
    return study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=ROOT)
    parser.add_argument('--group', choices=('original8', 'cap48'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('design', 'assessment', 'all'), default='design')
    args = parser.parse_args()
    study = export_inputs(args.output, args.group, args.stage, args.source_dir)
    print(json.dumps({'group': args.group, 'stage': args.stage,
                      'development_cases': len(study['development_cases']),
                      'assessment_cases': len(study['assessment_cases'])}, sort_keys=True))


if __name__ == '__main__':
    main()
