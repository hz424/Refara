"""Convert allocation-level protocol MSE scores into declared organizer inputs."""
import argparse
import hashlib
from pathlib import Path
import tempfile

from reference_design.cli import _destination, _publish, _schema, _tsv
from reference_design.io import label, table
from reference_design.organizer import (
    BASE_POLICIES, CONFIG_FIELDS, _configuration, _digest, _encoded, _json, _scores,
)


def from_scores(scores_path, config_path, output_path):
    scores_path, config_path, output = Path(scores_path), Path(config_path), Path(output_path)
    _destination(output)
    spec = _json(config_path)
    split_fields = {'development_cases', 'assessment_cases', 'budget_by_depth'}
    _schema(spec, CONFIG_FIELDS | split_fields, set(), 'Score conversion configuration')
    config = _configuration({**{key: spec[key] for key in CONFIG_FIELDS}, 'policies': list(BASE_POLICIES)})
    if config['metric'] != {'name': 'MSE', 'direction': 'lower'}:
        raise ValueError('Protocol scores require metric name MSE and direction lower')
    splits = {}
    for field in ('development_cases', 'assessment_cases'):
        values = spec[field]
        if not isinstance(values, list) or not values:
            raise ValueError(f'{field} must be a nonempty list')
        for value in values:
            label(value, field)
        if len(set(values)) != len(values):
            raise ValueError(f'Duplicate case in {field}')
        splits[field] = set(values)
    development, assessment = splits['development_cases'], splits['assessment_cases']
    if development & assessment:
        raise ValueError('Development and assessment cases must be disjoint')
    mapping = spec['budget_by_depth']
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError('budget_by_depth must be a nonempty object')
    for depth, budget in mapping.items():
        label(depth, 'Depth')
        label(budget, 'Mapped budget')
    if (set(mapping.values()) != {budget['id'] for budget in config['budgets']}
            or len(set(mapping.values())) != len(mapping)):
        raise ValueError('budget_by_depth must map one-to-one onto every declared budget')
    headers, source_rows = table(scores_path)
    if set(headers) != {'task', 'unit', 'depth', 'allocation', 'model', 'kind', 'MSE'}:
        raise ValueError('Expected primary_allocation_scores.tsv columns: task, unit, depth, allocation, model, kind, MSE')
    if {row['task'] for row in source_rows} != development | assessment:
        raise ValueError('Development and assessment cases must cover exactly the source tasks')
    if {row['depth'] for row in source_rows} != set(mapping):
        raise ValueError('budget_by_depth must cover exactly the source depths')
    converted, kinds = [], {}
    for row in source_rows:
        if row['kind'] not in ('state', 'effect'):
            raise ValueError('Model kind must be state or effect')
        if kinds.setdefault(row['model'], row['kind']) != row['kind']:
            raise ValueError('Each model must have one consistent state/effect kind')
        converted.append(dict(case=row['task'], unit=row['unit'], budget=mapping[row['depth']],
                              realization=row['allocation'], model=row['model'], score=row['MSE']))
    if config['target'] == 'treated_state' and 'effect' in kinds.values():
        raise ValueError('Treated-state scores require state predictions')
    columns = ['case', 'unit', 'budget', 'realization', 'model', 'score']
    development_rows = [row for row in converted if row['case'] in development]
    assessment_rows = [row for row in converted if row['case'] in assessment]
    contents = {
        'development_scores.tsv': _tsv(columns, development_rows),
        'assessment_scores.tsv': _tsv(columns, assessment_rows),
        'assessment_anchors.tsv': _tsv(columns, [row for row in assessment_rows
                                               if row['realization'] == config['anchor_realization']]),
    }
    with tempfile.TemporaryDirectory(prefix='organizer-scores-') as temporary:
        staging = Path(temporary)
        for name, data in contents.items():
            (staging/name).write_bytes(data)
        cases = {**_scores(staging/'development_scores.tsv', config),
                 **_scores(staging/'assessment_scores.tsv', config)}
        _scores(staging/'assessment_anchors.tsv', config, anchor_only=True)
    support = None
    for case in cases.values():
        actual = {(budget, realization) for budget, values in case['scores'].items() for realization in values}
        if support is not None and actual != support:
            raise ValueError('All source tasks must have the same budget and realization support')
        support = actual
        if any(score < 0 for values in case['scores'].values() for models in values.values() for score in models.values()):
            raise ValueError('MSE scores must be nonnegative')
    manifest = {key: spec[key] for key in CONFIG_FIELDS}
    manifest.update(development_scores='development_scores.tsv', assessment_anchors='assessment_anchors.tsv')
    contents['manifest.json'] = _encoded(manifest)
    receipt = dict(
        schema_version=1,
        inputs={name: dict(name=path.name, sha256=_digest(path))
                for name, path in [('scores', scores_path), ('configuration', config_path)]},
        column_mapping=dict(task='case', unit='unit', depth='budget', allocation='realization', model='model', MSE='score'),
        budget_by_depth=mapping, model_kinds=kinds,
        development_cases=spec['development_cases'], assessment_cases=spec['assessment_cases'],
        outputs={name: hashlib.sha256(data).hexdigest() for name, data in contents.items()},
    )
    contents['receipt.json'] = _encoded(receipt)
    _publish(output, contents)
    return output/'manifest.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        print(from_scores(args.scores, args.config, args.output))
    except (ValueError, OSError) as error:
        parser.exit(2, f'{error}\n')
