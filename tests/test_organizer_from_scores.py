"""Protocol allocation scores can feed an independently declared organizer study."""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from reference_design.organizer import design, check_design
from reference_design.protocol import run_plan


SCRIPT = Path(__file__).parents[1]/'examples/organizer/from_scores.py'
SPEC = importlib.util.spec_from_file_location('organizer_from_scores', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_rows(path, values):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(values)


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


@pytest.fixture
def protocol_scores(tmp_path):
    references = []
    for depth, allocations in [(1, [([1, 2], [3, 4]), ([2, 1], [4, 3])]),
                               (2, [([0, 0], [1, 1]), ([.5, -.5], [1.5, .5])])]:
        for allocation, blocks in enumerate(allocations):
            for block, values in zip(['B1', 'B2'], blocks):
                references.append(dict(allocation=allocation, depth=depth, block=block,
                                       cells_per_block=depth, g1=values[0], g2=values[1]))
    write_rows(tmp_path/'references.tsv', references)
    for name, values in [('effect', [2, 2]), ('state', [4, 5])]:
        write_rows(tmp_path/f'{name}.tsv', [dict(gene=f'g{i + 1}', value=value) for i, value in enumerate(values)])
    models = [dict(name='native_effect', kind='effect', prediction='effect.tsv'),
              dict(name='fixed_state', kind='state', prediction='state.tsv')]
    tasks = []
    for index in range(4):
        write_rows(tmp_path/f'treated_{index}.tsv', [dict(gene='g1', value=5 + index), dict(gene='g2', value=6 + index)])
        tasks.append(dict(id=f'task_{index}', unit=f'unit_{index}', treated=f'treated_{index}.tsv',
                          references='references.tsv', models=models))
    plan = dict(target='heldout_effect', prediction_controls='available', tasks=tasks)
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    run_plan(tmp_path/'plan.json', tmp_path/'protocol')
    config = dict(
        schema_version=1, target='heldout_effect', metric=dict(name='MSE', direction='lower'),
        models=['native_effect', 'fixed_state'],
        budgets=[dict(id=name, control_cells_available=2 * depth, input_cells=depth,
                      observation_cells=depth, description=f'Two blocks of {depth} cells.')
                 for depth, name in [(1, 'small'), (2, 'large')]],
        roles=dict(model_input='Fixed supplied predictions.', prediction_centring='B1 for state predictions.',
                   observation_centring='B2 for every prediction.', participant_information='Supplied prediction files.',
                   evaluator_information='Treated vectors and reference means.'),
        anchor_realization='0', minimum_gap=.1, numerical_tolerance=1e-12,
        evaluation_scope='Four synthetic tasks and two recorded allocations.',
        development_cases=['task_0', 'task_1'], assessment_cases=['task_2', 'task_3'],
        budget_by_depth={'1': 'small', '2': 'large'},
    )
    config_path = tmp_path/'organizer_config.json'
    config_path.write_text(json.dumps(config))
    return tmp_path/'protocol/primary_allocation_scores.tsv', config_path


def test_protocol_scores_export_design_and_check(tmp_path, protocol_scores):
    scores_path, config_path = protocol_scores
    output = tmp_path/'organizer_inputs'
    result = subprocess.run([sys.executable, str(SCRIPT), '--scores', str(scores_path),
                             '--config', str(config_path), '--output', str(output)],
                            env=dict(os.environ, PYTHONPATH=str(SCRIPT.parents[2]/'src')),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert set(path.name for path in output.iterdir()) == {
        'manifest.json', 'development_scores.tsv', 'assessment_anchors.tsv', 'assessment_scores.tsv', 'receipt.json'}
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['roles'] == json.loads(config_path.read_text())['roles']
    assert 'geometry' not in manifest and 'geometry_tolerance' not in manifest
    converted = rows(output/'development_scores.tsv') + rows(output/'assessment_scores.tsv')
    values = {(r['case'], r['budget'], r['realization'], r['model']): float(r['score']) for r in converted}
    assert len(values) == 4 * 2 * 2 * 2
    # Independently enumerated squared residuals, before any organizer calculation.
    assert values[('task_0', 'small', '0', 'native_effect')] == 0
    assert values[('task_0', 'small', '1', 'native_effect')] == 1
    assert values[('task_0', 'small', '0', 'fixed_state')] == 1
    assert values[('task_0', 'large', '0', 'native_effect')] == 6.5
    assert values[('task_2', 'small', '0', 'native_effect')] == 4
    assert {r['unit'] for r in converted if r['case'] == 'task_2'} == {'unit_2'}
    anchors = rows(output/'assessment_anchors.tsv')
    assert {r['case'] for r in anchors} == {'task_2', 'task_3'}
    assert {r['realization'] for r in anchors} == {'0'}
    receipt = json.loads((output/'receipt.json').read_text())
    assert receipt['model_kinds'] == {'native_effect': 'effect', 'fixed_state': 'state'}
    assert receipt['budget_by_depth'] == {'1': 'small', '2': 'large'}
    for name, path in [('scores', scores_path), ('configuration', config_path)]:
        assert receipt['inputs'][name]['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    for name, digest in receipt['outputs'].items():
        assert digest == hashlib.sha256((output/name).read_bytes()).hexdigest()
    frozen = design(output/'manifest.json', tmp_path/'design')
    assert {r['budget']: r['B'] for r in frozen['calibration']} == {'small': 1, 'large': .75}
    checked = check_design(tmp_path/'design/design.json', output/'assessment_scores.tsv', tmp_path/'check')
    assert len(checked['summaries']) == 4
    assert all(r['released'] == 2 and r['unsupported_releases'] == 0 for r in checked['summaries'])
    assert all(r['budget'] == 'small' and r['model'] == 'fixed_state' for r in checked['budget_assessments'])


@pytest.mark.parametrize('failure', ['overlap', 'omitted_case', 'unmapped_depth', 'duplicate_budget_mapping',
                                     'unknown_config', 'wrong_metric', 'inconsistent_kind', 'inconsistent_unit',
                                     'duplicate_score', 'missing_model', 'missing_depth', 'missing_realization',
                                     'different_realization_support', 'missing_anchor', 'nonfinite_score',
                                     'negative_score', 'extra_column'])
def test_invalid_exports_publish_nothing(tmp_path, protocol_scores, failure):
    scores_path, config_path = protocol_scores
    config, data = json.loads(config_path.read_text()), rows(scores_path)
    if failure == 'overlap':
        config['assessment_cases'].append('task_0')
    elif failure == 'omitted_case':
        config['assessment_cases'].pop()
    elif failure == 'unmapped_depth':
        config['budget_by_depth'] = {'1': 'small', '3': 'large'}
    elif failure == 'duplicate_budget_mapping':
        config['budget_by_depth']['2'] = 'small'
    elif failure == 'unknown_config':
        config['geometry'] = 'unused.tsv'
    elif failure == 'wrong_metric':
        config['metric']['direction'] = 'higher'
    elif failure == 'inconsistent_kind':
        data[0]['kind'] = 'state'
    elif failure == 'inconsistent_unit':
        data[0]['unit'] = 'another_unit'
    elif failure == 'duplicate_score':
        data.append(data[0])
    elif failure == 'missing_model':
        data.pop()
    elif failure == 'missing_depth':
        data = [r for r in data if not (r['task'] == 'task_3' and r['depth'] == '2')]
    elif failure == 'missing_realization':
        data = [r for r in data if not (r['task'] == 'task_3' and r['depth'] == '2' and r['allocation'] == '1')]
    elif failure == 'different_realization_support':
        for row in data:
            if row['task'] == 'task_3' and row['allocation'] == '1':
                row['allocation'] = '2'
    elif failure == 'missing_anchor':
        config['anchor_realization'] = 'unrecorded'
    elif failure == 'nonfinite_score':
        data[0]['MSE'] = 'nan'
    elif failure == 'negative_score':
        data[0]['MSE'] = '-1'
    else:
        data = [dict(row, extra='unrecognized') for row in data]
    config_path.write_text(json.dumps(config))
    write_rows(scores_path, data)
    with pytest.raises(ValueError):
        MODULE.from_scores(scores_path, config_path, tmp_path/'out')
    assert not (tmp_path/'out').exists()
