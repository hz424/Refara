"""Independent residual and hierarchical-weighting checks for declared protocols."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from reference_design.protocol import run_plan


def example(tmp_path):
    script = Path(__file__).parents[1]/'examples/reference_protocol/make_example.py'
    spec = importlib.util.spec_from_file_location('protocol_example', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_example(tmp_path/'inputs')


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def save(path, plan):
    path.write_text(json.dumps(plan))


def test_independent_scores_and_unbalanced_units(tmp_path):
    plan = example(tmp_path)
    output = tmp_path/'result'
    receipt = run_plan(plan, output)
    # Direct residual enumeration, independent of protocol/core helpers.
    treated = np.array([[5, 6], [3, 4], [7, 8]])
    b1 = np.array([[1, 2], [2, 1]])
    b2 = np.array([[3, 4], [4, 3]])
    expected = {}
    for name in ['native_effect', 'fixed_state', 'conditioned_state']:
        task_scores = []
        for target in treated:
            losses = []
            for first, heldout in zip(b1, b2):
                effect = np.array([2, 2]) if name != 'fixed_state' else np.array([4, 5])-first
                losses.append(np.mean((effect-(target-heldout))**2))
            task_scores.append(np.mean(losses))
        expected[name] = (task_scores[0]+np.mean(task_scores[1:]))/2
    for row in rows(output/'primary_summary_scores.tsv'):
        assert float(row['MSE']) == pytest.approx(expected[row['model']])
    for row in rows(output/'primary_summary_pairs.tsv'):
        assert float(row['margin']) == pytest.approx(expected[row['model_b']]-expected[row['model_a']])
    assert not receipt['diagnostics_emitted']
    assert all('verified' in task['references'] for task in receipt['tasks'])
    assert len(receipt['inputs']) == 10  # plan, controls, membership, references, 3 predictions, 3 targets


def test_state_without_references_and_unavailable_controls(tmp_path):
    path = example(tmp_path)
    plan = json.loads(path.read_text())
    plan.update(target='treated_state', prediction_controls='unavailable')
    for task in plan['tasks']:
        for key in ['references', 'controls', 'membership']:
            task.pop(key)
        task['models'] = [dict(name='a', kind='state', prediction='state.tsv'), dict(name='b', kind='state', prediction='effect.tsv')]
    save(path, plan)
    receipt = run_plan(path, tmp_path/'out')
    assert receipt['roles']['observation'] == 'unused'
    assert {row['depth'] for row in rows(tmp_path/'out/primary_summary_scores.tsv')} == {'none'}


@pytest.mark.parametrize('mutation', ['family', 'membership', 'missing_block_predictions', 'unavailable', 'provenance'])
def test_rejects_before_publication(tmp_path, mutation):
    path = example(tmp_path)
    plan = json.loads(path.read_text())
    if mutation == 'family':
        plan['tasks'][1]['models'].pop()
    elif mutation == 'membership':
        member = path.parent/'membership.tsv'
        member.write_text(member.read_text().replace('c2', 'c1'))
    elif mutation == 'missing_block_predictions':
        plan['tasks'][0]['models'][0]['conditioning'] = 'block'
    elif mutation == 'unavailable':
        plan['prediction_controls'] = 'unavailable'
    else:
        (path.parent/'provenance.json').write_text(json.dumps({'files': [{'path': 'effect.tsv', 'sha256': '0'*64}]}))
        plan['tasks'][0]['models'][0]['provenance'] = 'provenance.json'
    save(path, plan)
    with pytest.raises(ValueError):
        run_plan(path, tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_provenance_and_no_overwrite(tmp_path):
    path = example(tmp_path)
    plan = json.loads(path.read_text())
    fitted = path.parent/'fitted_state.bin'
    fitted.write_bytes(b'synthetic fitted state')
    digest = hashlib.sha256(fitted.read_bytes()).hexdigest()
    (path.parent/'provenance.json').write_text(json.dumps({'files': [{'path': fitted.name, 'sha256': digest}]}))
    plan['tasks'][0]['models'][0]['provenance'] = 'provenance.json'
    save(path, plan)
    receipt = run_plan(path, tmp_path/'out')
    assert any(item['sha256'] == digest for item in receipt['inputs'])
    before = (tmp_path/'out/receipt.json').read_bytes()
    with pytest.raises(FileExistsError):
        run_plan(path, tmp_path/'out')
    assert (tmp_path/'out/receipt.json').read_bytes() == before


def test_three_block_diagnostic_scores_enumerated(tmp_path):
    import itertools
    path = example(tmp_path)
    plan = json.loads(path.read_text())
    # Use one task so the oracle directly averages allocation and role tuples.
    plan['tasks'] = plan['tasks'][:1]
    task = plan['tasks'][0]
    task.pop('controls')
    task.pop('membership')
    with (path.parent/'references.tsv').open('a') as stream:
        stream.write('0\t1\tB3\t1\t6\t7\n1\t1\tB3\t1\t7\t6\n')
    with (path.parent/'conditioned.tsv').open('a') as stream:
        stream.write('0\t1\tB3\t8\t9\n1\t1\tB3\t9\t8\n')
    save(path, plan)
    output = tmp_path/'out'
    receipt = run_plan(path, output)
    assert receipt['diagnostics_emitted']
    actual = {(r['pattern'], r['model']): float(r['MSE']) for r in rows(output/'diagnostic_summary_scores.tsv')}
    blocks_by_allocation = [np.array([[1, 2], [3, 4], [6, 7]]), np.array([[2, 1], [4, 3], [7, 6]])]
    losses = {}
    for blocks in blocks_by_allocation:
        for o, p, m in itertools.product(range(3), repeat=3):
            pattern = 'S' if o == p == m else 'M' if o == p else 'P' if o == m else 'O' if p == m else 'D'
            for name in ['native_effect', 'fixed_state', 'conditioned_state']:
                if name == 'native_effect':
                    residual = np.array([2, 2])-np.array([5, 6])+blocks[o]
                else:
                    state = np.array([4, 5]) if name == 'fixed_state' else blocks[m]+2
                    residual = state-np.array([5, 6])+blocks[o]-blocks[p]
                losses.setdefault((pattern, name), []).append(np.mean(residual**2))
    assert set(actual) == set(losses)
    for key, values in losses.items():
        assert actual[key] == pytest.approx(np.mean(values))
    assert len(rows(output/'diagnostic_unit_pairs.tsv')) == 15


def test_report_escapes_labels_without_changing_score_identifiers(tmp_path):
    path = example(tmp_path)
    plan = json.loads(path.read_text())
    name = 'native | effect\nsecond line'
    for task in plan['tasks']:
        task['models'][0]['name'] = name
    plan['tasks'][0]['id'] = 'task | first\ncontinued'
    save(path, plan)
    output = tmp_path/'out'
    run_plan(path, output)
    report = (output/'report.md').read_text()
    assert 'native \\| effect<br>second line' in report
    assert 'task \\| first<br>continued' in report
    assert name in {row['model'] for row in rows(output/'primary_summary_scores.tsv')}
    assert '| Reference role | Primary use |' in report
    assert '| Depth | Model a | Model b | Equal-unit margin |' in report


@pytest.mark.parametrize('target', ['heldout_effect', 'shared_effect'])
def test_block_native_effect_primary_uses_input_but_no_prediction_centring(tmp_path, target):
    path = example(tmp_path)
    plan = json.loads(path.read_text())
    plan['target'] = target
    for task in plan['tasks']:
        task['models'][0].update(conditioning='block', prediction='conditioned.tsv')
    save(path, plan)
    output = tmp_path/'out'
    receipt = run_plan(path, output)
    # Supplied delta(B1)=B1+2. Independently enumerate cells, allocations and units.
    inputs = np.array([[1.,2.],[2.,1.]])
    observations = np.array([[3.,4.],[4.,3.]]) if target == 'heldout_effect' else inputs
    targets = np.array([[5.,6.],[3.,4.],[7.,8.]])
    task_risks = [np.mean([np.mean((input_cells+2-y+obs)**2)
                          for input_cells,obs in zip(inputs,observations)]) for y in targets]
    expected = (task_risks[0] + np.mean(task_risks[1:]))/2
    actual = {r['model']:float(r['MSE']) for r in rows(output/'primary_summary_scores.tsv')}
    assert actual['native_effect'] == pytest.approx(expected)
    assert not receipt['diagnostics_emitted']  # Only two blocks are supplied.
    assert receipt['roles']['model'] == 'B1 for block-conditioned predictions'
    assert all(item['evidence'] == {'level':'declaration','verified_input_use':False,'artifacts':{}}
               for task in receipt['tasks'] for item in task['prediction_evidence'])


def test_block_native_effect_three_block_diagnostics_are_descriptive(tmp_path):
    import itertools
    path = example(tmp_path)
    plan = json.loads(path.read_text())
    plan['tasks'] = plan['tasks'][:1]
    task = plan['tasks'][0]
    task.pop('controls'); task.pop('membership')
    task['models'][0].update(conditioning='block', prediction='conditioned.tsv')
    with (path.parent/'references.tsv').open('a') as stream:
        stream.write('0\t1\tB3\t1\t6\t7\n1\t1\tB3\t1\t7\t6\n')
    with (path.parent/'conditioned.tsv').open('a') as stream:
        stream.write('0\t1\tB3\t8\t9\n1\t1\tB3\t9\t8\n')
    save(path, plan)
    output = tmp_path/'out'
    receipt = run_plan(path, output)
    assert receipt['diagnostics_emitted']
    expected = {}
    for blocks in (np.array([[1.,2.],[3.,4.],[6.,7.]]),np.array([[2.,1.],[4.,3.],[7.,6.]])):
        for o,p,m in itertools.product(range(3),repeat=3):
            pattern = 'S' if o == p == m else 'M' if o == p else 'P' if o == m else 'O' if p == m else 'D'
            loss = np.mean((blocks[m]+2-np.array([5.,6.])+blocks[o])**2)
            expected.setdefault(pattern,[]).append(loss)
    scores = {r['pattern']:float(r['MSE']) for r in rows(output/'diagnostic_summary_scores.tsv') if r['model']=='native_effect'}
    for pattern, values in expected.items():
        assert scores[pattern] == pytest.approx(np.mean(values))
    pairs = rows(output/'diagnostic_allocation_pairs.tsv')
    for pair in pairs:
        if pair['model_a'] == 'native_effect':
            assert pair['identity_applicable'] == 'False'
            assert pair['identity_verified'] == pair['predicted_d_D'] == ''
            assert pair['theory'] == 'not_applicable_input_conditioned_effect'
        else:
            assert pair['identity_applicable'] == pair['identity_verified'] == 'True'


@pytest.mark.parametrize('conditioned', [False, True])
def test_report_shows_reference_reversal_and_identity_scope(tmp_path, conditioned):
    (tmp_path/'zero.tsv').write_text('gene\tvalue\ngene1\t0\n')
    (tmp_path/'references.tsv').write_text(
        'allocation\tdepth\tblock\tgene1\n0\t1\tB1\t0\n0\t1\tB2\t1\n0\t1\tB3\t2\n')
    effect = {'name': 'effect', 'kind': 'effect', 'prediction': 'zero.tsv'}
    if conditioned:
        effect.update(conditioning='block', prediction='references.tsv')
    plan = {'target': 'heldout_effect', 'prediction_controls': 'available', 'diagnostics': True,
            'tasks': [{'id': 'task1', 'unit': 'donor1', 'treated': 'zero.tsv',
                       'references': 'references.tsv', 'models': [effect,
                           {'name': 'state', 'kind': 'state', 'prediction': 'zero.tsv'}]}]}
    save(tmp_path/'plan.json', plan)
    run_plan(tmp_path/'plan.json', tmp_path/'results')
    report = (tmp_path/'results/report.md').read_text()
    # For Y=E=M=0 and controls (0,1,2), effect MSE=5/3. The state
    # MSE is zero in S/M and two in P/O/D, so the average ranking reverses.
    # If E(Cmodel)=Cmodel, effect MSE instead is 20/3 in S/P and 14/3
    # in M/O/D. It loses throughout, outside the fixed-effect identity.
    if conditioned:
        expected = '| 1 | effect | state | -6.6666667 | -4.6666667 | -4.6666667 | -2.6666667 | -2.6666667 | same direction | not applicable (input-conditioned effect) |'
    else:
        expected = '| 1 | effect | state | -1.6666667 | -1.6666667 | 0.33333333 | 0.33333333 | 0.33333333 | reversal | passed |'
    assert expected in report
    for filename in ('diagnostic_summary_pairs.tsv', 'diagnostic_summary_scores.tsv',
                     'diagnostic_allocation_pairs.tsv'):
        assert f']({filename})' in report
        assert (tmp_path/'results'/filename).is_file()


@pytest.mark.parametrize('mutation', ['unavailable', 'missing_block', 'non_native_representation'])
def test_invalid_block_effect_fails_before_publication(tmp_path, mutation):
    path = example(tmp_path)
    plan = json.loads(path.read_text())
    for task in plan['tasks']:
        task['models'][0].update(conditioning='block', prediction='conditioned.tsv')
    if mutation == 'unavailable':
        plan['prediction_controls'] = 'unavailable'
    elif mutation == 'non_native_representation':
        for task in plan['tasks']:
            task['models'][0]['representation'] = 'effect'
    else:
        pred = path.parent/'conditioned.tsv'
        lines = pred.read_text().splitlines(True)
        pred.write_text(''.join(lines[:-1]))
    save(path,plan)
    with pytest.raises(ValueError):
        run_plan(path,tmp_path/'out')
    assert not (tmp_path/'out').exists()
