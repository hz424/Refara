"""Organizer calibration and independent-check behavior on synthetic cases."""
import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from reference_design.organizer import design, check_design


def example(tmp_path):
    path = Path(__file__).parents[1]/'examples/organizer/make_example.py'
    spec = importlib.util.spec_from_file_location('organizer_example', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_example(tmp_path/'inputs')


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def write_rows(path, values):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(values)


def summary(result, budget, policy):
    return next(r for r in result['summaries'] if r['budget'] == budget and r['policy'] == policy)


def test_frozen_design_and_independent_check_report_real_failures(tmp_path):
    manifest = example(tmp_path)
    artifact = design(manifest, tmp_path/'design')
    assert len(artifact['calibration']) == 6
    pair = next(r for r in artifact['calibration'] if r['budget'] == 'small' and r['model_b'] == 'model_b')
    assert pair['B'] == 2 and pair['K'] == 2 and pair['geometry_available']
    chosen = next(r for r in artifact['budget_recommendations'] if r['case'] == 'reversal' and r['policy'] == 'repeated_primary')
    assert chosen['budget'] == 'large' and chosen['model'] == 'model_a'
    assert len(artifact['decisions']) == 54
    report = (tmp_path/'design/report.md').read_text()
    assert 'provisional' in report and 'not confidence bounds' in report
    assert str(tmp_path.resolve()) not in report
    result = check_design(tmp_path/'design/design.json', manifest.parent/'assessment.tsv', tmp_path/'check')
    repeated = summary(result, 'small', 'repeated_primary')
    geometry = summary(result, 'small', 'geometry_scaled')
    assert (repeated['released'], repeated['unsupported_releases']) == (8, 1)
    assert (geometry['released'], geometry['unsupported_releases']) == (9, 2)
    assert geometry['strict_reversals'] == 1 and geometry['within_gap'] == 1
    assert result['geometry_vs_repeated'][0]['outcome'] == 'no_gain'
    assert result['geometry_vs_repeated'][1]['outcome'] == 'parity'
    budgets = {(r['case'], r['policy']): r for r in result['budget_assessments']}
    assert budgets[('stable', 'geometry_scaled')]['supported'] is True
    failure = budgets[('reversal', 'geometry_scaled')]
    assert failure['budget'] == 'small' and failure['unsupported_recommendation']
    assert failure['opponents'] == 2 and failure['supported_opponents'] == 1
    assert failure['strict_reversal'] and failure['realizations'] == 3
    assert (tmp_path/'check/budget_assessments.tsv').is_file()
    assert json.loads((tmp_path/'design/design.json').read_text()) == artifact
    assert json.loads((tmp_path/'check/check.json').read_text()) == result
    assert not any('assessment_scores' in item for item in artifact['inputs'])
    # A frozen design and the complete check table suffice after development inputs move.
    for name in ('development.tsv', 'anchors.tsv', 'geometry.tsv', 'manifest.json'):
        (manifest.parent/name).unlink()
    second = check_design(tmp_path/'design/design.json', manifest.parent/'assessment.tsv', tmp_path/'second_check')
    assert second == result


@pytest.mark.parametrize('failure', ['duplicate_score', 'missing_model', 'missing_budget', 'missing_realization',
                                     'nonfinite_score', 'bad_unit', 'overlap', 'extra_geometry', 'missing_geometry',
                                     'duplicate_geometry', 'negative_geometry', 'check_nonanchor', 'unknown_field',
                                     'nan_config', 'duplicate_json', 'over_budget'])
def test_invalid_design_inputs_publish_nothing(tmp_path, failure):
    manifest_path = example(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    development_path = manifest_path.parent/'development.tsv'
    development = rows(development_path)
    geometry_path = manifest_path.parent/'geometry.tsv'
    geometry = rows(geometry_path)
    if failure == 'duplicate_score':
        development.append(development[0])
    elif failure == 'missing_model':
        development.pop()
    elif failure == 'missing_budget':
        development = [r for r in development if not (r['case'] == 'dev_1' and r['budget'] == 'large')]
    elif failure == 'missing_realization':
        development = [r for r in development if not (r['case'] == 'dev_1' and r['budget'] == 'large' and r['realization'] == 'repeat_2')]
    elif failure == 'nonfinite_score':
        development[0]['score'] = 'nan'
    elif failure == 'bad_unit':
        development[0]['unit'] = 'another_unit'
    elif failure == 'overlap':
        for row in development:
            if row['case'] == 'dev_1':
                row['case'] = 'stable'
    elif failure == 'extra_geometry':
        geometry.append(dict(case='extra', unit='synthetic_unit', budget='small', geometry='1'))
    elif failure == 'missing_geometry':
        geometry.pop()
    elif failure == 'duplicate_geometry':
        geometry.append(geometry[0])
    elif failure == 'negative_geometry':
        geometry[0]['geometry'] = '-1'
    elif failure == 'check_nonanchor':
        path = manifest_path.parent/'anchors.tsv'
        anchors = rows(path)
        anchors[0]['realization'] = 'repeat_1'
        write_rows(path, anchors)
    elif failure == 'unknown_field':
        manifest['confidence'] = .95
    elif failure == 'nan_config':
        manifest['minimum_gap'] = float('nan')
    elif failure == 'over_budget':
        manifest['budgets'][0]['input_cells'] = 51
    write_rows(development_path, development)
    write_rows(geometry_path, geometry)
    text = json.dumps(manifest)
    if failure == 'duplicate_json':
        text = text[:-1] + ', "schema_version": 1}'
    manifest_path.write_text(text)
    with pytest.raises(ValueError):
        design(manifest_path, tmp_path/'out')
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('development_zero', [False, True])
def test_zero_geometry_never_implies_zero_risk(tmp_path, development_zero):
    manifest = example(tmp_path)
    path = manifest.parent/'geometry.tsv'
    geometry = rows(path)
    for row in geometry:
        if row['case'] == ('dev_1' if development_zero else 'stable'):
            row['geometry'] = '0'
    write_rows(path, geometry)
    artifact = design(manifest, tmp_path/'design')
    decisions = [r for r in artifact['decisions'] if r['policy'] == 'geometry_scaled' and r['model_b'] == 'model_b']
    if development_zero:
        assert all(r['decision'] == 'hold' and r['reason'] == 'development_zero_geometry_shift' for r in decisions)
    else:
        assert all(r['decision'] == 'hold' and r['reason'] == 'assessment_geometry_at_or_below_tolerance'
                   for r in decisions if r['case'] == 'stable')


def test_all_zero_geometry_has_no_geometry_coefficient(tmp_path):
    manifest = example(tmp_path)
    path = manifest.parent/'geometry.tsv'
    geometry = rows(path)
    for row in geometry:
        row['geometry'] = '0'
    write_rows(path, geometry)
    artifact = design(manifest, tmp_path/'design')
    assert all(row['K'] is None and not row['geometry_available'] for row in artifact['calibration'])
    result = check_design(tmp_path/'design/design.json', manifest.parent/'assessment.tsv', tmp_path/'check')
    assert summary(result, 'small', 'geometry_scaled')['unsupported_fraction'] is None
    assert all(row['outcome'] == 'insufficient_releases' for row in result['geometry_vs_repeated'])


@pytest.mark.parametrize('keep_tolerance', [False, True])
def test_omitting_geometry_preserves_single_split_and_repeated_primary(tmp_path, keep_tolerance):
    manifest_path = example(tmp_path)
    original = design(manifest_path, tmp_path/'original')
    original_check = check_design(tmp_path/'original/design.json', manifest_path.parent/'assessment.tsv', tmp_path/'original_check')
    # Existing version 1 configurations do not declare a policy list.
    assert 'policies' not in original['configuration']
    manifest = json.loads(manifest_path.read_text())
    del manifest['geometry']
    if not keep_tolerance:
        del manifest['geometry_tolerance']
    manifest_path.write_text(json.dumps(manifest))
    (manifest_path.parent/'geometry.tsv').unlink()
    artifact = design(manifest_path, tmp_path/'design')
    policies = ['single_split', 'repeated_primary']
    assert artifact['configuration']['policies'] == policies
    assert 'geometry' not in artifact['inputs']
    assert all(case['geometry'] is None for case in artifact['cases'])
    assert artifact['decisions'] == [dict(row, geometry=None) for row in original['decisions'] if row['policy'] in policies]
    assert artifact['budget_recommendations'] == [row for row in original['budget_recommendations'] if row['policy'] in policies]
    assert artifact['calibration'] == [dict(row, K=None, geometry_available=False,
                                          geometry_reason='geometry_not_supplied', positive_geometry_cases=0,
                                          zero_geometry_shift_cases=0) for row in original['calibration']]
    assert 'geometry_scaled' not in (tmp_path/'design/report.md').read_text()
    # The new frozen form must also remain sufficient after development data move.
    for name in ('development.tsv', 'anchors.tsv', 'manifest.json'):
        (manifest_path.parent/name).unlink()
    result = check_design(tmp_path/'design/design.json', manifest_path.parent/'assessment.tsv', tmp_path/'check')
    assert result['geometry_vs_repeated'] == []
    for field in ('summaries', 'assessments', 'budget_assessments'):
        assert result[field] == [row for row in original_check[field] if row['policy'] in policies]
    assert summary(result, 'small', 'repeated_primary')['unsupported_releases'] == 1
    assert 'Geometry versus' not in (tmp_path/'check/report.md').read_text()


def test_geometry_requires_its_explicit_tolerance(tmp_path):
    manifest_path = example(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    del manifest['geometry_tolerance']
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='geometry_tolerance is required'):
        design(manifest_path, tmp_path/'out')
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('failure', ['policies_removed', 'geometry_policy_added', 'geometry_injected',
                                     'coefficient_injected', 'reason_changed', 'decision_changed',
                                     'budget_changed', 'geometry_receipt_added', 'unknown_policies'])
def test_geometry_free_frozen_design_rejects_inconsistent_edits(tmp_path, failure):
    manifest_path = example(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    del manifest['geometry']
    manifest_path.write_text(json.dumps(manifest))
    artifact = design(manifest_path, tmp_path/'design')
    if failure == 'policies_removed':
        del artifact['configuration']['policies']
    elif failure == 'geometry_policy_added':
        artifact['decisions'].append(dict(artifact['decisions'][0], policy='geometry_scaled'))
    elif failure == 'geometry_injected':
        artifact['cases'][0]['geometry'] = {'small': 0, 'large': 0}
    elif failure == 'coefficient_injected':
        row = artifact['calibration'][0]
        row.update(K=1, geometry_available=True, positive_geometry_cases=1, geometry_reason='available')
    elif failure == 'reason_changed':
        artifact['calibration'][0]['geometry_reason'] = 'no_positive_development_geometry'
    elif failure == 'decision_changed':
        artifact['decisions'][0]['decision'] = 'hold'
    elif failure == 'budget_changed':
        artifact['budget_recommendations'][0]['model'] = 'model_b'
    elif failure == 'geometry_receipt_added':
        artifact['inputs']['geometry'] = dict(name='geometry.tsv', sha256='0' * 64)
    else:
        artifact['configuration']['policies'] = ['single_split']
    frozen = tmp_path/'design/design.json'
    frozen.write_text(json.dumps(artifact))
    with pytest.raises(ValueError):
        check_design(frozen, manifest_path.parent/'assessment.tsv', tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_all_hold_error_fraction_is_undefined(tmp_path):
    path = example(tmp_path)
    manifest = json.loads(path.read_text())
    manifest['minimum_gap'] = 100
    path.write_text(json.dumps(manifest))
    design(path, tmp_path/'design')
    result = check_design(tmp_path/'design/design.json', path.parent/'assessment.tsv', tmp_path/'check')
    assert all(r['released'] == 0 and r['coverage'] == 0 and r['unsupported_fraction'] is None for r in result['summaries'])
    assert all(r['outcome'] == 'insufficient_releases' for r in result['geometry_vs_repeated'])
    assert 'undefined' in (tmp_path/'check/report.md').read_text()
    assert all(row['decision'] == 'hold' and row['supported'] is None and not row['unsupported_recommendation']
               and row['assessment_reason'] == 'held_by_design' for row in result['budget_assessments'])


@pytest.mark.parametrize('failure', ['anchor_changed', 'extra_case', 'missing_model', 'one_realization', 'unit_changed', 'frozen_decision'])
def test_invalid_independent_check_publishes_nothing(tmp_path, failure):
    manifest = example(tmp_path)
    design(manifest, tmp_path/'design')
    path = manifest.parent/'assessment.tsv'
    data = rows(path)
    if failure == 'anchor_changed':
        data[0]['score'] = '2'
    elif failure == 'extra_case':
        data.extend([dict(row, case='extra') for row in data if row['case'] == 'stable'])
    elif failure == 'missing_model':
        data.pop()
    elif failure == 'one_realization':
        data = [row for row in data if row['realization'] == 'anchor']
    elif failure == 'unit_changed':
        for row in data:
            row['unit'] = 'changed'
    else:
        frozen = tmp_path/'design/design.json'
        artifact = json.loads(frozen.read_text())
        artifact['decisions'][0]['decision'] = 'hold'
        frozen.write_text(json.dumps(artifact))
    write_rows(path, data)
    with pytest.raises(ValueError):
        check_design(tmp_path/'design/design.json', path, tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_higher_metric_preserves_decisions_after_sign_conversion(tmp_path):
    manifest_path = example(tmp_path)
    lower = design(manifest_path, tmp_path/'lower')
    manifest = json.loads(manifest_path.read_text())
    manifest['metric']['direction'] = 'higher'
    manifest_path.write_text(json.dumps(manifest))
    for name in ('development.tsv', 'anchors.tsv', 'assessment.tsv'):
        path = manifest_path.parent/name
        data = rows(path)
        for row in data:
            row['score'] = str(-float(row['score']))
        write_rows(path, data)
    higher = design(manifest_path, tmp_path/'higher')
    assert lower['decisions'] == higher['decisions']
    result = check_design(tmp_path/'higher/design.json', manifest_path.parent/'assessment.tsv', tmp_path/'check')
    assert result['geometry_vs_repeated'][0]['outcome'] == 'no_gain'


def test_ties_count_as_unsupported_without_strict_reversal(tmp_path):
    manifest = example(tmp_path)
    design(manifest, tmp_path/'design')
    path = manifest.parent/'assessment.tsv'
    data = rows(path)
    for row in data:
        if row['case'] == 'reversal' and row['model'] == 'model_b' and row['realization'] != 'anchor':
            row['score'] = '1'
    write_rows(path, data)
    result = check_design(tmp_path/'design/design.json', path, tmp_path/'check')
    geometry = summary(result, 'small', 'geometry_scaled')
    assert geometry['unsupported_releases'] == 2 and geometry['numerical_ties'] == 1
    assert geometry['strict_reversals'] == 0


def test_geometry_can_improve_an_all_hold_baseline(tmp_path):
    manifest_path = example(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest['models'] = ['model_a', 'model_b']
    manifest_path.write_text(json.dumps(manifest))
    for name in ('development.tsv', 'anchors.tsv', 'assessment.tsv'):
        path = manifest_path.parent/name
        data = [row for row in rows(path) if row['model'] != 'model_c']
        for row in data:
            if row['model'] == 'model_b':
                row['score'] = '2.5' if not row['case'].startswith('dev_') or row['realization'] == 'anchor' else '4.5'
        write_rows(path, data)
    path = manifest_path.parent/'geometry.tsv'
    data = rows(path)
    for row in data:
        if not row['case'].startswith('dev_'):
            row['geometry'] = '.1'
    write_rows(path, data)
    design(manifest_path, tmp_path/'design')
    result = check_design(tmp_path/'design/design.json', manifest_path.parent/'assessment.tsv', tmp_path/'check')
    assert summary(result, 'small', 'repeated_primary')['released'] == 0
    assert summary(result, 'small', 'geometry_scaled')['released'] == 3
    assert all(row['outcome'] == 'strict_gain' for row in result['geometry_vs_repeated'])


def test_cli_outputs_are_identical_across_python_hash_seeds(tmp_path):
    manifest = example(tmp_path)
    outputs = []
    source = str(Path(__file__).parents[1]/'src')
    for seed in ('1', '817'):
        output = tmp_path/f'design_{seed}'
        environment = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=source, PYTHONDONTWRITEBYTECODE='1')
        subprocess.run([sys.executable, '-m', 'reference_design', 'design', str(manifest), '--output', str(output)],
                       env=environment, check=True, capture_output=True, text=True)
        outputs.append({path.name: path.read_bytes() for path in output.iterdir()})
    assert outputs[0] == outputs[1]
