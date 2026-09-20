"""Synthetic tests for the portable external reference-design numerical replay."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
from scipy.stats import t


CAPSULE = Path(__file__).resolve().parents[1] / 'evidence' / 'gse181897_reference_design'


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(CAPSULE))
    for name in ('paired_statistics', 'replay', 'build_export'):
        monkeypatch.delitem(sys.modules, name, raising=False)
    import replay
    import build_export
    return replay, build_export


@pytest.fixture
def synthetic(modules, tmp_path):
    replay, exporter = modules
    directory = tmp_path / 'capsule'
    directory.mkdir()
    (directory / 'source_data').mkdir()
    for name in exporter.CODE_FILES:
        shutil.copyfile(CAPSULE / name, directory / name)
    tasks = []
    donor_contexts = [('S001', 'selection', ('B Naive', 'CD4 Naive')),
                      ('S002', 'selection', ('B Naive',)),
                      ('A001', 'assessment', ('B Naive', 'CD4 Naive')),
                      ('A002', 'assessment', ('B Naive',)),
                      ('A003', 'assessment', ('B Naive',)),
                      ('A004', 'assessment', ('B Naive',)),
                      ('A005', 'assessment', ('B Naive',)),
                      ('A006', 'assessment', ('B Naive',))]
    for donor, partition, contexts in donor_contexts:
        for context in contexts:
            for condition in replay.CONDITIONS:
                tasks.append({'donor': donor, 'partition': partition, 'context': context, 'condition': condition,
                              'secondary_32_eligible': 'True', 'control_cells_available': '32',
                              'within_donor_weight': str(1 / (3 * len(contexts)))})
    selection, assessment = [], []
    for task in tasks:
        if task['partition'] == 'selection':
            for budget in (8, 16, 32):
                for rule in replay.RULES + replay.DIAGNOSTICS:
                    selected = 0 if rule == 'shared_all_B' else 1
                    for fi, family in enumerate(replay.FAMILIES):
                        selection.append({key: task[key] for key in ('donor', 'context', 'condition')} |
                                         {'budget': str(budget), 'rule': rule, 'family': family,
                                          'loss': str(int(task['donor'][1:]) + .1 * (fi - selected) ** 2),
                                          'seeds': '1' if fi < 5 else '3'})
        else:
            for deployment_n in (8, 16):
                for fi, family in enumerate(replay.FAMILIES):
                    base = 9 + int(task['donor'][1:])
                    benefit = 1 if family == 'CM' else 0
                    assessment.append({key: task[key] for key in ('donor', 'context', 'condition')} |
                                      {'deployment_n': str(deployment_n), 'family': family,
                                       'loss': str(base - benefit), 'treated_state_loss': str(base + 10 - 2 * benefit),
                                       'seeds': '1' if fi < 5 else '3', 'control_pool': 'P01', 'treated_pool': 'P02'})
    protocol = {'schema': replay.SCHEMA, 'models': list(replay.FAMILIES), 'conditions': list(replay.CONDITIONS),
                'primary_budget': 16, 'secondary_budgets': [8, 32], 'recommended_rule': 'O_half',
                'primary_deployment_n': 8, 'secondary_deployment_n': 16, 'confidence': .95,
                'bootstrap_seed': 20260919, 'bootstrap_resamples': 19999, 'minimum_relative_reduction': .05,
                'max_inference_ratio': 4, 'minimum_inference_donors': 2, 'tie_tolerance': 1e-12,
                'source_frozen_protocol_sha256': '1' * 64,
                'statistics_executed_source_sha256': replay.EXECUTED_SOURCE_SHA256,
                'inference_scope': 'Synthetic donors conditional on fixed pools and selection',
                'primary_endpoint': 'Common independent measured-effect MSE',
                'treated_state_secondary': 'Descriptive common treated-state measurement'}
    replay.write_json(directory / 'model_contract.json', {'schema': 'SYNTHETIC_MODEL_CONTRACT', 'raw_assay_included': False})
    protocol['model_contract'] = {'path': 'model_contract.json', 'sha256': replay.sha256(directory / 'model_contract.json')}
    replay.write_json(directory / 'protocol.json', protocol)
    exporter.write_gzip(directory / 'source_data/tasks.tsv.gz', tasks, exporter.TASK_FIELDS)
    exporter.write_gzip(directory / 'source_data/selection_losses.tsv.gz', selection, exporter.SELECTION_FIELDS)
    exporter.write_gzip(directory / 'source_data/assessment_losses.tsv.gz', assessment, exporter.ASSESSMENT_FIELDS)
    values = replay.analyse(protocol, tasks, selection, assessment)
    replay.write_json(directory / 'source_data/expected.json', replay.expected_projection(values))
    def manifest():
        names = list(exporter.CODE_FILES) + ['protocol.json', 'model_contract.json', 'source_data/tasks.tsv.gz',
                'source_data/selection_losses.tsv.gz', 'source_data/assessment_losses.tsv.gz', 'source_data/expected.json']
        replay.write_json(directory / 'manifest.json', {'schema': 'REFARA_CAPSULE_MANIFEST_V1',
                          'files': {name: replay.sha256(directory / name) for name in names}})
    manifest()
    return directory, protocol, tasks, selection, assessment, manifest


def test_hand_mean_interval_family_choices_and_joint_result(modules, synthetic):
    replay, _ = modules
    _, protocol, tasks, selection, assessment, _ = synthetic
    result = replay.analyse(protocol, tasks, selection, assessment)
    assert result['selection_decisions']['budgets']['16']['shared_all_B']['family'] == 'NC'
    assert result['selection_decisions']['budgets']['16']['O_half']['family'] == 'CM'
    assert result['selection_decisions']['budgets']['16']['O_half']['family_risks']['CM'] == 1.5
    # Equal-donor baseline is mean(10,...,15)=12.5, despite A001 having twice as
    # many task rows. T5 donor values are 0.50, 0.45, ..., 0.25.
    assert result['paired_t']['mean_shared_loss'] == pytest.approx(12.5)
    assert result['paired_t']['relative_loss_reduction'] == pytest.approx(1 / 12.5)
    radius = t.ppf(.975, 5) * .05 * np.sqrt(3.5 / 6)
    assert result['paired_t']['practical_contrast_interval'] == pytest.approx([.375 - radius, .375 + radius])
    assert result['joint_practical_benefit'] == 'supported'
    assert result['joint_adoption'] == 'supported'
    assert result['compute']['ratio'] == .5
    assert len(result['policy_summary']) == 48
    assert len(result['mechanism_diagnostics']) == 12
    assert len(result['all_model_losses']) == 14
    assert next(r['mean_effect_loss'] for r in result['all_model_losses'] if
                r['deployment_n'] == 8 and r['family'] == 'NC') == pytest.approx(12.5)


def test_shared_recommendation_cannot_create_an_advantage(modules, synthetic):
    replay, _ = modules
    _, protocol, tasks, selection, assessment, _ = synthetic
    protocol = dict(protocol, recommended_rule='shared_all_B')
    result = replay.analyse(protocol, tasks, selection, assessment)
    assert result['paired_t']['relative_loss_reduction'] == 0
    assert result['joint_practical_benefit'] == 'insufficient'
    assert result['joint_adoption'] == 'not_supported'


def test_relocated_replay_verifies_expected_summary_and_refuses_overwrite(modules, synthetic, tmp_path):
    replay, _ = modules
    directory = synthetic[0]
    relocated = tmp_path / 'relocated'
    shutil.copytree(directory, relocated)
    output = tmp_path / 'result'
    result = replay.run(relocated, output)
    assert result['published_summary_verified']
    assert (output / 'RESULTS.json').is_file()
    assert (output / 'primary_donor_losses.tsv').is_file()
    assert (output / 'all_model_losses.tsv').is_file()
    with pytest.raises(ValueError, match='already exists'):
        replay.run(relocated, output)


def test_expected_result_is_not_replaced_with_new_computation(modules, synthetic, tmp_path):
    replay, _ = modules
    directory, _, _, _, _, manifest = synthetic
    path = directory / 'source_data/expected.json'
    expected = json.loads(path.read_text())
    expected['joint_adoption'] = 'deliberately_wrong'
    replay.write_json(path, expected)
    manifest()
    with pytest.raises(ValueError, match='Expected value differs'):
        replay.run(directory, tmp_path / 'result')
    assert not (tmp_path / 'result').exists()


def test_tampered_task_losses_are_rejected_by_manifest(modules, synthetic):
    replay, _ = modules
    directory = synthetic[0]
    with (directory / 'source_data/selection_losses.tsv.gz').open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='artifact changed'):
        replay.load_capsule(directory)


def test_manifest_paths_cannot_escape_capsule(modules, synthetic):
    replay, _ = modules
    directory = synthetic[0]
    payload = json.loads((directory / 'manifest.json').read_text())
    payload['files']['../outside'] = '0' * 64
    replay.write_json(directory / 'manifest.json', payload)
    with pytest.raises(ValueError, match='must remain relative'):
        replay.load_capsule(directory)


def test_missing_family_or_donor_overlap_is_rejected(modules, synthetic):
    replay, _ = modules
    _, protocol, tasks, selection, assessment, _ = synthetic
    with pytest.raises(ValueError, match='Assessment task/family coverage'):
        replay.analyse(protocol, tasks, selection, assessment[:-1])
    tasks = [dict(row, donor='S001') if row['donor'] == 'A001' else row for row in tasks]
    with pytest.raises(ValueError, match='Duplicate task|leaks between'):
        replay.analyse(protocol, tasks, selection, assessment)


def test_secondary_subset_requires_metadata_support(modules, synthetic):
    replay, _ = modules
    _, _, tasks, _, _, _ = synthetic
    tasks = [dict(row, secondary_32_eligible='False') if i == 0 else row for i, row in enumerate(tasks)]
    with pytest.raises(ValueError, match='Secondary subset'):
        replay.validate_tasks(tasks)


def test_numeric_expected_comparison_does_not_accept_wrong_ci(modules):
    replay, _ = modules
    with pytest.raises(ValueError, match='numerical value'):
        replay.compare_expected({'ci': [0., 1.]}, {'ci': [0., 2.]})


def test_statistics_relabeling_preserves_loo_and_donor_order(modules):
    _, exporter = modules
    value = {'units': [{'unit_id': 'D004', 'loss': 2}],
             'leave_one_unit_out': [{'omitted_unit': 'D010', 'mean': 3}]}
    result = exporter.recode_statistics(value, {'D004': 'A001', 'D010': 'A002'})
    assert result['units'][0]['unit_id'] == 'A001'
    assert result['leave_one_unit_out'][0]['omitted_unit'] == 'A002'
