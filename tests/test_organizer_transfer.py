"""Donor-transfer scores retain their hierarchy and frozen input boundary."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import runpy
import shutil

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / 'evidence' / 'organizer'
API = runpy.run_path(str(ROOT / 'transfer.py'))


def test_gene_score_matches_scalar_oracle_and_scores_seeds_before_averaging():
    treated = np.array([3., 5.])
    controls = np.array([[0., 1.], [2., 4.], [6., 7.]])
    effects = np.array([[[1., 3.], [2., 4.], [3., 5.]],
                        [[3., 5.], [4., 6.], [5., 7.]],
                        [[8., 2.], [9., 3.], [10., 4.]]])
    scales = np.array([2., 4.])
    weights = np.array([.25, .75])
    expected = []
    for block in range(3):
        other = [j for j in range(3) if j != block]
        total = 0.
        for seed in range(3):
            for gene in range(2):
                observed = treated[gene] - (controls[other[0], gene] + controls[other[1], gene]) / 2
                total += weights[gene] * ((effects[seed, block, gene] - observed) / scales[gene]) ** 2 / 3
        expected.append(total)
    actual = API['score_arrays'](treated, controls, effects, scales, weights)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
    wrong = API['score_arrays'](treated, controls, effects.mean(axis=0, keepdims=True), scales, weights)
    assert np.all(actual > wrong)
    with pytest.raises(ValueError, match='positive scales'):
        API['score_arrays'](treated, controls, effects, np.array([0., 4.]), weights)


@pytest.mark.parametrize('group', ['original8', 'cap48'])
def test_export_preserves_donor_units_and_assessment_boundary(tmp_path, group, monkeypatch):
    output = tmp_path / group
    # Decompression is forbidden for the full assessment during design export.
    forbidden = (ROOT / 'transfer' / group / 'assessment_scores.tsv.gz').read_bytes()
    decompress = gzip.decompress
    def design_only(payload):
        assert payload != forbidden
        return decompress(payload)
    monkeypatch.setattr(gzip, 'decompress', design_only)
    study = API['export_inputs'](output, group, stage='design')
    assert not (output / 'assessment_scores.tsv').exists()
    assert len(study['development_cases']) == 16
    assert len(study['assessment_cases']) == 78
    units = study['units']
    development = {units[c] for c in study['development_cases']}
    assessment = {units[c] for c in study['assessment_cases']}
    assert (len(development), len(assessment)) == (8, 39)
    assert not development & assessment
    frozen = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
    monkeypatch.setattr(gzip, 'decompress', decompress)
    API['export_inputs'](output, group, stage='assessment')
    assert all(hashlib.sha256((output / name).read_bytes()).hexdigest() == value
               for name, value in frozen.items())
    with (output / 'assessment_scores.tsv').open() as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    assert len(rows) == 78 * 30 * 2
    assert {r['unit'] for r in rows} == assessment
    for filename in frozen:
        assert filename in {'manifest.json', 'development_scores.tsv', 'assessment_anchors.tsv'}
    with pytest.raises(ValueError, match='already exported'):
        API['export_inputs'](output, group, stage='assessment')


def test_changed_development_input_blocks_assessment_export(tmp_path):
    output = tmp_path / 'inputs'
    API['export_inputs'](output, 'original8', stage='design')
    with (output / 'development_scores.tsv').open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='Previously exported design input changed'):
        API['export_inputs'](output, 'original8', stage='assessment')
    assert not (output / 'assessment_scores.tsv').exists()


def test_relocated_bundle_authenticates_and_rejects_changed_scores(tmp_path):
    relocated = tmp_path / 'source bundle'
    shutil.copytree(ROOT / 'transfer', relocated)
    API['export_inputs'](tmp_path / 'inputs', 'cap48', source_dir=relocated)
    path = relocated / 'cap48' / 'development_scores.tsv.gz'
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='Source binding differs'):
        API['export_inputs'](tmp_path / 'changed', 'cap48', source_dir=relocated)


def test_membership_and_protocol_records_are_consistent():
    API['authenticate']()
    membership = json.loads((ROOT / 'transfer' / 'membership.json').read_text())
    protocol = ROOT / 'comparison_protocol.json'
    assert membership['comparison_protocol_sha256'] == hashlib.sha256(protocol.read_bytes()).hexdigest()
    assert membership['development_assessment_donor_overlap'] == 0
    assert membership['development_assessment_control_cell_overlap'] == 0
    assert membership['development_selected_control_cells'] == 16 * 24
    assert membership['assessment_selected_control_cells'] == 78 * 24
    assert membership['shared_acquisition_batch_count'] == 8
    assert membership['scoring_validation']['independent_scalar_checks'] == 24
    assert membership['scoring_validation']['maximum_absolute_difference'] < 1e-12
    assert all(item == {'development': 0, 'assessment': 0}
               for item in membership['training_overlap'].values())
