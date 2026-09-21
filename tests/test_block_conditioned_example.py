"""Exercise prediction from recorded input cells through the public preparation path."""
import csv
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from reference_design.prepare import prepare_plan
from reference_design.protocol import run_plan


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


@pytest.fixture
def example(tmp_path, monkeypatch):
    script = Path(__file__).parents[1]/'examples/anndata_prepare/block_conditioned.py'
    monkeypatch.syspath_prepend(str(script.parent))
    spec = importlib.util.spec_from_file_location('block_conditioned_example', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_example(tmp_path/'example')


def test_predictions_and_scores_follow_actual_block_membership(example, tmp_path):
    inputs, root = example.parent, example.parents[1]
    manifest = json.loads(example.read_text())
    plan = json.loads((root/'prepared/plan.json').read_text())
    parameters = json.loads((inputs/'affine_parameters.json').read_text())
    cells = {row['cell_id']: row for row in rows(inputs/'cells.tsv')}
    conditions = {row['task']: row['condition'] for row in rows(inputs/'tasks.tsv')}
    score_rows = rows(root/'results/primary_allocation_scores.tsv')
    for task in plan['tasks']:
        member_rows = rows(root/'prepared'/task['membership'])
        prediction_rows = rows(inputs/manifest['models'][1]['prediction_files'][task['id']])
        observed_predictions = []
        for prediction in prediction_rows:
            key = tuple(prediction[column] for column in ('allocation', 'depth', 'block'))
            members = [row['cell_id'] for row in member_rows
                       if tuple(row[column] for column in ('allocation', 'depth', 'block')) == key]
            selected = np.array([[float(cells[cell][gene]) for gene in parameters['genes']]
                                 for cell in members])
            expected = selected.mean(axis=0)*parameters['gain']
            expected += parameters['offset_by_condition'][conditions[task['id']]]
            observed = np.array([float(prediction[gene]) for gene in parameters['genes']])
            np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-14)
            observed_predictions.append(observed)
            if key[2] == 'B1':
                heldout_ids = [row['cell_id'] for row in member_rows
                               if (row['allocation'], row['depth'], row['block']) == (*key[:2], 'B2')]
                heldout_mean = np.mean([[float(cells[cell][gene]) for gene in parameters['genes']]
                                        for cell in heldout_ids], axis=0)
                treated = np.array([float(row['value']) for row in rows(root/'prepared'/task['treated'])])
                expected_mse = np.mean((observed-selected.mean(axis=0)-(treated-heldout_mean))**2)
                score = next(row for row in score_rows if row['task'] == task['id']
                             and row['model'] == 'affine_state'
                             and (row['allocation'], row['depth']) == key[:2])
                assert float(score['MSE']) == pytest.approx(expected_mse, abs=1e-14)
        assert len(np.unique(observed_predictions, axis=0)) > 1

    # Everything required for re-scoring survives removal of the original inputs.
    shutil.rmtree(inputs)
    shutil.rmtree(root/'allocated')
    portable = tmp_path/'moved_prepared'
    shutil.move(str(root/'prepared'), portable)
    run_plan(portable/'plan.json', tmp_path/'portable_results')
    assert rows(tmp_path/'portable_results/primary_allocation_scores.tsv') == score_rows


@pytest.mark.parametrize('change', ['allocation', 'membership', 'prediction', 'cells'])
def test_changed_input_binding_is_rejected(example, tmp_path, change):
    manifest = json.loads(example.read_text())
    if change == 'allocation':
        manifest['allocation']['seed'] += 1
        example.write_text(json.dumps(manifest))
        error = 'membership digest'
    else:
        if change == 'membership':
            path = example.parents[1]/'allocated/t0_membership.tsv'
        elif change == 'cells':
            path = example.parent/'cells.tsv'
        else:
            path = example.parent/manifest['models'][1]['prediction_files']['a_pert_a']
        # A harmless trailing newline remains valid TSV, but breaks the recorded binding.
        path.write_text(path.read_text()+'\n')
        error = 'Provenance input digest mismatch'
    with pytest.raises(ValueError, match=error):
        prepare_plan(example, tmp_path/'rejected')
    assert not (tmp_path/'rejected').exists()
