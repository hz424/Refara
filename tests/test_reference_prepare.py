"""Preparation checks using independent synthetic cells and predictions."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from reference_design.core import allocate_controls
from reference_design.prepare import prepare_plan
from reference_design.protocol import run_plan


def example(tmp_path, format='tsv'):
    path = Path(__file__).parents[1]/'examples/anndata_prepare/make_example.py'
    spec = importlib.util.spec_from_file_location('prepare_example', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_example(tmp_path/'inputs', format)


def test_two_block_sampling_and_default_compatibility():
    cells = np.arange(30).reshape(10, 3)
    ids = [f'c{i}' for i in range(10)]
    two = allocate_controls(cells, ids, 2, 3, blocks=2)
    chosen = np.random.default_rng(3).permutation(np.arange(10))[:4].reshape(2, 2)
    np.testing.assert_array_equal(two.means, np.stack([cells[chosen[:, i]].mean(axis=0) for i in range(2)]))
    old = allocate_controls(cells, ids, 2, 3)
    explicit = allocate_controls(cells, ids, 2, 3, blocks=3)
    np.testing.assert_array_equal(old.means, explicit.means)
    assert old.membership == explicit.membership
    assert len({r['cell_id'] for r in two.membership}) == 4


def test_tsv_prepares_portable_runnable_plan(tmp_path):
    path = example(tmp_path)
    receipt = prepare_plan(path, tmp_path/'prepared')
    assert len(receipt['tasks']) == 3
    assert receipt['selected_cells'] == 18
    cells = receipt['cells']
    assert cells['format'] == 'tsv'
    assert cells['gene_columns'] == ['g1', 'g2', 'g3']
    assert cells['cell_id_column'] == 'cell_id'
    assert cells['unit_column'] == 'donor'
    assert cells['condition_column'] == 'condition'
    assert cells['context_columns'] == ['cell_type']
    assert cells['strata_columns'] == ['batch']
    assert 'layer' not in cells and 'path' not in cells
    assert cells['input_sha256'] == hashlib.sha256((path.parent/'cells.tsv').read_bytes()).hexdigest()
    assert receipt['models'][0]['predictions'] == {
        'format': 'tsv', 'task_column': 'task', 'gene_columns': ['g1', 'g2', 'g3'],
        'input_sha256': hashlib.sha256((path.parent/'effect.tsv').read_bytes()).hexdigest(),
    }
    assert json.loads((tmp_path/'prepared/prepare_receipt.json').read_text()) == receipt
    preview = (tmp_path/'prepared/preview.md').read_text()
    assert 'Cell input: tsv; cell ID column: cell_id; gene columns: g1, g2, g3.' in preview
    assert 'unit = donor; condition = condition; context = cell_type; strata = batch.' in preview
    assert '| effect | predictions | tsv | task column: task; gene columns: g1, g2, g3 |' in preview
    assert 'layer:' not in preview
    assert str(tmp_path.resolve()) not in preview
    plan = json.loads((tmp_path/'prepared/plan.json').read_text())
    with (tmp_path/'prepared'/plan['tasks'][0]['treated']).open() as stream:
        target = [float(r['value']) for r in csv.DictReader(stream, delimiter='\t')]
    assert target == [2, 3, 4]
    result = run_plan(tmp_path/'prepared/plan.json', tmp_path/'results')
    assert not result['diagnostics_emitted']
    assert all('verified' in t['references'] for t in result['tasks'])
    for file in receipt['prepared_files']:
        assert (tmp_path/'prepared'/file).is_file()


@pytest.mark.parametrize('failure', ['duplicate_task', 'duplicate_selector', 'missing_prediction', 'missing_gene', 'nonfinite', 'control_target', 'budget', 'unavailable'])
def test_bad_input_has_no_partial_output(tmp_path, failure):
    path = example(tmp_path)
    manifest = json.loads(path.read_text())
    inputs = path.parent
    if failure == 'duplicate_task':
        p=inputs/'tasks.tsv';p.write_text(p.read_text()+p.read_text().splitlines()[1]+'\n')
    elif failure == 'duplicate_selector':
        p=inputs/'tasks.tsv';line=p.read_text().splitlines()[1].replace('a_pert_a','extra');p.write_text(p.read_text()+line+'\n')
    elif failure == 'missing_prediction':
        p=inputs/'effect.tsv';p.write_text('\n'.join(p.read_text().splitlines()[:-1])+'\n')
    elif failure == 'missing_gene':
        p=inputs/'effect.tsv';p.write_text(p.read_text().replace('g3','other_gene'))
    elif failure == 'nonfinite':
        p=inputs/'cells.tsv';p.write_text(p.read_text().replace('1.0', 'nan'))
    elif failure == 'control_target':
        p=inputs/'tasks.tsv';p.write_text(p.read_text().replace('pert_a','control'))
    elif failure == 'budget':
        manifest['allocation']['depths']=[2]
    else:
        manifest['prediction_controls']='unavailable'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        prepare_plan(path, tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_direct_state_uses_no_control_allocation(tmp_path):
    path = example(tmp_path)
    m=json.loads(path.read_text());m.update(target='treated_state', prediction_controls='unavailable');m.pop('allocation')
    m['models'][0].update(kind='state')
    path.write_text(json.dumps(m))
    receipt=prepare_plan(path,tmp_path/'out')
    assert receipt['selected_cells']==6
    plan=json.loads((tmp_path/'out/plan.json').read_text())
    assert all('references' not in t for t in plan['tasks'])


def test_anndata_sparse_layer_and_prediction_matrix(tmp_path):
    ad = pytest.importorskip('anndata')
    from scipy.sparse import csr_matrix
    path = example(tmp_path, 'h5ad')
    manifest = json.loads(path.read_text())
    prediction_path = path.parent/'effect.h5ad'
    predictions = ad.read_h5ad(prediction_path)
    predictions.layers['predicted_effect'] = csr_matrix(predictions.X)
    predictions.X = np.full(predictions.shape, -123.)
    predictions.obs['task_id'] = predictions.obs_names
    predictions.obs_names = [f'row_{i}' for i in range(predictions.n_obs)]
    predictions.write_h5ad(prediction_path)
    manifest['cells']['path'] = str((path.parent/'cells.h5ad').resolve())
    manifest['models'][0]['predictions'].update(
        path=str(prediction_path.resolve()), layer='predicted_effect', task_column='task_id')
    path.write_text(json.dumps(manifest))
    receipt = prepare_plan(path, tmp_path/'out')
    assert receipt['selected_cells'] == 18
    assert receipt['cells']['format'] == 'h5ad'
    assert receipt['cells']['layer'] == 'normalized'
    assert receipt['cells']['unit_column'] == 'donor'
    assert receipt['cells']['condition_column'] == 'condition'
    assert receipt['cells']['context_columns'] == ['cell_type']
    assert receipt['cells']['strata_columns'] == ['batch']
    assert 'path' not in receipt['cells']
    assert receipt['cells']['input_sha256'] == hashlib.sha256((path.parent/'cells.h5ad').read_bytes()).hexdigest()
    assert receipt['models'][0]['predictions'] == {
        'format': 'h5ad', 'layer': 'predicted_effect', 'task_column': 'task_id',
        'input_sha256': hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
    }
    assert receipt['models'][1]['predictions']['layer'] == 'X'
    assert receipt['models'][1]['predictions']['task_column'] == '__index__'
    preview = (tmp_path/'out/preview.md').read_text()
    assert 'Cell input: h5ad; expression layer: normalized; cell IDs: obs_names; gene IDs: var_names.' in preview
    assert '| effect | predictions | h5ad | layer: predicted_effect; task IDs: obs column task_id; gene IDs: var_names |' in preview
    assert '| state | predictions | h5ad | layer: X; task IDs: obs_names; gene IDs: var_names |' in preview
    assert str(tmp_path.resolve()) not in preview
    plan = json.loads((tmp_path/'out/plan.json').read_text())
    with (tmp_path/'out'/plan['tasks'][0]['treated']).open() as stream:
        assert [float(r['value']) for r in csv.DictReader(stream, delimiter='\t')] == [2, 3, 4]
    with (tmp_path/'out'/plan['tasks'][0]['models'][0]['prediction']).open() as stream:
        assert [float(r['value']) for r in csv.DictReader(stream, delimiter='\t')] == [1, 1, 1]


def test_block_predictions_bind_declared_membership(tmp_path):
    path = example(tmp_path)
    prepare_plan(path,tmp_path/'first')
    initial=json.loads((tmp_path/'first/plan.json').read_text())
    manifest=json.loads(path.read_text())
    model=manifest['models'][1]
    model.pop('predictions')
    model['conditioning']='block'
    model['prediction_files']={task['id']:str((tmp_path/'first'/task['references']).resolve()) for task in initial['tasks']}
    model['membership_sha256']={task['id']:hashlib.sha256((tmp_path/'first'/task['membership']).read_bytes()).hexdigest() for task in initial['tasks']}
    path.write_text(json.dumps(manifest))
    receipt = prepare_plan(path, tmp_path/'bound')
    description = receipt['models'][1]
    first_task = initial['tasks'][0]
    assert description['membership_sha256'] == model['membership_sha256']
    assert description['prediction_files'][first_task['id']] == {
        'format': 'tsv', 'index_columns': ['allocation', 'depth', 'block'],
        'gene_columns': ['g1', 'g2', 'g3'],
        'input_sha256': hashlib.sha256((tmp_path/'first'/first_task['references']).read_bytes()).hexdigest(),
    }
    preview = (tmp_path/'bound/preview.md').read_text()
    assert '| state | prediction_files | tsv | index columns: allocation, depth, block; gene columns: g1, g2, g3 |' in preview
    assert str(tmp_path.resolve()) not in preview
    model['membership_sha256'][initial['tasks'][0]['id']]='0'*64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='membership digest'):
        prepare_plan(path,tmp_path/'bad')
    assert not (tmp_path/'bad').exists()
