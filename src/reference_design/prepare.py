"""Prepare portable declared-reference plans from explicitly mapped inputs."""
from __future__ import annotations

import hashlib
import html
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from .core import allocate_controls
from .io import label, table, read_vector
from .protocol import run_plan


def prepare_plan(manifest_path, output_path):
    """Prepare existing predictions; never normalize expression or fit a model."""
    from .cli import _schema, _json_object, _json_constant, _destination, _publish, _tsv
    manifest_path, output = Path(manifest_path), Path(output_path)
    _destination(output)
    hashes, contents = {}, {}

    def file_digest(path):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b''):
                digest.update(chunk)
        return digest.hexdigest()

    def source(value, base=None):
        path = ((manifest_path.parent if base is None else base)/label(value, 'Input path')).resolve()
        hashes[str(path)] = file_digest(path)
        return path

    def load_json(path):
        return json.loads(path.read_text(), object_pairs_hook=_json_object, parse_constant=_json_constant)

    spec = load_json(source(manifest_path.name, manifest_path.parent))
    _schema(spec, {'target', 'prediction_controls', 'cells', 'tasks', 'control_value', 'treated_aggregation', 'models'},
            {'allocation', 'diagnostics', 'scales', 'gene_panel'}, 'Preparation manifest')
    cell_spec = spec['cells']
    _schema(cell_spec, {'format', 'path', 'unit_column', 'condition_column', 'strata_columns', 'context_columns', 'normalization'},
            {'layer', 'gene_columns', 'cell_id_column'}, 'Cells')
    label(cell_spec['normalization'], 'Normalization description')
    for key in ('unit_column', 'condition_column'):
        label(cell_spec[key], key)
    for key in ('strata_columns', 'context_columns'):
        if not isinstance(cell_spec[key], list) or any(not isinstance(x, str) for x in cell_spec[key]):
            raise ValueError(f'{key} must be a list of column names')
        for item in cell_spec[key]:
            label(item, key)
        if len(set(cell_spec[key])) != len(cell_spec[key]):
            raise ValueError(f'{key} has duplicate columns')
    context = cell_spec['context_columns']
    if set(context) & {'task', 'unit', 'condition'}:
        raise ValueError('Context column names cannot be task, unit or condition')
    metadata = list(dict.fromkeys([cell_spec['unit_column'], cell_spec['condition_column'], *context, *cell_spec['strata_columns']]))
    if cell_spec['unit_column'] == cell_spec['condition_column']:
        raise ValueError('Unit and condition columns must differ')
    explicit_panel = spec.get('gene_panel')
    if explicit_panel is not None:
        if not isinstance(explicit_panel, list) or not explicit_panel:
            raise ValueError('gene_panel must be a nonempty gene-name list')
        for gene in explicit_panel:
            label(gene, 'Gene panel')
        if len(set(explicit_panel)) != len(explicit_panel):
            raise ValueError('Duplicate gene panel identifiers')

    def strings(values, field):
        result = tuple(label(value, field + '; explicitly convert numeric/missing annotations to meaningful strings') for value in values)
        return result

    def anndata(path):
        try:
            import anndata as ad
        except ImportError as error:
            raise ValueError('H5AD inputs require the optional empirical dependencies: pip install ".[empirical]"') from error
        return ad.read_h5ad(path)

    def layer_values(adata, layer, row_indices, gene_indices):
        label(layer, 'Expression layer')
        if layer != 'X' and layer not in adata.layers:
            raise ValueError(f'Unknown expression layer {layer!r}')
        matrix = adata.X if layer == 'X' else adata.layers[layer]
        if matrix is None:
            raise ValueError('Selected expression matrix is empty')
        matrix = matrix[row_indices, :][:, gene_indices]
        return matrix.toarray() if hasattr(matrix, 'toarray') else np.asarray(matrix)

    cell_path = source(cell_spec['path'])
    adata = None
    if cell_spec['format'] == 'h5ad':
        if 'layer' not in cell_spec or {'gene_columns', 'cell_id_column'} & set(cell_spec):
            raise ValueError('H5AD cells require layer and use obs_names/var_names identifiers')
        adata = anndata(cell_path)
        cell_ids = strings(adata.obs_names, 'Cell ID')
        all_genes = strings(adata.var_names, 'Gene ID')
        missing = set(metadata)-set(adata.obs.columns)
        if missing:
            raise ValueError(f'Missing cell metadata columns: {sorted(missing)}')
        meta = {column: strings(adata.obs[column].tolist(), column) for column in metadata}
    elif cell_spec['format'] == 'tsv':
        if not {'gene_columns', 'cell_id_column'}.issubset(cell_spec) or 'layer' in cell_spec:
            raise ValueError('TSV cells require gene_columns/cell_id_column and do not use layer')
        if not isinstance(cell_spec['gene_columns'], list) or not cell_spec['gene_columns']:
            raise ValueError('gene_columns must be a nonempty list')
        all_genes = strings(cell_spec['gene_columns'], 'Gene ID')
        headers, cell_rows = table(cell_path)
        required = {cell_spec['cell_id_column'], *metadata, *all_genes}
        if not required.issubset(headers):
            raise ValueError(f'Missing TSV cell columns: {sorted(required-set(headers))}')
        if set(all_genes) & {cell_spec['cell_id_column'], *metadata}:
            raise ValueError('Gene columns overlap cell metadata')
        cell_ids = strings([r[cell_spec['cell_id_column']] for r in cell_rows], 'Cell ID')
        meta = {column: strings([r[column] for r in cell_rows], column) for column in metadata}
    else:
        raise ValueError('Cells format must be h5ad or tsv')
    if len(set(cell_ids)) != len(cell_ids) or len(set(all_genes)) != len(all_genes):
        raise ValueError('Duplicate cell or gene identifiers')
    genes = tuple(explicit_panel) if explicit_panel is not None else all_genes
    if not set(genes).issubset(all_genes):
        raise ValueError('Declared gene panel has genes missing from cells')
    if set(genes) & {'cell_id', 'stratum', 'allocation', 'depth', 'block', 'cells_per_block', 'task'}:
        raise ValueError('Gene names conflict with prepared TSV metadata')
    headers, tasks = table(source(spec['tasks']))
    if set(headers) != {'task', 'unit', 'condition', *context}:
        raise ValueError('Tasks TSV needs exactly task, unit, condition and declared context columns')
    task_ids, selectors, selected = [], set(), set()
    control_value = label(spec['control_value'], 'Control value')
    aggregation = spec['treated_aggregation']
    if aggregation not in ('cell_mean', 'equal_stratum_mean'):
        raise ValueError('treated_aggregation must be cell_mean or equal_stratum_mean')
    models = spec['models']
    if not isinstance(models, list) or len(models) < 2:
        raise ValueError('Provide at least two models')
    for model in models:
        _schema(model, {'name', 'kind', 'conditioning', 'representation'},
                {'predictions', 'baseline', 'prediction_files', 'baseline_files', 'membership_sha256', 'provenance_files'}, 'Preparation model')
    needs_references = spec['target'] != 'treated_state' or any(m['conditioning'] == 'block' for m in models)
    if needs_references != ('allocation' in spec):
        raise ValueError('Allocation required exactly for effect targets or block-conditioned state models')
    if needs_references:
        allocation = spec['allocation']
        _schema(allocation, {'blocks', 'depths', 'allocations', 'seed'}, set(), 'Allocation')
        def integer(value, minimum):
            return isinstance(value, int) and not isinstance(value, bool) and value >= minimum
        if allocation['blocks'] not in (2, 3) or isinstance(allocation['blocks'], bool):
            raise ValueError('Allocation blocks must be 2 or 3')
        if not isinstance(allocation['depths'], list) or not allocation['depths'] or any(not integer(d, 1) for d in allocation['depths']):
            raise ValueError('Allocation depths must be positive integers')
        if len(set(allocation['depths'])) != len(allocation['depths']) or not integer(allocation['allocations'], 1) or not integer(allocation['seed'], 0):
            raise ValueError('Invalid allocation counts, duplicate depths or seed')
    strata = tuple(json.dumps([meta[column][i] for column in cell_spec['strata_columns']], ensure_ascii=False)
                   if cell_spec['strata_columns'] else 'all' for i in range(len(cell_ids)))
    task_indices = []
    for task in tasks:
        for column, value in task.items():
            label(value, f'Task {column}')
        if task['task'] in task_ids:
            raise ValueError('Duplicate task IDs')
        task_ids.append(task['task'])
        selector = tuple(task[key] for key in ['unit', 'condition', *context])
        if selector in selectors or task['condition'] == control_value:
            raise ValueError('Duplicate task selector or target condition equals control value')
        selectors.add(selector)
        pool = [i for i in range(len(cell_ids)) if meta[cell_spec['unit_column']][i] == task['unit'] and
                all(meta[column][i] == task[column] for column in context)]
        treated_idx = [i for i in pool if meta[cell_spec['condition_column']][i] == task['condition']]
        control_idx = [i for i in pool if meta[cell_spec['condition_column']][i] == control_value] if needs_references else []
        if not treated_idx or (needs_references and not control_idx):
            raise ValueError(f'{task["task"]}: missing treated cells or required matched controls')
        selected.update(treated_idx + control_idx)
        task_indices.append((treated_idx, control_idx))
    selected_order = sorted(selected)
    gene_indices = [all_genes.index(gene) for gene in genes]
    if adata is not None:
        values = layer_values(adata, cell_spec['layer'], selected_order, gene_indices)
    else:
        try:
            values = np.asarray([[float(cell_rows[i][gene]) for gene in genes] for i in selected_order])
        except (ValueError, TypeError) as error:
            raise ValueError('Selected expression must be numeric') from error
    if values.dtype.kind not in 'iuf' or not np.isfinite(values).all():
        raise ValueError('Selected expression must be finite')
    by_index = {original: values[position] for position, original in enumerate(selected_order)}

    def write_vector(name, vector):
        contents[name] = _tsv(['gene', 'value'], [dict(gene=g, value=float(v)) for g, v in zip(genes, vector)])
        return name

    def matrix(specification):
        if isinstance(specification, str):
            path = source(specification)
            headers, rows = table(path)
            if set(headers) != {'task', *genes}:
                raise ValueError('Prediction/baseline TSV gene set must exactly match selected genes')
            ids = strings([row['task'] for row in rows], 'Prediction task')
            try:
                matrix_values = np.asarray([[float(row[gene]) for gene in genes] for row in rows])
            except (ValueError, TypeError) as error:
                raise ValueError('Predictions/baselines must be numeric') from error
        else:
            _schema(specification, {'format', 'path', 'layer', 'task_column'}, set(), 'Prediction matrix')
            if specification['format'] != 'h5ad':
                raise ValueError('Prediction matrix object format must be h5ad')
            pred = anndata(source(specification['path']))
            column = label(specification['task_column'], 'Prediction task column')
            if column != '__index__' and column not in pred.obs:
                raise ValueError('Missing prediction task column')
            ids = strings(pred.obs_names if column == '__index__' else pred.obs[column].tolist(), 'Prediction task')
            pred_genes = strings(pred.var_names, 'Prediction gene')
            if len(set(pred_genes)) != len(pred_genes):
                raise ValueError('Duplicate prediction genes')
            if (explicit_panel is None and set(pred_genes) != set(genes)) or not set(genes).issubset(pred_genes):
                raise ValueError('Prediction/baseline H5AD gene set differs from declared panel')
            matrix_values = layer_values(pred, specification['layer'], np.arange(len(ids)), [pred_genes.index(g) for g in genes])
        if len(set(ids)) != len(ids) or set(ids) != set(task_ids):
            raise ValueError('Prediction/baseline tasks must match exactly, without duplicates')
        if matrix_values.dtype.kind not in 'iuf' or not np.isfinite(matrix_values).all():
            raise ValueError('Predictions/baselines must be finite')
        return dict(zip(ids, matrix_values))

    loaded_models = []
    for model in models:
        fixed = model['conditioning'] == 'fixed'
        converted = model['kind'] == 'state' and model['representation'] == 'effect'
        if fixed:
            if 'predictions' not in model or {'prediction_files', 'baseline_files', 'membership_sha256'} & set(model):
                raise ValueError('Fixed model requires predictions matrix, not block files')
            if ('baseline' in model) != converted:
                raise ValueError('Baseline required exactly for state stored in effect representation')
            prediction_values = matrix(model['predictions'])
            baseline_values = matrix(model['baseline']) if converted else None
        elif model['conditioning'] == 'block':
            if not {'prediction_files', 'membership_sha256'}.issubset(model) or {'predictions', 'baseline'} & set(model):
                raise ValueError('Block model requires prediction_files and membership_sha256')
            if ('baseline_files' in model) != converted:
                raise ValueError('baseline_files required exactly for block state stored in effect representation')
            for field in ('prediction_files', 'membership_sha256', 'baseline_files'):
                if field in model and (not isinstance(model[field], dict) or set(model[field]) != set(task_ids)):
                    raise ValueError(f'{field} must map every task exactly')
            prediction_values, baseline_values = None, None
        else:
            raise ValueError('Conditioning must be explicitly fixed or block')
        if 'provenance_files' in model and (not isinstance(model['provenance_files'], dict) or set(model['provenance_files']) != set(task_ids)):
            raise ValueError('provenance_files must map every task exactly')
        loaded_models.append((model, prediction_values, baseline_values))
    prepared_tasks, preview = [], []
    for ti, (task, (treated_idx, control_idx)) in enumerate(zip(tasks, task_indices)):
        treated = np.stack([by_index[i] for i in treated_idx])
        if aggregation == 'cell_mean':
            target = treated.mean(axis=0)
        else:
            target = np.mean([np.mean([by_index[i] for i in treated_idx if strata[i] == group], axis=0)
                              for group in sorted({strata[i] for i in treated_idx})], axis=0)
        prepared = dict(id=task['task'], unit=task['unit'], treated=write_vector(f't{ti}_treated.tsv', target), models=[])
        membership_data = None
        if needs_references:
            controls = np.stack([by_index[i] for i in control_idx])
            ids, groups = [cell_ids[i] for i in control_idx], [strata[i] for i in control_idx]
            controls_name, references_name, membership_name = f't{ti}_controls.tsv', f't{ti}_references.tsv', f't{ti}_membership.tsv'
            contents[controls_name] = _tsv(['cell_id', 'stratum', *genes], [dict(cell_id=cell, stratum=group, **dict(zip(genes, values))) for cell, group, values in zip(ids, groups, controls)])
            references, membership = [], []
            for ai in range(allocation['allocations']):
                for depth in sorted(allocation['depths']):
                    result = allocate_controls(controls, ids, depth, allocation['seed']+ai, groups, blocks=allocation['blocks'])
                    for bi, mean in enumerate(result.means):
                        references.append(dict(allocation=str(ai), depth=depth, block=f'B{bi+1}', cells_per_block=result.cells_per_block, **dict(zip(genes, mean))))
                    for row in result.membership:
                        membership.append(dict(allocation=str(ai), depth=depth, cell_id=row['cell_id'], stratum=row['stratum'], block=f'B{row["block"]+1}', within_block_index=row['within_block_index']))
            contents[references_name] = _tsv(['allocation', 'depth', 'block', 'cells_per_block', *genes], references)
            membership_data = _tsv(['allocation', 'depth', 'cell_id', 'stratum', 'block', 'within_block_index'], membership)
            contents[membership_name] = membership_data
            prepared.update(controls=controls_name, references=references_name, membership=membership_name)
        for mi, (model, predictions, baselines) in enumerate(loaded_models):
            out_model = {key: model[key] for key in ('name', 'kind', 'conditioning', 'representation')}
            if predictions is not None:
                out_model['prediction'] = write_vector(f't{ti}_m{mi}_prediction.tsv', predictions[task['task']])
                if baselines is not None:
                    out_model['baseline'] = write_vector(f't{ti}_m{mi}_baseline.tsv', baselines[task['task']])
            else:
                digest = hashlib.sha256(membership_data).hexdigest()
                if model['membership_sha256'][task['task']] != digest:
                    raise ValueError('Block prediction declared membership digest does not match allocated controls')
                for field, key in [('prediction_files', 'prediction'), ('baseline_files', 'baseline')]:
                    if field in model:
                        name = f't{ti}_m{mi}_{key}.tsv'
                        contents[name] = source(model[field][task['task']]).read_bytes()
                        out_model[key] = name
            if 'provenance_files' in model:
                path = source(model['provenance_files'][task['task']])
                provenance = load_json(path)
                _schema(provenance, {'files'}, {'description'}, 'Provenance')
                if not isinstance(provenance['files'], list) or not provenance['files']:
                    raise ValueError('Provenance files must be nonempty')
                rewritten = []
                for pi, item in enumerate(provenance['files']):
                    _schema(item, {'path', 'sha256'}, set(), 'Provenance file')
                    original = source(item['path'], path.parent)
                    if hashes[str(original)] != item['sha256']:
                        raise ValueError('Provenance input digest mismatch')
                    name = f't{ti}_m{mi}_provenance_{pi}.bin'
                    contents[name] = original.read_bytes()
                    rewritten.append(dict(path=name, sha256=item['sha256']))
                provenance['files'] = rewritten
                name = f't{ti}_m{mi}_provenance.json'
                contents[name] = (json.dumps(provenance, indent=2)+'\n').encode()
                out_model['provenance'] = name
            prepared['models'].append(out_model)
        prepared_tasks.append(prepared)
        preview.append(dict(task=task['task'], unit=task['unit'], condition=task['condition'], context={column: task[column] for column in context}, treated_cells=len(treated_idx), control_cells=len(control_idx),
                            treated_strata=len({strata[i] for i in treated_idx}), control_strata=len({strata[i] for i in control_idx})))
    plan = dict(target=spec['target'], prediction_controls=spec['prediction_controls'], tasks=prepared_tasks, diagnostics=spec.get('diagnostics', False))
    if 'scales' in spec:
        _, scales = read_vector(source(spec['scales']), genes)
        plan['scales'] = write_vector('scales.tsv', scales)
    contents['plan.json'] = (json.dumps(plan, indent=2, allow_nan=False)+'\n').encode()
    with tempfile.TemporaryDirectory(prefix='reference-prepare-') as directory:
        temporary = Path(directory)
        for name, data in contents.items():
            (temporary/name).write_bytes(data)
        run_plan(temporary/'plan.json', temporary/'validation')
    def describe_input(specification, block=False):
        if isinstance(specification, str):
            description = dict(format='tsv', gene_columns=list(genes))
            if block:
                description['index_columns'] = ['allocation', 'depth', 'block']
            else:
                description['task_column'] = 'task'
            path = specification
        else:
            description = {key: value for key, value in specification.items() if key != 'path'}
            path = specification['path']
        description['input_sha256'] = hashes[str((manifest_path.parent/path).resolve())]
        return description

    cell_input = describe_input(cell_spec)
    model_inputs = []
    for model in models:
        description = {key: model[key] for key in ('name', 'kind', 'conditioning', 'representation')}
        for field in ('predictions', 'baseline'):
            if field in model:
                description[field] = describe_input(model[field])
        for field in ('prediction_files', 'baseline_files'):
            if field in model:
                description[field] = {task: describe_input(path, block=True) for task, path in model[field].items()}
        if 'membership_sha256' in model:
            description['membership_sha256'] = model['membership_sha256']
        model_inputs.append(description)
    receipt = dict(normalization=cell_spec['normalization'], normalization_performed=False, treated_aggregation=aggregation,
                   cells=cell_input, models=model_inputs,
                   gene_panel=list(genes), selected_cells=len(selected), ignored_cells=len(cell_ids)-len(selected), tasks=preview,
                   inputs=[dict(path=path, sha256=digest) for path, digest in sorted(hashes.items())],
                   prepared_files={name: hashlib.sha256(data).hexdigest() for name, data in sorted(contents.items())},
                   verification='Annotations, gene/task support, numerical values and reference membership checked. Supplied predictions are not verified as actual outputs of training or declared input-cell use.',
                   next_command='reference-design run plan.json --output results')
    contents['prepare_receipt.json'] = (json.dumps(receipt, indent=2, allow_nan=False)+'\n').encode()
    def display(value):
        return html.escape(str(value)).replace('\\', '\\\\').replace('|', '\\|').replace('\r', '').replace('\n', '<br>')
    def column_list(columns):
        return ', '.join(display(column) for column in columns) or '(none)'

    def matrix_selection(description):
        if description['format'] == 'h5ad':
            task_ids = 'obs_names' if description['task_column'] == '__index__' else f'obs column {display(description["task_column"])}'
            return f'layer: {display(description["layer"])}; task IDs: {task_ids}; gene IDs: var_names'
        indices = 'index columns: '+column_list(description['index_columns']) if 'index_columns' in description else 'task column: task'
        return indices+'; gene columns: '+column_list(description['gene_columns'])

    if cell_input['format'] == 'h5ad':
        cell_selection = f'expression layer: {display(cell_input["layer"])}; cell IDs: obs_names; gene IDs: var_names'
    else:
        cell_selection = f'cell ID column: {display(cell_input["cell_id_column"])}; gene columns: {column_list(cell_input["gene_columns"])}'
    lines = ['# Prepared reference protocol', '', f'{len(tasks)} tasks, {len(set(t["unit"] for t in tasks))} declared units, {len(genes)} genes.',
             f'Cell input: {cell_input["format"]}; {cell_selection}.',
             f'Metadata mapping: unit = {display(cell_input["unit_column"])}; condition = {display(cell_input["condition_column"])}; '
             f'context = {column_list(cell_input["context_columns"])}; strata = {column_list(cell_input["strata_columns"])}.',
             f'Control value: {display(control_value)}.',
             f'Expression is used as supplied: {display(cell_spec["normalization"])}. No normalization was performed.',
             f'Treated aggregation: {aggregation}. Controls are sampled at equal depth within each declared stratum.', '',
             '| Model | Input | Format | Selection |', '| --- | --- | --- | --- |']
    for model in model_inputs:
        for field in ('predictions', 'baseline', 'prediction_files', 'baseline_files'):
            if field in model:
                description = next(iter(model[field].values())) if field.endswith('_files') else model[field]
                lines.append(f'| {display(model["name"])} | {field} | {description["format"]} | {matrix_selection(description)} |')
    lines += ['', '| Task | Unit | Condition | Treated cells | Control cells | Treated strata | Control strata |',
              '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    lines += ['| '+' | '.join(display(row[key]) for key in ('task', 'unit', 'condition', 'treated_cells', 'control_cells', 'treated_strata', 'control_strata'))+' |' for row in preview]
    lines += ['', f'{len(cell_ids)-len(selected)} cells outside declared selections were ignored. Detailed input and prepared-file hashes are in prepare_receipt.json.',
              'Membership checks verify declared cell selections and means; they do not establish biological independence or actual training/prediction input use.', '',
              'Run `reference-design run plan.json --output results` from this directory.', '']
    contents['preview.md'] = '\n'.join(lines).encode()
    _publish(output, contents)
    return receipt
