"""Declared reference protocols for supplied predictions; no model fitting or inference."""
from __future__ import annotations

import hashlib
import itertools
import json
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from .core import Prediction, effect_to_state, score_references
from .io import label, match_blocks, read_cells, read_vector, verify_membership
from .protocol_io import read_protocol_blocks


def _average(rows, keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row['MSE'])
    return [dict(zip(keys, key), MSE=float(np.mean(values)))
            for key, values in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0])))]


def _pairs(rows, keys):
    grouped = defaultdict(dict)
    kinds = {}
    for row in rows:
        grouped[tuple(row[key] for key in keys)][row['model']] = row['MSE']
        kinds[row['model']] = row['kind']
    result = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        for first, second in itertools.combinations(sorted(values), 2):
            a, b = (second, first) if kinds[first] == 'state' and kinds[second] == 'effect' else (first, second)
            result.append(dict(zip(keys, key), model_a=a, model_b=b, margin=values[b]-values[a]))
    return result


def _diagnostic_report(summary, allocation_pairs, display):
    margins = defaultdict(dict)
    identity_checks = defaultdict(list)
    for row in _pairs(summary, ['depth', 'pattern']):
        margins[(row['depth'], row['model_a'], row['model_b'])][row['pattern']] = row['margin']
    for row in allocation_pairs:
        identity_checks[(row['depth'], row['model_a'], row['model_b'])].append(row)
    lines = [
        '', '## Reference sensitivity', '',
        'Pair margins use the same task and unit weights as the primary comparison. Positive values favour model a.',
        'S shares all three roles; M shares the two scoring references; P shares observation and model input; '
        'O shares prediction centring and model input; D uses a separate block for each role.', '',
        '| Depth | Model a | Model b | S | M | P | O | D | S to D | Identity checks |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |',
    ]
    for key, values in margins.items():
        shared, separated = values['S'], values['D']
        signs = [0 if abs(value) <= 1e-12 else 1 if value > 0 else -1
                 for value in (shared, separated)]
        change = ('tie in both' if signs == [0, 0] else 'tie transition' if 0 in signs
                  else 'reversal' if signs[0] != signs[1] else 'same direction')
        checks = identity_checks[key]
        if not all(row['identity_applicable'] for row in checks):
            identity = 'not applicable (input-conditioned effect)'
        else:
            identity = 'passed' if all(row['identity_verified'] for row in checks) else 'failed'
        cells = [*(display(value) for value in key),
                 *(f'{values[pattern]:.8g}' for pattern in ('S', 'M', 'P', 'O', 'D')),
                 change, identity]
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += [
        '', 'S-to-D changes compare the equal-unit margins, using a numerical tie tolerance of 1e-12. '
        'Identity checks cover each task and allocation before averaging.',
        'Full values: [pattern margins](diagnostic_summary_pairs.tsv), '
        '[model scores](diagnostic_summary_scores.tsv), and '
        '[allocation-level identities and reversals](diagnostic_allocation_pairs.tsv).',
    ]
    return lines


def run_plan(plan_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Validate a plan and atomically publish descriptive task/unit summaries.

    Inputs are supplied predictions; membership verification does not prove how a
    model was trained or that the supplied prediction consumed the declared cells.
    """
    from .cli import _schema, _json_object, _json_constant, _destination, _publish, _tsv
    plan_path, output = Path(plan_path), Path(output_path)
    _destination(output)
    hashes = {}

    def record(path):
        path = Path(path).resolve()
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def source(value, base=plan_path.parent):
        return record(base / label(value, 'Input path'))

    def read_json(path):
        return json.loads(path.read_text(), object_pairs_hook=_json_object, parse_constant=_json_constant)

    spec = read_json(record(plan_path))
    _schema(spec, {'target', 'prediction_controls', 'tasks'}, {'diagnostics', 'scales'}, 'Plan')
    target = spec['target']
    if target not in ('heldout_effect', 'shared_effect', 'treated_state'):
        raise ValueError('target must be heldout_effect, shared_effect or treated_state')
    if spec['prediction_controls'] not in ('available', 'unavailable'):
        raise ValueError('prediction_controls must be available or unavailable')
    if target == 'shared_effect' and spec['prediction_controls'] != 'available':
        raise ValueError('Shared-input effect target requires available prediction controls')
    diagnostic_requested = spec.get('diagnostics', False)
    if not isinstance(diagnostic_requested, bool):
        raise ValueError('diagnostics must be boolean')
    tasks = spec['tasks']
    if not isinstance(tasks, list) or not tasks:
        raise ValueError('tasks must be a nonempty list')
    primary, diagnostic, diagnostic_pairs, checks = [], [], [], []
    seen, family, common_depths = set(), None, None
    for task in tasks:
        _schema(task, {'id', 'unit', 'treated', 'models'}, {'references', 'controls', 'membership'}, 'Task')
        task_id, unit = label(task['id'], 'Task ID'), label(task['unit'], 'Unit ID')
        if task_id in seen:
            raise ValueError(f'Duplicate task ID {task_id}')
        seen.add(task_id)
        genes, treated = read_vector(source(task['treated']))
        scales = np.ones(len(genes))
        if 'scales' in spec:
            _, scales = read_vector(source(spec['scales']), genes)
        if np.any(scales <= 0):
            raise ValueError('scales must be strictly positive')
        references = read_protocol_blocks(source(task['references']), genes) if 'references' in task else None
        if target != 'treated_state' and references is None:
            raise ValueError('Effect targets require references')
        if ('controls' in task) != ('membership' in task):
            raise ValueError('Provide controls and membership together')
        membership_check = 'not used' if references is None else 'supplied reference means'
        if 'controls' in task:
            if references is None:
                raise ValueError('Membership verification requires references')
            verify_membership(read_cells(source(task['controls']), genes), references, source(task['membership']))
            membership_check = 'cell membership and reference means verified'
        groups = references.groups if references is not None else [('none', 'none')]
        depths = {group[1] for group in groups}
        if common_depths is not None and depths != common_depths:
            raise ValueError('All tasks must have the same depth support')
        common_depths = depths
        model_specs = task['models']
        if not isinstance(model_specs, list) or len(model_specs) < 2:
            raise ValueError('Each task needs at least two models')
        models, signatures = [], []
        for model in model_specs:
            _schema(model, {'name', 'kind', 'prediction'}, {'conditioning', 'representation', 'baseline', 'provenance', 'generation_record'}, 'Model')
            name = label(model['name'], 'Model name')
            kind, conditioning = model['kind'], model.get('conditioning', 'fixed')
            representation = model.get('representation', 'native')
            if kind not in ('state', 'effect') or conditioning not in ('fixed', 'block'):
                raise ValueError(f'{name}: unsupported kind or conditioning')
            if representation not in ('native', 'effect'):
                raise ValueError(f'{name}: unknown baseline/representation semantics')
            converted = kind == 'state' and representation == 'effect'
            if ('baseline' in model) != converted:
                raise ValueError(f'{name}: baseline required only for state stored in effect representation')
            if kind == 'effect' and representation != 'native':
                raise ValueError(f'{name}: native effects require native representation')
            if target == 'treated_state' and kind != 'state':
                raise ValueError('Treated-state target requires state predictions')
            if conditioning == 'block' and (spec['prediction_controls'] != 'available' or references is None):
                raise ValueError(f'{name}: block conditioning requires available prediction controls and references')
            # An unavailable deployment reference cannot be used to turn a state
            # prediction into an effect; native effects remain evaluable.
            if target != 'treated_state' and kind == 'state' and spec['prediction_controls'] == 'unavailable':
                raise ValueError(f'{name}: state-to-effect centring requires available prediction controls')
            signatures.append((name, kind, conditioning, representation))
            path = source(model['prediction'])
            if conditioning == 'fixed':
                _, values = read_vector(path, genes)
                if converted:
                    _, baseline = read_vector(source(model['baseline']), genes)
                    values = effect_to_state(values, baseline)
            else:
                blocks = read_protocol_blocks(path, genes)
                match_blocks(blocks, references, path)
                values = blocks.values
                if converted:
                    base_path = source(model['baseline'])
                    baseline = read_protocol_blocks(base_path, genes)
                    match_blocks(baseline, references, base_path)
                    values = {key: effect_to_state(value, baseline.values[key]) for key, value in values.items()}
            if 'provenance' in model:
                provenance_path = source(model['provenance'])
                provenance = read_json(provenance_path)
                _schema(provenance, {'files'}, {'description'}, 'Provenance')
                if not isinstance(provenance['files'], list) or not provenance['files']:
                    raise ValueError('Provenance files must be nonempty')
                for item in provenance['files']:
                    _schema(item, {'path', 'sha256'}, set(), 'Provenance file')
                    bound = source(item['path'], provenance_path.parent)
                    if item['sha256'] != hashes[str(bound)]:
                        raise ValueError(f'Provenance digest mismatch: {bound}')
            evidence = dict(level='declaration', verified_input_use=False, artifacts={})
            if 'generation_record' in model:
                from .input_binding import verify_generation_record
                evidence = verify_generation_record(
                    source(model['generation_record']), prediction_path=path,
                    model_name=name, kind=kind, conditioning=conditioning,
                    representation=representation,
                    controls_path=source(task['controls']) if 'controls' in task else None,
                    membership_path=source(task['membership']) if 'membership' in task else None,
                    references_path=source(task['references']) if 'references' in task else None,
                    baseline_path=source(model['baseline']) if 'baseline' in model else None,
                    expected_input_keys=sorted(references.values) if conditioning == 'block' else None,
                )
            models.append(dict(name=name, kind=kind, conditioning=conditioning, values=values,
                               generation_evidence=evidence))
        if len({item[0] for item in signatures}) != len(signatures):
            raise ValueError('Duplicate model names')
        signature = set(signatures)
        if family is not None and signature != family:
            raise ValueError('All tasks must have the same complete model family and semantics')
        family = signature
        if references is not None:
            required_blocks = set()
            if target != 'treated_state':
                required_blocks.add('B2' if target == 'heldout_effect' else 'B1')
            if any(m['conditioning'] == 'block' or (target != 'treated_state' and m['kind'] == 'state') for m in models):
                required_blocks.add('B1')
            if not required_blocks.issubset({key[2] for key in references.values}):
                raise ValueError(f'Target/model roles require reference blocks {sorted(required_blocks)}')
        eligible = (target != 'treated_state' and references is not None and
                    {key[2] for key in references.values} == {'B1', 'B2', 'B3'})
        reason = ('available' if eligible else 'unavailable: treated-state target' if target == 'treated_state'
                  else 'unavailable: three blocks required for five-pattern diagnostics')
        checks.append(dict(task=task_id, unit=unit, references=membership_check,
                           predictions='supplied predictions; use of declared controls is not independently verified',
                           prediction_evidence=[dict(model=m['name'], evidence=m['generation_evidence']) for m in models],
                           diagnostics=reason if diagnostic_requested else 'not requested'))
        for allocation, depth in groups:
            meta = dict(task=task_id, unit=unit, allocation=allocation, depth=depth)
            for model in models:
                pred = model['values'] if model['conditioning'] == 'fixed' else model['values'][(allocation, depth, 'B1')]
                residual = pred - treated
                if target != 'treated_state':
                    obs = references.values[(allocation, depth, 'B2' if target == 'heldout_effect' else 'B1')]
                    residual = residual + obs
                    if model['kind'] == 'state':
                        residual = residual - references.values[(allocation, depth, 'B1')]
                with np.errstate(over='raise', invalid='raise', divide='raise'):
                    loss = float(np.mean((residual/scales)**2))
                primary.append(dict(**meta, model=model['name'], kind=model['kind'], MSE=loss))
            if diagnostic_requested and eligible:
                keys = [(allocation, depth, block) for block in ('B1', 'B2', 'B3')]
                predictions = [Prediction(m['name'], m['values'] if m['conditioning'] == 'fixed' else
                                          np.stack([m['values'][key] for key in keys]), m['kind']) for m in models]
                result = score_references(treated, np.stack([references.values[key] for key in keys]), predictions, scales)
                diagnostic.extend(dict(**meta, pattern=r['pattern'], model=r['model'], kind=r['kind'], MSE=r['mse']) for r in result.rankings)
                diagnostic_pairs.extend(dict(**meta, **r) for r in result.pairwise)
    # Partial diagnostic support cannot be pooled as if it were the primary
    # complete task family. Keep diagnostics unavailable for the whole plan.
    if diagnostic_requested and not all(c['diagnostics'] == 'available' for c in checks):
        diagnostic, diagnostic_pairs = [], []
    contents = {}

    def emit(name, rows):
        if rows:
            contents[name + '.tsv'] = _tsv(list(rows[0]), rows)

    def hierarchy(prefix, rows, extra):
        emit(prefix + '_allocation_scores', rows)
        task_rows = _average(rows, ['task', 'unit', 'depth', *extra, 'model', 'kind'])
        unit_rows = _average(task_rows, ['unit', 'depth', *extra, 'model', 'kind'])
        summary = _average(unit_rows, ['depth', *extra, 'model', 'kind'])
        for level, data, keys in [('task', task_rows, ['task', 'unit', 'depth', *extra]),
                                  ('unit', unit_rows, ['unit', 'depth', *extra]),
                                  ('summary', summary, ['depth', *extra])]:
            emit(prefix + '_' + level + '_scores', data)
            emit(prefix + '_' + level + '_pairs', _pairs(data, keys))
        return summary

    summary = hierarchy('primary', primary, [])
    if diagnostic:
        diagnostic_summary = hierarchy('diagnostic', diagnostic, ['pattern'])
        emit('diagnostic_allocation_pairs', diagnostic_pairs)
    try:
        software_version = version('reference-cell-benchmark-design')
    except PackageNotFoundError:
        software_version = 'uninstalled'
    roles = dict(observation='unused', prediction_centring='unused',
                 model='B1 for block-conditioned predictions' if any(item[2] == 'block' for item in family) else 'not varied (fixed supplied predictions)')
    if target != 'treated_state':
        roles.update(observation='B2' if target == 'heldout_effect' else 'B1',
                     prediction_centring='B1 for state predictions; unused for native effects' if any(item[1] == 'state' for item in family) else 'unused')
    receipt = dict(software_version=software_version, numpy_version=np.__version__, target=target,
                   prediction_controls=spec['prediction_controls'], roles=roles,
                   weighting='equal allocations within task, equal tasks within unit, equal units; depths separate',
                   loss='mean squared residual after fixed positive gene divisors',
                   inputs=[dict(path=path, sha256=digest) for path, digest in sorted(hashes.items())],
                   tasks=checks, diagnostics_emitted=bool(diagnostic),
                   provenance_limit='Hashes bind supplied files; they do not prove actual training, prediction inputs or biological independence.',
                   inference='Descriptive finite-reference comparisons; no inferential P values.')
    contents['receipt.json'] = (json.dumps(receipt, indent=2, allow_nan=False)+'\n').encode()
    target_description = {
        'heldout_effect': 'Native effects are compared with treated expression minus held-out B2. State predictions are first centred on B1, then compared with that same measured effect.',
        'shared_effect': 'Native effects and B1-centred state predictions are compared with treated expression minus the shared B1 reference.',
        'treated_state': 'State predictions are compared directly with treated expression. Observation and prediction-centring references are unused.',
    }[target]
    def display(value):
        import html
        return html.escape(str(value)).replace('\\', '\\\\').replace('|', '\\|').replace('\r\n', '\n').replace('\r', '\n').replace('\n', '<br>')

    conditioning_text = f'Prediction controls are {spec["prediction_controls"]}.'
    if any(item[2] == 'fixed' for item in family):
        conditioning_text += ' Fixed conditioning means a fixed supplied prediction; it does not assert that the original model used no controls.'
    if any(item[2] == 'block' for item in family):
        conditioning_text += ' Block-conditioned predictions of either output kind select B1 for the primary score. Native effects use no prediction-centring subtraction.'
    lines = ['# Reference protocol results', '', target_description, conditioning_text,
             '', 'Scores average allocations within each task, tasks within each declared unit, and units equally. Depths remain separate. Lower MSE is better.',
             '', '| Depth | Model | Kind | Equal-unit MSE |', '| --- | --- | --- | ---: |']
    lines += [f'| {display(r["depth"])} | {display(r["model"])} | {display(r["kind"])} | {r["MSE"]:.8g} |' for r in summary]
    lines += ['', 'Pair margins are MSE(b) minus MSE(a); positive values favour model a.', '',
              '| Depth | Model a | Model b | Equal-unit margin |', '| --- | --- | --- | ---: |']
    lines += [f'| {display(r["depth"])} | {display(r["model_a"])} | {display(r["model_b"])} | {r["margin"]:.8g} |' for r in _pairs(summary, ['depth'])]
    if diagnostic:
        lines += _diagnostic_report(diagnostic_summary, diagnostic_pairs, display)
    lines += ['', '| Reference role | Primary use |', '| --- | --- |']
    role_labels = {'observation': 'Observation', 'prediction_centring': 'Prediction centring', 'model': 'Model input'}
    lines += [f'| {role_labels[key]} | {display(value)} |' for key, value in roles.items()]
    lines += ['', '| Task | Reference verification | Five-pattern diagnostics |', '| --- | --- | --- |']
    check_groups = defaultdict(list)
    for check in checks:
        check_groups[(check['references'], check['diagnostics'])].append(check['task'])
    for (reference_check, diagnostic_check), task_names in check_groups.items():
        task_display = ', '.join(display(name) for name in task_names) if len(task_names) <= 5 else f'{len(task_names)} tasks'
        lines.append(f'| {task_display} | {display(reference_check)} | {display(diagnostic_check)} |')
    lines += ['', 'Per-task verification details are recorded in receipt.json.']
    if diagnostic_requested and not diagnostic:
        lines += ['', 'No diagnostic tables were pooled because the complete task family is not eligible for three-block effect diagnostics. Primary results remain available.']
    lines += ['', 'Results describe the supplied predictions, tasks and reference allocations. Unit labels do not establish biological independence. Input hashes and optional provenance checks are recorded in receipt.json; they do not verify actual training or prediction inputs.', '']
    contents['report.md'] = '\n'.join(lines).encode()
    _publish(output, contents)
    return receipt
