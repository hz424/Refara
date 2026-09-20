"""Empirical organizer decisions from development variation and frozen anchors."""
from __future__ import annotations

import hashlib
import html
import itertools
import json
import math
from pathlib import Path

from .io import label, table

POLICIES = ('single_split', 'repeated_primary', 'geometry_scaled')
BASE_POLICIES = POLICIES[:2]
CONFIG_FIELDS = {'schema_version', 'target', 'metric', 'models', 'budgets', 'roles',
                 'anchor_realization', 'minimum_gap', 'numerical_tolerance',
                 'evaluation_scope'}
PATH_FIELDS = {'development_scores', 'assessment_anchors'}
ROLE_FIELDS = {'model_input', 'prediction_centring', 'observation_centring',
               'participant_information', 'evaluator_information'}
SCORE_COLUMNS = {'case', 'unit', 'budget', 'realization', 'model', 'score'}


def _schema(value, required, optional, name):
    from .cli import _schema as validate
    validate(value, required, optional, name)


def _json(path):
    from .cli import _json_object, _json_constant
    return json.loads(Path(path).read_text(), object_pairs_hook=_json_object,
                      parse_constant=_json_constant)


def _number(value, name, minimum=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    if minimum is not None and value < minimum or positive and value <= 0:
        raise ValueError(f'{name} is outside its permitted range')
    return float(value)


def _policies(config):
    # Version 1 designs made before optional geometry always contain all policies.
    return tuple(config.get('policies', POLICIES))


def _configuration(spec):
    _schema(spec, CONFIG_FIELDS, {'geometry_tolerance', 'policies'}, 'Organizer configuration')
    if 'policies' in spec and spec['policies'] not in (list(BASE_POLICIES), list(POLICIES)):
        raise ValueError('Frozen policies must be single_split and repeated_primary, optionally followed by geometry_scaled')
    if type(spec['schema_version']) is not int or spec['schema_version'] != 1:
        raise ValueError('schema_version must be 1')
    if spec['target'] not in ('heldout_effect', 'treated_state', 'shared_effect'):
        raise ValueError('Unsupported target')
    _schema(spec['metric'], {'name', 'direction'}, set(), 'Metric')
    label(spec['metric']['name'], 'Metric name')
    if spec['metric']['direction'] not in ('lower', 'higher'):
        raise ValueError('Metric direction must be lower or higher')
    models = spec['models']
    if not isinstance(models, list) or len(models) < 2:
        raise ValueError('models must contain at least two distinct labels')
    for model in models:
        label(model, 'Model')
    if len(set(models)) != len(models):
        raise ValueError('Duplicate model labels')
    budgets = spec['budgets']
    if not isinstance(budgets, list) or not budgets:
        raise ValueError('budgets must be a nonempty list')
    ids = []
    for budget in budgets:
        _schema(budget, {'id', 'control_cells_available', 'input_cells', 'observation_cells', 'description'}, set(), 'Budget')
        ids.append(label(budget['id'], 'Budget ID'))
        label(budget['description'], 'Budget description')
        for field in ('control_cells_available', 'input_cells', 'observation_cells'):
            if type(budget[field]) is not int or budget[field] < 0:
                raise ValueError(f'{field} must be a nonnegative integer')
        if budget['input_cells'] + budget['observation_cells'] > budget['control_cells_available']:
            raise ValueError('Disjoint input and observation cells exceed the available control budget')
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate budget IDs')
    _schema(spec['roles'], ROLE_FIELDS, set(), 'Roles')
    for field, value in spec['roles'].items():
        label(value, field)
    label(spec['anchor_realization'], 'Anchor realization')
    label(spec['evaluation_scope'], 'Evaluation scope')
    _number(spec['minimum_gap'], 'minimum_gap', minimum=0)
    _number(spec['numerical_tolerance'], 'numerical_tolerance', positive=True)
    if 'geometry_scaled' in _policies(spec) and 'geometry_tolerance' not in spec:
        raise ValueError('geometry_tolerance is required when geometry is supplied')
    if 'geometry_tolerance' in spec:
        _number(spec['geometry_tolerance'], 'geometry_tolerance', positive=True)
    return spec


def _scores(path, config, anchor_only=False):
    headers, rows = table(path)
    if set(headers) != SCORE_COLUMNS:
        raise ValueError('Scores TSV needs exactly case, unit, budget, realization, model, score')
    cases = {}
    budget_ids = {b['id'] for b in config['budgets']}
    model_ids = set(config['models'])
    anchor = config['anchor_realization']
    for row in rows:
        for field in SCORE_COLUMNS - {'score'}:
            label(row[field], field)
        if row['budget'] not in budget_ids or row['model'] not in model_ids:
            raise ValueError('Score rows contain undeclared budgets or models')
        if anchor_only and row['realization'] != anchor:
            raise ValueError('Assessment anchors must contain only anchor_realization')
        try:
            value = float(row['score'])
        except ValueError as error:
            raise ValueError('Scores must be finite numbers') from error
        _number(value, 'Score')
        case = cases.setdefault(row['case'], {'unit': row['unit'], 'scores': {}})
        if case['unit'] != row['unit']:
            raise ValueError('Each case must have one consistent unit')
        values = case['scores'].setdefault(row['budget'], {}).setdefault(row['realization'], {})
        if row['model'] in values:
            raise ValueError('Duplicate score row')
        values[row['model']] = value
    for case in cases.values():
        if set(case['scores']) != budget_ids:
            raise ValueError('Every case must cover all declared budgets')
        support = None
        for realizations in case['scores'].values():
            if anchor not in realizations or (not anchor_only and len(realizations) < 2):
                raise ValueError('Every case/budget needs the anchor and at least two realizations')
            if support is not None and set(realizations) != support:
                raise ValueError('Realization support must agree across budgets within each case')
            support = set(realizations)
            if any(set(values) != model_ids for values in realizations.values()):
                raise ValueError('Every score realization must cover all declared models')
    return cases


def _geometry(path, cases, config):
    headers, rows = table(path)
    if set(headers) != {'case', 'unit', 'budget', 'geometry'}:
        raise ValueError('Geometry TSV needs exactly case, unit, budget, geometry')
    budget_ids = {b['id'] for b in config['budgets']}
    result = {}
    for row in rows:
        for field in ('case', 'unit', 'budget'):
            label(row[field], field)
        key = (row['case'], row['budget'])
        if row['case'] not in cases or row['budget'] not in budget_ids:
            raise ValueError('Geometry contains undeclared case/budget rows')
        if row['unit'] != cases[row['case']]['unit']:
            raise ValueError('Geometry unit differs from the score unit')
        if key in result:
            raise ValueError('Duplicate geometry row')
        try:
            value = float(row['geometry'])
        except ValueError as error:
            raise ValueError('Geometry must be finite and nonnegative') from error
        result[key] = _number(value, 'Geometry', minimum=0)
    if set(result) != set(itertools.product(cases, budget_ids)):
        raise ValueError('Geometry must cover exactly every development and assessment case/budget')
    return result


def _margin(values, a, b, config):
    margin = values[b] - values[a] if config['metric']['direction'] == 'lower' else values[a] - values[b]
    return _number(margin, 'Pair margin')


def _calibrate(development, geometry, config):
    rows = []
    for budget in config['budgets']:
        bid = budget['id']
        for a, b in itertools.combinations(config['models'], 2):
            shifts, ratios, zero_shifts = [], [], 0
            for case_id, case in development.items():
                values = case['scores'][bid]
                anchor = _margin(values[config['anchor_realization']], a, b, config)
                shift = max(abs(_margin(scores, a, b, config) - anchor) for scores in values.values())
                _number(shift, 'Development shift', minimum=0)
                shifts.append(shift)
                if geometry is None:
                    continue
                g = geometry[(case_id, bid)]
                if g > config['geometry_tolerance']:
                    ratios.append(_number(shift / g, 'Geometry coefficient', minimum=0))
                elif shift > config['numerical_tolerance']:
                    zero_shifts += 1
            available = bool(ratios) and zero_shifts == 0
            reason = ('geometry_not_supplied' if geometry is None else 'available' if available else
                      'development_zero_geometry_shift' if zero_shifts else 'no_positive_development_geometry')
            rows.append(dict(budget=bid, model_a=a, model_b=b, B=max(shifts), K=max(ratios) if ratios else None,
                             geometry_available=available, geometry_reason=reason, development_cases=len(shifts),
                             positive_geometry_cases=len(ratios), zero_geometry_shift_cases=zero_shifts))
    return rows


def _decisions(cases, calibration, config):
    rows = []
    for case in cases:
        for calibrated in calibration:
            bid, a, b = (calibrated[k] for k in ('budget', 'model_a', 'model_b'))
            margin = _margin(case['anchor_scores'][bid], a, b, config)
            g = case['geometry'][bid] if case['geometry'] is not None else None
            for policy in _policies(config):
                reason = None
                if policy == 'single_split':
                    bound = 0.
                elif policy == 'repeated_primary':
                    bound = calibrated['B']
                elif not calibrated['geometry_available']:
                    bound, reason = None, calibrated['geometry_reason']
                elif g <= config['geometry_tolerance']:
                    bound, reason = None, 'assessment_geometry_at_or_below_tolerance'
                else:
                    bound = _number(calibrated['K'] * g, 'Scaled empirical range', minimum=0)
                released = reason is None and abs(margin) > bound + config['minimum_gap'] + config['numerical_tolerance']
                favored = a if margin > config['numerical_tolerance'] else b if margin < -config['numerical_tolerance'] else None
                rows.append(dict(case=case['case'], unit=case['unit'], budget=bid, policy=policy,
                                 model_a=a, model_b=b, anchor_margin=margin, geometry=g, empirical_range=bound,
                                 favored_model=favored, decision='release' if released else 'hold',
                                 reason=reason or ('margin_exceeds_range_and_gap' if released else 'margin_within_range_or_gap')))
    return rows


def _budget_recommendations(cases, decisions, config):
    by_key = {(r['case'], r['budget'], r['policy'], r['model_a'], r['model_b']): r for r in decisions}
    budgets = sorted(enumerate(config['budgets']), key=lambda item: (item[1]['control_cells_available'], item[0]))
    rows = []
    for case in cases:
        for policy in _policies(config):
            chosen = None
            for _, budget in budgets:
                bid = budget['id']
                scores = case['anchor_scores'][bid]
                winner = min(config['models'], key=lambda m: scores[m]) if config['metric']['direction'] == 'lower' else max(config['models'], key=lambda m: scores[m])
                comparisons = [by_key[(case['case'], bid, policy, a, b)]
                               for a, b in itertools.combinations(config['models'], 2) if winner in (a, b)]
                if all(r['decision'] == 'release' and r['favored_model'] == winner for r in comparisons):
                    chosen = (budget, winner)
                    break
            budget, winner = chosen if chosen else ({}, None)
            rows.append(dict(case=case['case'], unit=case['unit'], policy=policy,
                             decision='release' if chosen else 'hold', model=winner, budget=budget.get('id'),
                             control_cells_available=budget.get('control_cells_available'), input_cells=budget.get('input_cells'),
                             observation_cells=budget.get('observation_cells'),
                             reason='all_winner_comparisons_released' if chosen else 'no_declared_budget_releases_a_unique_winner'))
    return rows


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False, sort_keys=True) + '\n').encode()


def _emit(output, contents, inputs):
    from .cli import _publish
    receipt = dict(inputs=inputs, outputs={name: hashlib.sha256(data).hexdigest() for name, data in sorted(contents.items())})
    contents['receipt.json'] = _encoded(receipt)
    _publish(Path(output), contents)


def _tabular(rows):
    from .cli import _tsv
    return _tsv(list(rows[0]), rows)


def _display(value):
    return html.escape(str(value)).replace('|', '\\|').replace('\n', ' ').replace('\r', '')


def design(manifest_path, output_path):
    """Freeze empirical release/hold recommendations before assessment scoring."""
    from .cli import _destination
    manifest_path, output = Path(manifest_path), Path(output_path)
    _destination(output)
    manifest = _json(manifest_path)
    _schema(manifest, CONFIG_FIELDS | PATH_FIELDS, {'geometry', 'geometry_tolerance'}, 'Organizer manifest')
    config = {k: manifest[k] for k in CONFIG_FIELDS | {'geometry_tolerance'} if k in manifest}
    if 'geometry' not in manifest:
        config['policies'] = list(BASE_POLICIES)
    config = _configuration(config)
    paths = {field: manifest_path.parent / label(manifest[field], field)
             for field in PATH_FIELDS | {'geometry'} if field in manifest}
    development = _scores(paths['development_scores'], config)
    anchors = _scores(paths['assessment_anchors'], config, anchor_only=True)
    if set(development) & set(anchors):
        raise ValueError('Development and assessment cases must be disjoint')
    geometry = _geometry(paths['geometry'], {**development, **anchors}, config) if 'geometry' in paths else None
    calibration = _calibrate(development, geometry, config)
    cases = [dict(case=case_id, unit=case['unit'],
                  anchor_scores={bid: values[config['anchor_realization']] for bid, values in case['scores'].items()},
                  geometry={budget['id']: geometry[(case_id, budget['id'])] for budget in config['budgets']} if geometry is not None else None)
             for case_id, case in sorted(anchors.items())]
    decisions = _decisions(cases, calibration, config)
    recommendations = _budget_recommendations(cases, decisions, config)
    inputs = {field: dict(name=path.name, sha256=_digest(path)) for field, path in {'manifest': manifest_path, **paths}.items()}
    artifact = dict(artifact_type='reference_design_organizer', schema_version=1, configuration=config,
                    inputs=inputs, development_cases=[dict(case=k, unit=v['unit']) for k, v in sorted(development.items())],
                    calibration=calibration, cases=cases, decisions=decisions, budget_recommendations=recommendations)
    report = ['# Organizer recommendations', '',
              'These provisional recommendations apply empirical reference variation measured in the development cases to the declared assessment anchors.',
              f'Scope: {_display(config["evaluation_scope"])}',
              f'Target: {_display(config["target"])}. Metric: {_display(config["metric"]["name"])} ({config["metric"]["direction"]} is better).',
              f'The user-selected minimum gap is {config["minimum_gap"]}; it is a decision setting, not an established biological threshold.',
              'The calibrated ranges are descriptive empirical ranges, not confidence bounds or guarantees for unseen models.', '',
              '| Policy | Released comparisons | Total comparisons |', '| --- | ---: | ---: |']
    for policy in _policies(config):
        selected = [r for r in decisions if r['policy'] == policy]
        report.append(f'| {policy} | {sum(r["decision"] == "release" for r in selected)} | {len(selected)} |')
    report += ['', 'budget_recommendations.tsv selects the smallest declared control budget that releases every comparison involving the anchor winner.',
               'Freeze this directory, then run check-design with the complete assessment realization scores to measure support and coverage.', '']
    _emit(output, {'design.json': _encoded(artifact), 'recommendations.tsv': _tabular(decisions),
                   'calibration.tsv': _tabular(calibration), 'budget_recommendations.tsv': _tabular(recommendations),
                   'report.md': '\n'.join(report).encode()}, inputs)
    return artifact


def _validated_design(path):
    artifact = _json(path)
    _schema(artifact, {'artifact_type', 'schema_version', 'configuration', 'inputs', 'development_cases',
                       'calibration', 'cases', 'decisions', 'budget_recommendations'}, set(), 'Frozen design')
    if artifact['artifact_type'] != 'reference_design_organizer' or type(artifact['schema_version']) is not int or artifact['schema_version'] != 1:
        raise ValueError('Unsupported frozen design')
    config = _configuration(artifact['configuration'])
    has_geometry = 'geometry_scaled' in _policies(config)
    budgets = {b['id'] for b in config['budgets']}
    pairs = set(itertools.combinations(config['models'], 2))
    calibration = artifact['calibration']
    if not isinstance(calibration, list) or len(calibration) != len(budgets) * len(pairs):
        raise ValueError('Frozen calibration support differs from its configuration')
    keys = set()
    for row in calibration:
        _schema(row, {'budget', 'model_a', 'model_b', 'B', 'K', 'geometry_available', 'geometry_reason',
                      'development_cases', 'positive_geometry_cases', 'zero_geometry_shift_cases'}, set(), 'Calibration row')
        for field in ('budget', 'model_a', 'model_b'):
            label(row[field], field)
        key = (row['budget'], row['model_a'], row['model_b'])
        if key in keys or row['budget'] not in budgets or (row['model_a'], row['model_b']) not in pairs:
            raise ValueError('Duplicate or unsupported frozen calibration row')
        keys.add(key)
        _number(row['B'], 'B', minimum=0)
        if row['K'] is not None:
            _number(row['K'], 'K', minimum=0)
        if type(row['geometry_available']) is not bool or row['geometry_available'] and row['K'] is None:
            raise ValueError('Invalid geometry availability in frozen calibration')
        label(row['geometry_reason'], 'Geometry reason')
        for field in ('development_cases', 'positive_geometry_cases', 'zero_geometry_shift_cases'):
            if type(row[field]) is not int or row[field] < (1 if field == 'development_cases' else 0):
                raise ValueError('Invalid frozen calibration case counts')
        positive, zero = row['positive_geometry_cases'], row['zero_geometry_shift_cases']
        if positive + zero > row['development_cases'] or row['geometry_available'] != (positive > 0 and zero == 0):
            raise ValueError('Inconsistent frozen geometry calibration counts')
        if not has_geometry and (row['K'] is not None or positive or zero or row['geometry_available']):
            raise ValueError('Frozen calibration contains geometry without a geometry policy')
        expected_reason = ('geometry_not_supplied' if not has_geometry else 'available' if row['geometry_available'] else
                           'development_zero_geometry_shift' if zero else 'no_positive_development_geometry')
        if row['geometry_reason'] != expected_reason or (row['K'] is None) != (positive == 0):
            raise ValueError('Inconsistent frozen geometry calibration status')
    development = artifact['development_cases']
    if not isinstance(development, list) or not development:
        raise ValueError('Frozen development cases must be nonempty')
    development_ids = set()
    for case in development:
        _schema(case, {'case', 'unit'}, set(), 'Frozen development case')
        label(case['case'], 'Case')
        label(case['unit'], 'Unit')
        if case['case'] in development_ids:
            raise ValueError('Duplicate frozen development case')
        development_ids.add(case['case'])
    if any(row['development_cases'] != len(development) for row in calibration):
        raise ValueError('Frozen development case counts differ from calibration')
    input_fields = {'manifest', *PATH_FIELDS} | ({'geometry'} if has_geometry else set())
    _schema(artifact['inputs'], input_fields, set(), 'Frozen inputs')
    for item in artifact['inputs'].values():
        _schema(item, {'name', 'sha256'}, set(), 'Frozen input')
        label(item['name'], 'Input name')
        digest = item['sha256']
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid frozen input digest')
    cases = artifact['cases']
    if not isinstance(cases, list) or not cases:
        raise ValueError('Frozen design cases must be nonempty')
    seen = set()
    for case in cases:
        _schema(case, {'case', 'unit', 'anchor_scores', 'geometry'}, set(), 'Frozen case')
        label(case['case'], 'Case')
        label(case['unit'], 'Unit')
        if case['case'] in seen or case['case'] in development_ids:
            raise ValueError('Duplicate or overlapping frozen assessment case')
        seen.add(case['case'])
        _schema(case['anchor_scores'], budgets, set(), 'Frozen anchors')
        if has_geometry:
            _schema(case['geometry'], budgets, set(), 'Frozen geometry')
        elif case['geometry'] is not None:
            raise ValueError('Frozen geometry must be null when geometry was not supplied')
        for bid in budgets:
            _schema(case['anchor_scores'][bid], set(config['models']), set(), 'Frozen model scores')
            for value in case['anchor_scores'][bid].values():
                _number(value, 'Frozen anchor score')
            if has_geometry:
                _number(case['geometry'][bid], 'Frozen geometry', minimum=0)
    if artifact['decisions'] != _decisions(cases, calibration, config):
        raise ValueError('Frozen decisions differ from their calibration and anchors')
    if artifact['budget_recommendations'] != _budget_recommendations(cases, artifact['decisions'], config):
        raise ValueError('Frozen budget recommendations differ from their decisions')
    return artifact


def check_design(design_path, assessment_scores_path, output_path):
    """Evaluate frozen decisions against complete supplied assessment realizations."""
    from .cli import _destination
    design_path, assessment_scores_path, output = Path(design_path), Path(assessment_scores_path), Path(output_path)
    _destination(output)
    artifact = _validated_design(design_path)
    config = artifact['configuration']
    assessment = _scores(assessment_scores_path, config)
    if set(assessment) != {c['case'] for c in artifact['cases']}:
        raise ValueError('Assessment scores must cover exactly the frozen assessment cases')
    for case in artifact['cases']:
        actual = assessment[case['case']]
        if actual['unit'] != case['unit']:
            raise ValueError('Assessment unit differs from frozen anchor unit')
        for bid, scores in case['anchor_scores'].items():
            if actual['scores'][bid][config['anchor_realization']] != scores:
                raise ValueError('Assessment anchor values differ from the frozen design')
    gap, tau = config['minimum_gap'], config['numerical_tolerance']
    rows = []
    for decision in artifact['decisions']:
        values = assessment[decision['case']]['scores'][decision['budget']]
        a, b = decision['model_a'], decision['model_b']
        sign = 1 if decision['favored_model'] == a else -1
        margins = [_margin(scores, a, b, config) * sign for scores in values.values()]
        released = decision['decision'] == 'release'
        supported = all(m > gap + tau for m in margins) if released else None
        rows.append(dict(case=decision['case'], unit=decision['unit'], budget=decision['budget'],
                         policy=decision['policy'], model_a=a, model_b=b, decision=decision['decision'],
                         favored_model=decision['favored_model'], realizations=len(margins), supported=supported,
                         unsupported_release=released and not supported,
                         strict_reversal=released and any(m < -tau for m in margins),
                         numerical_tie=released and any(abs(m) <= tau for m in margins),
                         within_gap=released and any(0 < m <= gap + tau for m in margins)))
    summary, comparisons = [], []
    for budget in config['budgets']:
        bid = budget['id']
        budget_summary = []
        for policy in _policies(config):
            selected = [row for row in rows if row['budget'] == bid and row['policy'] == policy]
            released = sum(row['decision'] == 'release' for row in selected)
            unsupported = sum(row['unsupported_release'] for row in selected)
            budget_summary.append(dict(budget=bid, policy=policy, comparisons=len(selected), released=released,
                                       coverage=released / len(selected), unsupported_releases=unsupported,
                                       unsupported_fraction=unsupported / released if released else None,
                                       strict_reversals=sum(row['strict_reversal'] for row in selected),
                                       numerical_ties=sum(row['numerical_tie'] for row in selected),
                                       within_gap=sum(row['within_gap'] for row in selected)))
        summary.extend(budget_summary)
        if 'geometry_scaled' not in _policies(config):
            continue
        repeated, geometry = budget_summary[1], budget_summary[2]
        comparable = geometry['released'] > 0
        favorable = comparable and geometry['unsupported_releases'] <= repeated['unsupported_releases'] and geometry['coverage'] >= repeated['coverage']
        strict_gain = favorable and (geometry['unsupported_releases'] < repeated['unsupported_releases'] or geometry['coverage'] > repeated['coverage'])
        comparisons.append(dict(budget=bid, comparable=comparable,
                                no_more_unsupported_and_no_lower_coverage=favorable,
                                outcome='strict_gain' if strict_gain else 'parity' if favorable else 'no_gain' if comparable else 'insufficient_releases'))
    pair_assessments = {(row['case'], row['budget'], row['policy'], row['model_a'], row['model_b']): row for row in rows}
    budget_assessments = []
    for recommendation in artifact['budget_recommendations']:
        released = recommendation['decision'] == 'release'
        opponents = []
        if released:
            winner = recommendation['model']
            opponents = [pair_assessments[(recommendation['case'], recommendation['budget'], recommendation['policy'], a, b)]
                         for a, b in itertools.combinations(config['models'], 2) if winner in (a, b)]
        supported = all(row['supported'] for row in opponents) if released else None
        budget_assessments.append(dict(**recommendation, supported=supported,
                                       unsupported_recommendation=released and not supported,
                                       opponents=len(opponents), supported_opponents=sum(row['supported'] for row in opponents),
                                       realizations=opponents[0]['realizations'] if opponents else None,
                                       strict_reversal=any(row['strict_reversal'] for row in opponents),
                                       numerical_tie=any(row['numerical_tie'] for row in opponents),
                                       within_gap=any(row['within_gap'] for row in opponents),
                                       assessment_reason='held_by_design' if not released else 'all_opponents_supported' if supported else 'one_or_more_opponents_unsupported'))
    result = dict(schema_version=1, scope=config['evaluation_scope'], summaries=summary,
                  geometry_vs_repeated=comparisons, assessments=rows, budget_assessments=budget_assessments,
                  inputs={'design': dict(name=design_path.name, sha256=_digest(design_path)),
                          'assessment_scores': dict(name=assessment_scores_path.name, sha256=_digest(assessment_scores_path))})
    report = ['# Frozen organizer check', '', f'Scope: {_display(config["evaluation_scope"])}',
              'A release is supported when its oriented margin exceeds the declared gap and numerical tolerance in every recorded assessment realization.',
              'Coverage is the fraction of declared case/budget/model-pair comparisons released. Unsupported fractions use released comparisons as the denominator; an all-hold fraction is undefined.', '',
              '| Budget | Policy | Released | Coverage | Unsupported | Unsupported fraction |', '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in summary:
        fraction = 'undefined' if row['unsupported_fraction'] is None else f'{row["unsupported_fraction"]:.6g}'
        report.append(f'| {_display(row["budget"])} | {row["policy"]} | {row["released"]} | {row["coverage"]:.6g} | {row["unsupported_releases"]} | {fraction} |')
    report += [''] + [f'Geometry versus repeated primary at {_display(row["budget"])}: {row["outcome"]}.' for row in comparisons]
    report += ['budget_assessments.tsv checks each frozen minimum-budget winner against every opponent across all recorded realizations; held recommendations remain marked hold.',
               'These results describe the recorded realizations under the frozen development calibration.', '']
    _emit(output, {'check.json': _encoded(result), 'assessments.tsv': _tabular(rows),
                   'policy_summary.tsv': _tabular(summary), 'budget_assessments.tsv': _tabular(budget_assessments),
                   'report.md': '\n'.join(report).encode()}, result['inputs'])
    return result
