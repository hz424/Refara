"""Recompute frozen model choices and paired donor inference from task losses.

This numerical replay needs NumPy and SciPy, and no model weights, PyTorch,
raw assay files, or private paths. A local pre-outcome freeze is not a public
preregistration. Read protocol.json for the scope of this external validation.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy

from paired_statistics import (EXECUTED_SOURCE_SHA256, donor_weights, paired_loss_summary,
                               studentized_bootstrap_sensitivity)


SCHEMA = 'REFARA_GSE181897_REFERENCE_DESIGN_V1'
FAMILIES = ('NC', 'CM', 'TW', 'PCA', 'RBF', 'scGen', 'CellOT')
CONDITIONS = ('IFN-beta', 'IFN-gamma', 'TNF-alpha')
RULES = ('shared_all_B', 'O_quarter', 'O_half', 'O_three_quarters', 'crossfit_2', 'crossfit_4')
DIAGNOSTICS = ('overlap_half', 'crossfit2_overlap_all')
EXPECTED_RTOL = 1e-11
EXPECTED_ATOL = 1e-12


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path):
    path = Path(path)
    with (gzip.open if path.suffix == '.gz' else open)(path, 'rt', newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def write_rows(path, rows):
    require(bool(rows), 'Empty replay output table')
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def portable_path(root, relative):
    root = Path(root).resolve()
    path = Path(relative)
    require(not path.is_absolute() and '..' not in path.parts, 'Capsule paths must remain relative')
    resolved = (root / path).resolve()
    require(resolved.is_relative_to(root), 'Capsule path or symlink escapes its directory')
    return resolved


def load_capsule(directory):
    root = Path(directory).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    require(manifest['schema'] == 'REFARA_CAPSULE_MANIFEST_V1', 'Unknown capsule manifest')
    for name, digest in manifest['files'].items():
        require(sha256(portable_path(root, name)) == digest, f'Capsule artifact changed: {name}')
    required = {'protocol.json', 'model_contract.json', 'replay.py', 'paired_statistics.py',
                'source_data/tasks.tsv.gz', 'source_data/selection_losses.tsv.gz',
                'source_data/assessment_losses.tsv.gz', 'source_data/expected.json'}
    require(required <= set(manifest['files']), 'Capsule manifest is incomplete')
    protocol = json.loads((root / 'protocol.json').read_text())
    require(protocol['schema'] == SCHEMA, 'Unknown scientific capsule schema')
    require(protocol['statistics_executed_source_sha256'] == EXECUTED_SOURCE_SHA256,
            'Frozen numerical source identity differs')
    require(sha256(root / 'model_contract.json') == protocol['model_contract']['sha256'],
            'Public model contract differs from protocol')
    require(tuple(protocol['models']) == FAMILIES and tuple(protocol['conditions']) == CONDITIONS,
            'Frozen model or condition order differs')
    require(protocol['recommended_rule'] in RULES, 'A diagnostic cannot become the recommended rule')
    require(protocol['primary_budget'] == 16 and protocol['secondary_budgets'] == [8, 32] and
            protocol['primary_deployment_n'] == 8 and protocol['secondary_deployment_n'] == 16,
            'Budget or deployment target differs')
    require(protocol['confidence'] == .95 and protocol['minimum_relative_reduction'] == .05 and
            protocol['bootstrap_resamples'] == 19999 and protocol['bootstrap_seed'] == 20260919 and
            protocol['max_inference_ratio'] == 4, 'Frozen decision criterion differs')
    return protocol, read_rows(root / 'source_data/tasks.tsv.gz'), read_rows(root / 'source_data/selection_losses.tsv.gz'), read_rows(root / 'source_data/assessment_losses.tsv.gz')


def rule_cost(budget, rule):
    require(rule in RULES + DIAGNOSTICS, 'Unknown reference policy')
    if rule == 'O_quarter':
        total = unique = budget // 4
    elif rule in ('O_half', 'overlap_half'):
        total = unique = budget // 2
    elif rule == 'O_three_quarters':
        total = unique = 3 * budget // 4
    elif rule == 'crossfit_4':
        total, unique = 3 * budget, budget
    else:
        total = unique = budget
    return {'input_control_forwards_without_cache': total,
            'unique_input_control_forwards_with_cache': unique,
            'uncached_ratio_to_shared': total / budget,
            'cached_ratio_to_shared': unique / budget,
            'physical_control_cells': budget}


def mean_by_donor(rows, field='loss'):
    donors = defaultdict(list)
    for row in rows:
        donors[row['donor']].append(float(row[field]))
    require(bool(donors), 'No eligible donors')
    return float(np.mean([np.mean(values) for values in donors.values()])), len(donors)


def validate_tasks(tasks):
    keys = [(r['donor'], r['context'], r['condition']) for r in tasks]
    require(len(keys) == len(set(keys)), 'Duplicate task')
    contexts, partitions, counts = defaultdict(set), defaultdict(set), defaultdict(int)
    for row in tasks:
        require(row['partition'] in ('selection', 'assessment') and int(row['control_cells_available']) >= 16,
                'Task outside frozen eligibility')
        require((row['secondary_32_eligible'] == 'True') == (int(row['control_cells_available']) >= 32),
                'Secondary subset does not match metadata support')
        contexts[row['donor'], row['context']].add(row['condition'])
        partitions[row['donor']].add(row['partition'])
        counts[row['donor']] += 1
    require(all(value == set(CONDITIONS) for value in contexts.values()), 'Incomplete three-condition context')
    require(all(len(value) == 1 for value in partitions.values()), 'Donor leaks between selection and assessment')
    require(all(abs(float(r['within_donor_weight']) - 1 / counts[r['donor']]) < 1e-13 for r in tasks),
            'Frozen donor/context/condition weights differ')


def select_families(protocol, tasks, losses):
    selection_tasks = [r for r in tasks if r['partition'] == 'selection']
    expected = {(r['donor'], r['context'], r['condition'], budget, rule, family)
                for r in selection_tasks for budget in (8, 16, 32)
                if budget != 32 or r['secondary_32_eligible'] == 'True'
                for rule in RULES + DIAGNOSTICS for family in FAMILIES}
    keys = [(r['donor'], r['context'], r['condition'], int(r['budget']), r['rule'], r['family']) for r in losses]
    require(len(keys) == len(set(keys)) and set(keys) == expected, 'Selection table does not cover exactly the frozen eligible tasks')
    buckets = defaultdict(list)
    for row in losses:
        require(np.isfinite(float(row['loss'])) and float(row['loss']) >= 0, 'Invalid selection MSE')
        require(int(row['seeds']) == (1 if row['family'] in FAMILIES[:5] else 3), 'Selection seed aggregation differs')
        buckets[int(row['budget']), row['rule'], row['family']].append(row)
    decisions = {'families': list(FAMILIES), 'recommended_rule': protocol['recommended_rule'],
                 'tie_tolerance': protocol['tie_tolerance'], 'budgets': {}}
    for budget in (8, 16, 32):
        decisions['budgets'][str(budget)] = {}
        for rule in RULES + DIAGNOSTICS:
            risks = []
            for family in FAMILIES:
                rows = buckets[budget, rule, family]
                mean, count = mean_by_donor(rows)
                risks.append(mean)
            chosen = int(np.flatnonzero(np.asarray(risks) <= min(risks) + protocol['tie_tolerance'])[0])
            decisions['budgets'][str(budget)][rule] = {
                'family': FAMILIES[chosen], 'family_risks': dict(zip(FAMILIES, risks)),
                'donors': count, 'tasks': len(rows), 'cost': rule_cost(budget, rule)}
    return decisions


def analyse(protocol, tasks, selection_losses, assessment_losses):
    validate_tasks(tasks)
    decisions = select_families(protocol, tasks, selection_losses)
    assessment_tasks = [r for r in tasks if r['partition'] == 'assessment']
    lookup = {(r['donor'], r['context'], r['condition'], int(r['deployment_n']), r['family']): r for r in assessment_losses}
    expected = {(r['donor'], r['context'], r['condition'], n, family)
                for r in assessment_tasks for n in (8, 16)
                if n == 8 or int(r['control_cells_available']) >= 24
                for family in FAMILIES}
    require(len(lookup) == len(assessment_losses) and set(lookup) == expected, 'Assessment task/family coverage differs')
    for row in assessment_losses:
        require(all(np.isfinite(float(row[key])) and float(row[key]) >= 0 for key in ('loss', 'treated_state_loss')),
                'Invalid assessment MSE')
        require(int(row['seeds']) == (1 if row['family'] in FAMILIES[:5] else 3), 'Assessment seed aggregation differs')
    all_model_losses = []
    for n in (8, 16):
        eligible = [r for r in assessment_tasks if n == 8 or int(r['control_cells_available']) >= 24]
        weights = donor_weights([r['donor'] for r in eligible], [float(r['within_donor_weight']) for r in eligible])
        for family in FAMILIES:
            rows = [lookup[r['donor'], r['context'], r['condition'], n, family] for r in eligible]
            all_model_losses.append({'deployment_n': n, 'family': family,
                                     'donors': len({r['donor'] for r in eligible}), 'tasks': len(eligible),
                                     'mean_effect_loss': float(np.average([float(r['loss']) for r in rows], weights=weights)),
                                     'mean_treated_state_loss': float(np.average([float(r['treated_state_loss']) for r in rows], weights=weights))})
    summaries, primary_rows = [], []
    for budget in (8, 16, 32):
        for n in (8, 16):
            eligible = [r for r in assessment_tasks if (budget != 32 or r['secondary_32_eligible'] == 'True') and
                        (n != 16 or int(r['control_cells_available']) >= 24)]
            require(bool(eligible), 'Frozen evaluation subset is empty')
            shared_family = decisions['budgets'][str(budget)]['shared_all_B']['family']
            for rule in RULES + DIAGNOSTICS:
                family = decisions['budgets'][str(budget)][rule]['family']
                paired = []
                for task in eligible:
                    key = task['donor'], task['context'], task['condition'], n
                    shared, selected = lookup[key + (shared_family,)], lookup[key + (family,)]
                    require(shared['control_pool'] == selected['control_pool'] and shared['treated_pool'] == selected['treated_pool'],
                            'Compared family measurements belong to different physical pools')
                    paired.append({'donor': task['donor'], 'shared_loss': float(shared['loss']), 'rule_loss': float(selected['loss']),
                                   'shared_treated_state_loss': float(shared['treated_state_loss']),
                                   'rule_treated_state_loss': float(selected['treated_state_loss']),
                                   'within_donor_weight': float(task['within_donor_weight'])})
                weights = donor_weights([r['donor'] for r in paired], [r['within_donor_weight'] for r in paired])
                means = {key: float(np.average([r[key] for r in paired], weights=weights))
                         for key in ('shared_loss', 'rule_loss', 'shared_treated_state_loss', 'rule_treated_state_loss')}
                s, r = means['shared_loss'], means['rule_loss']
                ss, sr = means['shared_treated_state_loss'], means['rule_treated_state_loss']
                primary = budget == 16 and n == 8 and rule == protocol['recommended_rule']
                summaries.append({'budget': budget, 'deployment_n': n, 'rule': rule,
                                  'selected_family': family, 'shared_family': shared_family,
                                  'donors': len({x['donor'] for x in paired}), 'tasks': len(paired),
                                  'mean_shared_loss': s, 'mean_rule_loss': r,
                                  'relative_loss_reduction': (s - r) / s if s else '',
                                  'mean_shared_treated_state_loss': ss, 'mean_rule_treated_state_loss': sr,
                                  'treated_state_relative_loss_reduction': (ss - sr) / ss if ss else '',
                                  'uncached_inference_ratio': rule_cost(budget, rule)['uncached_ratio_to_shared'],
                                  'confirmatory_primary': primary})
                if primary:
                    primary_rows, primary_weights = paired, weights
    require(len(primary_rows) == len(assessment_tasks), 'Primary endpoint excludes eligible assessment tasks')
    recommended = protocol['recommended_rule']
    selection_tasks = [r for r in tasks if r['partition'] == 'selection']
    shared_work = len(selection_tasks) * 6 * 16
    recommended_work = len(selection_tasks) * 6 * rule_cost(16, recommended)['input_control_forwards_without_cache']
    kwargs = {'target_reduction': protocol['minimum_relative_reduction'], 'confidence': protocol['confidence'],
              'minimum_units': protocol['minimum_inference_donors'], 'shared_inference_calls': shared_work,
              'rule_inference_calls': recommended_work, 'inference_cap': protocol['max_inference_ratio'],
              'scope': protocol['inference_scope']}
    shared = [r['shared_loss'] for r in primary_rows]
    rule = [r['rule_loss'] for r in primary_rows]
    units = [r['donor'] for r in primary_rows]
    paired_t = paired_loss_summary(shared, rule, units, primary_weights, **kwargs)
    bootstrap = studentized_bootstrap_sensitivity(shared, rule, units, primary_weights,
                seed=protocol['bootstrap_seed'], resamples=protocol['bootstrap_resamples'], **kwargs)
    summary_lookup = {(r['budget'], r['deployment_n'], r['rule']): r for r in summaries}
    diagnostics = []
    for budget in (8, 16, 32):
        for n in (8, 16):
            for overlap, disjoint in (('overlap_half', 'O_half'), ('crossfit2_overlap_all', 'crossfit_2')):
                left, right = summary_lookup[budget, n, overlap], summary_lookup[budget, n, disjoint]
                diagnostics.append({'budget': budget, 'deployment_n': n, 'overlap_rule': overlap, 'disjoint_rule': disjoint,
                                    'overlap_selected_family': left['selected_family'], 'disjoint_selected_family': right['selected_family'],
                                    'donors': left['donors'], 'tasks': left['tasks'], 'benchmark_input_cells_per_fold': budget // 2,
                                    'mean_overlap_selected_loss': left['mean_rule_loss'], 'mean_disjoint_selected_loss': right['mean_rule_loss'],
                                    'disjoint_loss_reduction_vs_overlap':
                                    (left['mean_rule_loss'] - right['mean_rule_loss']) / left['mean_rule_loss'] if left['mean_rule_loss'] else ''})
    return {'schema': SCHEMA, 'source_frozen_protocol_sha256': protocol['source_frozen_protocol_sha256'],
            'recommended_rule': recommended, 'primary_budget': 16, 'primary_deployment_n': 8,
            'scope': protocol['inference_scope'], 'endpoint': protocol['primary_endpoint'],
            'joint_practical_benefit': bootstrap['joint_practical_benefit'], 'joint_adoption': bootstrap['joint_adoption'],
            'paired_t': paired_t, 'studentized_bootstrap': bootstrap,
            'selection_decisions': decisions, 'policy_summary': summaries, 'donor_losses': paired_t['units'],
            'all_model_losses': all_model_losses,
            'mechanism_diagnostics': diagnostics,
            'pool_sensitivity': {'status': 'unavailable_multiple_membership',
                                 'reason': 'Each loss depends on distinct control and treated physical pools; a one-pool-per-row covariance is not applicable.'},
            'compute': {'shared_work': shared_work, 'recommended_work': recommended_work, 'ratio': recommended_work / shared_work},
            'treated_state_secondary': protocol['treated_state_secondary']}


def expected_projection(result):
    """Stable scientific values compared with the executed full-data analysis."""
    return {key: result[key] for key in ('recommended_rule', 'joint_practical_benefit', 'joint_adoption',
            'paired_t', 'studentized_bootstrap', 'selection_decisions', 'policy_summary',
            'donor_losses', 'mechanism_diagnostics', 'all_model_losses', 'compute')}


def compare_expected(actual, expected, path='result'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f'Expected keys differ: {path}')
        for key in expected:
            compare_expected(actual[key], expected[key], f'{path}.{key}')
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f'Expected list differs: {path}')
        for index, (a, e) in enumerate(zip(actual, expected)):
            compare_expected(a, e, f'{path}[{index}]')
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        require(isinstance(actual, (int, float)) and np.isclose(actual, expected, rtol=EXPECTED_RTOL, atol=EXPECTED_ATOL),
                f'Expected numerical value differs: {path}')
    else:
        require(actual == expected, f'Expected value differs: {path}')


def run(capsule, output, *, verify_expected=True):
    output = Path(output)
    require(not output.exists(), 'Replay output already exists; choose a new directory')
    protocol, tasks, selection, assessment = load_capsule(capsule)
    result = analyse(protocol, tasks, selection, assessment)
    if verify_expected:
        expected = json.loads((Path(capsule) / 'source_data' / 'expected.json').read_text())
        compare_expected(expected_projection(result), expected)
    result['published_summary_verified'] = verify_expected
    result['replay_runtime'] = {'numpy': np.__version__, 'scipy': scipy.__version__,
                                'expected_rtol': EXPECTED_RTOL, 'expected_atol': EXPECTED_ATOL,
                                'decision_boundary_tolerance': 0,
                                'note': 'Expected-value tolerance accommodates special-function rounding across library versions; practical-benefit boundaries remain strictly zero.'}
    output.mkdir(parents=True)
    write_json(output / 'RESULTS.json', result)
    write_rows(output / 'policy_summary.tsv', result['policy_summary'])
    write_rows(output / 'primary_donor_losses.tsv', result['donor_losses'])
    write_rows(output / 'mechanism_diagnostics.tsv', result['mechanism_diagnostics'])
    write_rows(output / 'all_model_losses.tsv', result['all_model_losses'])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capsule', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.capsule, args.output)
    print(json.dumps({'published_summary_verified': result['published_summary_verified'],
                      'joint_practical_benefit': result['joint_practical_benefit'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
