#!/usr/bin/env python3
"""Check the delivered Figure 3 metric summaries against their original tables."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

COMPONENT = 'Figure_3_metric_summaries_v1818'
NUMERIC = tuple(prefix + suffix for prefix in ('raw_', 'oriented_')
                for suffix in ('mean', 'q05', 'q50', 'q95'))
KEY = ('protocol_id', 'tier', 'resource_id', 'depth', 'pattern_id', 'method_id')
FAMILY = ('protocol_id', 'tier', 'resource_id', 'depth', 'pattern_id')
NC = 'NO_CHANGE_DIRECT_V1'
PEARSON = 'M8_PEARSON_DELTA_V1'
COSINE = 'M8_LOGFC_COSINE_V1'
RETRIEVAL = 'M8_CENTROID_RETRIEVAL_V1'
TABLES = {
    'resource': (
        'data/derived/v8/f4/F4_RESOURCE_METHOD_SUMMARY_V1.tsv',
        '525f8ad97d4aa0802d972495c784808b5c7230cbf53dc992a716c22a91fd1d44',
        COMPONENT + '/F4_RESOURCE_METHOD_SUMMARY_MODEL_COMPLETE_V2.tsv',
        '7e69fd3510885c092fd7505b06b2376804aed10e7080eed6edf5038f57605350'),
    'root': (
        'data/derived/v8/f4/F4_ROOT_METHOD_SUMMARY_V1.tsv',
        'b99a8778910e8b1250d8c255ea4cfbdcad5fcefadd18e6b373c5088646c9484f',
        COMPONENT + '/F4_ROOT_METHOD_SUMMARY_MODEL_COMPLETE_V2.tsv',
        '0cca03fac97658ea78a7a8e3dd505aec59b36522738bce61cba250357e753607'),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Expected regular file: ' + path.name)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        require(len(reader.fieldnames) == len(set(reader.fieldnames)), 'Duplicate column')
        rows = list(reader)
    require(all(None not in row and None not in row.values() for row in rows), 'Malformed table row')
    return rows


def identifier(row, fields=KEY):
    return tuple(row[field] for field in fields)


def check_row(row):
    present = [row[name] != '' for name in NUMERIC]
    require(all(present) or not any(present), 'Partially populated numeric summary')
    defined = all(present)
    require(defined == row['status'].startswith('PASS_'), 'Score availability and status disagree')
    require(row['complete_family_status'] in {'COMPLETE', 'INCOMPLETE'}, 'Unknown family status')
    require(row['allocation_count'] == '1000', 'Allocation count differs')
    if not defined:
        require(bool(row['failure_reason']), 'Undefined summary lacks a reason')
        require(row['complete_family_status'] == 'INCOMPLETE', 'Undefined score in a complete family')
        return False
    require(not row['failure_reason'], 'Defined summary has a failure reason')
    require(not (row['method_id'] == NC and row['protocol_id'] in {PEARSON, COSINE}),
            'Zero-effect Pearson or cosine must remain undefined')
    require(row['protocol_id'] != RETRIEVAL, 'Incomplete retrieval strata must remain undefined')
    values = {name: float(row[name]) for name in NUMERIC}
    require(all(math.isfinite(x) for x in values.values()), 'Nonfinite summary')
    for prefix in ('raw_', 'oriented_'):
        require(values[prefix+'q05'] <= values[prefix+'q50'] <= values[prefix+'q95'],
                'Quantiles are not ordered')
    require(row['direction'] in {'HIGHER_IS_BETTER', 'LOWER_IS_BETTER'}, 'Unknown direction')
    reverse = row['direction'] == 'LOWER_IS_BETTER'
    for suffix in ('mean', 'q05', 'q50', 'q95'):
        other = {'q05': 'q95', 'q95': 'q05'}.get(suffix, suffix) if reverse else suffix
        expected = (-1 if reverse else 1) * values['raw_'+other]
        require(math.isclose(values['oriented_'+suffix], expected, rel_tol=1e-12, abs_tol=1e-14),
                'Raw and oriented summaries disagree')
    return True


def compare_tables(original, current, level):
    fields = KEY + (('root_id',) if level == 'root' else ())
    old = {identifier(row, fields): row for row in original}
    new = {identifier(row, fields): row for row in current}
    require(len(old) == len(original) and len(new) == len(current), 'Duplicate summary key')
    require(set(old) == set(new), 'Summary row roster changed')
    counts = Counter()
    family_states = defaultdict(set)
    for key, row in new.items():
        before = old[key]
        defined = check_row(row)
        original_defined = bool(before['raw_mean'])
        expected_family = 'COMPLETE' if original_defined else 'INCOMPLETE'
        require(row['complete_family_status'] == expected_family, 'Original family eligibility changed')
        require(row['complete_family_failure_reason'] == before['failure_reason'],
                'Original family failure reason changed')
        family_states[identifier(row, FAMILY)].add(row['complete_family_status'])
        if original_defined:
            require(all(row[name] == value for name, value in before.items()),
                    'Previously populated summary changed')
            counts['preserved_rows'] += 1
            counts['preserved_numeric_strings'] += len(NUMERIC)
        else:
            unchanged = set(before) - set(NUMERIC) - {'status', 'failure_reason'}
            require(all(row[name] == before[name] for name in unchanged), 'Summary metadata changed')
            counts['restored_rows' if defined else 'undefined_rows'] += 1
            if defined:
                require(row['protocol_id'] in {PEARSON, COSINE} and row['method_id'] != NC,
                        'Unexpected restored metric or model')
        counts['total_rows'] += 1
    require(all(len(states) == 1 for states in family_states.values()), 'Inconsistent family eligibility')
    counts['complete_families'] = sum(states == {'COMPLETE'} for states in family_states.values())
    counts['incomplete_families'] = sum(states == {'INCOMPLETE'} for states in family_states.values())
    return dict(counts)


def check_root_means(resource_rows, root_rows):
    groups = defaultdict(list)
    for row in root_rows:
        groups[identifier(row)].append(row)
    require(set(groups) == {identifier(row) for row in resource_rows}, 'Root and resource rosters differ')
    maximum = 0.0
    checked = 0
    for row in resource_rows:
        roots = groups[identifier(row)]
        require(len(roots) == 8 and len({r['root_id'] for r in roots}) == 8,
                'A resource summary requires all eight distinct roots')
        require(row['root_count'] == '8', 'Resource root count differs')
        require({r['complete_family_status'] for r in roots} == {row['complete_family_status']},
                'Resource-wide family eligibility differs on root rows')
        defined = [bool(r['raw_mean']) for r in roots]
        require(bool(row['raw_mean']) == all(defined), 'Incomplete roots cannot yield a resource summary')
        if not row['raw_mean']:
            continue
        for name in ('raw_mean', 'oriented_mean'):
            expected = math.fsum(float(r[name]) for r in roots) / 8
            error = abs(float(row[name]) - expected)
            maximum = max(maximum, error)
            require(math.isclose(float(row[name]), expected, rel_tol=1e-12, abs_tol=1e-14),
                    'Resource mean differs from the equal eight-root mean')
        checked += 1
    return {'defined_resource_means_checked': checked, 'maximum_absolute_difference': maximum}


def validate(source_data_root):
    root = Path(source_data_root).resolve()
    current = {}
    checks = {}
    for level, (old_name, old_sha, new_name, new_sha) in TABLES.items():
        require(sha(root/old_name) == old_sha, 'Original table hash differs: '+old_name)
        require(sha(root/new_name) == new_sha, 'V2 table hash differs: '+new_name)
        current[level] = read_rows(root/new_name)
        checks[level] = compare_tables(read_rows(root/old_name), current[level], level)
    expected = {'resource': (1480, 640, 560, 280), 'root': (11840, 5120, 4480, 2240)}
    for level, values in expected.items():
        require(tuple(checks[level][k] for k in ('total_rows', 'preserved_rows', 'restored_rows', 'undefined_rows'))
                == values, 'Summary counts differ: '+level)
        require(checks[level]['complete_families'] == 80 and checks[level]['incomplete_families'] == 105,
                'Complete-family roster differs')
    aggregate = check_root_means(current['resource'], current['root'])
    require(aggregate['defined_resource_means_checked'] == 1200, 'Defined resource count differs')
    return {'status': 'PASS_FIGURE_3_METRIC_SUMMARIES_V1818',
            'validation_scope': 'DELIVERED_TABLE_HASHES_PRESERVATION_AND_AGGREGATION',
            'raw_score_recomputation': False, 'tables': checks, 'aggregation': aggregate}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).resolve().parent.parent
    parser.add_argument('--source-data-root', type=Path,
                        default=default if (default/COMPONENT).is_dir() else None)
    args = parser.parse_args()
    if args.source_data_root is None:
        parser.error('--source-data-root is required for the public-code copy')
    print(json.dumps(validate(args.source_data_root), indent=2, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as error:
        print(json.dumps({'status': 'FAIL_FIGURE_3_METRIC_SUMMARIES_V1818', 'error': str(error)}))
        raise SystemExit(1)
