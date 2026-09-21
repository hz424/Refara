#!/usr/bin/env python3
"""Combine all new counts with the unchanged archived Figure 5 source rows."""
from __future__ import annotations
import csv
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
COUNT = 'outer_replicates_any_null_direction_rejected'


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def write(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=HERE / 'results')
    args = parser.parse_args()
    original = HERE / 'source_data/F2D_PREEXISTING_NEGATIVE_CONTROLS_V1.tsv'
    old = read(original)
    assert len(old) == 20
    reference = {(int(r['root_count_R']), r['procedure_id']): r for r in read(HERE / 'source_data/FIGURE_5B_ARCHIVED_EVENT_COUNTS.tsv')}
    assert all(int(r['events']) == int(reference[(int(r['root_count_R']), r['procedure_id'])]['events']) for r in old)
    templates = {r['procedure_id']: r for r in old if r['root_count_R'] == '8'}
    added = []
    for n in (2, 3, 4, 6):
        path = args.results_dir / f'root_{n:02d}_counts.json'
        payload = json.loads(path.read_text())
        assert payload['low_n_frozen_spec_sha256'] == hashlib.sha256((HERE / 'LOW_N_FROZEN_SPEC.json').read_bytes()).hexdigest()
        cell = payload['cell']
        assert cell['root_count'] == n and cell['law_id'] == 'C02'
        assert cell['outer_counts'] == dict(attempted_replicates=2500, successful_replicates=2500, failed_replicates=0)
        assert set(cell['inferential_procedures']) == set(templates)
        for procedure, record in cell['inferential_procedures'].items():
            events = record['counts'][COUNT]
            trials = 2500
            row = dict(templates[procedure])
            row.update(root_count_R=str(n), events=str(events), trials=str(trials),
                       estimate=format(events / trials, '.17g'),
                       interval_lower=format(0. if events == 0 else beta.ppf(.025, events, trials - events + 1), '.17g'),
                       interval_upper=format(1. if events == trials else beta.ppf(.975, events + 1, trials - events), '.17g'),
                       source_record_id=cell['cell_id'] + ':' + procedure,
                       upstream_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            added.append(row)
        assert cell['inferential_procedures']['EXACT_ROOT_SIGN_HOLM']['counts'][COUNT] == 0
    added.sort(key=lambda r: (int(r['root_count_R']), int(r['procedure_order'])))
    combined = added + old
    assert len(combined) == 40
    assert combined[20:] == old
    write(HERE / 'source_data/FIGURE_5B_LOW_N_ADDED.tsv', added)
    write(HERE / 'source_data/FIGURE_5B_ALL_ROOT_COUNTS.tsv', combined)
    # Check literal archival rows, including precision and field order.
    assert (HERE / 'source_data/FIGURE_5B_ALL_ROOT_COUNTS.tsv').read_text().splitlines()[21:] == original.read_text().splitlines()[1:]
    analytic = []
    for roots in range(1, 9):
        minimum_p = Fraction(1, 2 ** roots)
        threshold = Fraction(1, 20 * 12)
        possible = minimum_p <= threshold
        assert possible == (roots >= 8)
        analytic.append(dict(independent_roots_R=roots, minimum_one_sided_sign_p=str(minimum_p), first_holm_threshold=str(threshold),
                             rejection_possible=str(possible).lower(), exact_family_recovery=0 if not possible else '',
                             true_edge_recovery=0 if not possible else '', basis='ANALYTIC_DISCRETENESS',
                             monte_carlo_replicates='', interval_lower='', interval_upper=''))
    write(HERE / 'source_data/FIGURE_5C_ANALYTIC_LOWER_ROOT_REGION.tsv', analytic)
    receipt = dict(status='PASS', new_root_counts=[2,3,4,6], new_procedure_rows=20, total_procedure_rows=40,
                   old_20_source_rows_preserved_verbatim=True, failed_replicates=0,
                   low_n_exact_sign_events=0, analytic_zero_roots=list(range(1,8)),
                   minimum_roots_for_any_possible_rejection=8,
                   numerical_validation='qa/NUMERICAL_VALIDATION.json')
    (HERE / 'qa/SOURCE_DATA_VALIDATION.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
