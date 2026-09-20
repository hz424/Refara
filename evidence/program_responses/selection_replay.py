"""Recompute S/O/D family selection from bundled, already-scored utilities."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FAMILIES = ['NC', 'CM', 'TW', 'PCA', 'RBF', 'scGen', 'CPA', 'CellOT']
REGIMES = ['original8', 'cap48']
POLICIES = ['S', 'O', 'D']
TOLERANCE = 1e-12


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose(scores):
    """Maximize utility; ties within 1e-12 follow the frozen family order."""
    if set(scores) != set(FAMILIES) or not all(math.isfinite(v) for v in scores.values()):
        raise ValueError('Expected finite utilities for all eight families')
    best = max(scores.values())
    ties = [family for family in FAMILIES if best - scores[family] <= TOLERANCE]
    ordered = sorted(scores.values(), reverse=True)
    outside = [value for family, value in scores.items() if family not in ties]
    return dict(family=ties[0], maximizers=ties, family_utilities=scores,
                top_two_gap=ordered[0] - ordered[1],
                gap_to_best_outside_tie=best - max(outside) if outside else None)


def calculate(rows, identities):
    """Average seven or eight donor utilities before applying the selection rule."""
    values = {}
    for row in rows:
        key = (row['regime'], row['policy'], int(row['root_index']), row['family'])
        if key in values:
            raise ValueError('Duplicate candidate utility')
        values[key] = float(row['utility'])
    expected = set(itertools.product(REGIMES, POLICIES, range(8), FAMILIES))
    if set(values) != expected or not all(math.isfinite(v) for v in values.values()):
        raise ValueError('Incomplete or nonfinite candidate utility grid')
    if set(identities) != set(REGIMES) or any(set(identities[r]) != set(FAMILIES) for r in REGIMES):
        raise ValueError('Incomplete model identity map')
    choices = []
    comparisons = []
    for regime in REGIMES:
        for held_out in [*range(8), None]:
            roots = [root for root in range(8) if root != held_out]
            group = {}
            for policy in POLICIES:
                scores = {family: math.fsum(values[(regime, policy, root, family)] for root in roots) / len(roots)
                          for family in FAMILIES}
                result = choose(scores)
                identity = identities[regime][result['family']]
                result.update(regime=regime, scope='original8' if held_out is not None else 'additional39',
                              held_out_root_index=held_out, policy=policy, selection_root_indices=roots,
                              candidate_id=identity['candidate_id'], model_identity_sha256=identity['model_identity_sha256'])
                choices.append(result)
                group[policy] = result
            comparisons.append(dict(regime=regime, scope=group['S']['scope'], held_out_root_index=held_out,
                                    S_family=group['S']['family'], O_family=group['O']['family'], D_family=group['D']['family'],
                                    S_O_same_model=group['S']['model_identity_sha256'] == group['O']['model_identity_sha256'],
                                    O_D_same_model=group['O']['model_identity_sha256'] == group['D']['model_identity_sha256']))
    return dict(choices=choices, comparisons=comparisons,
                counts=dict(choices=len(choices), selection_cases=len(comparisons),
                            S_O_changed=sum(not row['S_O_same_model'] for row in comparisons),
                            O_D_same=sum(row['O_D_same_model'] for row in comparisons)))


def verify_expected(result, expected, atol):
    if result['counts'] != expected['counts'] or len(result['choices']) != len(expected['choices']):
        raise ValueError('Selection counts differ from verified roster')
    maximum_error = 0.0
    for actual, recorded in zip(result['choices'], expected['choices']):
        for key in ('regime', 'scope', 'held_out_root_index', 'policy', 'family', 'candidate_id',
                    'maximizers', 'selection_root_indices'):
            if actual[key] != recorded[key]:
                raise ValueError('Selection differs from verified roster: ' + key)
        for key in ('top_two_gap', 'gap_to_best_outside_tie'):
            a, b = actual[key], recorded[key]
            if a is None or b is None:
                if a is not b:
                    raise ValueError('Tie-gap null differs')
            else:
                maximum_error = max(maximum_error, abs(a - b))
        if set(recorded['family_utilities']) != set(FAMILIES):
            raise ValueError('Expected utility family incomplete')
        maximum_error = max(maximum_error, *(abs(actual['family_utilities'][f] - recorded['family_utilities'][f]) for f in FAMILIES))
    if maximum_error > atol:
        raise ValueError('Candidate utility or selection margin differs from verified roster')
    return maximum_error


def replay(output, bundle=ROOT):
    bundle, output = Path(bundle), Path(output)
    manifest = json.loads((bundle / 'selection_inputs.json').read_text())
    if manifest['families'] != FAMILIES or manifest['tie_tolerance'] != TOLERANCE:
        raise ValueError('Frozen selection rule changed')
    for name, digest in manifest['bundled_sha256'].items():
        if sha(bundle / name) != digest:
            raise ValueError('Bundled input hash differs: ' + name)
    with (bundle / 'inputs/selection_root_utilities.tsv').open(newline='') as stream:
        result = calculate(list(csv.DictReader(stream, delimiter='\t')), manifest['model_identities'])
    expected = json.loads((bundle / 'selection_expected.json').read_text())
    error = verify_expected(result, expected, manifest['replay_numeric_atol'])
    result.update(schema_version=1, status='PASS', maximum_absolute_error_against_verified_roster=error,
                  input_sha256=manifest['bundled_sha256'], aggregation=manifest['aggregation'],
                  interpretation=manifest['interpretation'])
    # Finish validation before creating any output; existing destinations are refused.
    if output.exists() or output.is_symlink():
        raise ValueError('Output must be a new directory')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'selection_results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    fields = ['regime', 'scope', 'held_out_root_index', 'policy', 'family', 'candidate_id',
              'selection_root_indices', 'maximizers', 'top_two_gap', 'gap_to_best_outside_tie']
    with (output / 'selection_roster.tsv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        for choice in result['choices']:
            row = {key: choice[key] for key in fields}
            for key in ('selection_root_indices', 'maximizers'):
                row[key] = ','.join(map(str, row[key]))
            writer.writerow(row)
    with (output / 'selection_family_utilities.tsv').open('w', newline='') as stream:
        fields = ['regime', 'scope', 'held_out_root_index', 'policy', 'family', 'utility']
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        for choice in result['choices']:
            for family, value in choice['family_utilities'].items():
                writer.writerow({**{key: choice[key] for key in fields[:4]}, 'family': family, 'utility': value})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(args.output)['counts']))
