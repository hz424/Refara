"""Write a synthetic organizer study with a recorded geometry-policy failure."""
import argparse
import csv
import json
from pathlib import Path


def _write(path, columns, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def make_example(output):
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Example directory must be absent or empty')
    output.mkdir(parents=True, exist_ok=True)
    budgets = [dict(id='small', control_cells_available=100, input_cells=50, observation_cells=50,
                    description='Two disjoint blocks of 50 control cells.'),
               dict(id='large', control_cells_available=200, input_cells=100, observation_cells=100,
                    description='Two disjoint blocks of 100 control cells.')]
    models = ['model_a', 'model_b', 'model_c']
    development, assessment, geometry = [], [], []
    for case in ('dev_1', 'dev_2', 'stable', 'reversal', 'small_gap'):
        for budget in budgets:
            bid = budget['id']
            dev = case.startswith('dev_')
            g = 1. if dev or bid == 'large' else .25 if case == 'reversal' else .5
            geometry.append(dict(case=case, unit='synthetic_unit', budget=bid, geometry=g))
            for realization in ('anchor', 'repeat_1', 'repeat_2'):
                values = [1., 4., 8.]
                if dev and realization != 'anchor':
                    values[1] += 2. if bid == 'small' else 1.
                if case == 'reversal':
                    values[1] = (2.5 if bid == 'small' else 3.5) if realization == 'anchor' else 0.
                if case == 'small_gap' and realization != 'anchor':
                    values[1] = 1.3
                for model, score in zip(models, values):
                    (development if dev else assessment).append(dict(case=case, unit='synthetic_unit', budget=bid,
                                                                     realization=realization, model=model, score=score))
    columns = ['case', 'unit', 'budget', 'realization', 'model', 'score']
    _write(output/'development.tsv', columns, development)
    _write(output/'anchors.tsv', columns, [r for r in assessment if r['realization'] == 'anchor'])
    _write(output/'assessment.tsv', columns, assessment)
    _write(output/'geometry.tsv', ['case', 'unit', 'budget', 'geometry'], geometry)
    manifest = dict(schema_version=1, target='heldout_effect', metric=dict(name='synthetic_MSE', direction='lower'),
                    models=models, budgets=budgets,
                    roles=dict(model_input='Declared input block supplied to the state predictors.',
                               prediction_centring='State predictions centred using the input block.',
                               observation_centring='Treated observations centred using the disjoint observation block.',
                               participant_information='Task definitions and input-block cells.',
                               evaluator_information='Treated observations and observation-block cells.'),
                    development_scores='development.tsv', assessment_anchors='anchors.tsv', geometry='geometry.tsv',
                    anchor_realization='anchor', minimum_gap=.5, numerical_tolerance=1e-12,
                    geometry_tolerance=1e-12, evaluation_scope='Synthetic fixed three-model family and recorded control-reference realizations.')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return output/'manifest.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(make_example(args.output))
