"""Create a small, fully synthetic two-block supplied-prediction protocol."""
import argparse
import csv
import json
from pathlib import Path


def write(path, columns, rows):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
        writer.writerow(columns)
        writer.writerows(rows)


def make_example(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Example destination must be absent or empty')
    output.mkdir(parents=True, exist_ok=True)
    write(output/'controls.tsv', ['cell_id', 'stratum', 'g1', 'g2'],
          [['c1', 'all', 1, 2], ['c2', 'all', 3, 4], ['c3', 'all', 2, 1], ['c4', 'all', 4, 3]])
    refs, membership, conditioned = [], [], []
    cells = [('c1', [1, 2]), ('c2', [3, 4]), ('c3', [2, 1]), ('c4', [4, 3])]
    for allocation, chosen in enumerate([(0, 1), (2, 3)]):
        for block, index in zip(['B1', 'B2'], chosen):
            cell, values = cells[index]
            refs.append([allocation, 1, block, 1, *values])
            membership.append([allocation, 1, cell, 'all', block, 0])
            conditioned.append([allocation, 1, block, *[x+2 for x in values]])
    write(output/'references.tsv', ['allocation', 'depth', 'block', 'cells_per_block', 'g1', 'g2'], refs)
    write(output/'membership.tsv', ['allocation', 'depth', 'cell_id', 'stratum', 'block', 'within_block_index'], membership)
    write(output/'conditioned.tsv', ['allocation', 'depth', 'block', 'g1', 'g2'], conditioned)
    write(output/'effect.tsv', ['gene', 'value'], [['g1', 2], ['g2', 2]])
    write(output/'state.tsv', ['gene', 'value'], [['g1', 4], ['g2', 5]])
    tasks = []
    for i, (unit, values) in enumerate([('unit_a', [5, 6]), ('unit_b', [3, 4]), ('unit_b', [7, 8])]):
        write(output/f'treated_{i}.tsv', ['gene', 'value'], list(zip(['g1', 'g2'], values)))
        tasks.append(dict(id=f'task_{i}', unit=unit, treated=f'treated_{i}.tsv', references='references.tsv',
                          controls='controls.tsv', membership='membership.tsv', models=[
                              dict(name='native_effect', kind='effect', prediction='effect.tsv'),
                              dict(name='fixed_state', kind='state', prediction='state.tsv'),
                              dict(name='conditioned_state', kind='state', conditioning='block', prediction='conditioned.tsv')]))
    plan = dict(target='heldout_effect', prediction_controls='available', diagnostics=True, tasks=tasks)
    (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    return output/'plan.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    print(make_example(parser.parse_args().output))
