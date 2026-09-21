"""Create synthetic AnnData/TSV inputs for the declared preparation workflow."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def write(path, columns, rows):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
        writer.writerow(columns)
        writer.writerows(rows)


def make_example(output, format='h5ad'):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Example directory must be absent or empty')
    output.mkdir(parents=True, exist_ok=True)
    genes = ['g1', 'g2', 'g3']
    records, values = [], []
    for donor in ['donor_a', 'donor_b']:
        for batch in ['batch_1', 'batch_2']:
            for i in range(3):
                records.append([f'{donor}_{batch}_c{i}', donor, 'control', batch, 'type_a'])
                values.append([1+i/10, 2+i/10, 3+i/10])
        for condition in (['pert_a', 'pert_b'] if donor == 'donor_a' else ['pert_a']):
            for batch in ['batch_1', 'batch_2']:
                records.append([f'{donor}_{condition}_{batch}', donor, condition, batch, 'type_a'])
                values.append([2, 3, 4] if condition == 'pert_a' else [3, 4, 5])
    tasks = [['a_pert_a', 'donor_a', 'pert_a', 'type_a'], ['a_pert_b', 'donor_a', 'pert_b', 'type_a'], ['b_pert_a', 'donor_b', 'pert_a', 'type_a']]
    write(output/'tasks.tsv', ['task', 'unit', 'condition', 'cell_type'], tasks)
    cells = dict(format=format, path='cells.h5ad' if format == 'h5ad' else 'cells.tsv', unit_column='donor', condition_column='condition',
                 strata_columns=['batch'], context_columns=['cell_type'], normalization='Synthetic values already on a common expression scale; no further transform.')
    prediction_values = {'effect': np.array([[1, 1, 1], [2, 2, 2], [1, 1, 1]], dtype=float),
                         'state': np.array([[2.2, 3.2, 4.2], [2.2, 3.2, 4.2], [2.2, 3.2, 4.2]])}
    models = []
    if format == 'h5ad':
        import anndata as ad
        import pandas as pd
        from scipy.sparse import csr_matrix
        obs = pd.DataFrame(records, columns=['cell_id', 'donor', 'condition', 'batch', 'cell_type']).set_index('cell_id')
        data = ad.AnnData(X=np.zeros((len(values), len(genes))), obs=obs, var=pd.DataFrame(index=genes))
        data.layers['normalized'] = csr_matrix(np.asarray(values))
        data.write_h5ad(output/'cells.h5ad')
        cells['layer'] = 'normalized'
        for kind, matrix in prediction_values.items():
            pred = ad.AnnData(X=matrix, obs=pd.DataFrame(index=[t[0] for t in tasks]), var=pd.DataFrame(index=genes))
            pred.write_h5ad(output/f'{kind}.h5ad')
            models.append(dict(name=kind, kind=kind, conditioning='fixed', representation='native',
                               predictions=dict(format='h5ad', path=f'{kind}.h5ad', layer='X', task_column='__index__')))
    elif format == 'tsv':
        write(output/'cells.tsv', ['cell_id', 'donor', 'condition', 'batch', 'cell_type', *genes], [r+list(v) for r, v in zip(records, values)])
        cells.update(gene_columns=genes, cell_id_column='cell_id')
        for kind, matrix in prediction_values.items():
            write(output/f'{kind}.tsv', ['task', *genes], [[t[0], *v] for t, v in zip(tasks, matrix)])
            models.append(dict(name=kind, kind=kind, conditioning='fixed', representation='native', predictions=f'{kind}.tsv'))
    else:
        raise ValueError('format must be h5ad or tsv')
    manifest = dict(target='heldout_effect', prediction_controls='available', cells=cells, tasks='tasks.tsv', control_value='control',
                    treated_aggregation='cell_mean', allocation=dict(blocks=2, depths=[1], allocations=2, seed=7), models=models, diagnostics=True)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return output/'manifest.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--format', choices=['h5ad', 'tsv'], default='h5ad')
    args = parser.parse_args()
    print(make_example(args.output, args.format))
