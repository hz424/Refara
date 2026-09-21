"""Replay fixed-panel model comparisons from frozen losses."""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def directions(values, tolerance):
    values = np.asarray(values)
    return np.where(values > tolerance, 1, np.where(values < -tolerance, -1, 0))


def compare_table(path, rows, atol):
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        expected = list(reader)
        require(reader.fieldnames == list(rows[0]), 'output columns changed: ' + path.name)
    require(len(expected) == len(rows), 'output row count changed: ' + path.name)
    for i, (saved, computed) in enumerate(zip(expected, rows)):
        for key, value in computed.items():
            if isinstance(value, (float, np.floating)):
                number = float(saved[key])
                require(np.isfinite(number) and abs(number - value) <= atol,
                        'stale numeric output: %s row %s %s' % (path.name, i, key))
            else:
                require(saved[key] == str(value), 'stale output: %s row %s %s' % (path.name, i, key))


def write_table(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def replay(base=BASE, output=None):
    base = Path(base)
    protocol = json.loads((base / 'manifest.json').read_bytes())
    require(sha(base / 'losses.npz') == protocol['losses_sha256'], 'frozen loss checksum differs')
    require(protocol['scope_id'] == 'retrospective_fixed_observed_panel', 'unsupported scope')
    require(protocol['root_count'] == 20 and protocol['root_task_count'] == 160,
            'declared root geometry differs')
    tasks, models = protocol['tasks'], protocol['models']
    require(len(tasks) == len(set(tasks)) == 8 and len(models) == len(set(models)) == 3, 'axis roster differs')
    with np.load(base / 'losses.npz', allow_pickle=False) as bank:
        require(set(bank.files) == {'S','O','O_fixed_input','crossfit'}, 'loss array roster differs')
        s, o, fixed, saved_cf = (bank[x] for x in ('S','O','O_fixed_input','crossfit'))
    require(s.shape == (160,3) and o.shape == fixed.shape == (160,16,3) and saved_cf.shape == (160,8,3), 'loss geometry differs')
    for array in (s,o,fixed,saved_cf):
        require(array.dtype == np.float64 and np.isfinite(array).all() and (array >= 0).all(), 'invalid loss array')
    atol = protocol['replay_absolute_tolerance']
    tolerance = protocol['direction_absolute_tolerance']
    require(atol == 2e-14 and tolerance == 1e-10, 'numeric tolerances changed')
    cf = (o[:,0::2,:] + o[:,1::2,:]) / 2
    require(np.allclose(cf, saved_cf, rtol=0, atol=atol), 'complement-pair means differ')
    require(np.allclose(cf.mean(axis=1),o.mean(axis=1),rtol=0,atol=atol), 'O/crossfit identity failed')
    task_s = s.reshape(20,8,3).mean(axis=0)
    task_o = o.reshape(20,8,16,3).mean(axis=0)
    task_fixed = fixed.reshape(20,8,16,3).mean(axis=0)
    task_cf = cf.reshape(20,8,8,3).mean(axis=0)
    pairs = list(itertools.combinations(range(3),2))
    root_rows, task_rows, pair_summary, integration = [], [], [], []
    for rt in range(160):
        for a,b in pairs:
            anchor = float(s[rt,b]-s[rt,a])
            for allocation in range(16):
                margin = float(o[rt,allocation,b]-o[rt,allocation,a])
                fixed_margin = float(fixed[rt,allocation,b]-fixed[rt,allocation,a])
                ds,do,df = [int(x) for x in directions([anchor,margin,fixed_margin],tolerance)]
                root_rows.append(dict(root_task_index=rt,root_index=rt//8,task=tasks[rt%8],
                    model_a=models[a],model_b=models[b],allocation=allocation,
                    S_margin=anchor,O_margin=margin,O_fixed_input_margin=fixed_margin,
                    S_direction=ds,O_direction=do,O_fixed_input_direction=df,
                    strict_S_O_reversal=ds*do == -1,actual_vs_fixed_input_direction_changed=do != df))
    for task,name in enumerate(tasks):
        for a,b in pairs:
            anchor = float(task_s[task,b]-task_s[task,a])
            these = []
            for allocation in range(16):
                margin = float(task_o[task,allocation,b]-task_o[task,allocation,a])
                fixed_margin = float(task_fixed[task,allocation,b]-task_fixed[task,allocation,a])
                ds,do,df = [int(x) for x in directions([anchor,margin,fixed_margin],tolerance)]
                row=dict(task=name,model_a=models[a],model_b=models[b],allocation=allocation,
                    S_margin=anchor,O_margin=margin,O_fixed_input_margin=fixed_margin,
                    S_direction=ds,O_direction=do,O_fixed_input_direction=df,
                    strict_S_O_reversal=ds*do == -1,actual_vs_fixed_input_direction_changed=do != df)
                task_rows.append(row); these.append(row)
            pair_summary.append(dict(task=name,model_a=models[a],model_b=models[b],S_direction=these[0]['S_direction'],
                O_favors_a=sum(r['O_direction']==1 for r in these),O_favors_b=sum(r['O_direction']==-1 for r in these),
                O_ties=sum(r['O_direction']==0 for r in these),strict_S_O_reversals=sum(r['strict_S_O_reversal'] for r in these),
                actual_vs_fixed_input_direction_changes=sum(r['actual_vs_fixed_input_direction_changed'] for r in these)))
        for model,name_model in enumerate(models):
            values=task_cf[task,:,model]
            integration.append(dict(task=name,model=name_model,S_loss=float(task_s[task,model]),
                O_mean=float(task_o[task,:,model].mean()),crossfit_mean=float(values.mean()),
                allocation_MC_standard_error=float(values.std(ddof=1)/np.sqrt(8)),
                O_fixed_input_mean=float(task_fixed[task,:,model].mean())))
    tables={'root_task_pair_directions.tsv':root_rows,'task_pair_directions.tsv':task_rows,
            'allocation_integration_summary.tsv':integration}
    for name,rows in tables.items():
        compare_table(base / name,rows,atol)
    def counts(rows):
        return dict(events=len(rows),strict_S_O_reversal_events=sum(r['strict_S_O_reversal'] for r in rows),
            S_tie_events=sum(r['S_direction']==0 for r in rows),O_tie_events=sum(r['O_direction']==0 for r in rows),
            fixed_input_tie_events=sum(r['O_fixed_input_direction']==0 for r in rows),
            actual_vs_fixed_input_direction_changes=sum(r['actual_vs_fixed_input_direction_changed'] for r in rows))
    result=dict(status='PASS_PORTABLE_FIXED_PANEL_LOSS_REPLAY',scope=protocol['scope'],
        replay_computes='Means, margins, directions, ties, complement-pair means and allocation Monte Carlo standard errors from frozen model losses.',
        raw_model_inference_replayed=False,independent_experiment_validation=False,donor_population_CI=False,
        source_result_sha256=protocol['provenance']['result_sha256'],
        root_task_pair_allocation=counts(root_rows),task_pair_allocation=counts(task_rows),
        task_pairs_with_any_strict_reversal=sum(r['strict_S_O_reversals']>0 for r in pair_summary),
        task_pairs_reversed_in_all_16=sum(r['strict_S_O_reversals']==16 for r in pair_summary),
        O_crossfit_overall_mean_identity=True,
        maximum_actual_vs_fixed_loss_difference_by_model={model:float(np.max(np.abs(o[:,:,m]-fixed[:,:,m]))) for m,model in enumerate(models)},
        task_pair_summary=pair_summary)
    for key,value in protocol['expected_counts'].items():
        require(result[key] == value,'summary differs: '+key)
    if output is not None:
        output=Path(output)
        require(not output.exists(),'output path already exists')
        output.mkdir(parents=True)
        for name,rows in tables.items(): write_table(output/name,rows)
        write_table(output/'task_pair_summary.tsv',pair_summary)
        (output/'RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=replay(output=args.output)
    print(json.dumps({k:v for k,v in result.items() if k != 'task_pair_summary'},indent=2,allow_nan=False))
