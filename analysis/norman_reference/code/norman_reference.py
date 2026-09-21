#!/usr/bin/env python3
"""Frozen Norman baselines and balanced reference-role scoring.

The input cells are technical partitions of one pooled experiment. Outputs are
descriptive finite-set summaries, without p values or biological intervals.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

MODELS = ('additive', 'compositional_ridge', 'CPA_0.8.8', 'zero_effect')
PATTERNS = ('S', 'M', 'P', 'O', 'D')
DEPTHS = (8, 16, 32, 64, 128, 135)
LABELS = 30
TIE_TOL = 1e-12


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')


def protocol_check(path):
    """Require a sidecar hash, so a mutable proposal cannot drive scores."""
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + '.sha256')
    expected = sidecar.read_text().split()[0]
    actual = sha256(path)
    if actual != expected:
        raise ValueError('Scoring protocol hash mismatch')
    protocol = json.loads(path.read_text())
    if protocol['version'] != 2:
        raise ValueError('Use scoring protocol v2 with frozen source allocation membership')
    if protocol['allocations']['labels'] != LABELS or tuple(
            protocol['allocations']['per_gemgroup_per_role_depths']) != DEPTHS:
        raise ValueError('Scoring constants differ from protocol')
    return actual


def quality_check(data,path):
    quality=json.loads(path.read_text())
    if quality.get('status')!='PASS':
        raise ValueError('Complete input-quality audit must pass before score access')
    required=('training_evaluation_control_ids_disjoint','no_test_conditions_in_training',
              'features_and_scales_use_only_training_controls','whole_library_normalization',
              'arrays_finite','all_reference_controls_follow_author_rule')
    if not all(quality.get(key) is True for key in required):
        raise ValueError('Input audit does not establish the frozen data contract')
    for filename,record in quality['files'].items():
        if sha256(data/filename)!=record['sha256']:
            raise ValueError(f'Input file changed after quality audit: {filename}')
    return quality


def condition_design(conditions, gene_terms):
    lookup = {gene: i for i, gene in enumerate(gene_terms)}
    design = np.zeros((len(conditions), len(gene_terms)), dtype=np.float64)
    for i, condition in enumerate(conditions):
        if condition == 'ctrl':
            continue
        terms = condition.split('+')
        if len(terms) != len(set(terms)) or not 1 <= len(terms) <= 2:
            raise ValueError(f'Invalid condition {condition}')
        for term in terms:
            design[i, lookup[term]] = 1.
    return design


def fit_baselines(train_conditions, train_means, test_conditions):
    """Equal-condition ridge with fixed sum-loss alpha=1 and free intercept."""
    train_conditions = list(map(str, train_conditions))
    test_conditions = list(map(str, test_conditions))
    train_means = np.asarray(train_means, dtype=np.float64)
    if train_means.ndim!=2 or len(train_means)!=len(train_conditions) or not np.isfinite(train_means).all():
        raise ValueError('Invalid training-condition means')
    if len(set(train_conditions)) != len(train_conditions):
        raise ValueError('Duplicate training conditions')
    row = {value: i for i, value in enumerate(train_conditions)}
    if 'ctrl' not in row:
        raise ValueError('No training NTC condition')
    singles = sorted(x for x in train_conditions if x != 'ctrl' and '+' not in x)
    for task in test_conditions:
        terms = task.split('+')
        if len(terms) != 2 or terms[0] == terms[1] or any(t not in singles for t in terms):
            raise ValueError(f'Unsupported test combination {task}')
        if task in row or '+'.join(reversed(terms)) in row:
            raise ValueError(f'Test combination was fitted: {task}')
    train_design = condition_design(train_conditions, singles)
    test_design = condition_design(test_conditions, singles)
    xmean = train_design.mean(axis=0)
    ymean = train_means.mean(axis=0)
    xc = train_design - xmean
    yc = train_means - ymean
    slope = np.linalg.solve(xc.T @ xc + np.eye(len(singles)), xc.T @ yc)
    intercept = ymean - xmean @ slope
    ridge = intercept + test_design @ slope
    additive = np.stack([train_means[row[a]] + train_means[row[b]] - train_means[row['ctrl']]
                         for a, b in (task.split('+') for task in test_conditions)])
    residual = xc.T @ (yc - xc @ slope) - slope
    return dict(additive=additive, compositional_ridge=ridge, ridge_slope=slope,
                ridge_intercept=intercept, gene_terms=np.asarray(singles),
                ridge_normal_equation_max_abs=float(np.max(np.abs(residual))))


def pattern_for(obs, pred, model):
    if obs == pred == model:
        return 'S'
    if obs == pred:
        return 'M'
    if obs == model:
        return 'P'
    if pred == model:
        return 'O'
    return 'D'


TUPLES = tuple(itertools.product(range(3), repeat=3))
PATTERN_INDEX = {p: [i for i, t in enumerate(TUPLES) if pattern_for(*t) == p] for p in PATTERNS}


def score_arrays(control_means, state_means, treated, scales):
    """Return all 27 assignment utilities and algebra/representation diagnostics.

    control_means: [label, depth, block, gene]; state_means:
    [label, depth, state_model, conditioning_block, gene]. No averaging of
    predictions before squaring across reference-role tuples is permitted.
    """
    b = np.asarray(control_means, dtype=np.float64) / scales
    states = np.asarray(state_means, dtype=np.float64) / scales
    y = np.asarray(treated, dtype=np.float64) / scales
    base_residual = states - y
    shape = b.shape[:2]
    atomic = np.empty((*shape, len(MODELS), len(TUPLES)), dtype=np.float64)
    representation_error = 0.
    reconstruction_error = 0.
    representation_utility_error = 0.
    for ti, (obs, pred, model) in enumerate(TUPLES):
        residual = base_residual[..., model, :] + (b[..., obs, :] - b[..., pred, :])[..., None, :]
        atomic[..., :3, ti] = -np.mean(residual * residual, axis=-1)
        direct_residual = b[..., obs, :] - y
        atomic[..., 3, ti] = -np.mean(direct_residual * direct_residual, axis=-1)
        # A state predictor can be stored as an effect using the same baseline.
        # Add the baseline back before any new-role centring; do not relabel the
        # cached effect as a baseline-invariant direct-effect model.
        state = states[..., model, :]
        baseline = b[..., pred, :][..., None, :]
        converted_effect = state - baseline
        reconstructed = converted_effect + baseline
        converted_residual = converted_effect - (y - b[..., obs, :])[..., None, :]
        representation_error = max(representation_error, float(np.max(np.abs(converted_residual-residual))))
        reconstruction_error = max(reconstruction_error, float(np.max(np.abs(reconstructed-state))))
        converted_utility = -np.mean(converted_residual**2, axis=-1)
        representation_utility_error = max(representation_utility_error,
            float(np.max(np.abs(converted_utility-atomic[..., :3, ti]))))
    utility = np.stack([atomic[..., ids].mean(axis=-1) for ids in PATTERN_INDEX.values()], axis=-1)
    v = np.zeros(shape)
    k = np.zeros((*shape, 3))
    input_distance = np.zeros((*shape, 3))
    for i, j in itertools.permutations(range(3), 2):
        delta = b[..., i, :] - b[..., j, :]
        v += np.mean(delta**2, axis=-1) / 6
        k += np.mean(base_residual[..., i, :] * delta[..., None, :], axis=-1) / 6
        input_distance += np.mean((states[..., i, :] - states[..., j, :])**2, axis=-1) / 6
    us, um, up, uo, ud = [utility[..., :3, i] for i in range(5)]
    checks = {
        'M_minus_S': um-us,
        'D_minus_S_plus_V': ud-us+v[..., None],
        'P_minus_S_plus_V_plus_2K': up-us+v[..., None]+2*k,
        'O_minus_S_plus_V_minus_2K': uo-us+v[..., None]-2*k,
        'D_minus_PO_midpoint': ud-(up+uo)/2,
        'zero_effect_pattern_spread': np.ptp(utility[..., 3, :], axis=-1),
        'state_pair_S_D_margin_displacement': (ud-us)-((ud-us)[..., :1]),
    }
    audit = {key: float(np.max(np.abs(value))) for key, value in checks.items()}
    audit.update(representation_residual_max_abs=representation_error,
                 representation_state_reconstruction_max_abs=reconstruction_error,
                 representation_utility_max_abs=representation_utility_error)
    if max(audit.values()) > 1e-9:
        raise AssertionError(f'Quadratic or representation identity failed: {audit}')
    return atomic, utility, v, k, input_distance, audit


def allocation_matrix(cell_ids, gemgroups, manifest, depths=DEPTHS, labels=LABELS):
    """Validate frozen membership and construct sparse incremental block sums."""
    cell_ids = np.asarray(cell_ids, dtype=str)
    gemgroups = np.asarray(gemgroups, dtype=str)
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError('Duplicate evaluation-control cells')
    groups = sorted(set(gemgroups))
    if any(np.count_nonzero(gemgroups == group) < 3*max(depths) for group in groups):
        raise ValueError('Insufficient controls for frozen depths')
    if len(manifest) != labels*len(groups)*3*max(depths):
        raise ValueError('Allocation membership is incomplete')
    if manifest.duplicated(['assignment','eval_control_index']).any():
        raise ValueError('Reference blocks overlap within an allocation')
    if manifest.duplicated(['assignment','gemgroup','block','within_block_index']).any():
        raise ValueError('Repeated allocation slot')
    col_indices=manifest.eval_control_index.to_numpy(dtype=int)
    if np.any(col_indices<0) or np.any(col_indices>=len(cell_ids)):
        raise ValueError('Out of range control cell index')
    if not np.array_equal(cell_ids[col_indices],manifest.cell_id.astype(str)):
        raise ValueError('Allocation control cell IDs do not match the control matrix')
    if not np.array_equal(gemgroups[col_indices],manifest.gemgroup.astype(str)):
        raise ValueError('Allocation gemgroups do not match controls')
    if set(manifest.assignment)!=set(range(labels)) or set(manifest.block)!=set(range(3)):
        raise ValueError('Allocation labels or block IDs differ')
    for _,block in manifest.groupby(['assignment','gemgroup','block']):
        if sorted(block.within_block_index)!=list(range(max(depths))):
            raise ValueError('Block ranks do not give complete nested prefixes')
    depth_index=np.searchsorted(depths,manifest.within_block_index.to_numpy()+1)
    row_indices=(manifest.assignment.to_numpy()*len(depths)+depth_index)*3+manifest.block.to_numpy()
    matrix = sparse.csr_matrix((np.ones(len(row_indices)), (row_indices, col_indices)),
                               shape=(labels*len(depths)*3, len(cell_ids)))
    return matrix, len(groups)


def allocated_means(matrix, cell_values, group_count, depths=DEPTHS, labels=LABELS):
    sums = matrix @ np.asarray(cell_values, dtype=np.float64)
    sums = np.asarray(sums).reshape(labels, len(depths), 3, -1)
    sums = np.cumsum(sums, axis=1)
    return sums / (group_count*np.asarray(depths)[None, :, None, None])


def axis_values(frame, candidates):
    for name in candidates:
        if name in frame:
            return frame[name].astype(str).tolist()
    raise ValueError(f'Missing axis column: {candidates}')


def fit_command(args):
    protocol_sha = protocol_check(args.protocol)
    quality_check(args.data,args.quality)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'baseline_fit.json').exists():
        raise FileExistsError('Refusing to replace completed baseline fit')
    train = np.load(args.data/'training_means.npz', allow_pickle=False)
    tasks = pd.read_csv(args.data/'test_tasks.tsv', sep='\t', dtype=str)
    conditions = tasks['condition'].tolist()
    fitted = fit_baselines(train['conditions'], train['mean_equal_gemgroup'], conditions)
    diagnostics = fitted.pop('ridge_normal_equation_max_abs')
    np.savez_compressed(args.output/'baselines.npz', conditions=np.asarray(conditions), **fitted)
    write_json(args.output/'baseline_fit.json', dict(
        native_output={'additive':'state', 'compositional_ridge':'state'},
        protocol_sha256=protocol_sha, training_means_sha256=sha256(args.data/'training_means.npz'),
        input_quality_sha256=sha256(args.quality),
        test_task_axis_sha256=sha256(args.data/'test_tasks.tsv'),
        baseline_file_sha256=sha256(args.output/'baselines.npz'),
        script_sha256=sha256(__file__), ridge_normal_equation_max_abs=diagnostics,
        n_training_conditions=len(train['conditions']), n_test_tasks=len(conditions),
        n_genes=int(train['mean_equal_gemgroup'].shape[1]), alpha=1.,
        loss_normalization='sum over conditions', intercept='unpenalized',
        means='equal gemgroup means', test_expression_used=False))


def paired_tables(utility, tasks):
    """Every frozen pair, including stable and zero-baseline comparisons."""
    # utility [task,label,depth,model,pattern]
    rows = []
    pair_ids = list(itertools.combinations(range(4), 2))
    for task_index, condition in enumerate(tasks):
        for a, b in pair_ids:
            margin = utility[task_index, :, :, a, :] - utility[task_index, :, :, b, :]
            for label in range(LABELS):
                for di, depth in enumerate(DEPTHS):
                    anchor = margin[label, di, 0]
                    for pi, pattern in enumerate(PATTERNS):
                        m = margin[label, di, pi]
                        crossing = bool(anchor*m < 0 and abs(anchor)>TIE_TOL and abs(m)>TIE_TOL)
                        rows.append((condition, label, depth, MODELS[a], MODELS[b],
                                     'primary_state_pair' if b<3 else 'zero_effect_diagnostic',
                                     pattern, m, anchor, m-anchor, crossing))
    return pd.DataFrame(rows, columns=['condition','allocation','depth_per_gemgroup','model_a','model_b',
        'pair_role','pattern','margin','S_margin','displacement','strict_crossing'])


def score_command(args):
    protocol_sha = protocol_check(args.protocol)
    quality_check(args.data,args.quality)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'score_report.json').exists():
        raise FileExistsError('Refusing to replace completed scores')
    tasks = pd.read_csv(args.data/'test_tasks.tsv', sep='\t', dtype=str)['condition'].tolist()
    genes = pd.read_csv(args.data/'gene_panel.tsv', sep='\t')
    gene_ids = axis_values(genes, ['feature_id','gene_id','ensembl_id','gene'])
    scales = genes['scale'].to_numpy(dtype=np.float64)
    if len(scales)!=2000 or not np.isfinite(scales).all() or scales.min()<.1:
        raise ValueError('Invalid frozen gene scales')
    cells = pd.read_csv(args.data/'eval_control_cells.tsv', sep='\t', dtype=str)
    cell_ids = axis_values(cells, ['cell_id','barcode'])
    groups = cells['gemgroup'].to_numpy(dtype=str)
    if len(set(groups)) != 8 or min(np.count_nonzero(groups==g) for g in set(groups))//3 !=135:
        raise ValueError('Evaluation control support differs from metadata-fixed allocation')
    controls = np.load(args.data/'eval_controls.npy', mmap_mode='r')
    observed = np.load(args.data/'test_means.npz', allow_pickle=False)
    if list(observed['conditions']) != tasks:
        raise ValueError('Observed task axis differs')
    if controls.shape != (len(cells),len(scales)):
        raise ValueError('Control matrix axes differ')
    if observed['mean_equal_gemgroup'].shape != (55,2000):
        raise ValueError('Observed data dimensions differ')
    baseline_record = json.loads((args.baselines/'baseline_fit.json').read_text())
    if baseline_record['protocol_sha256'] != protocol_sha or baseline_record['baseline_file_sha256'] != sha256(args.baselines/'baselines.npz'):
        raise ValueError('Frozen baseline hash differs')
    baselines = np.load(args.baselines/'baselines.npz', allow_pickle=False)
    if list(baselines['conditions']) != tasks:
        raise ValueError('Baseline task axis differs')
    pred = pd.read_csv(args.cpa/'predictions.tsv', sep='\t')
    if pred['condition'].tolist() != tasks:
        raise ValueError('CPA task axis differs')
    cpa_cells = pd.read_csv(args.cpa/'control_cells.tsv', sep='\t', dtype=str)['cell_id'].tolist()
    cpa_genes = pd.read_csv(args.cpa/'genes.tsv', sep='\t', dtype=str)['feature_id'].tolist()
    if cpa_cells != cell_ids or cpa_genes != gene_ids:
        raise ValueError('CPA control or gene axis differs')
    cpa_record=json.loads((args.cpa/'prediction_record.json').read_text())
    if (cpa_record['native_output']!='state' or cpa_record['training_control_overlap']!=0 or
        cpa_record['treated_cells_used_as_input']!=0 or
        cpa_record['controls_sha256']!=sha256(args.data/'eval_controls.h5ad') or
        cpa_record['tasks_sha256']!=sha256(args.data/'test_tasks.tsv')):
        raise ValueError('CPA prediction record differs from the frozen evaluation inputs')
    # Verify every output before generating even the first task score.
    for record in pred.to_dict('records'):
        path = args.cpa/record['file']
        if sha256(path) != record['sha256']:
            raise ValueError('CPA prediction hash differs')
        slab = np.load(path, mmap_mode='r')
        if slab.shape != controls.shape or not np.isfinite(slab).all():
            raise ValueError('Invalid CPA prediction matrix')
    protocol=json.loads(args.protocol.read_text())
    if sha256(args.data/'control_allocations.tsv.gz')!=protocol['allocations']['source_membership_sha256']:
        raise ValueError('Frozen source allocation membership hash differs')
    allocation_manifest=pd.read_csv(args.data/'control_allocations.tsv.gz',sep='\t',dtype={'gemgroup':str,'cell_id':str})
    matrix, n_groups = allocation_matrix(cell_ids,groups,allocation_manifest)
    allocation_manifest.to_csv(args.output/'allocation_membership.tsv.gz', sep='\t', index=False,
                               compression={'method':'gzip','mtime':0})
    b = allocated_means(matrix,controls,n_groups)
    np.save(args.output/'control_block_means.npy',b)
    utility = np.empty((len(tasks),LABELS,len(DEPTHS),4,5),dtype=np.float64)
    atomic = np.empty((len(tasks),LABELS,len(DEPTHS),4,27),dtype=np.float64)
    k_values = np.empty((len(tasks),LABELS,len(DEPTHS),3))
    distance = np.empty_like(k_values)
    audit = {}
    for task_index, task in enumerate(tasks):
        state = np.empty((LABELS,len(DEPTHS),3,3,len(scales)))
        state[:,:,0,:,:] = baselines['additive'][task_index]
        state[:,:,1,:,:] = baselines['compositional_ridge'][task_index]
        slab = np.load(args.cpa/pred.iloc[task_index]['file'],mmap_mode='r')
        state[:,:,2,:,:] = allocated_means(matrix,slab,n_groups)
        out = score_arrays(b,state,observed['mean_equal_gemgroup'][task_index],scales)
        atomic[task_index],utility[task_index],v,k_values[task_index],distance[task_index],checks = out
        for key,value in checks.items():
            audit[key]=max(audit.get(key,0.),value)
        if (task_index+1)%10==0:
            print(f'Completed {task_index+1}/{len(tasks)} frozen tasks',flush=True)
    np.savez_compressed(args.output/'utilities.npz', utility=utility, atomic_utility=atomic,
        V=v,K=k_values,conditioning_state_distance_squared=distance,
        tasks=np.asarray(tasks),models=np.asarray(MODELS),patterns=np.asarray(PATTERNS),
        depths=np.asarray(DEPTHS),role_tuples=np.asarray(TUPLES))
    utility_rows=[]
    for ti,task in enumerate(tasks):
        for label in range(LABELS):
            for di,depth in enumerate(DEPTHS):
                for mi,model in enumerate(MODELS):
                    for pi,pattern in enumerate(PATTERNS):
                        utility_rows.append((task,label,depth,model,pattern,utility[ti,label,di,mi,pi]))
    utilities=pd.DataFrame(utility_rows,columns=['condition','allocation','depth_per_gemgroup','model','pattern','utility'])
    utilities.to_csv(args.output/'task_pattern_utilities.tsv.gz',sep='\t',index=False,compression={'method':'gzip','mtime':0})
    margins=paired_tables(utility,tasks)
    margins.to_csv(args.output/'task_pair_margins.tsv.gz',sep='\t',index=False,compression={'method':'gzip','mtime':0})
    summarize_outputs(args.output,utility,k_values,distance,v,tasks,margins)
    write_json(args.output/'score_report.json',dict(status='PASS',
        scope='One pooled experiment; finite-allocation descriptive summaries',
        n_tasks=len(tasks),n_gemgroups=n_groups,n_independent_pooled_experiments=1,
        n_allocations=LABELS,depths_per_gemgroup=list(DEPTHS),models=list(MODELS),
        model_pairs=6,primary_state_pairs=3,diagnostic_zero_pairs=3,
        identities=audit,protocol_sha256=protocol_sha,quality_sha256=sha256(args.quality),
        baseline_fit_sha256=sha256(args.baselines/'baseline_fit.json'),
        cpa_record_sha256=sha256(args.cpa/'prediction_record.json'),
        script_sha256=sha256(__file__),output_sha256={p.name:sha256(p) for p in sorted(args.output.iterdir()) if p.is_file()}))


def summarize_outputs(output,utility,k,distance,v,tasks,task_margins):
    # First average complete task utilities, then compare the resulting margins.
    task_mean=utility.mean(axis=0)
    rows=[]
    for label in range(LABELS):
        for di,depth in enumerate(DEPTHS):
            for a,b in itertools.combinations(range(4),2):
                anchor=task_mean[label,di,a,0]-task_mean[label,di,b,0]
                for pi,pattern in enumerate(PATTERNS):
                    margin=task_mean[label,di,a,pi]-task_mean[label,di,b,pi]
                    rows.append((label,depth,MODELS[a],MODELS[b],
                        'primary_state_pair' if b<3 else 'zero_effect_diagnostic',pattern,
                        anchor,margin,margin-anchor,bool(anchor*margin<0 and abs(anchor)>TIE_TOL and abs(margin)>TIE_TOL)))
    frame=pd.DataFrame(rows,columns=['allocation','depth_per_gemgroup','model_a','model_b','pair_role',
                                   'pattern','S_margin','margin','displacement','strict_crossing'])
    frame.to_csv(output/'allocation_pair_margins.tsv',sep='\t',index=False)
    grouping=['depth_per_gemgroup','model_a','model_b','pair_role','pattern']
    mean_frame=frame.groupby(grouping,sort=False).agg(
        S_margin=('S_margin','mean'),margin=('margin','mean'),displacement=('displacement','mean'),
        allocation_min_margin=('margin','min'),allocation_max_margin=('margin','max'),
        crossing_allocation_count=('strict_crossing','sum'),n_allocations=('allocation','size')).reset_index()
    mean_frame['strict_crossing_after_task_and_allocation_means']=(
        (mean_frame.S_margin*mean_frame.margin<0)&(mean_frame.S_margin.abs()>TIE_TOL)&(mean_frame.margin.abs()>TIE_TOL))
    mean_frame.to_csv(output/'depth_pair_summary.tsv',sep='\t',index=False)
    task_summary=task_margins.groupby(['condition']+grouping,sort=False).agg(
        S_margin=('S_margin','mean'),margin=('margin','mean'),displacement=('displacement','mean'),
        crossing_allocation_count=('strict_crossing','sum'),n_allocations=('allocation','size')).reset_index()
    task_summary['strict_crossing_after_allocation_mean']=(
        (task_summary.S_margin*task_summary.margin<0)&(task_summary.S_margin.abs()>TIE_TOL)&(task_summary.margin.abs()>TIE_TOL))
    task_summary.to_csv(output/'task_depth_pair_summary.tsv',sep='\t',index=False)
    utility_rows=[]
    for di,depth in enumerate(DEPTHS):
        for mi,model in enumerate(MODELS):
            for pi,pattern in enumerate(PATTERNS):
                values=task_mean[:,di,mi,pi]
                utility_rows.append((depth,model,pattern,float(values.mean()),float(values.min()),float(values.max())))
    pd.DataFrame(utility_rows,columns=['depth_per_gemgroup','model','pattern','mean_utility',
        'allocation_min_utility','allocation_max_utility']).to_csv(output/'depth_utility_summary.tsv',sep='\t',index=False)
    diagnostics=[]
    for ti,task in enumerate(tasks):
        for label in range(LABELS):
            for di,depth in enumerate(DEPTHS):
                for mi,model in enumerate(MODELS[:3]):
                    diagnostics.append((task,label,depth,model,v[label,di],k[ti,label,di,mi],
                                        distance[ti,label,di,mi],float(np.sqrt(distance[ti,label,di,mi]))))
    pd.DataFrame(diagnostics,columns=['condition','allocation','depth_per_gemgroup','model','V','K',
        'conditioning_state_distance_squared','conditioning_state_rms_distance']).to_csv(
        output/'geometry_and_input_displacement.tsv.gz',sep='\t',index=False,compression={'method':'gzip','mtime':0})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='command',required=True)
    fit=subs.add_parser('fit')
    fit.add_argument('--data',type=Path,required=True)
    fit.add_argument('--protocol',type=Path,required=True)
    fit.add_argument('--quality',type=Path,required=True)
    fit.add_argument('--output',type=Path,required=True)
    score=subs.add_parser('score')
    score.add_argument('--data',type=Path,required=True)
    score.add_argument('--protocol',type=Path,required=True)
    score.add_argument('--baselines',type=Path,required=True)
    score.add_argument('--cpa',type=Path,required=True)
    score.add_argument('--quality',type=Path,required=True)
    score.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='fit':
        fit_command(args)
    else:
        score_command(args)


if __name__=='__main__':
    main()
