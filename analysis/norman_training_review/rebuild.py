#!/usr/bin/env python3
"""Rebuild Norman tables and check figure data from the supplied numerical results."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
DEFAULT_SOURCE=HERE/'source_data'
PATTERNS=['S','M','P','O','D']
STATES=['additive','compositional_ridge','CPA_0.8.8']
COMMON_MODELS=['selected_CPA','original_CPA','additive','compositional_ridge','NC']
DEPTHS=[8,16,32,64,128,135]
TOL=1e-12
CHECKS=[]


def require(ok,message):
    if not ok:raise AssertionError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):return pd.read_csv(path,sep='\t',float_precision='round_trip')


def write(path,obj):path.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')


def manifest(directory):
    result={}
    for line in (directory/'MANIFEST.sha256').read_text().splitlines():
        digest,name=line.split(maxsplit=1);name=name.lstrip('*')
        path=(directory/name).resolve()
        require(directory.resolve() in path.parents,'Manifest path escapes its component')
        require(path.is_file() and sha(path)==digest,'Manifest mismatch: '+name)
        result[name]=digest
    return result


def compare(frame,filename,keys,columns=None):
    expected=read(filename)
    require(len(frame)==len(expected),'Row count differs: '+str(filename))
    require(not frame.duplicated(keys).any() and not expected.duplicated(keys).any(),'Duplicate comparison key')
    a=frame.sort_values(keys).reset_index(drop=True);b=expected.sort_values(keys).reset_index(drop=True)
    require(a[keys].equals(b[keys]),'Comparison membership differs: '+str(filename))
    columns=columns or [c for c in a.columns if c not in keys]
    maximum=0.
    for col in columns:
        require(col in b,'Missing source column: '+col)
        if pd.api.types.is_numeric_dtype(a[col]):
            aa=a[col].to_numpy(dtype=float);bb=b[col].to_numpy(dtype=float)
            require(np.array_equal(np.isnan(aa),np.isnan(bb)),'Defined values differ: '+col)
            require(np.allclose(aa,bb,rtol=1e-12,atol=1e-12,equal_nan=True),'Numeric mismatch: '+str(filename)+' '+col)
            finite=np.isfinite(aa)&np.isfinite(bb)
            if finite.any():maximum=max(maximum,float(np.max(np.abs(aa[finite]-bb[finite]))))
        else:require(a[col].equals(b[col]),'Non-numeric values differ: '+col)
    CHECKS.append(dict(file=str(filename.name),rows=len(frame),max_abs_difference=maximum))


def validation(source,out):
    table=read(source/'validation_summary.tsv');config=json.loads((HERE/'training_configuration.json').read_text())
    grids=[x['id'] for x in config['configurations']]
    candidates=table[(table.group=='heldout_combination')&table.model.isin(grids)].copy()
    require(len(candidates)==12 and not candidates.duplicated(['model','epoch']).any(),'Incomplete validation grid')
    require(set(zip(candidates.model,candidates.epoch))==set(itertools.product(grids,[100,200,400])),'Candidate membership differs')
    require((candidates.n_conditions==11).all() and np.isfinite(candidates.mean_standardized_effect_mse).all(),'Invalid validation means')
    best=candidates.mean_standardized_effect_mse.min()
    ties=candidates[candidates.mean_standardized_effect_mse<=best+TOL].copy()
    ties['reg_adv']=ties.model.map({c['id']:c['reg_adv'] for c in config['configurations']})
    ties['schedule_priority']=ties.model.str.startswith('per_epoch').astype(int)
    selected=ties.sort_values(['epoch','reg_adv','schedule_priority']).iloc[0]
    decision=json.loads((source/'validation_selection.json').read_text())
    require(selected.model==decision['selected_configuration']==config['selected_configuration'],'Selected configuration differs')
    require(int(selected.epoch)==decision['selected_epoch']==config['selected_epoch'],'Selected checkpoint differs')
    require(abs(selected.mean_standardized_effect_mse-decision['selected_validation_mse'])<TOL,'Selection score differs')
    membership=read(HERE/'selection/training_cell_membership.tsv.gz')
    require(len(membership)==68002 and membership.cell_id.nunique()==68002,'Full TRAIN membership differs')
    require(membership.condition.nunique()==161,'Full TRAIN condition count differs')
    require(membership.role.value_counts().to_dict()==config['role_counts'],'Internal training roles differ')
    conditions=read(HERE/'selection/validation_conditions.tsv')
    require(conditions.group.value_counts().to_dict()=={'seen_combination_cells':44,'heldout_combination':11},'Validation condition panel differs')
    held=set(conditions.loc[conditions.group=='heldout_combination','condition'])
    require(not held.intersection(membership.loc[membership.role=='optimizer','condition']),'Held-out combination entered optimizer cells')
    expected_roles=conditions.groupby('group').condition.apply(set).to_dict()
    for group in expected_roles:require(set(membership.loc[membership.role==group,'condition'])==expected_roles[group],'Validation membership differs')
    wide=candidates.pivot(index='model',columns='epoch',values='mean_standardized_effect_mse').reindex(grids).reset_index()
    wide.to_csv(out/'supplementary_table_9a.tsv',sep='\t',index=False)
    write(out/'selection.json',dict(selected_configuration=str(selected.model),selected_epoch=int(selected.epoch),
        selected_validation_mse=float(selected.mean_standardized_effect_mse),candidates=12,validation_conditions=11,
        selection_uses_only='Stored fixed-fold TRAIN validation summary; secondary cell holdouts and original test scores are not used.'))
    CHECKS.append(dict(file='validation_summary.tsv',rows=12,selected_configuration=str(selected.model),selected_epoch=int(selected.epoch)))


def common(source,out):
    with np.load(source/'common_target_effects.npz',allow_pickle=False) as z:
        models=z['models'].tolist();tasks=z['conditions'].tolist();effects=z['effects'];observed=z['observed_effect'];scales=z['scales']
        require(models==COMMON_MODELS and len(tasks)==len(set(tasks))==55,'Common-task axes differ')
        require(effects.shape==(5,55,2000) and observed.shape==(55,2000) and scales.shape==(2000,),'Common arrays differ')
        require(all(np.isfinite(x).all() for x in [effects,observed,scales]) and (scales>0).all(),'Invalid common arrays')
        np.testing.assert_array_equal(scales,np.load(HERE/'model/scales.npy',allow_pickle=False))
        genes=read(HERE/'model/gene_panel.tsv');require(z['gene_ids'].tolist()==genes.ensembl_id.tolist(),'Common gene panel differs')
    p=effects/scales;y=observed/scales;error=p-y
    mse=np.mean(error**2,axis=-1);mae=np.mean(np.abs(error),axis=-1)
    pmag=np.linalg.norm(p,axis=-1);ymag=np.linalg.norm(y,axis=-1)
    require((ymag>0).all(),'Undefined observed magnitude')
    ratio=pmag/ymag;denom=pmag*ymag
    cosine=np.divide(np.sum(p*y,axis=-1),denom,out=np.full_like(pmag,np.nan),where=denom>0)
    values=dict(standardized_effect_mse=mse,standardized_effect_mae=mae,predicted_magnitude=pmag,
        observed_magnitude=np.broadcast_to(ymag,pmag.shape),magnitude_ratio=ratio,cosine=cosine)
    rows=[]
    for mi,model in enumerate(models):
        for ti,task in enumerate(tasks):
            rows.append(dict(model=model,epoch=400 if mi==0 else 200 if mi==1 else 0,condition=task,group='original_test',
                             **{key:float(value[mi,ti]) for key,value in values.items()}))
    frame=pd.DataFrame(rows);compare(frame,source/'common_target_task_metrics.tsv',['model','condition'])
    summary=[]
    for mi,model in enumerate(models):
        row=dict(model=model,n_conditions=55)
        for key,value in values.items():
            finite=np.isfinite(value[mi]);row['defined_'+key]=int(finite.sum())
            row['mean_'+key]=float(value[mi,finite].mean()) if finite.any() else np.nan
        summary.append(row)
    summary=pd.DataFrame(summary);compare(summary,source/'common_target_model_summary.tsv',['model'])
    pairs=pd.DataFrame([dict(condition=task,model=models[0],reference=models[mi],mse_difference=float(mse[0,ti]-mse[mi,ti]))
        for mi in range(1,5) for ti,task in enumerate(tasks)])
    compare(pairs,source/'common_target_paired_differences.tsv',['condition','model','reference'])
    frame.to_csv(out/'common_target_task_metrics.tsv',sep='\t',index=False)
    summary.to_csv(out/'common_target_model_summary.tsv',sep='\t',index=False)
    pairs.to_csv(out/'common_target_paired_differences.tsv',sep='\t',index=False)
    summary[['model','mean_standardized_effect_mse','mean_standardized_effect_mae','mean_magnitude_ratio']].to_csv(out/'supplementary_table_9b.tsv',sep='\t',index=False)
    paired=[]
    for name,group in pairs.groupby('reference',sort=False):
        x=group.mse_difference;paired.append(dict(reference=name,mean_mse_difference=float(x.mean()),lower=int((x<0).sum()),equal=int((x==0).sum()),higher=int((x>0).sum())))
    pd.DataFrame(paired).to_csv(out/'paired_summary.tsv',sep='\t',index=False)


def reference(source,out):
    ref=np.load(source/'reference_results/utilities.npz',allow_pickle=False)
    ef=np.load(source/'direct_effect_results/effect_utilities.npz',allow_pickle=False)
    u=ref['utility'];eu=ef['effect_utility'];tasks=ref['tasks'].tolist();v=ef['V']
    require(u.shape==(55,30,6,4,5) and eu.shape==(55,30,6,5) and v.shape==(30,6),'Reference axes differ')
    require(ref['models'].tolist()==STATES+['zero_effect'] and ef['state_models'].tolist()==STATES,'Reference models differ')
    require(ref['patterns'].tolist()==ef['patterns'].tolist()==PATTERNS and ref['depths'].tolist()==ef['depths'].tolist()==DEPTHS,'Reference design differs')
    require(tasks==ef['conditions'].tolist() and len(set(tasks))==55,'Reference task panel differs')
    require(all(np.isfinite(x).all() for x in [u,eu,v]) and (v>0).all(),'Nonfinite reference results')
    tuples=list(itertools.product(range(3),repeat=3));np.testing.assert_array_equal(ref['role_tuples'],tuples)
    np.testing.assert_array_equal(ef['role_tuples'],tuples)
    require(np.isfinite(ref['K']).all() and np.isfinite(ref['conditioning_state_distance_squared']).all()
            and (ref['conditioning_state_distance_squared']>=0).all(),'Invalid reference diagnostics')
    groups={x:[] for x in PATTERNS}
    for i,(o,p,m) in enumerate(tuples):groups['S' if o==p==m else 'M' if o==p else 'P' if o==m else 'O' if p==m else 'D'].append(i)
    for pi,pattern in enumerate(PATTERNS):
        np.testing.assert_allclose(ref['atomic_utility'][...,groups[pattern]].mean(axis=-1),u[...,pi],rtol=0,atol=TOL)
        np.testing.assert_allclose(ef['effect_atomic_utility'][...,groups[pattern]].mean(axis=-1),eu[...,pi],rtol=0,atol=TOL)
    np.testing.assert_allclose(u[...,0],u[...,1],atol=TOL,rtol=0)
    np.testing.assert_allclose(u[...,:3,4]-u[...,:3,0],-np.broadcast_to(ref['V'][...,None],u[...,:3,0].shape),atol=TOL,rtol=0)
    np.testing.assert_allclose(u[...,:3,4],(u[...,:3,2]+u[...,:3,3])/2,atol=TOL,rtol=0)
    np.testing.assert_allclose(np.ptp(eu,axis=-1),0,atol=TOL,rtol=0)
    np.testing.assert_allclose(np.ptp(u[...,3,:],axis=-1),0,atol=TOL,rtol=0)
    ds=eu[...,0,None]-u[...,:3,0];dd=eu[...,4,None]-u[...,:3,4]
    np.testing.assert_allclose(dd,ds+v[...,None],atol=TOL,rtol=0)
    np.testing.assert_array_equal(ds,ef['d_S']);np.testing.assert_array_equal(dd,ef['d_D'])
    records=[(tasks[ti],ai,depth,depth*8,'direct_effect_ridge',model,float(ds[ti,ai,di,mi]),float(v[ai,di]),float(dd[ti,ai,di,mi]))
        for ti in range(55) for ai in range(30) for di,depth in enumerate(DEPTHS) for mi,model in enumerate(STATES)]
    pairs=pd.DataFrame(records,columns=['condition','allocation','depth_per_gemgroup','cells_per_role','effect_model','state_model','d_S','V','d_D'])
    pairs['observed_crossing']=(pairs.d_S< -TOL)&(pairs.d_D>TOL)
    compare(pairs,source/'direct_effect_results/task_allocation_pairs.tsv.gz',['condition','allocation','depth_per_gemgroup','state_model'])
    common=['depth_per_gemgroup','cells_per_role','effect_model','state_model'];summaries={}
    for name,by,ncol in [('task_mean_pairs',['condition']+common,'n_allocations'),('allocation_mean_pairs',['allocation']+common,'n_tasks'),('overall_mean_pairs',common,'n_task_allocation_pairs')]:
        avg=pairs.groupby(by,sort=False)[['d_S','V','d_D']].mean().reset_index()
        counts=pairs.groupby(by,sort=False).agg(**{ncol:('d_S','size')},crossing_unit_count=('observed_crossing','sum')).reset_index()
        avg=avg.merge(counts,on=by,validate='one_to_one');avg['observed_crossing']=(avg.d_S< -TOL)&(avg.d_D>TOL)
        compare(avg,source/'direct_effect_results'/(name+'.tsv'),by);summaries[name]=avg
        avg.to_csv(out/(name+'.tsv'),sep='\t',index=False)
    overall=summaries['overall_mean_pairs'];overall['d_S_over_V']=overall.d_S/overall.V
    compare(overall,source/'figure_source/d_summary.tsv',common)
    fig=out/'figure_source';fig.mkdir()
    overall.to_csv(fig/'d_summary.tsv',sep='\t',index=False)
    allu=np.concatenate([u,eu[...,None,:]],axis=3);names=STATES+['zero_effect','direct_effect_ridge'];means=allu.mean(axis=0)
    a=pd.DataFrame([(ai,depth,depth*8,model,pattern,float(-means[ai,di,mi,pi]))
        for ai in range(30) for di,depth in enumerate(DEPTHS) for mi,model in enumerate(names) for pi,pattern in enumerate(PATTERNS)],
        columns=['allocation','depth_per_gemgroup','cells_per_reference','model','pattern','scaled_squared_error'])
    bvalues=np.sqrt(ref['conditioning_state_distance_squared'][...,2].mean(axis=0))
    b=pd.DataFrame([(ai,depth,depth*8,float(bvalues[ai,di])) for ai in range(30) for di,depth in enumerate(DEPTHS)],
        columns=['allocation','depth_per_gemgroup','cells_per_reference','scaled_rms_displacement'])
    crows=[];after=[]
    for di,depth in enumerate(DEPTHS):
        for pi,pattern in enumerate(PATTERNS):
            counts=np.zeros(30,dtype=int);after_count=0
            for m,n in itertools.combinations(range(3),2):
                s=u[:,:,di,m,0]-u[:,:,di,n,0];t=u[:,:,di,m,pi]-u[:,:,di,n,pi]
                counts+=((s*t<0)&(np.abs(s)>TOL)&(np.abs(t)>TOL)).sum(axis=0)
                sm=s.mean(axis=1);tm=t.mean(axis=1);after_count+=int(((sm*tm<0)&(np.abs(sm)>TOL)&(np.abs(tm)>TOL)).sum())
            crows += [(ai,depth,depth*8,pattern,int(counts[ai]),165) for ai in range(30)]
            after.append((depth,depth*8,pattern,after_count,165))
    c=pd.DataFrame(crows,columns=['allocation','depth_per_gemgroup','cells_per_reference','pattern','reversed_task_pairs','eligible_task_pairs'])
    after=pd.DataFrame(after,columns=['depth_per_gemgroup','cells_per_reference','pattern','reversed_after_allocation_mean','eligible_task_pairs'])
    compare(after,source/'figure_source/crossings_after_mean.tsv',['depth_per_gemgroup','pattern'])
    for label,frame,value,by in [('a',a,'scaled_squared_error',['depth_per_gemgroup','cells_per_reference','model','pattern']),
            ('b',b,'scaled_rms_displacement',['depth_per_gemgroup','cells_per_reference']),
            ('c',c,'reversed_task_pairs',['depth_per_gemgroup','cells_per_reference','pattern'])]:
        compare(frame,source/f'figure_source/{label}_allocations.tsv',['allocation']+by)
        summary=frame.groupby(by,sort=False)[value].agg(mean='mean',allocation_min='min',allocation_max='max').reset_index()
        compare(summary,source/f'figure_source/{label}_summary.tsv',by)
        frame.to_csv(fig/f'{label}_allocations.tsv',sep='\t',index=False);summary.to_csv(fig/f'{label}_summary.tsv',sep='\t',index=False)
    write(out/'reference_summary.json',dict(tasks=55,allocations=30,depths=6,patterns=5,atomic_tuples=27,
        task_allocation_pair_crossings=int(pairs.observed_crossing.sum()),task_allocation_pairs=len(pairs),
        task_mean_pair_crossings=int(summaries['task_mean_pairs'].observed_crossing.sum()),task_mean_pairs=len(summaries['task_mean_pairs']),
        overall_pair_crossings=int(overall.observed_crossing.sum()),overall_pairs=len(overall),
        CPA_overall_pair_crossings=int(overall.loc[overall.state_model=='CPA_0.8.8','observed_crossing'].sum()),
        pooled_experiments=1,statistical_significance_test=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();source=args.source_dir.resolve();out=args.output_dir.resolve()
    require(out!=source and source not in out.parents and out not in source.parents,'Output must be separate from source data')
    require(out!=HERE and HERE not in out.parents and out not in HERE.parents,'Output must be separate from this code component')
    code_manifest=manifest(HERE);source_manifest=manifest(source)
    out.mkdir(parents=True,exist_ok=False)
    validation(source,out);common(source,out);reference(source,out)
    require(manifest(HERE)==code_manifest and manifest(source)==source_manifest,'Inputs changed during reconstruction')
    write(out/'REBUILD_RESULT.json',dict(status='PASS',new_training=False,new_inference=False,
        selection='Recomputed from all12 stored TRAIN-validation candidate means.',
        common_target='Recomputed all275 model-task scores and220 paired differences from saved effect arrays.',
        reference='Recomputed pattern means, all29700 direct-effect/state comparisons and ED8 a–d summaries from saved utility arrays.',
        checks=CHECKS,source_manifest_sha256=sha(source/'MANIFEST.sha256'),
        code_manifest_sha256=sha(HERE/'MANIFEST.sha256'),script_sha256=sha(__file__),
        runtime={'python':sys.version.split()[0],'numpy':np.__version__,'pandas':pd.__version__}))
    print(json.dumps({'status':'PASS','checks':len(CHECKS),'output_dir':str(out)},indent=2))


if __name__=='__main__':main()
