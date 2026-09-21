#!/usr/bin/env python3
"""Independently check fixed first/last tasks, allocations 0/29 and depths 8/135."""
import argparse, itertools, json
from pathlib import Path
import numpy as np
import pandas as pd
p = argparse.ArgumentParser()
for key in ['data', 'predictions', 'baselines', 'results', 'output']:
    p.add_argument('--'+key, required=True, type=Path)
a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
ref = dict(np.load(a.results/'utilities.npz', allow_pickle=False))
y = dict(np.load(a.data/'test_means.npz', allow_pickle=False))
base = dict(np.load(a.baselines/'baselines.npz', allow_pickle=False))
cells = pd.read_csv(a.data/'eval_control_cells.tsv', sep='\t')
alloc = pd.read_csv(a.data/'control_allocations.tsv.gz', sep='\t')
cpafiles = pd.read_csv(a.predictions/'predictions.tsv', sep='\t')
cpacells = pd.read_csv(a.predictions/'control_cells.tsv', sep='\t')
assert cells.cell_id.tolist() == cpacells.cell_id.tolist()
assert list(ref['tasks']) == list(y['conditions']) == list(base['conditions']) == cpafiles.condition.tolist()
assert list(ref['models']) == ['additive','compositional_ridge','CPA_0.8.8','zero_effect']
assert list(ref['patterns']) == ['S','M','P','O','D']
tuples = list(itertools.product(range(3), repeat=3)); assert tuples == [tuple(t) for t in ref['role_tuples']]
controls = np.load(a.data/'eval_controls.npy', mmap_mode='r')
scale = np.load(a.data/'scales.npy'); positions = dict(zip(cells.cell_id, range(len(cells))))
records, max_error = [], 0.
for task in [0, len(ref['tasks'])-1]:
    cpa = np.load(a.predictions/cpafiles.iloc[task]['file'], mmap_mode='r')
    for label, depth in itertools.product([0,29], [8,135]):
        chosen = alloc[(alloc.assignment==label) & (alloc.within_block_index<depth)]
        b, state = [], []
        for block in range(3):
            groups = [group for _,group in chosen[chosen.block==block].groupby('gemgroup')]
            assert len(groups)==8 and all(len(group)==depth for group in groups)
            idx = [[positions[cell] for cell in group.cell_id] for group in groups]
            b.append(np.mean([controls[ii].mean(0,dtype=np.float64) for ii in idx],axis=0))
            state.append(np.mean([cpa[ii].mean(0,dtype=np.float64) for ii in idx],axis=0))
        di = list(ref['depths']).index(depth)
        for model in range(4):
            scores, patterns = [], []
            for ti,(o,pr,m) in enumerate(tuples):
                observed = y['mean_equal_gemgroup'][task]-b[o]
                prediction = np.zeros_like(observed) if model==3 else ((base['additive'][task] if model==0 else base['compositional_ridge'][task] if model==1 else state[m])-b[pr])
                score = -float(np.mean(((prediction-observed)/scale)**2)); scores.append(score)
                pattern = 'S' if o==pr==m else 'M' if o==pr else 'P' if o==m else 'O' if pr==m else 'D'; patterns.append(pattern)
                error = abs(score-ref['atomic_utility'][task,label,di,model,ti]); max_error=max(max_error,error)
                records.append((str(ref['tasks'][task]),label,depth,str(ref['models'][model]),o,pr,m,error))
            for pi,pattern in enumerate(ref['patterns']):
                score = np.mean([value for value,tag in zip(scores,patterns) if tag==pattern])
                max_error=max(max_error,float(abs(score-ref['utility'][task,label,di,model,pi])))
report = dict(status='PASS' if max_error<1e-11 else 'FAIL', atomic_checks=len(records), pattern_checks=160, max_absolute_error=float(max_error), tolerance=1e-11, task_indices=[0,len(ref['tasks'])-1], allocation_labels=[0,29], depths=[8,135], role_tuples=27, scoring_module_imported=False)
pd.DataFrame(records,columns=['condition','allocation','depth','model','obs','pred','model_input','absolute_error']).to_csv(a.output/'tuple_checks.tsv',sep='\t',index=False)
(a.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if report['status']!='PASS': raise SystemExit(1)
