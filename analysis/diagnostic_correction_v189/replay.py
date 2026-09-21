#!/usr/bin/env python3
"""Design-based structural-zero correction to the full 56-member diagnostic.

All family members are retained. Under the fixed direct-effect squared-loss
design, equal-depth reference rotation averages the observation reference to
the Shared union mean; direct/direct rotation-minus-Shared contrasts are
therefore identically zero. The rule uses identities, never observed magnitude.
This is a numerical correction of descriptive working-model diagnostics and
does not establish calibrated coverage.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import platform
import numpy as np

COMPONENT='Diagnostic_correction_v189'
LEGACY_DIR='data/derived/public_held_out_evidence'
ROOT_NAME='GSE162632_PAIRED_CONSTRUCTION_CHANGES_ROOTS_PUBLIC_V1.tsv'
SUMMARY_NAME='GSE162632_PAIRED_CONSTRUCTION_CHANGES_FULL_56_PUBLIC_V1.tsv'
UTILITY_NAME='GSE162632_ROOT_UTILITIES_PUBLIC_V1.tsv'
CORRECTED_ROOT='PAIRED_CHANGES_ROOTS_CORRECTED_V189.tsv'
CORRECTED_SUMMARY='PAIRED_CHANGES_FULL_56_CORRECTED_V189.tsv'
DIRECT_METHODS=frozenset(('NO_CHANGE_DIRECT_V1','CONTEXT_MEAN_EFFECT_DIRECT_V1',
    'TWO_WAY_ADDITIVE_RIDGE_DIRECT_V1','PCA64_ADDITIVE_RIDGE_DIRECT_V1','RBF_KERNEL_RIDGE_DIRECT_V1'))
STRUCTURAL_RULE='fixed direct/direct pair; rotation-minus-Shared; equal-depth blocks exhaust Shared union; squared loss; score then equal-average'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def read_rows(path):
    with path.open(newline='') as stream:return list(csv.DictReader(stream,delimiter='\t'))

def table_bytes(rows):
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n')
    writer.writeheader();writer.writerows(rows)
    return stream.getvalue().encode()

def structural(row):
    return (row['source_scheme_id']=='NAIVE_SHARED' and row['target_scheme_id']=='UNIT_AWARE_CROSSFIT'
        and row['method_a'] in DIRECT_METHODS and row['method_b'] in DIRECT_METHODS)

def diagnose(raw,mask):
    values=np.array(raw,dtype=np.float64,copy=True)
    if values.shape!=(8,56) or np.asarray(mask).shape!=(56,) or not np.isfinite(values).all():
        raise ValueError('Expected finite eight-root complete 56-member matrix')
    values[:,mask]=0.0
    mean=values.mean(axis=0);se=values.std(axis=0,ddof=1)/np.sqrt(8)
    signs=np.array(list(itertools.product((-1.,1.),repeat=8)))
    signed=signs[:,:,None]*(values-mean)[None,:,:]
    numerators=signed.mean(axis=1);denominators=signed.std(axis=1,ddof=1)/np.sqrt(8)
    active=~mask
    fallback=bool(np.any(se[active]<=0) or np.any(denominators[:,active]<=0)
        or not np.isfinite(denominators[:,active]).all())
    statistics=np.zeros((256,56))
    if fallback:
        critical=float('inf');maxima=np.full(256,float('inf'))
        lower=np.full(56,-float('inf'));upper=np.full(56,float('inf'))
    else:
        statistics[:,active]=numerators[:,active]/denominators[:,active]
        maxima=np.max(np.abs(statistics),axis=1)
        critical=float(np.sort(maxima)[math.ceil(.95*256)-1])
        lower=mean-critical*se;upper=mean+critical*se
    return dict(values=values,point=mean,se=se,critical=critical,lower=lower,upper=upper,
        statistics=statistics,maxima=maxima,fallback=fallback,flags=(lower>0)|(upper<0))

def inputs(source):
    legacy=source/LEGACY_DIR
    roots=read_rows(legacy/ROOT_NAME);summaries=read_rows(legacy/SUMMARY_NAME)
    if len(roots)!=448 or len(summaries)!=56:raise ValueError('Incomplete source family')
    matrix=np.full((8,56),np.nan);mask=np.zeros(56,dtype=bool);seen=set()
    for row in roots:
        r=int(row['root_order'])-1;k=int(row['paired_change_index'])
        if (r,k) in seen:raise ValueError('Duplicate root-family key')
        seen.add((r,k));matrix[r,k]=float(row['root_paired_change']);mask[k]=structural(row)
    if [int(r['paired_change_index']) for r in summaries]!=list(range(56)):raise ValueError('Family order changed')
    if not np.isfinite(matrix).all() or len(seen)!=448:raise ValueError('Incomplete root matrix')
    assert np.flatnonzero(mask).tolist()==[28,29,30,31,35,36,37,41,42,46]
    assert all(structural(row)==mask[int(row['paired_change_index'])] for row in summaries)
    return roots,summaries,matrix,mask

def regenerate(source):
    roots,summaries,raw,mask=inputs(source)
    fixed=diagnose(raw,mask)
    legacy=diagnose(raw,np.zeros(56,dtype=bool))
    if fixed['fallback']:raise RuntimeError('Unexpected diagnostic fallback')
    assert abs(fixed['critical']-4.729828863130656)<3e-14
    oldflags=np.array([r['max_t_resolved_change_direction']!='UNRESOLVED' for r in summaries])
    assert np.array_equal(oldflags[~mask],fixed['flags'][~mask]) and sum(fixed['flags'][~mask])==33
    corrected_roots=[]
    for row in roots:
        r=int(row['root_order'])-1;k=int(row['paired_change_index'])
        corrected_roots.append({**row,'root_paired_change':format(fixed['values'][r,k],'.17g'),
            'legacy_float_root_paired_change':row['root_paired_change'],
            'design_structural_zero':str(bool(mask[k])).lower()})
    corrected_summaries=[]
    for k,row in enumerate(summaries):
        corrected_summaries.append({**row,
            'point_paired_change':format(fixed['point'][k],'.17g'),
            'root_standard_error':format(fixed['se'][k],'.17g'),
            'max_t_lower':format(fixed['lower'][k],'.17g'),
            'max_t_upper':format(fixed['upper'][k],'.17g'),
            'max_t_critical_value':format(fixed['critical'],'.17g'),
            'max_t_resolved_change_direction':'POSITIVE' if fixed['lower'][k]>0 else 'NEGATIVE' if fixed['upper'][k]<0 else 'UNRESOLVED',
            'design_structural_zero':str(bool(mask[k])).lower(),
            'legacy_float_critical_value':row['max_t_critical_value'],
            'diagnostic_scope':'WORKING_MODEL_DIAGNOSTIC_NO_CALIBRATED_COVERAGE'})
    # A second algebraically equivalent order uses each method's construction
    # shift before taking the method contrast, rather than contrast then shift.
    utility=read_rows(source/LEGACY_DIR/UTILITY_NAME)
    lookup={(int(r['root_order'])-1,r['scheme_id'],r['method_id']):float(r['root_utility']) for r in utility}
    alternative=np.empty_like(raw)
    for row in roots:
        r=int(row['root_order'])-1;k=int(row['paired_change_index']);a=row['method_a'];b=row['method_b']
        s=row['source_scheme_id'];t=row['target_scheme_id']
        alternative[r,k]=(lookup[r,t,a]-lookup[r,s,a])-(lookup[r,t,b]-lookup[r,s,b])
    alternate=diagnose(alternative,mask)
    assert np.array_equal(alternate['flags'],fixed['flags'])
    assert abs(alternate['critical']-fixed['critical'])<3e-12
    assert np.max(np.abs(alternate['lower']-fixed['lower']))<3e-12
    assert np.max(np.abs(alternate['upper']-fixed['upper']))<3e-12
    # Exact decimal float64 round trip and changed structural noise both leave
    # the corrected diagnostic bit-identical; no effect-size tolerance is used.
    roundtrip=np.array([[float(format(v,'.17g')) for v in row] for row in raw])
    assert np.array_equal(roundtrip,raw)
    noisy=raw.copy();noisy[:,mask]=np.arange(80).reshape(8,10)*np.finfo(float).eps
    noisy_fixed=diagnose(noisy,mask)
    assert np.array_equal(noisy_fixed['maxima'],fixed['maxima'])
    assert np.array_equal(noisy_fixed['lower'],fixed['lower'])
    assert (np.array([lookup[r,'NAIVE_SHARED','PCA64_ADDITIVE_RIDGE_DIRECT_V1']-lookup[r,'NAIVE_SHARED','SCGEN_2_1_1_ABSOLUTE_V1'] for r in range(8)])<0).all()
    for scheme in ('INDEPENDENT_SPLIT','UNIT_AWARE_CROSSFIT'):
        assert all(lookup[r,scheme,'PCA64_ADDITIVE_RIDGE_DIRECT_V1']>
            lookup[r,scheme,'SCGEN_2_1_1_ABSOLUTE_V1'] for r in range(8))
    # Focal shifts are retained in both constructions, all eight positive.
    focal=[r for r in corrected_summaries if r['method_a']=='PCA64_ADDITIVE_RIDGE_DIRECT_V1' and 'SCGEN' in r['method_b']]
    assert len(focal)==2 and all(float(r['max_t_lower'])>0 for r in focal)
    assert all(float(r['root_paired_change'])>0 for r in corrected_roots
        if r['method_a']=='PCA64_ADDITIVE_RIDGE_DIRECT_V1' and 'SCGEN' in r['method_b'])
    output={CORRECTED_ROOT:table_bytes(corrected_roots),CORRECTED_SUMMARY:table_bytes(corrected_summaries)}
    manifest=dict(component=COMPONENT,revision='1.8.9',status='PASS',
        structural_rule=STRUCTURAL_RULE,structural_zero_indices=np.flatnonzero(mask).tolist(),
        family_members=56,structural_zero_members=10,nondegenerate_members=46,root_rows=448,
        legacy_float_critical=legacy['critical'],corrected_critical=fixed['critical'],
        nondegenerate_zero_excluding_flags=33,nondegenerate_flags_unchanged=True,
        sign_vectors=256,ordered_quantile_index_one_based=244,family_members_removed=0,
        structural_statistics_zero_in_every_sign_vector=True,
        structural_mean_standard_error_and_bounds_exactly_zero=True,
        fallback='Invalid observed or sign-vector standard error in any nonstructural member makes the entire family unbounded; structural zeros use their algebraic zero statistics.',
        calibrated_coverage_established=False,within_construction_84_family_modified=False,
        tests=dict(equivalent_arithmetic_critical_absolute_difference=abs(alternate['critical']-fixed['critical']),
            equivalent_arithmetic_bound_max_absolute_difference=float(max(np.max(np.abs(alternate['lower']-fixed['lower'])),np.max(np.abs(alternate['upper']-fixed['upper'])))),
            float64_decimal_roundtrip_bit_identical=True,structural_noise_invariance_bit_identical=True,
            focal_shifts_positive_in_all_eight_roots=True),
        legacy_source_sha256={name:digest(source/LEGACY_DIR/name) for name in (ROOT_NAME,SUMMARY_NAME,UTILITY_NAME)},
        artifacts={name:hashlib.sha256(data).hexdigest() for name,data in output.items()})
    output['DIAGNOSTIC_CORRECTION_MANIFEST.json']=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode()
    return output,manifest

def validate(source):
    generated,manifest=regenerate(source)
    for name,data in generated.items():
        path=source/COMPONENT/name
        if path.read_bytes()!=data:raise RuntimeError('Corrected artifact differs from recomputation: '+name)
    return manifest

def apply_to_plot(source,legacy):
    """Retain validated historical metadata, substituting current full-family numerics."""
    manifest=validate(source)
    summaries=read_rows(source/COMPONENT/CORRECTED_SUMMARY)
    by_index={int(r['paired_change_index']):r for r in summaries}
    fields=('point_paired_change','root_standard_error','max_t_lower','max_t_upper','max_t_critical_value')
    for row in legacy['summary_rows']:
        new=by_index[row['paired_change_index']]
        for field in fields:row[field]=float(new[field])
        row['max_t_resolved_change_direction']=new['max_t_resolved_change_direction']
        row['design_structural_zero']=new['design_structural_zero']=='true'
    for row in legacy['root_rows']:
        if structural(row):row['root_paired_change']=0.0
    # selected_summaries / selected_roots reference these same dictionaries.
    legacy['current_correction']=manifest
    return legacy

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-data-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--verify',action='store_true')
    args=p.parse_args()
    if args.verify:manifest=validate(args.source_data_root)
    else:
        outputs,manifest=regenerate(args.source_data_root)
        output=args.output_dir or args.source_data_root/COMPONENT
        output.mkdir(parents=True,exist_ok=True)
        for name,data in outputs.items():(output/name).write_bytes(data)
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
