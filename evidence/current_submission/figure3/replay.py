#!/usr/bin/env python3
"""Recompute Figure 3 aggregates from bundled seed/root summaries, without fitting or rescoring cells."""
from pathlib import Path, PurePosixPath
import argparse, csv, hashlib, itertools, json, math
import numpy as np

ROOT=Path(__file__).resolve().parent
GROUPS=[('gse162632','original8',8),('gse162632','cap48',8),('parse','common48',48),('parse','common128',48)]
FAMILIES=['scGen','CPA','CellOT'];SEEDS=['17','29','43'];CONTRASTS=['D_to_O','D_to_P','S_to_M']


def require(ok,message):
    if not ok:raise ValueError(message)


def rows(path):
    with Path(path).open(newline='') as stream:return list(csv.DictReader(stream,delimiter='\t'))


def save_rows(path,records):
    with Path(path).open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]),delimiter='\t',lineterminator='\n');writer.writeheader();writer.writerows(records)


def authenticate(root=ROOT):
    root=Path(root).resolve();manifest=json.loads((root/'SOURCE_MANIFEST.json').read_text())
    for relative,rec in manifest['files'].items():
        rel=PurePosixPath(relative);require(not rel.is_absolute() and '..' not in rel.parts,'Unsafe source path')
        path=root/rel;require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root),'Missing/nonlocal source')
        data=path.read_bytes();require(len(data)==rec['bytes'] and hashlib.sha256(data).hexdigest()==rec['sha256'],'Changed source: '+relative)
    return manifest


def one(records,**keys):
    take=[r for r in records if all(str(r[k])==str(v) for k,v in keys.items())]
    require(len(take)==1,'Expected one source row for '+str(keys));return take[0]


def close(a,b,label):
    require(math.isfinite(a) and math.isfinite(b) and abs(a-b)<=2e-12,'Numerical disagreement: '+label)
    return abs(a-b)


def recompute(root=ROOT):
    root=Path(root);manifest=authenticate(root);plot=[];comparisons=[];errors=[];inherited_oracles=[]
    for dataset,regime,display_depth in GROUPS:
        udir=root/'source/utility'/dataset/regime;utility=rows(udir/'MODEL_ROOT_SUMMARIES.tsv');summary=json.loads((udir/'SUMMARY.json').read_text())
        require(summary['seeds']==[17,29,43] and summary['roots']==8 and summary['allocations']==1000,'Unexpected utility axes')
        values={}
        for depth,family,contrast in itertools.product(summary['depths'],FAMILIES,CONTRASTS):
            before=[];after=[];gain=[]
            for i in range(8):
                take=[one(utility,depth=depth,family=family,contrast=contrast,unit=f'root_{i:02d}',seed=seed) for seed in SEEDS]
                before.append(float(np.mean([float(r['utility_before']) for r in take])))
                after.append(float(np.mean([float(r['utility_after']) for r in take])))
                gain.append(float(np.mean([float(r['utility_gain']) for r in take])))
            values[depth,family,contrast]=dict(before=before,after=after,gain=gain)
        for family in FAMILIES:
            ddir=root/'source/displacement'/dataset/regime/family;displacement=rows(ddir/'root_rms.tsv');reference=rows(ddir/'reference_root_rms.tsv')
            inherited_oracles.append(json.loads((ddir/'NUMERICAL_QA.json').read_text()))
            for readout in ['raw_state','input_centered_effect','reference']+CONTRASTS:
                for i in range(8):
                    if readout in CONTRASTS:value=values[display_depth,family,readout]['gain'][i]
                    elif readout=='reference':value=float(one(reference,depth=display_depth,root_index=i)['rms'])
                    else:
                        seed_rms=[float(one(displacement,depth=display_depth,mode=readout,root_index=i,seed_summary=seed)['rms']) for seed in SEEDS]
                        value=float(np.sqrt(np.mean(np.square(seed_rms))))
                        saved=float(one(displacement,depth=display_depth,mode=readout,root_index=i,seed_summary='mean_of_squared_seed_displacements')['rms'])
                        errors.append(close(value,saved,'pooled seed RMS'))
                    plot.append(dict(dataset=dataset,regime=regime,reference_depth=display_depth,family=family,readout=readout,root_index=i,value=value))
        for depth,(a,b),contrast in itertools.product(summary['depths'],itertools.combinations(FAMILIES,2),CONTRASTS[:2]):
            left=values[depth,a,contrast];right=values[depth,b,contrast]
            before=float(np.mean(np.asarray(left['before'])-right['before']));after=float(np.mean(np.asarray(left['after'])-right['after']))
            classification='stable_order' if np.sign(before)==np.sign(after) and before!=0 and after!=0 else 'other'
            expected=one(summary['aggregate_pair_results'],depth=depth,family_a=a,family_b=b,contrast=contrast)
            errors.extend([close(before,float(expected['margin_before']),'pair before'),close(after,float(expected['margin_after']),'pair after')])
            require(classification==expected['exact_classification'],'Pair classification differs')
            comparisons.append(dict(dataset=dataset,regime=regime,depth=depth,family_a=a,family_b=b,contrast=contrast,margin_before=before,margin_after=after,margin_change=after-before,exact_classification=classification))
    expected=rows(root/'expected/FIGURE3_PRIMARY_ROOT_VALUES.tsv')
    require(len(plot)==len(expected)==576,'Incomplete plotted source axes')
    for row in plot:
        saved=one(expected,**{k:v for k,v in row.items() if k!='value'});errors.append(close(row['value'],float(saved['value']),'current plotted value'))
    primary=rows(root/'expected/FIGURE3_COMPLETE_PRIMARY_PAIRS.tsv')
    for saved in primary:
        actual=one(comparisons,**{k:saved[k] for k in ['dataset','regime','depth','family_a','family_b','contrast']})
        for key in ['margin_before','margin_after','margin_change']:errors.append(close(actual[key],float(saved[key]),'current primary pair'))
    counts={c:sum(r['contrast']==c for r in comparisons) for c in CONTRASTS[:2]}
    require(counts=={'D_to_O':48,'D_to_P':48} and len(primary)==24,'Incomplete pair-depth comparison axes')
    require(all(r['exact_classification']=='stable_order' for r in comparisons),'Current stable-order claim differs')
    report=dict(status='PASS_CURRENT_FIGURE3_SUMMARY_REPLAY',origin_release=manifest['origin_release'],plotted_root_values=576,primary_pair_comparisons=24,all_pair_depth_comparisons=counts,all_comparisons_stable_order=True,max_abs_reconstruction_difference=max(errors),source_level='saved per-seed/root summaries; equal-root and equal-seed reaggregation',fresh_model_fitting=False,fresh_cell_level_scoring=False,biological_roots_per_group=8,seeds=[17,29,43],allocations=1000,group_count=4,paired_regimes_reuse_roots=True,historical_representation_oracles=dict(case_count=sum(q['representation_case_count'] for q in inherited_oracles),max_abs=max(q['equivalent_representation_max_abs'] for q in inherited_oracles),reexecuted=False))
    return plot,comparisons,report


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output.expanduser().resolve();require(not out.exists() and not out.is_relative_to(ROOT),'Use a new output directory outside the capsule')
    plot,pairs,report=recompute();out.mkdir(parents=True)
    save_rows(out/'FIGURE3_PRIMARY_ROOT_VALUES.tsv',plot);save_rows(out/'ALL_PAIR_DEPTH_COMPARISONS.tsv',pairs)
    # Preserve the bound comparison table; all of its values were checked above.
    (out/'FIGURE3_COMPLETE_PRIMARY_PAIRS.tsv').write_bytes((ROOT/'expected/FIGURE3_COMPLETE_PRIMARY_PAIRS.tsv').read_bytes())
    (out/'REPLAY.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
