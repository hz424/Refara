"""Recompute metric ranks and concordance from current primary score summaries."""
from pathlib import Path, PurePosixPath
from collections import defaultdict, Counter
import argparse, csv, hashlib, itertools, json, math

ROOT=Path(__file__).resolve().parent
DEPTHS={('gse162632','original8'):(4,6,8),('gse162632','cap48'):(4,6,8),
        ('parse','common48'):(16,24,32,40,48),('parse','common128'):(16,24,32,40,48)}
FAMILIES={'NC','CM','TW','PCA','RBF','scGen','CPA','CellOT'}
POLICIES={'ALL_SHARED','OBS_PRED_SHARED_MODEL_SEPARATE','OBS_MODEL_SHARED_PRED_SEPARATE',
          'PRED_MODEL_SHARED_OBS_SEPARATE','ALL_DISJOINT'}
UNITS={f'root_{i:02d}' for i in range(8)}|{'equal_eight_root_mean'}

def require(ok,message):
    if not ok: raise ValueError(message)
def rows(p):
    with p.open(newline='') as f: return list(csv.DictReader(f,delimiter='\t'))
def key(row): return row['dataset'],row['regime'],int(row['depth']),row['unit'],row['policy']
def ranks(values):
    require(all(math.isfinite(x) for x in values),'Nonfinite score')
    return [1+sum(x<y for x in values)+(sum(x==y for x in values)-1)/2 for y in values]
def tau_b(a,b):
    concordant=discordant=ties_a=ties_b=0
    for i,j in itertools.combinations(range(len(a)),2):
        da=(a[i]>a[j])-(a[i]<a[j]);db=(b[i]>b[j])-(b[i]<b[j])
        concordant+=da*db>0;discordant+=da*db<0
        ties_a+=da==0 and db!=0;ties_b+=db==0 and da!=0
    denominator=math.sqrt((concordant+discordant+ties_a)*(concordant+discordant+ties_b))
    return (concordant-discordant)/denominator if denominator else None
def authenticate(root):
    manifest=json.loads((root/'SOURCE_MANIFEST.json').read_text())
    require(set(manifest['files'])=={'primary_metric_scores.tsv','expected_concordance.tsv'},'Unexpected input roster')
    for name,pin in manifest['files'].items():
        relative=PurePosixPath(name);require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe path')
        p=root/relative
        require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root.resolve()),'Missing/nonlocal input')
        require(p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['sha256'],'Input hash differs: '+name)
    return manifest
def recompute(root=ROOT):
    root=Path(root);manifest=authenticate(root);groups=defaultdict(list);expected={};output=[];summary=[];errors=[]
    for row in rows(root/'primary_metric_scores.tsv'):
        require(row['panel']=='primary' and row['seed']=='mean_of_scored_seeds','Unexpected input aggregation')
        require(row['allocations']=='1000','Unexpected allocation count')
        groups[key(row)].append(row)
    expected_keys={(d,r,depth,unit,policy) for (d,r),depths in DEPTHS.items()
                   for depth,unit,policy in itertools.product(depths,UNITS,POLICIES)}
    require(set(groups)==expected_keys,'Incomplete or unexpected metric-score axes')
    for row in rows(root/'expected_concordance.tsv'):
        k=key(row);require(k not in expected,'Duplicate validation row');expected[k]=row
    require(set(expected)==expected_keys,'Incomplete concordance validation axes')
    counts=defaultdict(Counter)
    for k,records in sorted(groups.items()):
        require(len(records)==8 and {r['family'] for r in records}==FAMILIES,'Incomplete eight-model roster')
        mse=[float(r['standardized_mse']) for r in records];delta=[float(r['delta_rmse']) for r in records]
        require(all(x>=0 for x in mse+delta),'Negative error metric')
        mr,dr=ranks(mse),ranks(delta)
        require(mr==[float(r['mse_rank']) for r in records] and dr==[float(r['delta_rmse_rank']) for r in records],'Saved metric ranks differ')
        ml={r['family'] for r,v in zip(records,mr) if v==min(mr)}
        dl={r['family'] for r,v in zip(records,dr) if v==min(dr)}
        agree=mr==dr;leaders=ml==dl;ref=expected[k];tau=tau_b(mse,delta)
        require(agree==(ref['ranks_identical']=='True') and leaders==(ref['leader_sets_identical']=='True'),'Concordance differs')
        require(ml==set(json.loads(ref['mse_leaders'])) and dl==set(json.loads(ref['delta_rmse_leaders'])),'Saved leaders differ')
        if tau is not None:
            error=abs(tau-float(ref['tau_b']));require(error<=2e-12,'Kendall concordance differs');errors.append(error)
        else: require(ref['tau_status']!='defined','Undefined Kendall concordance claimed defined')
        prefix='root' if k[3].startswith('root_') else 'aggregate'
        count=counts[k[:2]];count[prefix+'_designs']+=1;count[prefix+'_ranks_identical']+=agree;count[prefix+'_leaders_identical']+=leaders
        output.append(dict(dataset=k[0],regime=k[1],depth=k[2],unit=k[3],policy=k[4],ranks_identical=agree,leader_sets_identical=leaders,tau_b=tau,mse_leaders=json.dumps(sorted(ml)),delta_rmse_leaders=json.dumps(sorted(dl))))
    for (d,r),count in sorted(counts.items()):summary.append(dict(dataset=d,regime=r,**count))
    report={'status':'PASS_CURRENT_ED9_NUMERICAL_REPLAY','origin_release':manifest['origin_release'],
            'score_rows':sum(map(len,groups.values())),'root_designs':640,'aggregate_designs':80,
            'groups':summary,'maximum_tau_absolute_difference':max(errors,default=0),
            'scope':'Rank and concordance calculation from saved allocation-mean, mean-of-scored-seeds metrics; no cell-level rescoring or fitting.'}
    return output,report
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();require(not out.exists() and not out.is_relative_to(ROOT),'Use a new output directory outside the capsule')
    output,report=recompute();out.mkdir(parents=True)
    with (out/'CONCORDANCE.tsv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(output[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(output)
    (out/'REPLAY.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
