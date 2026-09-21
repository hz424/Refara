"""Parameterize the exact current CPA continuation and per-cell prediction operations.

check is read-only. resume trains into a new output directory. predict only
loads an authenticated fitted state and writes predictions into a new directory.
"""
from pathlib import Path
import argparse, csv, importlib.metadata, inspect, json, sys, time
from reconstruct import HERE, read_manifest, load_roots, path_for, choose, check, sha

def require(condition,message):
 if not condition:raise ValueError(message)

def setup(m,roots,regime,seed):
 recipe=next(r for r in m['recipes']['cpa2560'] if (r['regime'],r['seed'])==(regime,seed))
 byid={a['id']:a for a in m['assets']}
 def path(key):return path_for(byid[recipe[key]],roots)
 return recipe,byid,path

def check_runtime(expected):
 actual={k:importlib.metadata.version(k) for k in expected}
 if actual!=expected:raise ValueError('Native runtime differs: '+json.dumps({'expected':expected,'actual':actual}))
 return actual

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['check','resume','predict']);p.add_argument('--manifest',type=Path,default=HERE/'CURRENT_RECONSTRUCTION_MANIFEST.json');p.add_argument('--roots',type=Path,required=True);p.add_argument('--regime',choices=['original8','cap48'],required=True);p.add_argument('--seed',type=int,choices=[17,29,43],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--counts',type=Path);p.add_argument('--features',type=Path);p.add_argument('--labels',type=Path);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--check-runtime',action='store_true');a=p.parse_args()
 if a.output.exists():p.error('Use a new output path')
 started=time.monotonic();m=read_manifest(a.manifest);roots=load_roots(a.roots);r,byid,path=setup(m,roots,a.regime,a.seed)
 selected=choose(m,'cpa-predict' if a.command=='predict' else 'cpa-resume',recipe=a.regime+':'+str(a.seed))
 checks=check(selected,roots)
 if any(row['status']!='HASH_MATCH' for row in checks):raise ValueError('CPA input checks failed: '+str([x for x in checks if x['status']!='HASH_MATCH']))
 if a.command!='predict':
  binding=json.loads(path('source_binding').read_text());done=json.loads(path('source_completion').read_text());bm=json.loads(path('bundle_manifest').read_text())
  require(binding['maximum_epoch']==done['completed_epoch']==1280, 'CPA continuation metadata differs')
  require(binding['candidate']=={'lr':0.0003,'reg_adv':10.0} and binding['batch_size']==1024, 'CPA continuation metadata differs')
  require(binding['schedule']=='per_epoch' and binding['weighted_reconstruction'] and binding['omit_training_r2'], 'CPA continuation metadata differs')
  require(binding['seed']==a.seed and bm['dataset']=='gse162632' and bm['regime']==a.regime and bm['fold_id']=='full', 'CPA continuation metadata differs')
  require(len(bm['train_donors'])==40 and not bm['validation_donors'] and not bm['validation_tasks_count'], 'CPA continuation metadata differs')
  require(sha(path('bundle_manifest'))==binding['binding']['bundle_manifest_sha256'], 'CPA continuation metadata differs')
 native=roots['training_study']/'code'
 for name,expected in [('train_cpa.py',r['trainer_sha256']),('common.py',r['scorer_sha256']),('prepare_data.py',r['loader_sha256'])]:
  if sha(native/name)!=expected:raise ValueError('Native source differs: '+name)
 receipt={'schema':'current_cpa_operation.v1','regime':a.regime,'seed':a.seed,'command':a.command,'manifest_sha256':sha(a.manifest),'input_hashes_checked':len(checks),'native_training_performed':False,'native_inference_performed':False,'runtime_expected':r['runtime']}
 if a.command=='check':
  if a.check_runtime:receipt['runtime_actual']=check_runtime(r['runtime'])
  receipt.update(status='PASS_CPA_CONTINUATION_INPUTS',source_epoch=1280,target_epoch=2560,training_cells=r['training_cells'],training_donors=40,seconds=time.monotonic()-started)
  a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['status']);return
 receipt['runtime_actual']=check_runtime(r['runtime']);sys.path.insert(0,str(native))
 if a.command=='resume':
  if a.device!='cuda':p.error('resume requires --device cuda for the recorded continuation')
  from prepare_data import load_bundle
  from train_cpa import fit_bundle
  bundle=load_bundle(path('bundle_manifest').parent,load_counts=True,verify=True)
  a.output.mkdir(parents=True)
  _,result=fit_bundle(bundle,a.output,candidate=binding['candidate'],seed=a.seed,max_epochs=2560,checkpoints=[1280,2560],batch_size=1024,use_gpu=True,resume_from=path('checkpoint'),schedule=binding['schedule'],weighted=binding['weighted_reconstruction'],omit_r2=binding['omit_training_r2'],binding=binding['binding'])
  if result['status']!='COMPLETE' or result['completed_epoch']!=2560:raise ValueError('Continuation incomplete')
  receipt.update(status='COMPLETE_NATIVE_CPA_CONTINUATION',native_training_performed=True)
 else:
  if not all([a.counts,a.features,a.labels]):p.error('predict requires --counts, --features and --labels')
  import numpy as np
  from scipy import sparse
  from predict_cpa_points import load_predictor, predict_points
  group=next(g for g in m['groups'] if (g['dataset'],g['regime'])==('gse162632',a.regime))
  fit=next(f for f in group['fits'] if f['family']=='CPA' and f['setting']=='primary' and f['seed']==a.seed)
  model=byid[fit['asset']];state=path_for(model,roots)
  if sha(state)!=model['sha256']:raise ValueError('Selected 2560 model hash differs')
  with a.features.open(newline='') as f:features=[row['feature_id'] for row in csv.DictReader(f,delimiter='\t')]
  with a.labels.open(newline='') as f:labels=list(csv.DictReader(f,delimiter='\t'))
  counts=sparse.load_npz(a.counts)
  if len(labels)!=counts.shape[0]:raise ValueError('Labels/counts row counts differ')
  predictor=load_predictor(state,device=a.device)
  values=predict_points(predictor,counts,contexts=[x['context'] for x in labels],conditions=[x['condition'] for x in labels],feature_ids=features)
  a.output.mkdir(parents=True);np.save(a.output/'predicted_states.npy',values,allow_pickle=False)
  receipt.update(status='PASS_NATIVE_CPA_PREDICTION',native_inference_performed=True,model_asset_id=model['id'],shape=list(values.shape),output_sha256=sha(a.output/'predicted_states.npy'),inputs={k:{'sha256':sha(v),'bytes':v.stat().st_size} for k,v in [('counts',a.counts),('features',a.features),('labels',a.labels)]})
 receipt['seconds']=time.monotonic()-started;(a.output/'RECONSTRUCTION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['status'])
if __name__=='__main__':main()
