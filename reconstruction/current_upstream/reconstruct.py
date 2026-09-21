"""Resolve, check, select and export current reconstruction inputs without training."""
from pathlib import Path, PurePosixPath
import argparse, collections, hashlib, json, re, shutil, sys, time
HERE=Path(__file__).resolve().parent
PROFILES={
 'delivered':{'delivered_replay','delivered_provenance'},
 'code':{'native_custom_code','native_protocol','allocation_design'},
 'models':{'fitted_model','fitted_model_component','fit_completion'},
 'prepared':{'prepared_bundle_manifest','prepared_training_input'},
 'banks':{'fixed_prediction_bank','prediction_receipt'},
 'evaluation-inputs':{'registered_evaluation_input','native_input_metadata','allocation_design'},
 'cpa-resume':{'cpa_resume_input','cpa_completed_record'},
 'cpa-predict':set(),
}
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read_manifest(path=HERE/'CURRENT_RECONSTRUCTION_MANIFEST.json'):
 m=json.loads(Path(path).read_text()); seen=set()
 if any(not re.fullmatch(r'[a-z][a-z0-9_]*',key) for key in m['root_roles']):raise ValueError('Unsafe root name')
 for a in m['assets']:
  p=PurePosixPath(a['path'])
  if p.is_absolute() or '..' in p.parts or p.as_posix()!=a['path'] or a['root'] not in m['root_roles'] or a['id'] in seen:raise ValueError('Unsafe or duplicate asset')
  seen.add(a['id'])
 return m

def load_roots(filename, overrides=()):
 roots={};base=Path.cwd()
 if filename:
  f=Path(filename).resolve();base=f.parent;roots=json.loads(f.read_text())
 for item in overrides:
  key,sep,value=item.partition('=')
  if not sep or not key or not value:raise ValueError('Use --root name=/path')
  roots[key]=str(Path(value).expanduser().absolute())
 return {k:(Path(v).expanduser() if Path(v).expanduser().is_absolute() else base/Path(v)).absolute() for k,v in roots.items()}

def path_for(asset, roots):
 if asset['root'] not in roots:raise ValueError('Missing named root: '+asset['root'])
 return roots[asset['root']]/asset['path']

def choose(m,profile,group=None,recipe=None):
 assets=m['assets'];byid={a['id']:a for a in assets}
 if profile=='cpa-predict':
  if not recipe:raise ValueError('cpa-predict requires --recipe, for example original8:17')
  try:regime,seed=recipe.split(':');seed=int(seed)
  except (ValueError,TypeError):raise ValueError('Use a CPA recipe such as original8:17') from None
  recipes=[r for r in m.get('recipes',{}).get('cpa2560',[]) if r['regime']==regime and r['seed']==seed]
  if len(recipes)!=1:raise ValueError('Expected one CPA2560 recipe: '+recipe)
  if group and group!='gse162632/'+regime:raise ValueError('Group differs from selected CPA recipe')
  fits=[f for g in m.get('groups',[]) if (g['dataset'],g['regime'])==('gse162632',regime)
        for f in g['fits'] if f['family']=='CPA' and f['setting']=='primary' and f['seed']==seed]
  if len(fits)!=1 or fits[0]['asset'] not in byid:raise ValueError('Expected one recorded selected CPA2560 fit: '+recipe)
  model=byid[fits[0]['asset']]
  if 'fitted_model' not in model['categories']:raise ValueError('Selected CPA fit is not a fitted-model asset')
  wanted={model['id']}
  for name in ['train_cpa.py','common.py','prepare_data.py','predict_cpa_points.py']:
   sources=[a for a in assets if a['root']=='training_study' and a['path']=='code/'+name]
   if len(sources)!=1:raise ValueError('Missing or ambiguous CPA prediction source: '+name)
   wanted.add(sources[0]['id'])
  return [a for a in assets if a['id'] in wanted]
 if recipe:
  regime,seed=recipe.split(':');rr=next(r for r in m['recipes']['cpa2560'] if r['regime']==regime and r['seed']==int(seed))
  wanted={rr[k] for k in ['source_binding','source_completion','checkpoint','bundle_manifest']}
  b=byid[rr['bundle_manifest']];prefix=str(PurePosixPath(b['path']).parent)+'/'
  wanted.update(a['id'] for a in assets if (a['root']==b['root'] and a['path'].startswith(prefix)) or 'native_custom_code' in a['categories'])
  return [a for a in assets if a['id'] in wanted]
 selected=[a for a in assets if profile=='all' or set(a['categories'])&PROFILES[profile]]
 if group:selected=[a for a in selected if group in a['contexts'] or any(c.startswith(group+'/') for c in a['contexts']) or not a['contexts']]
 return selected

def check(assets,roots,hash_limit=None):
 rows=[]
 for a in assets:
  result={'id':a['id']}
  try:
   p=path_for(a,roots)
   if not p.is_file():result['status']='MISSING'
   elif a['bytes'] is not None and p.stat().st_size!=a['bytes']:result['status']='SIZE_MISMATCH'
   elif hash_limit is not None and p.stat().st_size>hash_limit:result['status']='SIZE_ONLY_HASH_NOT_CHECKED'
   elif not a['sha256']:result['status']='NO_EXPECTED_HASH'
   else:result['status']='HASH_MATCH' if sha(p)==a['sha256'] else 'HASH_MISMATCH'
  except ValueError:result['status']='ROOT_NOT_BOUND'
  rows.append(result)
 return rows

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['check','plan','export']);p.add_argument('--manifest',type=Path,default=HERE/'CURRENT_RECONSTRUCTION_MANIFEST.json');p.add_argument('--roots',type=Path);p.add_argument('--root',action='append',default=[]);p.add_argument('--profile',choices=['all']+list(PROFILES),default='delivered');p.add_argument('--group',choices=['gse162632/original8','gse162632/cap48','parse/common48','parse/common128']);p.add_argument('--recipe',help='CPA recipe, e.g. original8:17; continuation inputs by default, fitted inference inputs with --profile cpa-predict.');p.add_argument('--hash-limit-mib',type=float,help='Larger assets receive size checks only; 0 skips payload hashing. Omit to hash every selected asset.');p.add_argument('--output',type=Path,required=True);p.add_argument('--destination',type=Path,help='New directory for export; never uploads or publishes');p.add_argument('--dry-run',action='store_true');p.add_argument('--native-python',default='/path/to/native-cpa-python',help='Installed native CPA interpreter used in generated commands');a=p.parse_args()
 if a.output.exists():p.error('Use a new output receipt')
 m=read_manifest(a.manifest);roots=load_roots(a.roots,a.root);assets=choose(m,a.profile,a.group,a.recipe);started=time.monotonic();result={'schema':'reconstruction_action.v1','command':a.command,'manifest_sha256':sha(a.manifest),'selected_assets':len(assets),'selected_bytes':sum(x['bytes'] or 0 for x in assets),'training_performed':False,'publication_performed':False}
 code_root=roots.get('training_study');recipe=a.recipe or 'original8:17';regime,seed=recipe.split(':')
 if a.command=='plan':
  result.update(status='PLAN_ONLY_NOT_EXECUTED',asset_ids=[x['id'] for x in assets],required_roots=sorted({x['root'] for x in assets}),access={k:m['root_roles'][k] for k in sorted({x['root'] for x in assets})},commands={
   'validate_cpa_continuation':[sys.executable,str(HERE/'native_cpa.py'),'check','--roots',str(a.roots or 'ROOTS.json'),'--regime',regime,'--seed',seed,'--output','cpa-input-check.json'],
   'continue_cpa_training':[a.native_python,str(HERE/'native_cpa.py'),'resume','--roots',str(a.roots or 'ROOTS.json'),'--regime',regime,'--seed',seed,'--device','cuda','--output','new-cpa-fit'],
   'cpa_native_inference':[a.native_python,str(HERE/'native_cpa.py'),'predict','--roots',str(a.roots or 'ROOTS.json'),'--regime',regime,'--seed',seed,'--counts','native-control-counts.npz','--features','source-features.tsv','--labels','prediction-labels.tsv','--output','new-cpa-prediction'],
  },command_scope='Executable parameterized CPA entries. Other training families and full four-group evaluator remain native archived workflows; original nested bindings require an explicit relocation audit. Plan generation does not execute these commands.')
  if a.profile=='cpa-predict':
   result['commands']={'cpa_native_inference':result['commands']['cpa_native_inference']}
   result['command_scope']='CPA inference from the selected fitted state and supplied controls.'
 elif a.command=='check':
  if a.hash_limit_mib is not None and a.hash_limit_mib<0:p.error('Hash limit cannot be negative')
  rows=check(assets,roots,None if a.hash_limit_mib is None else int(a.hash_limit_mib*(1<<20)));counts=collections.Counter(r['status'] for r in rows);bad=set(counts)-{'HASH_MATCH','SIZE_ONLY_HASH_NOT_CHECKED'}
  result.update(status='FAIL_ASSET_CHECK' if bad else 'COMPLETE_PARTIAL_HASH_INVENTORY' if counts.get('SIZE_ONLY_HASH_NOT_CHECKED') else 'PASS_ALL_SELECTED_ASSET_HASHES',counts=dict(counts),assets=rows)
 else:
  if not a.destination:p.error('--destination is required for export')
  dest=a.destination.resolve()
  if dest.exists():p.error('Export destination must be new')
  result.update(status='EXPORT_PLAN_ONLY' if a.dry_run else 'EXPORT_PENDING',asset_ids=[x['id'] for x in assets],destination=str(dest),contains_original_provider_derived_assets=any(not x['included'] and 'native_custom_code' not in x['categories'] for x in assets),access_terms='Preserve provider terms; this local export is not authorization for public redistribution.')
  if not a.dry_run:
   dest.mkdir(parents=True)
   try:
    for x in assets:
     source=path_for(x,roots);target=dest/x['root']/x['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
     if sha(target)!=x['sha256'] or (x['bytes'] is not None and target.stat().st_size!=x['bytes']):raise ValueError('Exported bytes differ: '+x['id'])
    (dest/'ROOTS.json').write_text(json.dumps({k:k for k in sorted({x['root'] for x in assets})},indent=2)+'\n')
    (dest/'SELECTED_ASSETS.json').write_text(json.dumps(assets,indent=2)+'\n');result['status']='PASS_LOCAL_EXPORT_HASHES'
   except Exception as e:result.update(status='FAIL_LOCAL_EXPORT',error=str(e))
 result['seconds']=time.monotonic()-started;a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['assets','asset_ids','commands','access']},indent=2));return 1 if result['status'].startswith('FAIL') else 0
if __name__=='__main__':sys.exit(main())
