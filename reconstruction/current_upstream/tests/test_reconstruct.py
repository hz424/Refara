from pathlib import Path
import hashlib,json,shutil,subprocess,sys,tempfile,unittest
TOOL=Path(__file__).resolve().parents[1]/'reconstruct.py'
class PortableAssetChecks(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);(self.root/'assets').mkdir();(self.root/'assets/input.txt').write_bytes(b'bound scientific input\n')
  self.manifest=self.root/'manifest.json';self.roots=self.root/'ROOTS.json';self.roots.write_text(json.dumps({'study':'assets'}));self.manifest.write_text(json.dumps({'root_roles':{'study':{'status':'test'}},'assets':[{'id':'study:input.txt','root':'study','path':'input.txt','sha256':hashlib.sha256(b'bound scientific input\n').hexdigest(),'bytes':23,'categories':['native_custom_code'],'contexts':[],'included':False}]}))
 def run_tool(self,command,name,*extra,roots=None):
  out=self.root/name;cmd=[sys.executable,str(TOOL),command,'--manifest',str(self.manifest),'--roots',str(roots or self.roots),'--profile','code','--output',str(out),*map(str,extra)];done=subprocess.run(cmd,capture_output=True,text=True);return done,json.loads(out.read_text()) if out.exists() else None
 def test_relative_root_and_mutation_rejection(self):
  done,r=self.run_tool('check','good.json');self.assertEqual(done.returncode,0,done.stderr);self.assertEqual(r['status'],'PASS_ALL_SELECTED_ASSET_HASHES')
  (self.root/'assets/input.txt').write_bytes(b'altered scientific input')
  done,r=self.run_tool('check','bad.json');self.assertEqual(done.returncode,1);self.assertEqual(r['status'],'FAIL_ASSET_CHECK')
 def test_missing_root_is_not_a_pass(self):
  self.roots.write_text('{}');done,r=self.run_tool('check','missing.json');self.assertEqual(done.returncode,1);self.assertEqual(r['counts'],{'ROOT_NOT_BOUND':1})
 def test_export_remains_checkable_after_relocation(self):
  dest=self.root/'export';done,r=self.run_tool('export','copy.json','--destination',dest);self.assertEqual(done.returncode,0,done.stderr);self.assertEqual(r['status'],'PASS_LOCAL_EXPORT_HASHES')
  dest.rename(self.root/'moved');done,r=self.run_tool('check','moved.json',roots=self.root/'moved/ROOTS.json');self.assertEqual(done.returncode,0,done.stderr);self.assertEqual(r['status'],'PASS_ALL_SELECTED_ASSET_HASHES')
class CPAPredictionAssets(PortableAssetChecks):
 def setUp(self):
  super().setUp()
  assets=[]
  def add(root,path,category):
   target=self.root/'source'/root/path;target.parent.mkdir(parents=True,exist_ok=True)
   content=(root+':'+path+'\n').encode();target.write_bytes(content)
   asset=dict(id=root+':'+path,root=root,path=path,sha256=hashlib.sha256(content).hexdigest(),bytes=len(content),categories=[category],contexts=[],included=False)
   assets.append(asset);return asset['id']
  self.sources=[add('training_study','code/'+name,'native_custom_code') for name in ['train_cpa.py','common.py','prepare_data.py','predict_cpa_points.py']]
  self.state=add('cpa_continuation','results/original8/seed17/model_state.pt','fitted_model')
  recipe=dict(regime='original8',seed=17)
  for field in ['source_binding','source_completion','checkpoint']:
   recipe[field]=add('training_study','training/'+field,'cpa_resume_input')
  recipe['bundle_manifest']=add('training_study','data/manifest.json','prepared_bundle_manifest')
  self.training=add('training_study','data/training.npy','prepared_training_input')
  self.bank=add('training_study','banks/points.npy','fixed_prediction_bank')
  add('training_study','code/unrelated.py','native_custom_code')
  self.data=dict(root_roles={root:{} for root in ['training_study','cpa_continuation']},assets=assets,
                 recipes={'cpa2560':[recipe]},groups=[dict(dataset='gse162632',regime='original8',fits=[dict(family='CPA',setting='primary',seed=17,asset=self.state)])])
  self.manifest.write_text(json.dumps(self.data))
  self.roots.write_text(json.dumps({root:'source/'+root for root in self.data['root_roles']}))

 def predict_tool(self,command,name,*extra,roots=None):
  return self.run_tool(command,name,'--profile','cpa-predict','--recipe','original8:17',*extra,roots=roots)

 # The inherited tests describe the one-file fixture, not this CPA fixture.
 def test_relative_root_and_mutation_rejection(self):
  done,r=self.predict_tool('check','good.json');self.assertEqual(done.returncode,0,done.stderr)
  (self.root/'source/cpa_continuation/results/original8/seed17/model_state.pt').unlink()
  done,r=self.predict_tool('check','bad.json');self.assertEqual(done.returncode,1)
  self.assertEqual(r['counts'],{'HASH_MATCH':4,'MISSING':1})

 def test_missing_root_is_not_a_pass(self):
  self.roots.write_text('{}');done,r=self.predict_tool('check','missing.json')
  self.assertEqual(done.returncode,1);self.assertEqual(r['counts'],{'ROOT_NOT_BOUND':5})

 def test_export_remains_checkable_after_relocation(self):
  dest=self.root/'export';done,r=self.predict_tool('export','copy.json','--destination',dest)
  self.assertEqual(done.returncode,0,done.stderr)
  self.assertEqual(set(r['asset_ids']),{self.state,*self.sources})
  self.assertFalse((dest/'training_study/data').exists())
  self.assertFalse((dest/'training_study/training').exists())
  self.assertFalse((dest/'training_study/banks').exists())
  dest.rename(self.root/'moved');shutil.rmtree(self.root/'source')
  done,r=self.predict_tool('check','moved.json',roots=self.root/'moved/ROOTS.json')
  self.assertEqual(done.returncode,0,done.stderr);self.assertEqual(r['counts'],{'HASH_MATCH':5})

 def test_recipe_keeps_legacy_continuation_selection(self):
  done,r=self.run_tool('export','legacy.json','--recipe','original8:17','--dry-run','--destination',self.root/'export')
  self.assertEqual(done.returncode,0,done.stderr)
  self.assertIn(self.training,r['asset_ids']);self.assertNotIn(self.state,r['asset_ids']);self.assertNotIn(self.bank,r['asset_ids'])

 def test_missing_or_ambiguous_fit_cannot_export(self):
  for mode in ['missing_fit','ambiguous_fit','missing_asset']:
   with self.subTest(mode=mode):
    changed=json.loads(json.dumps(self.data))
    if mode=='missing_fit':changed['groups'][0]['fits']=[]
    elif mode=='ambiguous_fit':changed['groups'][0]['fits']*=2
    else:changed['assets']=[a for a in changed['assets'] if a['id']!=self.state]
    self.manifest.write_text(json.dumps(changed));dest=self.root/mode
    done,_=self.predict_tool('export',mode+'.json','--destination',dest)
    self.assertNotEqual(done.returncode,0);self.assertIn('Expected one recorded selected CPA2560 fit',done.stderr)
    self.assertFalse(dest.exists())

 def test_loader_source_is_required(self):
  self.data['assets']=[a for a in self.data['assets'] if a['path']!='code/prepare_data.py']
  self.manifest.write_text(json.dumps(self.data))
  done,_=self.predict_tool('check','loader.json')
  self.assertNotEqual(done.returncode,0);self.assertIn('prepare_data.py',done.stderr)

 def test_prediction_plan_has_only_inference_command(self):
  done,r=self.predict_tool('plan','plan.json')
  self.assertEqual(done.returncode,0,done.stderr)
  self.assertEqual(set(r['commands']),{'cpa_native_inference'})


if __name__=='__main__':unittest.main()
