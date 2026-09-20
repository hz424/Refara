"""Numerical replay of current program-response conclusions; no plotting or model fitting."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mean(rows):return float(np.mean(rows))
def balanced(rows,key):
 groups={}
 for r in rows:groups.setdefault((r['batch_id'],r['capture_orientation']),[]).append(float(r[key]))
 batches={}
 for (b,o),values in groups.items():batches.setdefault(b,[]).append(mean(values))
 if len(batches)!=8 or any(len(v)!=2 for v in batches.values()):raise ValueError('Expected eight batches and two capture orientations each')
 return mean([mean(v) for v in batches.values()])
def summarize_task_rows(rows):
 roots=sorted({r['root_id'] for r in rows});types=sorted({r['cell_type'] for r in rows});programs=sorted({r['program_id'] for r in rows});seeds=['17','29','43'];blocks=['1','2']
 if (len(roots),len(types),len(programs))!=(8,5,37):raise ValueError('Task/program axes differ')
 lookup={}
 for r in rows:
  key=(r['observation_block'],r['seed_summary'],r['root_id'],r['cell_type'],r['program_id'])
  if key in lookup:raise ValueError('Duplicate task/program/seed row')
  lookup[key]=r
 expected={(b,s,r,t,p) for b in blocks for s in seeds+['mean_program_score'] for r in roots for t in types for p in programs}
 if set(lookup)!=expected:raise ValueError('Incomplete task/program/seed Cartesian product')
 direction=0;counts=[];errors={b:{r:{k:[] for k in ['S','D']} for r in roots} for b in blocks};identity=0.;ensemble_gap=0.
 for r in roots:
  for t in types:
   for p in programs:
    values={b:np.array([[float(lookup[b,s,r,t,p][k]) for k in ['observed','S_selected','D_selected']] for s in seeds]) for b in blocks}
    if not np.array_equal(values['1'][:,1:],values['2'][:,1:]):raise ValueError('Prediction changed with observation reference')
    av=values['1'][:,1:].mean(axis=0)
    for b in blocks:
     recorded=np.array([float(lookup[b,'mean_program_score',r,t,p][k]) for k in ['S_selected','D_selected']])
     if not np.allclose(av,recorded,atol=2e-15,rtol=2e-13):raise ValueError('Recorded seed mean differs')
     a=values[b];e=(a[:,1:]-a[:,0,None])**2
     for i,k in enumerate(['S','D']):errors[b][r][k].extend(e[:,i].tolist())
     ensemble_gap=max(ensemble_gap,float(np.max(np.abs(e.mean(axis=0)-(av-a[:,0].mean())**2))))
    flag=bool(av[0]*av[1]<0 and min(abs(av))>.025);direction+=flag
    counts.append(dict(root=r,cell_type=t,program=p,discordant=flag))
    a=values['1'];b=values['2'];d1=(a[:,2]-a[:,0])**2-(a[:,1]-a[:,0])**2;d2=(b[:,2]-b[:,0])**2-(b[:,1]-b[:,0])**2
    identity=max(identity,float(np.max(np.abs((d2-d1)-2*(a[:,1]-a[:,2])*(b[:,0]-a[:,0])))))
 donor={b:{r:{k:mean(v) for k,v in d.items()} for r,d in byroot.items()} for b,byroot in errors.items()}
 centres={b:{k:mean([v[k] for v in donor[b].values()]) for k in ['S','D']} for b in blocks}
 for b in blocks:centres[b]['D_minus_S']=centres[b]['D']-centres[b]['S']
 return dict(direction_discordant=direction,direction_denominator=1480,threshold=.025,centres=centres,donor_errors=donor,observation_shift_identity_max_abs=identity,max_error_of_mean_vs_mean_error_gap=ensemble_gap,order_changes=centres['1']['D_minus_S']*centres['2']['D_minus_S']<0)
def replay(output):
 manifest=json.loads((ROOT/'MANIFEST.json').read_text())
 for n,h in manifest['inputs'].items():
  if sha(ROOT/n)!=h:raise ValueError('Input hash differs: '+n)
 def read(name):
  with (ROOT/name).open() as f:return list(csv.DictReader(f,delimiter='\t'))
 if sha(ROOT/'expected.json')!=manifest['expected_sha256']:raise ValueError('Expected result hash differs')
 expected=json.loads((ROOT/'expected.json').read_text());pairs=read('inputs/donor_errors.tsv');result={}
 for regime in ['original8','cap48']:
  current=summarize_task_rows(read('inputs/'+regime+'_task_programs.tsv'))
  if current['direction_discordant']!=expected[regime]['direction_discordant']:raise ValueError('Direction count differs')
  for b in ['1','2']:
   rows=[r for r in pairs if r['regime']==regime and r['cohort_display']=='original8' and r['observation_block']==b]
   if len(rows)!=8:raise ValueError('Original donor axes')
   for r in rows:
    for k in ['S','D']:
     if not np.isclose(current['donor_errors'][b][r['root_id']][k],float(r['mse_'+k]),atol=3e-14,rtol=2e-12):raise ValueError('Task-to-donor score mismatch')
  new39={}
  for b in ['1','2']:
   rows=[r for r in pairs if r['regime']==regime and r['cohort_display']=='new39' and r['observation_block']==b]
   if len(rows)!=39 or len({r['donor_id'] for r in rows})!=39:raise ValueError('Additional donor axes')
   vals={k:balanced(rows,'mse_'+k) for k in ['S','D']};vals['D_minus_S']=vals['D']-vals['S'];new39[b]=vals
  for cohort,centres in [('original8',current['centres']),('new39',new39)]:
   for b,values in centres.items():
    for k,v in values.items():
     if not np.isclose(v,expected[regime][cohort][b][k],atol=3e-14,rtol=2e-12):raise ValueError('Published centre differs')
  current['additional39_centres']=new39;current['additional39_order_changes']=new39['1']['D_minus_S']*new39['2']['D_minus_S']<0;result[regime]=current
 report=dict(status='PASS_CURRENT_PROGRAM_RESPONSE_NUMERICAL_REPLAY',groups=result,manifest_sha256=sha(ROOT/'MANIFEST.json'),scope='Original eight: recompute errors from saved task/program predictions and observations. Additional39: reaggregate saved donor errors with batch/capture balance. No training, selection, gene-to-program reconstruction or cell-level inference.',new_model_training=False)
 output.mkdir(parents=True,exist_ok=False);(output/'REPLAY.json').write_text(json.dumps(report,indent=2)+'\n');return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(replay(a.output),indent=2))
