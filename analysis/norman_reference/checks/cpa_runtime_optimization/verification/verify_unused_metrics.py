#!/usr/bin/env python3
"""Prove that bypassing unused CPA batch R2 diagnostics preserves training."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import argparse, hashlib, importlib.util, json, random, time
import numpy as np
import pandas as pd
import anndata as ad
import torch
import pytorch_lightning as pl
import scvi.train
from cpa._module import CPAModule

parser=argparse.ArgumentParser()
parser.add_argument('--driver',required=True,type=Path)
parser.add_argument('--output',required=True,type=Path)
parser.add_argument('--gpu',action='store_true')
args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('norman_driver',args.driver)
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


def digest(value):
 h=hashlib.sha256()
 def update(v):
  if isinstance(v,torch.Tensor):
   update(v.detach().cpu().numpy())
  elif isinstance(v,np.ndarray):
   h.update(str(v.dtype).encode());h.update(str(v.shape).encode());h.update(v.tobytes())
  elif isinstance(v,dict):
   for key in sorted(v,key=str):
    update(key);update(v[key])
  elif isinstance(v,(list,tuple)):
   h.update(type(v).__name__.encode())
   for x in v:update(x)
  else:h.update(repr(v).encode())
 update(value)
 return h.hexdigest()


def rng_digest():
 return digest([torch.get_rng_state(),torch.cuda.get_rng_state_all() if args.gpu else [],np.random.get_state(),random.getstate()])

class Audit(pl.Callback):
 def __init__(self):
  self.backward=[];self.batches=[];self.epochs=[]
 def on_after_backward(self,trainer,plan):
  self.backward.append(digest({k:p.grad for k,p in plan.named_parameters() if p.grad is not None}))
 def on_train_batch_end(self,trainer,plan,outputs,batch,batch_idx):
  # Includes loss terms but excludes the deliberately uncomputed logging values.
  values={k:v for k,v in outputs.items() if not k.startswith('r2_')}
  self.batches.append(dict(losses=digest(values),optimizer=digest([o.state_dict() for o in trainer.optimizers]),rng=rng_digest()))
 def on_train_epoch_end(self,trainer,plan):
  self.epochs.append(digest(plan.state_dict()))

OriginalRunner=scvi.train.TrainRunner
current_audit=None
class AuditRunner(OriginalRunner):
 def __init__(self,*a,**kw):
  kw['callbacks']=list(kw.get('callbacks',[]))+[current_audit]
  super().__init__(*a,**kw)
scvi.train.TrainRunner=AuditRunner
original_metric=CPAModule.r2_metric

def skipped_metric(self,*a,**kw):
 return 0.,0.

conditions=['ctrl']+[f'A{i:02d}' for i in range(30)]+[f'A{i:02d}+A{(i+1)%30:02d}' for i in range(30)]
labels=np.repeat(conditions,16)
rng=np.random.default_rng(5723)
X=np.log1p(rng.poisson(1.5,(len(labels),2000))).astype('float32')
base=ad.AnnData(X,obs=pd.DataFrame({'condition':labels},index=[f'synthetic_{i}' for i in range(len(labels))]),var=pd.DataFrame(index=[f'g{i}' for i in range(2000)]))
results={}
for mode in ['original','skip_unused_r2']:
 CPAModule.r2_metric=original_metric if mode=='original' else skipped_metric
 current_audit=Audit()
 start=time.monotonic()
 model,record=driver.fit(base.copy(),'condition',args.output/mode,'synthetic',args.gpu,epochs=12,smoke=True)
 results[mode]=dict(seconds=time.monotonic()-start,trace=dict(backward=current_audit.backward,batches=current_audit.batches,epochs=current_audit.epochs),
  terminal_model=digest(model.module.state_dict()),terminal_plan=digest(model.trainer.lightning_module.state_dict()),
  terminal_optimizer=digest([o.state_dict() for o in model.trainer.optimizers]),terminal_rng=rng_digest())
 (args.output/f'{mode}_trace.json').write_text(json.dumps(results[mode],indent=2)+'\n')
 del model
 if args.gpu:torch.cuda.empty_cache()
CPAModule.r2_metric=original_metric
checks={key:results['original'][key]==results['skip_unused_r2'][key] for key in ['trace','terminal_model','terminal_plan','terminal_optimizer','terminal_rng']}
report=dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,
 original_seconds=results['original']['seconds'],optimized_seconds=results['skip_unused_r2']['seconds'],
 speedup=results['original']['seconds']/results['skip_unused_r2']['seconds'],
 data='synthetic only',epochs=12,cells=len(labels),genes=2000,conditions=len(conditions),
 backward_events=len(results['original']['trace']['backward']),batch_events=len(results['original']['trace']['batches']),
 gpu=args.gpu,driver_sha256=driver.sha256(args.driver),original_r2_source='CPA 0.8.8 CPAModule.r2_metric',
 optimization='Replace unused per-batch R2 logging calculation with zero sentinels, omitting those columns from published training history; reconstruction/adversarial diagnostics retained.')
(args.output/'equivalence_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
if report['status']!='PASS':raise SystemExit(1)
