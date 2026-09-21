"""Compact claim replay and guards against wrong aggregation or incomplete axes."""
import importlib.util
from pathlib import Path
import pytest
P=Path(__file__).resolve().parents[1]/'evidence/program_responses/replay.py'
spec=importlib.util.spec_from_file_location('program_replay',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_current_program_claim_from_task_scores(tmp_path):
 r=m.replay(tmp_path/'replay')
 assert [r['groups'][k]['direction_discordant'] for k in ['original8','cap48']]==[75,65]
 for group in r['groups'].values():
  assert group['order_changes'] and group['additional39_order_changes']
  assert group['max_error_of_mean_vs_mean_error_gap']>0.001
  assert group['observation_shift_identity_max_abs']<1e-13

def test_balanced_aggregation_is_not_equal_donor_mean():
 rows=[]
 for b in range(8):
  rows.extend([{'batch_id':str(b),'capture_orientation':'A','value':'0'}])
  rows.extend([{'batch_id':str(b),'capture_orientation':'B','value':'2'}]*3)
 assert m.balanced(rows,'value')==1.0
 assert sum(float(r['value']) for r in rows)/len(rows)==1.5

def test_missing_orientation_is_rejected():
 with pytest.raises(ValueError,match='two capture orientations'):
  m.balanced([{'batch_id':str(b),'capture_orientation':'A','value':'0'} for b in range(8)],'value')

def test_missing_task_program_axes_are_rejected():
 with pytest.raises(ValueError,match='axes differ'):m.summarize_task_rows([])


SELECTION=P.with_name('selection_replay.py')
selection_spec=importlib.util.spec_from_file_location('program_selection_replay',SELECTION)
selection=importlib.util.module_from_spec(selection_spec)
selection_spec.loader.exec_module(selection)


def test_selection_capsule_runs_outside_repository(tmp_path):
 import json,shutil,subprocess,sys
 bundle=tmp_path/'portable'
 (bundle/'inputs').mkdir(parents=True)
 for name in ['selection_replay.py','selection_inputs.json','selection_expected.json','inputs/selection_root_utilities.tsv']:
  shutil.copyfile(SELECTION.parent/name,bundle/name)
 output=tmp_path/'results'
 completed=subprocess.run([sys.executable,'-I',str(bundle/'selection_replay.py'),'--output',str(output)],cwd=tmp_path,capture_output=True,text=True)
 assert completed.returncode==0,completed.stderr
 report=json.loads((output/'selection_results.json').read_text())
 assert report['counts']==dict(choices=54,selection_cases=18,S_O_changed=18,O_D_same=18)
 assert report['maximum_absolute_error_against_verified_roster']<2e-14
 assert len((output/'selection_roster.tsv').read_text().splitlines())==55
 assert len((output/'selection_family_utilities.tsv').read_text().splitlines())==433


def test_selection_replay_rejects_tampered_utilities_before_output(tmp_path):
 import shutil
 bundle=tmp_path/'bundle'
 (bundle/'inputs').mkdir(parents=True)
 for name in ['selection_inputs.json','selection_expected.json','inputs/selection_root_utilities.tsv']:
  shutil.copyfile(SELECTION.parent/name,bundle/name)
 table=bundle/'inputs/selection_root_utilities.tsv'
 table.write_text(table.read_text()+'\n')
 output=tmp_path/'results'
 with pytest.raises(ValueError,match='hash differs'):
  selection.replay(output,bundle)
 assert not output.exists()


def test_selection_uses_candidate_values_and_excludes_held_out_root():
 import json
 identities=json.loads((SELECTION.parent/'selection_inputs.json').read_text())['model_identities']
 rows=[dict(regime=r,policy=p,root_index=root,family=f,utility=100. if root==7 and f=='TW' else 1. if f=='scGen' else 0.)
       for r in selection.REGIMES for p in selection.POLICIES for root in range(8) for f in selection.FAMILIES]
 result=selection.calculate(rows,identities)
 choices={r['held_out_root_index']:r for r in result['choices'] if r['regime']=='original8' and r['policy']=='O'}
 assert choices[7]['family']=='scGen'
 assert choices[0]['family']==choices[None]['family']=='TW'
 assert 7 not in choices[7]['selection_root_indices']
 with pytest.raises(ValueError,match='Incomplete'):
  selection.calculate(rows[:-1],identities)
 with pytest.raises(ValueError,match='Duplicate'):
  selection.calculate(rows+[rows[0]],identities)


def test_selection_preserves_family_order_ties_and_all_tied_gap():
 scores={f:0. for f in selection.FAMILIES}
 all_tied=selection.choose(scores)
 assert all_tied['family']=='NC' and all_tied['maximizers']==selection.FAMILIES
 assert all_tied['gap_to_best_outside_tie'] is None
 scores['NC']=1.;scores['CM']=1.+.5e-12
 near_tie=selection.choose(scores)
 assert near_tie['family']=='NC' and near_tie['maximizers']==['NC','CM']
 scores['CM']=1.+2e-12
 assert selection.choose(scores)['family']=='CM'
 scores['NC']=float('nan')
 with pytest.raises(ValueError,match='finite'):
  selection.choose(scores)
