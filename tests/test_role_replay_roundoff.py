"""Portable Pearson replay keeps classifications and other numerical fields exact."""
from pathlib import Path
import json
import runpy
import numpy as np
import pytest

CODE=Path(__file__).resolve().parents[1]/'evidence/table1/code/replay_bundle.py'
replay=runpy.run_path(str(CODE))
ATOL=replay['PEARSON_ATOL']

def test_pearson_ulp_change_is_recorded_and_larger_change_fails(tmp_path):
    expected=tmp_path/'expected';actual=tmp_path/'actual';expected.mkdir();actual.mkdir()
    base=[{'case_id':'case_1','fits':1,'action_verified':True,
           'before_scores':{'perturbation_centred_Pearson':0.5,'standardized_MSE':0.1}}]
    (expected/'actions.json').write_text(json.dumps(base))
    changed=json.loads(json.dumps(base));changed[0]['before_scores']['perturbation_centred_Pearson']=np.nextafter(0.5,1.0)
    (actual/'actions.json').write_text(json.dumps(changed))
    result=replay['compare_texts'](expected,actual,('actions.json',))['actions.json']
    assert not result['equal_bytes'] and result['bounded_score_differences']==1
    changed[0]['before_scores']['perturbation_centred_Pearson']=0.5+10*ATOL
    (actual/'actions.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='beyond'):replay['compare_texts'](expected,actual,('actions.json',))

@pytest.mark.parametrize('field,value',[('case_id','case_2'),('fits',2),('action_verified',False)])
def test_action_identity_count_and_decision_remain_exact(field,value):
    left=[{'case_id':'case_1','fits':1,'action_verified':True}]
    right=[dict(left[0],**{field:value})]
    with pytest.raises(ValueError,match='Exact action JSON'):replay['compare_action_json'](left,right,(),[])

def test_only_pearson_component_column_allows_roundoff(tmp_path):
    left=np.array([[0.1,0.2,0.5,1.0],[0.2,0.3,np.nan,0.5]])
    right=left.copy();right[0,2]=np.nextafter(0.5,1.0)
    a=tmp_path/'a.npz';b=tmp_path/'b.npz'
    np.savez(a,case__before__scores=left);np.savez(b,case__before__scores=right)
    result=replay['compare_components'](a,b)
    assert result['bounded_pearson_differences']==1 and result['non_pearson_values_exact']
    right[0,0]=np.nextafter(left[0,0],1.0);np.savez(b,case__before__scores=right)
    with pytest.raises(ValueError,match='Non-Pearson'):replay['compare_components'](a,b)
    right=left.copy();right[0,2]+=10*ATOL;np.savez(b,case__before__scores=right)
    with pytest.raises(ValueError,match='beyond'):replay['compare_components'](a,b)
    right=left.copy();right[1,2]=0.0;np.savez(b,case__before__scores=right)
    with pytest.raises(ValueError,match='support'):replay['compare_components'](a,b)

def test_action_tsv_uses_explicit_score_column_only(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir()
    text='case_id\tfits\tchange_scores\tchange_residual\ncase_1\t1\t0.5\t0.0\n'
    (a/'action_summary.tsv').write_text(text)
    (b/'action_summary.tsv').write_text(text.replace('0.5','0.5000000000000001'))
    assert replay['compare_texts'](a,b,('action_summary.tsv',))['action_summary.tsv']['bounded_score_differences']==1
    (b/'action_summary.tsv').write_text(text.replace('case_1','case_2'))
    with pytest.raises(ValueError,match='Exact action field'):replay['compare_texts'](a,b,('action_summary.tsv',))
