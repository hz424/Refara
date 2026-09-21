"""Operational dependencies and directly evaluated coordinate controls."""
import json
import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from reference_design import audit
from reference_design.audit_cli import main, run_demo


def contract(**changes):
    base=dict(task='reference_effect',output_kind='state',input_conditioned=True,
              model_input_id='B1',prediction_reference_id='B1',observation_reference_id='B2',
              storage_encoding='state',storage_baseline_id=None)
    return dict(base,**changes)


def test_operation_planner_separates_input_scoring_and_storage_changes():
    before=contract()
    input_change=audit.plan_change(before,dict(before,model_input_id='B2'))
    assert input_change['prediction_action']=='load_or_generate_for_new_input'
    assert input_change['requires_score_recompute']
    rescoring=audit.plan_change(before,dict(before,prediction_reference_id='B2'))
    assert rescoring['prediction_action']=='reuse' and rescoring['prediction_effect_action']=='recentre'
    storage=audit.plan_change(before,dict(before,storage_encoding='effect',storage_baseline_id='B3'))
    assert not storage['requires_score_recompute']
    assert storage['role_dependencies']['after']['storage_baseline']=='decode_export'


def test_native_effect_new_input_selects_new_effect_without_centring():
    before=contract(output_kind='native_effect',storage_encoding='effect')
    changed=audit.plan_change(before,dict(before,model_input_id='B2'))
    assert changed['prediction_effect_action']=='select_new_native_effect_without_centring'
    unused=audit.plan_change(before,dict(before,prediction_reference_id='B3'))
    assert unused['prediction_effect_action']=='preserve' and not unused['requires_score_recompute']


@pytest.mark.parametrize('changes',[{'model_inpt_id':'B2'},{'observation_reference_id':None}])
def test_misspelled_or_missing_active_roles_fail(changes):
    with pytest.raises(ValueError): audit.plan_change(contract(),dict(contract(),**changes))


def test_same_baseline_id_with_changed_values_is_rejected():
    c=contract(storage_encoding='effect',storage_baseline_id='B3')
    refs={'B3':np.array([1.,3.])}
    stored,payload=audit.encode_prediction(np.array([4.,6.]),c,refs)
    with pytest.raises(ValueError,match='baseline values'):
        audit.decode_prediction(stored,c,{'B3':np.array([2.,3.])},payload)


def test_equivalent_storage_and_common_reference_have_equal_residuals():
    refs={'B1':np.array([1.,3.]),'B2':np.array([2.,4.]),'B3':np.array([4.,2.])}
    state=np.array([5.,8.]); y=np.array([7.,9.]); scales=np.array([1.,2.])
    c=contract(prediction_reference_id='B1',observation_reference_id='B1')
    c2=dict(c,prediction_reference_id='B2',observation_reference_id='B2',storage_encoding='effect',storage_baseline_id='B3')
    stored,payload=audit.encode_prediction(state,c2,refs)
    decoded=audit.decode_prediction(stored,c2,refs,payload)
    first=audit.canonical_components(state,c,refs,y,scales)
    second=audit.canonical_components(decoded,c2,refs,y,scales)
    np.testing.assert_array_equal(first['residual'],state[None,:]-y)
    np.testing.assert_array_equal(first['residual'],second['residual'])


def test_actual_demo_and_cli_plan_are_distinct_and_do_not_overwrite(tmp_path):
    result=run_demo(tmp_path/'demo')
    assert result['status']=='PASS_EXECUTED_SYNTHETIC_OPERATION_AUDIT'
    assert result['planned_only'] and len(result['negative_controls'])==4
    assert result['incorrect_representation_mse_change']>0
    with pytest.raises(FileExistsError): run_demo(tmp_path/'demo')
    before=tmp_path/'before.json'; after=tmp_path/'after.json'
    before.write_text(json.dumps(contract()));after.write_text(json.dumps(contract(model_input_id='B2')))
    assert main(['plan',str(before),str(after),'--output',str(tmp_path/'plan')])==0
    assert json.loads((tmp_path/'plan/AUDIT.json').read_text())['executed'] is False


def test_installed_api_matches_all_95_historical_small_cache_cases():
    """Implementation parity, separate from the companion's scientific audit."""
    root=Path(__file__).parents[1]/'evidence/table1'
    source=root/'code/role_actions.py'
    if not source.exists():
        pytest.skip('Historical data companion is included in the full source release only')
    spec=importlib.util.spec_from_file_location('historical_role_actions_for_parity',source)
    old=importlib.util.module_from_spec(spec)
    previous=sys.dont_write_bytecode; sys.dont_write_bytecode=True
    try:
        spec.loader.exec_module(old)
    finally:
        sys.dont_write_bytecode=previous
    directory=root/'reader_demo'
    manifest=json.loads((directory/'manifest.json').read_text())
    payload=directory/manifest['data_file']
    assert hashlib.sha256(payload.read_bytes()).hexdigest()==manifest['data_sha256']
    with np.load(payload,allow_pickle=False) as stored:
        arrays={name: stored[name] for name in stored.files}
    cases=old._cases(manifest)
    assert len(cases)==95
    for case in cases:
        model=manifest['resources'][case['resource']]['models'][case['model_index']]
        env=old._environment(manifest,arrays,case['resource'])
        assert audit.plan_change(case['before'],case['after'])==old.plan_change(case['before'],case['after'])
        for contract_value in (case['before'],case['after']):
            native=old._native(arrays,model,contract_value)
            encoded,binding=audit.encode_prediction(native,contract_value,env['references'])
            decoded=audit.decode_prediction(encoded,contract_value,env['references'],binding)
            actual=audit.canonical_components(decoded,contract_value,**env)
            old_encoded,old_binding=old.encode_prediction(native,contract_value,env['references'])
            expected=old._components(old.decode_prediction(old_encoded,contract_value,env['references'],old_binding),contract_value,env)
            for component in expected:
                np.testing.assert_array_equal(actual[component],expected[component],err_msg=case['case_id']+' '+component)
