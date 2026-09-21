"""Generation records must constrain actual scorer inputs, without self-certification."""
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from reference_design.input_binding import SCHEMA, verify_generation_record
from reference_design.protocol import run_plan


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bound_record(root, *, prediction, controls, membership, references, model, keys):
    checkpoint = root/'checkpoint.bin'; checkpoint.write_bytes(b'synthetic checkpoint')
    code = root/'generate.py'; code.write_text('# synthetic generator identity\n')
    def pin(path):
        return None if path is None else dict(path=path.name, sha256=digest(path))
    record = dict(schema=SCHEMA, model=model,
                  artifacts={name: pin(path) for name, path in dict(prediction=prediction, controls=controls,
                             membership=membership, references=references, baseline=None).items()},
                  input_keys=keys, checkpoint=pin(checkpoint), code=[pin(code)])
    path=root/'generation.json'; path.write_text(json.dumps(record))
    args=dict(prediction_path=prediction, model_name=model['name'], kind=model['kind'],
              conditioning=model['conditioning'], representation=model['representation'],
              controls_path=controls, membership_path=membership, references_path=references,
              expected_input_keys=keys)
    return path, record, args


@pytest.fixture
def bound(tmp_path):
    paths={}
    for name in ('prediction', 'controls', 'membership', 'references'):
        paths[name]=tmp_path/(name+'.tsv'); paths[name].write_text(name+' original\n')
    return bound_record(tmp_path, **paths,
                        model=dict(name='M',kind='state',conditioning='block',representation='native'),
                        keys=[['a',1,'B1'],['a',1,'B2']])


def test_no_record_is_declaration_only(bound):
    _,_,args=bound
    result=verify_generation_record(None, **args)
    assert result==dict(level='declaration',verified_input_use=False,artifacts={})


def test_complete_record_binds_files_without_attesting_generation(bound):
    path,_,args=bound
    result=verify_generation_record(path, **args)
    assert result['level']=='generation_record_binding'
    assert result['verified_input_use'] is False
    assert set(result['artifacts'])=={'prediction','controls','membership','references','checkpoint','code:0','record'}
    assert result['artifacts']['prediction']['sha256']==digest(args['prediction_path'])


@pytest.mark.parametrize('role', ['prediction','controls','membership','references','checkpoint','code'])
def test_every_stale_generation_artifact_fails(bound, role):
    path,record,args=bound
    spec=record['artifacts'][role] if role in record['artifacts'] else record[role][0] if role=='code' else record[role]
    (path.parent/spec['path']).write_bytes(b'changed after record')
    with pytest.raises(ValueError,match='Stale'):
        verify_generation_record(path,**args)


@pytest.mark.parametrize('change', ['keys','model','selected_path','self_attestation','missing_membership'])
def test_record_cannot_change_runtime_identity_or_self_attest(bound,change):
    path,record,args=bound
    if change=='keys': record['input_keys']=record['input_keys'][:1]
    elif change=='model': record['model']['name']='different'
    elif change=='self_attestation': record['level']='independently_recomputed'
    elif change=='missing_membership': args['membership_path']=None
    else:
        replacement=path.parent/'other.tsv'; replacement.write_bytes(args['prediction_path'].read_bytes())
        args['prediction_path']=replacement
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError): verify_generation_record(path,**args)


def test_only_executed_numeric_callback_earns_recomputed_level(bound):
    path,_,args=bound; calls=[]
    # Independently evaluate the simple synthetic rule from supplied controls.
    inputs=np.array([[1.,2.],[3.,4.]])
    expected=inputs*2+1
    def recompute(record):
        calls.append(record['model']['name'])
        return np.add(np.multiply(inputs,2),1)
    result=verify_generation_record(path,**args,recompute=recompute,expected_predictions=expected)
    assert calls==['M'] and result['level']=='independently_recomputed'
    assert result['verified_input_use'] is False
    for callback in (lambda _: {'status':'PASS'}, lambda _: True, lambda _: expected+1):
        with pytest.raises(ValueError):
            verify_generation_record(path,**args,recompute=callback,expected_predictions=expected)


def test_callback_cannot_modify_bound_files(bound):
    path,_,args=bound
    def mutate(_):
        args['controls_path'].write_text('changed during callback')
        return np.array([1.])
    with pytest.raises(ValueError,match='changed during'):
        verify_generation_record(path,**args,recompute=mutate,expected_predictions=np.array([1.]))


def test_actual_protocol_consumes_binding_and_rejects_stale_record(tmp_path):
    script=Path(__file__).parents[1]/'examples/reference_protocol/make_example.py'
    spec=importlib.util.spec_from_file_location('binding_protocol_example',script)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    plan_path=module.make_example(tmp_path/'inputs')
    plan=json.loads(plan_path.read_text()); plan['tasks']=plan['tasks'][:1]
    task=plan['tasks'][0]; model=next(m for m in task['models'] if m.get('conditioning')=='block')
    root=plan_path.parent
    with (root/task['references']).open() as f:
        keys=[[r['allocation'],int(r['depth']),r['block']] for r in csv.DictReader(f,delimiter='\t')]
    record,_,_=bound_record(root,prediction=root/model['prediction'],controls=root/task['controls'],
                           membership=root/task['membership'],references=root/task['references'],
                           model=dict(name=model['name'],kind=model['kind'],conditioning='block',representation='native'),keys=keys)
    model['generation_record']=record.name; plan_path.write_text(json.dumps(plan))
    receipt=run_plan(plan_path,tmp_path/'valid')
    evidence=next(x['evidence'] for x in receipt['tasks'][0]['prediction_evidence'] if x['model']==model['name'])
    assert evidence['level']=='generation_record_binding'
    # Valid prediction syntax, but different values: ordinary score parsing would succeed.
    prediction=root/model['prediction']; prediction.write_text(prediction.read_text().replace('\t3\t4','\t9\t4',1))
    # Ensure mutation even if the example's first numeric row changes in future.
    if digest(prediction)==evidence['artifacts']['prediction']['sha256']:
        prediction.write_text(prediction.read_text()+'\n')
    with pytest.raises(ValueError,match='Stale'):
        run_plan(plan_path,tmp_path/'invalid')
    assert not (tmp_path/'invalid').exists()
