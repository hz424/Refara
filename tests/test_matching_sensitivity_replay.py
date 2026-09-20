"""Numerical matching sensitivity, including changed inputs and exact ties."""
import copy
import importlib.util
import json
import shutil
from fractions import Fraction
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
CAPSULE=ROOT/'evidence/matching_sensitivity'
SPEC=importlib.util.spec_from_file_location('matching_sensitivity_replay',CAPSULE/'replay.py')
REPLAY=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


def actual_counts():
    return REPLAY.read_table(CAPSULE/'source/witness_counts.tsv',REPLAY.COUNTS_HEADER)


def synthetic_family(case,reverse=False):
    """A coherent strict order on eight models, observed on 39 matched units."""
    rows=[]
    for index,(source,target) in enumerate(REPLAY.PAIRS):
        higher=REPLAY.MODELS.index(source)>REPLAY.MODELS.index(target)
        if reverse:higher=not higher
        rows.append(dict(case_id=case,graph_id='G39',comparison_index=str(index),source=source,target=target,
            wins='39' if higher else '0',losses='0' if higher else '39',ties='0'))
    return rows


def utility_rows(matching,values):
    return [dict(matching_id=matching,graph_id='G12',cardinality='12',model=model,
        utility_sum_numerator=str(value.numerator),utility_sum_denominator=str(value.denominator))
        for model,value in zip(REPLAY.MODELS,values)]


def test_published_directional_conclusions_are_recomputed_from_counts():
    rows,sensitivity=REPLAY.replay_directions(actual_counts())
    assert len(rows)==560 and len(sensitivity)==112
    variable={(row['graph_id'],row['source'],row['target']) for row in sensitivity if row['matching_dependence_demonstrated']}
    assert variable=={('G39','RBF','TW'),('G39','RBF','PCA'),('G12','RBF','CM')}
    # Independently published exact fractions; no saved p-values are replay inputs.
    expected={
        ('G39','TW',28):Fraction(712182135,4294967296),
        ('G39','TW',30):Fraction(320195715,17179869184),
        ('G12','CM',11):Fraction(247,2048),
        ('G12','CM',12):Fraction(7,512),
    }
    for key,value in expected.items():
        hits=[row for row in rows if (row['graph_id'],row['target'],row['wins'])==key and row['source']=='RBF']
        assert hits
        assert all(Fraction(row['holm_p_numerator'],row['holm_p_denominator'])==value for row in hits)
        assert all(bool(row['reject'])==(value<=Fraction(1,20)) for row in hits)


def test_directions_respond_to_changed_witnesses_instead_of_fixed_verdicts():
    _,same=REPLAY.replay_directions(synthetic_family('A')+synthetic_family('B'))
    _,opposite=REPLAY.replay_directions(synthetic_family('A')+synthetic_family('B',reverse=True))
    assert sum(row['matching_dependence_demonstrated'] for row in same)==0
    assert sum(row['matching_dependence_demonstrated'] for row in opposite)==56


def test_complete_holm_family_and_tie_denominator():
    assert REPLAY.sign_tail(0,0)==1
    assert REPLAY.sign_tail(11,12)==Fraction(13,4096)
    assert REPLAY.sign_tail(11,11)==Fraction(1,2048)
    assert REPLAY.holm([Fraction(1,1024)]*56)==[Fraction(7,128)]*56
    assert REPLAY.holm([Fraction(1,2048)]*56)==[Fraction(7,256)]*56
    family=synthetic_family('all_tied')
    for row in family:row.update(wins='0',losses='0',ties='39')
    decisions,_=REPLAY.replay_directions(family)
    assert all(row['non_tied']==0 and row['holm_p']==1 and row['reject']==0 for row in decisions)
    with pytest.raises(ValueError,match='56-direction'):REPLAY.holm([Fraction(1,2048)]*55)


@pytest.mark.parametrize('corruption',['missing','duplicate','reverse','axis'])
def test_directional_replay_rejects_broken_matching_axes(corruption):
    rows=synthetic_family('A')
    if corruption=='missing':rows.pop()
    elif corruption=='duplicate':rows[-1]=dict(rows[0])
    elif corruption=='reverse':rows[0].update(wins='38',losses='1')
    else:rows[0]['source']='RBF'
    with pytest.raises(ValueError):REPLAY.replay_directions(rows)


def test_g12_leaders_recomputed_for_every_matching_and_model():
    rows=REPLAY.read_table(CAPSULE/'source/g12_matching_utility_sums.tsv',REPLAY.LEADER_HEADER)
    ranks,counts=REPLAY.replay_leaders(rows)
    assert len(ranks)==144*8
    assert {row['model']:row['leader_count'] for row in counts}==dict.fromkeys(REPLAY.MODELS,0)|{'NC':67,'RBF':77}
    assert len({row['matching_id'] for row in ranks})==144
    assert all(sum(row['is_leader'] for row in ranks if row['matching_id']==matching)==1 for matching in {r['matching_id'] for r in ranks})


def test_leader_changes_exact_ties_and_sub_float_differences():
    equal=[Fraction(-1)]*8
    _,tie_counts=REPLAY.replay_leaders(utility_rows('tie',equal))
    assert all(row['leader_count']==1 for row in tie_counts)
    first=[Fraction(1)]*8;first[0]+=Fraction(1,2**80)
    second=[Fraction(1)]*8;second[4]+=Fraction(1,2**80)
    # These two exact sums round to the same binary64 number; the exact leader still differs.
    assert float(first[0])==float(first[1])
    ranks,counts=REPLAY.replay_leaders(utility_rows('A',first)+utility_rows('B',second))
    assert {row['model']:row['leader_count'] for row in counts if row['leader_count']}=={'NC':1,'RBF':1}
    assert [row['model'] for row in ranks if row['matching_id']=='A' and row['rank_min']==1]==['NC']


@pytest.mark.parametrize('corruption',['missing','duplicate','zero_denominator','nan','fractional_numerator'])
def test_leader_replay_rejects_incomplete_or_invalid_aggregates(corruption):
    rows=utility_rows('A',[Fraction(1)]*8)
    if corruption=='missing':rows.pop()
    elif corruption=='duplicate':rows.append(dict(rows[0]))
    elif corruption=='zero_denominator':rows[0]['utility_sum_denominator']='0'
    elif corruption=='nan':rows[0]['utility_sum_numerator']='nan'
    else:rows[0]['utility_sum_numerator']='1.5'
    with pytest.raises(ValueError):REPLAY.replay_leaders(rows)


def test_bound_inputs_and_result_scope(tmp_path):
    result=REPLAY.run(tmp_path/'results')
    assert result['g12_multiple_models_can_lead']
    assert result['starting_point']=='Per-witness sign counts and per-matching model utility sums'
    assert not result['maximum_matching_graph_recomputed']
    assert not result['all_graph_directional_certificates_recomputed']
    assert json.loads((tmp_path/'results/SUMMARY.json').read_text())==result
    with pytest.raises(ValueError,match='new output directory'):REPLAY.run(tmp_path/'results')
    shutil.copytree(CAPSULE/'source',tmp_path/'changed_source')
    target=tmp_path/'changed_source/witness_counts.tsv'
    target.write_text(target.read_text()+'\n')
    with pytest.raises(ValueError,match='Changed bound aggregate input'):
        REPLAY.run(tmp_path/'rejected_results',tmp_path/'changed_source')
    assert not (tmp_path/'rejected_results').exists()
