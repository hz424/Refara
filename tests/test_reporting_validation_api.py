"""Hand-calculated reporting failures and exact kernel parity; no real outcomes."""
from copy import deepcopy
import numpy as np
import pytest

from reference_design.reporting_validation import (RULES, publication_candidates, common_coverage,
    working_assessment_intervals, finite_panel_risk_bounds, paired_rule_difference_bounds)
from reference_design.reporting_validation import _retrospective as k
from reference_design.reporting_validation.replay import compact_evaluate

CONTRACT = dict(design="antithetic_O_half", seed_aggregation="mean_after_scoring",
                basis="invented complementary input/observation control partitions")
ASSUMPTIONS = dict(unit_definition="fictional independent cultures", conditioning_scope="fixed fictional panel",
                   working_independence_basis="independent synthetic offsets; not actual admission")


def example():
    values = np.broadcast_to([4., 3., 2., 0.], (4, 4, 4)).copy()
    values[:, ::2, :3] -= .25
    values[:, 1::2, :3] += .25
    return publication_candidates(values, unit_ids=["u0", "u1", "u2", "u3"],
        comparison_ids=["c0", "c1", "c2", "c3"], calibration_penalty=[.5]*4,
        allocation_contract=CONTRACT, bootstrap_resamples=19, seed=100)


def test_hand_derived_five_scores_and_first_partition():
    result = example()
    rows = {(r['rule_id'], r['pair_id']): r for r in result['candidates']}
    assert [rows[r, 'c0']['reliability_score'] for r in RULES] == [3.75, 3.75, 4., 4., 3.25]
    assert result['companions'][0]['estimate'] == 4.
    assert all(not rows[r, 'c3']['eligible'] for r in RULES)
    plan = common_coverage(result['candidates'], comparison_ids=['c0','c1','c2','c3'])
    assert plan['common_achievable_M'] == [0,1,2,3]
    assert plan['primary_M'] == 2 and plan['primary_status'] == 'not_evaluable'


def test_reversal_unresolved_hold_and_shared_sign_risk_difference():
    records = {'c0': {'interval':[1.,2.], 'status':'assessable'},
               'c1': {'interval':[-2.,-1.], 'status':'assessable'},
               'c2': {'interval':[0.,1.], 'status':'assessable'},
               'c3': {'interval':None, 'status':'unassessable','reason':'missing'}}
    directions = dict(c0='a',c1='a',c2='a',c3='hold')
    result = finite_panel_risk_bounds(list(records), directions, records)
    assert result['counts']['supported'] == result['counts']['reversed'] == result['counts']['unresolved'] == result['counts']['held'] == 1
    assert result['reversal_fraction'] == result['unresolved_fraction'] == 1/3
    assert result['true_reversal_fraction_bounds'] == [1/3,1/3]
    other = dict(directions,c1='b')
    difference = paired_rule_difference_bounds(list(records), directions, other, records)
    assert difference['true_reversal_fraction_difference_bounds'] == [1/3,1/3]


def test_natural_coverage_survives_refara_all_hold_and_whole_ties():
    result = example()
    for row in result['candidates']:
        if row['rule_id'] == 'existing_refara_reporting':
            row.update(eligible=False,reliability_score=-1.,reason='score_below_floor')
    plan = common_coverage(result['candidates'], comparison_ids=['c0','c1','c2','c3'])
    assert plan['common_achievable_M'] == [0] and plan['primary_status'] == 'not_evaluable'
    assert [r['published_M'] for r in plan['natural_sets']] == [3,3,3,3,0]
    for row in result['candidates']:
        if row['pair_id'] in ('c0','c1') and row['rule_id'] != 'existing_refara_reporting':
            row['reliability_score'] = 5.
    plan = common_coverage(result['candidates'], comparison_ids=['c0','c1','c2','c3'])
    assert 1 not in plan['attainable_counts_by_rule']['cross_fitting']


def test_generic_candidates_match_executed_63_comparison_kernel():
    rng = np.random.default_rng(21)
    rows = [dict(native_donor=d,context='invented',condition=c) for d in ['u0','u1','u2','u3'] for c in k.CONDITIONS]
    losses = rng.uniform(.1,2.,size=(12,64,7))
    penalty = rng.uniform(0,.1,21)
    candidates, companions, receipt = k.candidates_for_context(rows,losses,'invented',penalty,bootstrap_count=19,seed=31)
    donors,margins = k.context_matrix(rows,losses,'invented')
    result = publication_candidates(margins,unit_ids=donors,comparison_ids=[r['pair_id'] for r in k.roster(['invented'])],
        calibration_penalty=np.tile(penalty,3),allocation_contract=CONTRACT,bootstrap_resamples=19,seed=31)
    assert result['candidates'] == candidates and result['companions'] == companions
    assert result['bootstrap_schedule_sha256'] == receipt['schedule_sha256']


def test_interval_declared_units_missing_and_constant_retain_full_N():
    x=np.array([[2+v,-3+v,np.nan if j==7 else v,0.] for j,v in enumerate(np.linspace(-.35,.35,8))])
    result=working_assessment_intervals(x,unit_ids=[f'u{j}' for j in range(8)],comparison_ids=['p','m','miss','const'],
        assumptions=ASSUMPTIONS,joint=False)
    assert result['multiplicity_N']==4
    assert result['records']['p']['interval'][0]>0 and result['records']['m']['interval'][1]<0
    assert result['records']['miss']['status']==result['records']['const']['status']=='unassessable'
    assert not result['scientific_confirmation'] and not result['interval_calibration_performed']
    with pytest.raises(ValueError,match='declarations'):
        working_assessment_intervals(x,unit_ids=[f'u{j}' for j in range(8)],comparison_ids=['p','m','miss','const'],assumptions={})
    with pytest.raises(ValueError,match='Multiplicity'):
        working_assessment_intervals(x,unit_ids=[f'u{j}' for j in range(8)],comparison_ids=['p','m','miss','const'],assumptions=ASSUMPTIONS,multiplicity=3)


def test_interval_zero_se_infinite_bootstrap_has_no_t_only_fallback():
    result=working_assessment_intervals([[0.],[1.]],unit_ids=['u0','u1'],comparison_ids=['c'],assumptions=ASSUMPTIONS,resamples=999,seed=3)
    assert result['bootstrap_critical'] is None and result['infinite_bootstrap_maxima']>0
    assert result['records']['c']['t_interval'] is not None
    assert result['records']['c']['interval'] is None


def test_unit_reordering_preserves_scores_and_intervals():
    x=np.random.default_rng(2).normal(size=(8,4,3))
    args=dict(comparison_ids=['a','b','c'],calibration_penalty=[0.,.1,.2],allocation_contract=CONTRACT,bootstrap_resamples=39)
    ids=[f'u{i}' for i in range(8)]
    assert publication_candidates(x,unit_ids=ids,**args)==publication_candidates(x[::-1],unit_ids=ids[::-1],**args)
    a=working_assessment_intervals(x.mean(1),unit_ids=ids,comparison_ids=['a','b','c'],assumptions=ASSUMPTIONS,resamples=99)
    b=working_assessment_intervals(x[::-1].mean(1),unit_ids=ids[::-1],comparison_ids=['a','b','c'],assumptions=ASSUMPTIONS,resamples=99)
    assert a==b


def test_bad_types_lossy_integers_and_duplicate_rosters_rejected():
    args=dict(unit_ids=['u0','u1'],comparison_ids=['a'],calibration_penalty=[0.],allocation_contract=CONTRACT)
    with pytest.raises(ValueError,match='real numeric'):
        publication_candidates(np.ones((2,2,1),dtype=bool),**args)
    with pytest.raises(ValueError,match='represented exactly'):
        publication_candidates(np.full((2,2,1),2**53+1,dtype=np.int64),**args)
    rows=example()['candidates'];rows[1]=deepcopy(rows[0])
    with pytest.raises(ValueError,match='Duplicate/missing'):
        common_coverage(rows,comparison_ids=['c0','c1','c2','c3'])


def test_nonzero_extended_precision_margins_cannot_silently_become_zero():
    if np.finfo(np.longdouble).tiny >= np.finfo(np.float64).tiny:
        pytest.skip('Platform long double has no wider exponent range')
    values=np.array([[np.longdouble('1e-400')],[np.longdouble('2e-400')]])
    assert (values != 0).all()
    with pytest.raises(ValueError,match='underflows'):
        working_assessment_intervals(values,unit_ids=['u0','u1'],comparison_ids=['c'],assumptions=ASSUMPTIONS)
    with pytest.raises(ValueError,match='underflows'):
        publication_candidates(np.repeat(values[:,None,:],2,axis=1),unit_ids=['u0','u1'],
            comparison_ids=['c'],calibration_penalty=[0.],allocation_contract=CONTRACT)
