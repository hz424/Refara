"""Independent examples separating output kind, input dependence and theorem scope."""
from itertools import product

import numpy as np
import pytest

from reference_design import Prediction, score_references


def test_input_conditioned_effect_scores_all_roles_without_prediction_centring():
    controls = np.array([[0.0], [1.0], [2.0]])
    result = score_references([0.0], controls,
                              [Prediction('effect', controls, 'effect'), Prediction('state', [0.0], 'state')])
    # Independent native-effect residual: delta(input) - treated + observation.
    expected = np.array([(float(controls[m, 0]) + float(controls[o, 0]))**2
                         for o, p, m in product(range(3), repeat=3)])
    np.testing.assert_array_equal(result.atomic_mse[0], expected)
    lookup = dict(zip(result.role_tuples, result.atomic_mse[0]))
    for observation, model_input in product(range(3), repeat=2):
        assert len({lookup[observation, pred, model_input] for pred in range(3)}) == 1
    assert lookup[0, 0, 0] != lookup[0, 0, 2]
    pair = result.pairwise[0]
    assert pair['d_S'] == pytest.approx(-20/3)
    assert pair['d_D'] == pytest.approx(-8/3)
    assert pair['V'] == pytest.approx(2)
    assert pair['d_D'] - pair['d_S'] == pytest.approx(4)  # Not the fixed-effect V identity.
    assert not pair['identity_applicable']
    for key in ('predicted_d_D', 'identity_residual', 'identity_verified',
                'identity_tolerance', 'condition_strict', 'expected_crossing'):
        assert pair[key] is None
    assert pair['theory'] == 'not_applicable_input_conditioned_effect'
    assert not result.checks['all_identities_verified']
    assert result.checks['all_applicable_identities_verified']
    assert result.checks['applicable_identity_count'] == 0
    assert result.checks['inapplicable_identity_count'] == 1
    assert result.checks['identity_failures'] == 0


@pytest.mark.parametrize('coefficient,state', [(1.0, 2.0), (-2.0, 1.5)])
def test_both_reversal_directions_are_descriptive_for_conditioned_effects(coefficient, state):
    controls = np.array([[0.0], [1.0], [2.0]])
    pair = score_references([0.0], controls,
            [Prediction('effect', coefficient*controls, 'effect'), Prediction('state', [state], 'state')]).pairwise[0]
    assert pair['d_S'] * pair['d_D'] < 0
    assert pair['observed_crossing']
    assert pair['classification'] == 'ranking_reversal'
    assert pair['expected_crossing'] is None


def test_theorem_scope_is_pair_specific_and_unchanged_pairs_keep_old_values():
    controls = np.array([[0.0], [1.0], [2.0]])
    old_models = [Prediction('fixed effect', [0.0], 'effect'),
                  Prediction('state a', controls*1.5, 'state'), Prediction('state b', [0.2], 'state')]
    old = score_references([0.5], controls, old_models)
    new = score_references([0.5], controls, old_models + [Prediction('conditioned effect', controls, 'effect')])
    np.testing.assert_array_equal(new.atomic_mse[:3], old.atomic_mse)
    np.testing.assert_array_equal(new.mse[:3], old.mse)
    old_pairs = {(p['model_a'],p['model_b']):p for p in old.pairwise}
    for pair in new.pairwise:
        key = pair['model_a'],pair['model_b']
        if key in old_pairs:
            assert pair == old_pairs[key]
        else:
            assert not pair['identity_applicable']
    assert new.checks['applicable_identity_count'] == 3
    assert new.checks['inapplicable_identity_count'] == 3
    assert new.checks['identity_failures'] == 0


def test_two_conditioned_effects_do_not_claim_zero_displacement():
    controls = np.array([[0.0], [1.0], [2.0]])
    pair = score_references([0.0], controls,
            [Prediction('a', controls, 'effect'),Prediction('b', -controls, 'effect')]).pairwise[0]
    assert pair['d_S'] != pytest.approx(pair['d_D'])
    assert not pair['identity_applicable']
    assert pair['identity_verified'] is None


def test_conditioned_effect_block_gene_and_expression_unit_invariants():
    controls = np.array([[0., 1.], [1., -1.], [2., 0.]])
    effect = 2*controls + np.array([0.2, -0.4])
    treated, scales = np.array([1., 2.]), np.array([0.5, 2.])
    original = score_references(treated, controls,
            [Prediction('effect', effect, 'effect'),Prediction('state', controls, 'state')], scales)
    blocks, genes = [2,0,1], [1,0]
    reordered = score_references(treated[genes],controls[blocks][:,genes],
            [Prediction('effect',effect[blocks][:,genes],'effect'),
             Prediction('state',controls[blocks][:,genes],'state')],scales[genes])
    np.testing.assert_allclose(reordered.mse, original.mse)
    units, location = np.array([3., 0.5]), np.array([7., -4.])
    transformed = score_references(treated*units+location, controls*units+location,
            [Prediction('effect',effect*units,'effect'),
             Prediction('state',controls*units+location,'state')],scales*units)
    np.testing.assert_allclose(transformed.atomic_mse, original.atomic_mse)
    assert transformed.V == pytest.approx(original.V)
