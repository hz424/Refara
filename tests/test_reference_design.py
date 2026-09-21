from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from reference_design import (
    Prediction,
    allocate_controls,
    effect_to_state,
    score_references,
    state_to_effect,
)


def _oracle(treated, controls, predictions, scales):
    """Calculate each residual before pooling the balanced role assignments."""
    groups = {key: [] for key in ("S", "M", "P", "O", "D")}
    for obs, pred, model in product(range(3), repeat=3):
        if obs == pred == model:
            key = "S"
        elif obs == pred:
            key = "M"
        elif obs == model:
            key = "P"
        elif pred == model:
            key = "O"
        else:
            key = "D"
        losses = []
        for prediction in predictions:
            values = np.asarray(prediction.values)
            if prediction.kind == "state":
                state = values if values.ndim == 1 else values[model]
                effect = state - controls[pred]
            else:
                effect = values if values.ndim == 1 else values[model]
            residual = (effect - (treated - controls[obs])) / scales
            losses.append(np.dot(residual, residual) / len(scales))
        groups[key].append(losses)
    return {key: np.mean(losses, axis=0) for key, losses in groups.items()}


def _simple(effect=0.0, state=0.0, **kwargs):
    return score_references(
        np.zeros(3),
        np.array([[-1.0] * 3, [0.0] * 3, [1.0] * 3]),
        [
            Prediction("effect", np.broadcast_to(effect, (3,)).copy(), "effect"),
            Prediction("state", np.broadcast_to(state, (3,)).copy(), "state"),
        ],
        **kwargs,
    )


def test_conditioned_predictions_match_independent_role_enumeration():
    treated = np.array([1.0, -2.0, 0.5])
    controls = np.array([[0.0, 1.0, -3.0], [2.0, -1.0, 1.0], [3.0, 2.0, 0.0]])
    scales = np.array([0.5, 2.0, 3.0])
    predictions = [
        Prediction("fixed effect", np.array([0.3, -0.4, 0.7]), "effect"),
        Prediction("fixed state", np.array([0.2, 1.3, -0.5]), "state"),
        Prediction("conditioned state", controls * 1.7 + 0.4, "state"),
    ]
    expected = _oracle(treated, controls, predictions, scales)
    result = score_references(treated, controls, predictions, scales)
    for index, pattern in enumerate(result.patterns):
        np.testing.assert_allclose(result.mse[:, index], expected[pattern], atol=1e-13)
    assert result.atomic_mse.shape == (3, 27)
    assert not np.isclose(expected["P"][2], expected["O"][2])
    np.testing.assert_allclose(expected["M"], expected["S"], atol=1e-13)
    np.testing.assert_allclose(expected["D"], (expected["P"] + expected["O"]) / 2)
    independent_v = 3 * np.mean(((controls - controls.mean(axis=0)) / scales) ** 2)
    assert result.V == pytest.approx(independent_v)
    for pair in result.pairwise:
        assert pair["identity_verified"]
        assert pair["identity_residual"] == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize(
    ("effect", "state", "expected_s", "expected_d", "crosses"),
    [
        (0.0, 0.0, -2 / 3, 4 / 3, True),
        (2.0, 0.0, -14 / 3, -8 / 3, False),
        (0.0, 2.0, 10 / 3, 16 / 3, False),
    ],
)
def test_exact_reversal_and_both_stable_outcomes(effect, state, expected_s, expected_d, crosses):
    result = _simple(effect, state)
    pair = result.pairwise[0]
    assert pair["model_a"] == "effect"
    assert pair["model_b"] == "state"
    assert pair["d_S"] == pytest.approx(expected_s)
    assert pair["V"] == pytest.approx(2.0)
    assert pair["d_D"] == pytest.approx(expected_d)
    assert bool(pair["condition_strict"]) is crosses
    assert bool(pair["expected_crossing"]) is crosses
    assert bool(pair["observed_crossing"]) is crosses


@pytest.mark.parametrize(
    ("effect", "state", "boundary"),
    [(0.0, np.array([1.0, 1.0, 0.0]), "S_tie"), (np.array([2.0, 0.0, 0.0]), 0.0, "D_tie")],
)
def test_boundary_ties_do_not_count_as_reversals(effect, state, boundary):
    pair = _simple(effect, state).pairwise[0]
    assert pair[boundary]
    assert not pair["observed_crossing"]
    assert not pair["expected_crossing"]


def test_tolerance_changes_classification_without_changing_scores():
    precise = _simple().pairwise[0]
    coarse = _simple(tie_tolerance=1.0).pairwise[0]
    for key in ("d_S", "d_D", "V", "predicted_d_D"):
        assert coarse[key] == precise[key]
    assert precise["observed_crossing"]
    assert coarse["S_tie"]
    assert not coarse["observed_crossing"]


def test_identical_control_means_have_no_reference_penalty():
    result = score_references(
        np.array([2.0, -1.0]),
        np.ones((3, 2)),
        [Prediction("effect", np.zeros(2), "effect"), Prediction("state", np.ones(2), "state")],
    )
    assert result.V == 0.0
    for row in result.mse:
        np.testing.assert_array_equal(row, np.repeat(row[0], 5))
    pair = result.pairwise[0]
    assert pair["S_tie"] and pair["D_tie"]
    assert not pair["observed_crossing"]


@pytest.mark.parametrize("kind", ["effect", "state"])
def test_same_output_kind_has_zero_margin_displacement(kind):
    controls = np.array([[-2.0, 1.0], [0.0, 0.0], [3.0, -1.0]])
    first = np.array([0.2, -0.7]) if kind == "effect" else controls * 2.0
    second = np.array([0.9, 0.2]) if kind == "effect" else controls * -0.5 + 0.3
    result = score_references(
        np.array([1.0, -1.0]), controls,
        [Prediction("a", first, kind), Prediction("b", second, kind)],
    )
    pair = result.pairwise[0]
    assert pair["d_D"] == pytest.approx(pair["d_S"], abs=1e-12)
    assert pair["condition_strict"] is None
    assert pair["theory"] == "zero_displacement"
    assert not pair["observed_crossing"]


def test_mixed_pair_orientation_does_not_depend_on_input_model_order():
    predictions = [Prediction("z effect", np.zeros(3), "effect"), Prediction("a state", np.zeros(3), "state")]
    controls = np.array([[-1.0] * 3, [0.0] * 3, [1.0] * 3])
    forward = score_references(np.zeros(3), controls, predictions).pairwise[0]
    reverse = score_references(np.zeros(3), controls, list(reversed(predictions))).pairwise[0]
    assert forward == reverse
    assert forward["model_a"] == "z effect"


def test_block_and_gene_permutations_preserve_scores():
    rng = np.random.default_rng(14)
    treated, effect = rng.normal(size=(2, 7))
    controls, states = rng.normal(size=(2, 3, 7))
    scales = np.linspace(0.3, 2.4, 7)
    original = score_references(
        treated, controls,
        [Prediction("effect", effect, "effect"), Prediction("state", states, "state")], scales,
    )
    blocks = np.array([2, 0, 1])
    genes = np.array([6, 0, 4, 1, 5, 2, 3])
    permuted = score_references(
        treated[genes], controls[blocks][:, genes],
        [Prediction("effect", effect[genes], "effect"), Prediction("state", states[blocks][:, genes], "state")],
        scales[genes],
    )
    np.testing.assert_allclose(original.mse, permuted.mse, atol=1e-13)
    assert original.V == pytest.approx(permuted.V)


def test_expression_units_and_location_do_not_change_standardized_scores():
    treated = np.array([1.0, 3.0])
    controls = np.array([[0.0, 1.0], [0.5, 0.0], [1.0, -1.0]])
    effect, state = np.array([0.2, 0.3]), controls * 0.5
    scales = np.array([0.2, 2.0])
    unit = np.array([3.0, 0.5])
    shift = np.array([7.0, -4.0])
    original = score_references(treated, controls, [Prediction("e", effect, "effect"), Prediction("s", state, "state")], scales)
    transformed = score_references(
        treated * unit + shift, controls * unit + shift,
        [Prediction("e", effect * unit, "effect"), Prediction("s", state * unit + shift, "state")], scales * unit,
    )
    np.testing.assert_allclose(original.mse, transformed.mse, atol=1e-12)
    assert transformed.V == pytest.approx(original.V)


def test_common_baseline_conversion_preserves_predictions_and_residuals():
    states = np.array([[1.0, 3.0, -2.0], [2.0, 0.0, 4.0], [-1.0, 5.0, 2.0]])
    target = np.array([0.3, 1.0, -0.7])
    baseline = np.array([2.0, -1.0, 0.5])
    scales = np.array([0.5, 2.0, 3.0])
    effects = state_to_effect(states, baseline)
    np.testing.assert_allclose(effect_to_state(effects, baseline), states)
    state_residual = (states - target) / scales
    effect_residual = (effects - state_to_effect(target, baseline)) / scales
    np.testing.assert_allclose(state_residual, effect_residual)
    np.testing.assert_allclose(np.mean(state_residual ** 2, axis=1), np.mean(effect_residual ** 2, axis=1))


@pytest.mark.parametrize("baseline", [float(2**54), -float(2**54)])
@pytest.mark.parametrize("effect", [1.0, -1.0])
@pytest.mark.parametrize("conditioned", [False, True])
def test_small_native_effect_survives_equal_large_target_and_observation(baseline, effect, conditioned):
    values = np.full((3, 2) if conditioned else (2,), effect)
    result = score_references(
        np.full(2, baseline), np.full((3, 2), baseline),
        [Prediction("effect", values, "effect")], scales=np.array([1.0, 2.0]),
    )
    # The observed effect is exactly zero; signed unit predictions have MSE 5/8.
    np.testing.assert_array_equal(result.atomic_mse, np.full((1, 27), 0.625))
    np.testing.assert_array_equal(result.mse, np.full((1, 5), 0.625))


def test_signed_effect_and_equivalent_input_state_agree_under_S_and_O():
    controls = np.array([[-2.0, 4.0], [1.0, -1.0], [3.0, 2.0]])
    treated = np.array([0.5, -2.0])
    scales = np.array([0.5, 2.0])
    effects = np.array([[0.25, -0.5], [-1.0, 0.75], [0.5, -0.25]])
    predictions = [Prediction("effect", effects, "effect"),
                   Prediction("state", controls + effects, "state")]
    result = score_references(treated, controls, predictions, scales)
    oracle = _oracle(treated, controls, predictions, scales)
    for index, pattern in enumerate(result.patterns):
        np.testing.assert_allclose(result.mse[:, index], oracle[pattern], rtol=0, atol=1e-13)
    for index, (_, centring, model) in enumerate(result.role_tuples):
        if centring == model:
            assert result.atomic_mse[0, index] == result.atomic_mse[1, index]


def test_state_shared_reference_cancels_before_a_small_state_is_lost():
    result = score_references(
        [0.0], np.full((3, 1), float(2**54)), [Prediction("state", [1.0], "state")],
    )
    # Subtracting the huge baseline separately from state and target would lose 1.
    np.testing.assert_array_equal(result.atomic_mse, np.ones((1, 27)))


def test_scoring_does_not_modify_frozen_inputs():
    arrays = [np.array([1.0, 2.0]), np.arange(6.0).reshape(3, 2), np.array([0.1, 0.2]), np.array([0.5, 2.0])]
    snapshots = [array.copy() for array in arrays]
    for array in arrays:
        array.flags.writeable = False
    score_references(arrays[0], arrays[1], [Prediction("e", arrays[2], "effect"), Prediction("s", arrays[0], "state")], arrays[3])
    for actual, expected in zip(arrays, snapshots):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_inputs_fail_before_scoring(bad):
    with pytest.raises(ValueError):
        score_references(np.array([bad]), np.zeros((3, 1)), [Prediction("e", np.zeros(1), "effect")])
    with pytest.raises(ValueError):
        score_references(np.zeros(1), np.array([[0.0], [bad], [1.0]]), [Prediction("e", np.zeros(1), "effect")])
    with pytest.raises(ValueError):
        score_references(np.zeros(1), np.zeros((3, 1)), [Prediction("e", np.array([bad]), "effect")])


@pytest.mark.parametrize("scales", [[0.0, 1.0], [-1.0, 1.0], [np.nan, 1.0], [1.0], [[1.0, 1.0]]])
def test_invalid_scales_are_rejected(scales):
    with pytest.raises(ValueError):
        score_references(np.zeros(2), np.zeros((3, 2)), [Prediction("e", np.zeros(2), "effect")], np.array(scales))


@pytest.mark.parametrize(
    ("treated", "controls", "values", "kind"),
    [
        (np.zeros((1, 2)), np.zeros((3, 2)), np.zeros(2), "effect"),
        (np.zeros(2), np.zeros((2, 2)), np.zeros(2), "effect"),
        (np.zeros(2), np.zeros((3, 3)), np.zeros(2), "effect"),
        (np.zeros(2), np.zeros((3, 2)), np.zeros((2, 2)), "effect"),
        (np.zeros(2), np.zeros((3, 2)), np.zeros((2, 2)), "state"),
        (np.zeros(2), np.zeros((3, 2)), np.zeros(3), "state"),
        (np.zeros(0), np.zeros((3, 0)), np.zeros(0), "effect"),
    ],
)
def test_incompatible_prediction_shapes_are_rejected(treated, controls, values, kind):
    with pytest.raises(ValueError):
        score_references(treated, controls, [Prediction("model", values, kind)])


def test_duplicate_model_names_are_rejected():
    with pytest.raises(ValueError):
        score_references(np.zeros(2), np.zeros((3, 2)), [Prediction("same", np.zeros(2), "effect"), Prediction("same", np.zeros(2), "state")])


@pytest.mark.parametrize("setting", ["tie_tolerance", "identity_atol", "identity_rtol"])
@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf])
def test_invalid_numerical_tolerances_are_rejected(setting, value):
    with pytest.raises(ValueError):
        _simple(**{setting: value})


def _cells():
    values = np.arange(120, dtype=float).reshape(60, 2)
    ids = [f"cell_{index:03d}" for index in range(60)]
    strata = ["capture_a"] * 24 + ["capture_b"] * 36
    return values, ids, strata


def test_stratified_allocation_is_disjoint_and_means_match_membership():
    values, ids, strata = _cells()
    allocation = allocate_controls(values, ids, depth=5, seed=21, strata=strata)
    records = allocation.membership
    assert len(records) == 3 * 5 * 2
    assert len({row["cell_id"] for row in records}) == len(records)
    for block in range(3):
        block_rows = [row for row in records if row["block"] == block]
        assert len(block_rows) == 10
        for stratum in ("capture_a", "capture_b"):
            assert sum(row["stratum"] == stratum for row in block_rows) == 5
        for row in block_rows:
            assert ids[row["cell_index"]] == row["cell_id"]
            assert strata[row["cell_index"]] == row["stratum"]
        selected = [row["cell_index"] for row in block_rows]
        np.testing.assert_allclose(allocation.means[block], values[selected].mean(axis=0))


def test_control_allocation_reproducible_and_row_order_invariant():
    values, ids, strata = _cells()
    original = allocate_controls(values, ids, depth=5, seed=21, strata=strata)
    perm = np.random.default_rng(5).permutation(len(ids))
    reordered = allocate_controls(values[perm], [ids[index] for index in perm], depth=5, seed=21, strata=[strata[index] for index in perm])
    repeated = allocate_controls(values, ids, depth=5, seed=21, strata=strata)
    assignment = lambda allocation: {(row["cell_id"], row["block"]) for row in allocation.membership}
    assert assignment(original) == assignment(reordered) == assignment(repeated)
    np.testing.assert_array_equal(original.means, reordered.means)
    np.testing.assert_array_equal(original.means, repeated.means)
    changed = allocate_controls(values, ids, depth=5, seed=22, strata=strata)
    assert assignment(original) != assignment(changed)


def test_control_depths_are_nested_within_each_seed_and_stratum():
    values, ids, strata = _cells()
    shallow = allocate_controls(values, ids, depth=2, seed=21, strata=strata)
    deep = allocate_controls(values, ids, depth=6, seed=21, strata=strata)
    for block in range(3):
        low = {row["cell_id"] for row in shallow.membership if row["block"] == block}
        high = {row["cell_id"] for row in deep.membership if row["block"] == block}
        assert low < high


def test_control_support_is_checked_within_every_stratum():
    values, ids, strata = _cells()
    with pytest.raises(ValueError):
        allocate_controls(values, ids, depth=9, seed=21, strata=strata)


@pytest.mark.parametrize("problem", ["duplicate", "missing", "empty", "nonstrings"])
def test_invalid_cell_identifiers_are_rejected(problem):
    values, ids, strata = _cells()
    if problem == "duplicate":
        ids[1] = ids[0]
    elif problem == "missing":
        ids = ids[:-1]
    elif problem == "empty":
        ids[0] = ""
    else:
        ids[0] = 7
    with pytest.raises(ValueError):
        allocate_controls(values, ids, depth=2, seed=21, strata=strata)


@pytest.mark.parametrize(("argument", "value"), [("depth", 0), ("depth", 1.5), ("depth", True), ("seed", -1), ("seed", 1.5), ("seed", True)])
def test_allocation_requires_integer_depth_and_seed(argument, value):
    values, ids, strata = _cells()
    options = {"depth": 2, "seed": 21, "strata": strata}
    options[argument] = value
    with pytest.raises(ValueError):
        allocate_controls(values, ids, **options)
