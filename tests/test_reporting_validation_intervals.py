"""Hand-computed and independently enumerated synthetic checks, not calibration."""
from __future__ import annotations

import itertools
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.stats import t as student_t


from reference_design.reporting_validation.assessment_intervals import (MODE, paired_matrix_intervals,
                                 finite_panel_risk_bounds, paired_rule_difference_bounds,
                                 possible_signs)


def run(matrix, ids=None, **kwargs):
    matrix = np.asarray(matrix, dtype=float)
    units = kwargs.pop("unit_ids", [f"u{i}" for i in range(matrix.shape[0])])
    return paired_matrix_intervals(matrix, unit_ids=units,
        comparison_ids=ids or [f"c{i}" for i in range(matrix.shape[1])],
        mode=MODE, minimum_units=kwargs.pop("minimum_units", 2),
        resamples=kwargs.pop("resamples", 100), **kwargs)


def balanced_indices(n, count):
    return np.tile(np.arange(n), (count, 1))


def structural_evidence(identity, units):
    n = len(units)
    return dict(basis="identical_frozen_predictor_and_scoring_contract",
                comparison_id=identity, unit_ids=units,
                predictor_function_sha256_a="1" * 64, predictor_function_sha256_b="1" * 64,
                prediction_contract_sha256_a="2" * 64, prediction_contract_sha256_b="2" * 64,
                score_contract_sha256_a="3" * 64, score_contract_sha256_b="3" * 64,
                target_sha256_a=["4" * 64] * n, target_sha256_b=["4" * 64] * n,
                predictions_a=np.arange(n * 2, dtype=np.float32).reshape(n, 2),
                predictions_b=np.arange(n * 2, dtype=np.float32).reshape(n, 2),
                loss_a=np.ones(n), loss_b=np.ones(n))


def test_two_unit_cauchy_golden_and_full_N_bonferroni():
    result = run([[1, 2], [3, 6]], bootstrap_indices=balanced_indices(2, 100))
    # df=1 is Cauchy: its upper quantile is cot(pi * tail_probability).
    expected_t = 1 / math.tan(math.pi * 0.05 / 4)
    assert result["multiplicity_N"] == 2
    # scipy 1.12's df=1 inverse survival approximation differs from the
    # analytic cotangent by about 2.6e-12 relatively at this probability.
    assert result["t_critical"] == pytest.approx(expected_t, rel=5e-12)
    for name, mean, se in [("c0", 2, 1), ("c1", 4, 2)]:
        row = result["records"][name]
        assert row["mean"] == mean and row["standard_error"] == se
        assert row["bootstrap_interval"] == [mean, mean]
        assert row["interval"] == pytest.approx([mean - expected_t * se, mean + expected_t * se], rel=5e-12)


@pytest.mark.parametrize("constant", [0.0, 0.1, 7.0])
def test_unknown_constant_is_unassessable_not_zero_width_confidence(constant):
    result = run([[constant], [constant], [constant]])
    row = result["records"]["c0"]
    assert row["status"] == "unassessable"
    assert row["reason"] == "zero_original_SE_without_structural_identity"
    assert row["interval"] is None
    assert result["bootstrap_critical"] is None


def test_structural_zero_requires_and_checks_model_and_numeric_identity():
    proof = structural_evidence("same", ["u0", "u1", "u2"])
    result = run([[0], [0], [0]], ids=["same"], structural_zero_evidence={"same": proof})
    row = result["records"]["same"]
    assert row["status"] == "structural_zero" and row["interval"] == [0, 0]
    assert row["structural_zero_evidence"]["predictions_sha256"]
    changed = dict(proof, predictor_function_sha256_b="5" * 64)
    with pytest.raises(ValueError, match="predictor_function identity differs"):
        run([[0], [0], [0]], ids=["same"], structural_zero_evidence={"same": changed})
    changed = dict(proof, predictions_b=proof["predictions_b"] + 1)
    with pytest.raises(ValueError, match="not byte-identical"):
        run([[0], [0], [0]], ids=["same"], structural_zero_evidence={"same": changed})
    with pytest.raises(ValueError, match="input matrix"):
        run([[1], [1], [1]], ids=["same"], structural_zero_evidence={"same": proof})


def test_structural_zero_cannot_merge_unequal_integer_losses_in_float64():
    proof = structural_evidence("same", ["u0", "u1", "u2"])
    proof["loss_a"] = np.full(3, 2**53 + 1, dtype=np.int64)
    proof["loss_b"] = np.full(3, 2**53, dtype=np.int64)
    with pytest.raises(ValueError, match="raw paired losses differ"):
        run([[0], [0], [0]], ids=["same"], structural_zero_evidence={"same": proof})
    proof["loss_b"] = proof["loss_a"].copy()
    with pytest.raises(ValueError, match="loses integer precision"):
        run([[0], [0], [0]], ids=["same"], structural_zero_evidence={"same": proof})


def test_input_matrix_rejects_lossy_integer_cast_before_estimation():
    with pytest.raises(ValueError, match="loses integer precision"):
        paired_matrix_intervals(np.array([[2**53 + 1], [2**53]], dtype=np.int64),
                                unit_ids=["u0", "u1"], comparison_ids=["c0"], mode=MODE,
                                minimum_units=2, resamples=2)


def test_missing_column_keeps_all_N_and_does_not_drop_units():
    matrix = [[1, np.nan, 0], [3, 2, 0], [5, 3, 0]]
    result = run(matrix, bootstrap_indices=balanced_indices(3, 100))
    assert result["multiplicity_N"] == 3 and result["independent_unit_count"] == 3
    assert result["t_critical"] == pytest.approx(student_t.isf(0.05 / 6, 2))
    assert result["records"]["c0"]["status"] == "assessable"
    assert result["records"]["c1"]["reason"] == "missing_unit_margin"
    assert result["records"]["c1"]["observed_units"] == 2
    assert result["records"]["c2"]["interval"] is None


def test_zero_over_zero_draws_are_zero_and_retained():
    result = run([[-1], [0], [1]], bootstrap_indices=np.ones((100, 3), dtype=int))
    assert result["extended_studentization_counts"]["zero_over_zero"] == 100
    assert result["bootstrap_maxima"]["zero_draws"] == 100
    assert result["bootstrap_critical"] == 0
    assert result["records"]["c0"]["status"] == "assessable"


def test_isolated_infinite_draw_does_not_automatically_disqualify_interval():
    indices = balanced_indices(2, 100)
    indices[:4] = 0
    result = run([[1], [3]], bootstrap_indices=indices)
    assert result["extended_studentization_counts"]["nonzero_over_zero"] == 4
    assert result["bootstrap_maxima"]["positive_infinite_draws"] == 4
    assert result["bootstrap_maxima"]["quantile_index"] == 95
    assert result["bootstrap_critical"] == 0
    assert result["records"]["c0"]["status"] == "assessable"


def test_infinite_critical_quantile_cannot_fallback_to_narrower_t_interval():
    indices = balanced_indices(2, 100)
    indices[:5] = 0
    result = run([[1, 2], [3, 6]], bootstrap_indices=indices)
    assert result["bootstrap_critical"] is None
    assert result["bootstrap_status"] == "unbounded_positive_infinite_critical_quantile"
    for row in result["records"].values():
        assert row["t_interval"] is not None
        assert row["interval"] is None and row["status"] == "unassessable"


def independent_enumeration(matrix, indices, confidence):
    """Direct scalar implementation with fsum and sample SD, independent of module helpers."""
    n, N = matrix.shape
    means = [math.fsum(column) / n for column in matrix.T]
    ses = [math.sqrt(math.fsum((x - mu) ** 2 for x in column) / (n * (n - 1)))
           for column, mu in zip(matrix.T, means)]
    maxima = []
    for draw in indices:
        t_values = []
        for column, mu in zip(matrix.T, means):
            sampled = [float(column[i]) for i in draw]
            mean = math.fsum(sampled) / n
            se = math.sqrt(math.fsum((x - mean) ** 2 for x in sampled) / (n * (n - 1)))
            numerator = mean - mu
            t_values.append(abs(numerator / se) if se else 0 if numerator == 0 else math.inf)
        maxima.append(max(t_values))
    critical = sorted(maxima)[math.ceil(confidence * (len(indices) - 1))]
    t = student_t.isf((1 - confidence) / (2 * N), n - 1)
    intervals = [[mu - max(t, critical) * se, mu + max(t, critical) * se] for mu, se in zip(means, ses)]
    return critical, intervals


def test_joint_resampling_matches_complete_small_independent_enumeration():
    matrix = np.array([[-2, 4, 2], [1, 1, 3], [3, -2, 1], [8, 3, 11]], dtype=float)
    indices = np.array(list(itertools.product(range(4), repeat=4)))
    expected_q, expected_intervals = independent_enumeration(matrix, indices, 0.95)
    result = run(matrix, resamples=256, bootstrap_indices=indices)
    assert math.isfinite(expected_q)
    assert result["bootstrap_critical"] == pytest.approx(expected_q, rel=2e-14)
    np.testing.assert_allclose([r["interval"] for r in result["records"].values()], expected_intervals, rtol=2e-14, atol=1e-14)


def test_joint_max_quantile_with_discrete_column_matches_enumeration():
    # In the third column half the units equal 2: drawing only those units
    # has probability 1/16, enough to make the 95% max-|T| quantile infinite.
    matrix = np.array([[-2, 4, 2], [1, 1, 2], [3, -2, 1], [8, 3, 11]], dtype=float)
    indices = np.array(list(itertools.product(range(4), repeat=4)))
    expected_q, _ = independent_enumeration(matrix, indices, 0.95)
    result = run(matrix, resamples=256, bootstrap_indices=indices)
    assert math.isinf(expected_q)
    assert result["bootstrap_status"] == "unbounded_positive_infinite_critical_quantile"
    assert all(row["interval"] is None for row in result["records"].values())


def test_sign_flip_and_unit_comparison_order_invariance():
    matrix = np.array([[-2, 4], [1, 1], [3, -2], [8, 3]], dtype=float)
    kwargs = dict(unit_ids=["u2", "u0", "u3", "u1"], ids=["A", "B"], resamples=499, seed=55)
    original = run(matrix, **kwargs)
    flipped = run(-matrix, **kwargs)
    assert original["bootstrap_critical"] == pytest.approx(flipped["bootstrap_critical"])
    for identity in ["A", "B"]:
        lo, hi = original["records"][identity]["interval"]
        assert flipped["records"][identity]["interval"] == pytest.approx([-hi, -lo])
    perm = [2, 0, 3, 1]
    reordered = run(matrix[perm][:, ::-1], unit_ids=[kwargs["unit_ids"][i] for i in perm], ids=["B", "A"], resamples=499, seed=55)
    assert reordered["bootstrap_indices_sha256"] == original["bootstrap_indices_sha256"]
    assert reordered["bootstrap_critical"] == original["bootstrap_critical"]
    for identity in ["A", "B"]:
        assert reordered["records"][identity] == original["records"][identity]


def test_minimum_unit_gate_is_not_replaced_by_cell_or_bootstrap_count():
    result = run([[1], [2], [3]], minimum_units=24, resamples=999)
    assert result["records"]["c0"]["reason"] == "insufficient_independent_units"
    assert result["records"]["c0"]["interval"] is None


@pytest.mark.parametrize("change", ["mode", "shape", "duplicate_units", "bool_matrix", "infinity", "bootstrap_float", "bootstrap_out_of_range"])
def test_invalid_contracts_rejected(change):
    options = dict(margins=np.array([[1.], [3.]]), unit_ids=["u0", "u1"], comparison_ids=["c0"], mode=MODE, minimum_units=2, resamples=2)
    if change == "mode": options["mode"] = "real_data"
    if change == "shape": options["margins"] = [1, 3]
    if change == "duplicate_units": options["unit_ids"] = ["u", "u"]
    if change == "bool_matrix": options["margins"] = [[True], [False]]
    if change == "infinity": options["margins"] = [[1], [math.inf]]
    if change == "bootstrap_float": options["bootstrap_indices"] = [[0., 1.], [0., 1.]]
    if change == "bootstrap_out_of_range": options["bootstrap_indices"] = [[0, 2], [0, 1]]
    with pytest.raises(ValueError): paired_matrix_intervals(**options)


def records_fixture():
    return {"c1": dict(status="assessable", interval=[1., 2.]),
            "c2": dict(status="assessable", interval=[-2., -1.]),
            "c3": dict(status="assessable", interval=[-1., 1.]),
            "c4": dict(status="unassessable", interval=None, reason="missing_unit_margin")}


def test_finite_panel_denominators_and_hand_computed_bounds():
    records = records_fixture()
    directions = dict(c1="a", c2="a", c3="b", c4="hold")
    result = finite_panel_risk_bounds(list(records), directions, records)
    assert result["counts"] == dict(candidates=4, published=3, held=1, supported=1, reversed=1, unresolved=1, unassessable_publications=0)
    assert result["coverage"] == 0.75
    assert result["true_reversal_fraction_bounds"] == [1/3, 2/3]
    assert result["reversal_fraction"] == result["unresolved_fraction"] == 1/3


def test_paired_bounds_use_same_candidate_sign_not_separate_rule_extrema():
    records = records_fixture()
    j = dict(c1="a", c2="a", c3="b", c4="hold")
    k = dict(c1="b", c2="a", c3="hold", c4="a")
    comparison = paired_rule_difference_bounds(list(records), j, k, records)
    assert comparison["true_reversal_fraction_difference_bounds"] == [-2/3, 0]
    assert comparison["observed_fewer_reversals_without_more_unresolved"]
    assert not comparison["interval_supported_finite_panel_improvement"]
    identical = paired_rule_difference_bounds(list(records), j, j, records)
    assert identical["true_reversal_fraction_difference_bounds"] == [0, 0]
    # Independently enumerate every joint sign assignment allowed by these CIs.
    values = []
    for signs in itertools.product(*(possible_signs(records[c]) for c in records)):
        score = 0
        for c, sign in zip(records, signs):
            dj, dk = ({"a": 1, "b": -1, "hold": 0}[p[c]] for p in [j, k])
            score += int(dj * sign < 0) - int(dk * sign < 0)
        values.append(score / 3)
    assert [min(values), max(values)] == comparison["true_reversal_fraction_difference_bounds"]


def test_supported_finite_panel_improvement_and_swap_antisymmetry():
    records = records_fixture()
    j = dict(c1="a", c2="b", c3="hold", c4="hold")
    k = dict(c1="b", c2="a", c3="hold", c4="hold")
    result = paired_rule_difference_bounds(list(records), j, k, records)
    assert result["true_reversal_fraction_difference_bounds"] == [-1, -1]
    assert result["interval_supported_finite_panel_improvement"]
    reverse = paired_rule_difference_bounds(list(records), k, j, records)
    assert reverse["true_reversal_fraction_difference_bounds"] == [1, 1]


def test_touching_zero_missing_assessment_and_all_hold():
    records = {"a": dict(status="assessable", interval=[0, 1]),
               "b": dict(status="assessable", interval=[-1, 0]),
               "c": dict(status="unassessable", interval=None, reason="missing")}
    result = finite_panel_risk_bounds(list(records), dict(a="a", b="b", c="a"), records)
    assert result["counts"]["unresolved"] == result["counts"]["published"] == 3
    assert result["counts"]["unassessable_publications"] == 1
    holds = dict.fromkeys(records, "hold")
    result = paired_rule_difference_bounds(list(records), holds, holds, records)
    assert result["status"] == "no_publications"
    assert result["true_reversal_fraction_difference_bounds"] is None


def test_no_silent_roster_deletion_unequal_coverage_or_reversed_raw_bounds():
    records = records_fixture()
    j = dict(c1="a", c2="b", c3="hold", c4="hold")
    with pytest.raises(ValueError, match="same achieved publication count"):
        paired_rule_difference_bounds(list(records), j, dict.fromkeys(records, "a"), records)
    with pytest.raises(ValueError, match="complete frozen roster"):
        finite_panel_risk_bounds(list(records), j, {k:v for k,v in records.items() if k != "c4"})
    records["c1"]["interval"] = [2**53 + 1, 2**53]
    with pytest.raises(ValueError, match="reversed before conversion"):
        finite_panel_risk_bounds(list(records), j, records)
