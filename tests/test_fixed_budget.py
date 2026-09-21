from __future__ import annotations

import csv
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

from reference_design.fixed_budget import (
    DIAGNOSTIC_RULES, RULES, fixed_argmin, reduce_seed_losses, rule_cost,
    rule_partitions, score_independent_effect, score_reference_design,
)


@pytest.mark.parametrize("rule,expected", [
    ("shared_all_B", 0.), ("O_half", 16.), ("overlap_half", 4.),
    ("crossfit_2", 16.), ("crossfit2_overlap_all", 4.),
    ("crossfit_4", 64. / 9.),
])
def test_hand_calculated_state_losses(rule, expected):
    controls = np.array([[-2.], [-2.], [2.], [2.]])
    actual = score_reference_design(controls, np.zeros((4, 1)), [0.], [1.], rule)
    assert actual == pytest.approx(expected)


def test_direct_effect_is_not_centred_again():
    # Treated 7 minus observation mean 5 is an effect of 2, independent of input.
    controls = np.array([[1.], [1.], [5.], [5.]])
    assert score_reference_design(controls, [2.], [7.], [1.], "O_half") == 0.
    assert score_independent_effect(controls, [2.], [7.], [1.], [0, 1], [2, 3]) == 0.


def test_crossfit_squares_each_direction_before_averaging():
    controls = np.array([[-2.], [-2.], [2.], [2.]])
    # Directional residuals +4 and -4 average to zero, but mean squared loss is 16.
    assert score_reference_design(controls, np.zeros((4, 1)), [0.], [1.], "crossfit_2") == 16.


def test_seed_losses_preserve_variation_between_fits():
    control = np.zeros((4, 1))
    fits = [np.full((4, 1), -2.), np.full((4, 1), 2.)]
    losses = [score_reference_design(control, p, [0.], [1.], "shared_all_B") for p in fits]
    assert reduce_seed_losses(losses) == 4.
    assert score_reference_design(control, np.mean(fits, axis=0), [0.], [1.], "shared_all_B") == 0.
    np.testing.assert_array_equal(reduce_seed_losses([[1., 4.], [3., 8.]]), [2., 6.])


def test_weights_and_scales_have_a_hand_calculated_score():
    loss = score_reference_design(np.zeros((4, 2)), [2., 6.], [0., 0.], [2., 3.],
                                  "shared_all_B", weights=[0.25, 0.75])
    assert loss == 3.25


def test_independent_endpoint_accepts_unequal_nondivisible_roles():
    controls = np.array([[1.], [2.], [3.]])
    state = controls + 2.
    assert score_independent_effect(controls, state, [4.5], [1.], [0], [1, 2]) == 0.


@pytest.mark.parametrize("rule", RULES + DIAGNOSTIC_RULES)
def test_every_rule_uses_the_same_control_pool(rule):
    roles = rule_partitions(np.int64(16), rule)
    for inputs, observations in roles:
        np.testing.assert_array_equal(np.union1d(inputs, observations), np.arange(16))
        if rule in RULES and rule != "shared_all_B":
            assert not np.intersect1d(inputs, observations).size


@pytest.mark.parametrize("rule,uncached,cached", [
    ("shared_all_B", 16, 16), ("O_quarter", 4, 4), ("O_half", 8, 8),
    ("O_three_quarters", 12, 12), ("crossfit_2", 16, 16), ("crossfit_4", 48, 16),
    ("overlap_half", 8, 8), ("crossfit2_overlap_all", 16, 16),
])
def test_cost_counts_state_cell_forwards(rule, uncached, cached):
    cost = rule_cost(16, rule)
    assert cost["input_control_forwards_without_cache"] == uncached
    assert cost["unique_input_control_forwards_with_cache"] == cached
    assert cost["physical_control_cells"] == 16
    assert cost["uncached_ratio_to_shared"] == uncached / 16


def test_fixed_order_breaks_near_ties():
    assert fixed_argmin([1. + 5e-13, 1., 2.]) == 0
    assert fixed_argmin([1. + 5e-13, 1., 2.], tolerance=0) == 1


@pytest.mark.parametrize("budget", [0, -4, 2, 6, 4.0, True, np.bool_(True)])
def test_invalid_budgets_are_rejected(budget):
    with pytest.raises(ValueError):
        rule_partitions(budget, "O_half")


def test_unknown_rule_is_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        rule_partitions(4, "new_rule")


@pytest.mark.parametrize("inputs,observations", [
    ([], [1]), ([0], []), ([0.9], [1]), ([0.], [1]), ([True], [1]),
    ([0, 0], [1]), ([0], [1, 1]), ([0, 1], [1, 2]), ([-1], [1]),
    ([4], [1]), ([[0]], [1]), ([np.nan], [1]),
])
def test_invalid_independent_membership_is_rejected(inputs, observations):
    with pytest.raises(ValueError):
        score_independent_effect(np.zeros((4, 1)), [0.], [0.], [1.], inputs, observations)


@pytest.mark.parametrize("argument,value", [
    ("control_expression", np.empty((0, 1))),
    ("control_expression", np.empty((4, 0))),
    ("control_expression", [[np.nan]] * 4),
    ("state_points_or_effect", [np.inf]),
    ("state_points_or_effect", [[1.], [2.]]),
    ("state_points_or_effect", [1j]),
    ("treated_mean", [np.nan]), ("treated_mean", [1., 2.]),
    ("scales", [0.]), ("scales", [-1.]), ("scales", [np.inf]),
    ("weights", [0.5]), ("weights", [-1.]), ("weights", [np.nan]),
])
def test_invalid_scoring_inputs_are_rejected(argument, value):
    kwargs = dict(control_expression=np.zeros((4, 1)), state_points_or_effect=[0.],
                  treated_mean=[0.], scales=[1.], design="O_half")
    kwargs[argument] = value
    with pytest.raises(ValueError):
        score_reference_design(**kwargs)


@pytest.mark.parametrize("values", [[], [[]], -1., [np.nan], [-1.], [1j]])
def test_invalid_seed_losses_are_rejected(values):
    with pytest.raises(ValueError):
        reduce_seed_losses(values)


@pytest.mark.parametrize("tolerance", [-1., np.nan, np.inf, [0.], True])
def test_invalid_tolerance_is_rejected(tolerance):
    with pytest.raises(ValueError):
        fixed_argmin([0., 1.], tolerance)


@pytest.mark.parametrize("values", [[], [[1.]], [np.inf], [np.nan]])
def test_invalid_candidate_risks_are_rejected(values):
    with pytest.raises(ValueError):
        fixed_argmin(values)


def test_score_overflow_is_rejected_and_large_finite_seed_mean_is_supported():
    with pytest.raises(ValueError, match="overflowed"):
        score_reference_design(np.zeros((4, 1)), [1e300], [0.], [1.], "shared_all_B")
    assert reduce_seed_losses([1e308, 1e308]) == 1e308


def test_example_assesses_the_selected_candidates_with_the_same_seed_aggregation(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "examples" / "fixed_budget.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--output", str(tmp_path)])
    runpy.run_path(str(script), run_name="__main__")
    result = json.loads((tmp_path / "assessment.json").read_text())
    rows = list(csv.DictReader((tmp_path / "allocation_scores.tsv").open(), delimiter="\t"))
    rng = np.random.default_rng(7)
    rng.normal(size=(16, 3))
    controls = rng.normal(size=(24, 3))
    target = np.array([0.7, -0.2, 0.4]) - controls[8:].mean(axis=0)
    losses = []
    for shift in (0.2, 0.3):
        effect = (0.8 * controls[:8] + [0.5, -0.1, shift]).mean(axis=0) - controls[:8].mean(axis=0)
        losses.append(np.mean(((effect - target) / [1., 2., 0.5]) ** 2))
    assert result["candidate_independent_losses"]["state"] == pytest.approx(np.mean(losses))
    direct_loss = np.mean(((np.array([0.5, -0.1, 0.3]) - target) / [1., 2., 0.5]) ** 2)
    assert result["candidate_independent_losses"]["direct"] == pytest.approx(direct_loss)
    assert len(rows) == len(RULES)
    for row in rows:
        expected = result["candidate_independent_losses"][row["selected_candidate"]]
        assert float(row["independent_assessment_loss"]) == expected
        assert result["selected_by_rule"][row["rule"]]["independent_loss"] == expected


def test_example_rejects_a_nonempty_output_directory(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "examples" / "fixed_budget.py"
    existing = tmp_path / "assessment.json"
    existing.write_text("existing result")
    monkeypatch.setattr(sys, "argv", [str(script), "--output", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(script), run_name="__main__")
    assert error.value.code == 2
    assert existing.read_text() == "existing result"
    assert not (tmp_path / "allocation_scores.tsv").exists()
