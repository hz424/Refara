"""Score reference allocations using a fixed pool of control cells.

State predictions must be one fixed, deterministic prediction per control cell:
changing the input subset, batch or its ordering must not change that cell's
prediction. This permits a point bank to be reused across allocations. Models
that depend on the input set require separate inference and are not supported
by this API. A direct effect is a fixed feature vector independent of controls.

Preprocess cells and fix feature scales before calling these functions. Row
order determines allocation; callers should randomize it before seeing outcomes
and use the same order for every candidate. These functions neither fit models
nor choose which design should be used for a new study.
"""
from __future__ import annotations

import numpy as np

RULES = (
    "shared_all_B", "O_quarter", "O_half", "O_three_quarters",
    "crossfit_2", "crossfit_4",
)
DIAGNOSTIC_RULES = ("overlap_half", "crossfit2_overlap_all")

__all__ = [
    "RULES", "DIAGNOSTIC_RULES", "rule_partitions", "rule_cost",
    "score_reference_design", "score_independent_effect", "reduce_seed_losses",
    "fixed_argmin",
]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _real_array(values, name):
    array = np.asarray(values)
    _require(array.dtype.kind in "iuf", f"{name} must contain real numbers")
    array = np.asarray(array, dtype=np.float64)
    _require(np.isfinite(array).all(), f"{name} must be finite")
    return array


def rule_partitions(n_controls: int, design: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return ordered (input rows, observed-reference rows) for each direction.

    Every rule uses the same pool of B cells; B must be a multiple of four.
    ``O_*`` puts the indicated fraction in input and the remainder in the
    observed reference. Cross-fitting holds out one observation fold at a time.
    The two diagnostic rules reuse all B observed controls while keeping the
    same input roles as O_half or crossfit_2.
    """
    _require(isinstance(n_controls, (int, np.integer))
             and not isinstance(n_controls, (bool, np.bool_))
             and n_controls >= 4 and n_controls % 4 == 0,
             "Control budget must be a positive multiple of four")
    _require(isinstance(design, str) and design in RULES + DIAGNOSTIC_RULES,
             "Unknown reference design")
    rows = np.arange(n_controls, dtype=np.intp)
    if design == "shared_all_B":
        return [(rows, rows)]
    if design == "overlap_half":
        return [(rows[:n_controls // 2], rows)]
    if design == "crossfit2_overlap_all":
        return [(fold, rows) for fold in np.split(rows, 2)]
    if design.startswith("O_"):
        numerator = {"O_quarter": 1, "O_half": 2, "O_three_quarters": 3}[design]
        cut = n_controls * numerator // 4
        return [(rows[:cut], rows[cut:])]
    folds = np.split(rows, 2 if design == "crossfit_2" else 4)
    return [(np.setdiff1d(rows, observation), observation) for observation in folds]


def rule_cost(n_controls: int, design: str) -> dict[str, int | float]:
    """Count input-cell forwards per state-model fit, relative to shared_all_B.

    Cached counts require the point-model property stated in this module.
    These are conceptual inference counts, not wall time or preprocessing cost.
    Multiply by the number of tasks and fits actually processed. Direct effect
    vectors need zero cell forwards and do not define a standalone cost ratio.
    When several rules share one bank, count their union of input cells once.
    """
    parts = rule_partitions(n_controls, design)
    total = sum(len(input_rows) for input_rows, _ in parts)
    unique = len(np.unique(np.concatenate([input_rows for input_rows, _ in parts])))
    return {
        "input_control_forwards_without_cache": total,
        "unique_input_control_forwards_with_cache": unique,
        "uncached_ratio_to_shared": total / n_controls,
        "cached_ratio_to_shared": unique / n_controls,
        "physical_control_cells": int(n_controls),
    }


def _validate_scoring(control_expression, prediction, treated_mean, scales, weights):
    control = _real_array(control_expression, "Control expression")
    predicted = _real_array(prediction, "Prediction")
    treated = _real_array(treated_mean, "Treated mean")
    scale = _real_array(scales, "Scales")
    _require(control.ndim == 2 and all(control.shape), "Controls must be a nonempty cells by features matrix")
    genes = control.shape[1]
    _require(predicted.shape in ((genes,), control.shape),
             "Prediction must be one direct effect or one state per control cell")
    _require(treated.shape == scale.shape == (genes,), "Target or scale feature axis differs")
    _require(np.all(scale > 0), "Scales must be positive")
    weight = np.full(genes, 1. / genes) if weights is None else _real_array(weights, "Feature weights")
    _require(weight.shape == (genes,) and np.all(weight >= 0)
             and abs(float(weight.sum()) - 1.) <= 1e-12,
             "Feature weights must be nonnegative and sum to one")
    return control, predicted, treated, scale, weight


def _role_loss(control, predicted, treated, scale, weights, input_rows, observation_rows):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        effect = (predicted if predicted.ndim == 1 else
                  predicted[input_rows].mean(axis=0) - control[input_rows].mean(axis=0))
        target = treated - control[observation_rows].mean(axis=0)
        error = (effect - target) / scale
        loss = float(np.sum(weights * error * error))
    _require(np.isfinite(loss), "Loss overflowed; check expression and scales")
    return loss


def score_reference_design(control_expression, state_points_or_effect, treated_mean,
                           scales, design: str, *, weights=None) -> float:
    """Return the mean directional MSE on fixed scales and feature weights.

    Controls and state points have shape (B, G); a direct effect has shape (G,).
    Native states are centred once using their input controls. Direct effects
    retain their existing baseline. The target is treated_mean minus the mean
    observed-reference controls. Each direction is squared before averaging;
    cross-fitting does not average predictions before scoring. All arithmetic
    uses float64. Default feature weights are uniform over the supplied G genes.
    """
    values = _validate_scoring(control_expression, state_points_or_effect, treated_mean, scales, weights)
    losses = [_role_loss(*values, i, o) for i, o in rule_partitions(len(values[0]), design)]
    return float(reduce_seed_losses(losses))


def _indices(values, size, name):
    rows = np.asarray(values)
    _require(rows.ndim == 1 and len(rows) > 0, f"{name} must be a nonempty index vector")
    _require(rows.dtype.kind in "iu", f"{name} must contain integer indices")
    _require(np.all(rows >= 0) and np.all(rows < size), f"{name} index outside controls")
    _require(len(np.unique(rows)) == len(rows), f"{name} contains repeated indices")
    return rows.astype(np.intp, copy=False)


def score_independent_effect(control_expression, state_points_or_effect, treated_mean,
                             scales, input_rows, observation_rows, *, weights=None) -> float:
    """Score a fixed prediction against disjoint observed-reference controls.

    Both role vectors contain unique integer row indices into the full control
    matrix. State points have that same full shape; only their input rows enter
    the score. No multiple-of-four budget constraint applies to this endpoint.
    """
    values = _validate_scoring(control_expression, state_points_or_effect, treated_mean, scales, weights)
    i = _indices(input_rows, len(values[0]), "Input role")
    o = _indices(observation_rows, len(values[0]), "Observation role")
    _require(not np.intersect1d(i, o).size, "Independent input and observation controls overlap")
    return _role_loss(*values, i, o)


def reduce_seed_losses(losses):
    """Average already-squared losses over axis 0, preserving remaining axes.

    Calculate a loss for each fit first. Averaging predictions before squaring
    instead evaluates an ensemble and answers a different question. Seeds and
    repeated allocations do not add independent biological replicates.
    """
    values = _real_array(losses, "Seed losses")
    _require(values.ndim >= 1 and all(values.shape), "Seed losses must have nonempty axes")
    _require(np.all(values >= 0), "Seed losses must be nonnegative")
    # Divide first so a finite mean cannot overflow while summing finite losses.
    return (values / values.shape[0]).sum(axis=0)


def fixed_argmin(values, tolerance: float = 1e-12) -> int:
    """Select the first candidate within tolerance of the smallest finite risk.

    The input order is the tie-breaking order; freeze it before evaluating data.
    Tolerance is an absolute loss difference, not a relative percentage.
    """
    risks = _real_array(values, "Candidate risks")
    _require(risks.ndim == 1 and len(risks) > 0, "Candidate risks must be a nonempty vector")
    tol = _real_array(tolerance, "Tie tolerance")
    _require(tol.ndim == 0 and tol >= 0, "Tie tolerance must be a nonnegative scalar")
    with np.errstate(over="ignore"):
        return int(np.flatnonzero(risks - risks.min() <= tol)[0])
