"""Small scored-matrix interface; every inferential use requires a declaration.

Margins are loss_b - loss_a, with individual fit losses scored before seed means.
Rows are equally weighted sampling units, never cells, seeds or random splits.
Positive directions favour model a. Calibration penalties are fixed before P.
"""
from collections.abc import Mapping
import hashlib
import math

import numpy as np

from . import _retrospective as kernel
from .reporting_candidate import RULES, coverage_plans


def _labels(values, name):
    kernel.require(not isinstance(values, (str, bytes)), "Invalid " + name)
    values = list(values)
    kernel.require(values and all(isinstance(v, str) and v for v in values)
                   and len(values) == len(set(values)), "Invalid " + name)
    return values


def _array(values, shape, missing=False):
    raw = np.asarray(values)
    kernel.require(raw.dtype.kind in "fiu", "Expected real numeric margins")
    if raw.dtype.kind in "iu":
        kernel.require(np.all(raw == raw.astype(float).astype(raw.dtype)),
                       "Integer margins cannot be represented exactly")
    arr = raw.astype(np.float64)
    kernel.require(not np.any((raw != 0) & (arr == 0)),
                   "Nonzero margin underflows to float64 zero")
    kernel.require(arr.shape == shape, "Matrix axes differ from declared identities")
    kernel.require(not np.isinf(arr).any() and (missing or np.isfinite(arr).all()),
                   "Nonfinite publication margin or infinite assessment margin")
    return arr


def publication_candidates(margins, *, unit_ids, comparison_ids, calibration_penalty,
                           allocation_contract, bootstrap_resamples=9999, seed=20260922):
    """Rank the five rules on one complete [unit, oriented allocation, pair] panel.

    Consecutive allocations (0,1), (2,3), ... must be complementary equal-cell
    O-half folds, each scored on its own held-out observation controls. Allocation
    0 is the prespecified anchor. Input-dependent models require corresponding
    predictions for each input set. This function cannot verify those facts from
    loss summaries; ``allocation_contract`` must state their upstream basis.

    Calibration penalties, one per comparison, are the maximum absolute change
    from the anchor on development/calibration units, pooled over the declared
    task/model-pair group. Assessment inputs must not determine them.
    """
    units = _labels(unit_ids, "publication units")
    ids = _labels(comparison_ids, "comparison roster")
    kernel.require(len(units) >= 2, "At least two publication units required")
    kernel.require(isinstance(allocation_contract, Mapping)
                   and allocation_contract.get("design") == "antithetic_O_half"
                   and allocation_contract.get("seed_aggregation") == "mean_after_scoring"
                   and isinstance(allocation_contract.get("basis"), str)
                   and bool(allocation_contract["basis"].strip()),
                   "Explicit scored-fold contract required")
    raw = np.asarray(margins)
    kernel.require(raw.ndim == 3 and raw.shape[1] >= 2 and raw.shape[1] % 2 == 0,
                   "Expected [units, even oriented allocations, comparisons]")
    x = _array(raw, (len(units), raw.shape[1], len(ids)))
    order = np.argsort(units)
    x = x[order]
    units = [units[j] for j in order]
    penalty = _array(calibration_penalty, (len(ids),))
    kernel.require(np.all(penalty >= 0), "Negative calibration penalty")
    kernel.require(type(bootstrap_resamples) is int and bootstrap_resamples >= 2
                   and type(seed) is int and seed >= 0, "Invalid bootstrap plan")
    anchor = x[:, 0].mean(axis=0)
    unit_mean = x.mean(axis=1)
    mean = unit_mean.mean(axis=0)
    pooled = x.mean(axis=0)
    cf2 = x[:, :2].mean(axis=(0, 1))
    schedule = np.random.default_rng(seed).integers(0, len(units), size=(bootstrap_resamples, len(units)))
    bootstrap = np.empty((bootstrap_resamples, len(ids)))
    for start in range(0, bootstrap_resamples, 128):
        stop = min(start + 128, bootstrap_resamples)
        bootstrap[start:stop] = unit_mean[schedule[start:stop]].mean(axis=1)
    kernel.require(all(np.isfinite(v).all() for v in (anchor, unit_mean, mean, pooled, cf2, bootstrap)),
                   "Nonfinite publication arithmetic")
    rows, companions = [], []
    for j, identity in enumerate(ids):
        for rule in RULES:
            estimate = anchor[j] if rule in ("score_margin", "existing_refara_reporting") else mean[j]
            d = kernel.sign(estimate)
            if rule == "repeated_splitting":
                score = np.sort(d * pooled[:, j])[math.floor(.05 * (x.shape[1] - 1))]
            elif rule == "independent_unit_resampling":
                score = np.sort(d * bootstrap[:, j])[math.floor(.05 * (bootstrap_resamples - 1))]
            elif rule == "existing_refara_reporting":
                score = abs(estimate) - penalty[j]
            else:
                score = abs(estimate)
            eligible = bool(d and score > kernel.TAU)
            rows.append(dict(rule_id=rule, pair_id=identity, estimate=float(estimate),
                             candidate_direction=kernel.direction(estimate), reliability_score=float(score),
                             eligible=eligible, reason="eligible" if eligible else "direction_tie" if not d else "score_below_floor"))
        value = float(cf2[j])
        companions.append(dict(rule_id="cross_fitting_first_partition", pair_id=identity,
                               estimate=value, candidate_direction=kernel.direction(value),
                               reliability_score=abs(value), eligible=bool(kernel.sign(value)),
                               reason="eligible" if kernel.sign(value) else "direction_tie"))
    return dict(candidates=rows, companions=companions, unit_ids=units,
                bootstrap_schedule_sha256=hashlib.sha256(schedule.tobytes()).hexdigest(),
                allocation_contract=dict(allocation_contract),
                scope="Scored-input calculation, not authentication of folds, independence or calibration.")


def common_coverage(candidates, *, comparison_ids):
    """Whole-score-tie prefixes common to all five rules, plus every natural set.

    Primary M is the largest achievable M <= 50% of N; evaluation floor is
    max(10, 25% of N). All-hold retains natural outcomes and M=0 in the curve.
    """
    ids = _labels(comparison_ids, "comparison roster")
    rows = list(candidates)
    kernel.require(len(rows) == len(ids) * len(RULES), "Incomplete five-rule roster")
    for rule in RULES:
        selected = [r for r in rows if r["rule_id"] == rule]
        kernel.require(len(selected) == len(ids) and {r["pair_id"] for r in selected} == set(ids),
                       "Duplicate/missing rule comparison")
    for row in rows:
        kernel.require(row["candidate_direction"] in ("a", "b", "hold")
                       and type(row["eligible"]) is bool
                       and math.isfinite(row["reliability_score"]), "Invalid candidate")
        kernel.require(row["eligible"] == (row["candidate_direction"] != "hold"
                       and row["reliability_score"] > kernel.TAU), "Inconsistent eligibility")
    return coverage_plans(rows, kernel.coverage_protocol(), len(ids))


def assessment_intervals(margins, *, unit_ids, comparison_ids, assumptions,
                         multiplicity=None, joint=True, resamples=19999, seed=20260923):
    """95% working Bonferroni-t / joint studentized-bootstrap envelope.

    Declare the sampling-unit definition, conditioning scope and rationale for
    working independence. A shared culture is not made independent by relabelling
    its donors. This API verifies declarations, not biological independence.
    NaN columns and constants without an authenticated identity are unassessable.
    All prespecified comparisons remain in the multiplicity family. Bootstrap
    0/0 is zero; nonzero/0 is infinite, with no narrower t-only fallback.
    These are approximate, assumption-conditional intervals, not calibrated
    distribution-free guarantees. No structural-zero shortcut is exposed here.
    """
    units = _labels(unit_ids, "assessment units")
    ids = _labels(comparison_ids, "comparison roster")
    kernel.require(len(units) >= 2, "At least two assessment units required")
    fields = {"unit_definition", "conditioning_scope", "working_independence_basis"}
    kernel.require(isinstance(assumptions, Mapping) and set(assumptions) == fields
                   and all(isinstance(v, str) and v.strip() for v in assumptions.values()),
                   "Explicit native-unit and working-independence declarations required")
    n_family = len(ids) if multiplicity is None else multiplicity
    kernel.require(type(n_family) is int and n_family >= len(ids), "Multiplicity cannot omit frozen comparisons")
    kernel.require(type(joint) is bool and type(resamples) is int and resamples >= 2
                   and type(seed) is int and seed >= 0, "Invalid interval plan")
    x = _array(margins, (len(units), len(ids)), missing=True)
    order = np.argsort(units)
    result = kernel.working_intervals(x[order], ids, multiplicity=n_family,
                                     joint=joint, resamples=resamples, seed=seed)
    result.update(unit_ids=[units[j] for j in order], assumptions=dict(assumptions),
                  scientific_confirmation=False, interval_calibration_performed=False,
                  scope="Working-independence equal-unit diagnostic, conditional on the supplied sampling declarations; nominal coverage is not validated by this API.")
    return result
