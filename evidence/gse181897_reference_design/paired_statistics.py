"""Paired donor statistics copied from the executed, locally frozen analysis.

Function bodies retain the frozen numerical implementation. Pool estimators and
private-data command-line code are omitted; this study's losses have multiple
physical-pool membership. Intervals are approximate, conditional-donor inference.
"""
from __future__ import annotations
import math
from collections import defaultdict
from typing import Any, Sequence
import numpy as np
from scipy.stats import t as student_t

EXECUTED_SOURCE_SHA256 = '0e3b704744a88cd4612f5f9ac4993a4ca55d57979988cbb29efa7a60fdccd2ac'
COPIED_FUNCTION_SHA256 = {'_inputs': '88f22aec0ca9feec0b285cc047ba43e08d0c367d1c72131dc622f512c3e1e7ae', 'donor_weights': '322662674044af1fefde4117bea7adf930bd091e70e98d3d604862d979acc35d', '_groups': 'b8cb2d16b203e4de4fe4b8103efada56ea08bb3b431fb3590e007705c284726e', '_cluster_covariance': 'a04bcdb1de423fc271e7046b59535a7f5ac14f813a454871327771af60cbcdae', '_interval': 'ca2892b08095a970e4c0e23bb2ae7f5f3e51ecea60d15ffed30495cca7c04ac1', '_classify': '5908bd564afbd3c1e816c68377d12ad2b37f42abed35673070dfe5967cdb7fe0', '_fieller': '6050663c1d3c7d0f87bbdccc6fc67009721139b5289f45f7e1e95537f85ef1de', '_summarize': '899b5bcca3c79263ea63f0a0f78e52b1c43283e3e17a6daf26c06de0e2d9735d', 'paired_loss_summary': 'f6c7ef5780568ef8cf806e05303f1c0beff9a82ead5bb601dcf4ff3a7c5f8e7c', 'studentized_bootstrap_sensitivity': 'ab2168fbe59e2d6f5813eb9df7eeb6892e1d6f6d43d9ab5a992c1c475fddb7c1'}

def _inputs(shared_loss, rule_loss, unit_ids, weights):
    shared = np.asarray(shared_loss, dtype=float)
    rule = np.asarray(rule_loss, dtype=float)
    units = [str(x) for x in unit_ids]
    weight = np.ones(len(shared)) if weights is None else np.asarray(weights, dtype=float)
    if shared.ndim != 1 or rule.shape != shared.shape or weight.shape != shared.shape:
        raise ValueError("Losses and weights must be equally sized one-dimensional arrays")
    if not len(shared) or len(units) != len(shared):
        raise ValueError("At least one paired observation and one unit ID per row are required")
    if not np.all(np.isfinite(shared)) or not np.all(np.isfinite(rule)):
        raise ValueError("Paired losses must be finite; missing outcomes cannot be silently excluded")
    if np.any(shared < 0) or np.any(rule < 0):
        raise ValueError("Prediction losses must be nonnegative")
    if not np.all(np.isfinite(weight)) or np.any(weight <= 0):
        raise ValueError("Prespecified weights must be positive and finite")
    if any(not x for x in units):
        raise ValueError("Unit IDs cannot be empty")
    return shared, rule, units, weight

def donor_weights(donor_ids: Sequence[str], task_weights=None) -> np.ndarray:
    """Normalize prespecified task weights so every donor has total weight one.

    All paired rows must already satisfy frozen eligibility rules. Repeated
    allocations should be averaged per task first or be given an explicit
    fraction of that task's total weight.
    """
    ids = [str(x) for x in donor_ids]
    weights = np.ones(len(ids)) if task_weights is None else np.asarray(task_weights, dtype=float)
    if weights.shape != (len(ids),) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Task weights must be positive, finite, and aligned to donors")
    totals = defaultdict(float)
    for donor, weight in zip(ids, weights):
        if not donor:
            raise ValueError("Donor IDs cannot be empty")
        totals[donor] += float(weight)
    return np.asarray([weight / totals[donor] for donor, weight in zip(ids, weights)])

def _groups(values, weights, labels):
    grouped = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[label].append(index)
    output = []
    for label in sorted(grouped):
        ix = grouped[label]
        total = float(np.sum(weights[ix]))
        output.append((label, ix, total, np.average(values[ix], weights=weights[ix], axis=0)))
    return output

def _cluster_covariance(values, weights, labels):
    """CR1 covariance of a weighted paired mean, with cluster score correction."""
    grouped = _groups(values, weights, labels)
    if len(grouped) < 2:
        return None, grouped
    mean = np.average(values, weights=weights, axis=0)
    scores = np.asarray([np.sum(weights[ix, None] * (values[ix] - mean), axis=0)
                         for _, ix, _, _ in grouped])
    covariance = (len(grouped) / (len(grouped) - 1)) * (scores.T @ scores) / np.sum(weights) ** 2
    return covariance, grouped

def _interval(estimate, variance, critical):
    if variance is None or critical is None or variance < 0:
        return None
    radius = critical * math.sqrt(variance)
    return [float(estimate - radius), float(estimate + radius)]

def _classify(interval):
    if interval is None:
        return "inconclusive"
    if interval[0] > 0:
        return "supported"
    if interval[1] < 0:
        return "insufficient"
    return "inconclusive"

def _fieller(shared_mean, improvement_mean, covariance, critical):
    """Finite Fieller interval only; otherwise expose the identification problem."""
    if covariance is None or critical is None or shared_mean <= 0:
        return {"interval": None, "status": "unavailable"}
    c2 = critical * critical
    a = shared_mean ** 2 - c2 * covariance[0, 0]
    b = -2 * (shared_mean * improvement_mean - c2 * covariance[0, 1])
    c = improvement_mean ** 2 - c2 * covariance[1, 1]
    discriminant = b * b - 4 * a * c
    if a <= 0:
        return {"interval": None, "status": "unbounded_or_disjoint"}
    if discriminant < 0:
        return {"interval": None, "status": "empty_or_numerically_unstable"}
    radius = math.sqrt(discriminant)
    return {"interval": [float((-b - radius) / (2 * a)), float((-b + radius) / (2 * a))],
            "status": "finite"}

def _summarize(shared, rule, weight, covariance, df, group_count, target_reduction,
               confidence, minimum_units, shared_inference_calls, rule_inference_calls,
               inference_cap, scope):
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be strictly between zero and one")
    if not 0 <= target_reduction < 1:
        raise ValueError("Target reduction must lie in [0, 1)")
    if isinstance(minimum_units, bool) or int(minimum_units) != minimum_units or minimum_units < 2:
        raise ValueError("At least two independent units are required; protocol may require more")
    mean_shared = float(np.average(shared, weights=weight))
    mean_rule = float(np.average(rule, weights=weight))
    improvement = mean_shared - mean_rule
    contrast = improvement - target_reduction * mean_shared
    valid = covariance is not None and group_count >= minimum_units and df >= 1
    critical = float(student_t.ppf((1 + confidence) / 2, df)) if valid else None
    direction = np.asarray([-target_reduction, 1.0])
    variance_t = float(direction @ covariance @ direction) if valid else None
    variance_d = float(covariance[1, 1]) if valid else None
    practical_ci = _interval(contrast, variance_t, critical)
    zero_ci = _interval(improvement, variance_d, critical)
    practical_state = _classify(practical_ci) if mean_shared > 0 else "inconclusive"
    if (shared_inference_calls is None) != (rule_inference_calls is None):
        raise ValueError("Inference costs must be supplied for both strategies or neither")
    cost_ratio = None
    cost_condition = "unknown"
    if shared_inference_calls is not None:
        if not (np.isfinite(shared_inference_calls) and np.isfinite(rule_inference_calls)):
            raise ValueError("Inference costs must be finite")
        if shared_inference_calls <= 0 or rule_inference_calls < 0:
            raise ValueError("Shared inference cost must be positive and rule cost nonnegative")
        cost_ratio = float(rule_inference_calls / shared_inference_calls)
        cost_condition = "satisfied" if cost_ratio <= inference_cap else "exceeded"
    if not np.isfinite(inference_cap) or inference_cap <= 0:
        raise ValueError("Inference cost cap must be positive and finite")
    adoption = ("supported" if practical_state == "supported" and cost_condition == "satisfied"
                else "not_supported" if practical_state == "insufficient" or cost_condition == "exceeded"
                else "inconclusive")
    warnings = []
    if group_count < minimum_units:
        warnings.append("Fewer independent units than the declared minimum; no confirmatory interval")
    if group_count < 10:
        warnings.append("Fewer than 10 independent units: Student-t calibration may be inaccurate")
    if mean_shared == 0:
        warnings.append("Shared loss is zero; a relative reduction is not defined")
    if covariance is None:
        warnings.append("Sampling covariance unavailable")
    return {
        "scope": scope,
        "method": "paired_weighted_mean_CR1_Student_t",
        "interval_status": "approximate" if valid else "unavailable",
        "confidence": confidence,
        "degrees_of_freedom": int(df),
        "independent_unit_count": int(group_count),
        "minimum_units": int(minimum_units),
        "mean_shared_loss": mean_shared,
        "mean_rule_loss": mean_rule,
        "mean_loss_improvement": improvement,
        "relative_loss_reduction": improvement / mean_shared if mean_shared > 0 else None,
        "relative_loss_reduction_fieller": _fieller(mean_shared, improvement, covariance if valid else None, critical),
        "target_reduction": target_reduction,
        "practical_contrast": contrast,
        "practical_contrast_formula": "(1 - target_reduction) * mean_shared_loss - mean_rule_loss",
        "practical_contrast_interval": practical_ci,
        "practical_benefit": practical_state,
        "zero_effect_interval": zero_ci,
        "zero_effect": ("beneficial" if _classify(zero_ci) == "supported"
                        else "harmful" if _classify(zero_ci) == "insufficient" else "inconclusive"),
        "inference_cost_ratio": cost_ratio,
        "inference_cost_cap": inference_cap,
        "inference_cost_condition": cost_condition,
        "adoption": adoption,
        "covariance_shared_improvement": None if covariance is None else covariance.tolist(),
        "warnings": warnings,
        "assumptions": [
            "Independent units, scope, model selection, tasks, budgets, weights, and exclusions were frozen before assessment outcomes",
            "Units are independent for the declared target population, conditional on all shared fixed resources named in scope",
            "Cluster Student-t intervals are approximate; their nominal coverage is not guaranteed for few, skewed, or uneven units",
            "This comparison alone does not establish transfer to another study",
        ],
    }

def paired_loss_summary(shared_loss, rule_loss, unit_ids, weights=None, *,
                        target_reduction=0.05, confidence=0.95, minimum_units=2,
                        shared_inference_calls=None, rule_inference_calls=None,
                        inference_cap=4.0, scope="unspecified") -> dict[str, Any]:
    """Summarize a frozen comparison; independent-unit identity is explicit.

    By default observations are equally weighted. Use ``donor_weights`` when
    donor task counts vary and the estimand gives equal weight to each donor.
    ``minimum_units=2`` is a mathematical minimum, not a validity endorsement.
    A protocol can declare a higher minimum before outcomes are inspected.
    """
    shared, rule, units, weight = _inputs(shared_loss, rule_loss, unit_ids, weights)
    values = np.column_stack((shared, shared - rule))
    covariance, grouped = _cluster_covariance(values, weight, units)
    result = _summarize(shared, rule, weight, covariance, len(grouped) - 1, len(grouped),
                        target_reduction, confidence, minimum_units, shared_inference_calls,
                        rule_inference_calls, inference_cap, scope)
    total_weight = float(np.sum(weight))
    shares = np.asarray([w / total_weight for _, _, w, _ in grouped])
    result["effective_weighted_unit_count"] = float(1 / np.sum(shares ** 2))
    result["maximum_unit_weight_share"] = float(np.max(shares))
    result["observation_count"] = len(shared)
    result["units"] = [
        {"unit_id": label, "observation_count": len(ix), "weight": w,
         "shared_loss": float(means[0]), "rule_loss": float(means[0] - means[1]),
         "loss_improvement": float(means[1]),
         "practical_contrast": float(means[1] - target_reduction * means[0])}
        for label, ix, w, means in grouped
    ]
    result["leave_one_unit_out"] = []
    for label, _, _, _ in grouped:
        keep = np.asarray([unit != label for unit in units])
        if not np.any(keep):
            continue
        s = float(np.average(shared[keep], weights=weight[keep]))
        r = float(np.average(rule[keep], weights=weight[keep]))
        result["leave_one_unit_out"].append(
            {"omitted_unit": label, "mean_shared_loss": s, "mean_rule_loss": r,
             "relative_loss_reduction": (s - r) / s if s > 0 else None,
             "practical_contrast": (1 - target_reduction) * s - r})
    if np.max(shares) > 0.25:
        result["warnings"].append("One independent unit carries more than 25% of total weight")
    return result

def studentized_bootstrap_sensitivity(shared_loss, rule_loss, unit_ids, weights=None, *,
                                      seed: int, resamples=19999,
                                      target_reduction=0.05, confidence=0.95, minimum_units=2,
                                      shared_inference_calls=None, rule_inference_calls=None,
                                      inference_cap=4.0, scope="unspecified") -> dict[str, Any]:
    """Resample entire declared units, preserving their paired losses and weights.

    The bootstrap is studentized with the same CR1 mean standard error in each
    sample. A joint decision requires both t and bootstrap intervals to lie on
    the same side of the practical threshold. This is an approximate, frozen
    decision rule; neither interval repairs an invalid independence assumption.
    """
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("A nonnegative integer bootstrap seed is required")
    if isinstance(resamples, bool) or int(resamples) != resamples or resamples < 99:
        raise ValueError("At least 99 bootstrap resamples are required")
    shared, rule, units, weight = _inputs(shared_loss, rule_loss, unit_ids, weights)
    base = paired_loss_summary(
        shared, rule, units, weight, target_reduction=target_reduction, confidence=confidence,
        minimum_units=minimum_units, shared_inference_calls=shared_inference_calls,
        rule_inference_calls=rule_inference_calls, inference_cap=inference_cap, scope=scope)
    grouped = _groups(np.column_stack((shared, shared - rule)), weight, units)
    group_count = len(grouped)
    output = {
        "method": "independent_unit_studentized_bootstrap",
        "scope": scope, "confidence": confidence, "seed": int(seed),
        "resamples": int(resamples), "independent_unit_count": group_count,
        "practical_contrast": base["practical_contrast"],
        "mean_loss_improvement": base["mean_loss_improvement"],
        "relative_loss_reduction": base["relative_loss_reduction"],
        "quantile_method": "inverted_cdf",
        "practical_contrast_interval": None,
        "zero_effect_interval": None,
        "practical_benefit": "inconclusive",
        "zero_effect": "inconclusive",
        "paired_t": {key: base[key] for key in (
            "practical_contrast_interval", "practical_benefit", "zero_effect_interval", "zero_effect")},
        "inference_cost_ratio": base["inference_cost_ratio"],
        "inference_cost_condition": base["inference_cost_condition"],
        "joint_practical_benefit": "inconclusive",
        "joint_adoption": "inconclusive",
        "warnings": list(base["warnings"]),
        "assumptions": list(base["assumptions"]),
    }
    output["assumptions"].append(
        "Bootstrap resamples the declared independent units; it is not distribution-free or guaranteed under arbitrary rare tails")
    if base["interval_status"] == "unavailable":
        if base["inference_cost_condition"] == "exceeded":
            output["joint_adoption"] = "not_supported"
        return output
    means = np.asarray([means for _, _, _, means in grouped])
    group_weights = np.asarray([w for _, _, w, _ in grouped])
    # Columns are the practical-margin contrast and zero-effect contrast.
    unit_values = np.column_stack((means[:, 1] - target_reduction * means[:, 0], means[:, 1]))
    estimate = np.average(unit_values, weights=group_weights, axis=0)
    original_scores = group_weights[:, None] * (unit_values - estimate)
    original_se = np.sqrt(group_count / (group_count - 1) * np.sum(original_scores ** 2, axis=0)
                          / np.sum(group_weights) ** 2)
    pivots = np.empty((resamples, 2))
    generator = np.random.default_rng(seed)
    zero_se_counts = np.zeros(2, dtype=int)
    numerical_scale = np.maximum(np.max(np.abs(unit_values), axis=0), np.finfo(float).tiny)
    tolerance = np.finfo(float).eps * numerical_scale * 32
    for start in range(0, resamples, 256):
        end = min(start + 256, resamples)
        indexes = generator.integers(0, group_count, size=(end - start, group_count))
        values = unit_values[indexes]
        draw_weights = group_weights[indexes]
        total_weight = np.sum(draw_weights, axis=1)
        draw_mean = np.sum(draw_weights[:, :, None] * values, axis=1) / total_weight[:, None]
        scores = draw_weights[:, :, None] * (values - draw_mean[:, None, :])
        draw_se = np.sqrt(group_count / (group_count - 1) * np.sum(scores ** 2, axis=1)
                          / total_weight[:, None] ** 2)
        numerator = draw_mean - estimate
        valid = draw_se > tolerance
        pivot = np.zeros_like(draw_mean)
        np.divide(numerator, draw_se, out=pivot, where=valid)
        pivot[(~valid) & (numerator > tolerance)] = np.inf
        pivot[(~valid) & (numerator < -tolerance)] = -np.inf
        pivots[start:end] = pivot
        zero_se_counts += np.sum(~valid, axis=0)
    alpha = (1 - confidence) / 2
    quantiles = np.quantile(pivots, [alpha, 1 - alpha], axis=0, method="inverted_cdf")
    with np.errstate(invalid="ignore"):
        low = estimate - quantiles[1] * original_se
        high = estimate - quantiles[0] * original_se
    for index, name in enumerate(("practical_contrast_interval", "zero_effect_interval")):
        if np.isfinite(low[index]) and np.isfinite(high[index]):
            output[name] = [float(low[index]), float(high[index])]
        else:
            output["warnings"].append(f"Nonfinite studentized-bootstrap {name}; result is inconclusive")
    output["zero_standard_error_resample_counts"] = zero_se_counts.tolist()
    if np.any(original_se <= tolerance):
        output["warnings"].append("Observed unit variance is numerically zero for at least one contrast; unseen rare tails cannot be assessed")
    bootstrap_state = (_classify(output["practical_contrast_interval"])
                       if base["mean_shared_loss"] > 0 else "inconclusive")
    output["practical_benefit"] = bootstrap_state
    zero_state = _classify(output["zero_effect_interval"])
    output["zero_effect"] = "beneficial" if zero_state == "supported" else "harmful" if zero_state == "insufficient" else "inconclusive"
    joint_state = bootstrap_state if bootstrap_state == base["practical_benefit"] else "inconclusive"
    output["joint_practical_benefit"] = joint_state
    output["joint_adoption"] = (
        "supported" if joint_state == "supported" and base["inference_cost_condition"] == "satisfied"
        else "not_supported" if joint_state == "insufficient" or base["inference_cost_condition"] == "exceeded"
        else "inconclusive")
    return output
