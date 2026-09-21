#!/usr/bin/env python3
"""Pure V21 multi-realization aggregation and working-law calculations."""

from __future__ import annotations

import itertools
import math
from types import ModuleType
from typing import Any

import numpy as np


ROOTS = 8
TASKS = 5
CONSTRUCTIONS = (
    "NAIVE_SHARED",
    "INDEPENDENT_SPLIT",
    "UNIT_AWARE_CROSSFIT",
)
REALIZATIONS = ("q1", "q2", "q3", "q4", "q5")
SEEDS = (17, 29, 43, 763929762, 85508424)
DIRECT_METHODS = (
    "NO_CHANGE_DIRECT_V1",
    "CONTEXT_MEAN_EFFECT_DIRECT_V1",
    "TWO_WAY_ADDITIVE_RIDGE_DIRECT_V1",
    "PCA64_ADDITIVE_RIDGE_DIRECT_V1",
    "RBF_KERNEL_RIDGE_DIRECT_V1",
)
STOCHASTIC_METHODS = (
    "CPA_0_8_8_ABSOLUTE_V1",
    "SCGEN_2_1_1_ABSOLUTE_V1",
    "CELLOT_522D2B9_ABSOLUTE_V1",
)
METHODS = DIRECT_METHODS + STOCHASTIC_METHODS
METHOD_PAIRS = tuple(itertools.combinations(range(len(METHODS)), 2))
ALPHA = 0.05
SIGN_SEED = 1_046_527
SIGN_COUNT = 256
SIGN_SCHEDULE_SHA256 = (
    "12d6e0393ce9981498e8d9f37f0a04b3a7ce50584c200d9bc99658a26c98f3b7"
)
FOCAL_METHOD_A = "PCA64_ADDITIVE_RIDGE_DIRECT_V1"
FOCAL_METHOD_B = "SCGEN_2_1_1_ABSOLUTE_V1"


class MultiRealizationAnalysisError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MultiRealizationAnalysisError(message)


def _finite_array(values: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    require(array.shape == shape, f"{label} geometry differs: {array.shape}")
    require(np.isfinite(array).all(), f"{label} contains nonfinite values")
    return array


def aggregate_utilities(
    direct_root_task: np.ndarray,
    stochastic_root_task: np.ndarray,
) -> dict[str, np.ndarray]:
    """Task-average first, then realization-average within stochastic method."""

    direct = _finite_array(
        direct_root_task,
        (len(DIRECT_METHODS), ROOTS, TASKS, len(CONSTRUCTIONS)),
        "fixed DIRECT root-task utility",
    )
    stochastic = _finite_array(
        stochastic_root_task,
        (
            len(STOCHASTIC_METHODS),
            len(REALIZATIONS),
            ROOTS,
            TASKS,
            len(CONSTRUCTIONS),
        ),
        "stochastic root-task utility",
    )
    direct_task_mean = np.mean(direct, axis=2, dtype=np.float64)
    stochastic_task_mean = np.mean(stochastic, axis=3, dtype=np.float64)
    stochastic_realization_mean = np.mean(
        stochastic_task_mean, axis=1, dtype=np.float64
    )

    per_realization = np.empty(
        (len(REALIZATIONS), ROOTS, len(CONSTRUCTIONS), len(METHODS)),
        dtype=np.float64,
    )
    direct_root_construction = np.transpose(direct_task_mean, (1, 2, 0))
    for realization_index in range(len(REALIZATIONS)):
        per_realization[realization_index, :, :, : len(DIRECT_METHODS)] = (
            direct_root_construction
        )
        per_realization[realization_index, :, :, len(DIRECT_METHODS) :] = (
            np.transpose(
                stochastic_task_mean[:, realization_index, :, :], (1, 2, 0)
            )
        )

    seed_averaged = np.empty(
        (ROOTS, len(CONSTRUCTIONS), len(METHODS)), dtype=np.float64
    )
    seed_averaged[:, :, : len(DIRECT_METHODS)] = direct_root_construction
    seed_averaged[:, :, len(DIRECT_METHODS) :] = np.transpose(
        stochastic_realization_mean, (1, 2, 0)
    )

    require(
        np.array_equal(
            per_realization[:, :, :, : len(DIRECT_METHODS)],
            np.broadcast_to(
                direct_root_construction,
                (
                    len(REALIZATIONS),
                    ROOTS,
                    len(CONSTRUCTIONS),
                    len(DIRECT_METHODS),
                ),
            ),
        ),
        "DIRECT root utilities drift across realization panels",
    )
    require(
        np.allclose(
            seed_averaged[:, :, len(DIRECT_METHODS) :],
            np.mean(
                per_realization[:, :, :, len(DIRECT_METHODS) :],
                axis=0,
                dtype=np.float64,
            ),
            rtol=0.0,
            atol=0.0,
        ),
        "stochastic realization aggregation order differs",
    )
    return {
        "direct_task_mean": direct_task_mean,
        "stochastic_task_mean": stochastic_task_mean,
        "per_realization_root_utilities": per_realization,
        "seed_averaged_root_utilities": seed_averaged,
    }


def within_construction_root_contrasts(root_utilities: np.ndarray) -> np.ndarray:
    values = _finite_array(
        root_utilities,
        (ROOTS, len(CONSTRUCTIONS), len(METHODS)),
        "root utility",
    )
    columns = [
        values[:, construction_index, left] - values[:, construction_index, right]
        for construction_index in range(len(CONSTRUCTIONS))
        for left, right in METHOD_PAIRS
    ]
    result = np.column_stack(columns)
    require(result.shape == (ROOTS, 84), "within-construction family differs")
    return result


def within_family_metadata() -> list[dict[str, Any]]:
    """Return the frozen construction-major identity of all 84 contrasts."""

    rows: list[dict[str, Any]] = []
    for construction_index, construction_id in enumerate(CONSTRUCTIONS):
        for pair_index, (left, right) in enumerate(METHOD_PAIRS):
            rows.append(
                {
                    "family_index": construction_index * len(METHOD_PAIRS)
                    + pair_index,
                    "construction_index": construction_index,
                    "construction_id": construction_id,
                    "pair_index": pair_index,
                    "method_a": METHODS[left],
                    "method_b": METHODS[right],
                    "contrast_orientation": "METHOD_A_MINUS_METHOD_B",
                }
            )
    require(len(rows) == 84, "within-family metadata geometry differs")
    return rows


def paired_construction_root_changes(root_utilities: np.ndarray) -> np.ndarray:
    within = within_construction_root_contrasts(root_utilities)
    shared = within[:, :28]
    result = np.column_stack((within[:, 28:56] - shared, within[:, 56:84] - shared))
    require(result.shape == (ROOTS, 56), "paired construction family differs")
    return result


def paired_family_metadata() -> list[dict[str, Any]]:
    """Return the frozen target-minus-shared identity of all 56 changes."""

    rows: list[dict[str, Any]] = []
    for target_index in (1, 2):
        for pair_index, (left, right) in enumerate(METHOD_PAIRS):
            rows.append(
                {
                    "family_index": (target_index - 1) * len(METHOD_PAIRS)
                    + pair_index,
                    "source_construction_index": 0,
                    "source_construction_id": CONSTRUCTIONS[0],
                    "target_construction_index": target_index,
                    "target_construction_id": CONSTRUCTIONS[target_index],
                    "pair_index": pair_index,
                    "method_a": METHODS[left],
                    "method_b": METHODS[right],
                    "change_orientation": (
                        "TARGET_METHOD_A_MINUS_METHOD_B_MINUS_"
                        "SHARED_METHOD_A_MINUS_METHOD_B"
                    ),
                }
            )
    require(len(rows) == 56, "paired-family metadata geometry differs")
    return rows


def working_law(root_contributions: np.ndarray, generic: ModuleType) -> dict[str, Any]:
    values = np.asarray(root_contributions, dtype=np.float64)
    require(
        values.ndim == 2 and values.shape[0] == ROOTS and np.isfinite(values).all(),
        "working-law root matrix differs",
    )
    signs = generic.sign_schedule(ROOTS, monte_carlo_count=4_096, seed=SIGN_SEED)
    require(
        signs.shape == (SIGN_COUNT, ROOTS)
        and generic.sign_schedule_sha256(signs) == SIGN_SCHEDULE_SHA256,
        "exhaustive R8 sign schedule differs",
    )
    inference = generic.infer_current_joint_maxt(values[None, :, :], signs, alpha=ALPHA)
    return {
        "point": np.asarray(inference["point"][0], dtype=np.float64),
        "standard_error": np.asarray(
            inference["standard_errors"][0], dtype=np.float64
        ),
        "lower": np.asarray(inference["lower"][0], dtype=np.float64),
        "upper": np.asarray(inference["upper"][0], dtype=np.float64),
        "critical_value": float(inference["critical_values"][0]),
        "fallback": bool(inference["fallback"][0]),
        "sign_count": SIGN_COUNT,
        "sign_schedule_sha256": SIGN_SCHEDULE_SHA256,
    }


def family_working_law_rows(
    metadata: list[dict[str, Any]], inference: dict[str, Any]
) -> list[dict[str, Any]]:
    """Join a frozen family identity to its simultaneous working-law output."""

    family_size = len(metadata)
    arrays = {
        key: _finite_array(inference.get(key), (family_size,), f"{key} family")
        for key in ("point", "standard_error", "lower", "upper")
    }
    require(
        np.all(arrays["standard_error"] >= 0.0),
        "working-law standard error is negative",
    )
    require(
        np.all(arrays["lower"] <= arrays["point"])
        and np.all(arrays["point"] <= arrays["upper"]),
        "working-law interval does not contain its point estimate",
    )
    critical_value = float(inference.get("critical_value", math.nan))
    require(math.isfinite(critical_value), "working-law critical value is nonfinite")
    fallback = inference.get("fallback")
    require(isinstance(fallback, (bool, np.bool_)), "working-law fallback differs")
    require(inference.get("sign_count") == SIGN_COUNT, "working-law sign count differs")
    require(
        inference.get("sign_schedule_sha256") == SIGN_SCHEDULE_SHA256,
        "working-law sign identity differs",
    )
    rows: list[dict[str, Any]] = []
    for index, identity in enumerate(metadata):
        require(identity.get("family_index") == index, "family metadata order differs")
        rows.append(
            {
                **identity,
                "point_estimate": float(arrays["point"][index]),
                "standard_error": float(arrays["standard_error"][index]),
                "simultaneous_lower": float(arrays["lower"][index]),
                "simultaneous_upper": float(arrays["upper"][index]),
                "simultaneous_critical_value": critical_value,
                "working_law_fallback": bool(fallback),
                "root_count": ROOTS,
                "sign_count": SIGN_COUNT,
                "sign_schedule_sha256": SIGN_SCHEDULE_SHA256,
                "inferential_label": "SIMULTANEOUS_WORKING_LAW_SENSITIVITY",
                "randomization_inference_claimed": False,
                "population_confidence_interval_claimed": False,
            }
        )
    return rows


def analyze_complete_inputs(
    *,
    direct_root_task: np.ndarray,
    stochastic_root_task: np.ndarray,
    generic: ModuleType,
    structural_complete: bool,
) -> dict[str, Any]:
    """Run the complete frozen V21 aggregation and sensitivity calculation."""

    require(structural_complete is True, "formal V21 analysis requires complete inputs")
    aggregation = aggregate_utilities(direct_root_task, stochastic_root_task)
    root_utilities = aggregation["seed_averaged_root_utilities"]
    within_roots = within_construction_root_contrasts(root_utilities)
    paired_roots = paired_construction_root_changes(root_utilities)
    within_inference = working_law(within_roots, generic)
    paired_inference = working_law(paired_roots, generic)
    focal = focal_diagnostics(aggregation["per_realization_root_utilities"])
    outcome_branch = select_outcome_branch(
        structural_complete=True,
        seed_averaged_root_utilities=root_utilities,
        paired_working_law=paired_inference,
        focal=focal,
    )
    return {
        "aggregation": aggregation,
        "within_root_contributions": within_roots,
        "paired_root_contributions": paired_roots,
        "within_working_law": within_inference,
        "paired_working_law": paired_inference,
        "within_working_law_rows": family_working_law_rows(
            within_family_metadata(), within_inference
        ),
        "paired_working_law_rows": family_working_law_rows(
            paired_family_metadata(), paired_inference
        ),
        "focal": focal,
        "stochastic_cartesian_diagnostics": stochastic_cartesian_diagnostics(
            aggregation["stochastic_task_mean"]
        ),
        "reversal_membership": reversal_membership(root_utilities),
        "outcome_branch": outcome_branch,
    }


def _focal_values(points: np.ndarray) -> dict[str, Any]:
    vector = _finite_array(points, (len(CONSTRUCTIONS),), "focal construction point")
    shared, split, rotation = (float(value) for value in vector)
    split_shift = split - shared
    rotation_shift = rotation - shared
    passed = (
        shared < 0.0
        and split > 0.0
        and rotation > 0.0
        and split_shift > 0.0
        and rotation_shift > 0.0
    )
    return {
        "shared_pca64_minus_scgen": shared,
        "split_pca64_minus_scgen": split,
        "rotation_pca64_minus_scgen": rotation,
        "split_minus_shared_shift": split_shift,
        "rotation_minus_shared_shift": rotation_shift,
        "direction_pattern_passed": passed,
    }


def focal_diagnostics(per_realization_root_utilities: np.ndarray) -> dict[str, Any]:
    panels = _finite_array(
        per_realization_root_utilities,
        (len(REALIZATIONS), ROOTS, len(CONSTRUCTIONS), len(METHODS)),
        "per-realization root utility",
    )
    method_a = METHODS.index(FOCAL_METHOD_A)
    method_b = METHODS.index(FOCAL_METHOD_B)
    root_contrasts = panels[:, :, :, method_a] - panels[:, :, :, method_b]
    per_realization: list[dict[str, Any]] = []
    for realization_index, realization_id in enumerate(REALIZATIONS):
        per_realization.append(
            {
                "realization_id": realization_id,
                "seed": SEEDS[realization_index],
                **_focal_values(
                    np.mean(root_contrasts[realization_index], axis=0, dtype=np.float64)
                ),
            }
        )
    leave_one_out: list[dict[str, Any]] = []
    for omitted_index, omitted_id in enumerate(REALIZATIONS):
        keep = np.arange(len(REALIZATIONS)) != omitted_index
        points = np.mean(
            root_contrasts[keep], axis=(0, 1), dtype=np.float64
        )
        leave_one_out.append(
            {
                "omitted_realization_id": omitted_id,
                "omitted_seed": SEEDS[omitted_index],
                "retained_realization_count": 4,
                **_focal_values(points),
            }
        )
    return {
        "per_realization": per_realization,
        "leave_one_realization_out": leave_one_out,
        "all_per_realization_passed": all(
            row["direction_pattern_passed"] for row in per_realization
        ),
        "all_leave_one_out_passed": all(
            row["direction_pattern_passed"] for row in leave_one_out
        ),
    }


def stochastic_cartesian_diagnostics(
    stochastic_task_mean: np.ndarray,
) -> list[dict[str, Any]]:
    values = _finite_array(
        stochastic_task_mean,
        (
            len(STOCHASTIC_METHODS),
            len(REALIZATIONS),
            ROOTS,
            len(CONSTRUCTIONS),
        ),
        "stochastic task-averaged utility",
    )
    rows: list[dict[str, Any]] = []
    estimands = (
        ("NAIVE_SHARED_CONTRAST", 0, None),
        ("INDEPENDENT_SPLIT_CONTRAST", 1, None),
        ("UNIT_AWARE_CROSSFIT_CONTRAST", 2, None),
        ("SPLIT_MINUS_SHARED_CHANGE", 1, 0),
        ("CROSSFIT_MINUS_SHARED_CHANGE", 2, 0),
    )
    for left, right in itertools.combinations(range(len(STOCHASTIC_METHODS)), 2):
        for left_q, left_id in enumerate(REALIZATIONS):
            for right_q, right_id in enumerate(REALIZATIONS):
                root_contrast = values[left, left_q] - values[right, right_q]
                for estimand_id, target, source in estimands:
                    root_values = root_contrast[:, target]
                    if source is not None:
                        root_values = root_values - root_contrast[:, source]
                    rows.append(
                        {
                            "method_a": STOCHASTIC_METHODS[left],
                            "realization_a": left_id,
                            "seed_a": SEEDS[left_q],
                            "method_b": STOCHASTIC_METHODS[right],
                            "realization_b": right_id,
                            "seed_b": SEEDS[right_q],
                            "estimand_id": estimand_id,
                            "root_mean": float(np.mean(root_values, dtype=np.float64)),
                            "descriptive_only": True,
                            "same_number_realization_is_statistical_pairing": False,
                        }
                    )
    require(len(rows) == 375, "stochastic Cartesian table geometry differs")
    return rows


def reversal_membership(root_utilities: np.ndarray) -> dict[str, Any]:
    point = np.mean(
        within_construction_root_contrasts(root_utilities), axis=0, dtype=np.float64
    ).reshape(len(CONSTRUCTIONS), len(METHOD_PAIRS))
    rows: list[dict[str, Any]] = []
    for pair_index, (left, right) in enumerate(METHOD_PAIRS):
        shared, split, rotation = (float(point[index, pair_index]) for index in range(3))
        split_reversal = shared * split < 0.0
        rotation_reversal = shared * rotation < 0.0
        rows.append(
            {
                "pair_index": pair_index,
                "method_a": METHODS[left],
                "method_b": METHODS[right],
                "shared_contrast": shared,
                "split_contrast": split,
                "rotation_contrast": rotation,
                "shared_to_split_strict_sign_reversal": split_reversal,
                "shared_to_rotation_strict_sign_reversal": rotation_reversal,
                "any_shared_to_target_strict_sign_reversal": (
                    split_reversal or rotation_reversal
                ),
            }
        )
    return {
        "rows": rows,
        "unique_pair_reversal_count": sum(
            row["any_shared_to_target_strict_sign_reversal"] for row in rows
        ),
        "shared_to_split_reversal_count": sum(
            row["shared_to_split_strict_sign_reversal"] for row in rows
        ),
        "shared_to_rotation_reversal_count": sum(
            row["shared_to_rotation_strict_sign_reversal"] for row in rows
        ),
    }


def select_outcome_branch(
    *,
    structural_complete: bool,
    seed_averaged_root_utilities: np.ndarray,
    paired_working_law: dict[str, Any],
    focal: dict[str, Any],
) -> dict[str, Any]:
    values = _finite_array(
        seed_averaged_root_utilities,
        (ROOTS, len(CONSTRUCTIONS), len(METHODS)),
        "seed-averaged root utility",
    )
    method_a = METHODS.index(FOCAL_METHOD_A)
    method_b = METHODS.index(FOCAL_METHOD_B)
    seed_averaged_points = np.mean(
        values[:, :, method_a] - values[:, :, method_b], axis=0, dtype=np.float64
    )
    seed_averaged_focal = _focal_values(seed_averaged_points)
    pair_index = METHOD_PAIRS.index((method_a, method_b))
    lower = np.asarray(paired_working_law["lower"], dtype=np.float64)
    upper = np.asarray(paired_working_law["upper"], dtype=np.float64)
    require(lower.shape == (56,) and upper.shape == (56,), "paired interval family differs")
    require(
        np.isfinite(lower).all()
        and np.isfinite(upper).all()
        and np.all(lower <= upper),
        "paired interval bounds are invalid",
    )
    split_lower = float(lower[pair_index])
    split_upper = float(upper[pair_index])
    rotation_lower = float(lower[28 + pair_index])
    rotation_upper = float(upper[28 + pair_index])
    paired_intervals_positive = (
        math.isfinite(split_lower)
        and math.isfinite(split_upper)
        and math.isfinite(rotation_lower)
        and math.isfinite(rotation_upper)
        and split_lower > 0.0
        and rotation_lower > 0.0
        and not bool(paired_working_law["fallback"])
    )
    scientific_pass = (
        seed_averaged_focal["direction_pattern_passed"]
        and paired_intervals_positive
    )
    algorithmic_pass = bool(
        focal["all_per_realization_passed"]
        and focal["all_leave_one_out_passed"]
    )
    strict_pass = structural_complete and scientific_pass and algorithmic_pass
    if strict_pass:
        branch_id = "STRICT_PASS"
        title_action = "RETAIN_CURRENT_TITLE"
        required_language = "stable across five training realizations"
    elif structural_complete and scientific_pass:
        branch_id = "AGGREGATE_PASS_WITH_REALIZATION_HETEROGENEITY"
        title_action = "TITLE_MUST_EXPLICITLY_INCLUDE_TRAINING_REALIZATION"
        required_language = "training realization"
    else:
        branch_id = "AGGREGATE_OR_56_FAMILY_FAILURE"
        title_action = "NARROW_TITLE_AND_ABSTRACT_TO_DESIGN_AUDIT"
        required_language = "do not retain a general ranking-reversal claim"
    return {
        "branch_id": branch_id,
        "title_action": title_action,
        "required_language": required_language,
        "structural_complete": structural_complete,
        "scientific_sensitivity_passed": scientific_pass,
        "algorithmic_stability_passed": algorithmic_pass,
        "strict_passed": strict_pass,
        "seed_averaged_focal": seed_averaged_focal,
        "paired_focal_intervals": {
            "shared_to_split": {"lower": split_lower, "upper": split_upper},
            "shared_to_rotation": {
                "lower": rotation_lower,
                "upper": rotation_upper,
            },
            "both_strictly_positive": paired_intervals_positive,
        },
    }


def structural_failure_branch(
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    """Freeze the no-substitution disposition when the 15-state panel is incomplete."""

    require(bool(failures), "structural failure branch requires a retained failure")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for failure in failures:
        require(isinstance(failure, dict), "failure inventory row differs")
        method_id = failure.get("method_id")
        realization_id = failure.get("realization_id")
        require(method_id in STOCHASTIC_METHODS, "failed method is outside V21")
        require(realization_id in REALIZATIONS, "failed realization is outside V21")
        identity = (str(method_id), str(realization_id))
        require(identity not in seen, "failure inventory contains a duplicate")
        seen.add(identity)
        normalized.append(
            {
                "method_id": method_id,
                "realization_id": realization_id,
                "seed": SEEDS[REALIZATIONS.index(realization_id)],
                "failure_record": failure.get("failure_record"),
                "replacement_training_realization_used": False,
            }
        )
    return {
        "branch_id": "STRUCTURAL_INCOMPLETE_RETAINED_FAILURE",
        "title_action": "NARROW_TITLE_AND_ABSTRACT_TO_DESIGN_AUDIT",
        "required_language": "do not retain a general ranking-reversal claim",
        "structural_complete": False,
        "scientific_sensitivity_passed": False,
        "algorithmic_stability_passed": False,
        "strict_passed": False,
        "formal_15_state_archive_complete": False,
        "failed_realization_count": len(normalized),
        "failures": normalized,
        "same_seed_same_configuration_infrastructure_retry_only": True,
        "algorithmic_or_nonfinite_failures_replaced": False,
    }


__all__ = [
    "ALPHA",
    "CONSTRUCTIONS",
    "DIRECT_METHODS",
    "FOCAL_METHOD_A",
    "FOCAL_METHOD_B",
    "METHODS",
    "METHOD_PAIRS",
    "MultiRealizationAnalysisError",
    "REALIZATIONS",
    "ROOTS",
    "SEEDS",
    "SIGN_COUNT",
    "SIGN_SCHEDULE_SHA256",
    "STOCHASTIC_METHODS",
    "TASKS",
    "aggregate_utilities",
    "analyze_complete_inputs",
    "family_working_law_rows",
    "focal_diagnostics",
    "paired_family_metadata",
    "paired_construction_root_changes",
    "require",
    "reversal_membership",
    "select_outcome_branch",
    "structural_failure_branch",
    "stochastic_cartesian_diagnostics",
    "within_construction_root_contrasts",
    "within_family_metadata",
    "working_law",
]
