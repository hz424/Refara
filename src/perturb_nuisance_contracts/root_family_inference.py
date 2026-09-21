"""Outcome-independent inference for an eight-root, eight-method family.

This module contains only the root-level statistical transformation used by
the public 192-root-utility to 84-contrast replay.  It has no expression,
prediction, membership, model-training, or production-configuration I/O.
Root codes are release-local pseudonyms and carry no study identifier mapping.
"""

from __future__ import annotations

import math
from types import ModuleType
from typing import Any

import numpy as np


ROOTS = 8
TASKS = 5
METHODS = 8
CONTRASTS = 84
ALPHA = 0.05
SIGN_COUNT = 256
SIGN_SEED = 1_046_527
SIGN_SCHEDULE_SHA256 = (
    "12d6e0393ce9981498e8d9f37f0a04b3a7ce50584c200d9bc99658a26c98f3b7"
)

SCHEMES = ("NAIVE_SHARED", "INDEPENDENT_SPLIT", "UNIT_AWARE_CROSSFIT")
DIRECT_METHODS = (
    "NO_CHANGE_DIRECT_V1",
    "CONTEXT_MEAN_EFFECT_DIRECT_V1",
    "TWO_WAY_ADDITIVE_RIDGE_DIRECT_V1",
    "PCA64_ADDITIVE_RIDGE_DIRECT_V1",
    "RBF_KERNEL_RIDGE_DIRECT_V1",
)
ABSOLUTE_METHODS = (
    "CPA_0_8_8_ABSOLUTE_V1",
    "SCGEN_2_1_1_ABSOLUTE_V1",
    "CELLOT_522D2B9_ABSOLUTE_V1",
)
METHOD_IDS = DIRECT_METHODS + ABSOLUTE_METHODS
ROOT_IDS = tuple(f"ROOT_{index:02d}" for index in range(1, ROOTS + 1))


class RootFamilyInferenceError(RuntimeError):
    """Raised when root-level inference geometry or semantics differ."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RootFamilyInferenceError(message)


def finite_or_none(value: float) -> float | None:
    """Return a finite scalar, or ``None`` for a conservative fallback."""

    result = float(value)
    return result if math.isfinite(result) else None


def point_ranking(values: np.ndarray) -> list[dict[str, Any]]:
    """Return the stable descending point ranking for one method vector."""

    utilities = np.asarray(values, dtype=np.float64)
    require(utilities.shape == (METHODS,), "point-utility geometry differs")
    require(np.isfinite(utilities).all(), "point utilities are nonfinite")
    order = np.argsort(-utilities, kind="stable")
    return [
        {
            "position": position,
            "method_id": METHOD_IDS[int(method)],
            "point_utility": float(utilities[int(method)]),
        }
        for position, method in enumerate(order, start=1)
    ]


def analyse_root_utilities(
    root_utilities: np.ndarray,
    generic: ModuleType,
) -> dict[str, Any]:
    """Apply the fixed complete-family max-|T| analysis to root utilities."""

    values = np.asarray(root_utilities, dtype=np.float64)
    require(
        values.shape == (ROOTS, len(SCHEMES), METHODS),
        "root utility geometry differs",
    )
    require(np.isfinite(values).all(), "root utilities are nonfinite")
    family = generic.contrast_family(METHODS)
    signs = generic.sign_schedule(ROOTS, monte_carlo_count=4_096, seed=SIGN_SEED)
    require(
        family.size == CONTRASTS
        and signs.shape == (SIGN_COUNT, ROOTS)
        and generic.sign_schedule_sha256(signs) == SIGN_SCHEDULE_SHA256,
        "fixed contrast family or R8 signs differ",
    )
    root_contrast = generic.root_contrasts(values[None, ...], family)
    inference = generic.infer_current_joint_maxt(root_contrast, signs, alpha=ALPHA)
    decisions = generic.decision_sets(
        inference["lower"],
        inference["upper"],
        family,
        np.zeros((len(SCHEMES), METHODS), dtype=np.float64),
    )
    point = np.mean(values, axis=0, dtype=np.float64)
    utility_se = np.std(values, axis=0, ddof=1) / math.sqrt(ROOTS)

    intervals: list[dict[str, Any]] = []
    for index, (scheme_index, left, right) in enumerate(family.identities):
        lower = float(inference["lower"][0, index])
        upper = float(inference["upper"][0, index])
        intervals.append(
            {
                "contrast_index": index,
                "contrast_id": (
                    f"{SCHEMES[scheme_index]}::"
                    f"{METHOD_IDS[left]}__VS__{METHOD_IDS[right]}"
                ),
                "scheme": SCHEMES[scheme_index],
                "method_a": METHOD_IDS[left],
                "method_b": METHOD_IDS[right],
                "point_difference": float(inference["point"][0, index]),
                "standard_error": float(inference["standard_errors"][0, index]),
                "lower": finite_or_none(lower),
                "upper": finite_or_none(upper),
                "resolved_direction": (
                    "A_GREATER"
                    if lower > 0.0
                    else "B_GREATER"
                    if upper < 0.0
                    else "UNRESOLVED"
                ),
            }
        )

    best_sets = {
        scheme: [
            METHOD_IDS[index]
            for index in range(METHODS)
            if bool(decisions["best_set"][0, scheme_index, index])
        ]
        for scheme_index, scheme in enumerate(SCHEMES)
    }
    ranks = {
        scheme: {
            method: [
                int(decisions["rank_lower"][0, scheme_index, method_index]),
                int(decisions["rank_upper"][0, scheme_index, method_index]),
            ]
            for method_index, method in enumerate(METHOD_IDS)
        }
        for scheme_index, scheme in enumerate(SCHEMES)
    }
    rankings = {
        scheme: point_ranking(point[scheme_index])
        for scheme_index, scheme in enumerate(SCHEMES)
    }

    leave_one_root_out: list[dict[str, Any]] = []
    for held_out in range(ROOTS):
        keep = np.arange(ROOTS) != held_out
        loo_point = np.mean(values[keep], axis=0, dtype=np.float64)
        loo_contrasts = generic.root_contrasts(
            values[None, keep, :, :], family
        )[0].mean(axis=0, dtype=np.float64)
        leave_one_root_out.append(
            {
                "held_out_root_id": ROOT_IDS[held_out],
                "analysis_role": "DESCRIPTIVE_ROOT_INFLUENCE_NOT_NEW_INFERENCE",
                "point_utilities": [
                    {
                        "scheme": scheme,
                        "method_id": method,
                        "point_utility": float(
                            loo_point[scheme_index, method_index]
                        ),
                    }
                    for scheme_index, scheme in enumerate(SCHEMES)
                    for method_index, method in enumerate(METHOD_IDS)
                ],
                "point_contrasts": [
                    {
                        "contrast_id": intervals[index]["contrast_id"],
                        "point_difference": float(loo_contrasts[index]),
                    }
                    for index in range(CONTRASTS)
                ],
                "deterministic_rankings": {
                    scheme: point_ranking(loo_point[scheme_index])
                    for scheme_index, scheme in enumerate(SCHEMES)
                },
            }
        )

    scheme_pairs = ((0, 1), (0, 2), (1, 2))
    shifts = [
        {
            "method_id": method,
            "scheme_a": SCHEMES[left],
            "scheme_b": SCHEMES[right],
            "point_utility_difference": float(
                point[left, method_index] - point[right, method_index]
            ),
        }
        for method_index, method in enumerate(METHOD_IDS)
        for left, right in scheme_pairs
    ]
    point_records = [
        {
            "scheme": scheme,
            "method_id": method,
            "point_utility": float(point[scheme_index, method_index]),
            "root_standard_error": float(utility_se[scheme_index, method_index]),
        }
        for scheme_index, scheme in enumerate(SCHEMES)
        for method_index, method in enumerate(METHOD_IDS)
    ]
    return {
        "fallback": bool(inference["fallback"][0]),
        "critical_value": finite_or_none(inference["critical_values"][0]),
        "sign_schedule": {
            "mode": "EXHAUSTIVE",
            "seed": SIGN_SEED,
            "count": SIGN_COUNT,
            "sha256": generic.sign_schedule_sha256(signs),
        },
        "point_utility_records": point_records,
        "pairwise_interval_records": intervals,
        "best_model_confidence_sets": best_sets,
        "possible_rank_intervals": ranks,
        "deterministic_point_rankings": rankings,
        "leave_one_root_out": leave_one_root_out,
        "reference_scheme_point_shifts": shifts,
        "reference_scheme_decision_changes": {
            "best_sets_not_all_equal": len(
                {tuple(best_sets[scheme]) for scheme in SCHEMES}
            )
            > 1,
            "point_rankings_not_all_equal": len(
                {
                    tuple(row["method_id"] for row in rankings[scheme])
                    for scheme in SCHEMES
                }
            )
            > 1,
            "possible_rank_sets_not_all_equal": len(
                {
                    tuple(
                        (method, tuple(ranks[scheme][method]))
                        for method in METHOD_IDS
                    )
                    for scheme in SCHEMES
                }
            )
            > 1,
        },
    }


__all__ = [
    "ABSOLUTE_METHODS",
    "ALPHA",
    "CONTRASTS",
    "DIRECT_METHODS",
    "METHODS",
    "METHOD_IDS",
    "ROOTS",
    "ROOT_IDS",
    "RootFamilyInferenceError",
    "SCHEMES",
    "SIGN_COUNT",
    "SIGN_SCHEDULE_SHA256",
    "SIGN_SEED",
    "TASKS",
    "analyse_root_utilities",
    "finite_or_none",
    "point_ranking",
]
