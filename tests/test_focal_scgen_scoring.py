from __future__ import annotations

import numpy as np
import pytest

from perturb_nuisance_focal.io import CapsuleError
from perturb_nuisance_focal.scoring import (
    focal_contrast,
    standardized_squared_utility,
    validate_axis,
)


def test_standardized_score_matches_direct_calculation() -> None:
    observed = np.asarray([[2.0, 5.0], [0.0, 2.0]])
    predicted = np.asarray([[1.0, 1.0], [2.0, 2.0]])
    scales = np.asarray([1.0, 2.0])
    weights = np.asarray([0.25, 0.75])
    expected = -np.sum(((observed - predicted) / scales) ** 2 * weights, axis=1)
    actual = standardized_squared_utility(observed, predicted, scales, weights)
    np.testing.assert_array_equal(actual, expected)


def test_focal_route_subtracts_prediction_references_in_float32() -> None:
    axis = validate_axis([0, 1], [0, 0], ["R01", "R02"], ["T01"])
    treated = np.asarray([[1.0], [1.0]], dtype="<f8")
    c_obs = np.zeros((5, 2, 1), dtype="<f8")
    pca = np.asarray([[0.4], [0.4]], dtype="<f4")
    native = np.empty((5, 2, 1), dtype="<f4")
    c_pred = np.zeros_like(native)
    native[0] = 0.5
    native[1:] = 0.0
    result = focal_contrast(
        treated,
        c_obs,
        pca,
        native,
        c_pred,
        np.ones(1),
        np.ones(1),
        axis,
    )
    assert result["shared_pca64_minus_scgen"] < 0
    assert result["split_pca64_minus_scgen"] > 0
    assert result["rotation_pca64_minus_scgen"] > 0
    assert result["direction_pattern_passed"] is True


@pytest.mark.parametrize(
    ("roots", "tasks"),
    (
        ([0, 0, 1, 1, 1], [0, 1, 0, 1, 1]),
        ([0, 1, 0, 1], [0, 0, 1, 1]),
    ),
)
def test_root_task_axis_rejects_duplicates_and_non_root_major_order(
    roots: list[int], tasks: list[int]
) -> None:
    with pytest.raises(CapsuleError, match="root-major"):
        validate_axis(roots, tasks, ["R01", "R02"], ["T01", "T02"])


def test_standardized_score_requires_unit_feature_weight() -> None:
    with pytest.raises(CapsuleError, match="incompatible"):
        standardized_squared_utility(
            np.ones((1, 2)),
            np.ones((1, 2)),
            np.ones(2),
            np.ones(2),
        )
