"""Scoring and aggregation for the PCA-64 ridge versus scGen comparison."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import numpy as np

from .io import CapsuleError


REFERENCE_INSTANCES = (
    "D8_NAIVE_SHARED",
    "D8_INDEPENDENT_SPLIT",
    "D8_CROSSFIT_R0",
    "D8_CROSSFIT_R1",
    "D8_CROSSFIT_R2",
)
SCHEMES = ("NAIVE_SHARED", "INDEPENDENT_SPLIT", "UNIT_AWARE_CROSSFIT")
EVALUATION_ROOTS = 8
TASKS = 5
FEATURES = 2_000


@dataclass(frozen=True)
class RootTaskAxis:
    root_index: np.ndarray
    task_index: np.ndarray
    root_ids: tuple[str, ...]
    task_ids: tuple[str, ...]


def standardized_squared_utility(
    observed_effect: np.ndarray,
    predicted_effect: np.ndarray,
    scales: np.ndarray,
    weights: np.ndarray,
    feature_chunk: int = 256,
) -> np.ndarray:
    """Return negative standardized squared error for each root--task row."""

    observed = np.asarray(observed_effect)
    predicted = np.asarray(predicted_effect)
    scales = np.asarray(scales, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (
        observed.ndim != 2
        or observed.shape != predicted.shape
        or scales.ndim != 1
        or weights.ndim != 1
        or observed.shape[1] != scales.size
        or scales.shape != weights.shape
        or isinstance(feature_chunk, bool)
        or not isinstance(feature_chunk, int)
        or feature_chunk < 1
        or not np.isfinite(observed).all()
        or not np.isfinite(predicted).all()
        or not np.isfinite(scales).all()
        or not np.isfinite(weights).all()
        or np.any(scales <= 0)
        or np.any(weights <= 0)
        or not np.isclose(
            np.sum(weights, dtype=np.float64), 1.0, rtol=0.0, atol=1.0e-12
        )
    ):
        raise CapsuleError("The score inputs have incompatible shapes or values")
    result = np.zeros(observed.shape[0], dtype=np.float64)
    for start in range(0, observed.shape[1], feature_chunk):
        stop = min(start + feature_chunk, observed.shape[1])
        residual = (
            np.asarray(observed[:, start:stop], dtype=np.float64)
            - np.asarray(predicted[:, start:stop], dtype=np.float64)
        ) / scales[start:stop]
        result -= np.sum(
            np.square(residual) * weights[start:stop], axis=1, dtype=np.float64
        )
    if not np.isfinite(result).all():
        raise CapsuleError("The focal score is not finite")
    return result


def validate_axis(
    root_index: Sequence[int],
    task_index: Sequence[int],
    root_ids: Sequence[str],
    task_ids: Sequence[str],
) -> RootTaskAxis:
    root_values = tuple(root_index)
    task_values = tuple(task_index)
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in (*root_values, *task_values)
    ):
        raise CapsuleError("Root and task indices must be integers")
    try:
        roots = np.asarray(root_values, dtype=np.int64)
        tasks = np.asarray(task_values, dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as error:
        raise CapsuleError("Root and task indices must be integers") from error
    root_names = tuple(root_ids)
    task_names = tuple(task_ids)
    expected_rows = len(root_names) * len(task_names)
    expected_roots = np.repeat(np.arange(len(root_names), dtype=np.int64), len(task_names))
    expected_tasks = np.tile(np.arange(len(task_names), dtype=np.int64), len(root_names))
    if (
        roots.ndim != 1
        or roots.shape != tasks.shape
        or len(root_names) < 2
        or len(task_names) < 1
        or any(not isinstance(value, str) or not value for value in root_names)
        or any(not isinstance(value, str) or not value for value in task_names)
        or len(set(root_names)) != len(root_names)
        or len(set(task_names)) != len(task_names)
        or roots.size != expected_rows
        or not np.array_equal(roots, expected_roots)
        or not np.array_equal(tasks, expected_tasks)
    ):
        raise CapsuleError(
            "The root--task rows must form one root-major Cartesian panel"
        )
    return RootTaskAxis(roots, tasks, root_names, task_names)


def aggregate_root_utilities(
    instance_utilities: np.ndarray, axis: RootTaskAxis
) -> np.ndarray:
    values = np.asarray(instance_utilities, dtype=np.float64)
    expected = (len(REFERENCE_INSTANCES), axis.root_index.size, 2)
    if values.shape != expected or not np.isfinite(values).all():
        raise CapsuleError("The reference-instance utility array has the wrong shape")
    cubes = np.empty(
        (len(REFERENCE_INSTANCES), len(axis.root_ids), len(axis.task_ids), 2),
        dtype=np.float64,
    )
    for instance in range(len(REFERENCE_INSTANCES)):
        cubes[instance, axis.root_index, axis.task_index] = values[instance]
    task_means = np.mean(cubes, axis=2, dtype=np.float64)
    result = np.empty((len(axis.root_ids), len(SCHEMES), 2), dtype=np.float64)
    result[:, 0] = task_means[0]
    result[:, 1] = task_means[1]
    result[:, 2] = np.mean(task_means[2:5], axis=0, dtype=np.float64)
    return result


def focal_contrast(
    treated_means: np.ndarray,
    c_obs: np.ndarray,
    pca64_effect: np.ndarray,
    scgen_native: np.ndarray,
    c_pred: np.ndarray,
    scales: np.ndarray,
    weights: np.ndarray,
    axis: RootTaskAxis,
) -> dict[str, object]:
    treated = np.asarray(treated_means, dtype=np.float64)
    controls = np.asarray(c_obs, dtype=np.float64)
    pca = np.asarray(pca64_effect, dtype=np.float64)
    native = np.asarray(scgen_native)
    prediction_controls = np.asarray(c_pred)
    rows = axis.root_index.size
    features = np.asarray(scales).size
    if (
        treated.shape != (rows, features)
        or controls.shape != (len(REFERENCE_INSTANCES), rows, features)
        or pca.shape != (rows, features)
        or native.shape != controls.shape
        or prediction_controls.shape != controls.shape
    ):
        raise CapsuleError("The focal arrays do not share the expected axes")

    # The original materializer performed this subtraction in float32.
    scgen_effect = np.subtract(
        native.astype(np.float32, copy=False),
        prediction_controls.astype(np.float32, copy=False),
        dtype=np.float32,
    )
    instance_values = np.empty((len(REFERENCE_INSTANCES), rows, 2), dtype=np.float64)
    for index in range(len(REFERENCE_INSTANCES)):
        observed = treated - controls[index]
        instance_values[index, :, 0] = standardized_squared_utility(
            observed, pca, scales, weights
        )
        instance_values[index, :, 1] = standardized_squared_utility(
            observed, scgen_effect[index], scales, weights
        )
    root_values = aggregate_root_utilities(instance_values, axis)
    root_contrasts = root_values[:, :, 0] - root_values[:, :, 1]
    points = np.mean(root_contrasts, axis=0, dtype=np.float64)
    shared, split, rotation = (float(value) for value in points)
    return {
        "shared_pca64_minus_scgen": shared,
        "split_pca64_minus_scgen": split,
        "rotation_pca64_minus_scgen": rotation,
        "split_minus_shared_shift": split - shared,
        "rotation_minus_shared_shift": rotation - shared,
        "direction_pattern_passed": bool(
            shared < 0 and split > 0 and rotation > 0
        ),
        "root_contrasts": root_contrasts,
    }
