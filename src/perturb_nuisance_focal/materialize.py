"""Prediction materialization for a newly trained focal scGen model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .io import CapsuleError, load_array, load_manifest, read_tsv
from .scoring import (
    EVALUATION_ROOTS,
    FEATURES,
    REFERENCE_INSTANCES,
    TASKS,
    validate_axis,
)


CONTROL_COLUMNS = (
    "control_index",
    "root_task_index",
    "root_id",
    "root_index",
    "task_id",
    "task_index",
    "base_pool",
    "cell_id",
)
ROOT_TASK_COLUMNS = (
    "root_task_index",
    "root_id",
    "root_index",
    "task_id",
    "task_index",
)
DISPATCH_COLUMNS = (
    "reference_instance",
    "c_obs_pools",
    "c_pred_pools",
    "c_infer_pools",
)
REFERENCE_POOLS = ("B1", "B2", "B3")
CELLS_PER_REFERENCE_POOL = 8
EXPECTED_DISPATCH = {
    "D8_NAIVE_SHARED": ("B1|B2|B3", "B1|B2|B3", "B1|B2|B3"),
    "D8_INDEPENDENT_SPLIT": ("B1", "B2", "B3"),
    "D8_CROSSFIT_R0": ("B1", "B2", "B3"),
    "D8_CROSSFIT_R1": ("B2", "B3", "B1"),
    "D8_CROSSFIT_R2": ("B3", "B1", "B2"),
}


@dataclass(frozen=True)
class EvaluationControls:
    matrix: np.ndarray
    rows: tuple[Mapping[str, str], ...]
    root_tasks: tuple[Mapping[str, str], ...]
    dispatch: Mapping[str, Mapping[str, str]]
    feature_ids: tuple[str, ...]
    scales: np.ndarray
    weights: np.ndarray
    root_ids: tuple[str, ...]
    task_ids: tuple[str, ...]


def _integer(text: str, label: str) -> int:
    try:
        value = int(text)
    except (TypeError, ValueError, OverflowError) as error:
        raise CapsuleError(f"{label} must be an integer") from error
    if str(value) != text:
        raise CapsuleError(f"{label} must use canonical integer notation")
    return value


def _finite_float(text: str, label: str) -> float:
    try:
        value = float(text)
    except (TypeError, ValueError, OverflowError) as error:
        raise CapsuleError(f"{label} must be numeric") from error
    if not np.isfinite(value):
        raise CapsuleError(f"{label} must be finite")
    return value


def load_evaluation_controls(root: Path) -> EvaluationControls:
    root = Path(root)
    manifest = load_manifest(root, "gse162632_scgen_prepared_training")
    matrix = load_array(root, manifest, "evaluation_controls")
    rows = tuple(read_tsv(root, manifest, "evaluation_control_axis", CONTROL_COLUMNS))
    root_tasks = tuple(read_tsv(root, manifest, "root_tasks", ROOT_TASK_COLUMNS))
    dispatch_rows = read_tsv(root, manifest, "reference_dispatch", DISPATCH_COLUMNS)
    panel = read_tsv(
        root,
        manifest,
        "panel",
        ("feature_index", "feature_id", "scale", "weight"),
    )
    feature_ids = tuple(row["feature_id"] for row in panel)
    scales = np.asarray(
        [_finite_float(row["scale"], "panel scale") for row in panel], dtype="<f8"
    )
    weights = np.asarray(
        [_finite_float(row["weight"], "panel weight") for row in panel], dtype="<f8"
    )
    root_ids = tuple(dict.fromkeys(row["root_id"] for row in root_tasks))
    task_ids = tuple(dict.fromkeys(row["task_id"] for row in root_tasks))
    root_indices = [
        _integer(row["root_index"], "evaluation root index") for row in root_tasks
    ]
    task_indices = [
        _integer(row["task_index"], "evaluation task index") for row in root_tasks
    ]
    if (
        root_ids != tuple(f"R{index:02d}" for index in range(41, 49))
        or task_ids != tuple(f"T{index:02d}" for index in range(1, TASKS + 1))
    ):
        raise CapsuleError("The evaluation axis must use R41--R48 and T01--T05 in order")
    validate_axis(root_indices, task_indices, root_ids, task_ids)
    if (
        matrix.shape
        != (
            EVALUATION_ROOTS * TASKS * len(REFERENCE_POOLS) * CELLS_PER_REFERENCE_POOL,
            FEATURES,
        )
        or matrix.dtype not in (np.dtype("<f4"), np.dtype("<f8"))
        or np.any(matrix < 0)
        or len(rows) != matrix.shape[0]
        or len(root_tasks) != EVALUATION_ROOTS * TASKS
        or len(panel) != FEATURES
        or [
            _integer(row["control_index"], "evaluation control index")
            for row in rows
        ]
        != list(range(len(rows)))
        or [
            _integer(row["root_task_index"], "evaluation root--task index")
            for row in root_tasks
        ]
        != list(range(len(root_tasks)))
        or [_integer(row["feature_index"], "feature index") for row in panel]
        != list(range(FEATURES))
        or len(set(feature_ids)) != FEATURES
        or any(not value for value in (*feature_ids, *root_ids, *task_ids))
        or len({row["cell_id"] for row in rows}) != len(rows)
        or any(not row["cell_id"] for row in rows)
        or np.any(scales < 0.1)
        or not np.all(weights == np.float64(1.0 / FEATURES))
        or not np.isclose(
            np.sum(weights, dtype=np.float64), 1.0, rtol=0.0, atol=1.0e-12
        )
    ):
        raise CapsuleError("The evaluation controls do not match the 960-cell panel")

    for row in root_tasks:
        offset = _integer(row["root_task_index"], "evaluation root--task index")
        if (
            row["root_id"] != root_ids[root_indices[offset]]
            or row["task_id"] != task_ids[task_indices[offset]]
        ):
            raise CapsuleError("Evaluation labels and integer indices disagree")
    for row in rows:
        root_task_index = _integer(
            row["root_task_index"], "control root--task index"
        )
        if not 0 <= root_task_index < len(root_tasks):
            raise CapsuleError("An evaluation control has an out-of-range root--task index")
        expected = root_tasks[root_task_index]
        if (
            row["root_id"] != expected["root_id"]
            or row["task_id"] != expected["task_id"]
            or _integer(row["root_index"], "control root index")
            != _integer(expected["root_index"], "evaluation root index")
            or _integer(row["task_index"], "control task index")
            != _integer(expected["task_index"], "evaluation task index")
            or row["base_pool"] not in REFERENCE_POOLS
        ):
            raise CapsuleError("An evaluation control does not match its root--task row")

    if tuple(row["reference_instance"] for row in dispatch_rows) != REFERENCE_INSTANCES:
        raise CapsuleError("The reference instances are not in the expected order")
    for row in dispatch_rows:
        observed = (
            row["c_obs_pools"],
            row["c_pred_pools"],
            row["c_infer_pools"],
        )
        if observed != EXPECTED_DISPATCH[row["reference_instance"]]:
            raise CapsuleError(
                f"The pool assignment for {row['reference_instance']} is incorrect"
            )
    groups: dict[tuple[int, str], int] = {}
    for row in rows:
        key = (int(row["root_task_index"]), row["base_pool"])
        groups[key] = groups.get(key, 0) + 1
    expected_group_count = len(REFERENCE_POOLS) * len(root_tasks)
    if len(groups) != expected_group_count or any(
        count != CELLS_PER_REFERENCE_POOL for count in groups.values()
    ):
        raise CapsuleError("Each evaluation root--task pool must contain eight cells")
    expected_row_groups = [
        (root_task, pool)
        for root_task in range(len(root_tasks))
        for pool in REFERENCE_POOLS
        for _cell in range(CELLS_PER_REFERENCE_POOL)
    ]
    observed_row_groups = [
        (
            _integer(row["root_task_index"], "control root--task index"),
            row["base_pool"],
        )
        for row in rows
    ]
    if observed_row_groups != expected_row_groups:
        raise CapsuleError(
            "Evaluation controls must be ordered by root, task, pool and within-pool row"
        )
    return EvaluationControls(
        matrix=np.ascontiguousarray(matrix, dtype="<f4"),
        rows=rows,
        root_tasks=root_tasks,
        dispatch={row["reference_instance"]: row for row in dispatch_rows},
        feature_ids=feature_ids,
        scales=np.ascontiguousarray(scales),
        weights=np.ascontiguousarray(weights),
        root_ids=root_ids,
        task_ids=task_ids,
    )


def _pool_names(text: str) -> tuple[str, ...]:
    values = tuple(text.split("|"))
    if (
        not values
        or len(set(values)) != len(values)
        or not set(values) <= set(REFERENCE_POOLS)
    ):
        raise CapsuleError("A reference dispatch contains an invalid pool list")
    return values


def _selected_offsets(
    data: EvaluationControls, instance: str, role: str
) -> tuple[list[int], np.ndarray]:
    if role not in {"c_infer", "c_pred"}:
        raise CapsuleError("The prediction role must be c_infer or c_pred")
    pools = _pool_names(data.dispatch[instance][f"{role}_pools"])
    grouped: dict[tuple[int, str], list[int]] = {}
    for offset, row in enumerate(data.rows):
        grouped.setdefault((int(row["root_task_index"]), row["base_pool"]), []).append(
            offset
        )
    offsets: list[int] = []
    codes: list[int] = []
    for root_task in range(len(data.root_tasks)):
        selected = [offset for pool in pools for offset in grouped[(root_task, pool)]]
        offsets.extend(selected)
        codes.extend([root_task] * len(selected))
    return offsets, np.asarray(codes, dtype=np.int64)


def _fixed_group_means(
    values: np.ndarray, codes: np.ndarray, groups: int
) -> np.ndarray:
    result = np.empty((groups, values.shape[1]), dtype="<f4")
    for group in range(groups):
        selected = np.flatnonzero(codes == group)
        if selected.size == 0:
            raise CapsuleError("A prediction group is empty")
        accumulator = np.zeros(values.shape[1], dtype=np.float64)
        for offset in selected.tolist():
            accumulator += np.asarray(values[offset], dtype=np.float64)
        result[group] = accumulator / selected.size
    return result


def materialize_prediction_references(data: EvaluationControls) -> np.ndarray:
    """Reconstruct prediction-reference means from the prepared controls."""

    result = np.empty(
        (len(REFERENCE_INSTANCES), len(data.root_tasks), data.matrix.shape[1]),
        dtype="<f4",
    )
    for instance_index, instance in enumerate(REFERENCE_INSTANCES):
        offsets, codes = _selected_offsets(data, instance, "c_pred")
        result[instance_index] = _fixed_group_means(
            data.matrix[offsets], codes, len(data.root_tasks)
        )
    if not np.isfinite(result).all():
        raise CapsuleError("The reconstructed prediction-reference means are non-finite")
    return result


def predict_all_instances(
    model: Any,
    shifts: np.ndarray,
    data: EvaluationControls,
    *,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    import anndata
    import pandas as pd
    import torch

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise CapsuleError("The prediction batch size must be a positive integer")
    shift_values = np.asarray(shifts)
    if (
        shift_values.ndim != 2
        or shift_values.shape[0] != len(data.task_ids)
        or not np.isfinite(shift_values).all()
    ):
        raise CapsuleError("The latent-shift array has the wrong shape")
    device = next(model.module.parameters()).device
    native = np.empty(
        (len(REFERENCE_INSTANCES), len(data.root_tasks), data.matrix.shape[1]),
        dtype="<f4",
    )
    c_pred = materialize_prediction_references(data)
    for instance_index, instance in enumerate(REFERENCE_INSTANCES):
        offsets, codes = _selected_offsets(data, instance, "c_infer")
        values = np.ascontiguousarray(data.matrix[offsets], dtype="<f4")
        task_codes = np.asarray(
            [int(data.root_tasks[index]["task_index"]) for index in codes],
            dtype=np.int64,
        )
        observations = pd.DataFrame(
            {
                "arm": pd.Categorical(
                    ["NI"] * len(values), categories=["NI", "IAV"], ordered=True
                ),
                "task": pd.Categorical(
                    [data.task_ids[index] for index in task_codes],
                    categories=list(data.task_ids),
                    ordered=True,
                ),
            },
            index=pd.Index(
                [f"query_{instance_index}_{index}" for index in range(len(values))]
            ),
        )
        adata = anndata.AnnData(
            X=values,
            obs=observations,
            var=pd.DataFrame(index=pd.Index(data.feature_ids, name="feature_id")),
        )
        latent = np.asarray(
            model.get_latent_representation(
                adata=adata, give_mean=True, batch_size=batch_size
            ),
            dtype="<f4",
        )
        if latent.shape != (len(values), shift_values.shape[1]) or not np.isfinite(
            latent
        ).all():
            raise CapsuleError("scGen returned an invalid query latent matrix")
        latent = np.ascontiguousarray(
            latent + np.asarray(shift_values[task_codes], dtype="<f4"), dtype="<f4"
        )
        decoded = np.empty_like(values)
        with torch.inference_mode():
            for start in range(0, len(values), batch_size):
                stop = min(start + batch_size, len(values))
                tensor = torch.as_tensor(
                    latent[start:stop], dtype=torch.float32, device=device
                )
                generated = model.module.generative(tensor)
                if not isinstance(generated, Mapping) or "px" not in generated:
                    raise CapsuleError("The scGen decoder did not return a 'px' matrix")
                chunk = generated["px"].detach().cpu().numpy()
                if chunk.shape != (stop - start, data.matrix.shape[1]):
                    raise CapsuleError("The scGen decoder returned an unexpected shape")
                decoded[start:stop] = chunk
        if not np.isfinite(decoded).all():
            raise CapsuleError("scGen returned a non-finite decoded prediction")
        native[instance_index] = _fixed_group_means(
            decoded, codes, len(data.root_tasks)
        )

    return native, c_pred
