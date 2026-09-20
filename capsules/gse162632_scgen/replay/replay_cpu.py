#!/usr/bin/env python3
"""Recompute the five focal scGen contrasts from pseudonymized capsule arrays."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REFERENCE_INSTANCES = (
    "D8_NAIVE_SHARED",
    "D8_INDEPENDENT_SPLIT",
    "D8_CROSSFIT_R0",
    "D8_CROSSFIT_R1",
    "D8_CROSSFIT_R2",
)
CONSTRUCTIONS = ("shared", "split", "rotation")
REALIZATIONS = ("q1", "q2", "q3", "q4", "q5")
ROOTS = 8
TASKS = 5
ROOT_TASKS = ROOTS * TASKS
FEATURES = 2_000
TOLERANCE = 5.0e-13


class ReplayError(RuntimeError):
    """A capsule identity or numerical check failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(bundle: Path) -> Mapping[str, Any]:
    path = bundle / "manifest.json"
    require(path.is_file() and not path.is_symlink(), "manifest.json is absent")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "manifest.json must contain a JSON object")
    records = value.get("files")
    require(isinstance(records, list) and records, "manifest file records are absent")
    for record in records:
        require(isinstance(record, dict), "manifest record is not an object")
        relative = record.get("path")
        require(
            isinstance(relative, str)
            and relative
            and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts,
            "manifest record path is unsafe",
        )
        artifact = bundle / relative
        require(
            artifact.is_file()
            and not artifact.is_symlink()
            and artifact.stat().st_size == record.get("bytes")
            and sha256_file(artifact) == record.get("sha256"),
            f"file size or SHA-256 mismatch: {relative}",
        )
    return value


def load_array(
    bundle: Path,
    relative: str,
    shape: tuple[int, ...],
    dtype: str = "<f8",
) -> np.ndarray:
    values = np.load(bundle / relative, allow_pickle=False)
    require(
        values.shape == shape
        and values.dtype == np.dtype(dtype)
        and values.flags.c_contiguous
        and np.isfinite(values).all(),
        f"array has an unexpected shape, dtype or layout, or contains non-finite values: {relative}",
    )
    return values


def standardized_mse(
    observed: np.ndarray,
    predicted: np.ndarray,
    scales: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Compute equal-feature standardized MSE in fixed 256-feature chunks."""

    require(
        observed.shape == predicted.shape == (ROOT_TASKS, FEATURES),
        "effect geometry differs",
    )
    output = np.zeros(ROOT_TASKS, dtype=np.float64)
    for start in range(0, FEATURES, 256):
        stop = min(start + 256, FEATURES)
        residual = (observed[:, start:stop] - predicted[:, start:stop]) / scales[
            start:stop
        ]
        output += np.sum(
            np.square(residual) * weights[start:stop],
            axis=1,
            dtype=np.float64,
        )
    require(np.isfinite(output).all(), "standardized MSE is nonfinite")
    return output


def read_root_tasks(bundle: Path) -> tuple[np.ndarray, np.ndarray]:
    with (bundle / "root_tasks.tsv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require(
            tuple(reader.fieldnames or ())
            == ("root_task_index", "root_id", "root_index", "task_id", "task_index"),
            "root_tasks.tsv header differs",
        )
        rows = list(reader)
    require(len(rows) == ROOT_TASKS, "root-task row count differs")
    root_index = np.asarray([int(row["root_index"]) for row in rows], dtype=np.int64)
    task_index = np.asarray([int(row["task_index"]) for row in rows], dtype=np.int64)
    for offset, row in enumerate(rows):
        require(
            int(row["root_task_index"]) == offset
            and row["root_id"] == f"R{offset // TASKS + 41:02d}"
            and row["task_id"] == f"T{offset % TASKS + 1:02d}"
            and int(row["root_index"]) == offset // TASKS
            and int(row["task_index"]) == offset % TASKS,
            "pseudonymized root-task axis differs",
        )
    return root_index, task_index


def construction_utilities(instance_utilities: np.ndarray) -> np.ndarray:
    require(
        instance_utilities.shape == (len(REFERENCE_INSTANCES), ROOT_TASKS),
        "instance utility geometry differs",
    )
    return np.stack(
        (
            instance_utilities[0],
            instance_utilities[1],
            np.mean(instance_utilities[2:5], axis=0, dtype=np.float64),
        ),
        axis=0,
    )


def expected_focal(bundle: Path) -> Mapping[str, Mapping[str, float]]:
    with (bundle / "focal_contrasts.tsv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    require(len(rows) == len(REALIZATIONS), "focal contrast row count differs")
    output: dict[str, dict[str, float]] = {}
    for expected_id, row in zip(REALIZATIONS, rows):
        require(row.get("realization_id") == expected_id, "realization order differs")
        output[expected_id] = {
            construction: float(row[f"expected_{construction}_pca64_minus_scgen"])
            for construction in CONSTRUCTIONS
        }
    return output


def calculate(bundle: Path) -> dict[str, Any]:
    manifest = load_manifest(bundle)
    axes = manifest.get("axes")
    require(
        isinstance(axes, dict)
        and tuple(axes.get("reference_instances", ())) == REFERENCE_INSTANCES,
        "reference-instance axis differs",
    )
    treated = load_array(bundle, "arrays/treated_means.npy", (ROOT_TASKS, FEATURES))
    c_obs = load_array(
        bundle,
        "arrays/c_obs.npy",
        (len(REFERENCE_INSTANCES), ROOT_TASKS, FEATURES),
    )
    pca64 = load_array(
        bundle, "arrays/pca64_effect.npy", (ROOT_TASKS, FEATURES), "<f4"
    )
    c_pred = load_array(
        bundle,
        "arrays/c_pred.npy",
        (len(REFERENCE_INSTANCES), ROOT_TASKS, FEATURES),
        "<f4",
    )
    scales = load_array(bundle, "arrays/scales.npy", (FEATURES,))
    weights = load_array(bundle, "arrays/weights.npy", (FEATURES,))
    require(
        np.all(scales > 0.0)
        and np.all(weights > 0.0)
        and abs(float(np.sum(weights, dtype=np.float64)) - 1.0) < 1.0e-12,
        "scale or weight contract differs",
    )
    root_index, task_index = read_root_tasks(bundle)
    expected = expected_focal(bundle)
    records: list[dict[str, Any]] = []
    maximum_error = 0.0
    for realization_id in REALIZATIONS:
        native = load_array(
            bundle,
            f"arrays/scgen_native_{realization_id}.npy",
            (len(REFERENCE_INSTANCES), ROOT_TASKS, FEATURES),
            "<f4",
        )
        pca_instance = np.empty((len(REFERENCE_INSTANCES), ROOT_TASKS), dtype=np.float64)
        scgen_instance = np.empty_like(pca_instance)
        for instance_index in range(len(REFERENCE_INSTANCES)):
            observed_effect = treated - c_obs[instance_index]
            pca_instance[instance_index] = -standardized_mse(
                observed_effect, pca64, scales, weights
            )
            scgen_instance[instance_index] = -standardized_mse(
                observed_effect,
                native[instance_index] - c_pred[instance_index],
                scales,
                weights,
            )
        pca_construction = construction_utilities(pca_instance)
        scgen_construction = construction_utilities(scgen_instance)
        contrast = pca_construction - scgen_construction
        focal: dict[str, float] = {}
        focal_expected: dict[str, float] = {}
        errors: dict[str, float] = {}
        for construction_index, construction in enumerate(CONSTRUCTIONS):
            cube = np.empty((ROOTS, TASKS), dtype=np.float64)
            cube[root_index, task_index] = contrast[construction_index]
            root_values = np.mean(cube, axis=1, dtype=np.float64)
            value = float(np.mean(root_values, dtype=np.float64))
            target = expected[realization_id][construction]
            error = abs(value - target)
            maximum_error = max(maximum_error, error)
            focal[construction] = value
            focal_expected[construction] = target
            errors[construction] = error
        records.append(
            {
                "realization_id": realization_id,
                "role": (
                    "primary_analysis_realization"
                    if realization_id == "q1"
                    else "additional_realization"
                ),
                "focal_pca64_minus_scgen": focal,
                "expected_focal_pca64_minus_scgen": focal_expected,
                "absolute_error": errors,
            }
        )
    require(maximum_error <= TOLERANCE, "focal replay differs from expected values")
    return {
        "status": "PASS",
        "metric": "negative_equal_feature_standardized_mse",
        "aggregation": "task mean within root, then equal root mean; rotation utilities averaged after scoring",
        "maximum_absolute_error_vs_expected": maximum_error,
        "tolerance": TOLERANCE,
        "realizations": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="CPU replay bundle directory (defaults to the script directory)",
    )
    args = parser.parse_args()
    result = calculate(args.bundle.resolve())
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
