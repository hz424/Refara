"""Recalculate the focal empirical comparison from saved prediction arrays."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .io import CapsuleError, load_array, load_manifest, read_tsv
from .scoring import (
    EVALUATION_ROOTS,
    FEATURES,
    REFERENCE_INSTANCES,
    TASKS,
    focal_contrast,
    validate_axis,
)
from .training import EXPECTED_REALIZATIONS


ROOT_TASK_COLUMNS = (
    "root_task_index",
    "root_id",
    "root_index",
    "task_id",
    "task_index",
)
RESULT_FIELDS = (
    "shared_pca64_minus_scgen",
    "split_pca64_minus_scgen",
    "rotation_pca64_minus_scgen",
    "split_minus_shared_shift",
    "rotation_minus_shared_shift",
)
FOCAL_CONTRAST_COLUMNS = (
    "realization_id",
    "seed",
    "realization_role",
    "shared_pca64_minus_scgen",
    "expected_shared_pca64_minus_scgen",
    "shared_absolute_error",
    "split_pca64_minus_scgen",
    "expected_split_pca64_minus_scgen",
    "split_absolute_error",
    "rotation_pca64_minus_scgen",
    "expected_rotation_pca64_minus_scgen",
    "rotation_absolute_error",
    "split_minus_shared_shift",
    "rotation_minus_shared_shift",
    "direction_pattern_passed",
    "expected_value_check_passed",
)
MANIFEST_EXPECTED_ATOL = 5.0e-13


@dataclass(frozen=True)
class ReplayAssets:
    root: Path
    manifest: Mapping[str, Any]
    realizations: tuple[tuple[str, int], ...]
    axis: Any
    arrays: Mapping[str, np.ndarray]


def _integer(text: str, label: str) -> int:
    try:
        value = int(text)
    except (TypeError, ValueError, OverflowError) as error:
        raise CapsuleError(f"{label} must be an integer") from error
    if str(value) != text:
        raise CapsuleError(f"{label} must use canonical integer notation")
    return value


def _registered_realizations(
    manifest: Mapping[str, Any],
    expected_realizations: tuple[tuple[str, int], ...],
) -> tuple[tuple[str, int], ...]:
    rows = manifest.get("realizations")
    if not isinstance(rows, list) or len(rows) != 5:
        raise CapsuleError("The replay manifest must register five training realizations")
    result: list[tuple[str, int]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise CapsuleError("The realization registry is malformed")
        name, seed = row.get("id"), row.get("seed")
        if not isinstance(name, str) or not isinstance(seed, int) or isinstance(seed, bool):
            raise CapsuleError("A realization ID or seed is malformed")
        result.append((name, seed))
    if tuple(result) != expected_realizations:
        raise CapsuleError("The realization IDs or seeds differ from the reported analysis")
    return tuple(result)


def _root_task_axis(root: Path, manifest: Mapping[str, Any]):
    rows = read_tsv(root, manifest, "root_tasks", ROOT_TASK_COLUMNS)
    if [
        _integer(row["root_task_index"], "replay root--task index") for row in rows
    ] != list(range(len(rows))):
        raise CapsuleError("The root--task row order is not contiguous")
    root_order: list[str] = []
    task_order: list[str] = []
    for row in rows:
        if row["root_id"] not in root_order:
            root_order.append(row["root_id"])
        if row["task_id"] not in task_order:
            task_order.append(row["task_id"])
        if (
            _integer(row["root_index"], "replay root index")
            != root_order.index(row["root_id"])
            or _integer(row["task_index"], "replay task index")
            != task_order.index(row["task_id"])
        ):
            raise CapsuleError("The root--task labels and integer indices disagree")
    axis = validate_axis(
        [_integer(row["root_index"], "replay root index") for row in rows],
        [_integer(row["task_index"], "replay task index") for row in rows],
        root_order,
        task_order,
    )
    if (
        axis.root_ids != tuple(f"R{index:02d}" for index in range(41, 49))
        or axis.task_ids != tuple(f"T{index:02d}" for index in range(1, TASKS + 1))
    ):
        raise CapsuleError("The replay axis must use R41--R48 and T01--T05 in order")
    return axis


def read_expected(
    path: Path,
    expected_realizations: tuple[tuple[str, int], ...] = EXPECTED_REALIZATIONS,
) -> dict[str, dict[str, Any]]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise CapsuleError("The expected-result table is missing")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise CapsuleError("Could not read the expected-result table") from error
    expected_columns = (
        "realization_id",
        "seed",
        *RESULT_FIELDS,
        "direction_pattern_passed",
    )
    if tuple(reader.fieldnames or ()) != expected_columns:
        raise CapsuleError("The expected-result table has unexpected columns")
    if len(rows) != len(expected_realizations):
        raise CapsuleError("The expected-result table must contain five rows")
    if any(
        set(row) != set(expected_columns)
        or any(value is None for value in row.values())
        for row in rows
    ):
        raise CapsuleError("The expected-result table contains a malformed row")
    result: dict[str, dict[str, Any]] = {}
    for row, (expected_name, expected_seed) in zip(rows, expected_realizations):
        name = row["realization_id"]
        try:
            seed = int(row["seed"])
            values = {field: float(row[field]) for field in RESULT_FIELDS}
        except (TypeError, ValueError, OverflowError) as error:
            raise CapsuleError("The expected-result table contains a non-numeric value") from error
        direction_text = row["direction_pattern_passed"]
        if (
            name != expected_name
            or seed != expected_seed
            or str(seed) != row["seed"]
            or name in result
        ):
            raise CapsuleError("The expected realization IDs or seeds are incorrect")
        if not all(np.isfinite(value) for value in values.values()):
            raise CapsuleError("The expected-result table contains a non-finite value")
        if direction_text not in {"true", "false"}:
            raise CapsuleError("The expected direction check must be true or false")
        result[name] = {
            "realization_id": name,
            "seed": seed,
            **values,
            "direction_pattern_passed": direction_text == "true",
        }
    return result


def read_manifest_expected(
    root: Path,
    manifest: Mapping[str, Any],
    expected_realizations: tuple[tuple[str, int], ...] = EXPECTED_REALIZATIONS,
) -> dict[str, dict[str, Any]]:
    """Read acceptance values from the manifest-bound focal table."""

    rows = read_tsv(root, manifest, "focal_contrasts", FOCAL_CONTRAST_COLUMNS)
    if len(rows) != len(expected_realizations):
        raise CapsuleError("The focal contrast table must contain five rows")
    result: dict[str, dict[str, Any]] = {}
    for index, (row, (expected_name, expected_seed)) in enumerate(
        zip(rows, expected_realizations)
    ):
        name = row["realization_id"]
        try:
            seed = _integer(row["seed"], "focal contrast seed")
            actual = {
                "shared_pca64_minus_scgen": float(
                    row["shared_pca64_minus_scgen"]
                ),
                "split_pca64_minus_scgen": float(row["split_pca64_minus_scgen"]),
                "rotation_pca64_minus_scgen": float(
                    row["rotation_pca64_minus_scgen"]
                ),
            }
            expected = {
                "shared_pca64_minus_scgen": float(
                    row["expected_shared_pca64_minus_scgen"]
                ),
                "split_pca64_minus_scgen": float(
                    row["expected_split_pca64_minus_scgen"]
                ),
                "rotation_pca64_minus_scgen": float(
                    row["expected_rotation_pca64_minus_scgen"]
                ),
            }
            recorded_errors = {
                "shared_pca64_minus_scgen": float(row["shared_absolute_error"]),
                "split_pca64_minus_scgen": float(row["split_absolute_error"]),
                "rotation_pca64_minus_scgen": float(row["rotation_absolute_error"]),
            }
            recorded_shifts = (
                float(row["split_minus_shared_shift"]),
                float(row["rotation_minus_shared_shift"]),
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise CapsuleError("The focal contrast table contains a non-numeric value") from error
        numeric_values = (
            *actual.values(),
            *expected.values(),
            *recorded_errors.values(),
            *recorded_shifts,
        )
        expected_role = (
            "primary_analysis_realization" if index == 0 else "additional_realization"
        )
        if (
            name != expected_name
            or seed != expected_seed
            or name in result
            or row["realization_role"] != expected_role
            or not all(np.isfinite(value) for value in numeric_values)
            or row["direction_pattern_passed"] not in {"true", "false"}
            or row["expected_value_check_passed"] not in {"true", "false"}
        ):
            raise CapsuleError("The focal contrast registry is malformed")
        calculated_errors = {
            field: abs(actual[field] - expected[field])
            for field in actual
        }
        actual_shifts = (
            actual["split_pca64_minus_scgen"]
            - actual["shared_pca64_minus_scgen"],
            actual["rotation_pca64_minus_scgen"]
            - actual["shared_pca64_minus_scgen"],
        )
        actual_direction = (
            actual["shared_pca64_minus_scgen"] < 0
            and actual["split_pca64_minus_scgen"] > 0
            and actual["rotation_pca64_minus_scgen"] > 0
        )
        recorded_direction = row["direction_pattern_passed"] == "true"
        if (
            any(
                not np.isclose(
                    recorded_errors[field],
                    calculated_errors[field],
                    rtol=0.0,
                    atol=MANIFEST_EXPECTED_ATOL,
                )
                for field in actual
            )
            or any(
                not np.isclose(recorded, calculated, rtol=0.0, atol=MANIFEST_EXPECTED_ATOL)
                for recorded, calculated in zip(recorded_shifts, actual_shifts)
            )
            or recorded_direction is not actual_direction
            or row["expected_value_check_passed"] != "true"
            or any(error > MANIFEST_EXPECTED_ATOL for error in calculated_errors.values())
        ):
            raise CapsuleError("The focal contrast table contains inconsistent checks")
        expected_shifts = (
            expected["split_pca64_minus_scgen"]
            - expected["shared_pca64_minus_scgen"],
            expected["rotation_pca64_minus_scgen"]
            - expected["shared_pca64_minus_scgen"],
        )
        expected_direction = (
            expected["shared_pca64_minus_scgen"] < 0
            and expected["split_pca64_minus_scgen"] > 0
            and expected["rotation_pca64_minus_scgen"] > 0
        )
        if expected_direction is not recorded_direction:
            raise CapsuleError("The focal contrast direction check is inconsistent")
        result[name] = {
            "realization_id": name,
            "seed": seed,
            **expected,
            "split_minus_shared_shift": expected_shifts[0],
            "rotation_minus_shared_shift": expected_shifts[1],
            "direction_pattern_passed": expected_direction,
        }
    return result


def load_acceptance_values(
    assets: ReplayAssets,
    external_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load manifest-bound values and optionally verify a second copy."""

    expected_realizations = assets.realizations
    bound = read_manifest_expected(
        assets.root, assets.manifest, expected_realizations
    )
    if external_path is None:
        return bound
    external = read_expected(external_path, expected_realizations)
    for realization_id, _seed in expected_realizations:
        recorded = bound[realization_id]
        comparison = external[realization_id]
        if (
            recorded["seed"] != comparison["seed"]
            or recorded["direction_pattern_passed"]
            is not comparison["direction_pattern_passed"]
            or any(recorded[field] != comparison[field] for field in RESULT_FIELDS)
        ):
            raise CapsuleError(
                "The external expected-result table differs from the replay manifest"
            )
    return bound


def load_replay_assets(
    asset_root: Path,
    expected_realizations: tuple[tuple[str, int], ...] = EXPECTED_REALIZATIONS,
) -> ReplayAssets:
    root = Path(asset_root)
    manifest = load_manifest(root, "gse162632_scgen_focal_replay")
    if tuple(manifest.get("reference_instances", ())) != REFERENCE_INSTANCES:
        raise CapsuleError("The reference-instance order differs from the analysis")
    arrays = {
        name: load_array(root, manifest, name)
        for name in (
            "treated_means",
            "c_obs",
            "pca64_effect",
            "c_pred",
            "scales",
            "weights",
        )
    }
    rows = EVALUATION_ROOTS * TASKS
    if (
        arrays["treated_means"].shape != (rows, FEATURES)
        or arrays["c_obs"].shape != (len(REFERENCE_INSTANCES), rows, FEATURES)
        or arrays["pca64_effect"].shape != (rows, FEATURES)
        or arrays["c_pred"].shape
        != (len(REFERENCE_INSTANCES), rows, FEATURES)
        or arrays["scales"].shape != (FEATURES,)
        or arrays["weights"].shape != (FEATURES,)
        or np.any(arrays["scales"] < 0.1)
        or not np.all(arrays["weights"] == np.float64(1.0 / FEATURES))
        or not np.isclose(
            np.sum(arrays["weights"], dtype=np.float64),
            1.0,
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        raise CapsuleError("The replay arrays do not match the 40-row, 2000-feature panel")
    return ReplayAssets(
        root=root,
        manifest=manifest,
        realizations=_registered_realizations(manifest, expected_realizations),
        axis=_root_task_axis(root, manifest),
        arrays=arrays,
    )


def reproduce_focal(
    asset_root: Path,
    expected_path: Path | None = None,
    *,
    atol: float = 1.0e-12,
    expected_realizations: tuple[tuple[str, int], ...] = EXPECTED_REALIZATIONS,
) -> dict[str, Any]:
    """Verify the replay assets and recalculate all five focal contrasts."""

    if isinstance(atol, bool):
        raise CapsuleError("The comparison tolerance must be a non-negative finite number")
    try:
        tolerance = float(atol)
    except (TypeError, ValueError, OverflowError) as error:
        raise CapsuleError(
            "The comparison tolerance must be a non-negative finite number"
        ) from error
    if not np.isfinite(tolerance) or tolerance < 0:
        raise CapsuleError("The comparison tolerance must be a non-negative finite number")
    assets = load_replay_assets(asset_root, expected_realizations)
    root = assets.root
    manifest = assets.manifest
    realizations = assets.realizations
    expected = load_acceptance_values(assets, expected_path)
    axis = assets.axis
    shared = assets.arrays
    rows: list[dict[str, Any]] = []
    root_patterns: dict[str, dict[str, bool]] = {}
    for realization_id, seed in realizations:
        native = load_array(root, manifest, f"scgen_native_{realization_id}")
        if native.shape != (
            len(REFERENCE_INSTANCES),
            EVALUATION_ROOTS * TASKS,
            FEATURES,
        ):
            raise CapsuleError(
                f"The saved scGen predictions for {realization_id} have an unexpected shape"
            )
        result = focal_contrast(
            shared["treated_means"],
            shared["c_obs"],
            shared["pca64_effect"],
            native,
            shared["c_pred"],
            shared["scales"],
            shared["weights"],
            axis,
        )
        reference = expected[realization_id]
        if reference["seed"] != seed:
            raise CapsuleError(f"The seed differs for {realization_id}")
        for field in RESULT_FIELDS:
            if not np.isclose(
                float(result[field]), reference[field], rtol=0.0, atol=tolerance
            ):
                raise CapsuleError(
                    f"The recalculated {field} differs for {realization_id}"
                )
        if bool(result["direction_pattern_passed"]) is not reference[
            "direction_pattern_passed"
        ]:
            raise CapsuleError(f"The direction check differs for {realization_id}")
        root_contrasts = np.asarray(result.pop("root_contrasts"), dtype=np.float64)
        root_patterns[realization_id] = {
            "shared_negative_in_all_roots": bool(np.all(root_contrasts[:, 0] < 0)),
            "split_positive_in_all_roots": bool(np.all(root_contrasts[:, 1] > 0)),
            "rotation_positive_in_all_roots": bool(np.all(root_contrasts[:, 2] > 0)),
        }
        rows.append({"realization_id": realization_id, "seed": seed, **result})

    all_five_passed = all(bool(row["direction_pattern_passed"]) for row in rows)
    q1_all_roots = all(root_patterns["q1"].values())
    if not all_five_passed or not q1_all_roots:
        raise CapsuleError("The recalculated direction checks did not pass")
    return {
        "status": "PASS",
        "capsule_id": manifest.get("capsule_id"),
        "release_version": manifest["release_version"],
        "metric": "negative standardized mean squared error",
        "reference_instances": list(REFERENCE_INSTANCES),
        "realizations": rows,
        "root_direction_checks": root_patterns,
        "all_five_direction_patterns_passed": all_five_passed,
        "q1_reversed_in_all_eight_roots": q1_all_roots,
        "acceptance_values": "manifest-bound focal_contrasts.tsv",
        "external_expected_table_checked": expected_path is not None,
        "comparison_tolerance": {"absolute": tolerance, "relative": 0.0},
    }
