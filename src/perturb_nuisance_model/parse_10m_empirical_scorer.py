"""Frozen Parse 10M eight-model empirical scoring primitives.

The module has one outcome-bearing reader.  It can be reached only after the
caller validates an exact-SHA complete-family eligibility PASS.  The reader
opens only registered EVALUATION-treated CSR rows, in disjoint hyperslabs, and
immediately reduces them to the frozen 5,776 root-task means.

Inference is not reimplemented here.  The exact frozen generic V1.0 source is
hash checked and imported, and its contrast family, sign schedule, root
contrasts, joint max-|T| procedure, and decision-set function are called
directly.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import csv
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping, Sequence

import h5py
import numpy as np
from scipy import sparse

from perturb_nuisance_model.parse_10m_pipeline import (
    ABSOLUTE_METHOD_IDS,
    DIRECT_METHOD_IDS,
    METHOD_IDS,
    REFERENCE_INSTANCE_IDS,
)
from perturb_nuisance_model.parse_complete_family_eligibility import (
    PASS_STATUS as ELIGIBILITY_PASS_STATUS,
    RECORD_TYPE as ELIGIBILITY_RECORD_TYPE,
    contrast_axis as eligibility_contrast_axis,
)


SCORER_ID = "PARSE_10M_EIGHT_MODEL_EMPIRICAL_SCORER_V1"
PASS_STATUS = "PASS_FROZEN_PARSE_10M_EIGHT_MODEL_EMPIRICAL_ANALYSIS_V1"
SCHEMES = ("NAIVE_SHARED", "INDEPENDENT_SPLIT", "UNIT_AWARE_CROSSFIT")
DEPTHS = (16, 32, 48)
PRIMARY_DEPTH = 32
ROOTS = 8
TASKS = 722
ROOT_TASKS = ROOTS * TASKS
FEATURES = 2_000
METHODS = 8
CONTRASTS = 84
EVALUATION_TREATED_CELLS = 4_864_623
SOURCE_SHAPE = (9_697_974, 40_352)
CP10K_TOTAL = 10_000.0
CELL_AXIS_HEADER = (
    "row_index",
    "obs_index",
    "donor",
    "cytokine",
    "treatment",
    "cell_type",
    "analysis_role",
    "cell_role",
    "task_id",
)
ROOT_TASK_HEADER = (
    "root_task_offset",
    "control_group_id",
    "task_index",
    "task_id",
    "source_root_index",
    "root_id",
    "donor",
    "cytokine",
    "cell_type",
    "perturbed_cell_count",
    "matched_pbs_cell_count",
    "task_weight",
    "analysis_role",
    "analysis_root_index",
    "analysis_root_weight",
)
ROLE_HEADER = (
    "reference_instance_id",
    "depth",
    "scheme",
    "rotation",
    "role",
    "control_group_id",
    "donor",
    "cell_type",
    "role_cell_rank",
    "control_offset",
    "block",
    "block_rank",
    "row_index",
    "obs_index",
)


def reference_role_cell_count(reference_instance_id: str) -> int:
    """Return the frozen per-role cell count for one reference instance."""

    if reference_instance_id not in REFERENCE_INSTANCE_IDS:
        raise ParseEmpiricalScorerError("reference instance is not registered")
    depth = int(reference_instance_id.split("_", 1)[0][1:])
    return (
        3 * depth
        if reference_instance_id.endswith("_NAIVE_SHARED")
        else depth
    )


class ParseEmpiricalScorerError(RuntimeError):
    """A frozen empirical input or numerical invariant failed."""


@dataclass(frozen=True)
class EligibilityGate:
    """Validated, hash-bound complete-family eligibility state."""

    report_sha256: str
    input_paths: Mapping[str, Path]
    input_sha256: Mapping[str, str]


@dataclass(frozen=True)
class RootTaskAxis:
    """Exact scorer row order and root/task dispatch."""

    task_index: np.ndarray
    root_index: np.ndarray
    task_id: tuple[str, ...]
    root_id_by_index: tuple[str, ...]
    donor: tuple[str, ...]
    cytokine: tuple[str, ...]
    cell_type: tuple[str, ...]
    control_group_id: tuple[str, ...]
    perturbed_count: np.ndarray


@dataclass(frozen=True)
class TreatedMembership:
    """Physical source rows and their root-task output offsets."""

    row_indices: np.ndarray
    root_task_offsets: np.ndarray
    source_cell_axis_rows: int


@dataclass(frozen=True)
class TreatedMeans:
    """Root-task means and exact-reader audit."""

    values: np.ndarray
    counts: np.ndarray
    access_audit: Mapping[str, Any]


@dataclass(frozen=True)
class EvalBundle:
    """Only scorer-consumed axes and C_obs numeric values."""

    report_sha256: str
    root: Path
    root_tasks: RootTaskAxis
    panel_source_indices: np.ndarray
    feature_ids: tuple[str, ...]
    feature_weights: np.ndarray
    feature_scales: np.ndarray
    c_obs_by_instance: Mapping[str, np.ndarray]


def sha256_file(path: Path, *, block_size: int = 16 * 1024 * 1024) -> str:
    """Return a whole-file SHA-256 without following symlinks."""

    value = Path(path)
    if value.is_symlink() or not value.is_file():
        raise ParseEmpiricalScorerError(f"not a regular file: {value}")
    digest = hashlib.sha256()
    with value.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Return canonical JSON bytes, rejecting NaN and infinity."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ParseEmpiricalScorerError(f"{label} is not lowercase SHA-256")
    return value


def _text(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\0" in value
        or "\n" in value
        or "\r" in value
    ):
        raise ParseEmpiricalScorerError(f"{label} is not canonical text")
    return value


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ParseEmpiricalScorerError(f"{label} is not an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ParseEmpiricalScorerError(f"{label} is not an integer") from error
    if result < minimum or str(result) != str(value):
        raise ParseEmpiricalScorerError(
            f"{label} is not a canonical integer >= {minimum}"
        )
    return result


def _load_json(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[dict[str, Any], str]:
    expected = _sha256(expected_sha256, label=f"{label} expected SHA-256")
    observed = sha256_file(path)
    if observed != expected:
        raise ParseEmpiricalScorerError(f"{label} SHA-256 differs")
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ParseEmpiricalScorerError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ParseEmpiricalScorerError(f"{label} must be a JSON object")
    return value, observed


def _resolve_path(project_root: Path, value: object, *, label: str) -> Path:
    text = _text(value, label=label)
    candidate = Path(text)
    if candidate.is_absolute():
        return candidate
    if ".." in candidate.parts:
        raise ParseEmpiricalScorerError(f"{label} is unsafe")
    return Path(project_root) / candidate


def validate_contract(
    contract: Mapping[str, Any],
    *,
    project_root: Path,
) -> None:
    """Validate the frozen scientific dimensions and all static bindings."""

    if (
        contract.get("schema_version") != 1
        or contract.get("scorer_id") != SCORER_ID
        or contract.get("status")
        != "FROZEN_UNEXECUTED_OUTCOME_BEARING_APPLICATION"
        or contract.get("single_scientific_execution") is not True
        or contract.get("runtime_complete_family_eligibility_sha256_required")
        is not True
    ):
        raise ParseEmpiricalScorerError("scorer contract identity differs")
    geometry = contract.get("geometry")
    expected_geometry = {
        "tasks": TASKS,
        "roots": ROOTS,
        "root_tasks": ROOT_TASKS,
        "methods": METHODS,
        "features": FEATURES,
        "reference_schemes": 3,
        "reference_instances": 15,
        "crossfit_rotations": 3,
        "within_scheme_pairs": 28,
        "joint_contrasts": CONTRASTS,
        "evaluation_treated_cells": EVALUATION_TREATED_CELLS,
    }
    if geometry != expected_geometry:
        raise ParseEmpiricalScorerError("scorer geometry differs")
    if tuple(contract.get("method_order", ())) != tuple(METHOD_IDS):
        raise ParseEmpiricalScorerError("method order differs")
    reference = contract.get("reference_design")
    if (
        not isinstance(reference, Mapping)
        or tuple(reference.get("depths", ())) != DEPTHS
        or reference.get("primary_depth") != PRIMARY_DEPTH
        or tuple(reference.get("scheme_order", ())) != SCHEMES
        or tuple(reference.get("reference_instance_order", ()))
        != tuple(REFERENCE_INSTANCE_IDS)
        or reference.get("crossfit_rule")
        != (
            "COMPUTE_NONLINEAR_UTILITY_SEPARATELY_FOR_R0_R1_R2_"
            "THEN_EQUAL_AVERAGE_UTILITIES"
        )
        or reference.get("task_weight") != "1/722"
        or reference.get("root_weight") != "1/8"
        or reference.get("C_cal") != "DISABLED"
    ):
        raise ParseEmpiricalScorerError("reference design differs")
    metric = contract.get("metric")
    if (
        not isinstance(metric, Mapping)
        or metric.get("larger_is_better") is not True
        or metric.get("feature_weights") != "EQUAL_1_OVER_2000"
        or metric.get("scales")
        != "FROZEN_TRAIN_PBS_DDOF1_SD_WITH_0_1_FLOOR"
    ):
        raise ParseEmpiricalScorerError("metric contract differs")
    inference = contract.get("inference")
    if (
        not isinstance(inference, Mapping)
        or inference.get("procedure") != "CURRENT_JOINT_MAXT"
        or inference.get("alpha") != 0.05
        or inference.get("sign_schedule")
        != "EXHAUSTIVE_ALL_256_EIGHT_ROOT_SIGN_VECTORS"
    ):
        raise ParseEmpiricalScorerError("inference contract differs")
    static_inputs = contract.get("static_inputs")
    if not isinstance(static_inputs, Mapping):
        raise ParseEmpiricalScorerError("static input bindings are absent")
    for label, binding in static_inputs.items():
        if not isinstance(binding, Mapping):
            raise ParseEmpiricalScorerError(f"{label} binding is not an object")
        path = _resolve_path(project_root, binding.get("path"), label=f"{label} path")
        if label == "h5ad":
            expected_size = binding.get("size_bytes")
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != expected_size
            ):
                raise ParseEmpiricalScorerError("H5AD size or identity differs")
            continue
        if sha256_file(path) != _sha256(
            binding.get("sha256"), label=f"{label} SHA-256"
        ):
            raise ParseEmpiricalScorerError(f"{label} SHA-256 differs")


def validate_eligibility_gate(
    path: Path,
    *,
    expected_sha256: str,
    contract: Mapping[str, Any],
    project_root: Path,
) -> EligibilityGate:
    """Validate the sole gate that may precede treated-expression access."""

    report, digest = _load_json(
        path,
        expected_sha256=expected_sha256,
        label="complete-family eligibility report",
    )
    expected_geometry = {
        "tasks": TASKS,
        "roots": ROOTS,
        "root_tasks": ROOT_TASKS,
        "features": FEATURES,
        "methods": METHODS,
        "states": METHODS,
        "reference_instances": len(REFERENCE_INSTANCE_IDS),
        "absolute_method_instance_pairs": len(ABSOLUTE_METHOD_IDS)
        * len(REFERENCE_INSTANCE_IDS),
        "within_scheme_method_pairs": 28,
        "schemes": len(SCHEMES),
        "joint_within_scheme_contrasts": CONTRASTS,
    }
    boundary = report.get("access_boundary")
    artifact = report.get("artifact_audit")
    replay = report.get("replay")
    if (
        report.get("record_type") != ELIGIBILITY_RECORD_TYPE
        or report.get("status") != ELIGIBILITY_PASS_STATUS
        or report.get("complete_family_eligible") is not True
        or report.get("scoring_performed") is not False
        or report.get("failures") != []
        or report.get("geometry") != expected_geometry
        or tuple(report.get("method_order", ())) != tuple(METHOD_IDS)
        or tuple(report.get("reference_instance_order", ()))
        != tuple(REFERENCE_INSTANCE_IDS)
        or report.get("contrast_axis") != list(eligibility_contrast_axis())
        or not isinstance(artifact, Mapping)
        or artifact.get(
            "all_hashes_axes_semantics_and_once_only_subtractions_verified"
        )
        is not True
        or not isinstance(replay, Mapping)
        or replay.get("passed") is not True
        or not isinstance(boundary, Mapping)
        or boundary.get("h5ad_opened") is not False
        or boundary.get("evaluation_treated_expression_read") is not False
        or boundary.get("evaluation_perturbed_expression_read") is not False
        or boundary.get("scores_utilities_contrasts_intervals_computed") is not False
        or boundary.get("winner_best_set_or_rank_computed") is not False
        or boundary.get("metric_or_scorer_imported") is not False
    ):
        raise ParseEmpiricalScorerError(
            "complete-family eligibility is not an exact PASS"
        )

    bindings = report.get("input_bindings")
    expected_keys = (
        "panel",
        "direct_family",
        "eval_pbs_bundle",
        "cpa_scgen_family",
        "cellot_family",
    )
    if not isinstance(bindings, Mapping) or set(bindings) != set(expected_keys):
        raise ParseEmpiricalScorerError("eligibility input bindings differ")
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for key in expected_keys:
        binding = bindings[key]
        if not isinstance(binding, Mapping):
            raise ParseEmpiricalScorerError(f"eligibility {key} binding is absent")
        bound_path = _resolve_path(
            project_root, binding.get("path"), label=f"eligibility {key} path"
        )
        bound_hash = _sha256(
            binding.get("expected_sha256"), label=f"eligibility {key} SHA-256"
        )
        if sha256_file(bound_path) != bound_hash:
            raise ParseEmpiricalScorerError(
                f"eligibility-bound {key} changed after audit"
            )
        paths[key] = bound_path
        hashes[key] = bound_hash

    static_inputs = contract["static_inputs"]
    for key, contract_key in (
        ("panel", "model_panel"),
        ("direct_family", "direct_family_report"),
    ):
        contract_binding = static_inputs[contract_key]
        expected_path = _resolve_path(
            project_root,
            contract_binding["path"],
            label=f"{contract_key} path",
        ).resolve()
        if (
            paths[key].resolve() != expected_path
            or hashes[key] != contract_binding["sha256"]
        ):
            raise ParseEmpiricalScorerError(
                f"eligibility {key} differs from the frozen contract"
            )
    axes = report.get("axis_bindings")
    if (
        not isinstance(axes, Mapping)
        or axes.get("eval_bundle_report_sha256") != hashes["eval_pbs_bundle"]
    ):
        raise ParseEmpiricalScorerError("eligibility axis binding differs")
    return EligibilityGate(
        report_sha256=digest,
        input_paths=paths,
        input_sha256=hashes,
    )


def gated_treated_read(
    gate_builder: Callable[[], EligibilityGate],
    treated_reader: Callable[[EligibilityGate], Any],
) -> Any:
    """Make the eligibility-before-outcome ordering explicit and testable."""

    gate = gate_builder()
    if not isinstance(gate, EligibilityGate):
        raise ParseEmpiricalScorerError("eligibility check did not return an EligibilityGate")
    return treated_reader(gate)


def load_frozen_generic(
    source_path: Path,
    *,
    expected_sha256: str,
) -> ModuleType:
    """Hash check and import the exact generic V1.0 implementation."""

    expected = _sha256(expected_sha256, label="generic source SHA-256")
    if sha256_file(source_path) != expected:
        raise ParseEmpiricalScorerError("generic inference source SHA-256 differs")
    module_name = "_parse_frozen_generic_reference_framework_v1_0"
    prior = sys.modules.get(module_name)
    if prior is not None:
        if (
            Path(getattr(prior, "__file__", "")).resolve()
            != Path(source_path).resolve()
        ):
            raise ParseEmpiricalScorerError("generic module name was preoccupied")
        return prior
    specification = importlib.util.spec_from_file_location(module_name, source_path)
    if specification is None or specification.loader is None:
        raise ParseEmpiricalScorerError("generic source cannot be imported")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    required = (
        "contrast_family",
        "sign_schedule",
        "sign_schedule_sha256",
        "root_contrasts",
        "infer_current_joint_maxt",
        "decision_sets",
    )
    if any(not callable(getattr(module, name, None)) for name in required):
        raise ParseEmpiricalScorerError("generic inference API differs")
    return module


def validate_generic_qualification(
    report_path: Path,
    *,
    expected_sha256: str,
) -> None:
    """Require all 11 registered R8/M8 cells to have passed."""

    report, _ = _load_json(
        report_path,
        expected_sha256=expected_sha256,
        label="generic framework qualification report",
    )
    boundary = report.get("access_boundary")
    scenarios = report.get("scenarios")
    if (
        report.get("framework_id")
        != "GENERIC_REFERENCE_AWARE_ROOT_INFERENCE_V1_0"
        or not isinstance(boundary, Mapping)
        or boundary.get("dataset_opened") is not False
        or boundary.get("expression_or_counts_opened") is not False
        or boundary.get("predictions_opened") is not False
        or not isinstance(scenarios, list)
    ):
        raise ParseEmpiricalScorerError("generic qualification identity differs")
    selected = [
        row
        for row in scenarios
        if isinstance(row, Mapping)
        and isinstance(row.get("geometry"), Mapping)
        and row["geometry"].get("root_count") == ROOTS
        and row["geometry"].get("method_count") == METHODS
    ]
    if (
        len(selected) != 11
        or len({row.get("scenario_id") for row in selected}) != 11
        or any(row.get("joint_validity_pass") is not True for row in selected)
        or any(
            row.get("geometry", {}).get("joint_family_size") != CONTRASTS
            for row in selected
        )
    ):
        raise ParseEmpiricalScorerError(
            "generic qualification lacks the complete R8/M8 PASS"
        )


def _open_tsv(path: Path):
    return (
        gzip.open(path, "rt", encoding="utf-8", newline="")
        if path.suffix == ".gz"
        else path.open("rt", encoding="utf-8", newline="")
    )


def _artifact_path(
    root: Path,
    record: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    relative = Path(_text(record.get("path"), label=f"{label} path"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ParseEmpiricalScorerError(f"{label} artifact path is unsafe")
    path = Path(root) / relative
    expected_size = (
        _integer(record.get("size_bytes"), label=f"{label} size", minimum=1)
        if "size_bytes" in record
        else None
    )
    if (
        path.is_symlink()
        or not path.is_file()
        or (
            expected_size is not None
            and path.stat().st_size != expected_size
        )
        or sha256_file(path)
        != _sha256(record.get("sha256"), label=f"{label} SHA-256")
    ):
        raise ParseEmpiricalScorerError(f"{label} artifact identity differs")
    return path


def load_root_task_axis(path: Path) -> RootTaskAxis:
    """Load the exact 5,776-row bundle root-task axis."""

    task_index = np.empty(ROOT_TASKS, dtype=np.int64)
    root_index = np.empty(ROOT_TASKS, dtype=np.int64)
    perturbed_count = np.empty(ROOT_TASKS, dtype=np.int64)
    task_ids: list[str] = []
    donors: list[str] = []
    cytokines: list[str] = []
    cell_types: list[str] = []
    control_groups: list[str] = []
    root_id_by_index: list[str | None] = [None] * ROOTS
    with _open_tsv(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != ROOT_TASK_HEADER:
            raise ParseEmpiricalScorerError("root-task axis header differs")
        for offset, row in enumerate(reader):
            if offset >= ROOT_TASKS or row["root_task_offset"] != str(offset):
                raise ParseEmpiricalScorerError("root-task offset order differs")
            task = _integer(row["task_index"], label="task index")
            root = _integer(row["analysis_root_index"], label="root index")
            if (
                task not in range(TASKS)
                or root not in range(ROOTS)
                or row["task_weight"] != "1/722"
                or row["analysis_root_weight"] != "1/8"
                or row["analysis_role"] != "EVALUATION"
                or row["root_id"] != f"ROOT_{row['donor']}"
            ):
                raise ParseEmpiricalScorerError("root-task scientific axis differs")
            task_index[offset] = task
            root_index[offset] = root
            perturbed_count[offset] = _integer(
                row["perturbed_cell_count"],
                label="perturbed cell count",
                minimum=1,
            )
            task_ids.append(_text(row["task_id"], label="task ID"))
            donors.append(_text(row["donor"], label="donor"))
            cytokines.append(_text(row["cytokine"], label="cytokine"))
            cell_types.append(_text(row["cell_type"], label="cell type"))
            control_groups.append(
                _text(row["control_group_id"], label="control group")
            )
            prior = root_id_by_index[root]
            if prior is None:
                root_id_by_index[root] = row["root_id"]
            elif prior != row["root_id"]:
                raise ParseEmpiricalScorerError("root index is not functional")
    if len(task_ids) != ROOT_TASKS:
        raise ParseEmpiricalScorerError("root-task row count differs")
    keys = {(donors[i], task_ids[i]) for i in range(ROOT_TASKS)}
    if (
        len(keys) != ROOT_TASKS
        or np.any(np.bincount(task_index, minlength=TASKS) != ROOTS)
        or np.any(np.bincount(root_index, minlength=ROOTS) != TASKS)
        or any(value is None for value in root_id_by_index)
    ):
        raise ParseEmpiricalScorerError("root-task support is incomplete")
    task_index.setflags(write=False)
    root_index.setflags(write=False)
    perturbed_count.setflags(write=False)
    return RootTaskAxis(
        task_index=task_index,
        root_index=root_index,
        task_id=tuple(task_ids),
        root_id_by_index=tuple(str(value) for value in root_id_by_index),
        donor=tuple(donors),
        cytokine=tuple(cytokines),
        cell_type=tuple(cell_types),
        control_group_id=tuple(control_groups),
        perturbed_count=perturbed_count,
    )


def load_treated_membership(
    cell_axis_path: Path,
    *,
    expected_sha256: str,
    expected_rows: int,
    root_tasks: RootTaskAxis,
) -> TreatedMembership:
    """Map registered EVALUATION-treated cells to root-task offsets."""

    if sha256_file(cell_axis_path) != _sha256(
        expected_sha256, label="cell-axis SHA-256"
    ):
        raise ParseEmpiricalScorerError("cell-axis SHA-256 differs")
    key_to_offset = {
        (root_tasks.donor[offset], root_tasks.task_id[offset]): offset
        for offset in range(ROOT_TASKS)
    }
    rows = array("q")
    offsets = array("i")
    input_rows = 0
    prior_row = -1
    with _open_tsv(cell_axis_path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != CELL_AXIS_HEADER:
            raise ParseEmpiricalScorerError("cell-axis header differs")
        for raw in reader:
            input_rows += 1
            row_index = _integer(raw["row_index"], label="cell row index")
            if row_index <= prior_row or row_index >= SOURCE_SHAPE[0]:
                raise ParseEmpiricalScorerError("cell-axis row order differs")
            prior_row = row_index
            if (
                raw["analysis_role"] == "EVALUATION"
                and raw["cell_role"] == "EVALUATION_PERTURBED"
            ):
                key = (raw["donor"], raw["task_id"])
                offset = key_to_offset.get(key)
                if (
                    offset is None
                    or raw["treatment"] != "cytokine"
                    or raw["cytokine"] != root_tasks.cytokine[offset]
                    or raw["cell_type"] != root_tasks.cell_type[offset]
                ):
                    raise ParseEmpiricalScorerError(
                        "treated cell does not map to the frozen root-task axis"
                    )
                rows.append(row_index)
                offsets.append(offset)
    row_array = np.frombuffer(rows, dtype=np.int64).copy()
    offset_array = np.frombuffer(offsets, dtype=np.int32).astype(
        np.int64, copy=True
    )
    observed_counts = np.bincount(offset_array, minlength=ROOT_TASKS)
    if (
        input_rows != expected_rows
        or row_array.size != EVALUATION_TREATED_CELLS
        or offset_array.shape != row_array.shape
        or np.any(row_array[1:] <= row_array[:-1])
        or not np.array_equal(observed_counts, root_tasks.perturbed_count)
    ):
        raise ParseEmpiricalScorerError(
            "registered treated-cell support differs from the frozen axis"
        )
    row_array.setflags(write=False)
    offset_array.setflags(write=False)
    return TreatedMembership(
        row_indices=row_array,
        root_task_offsets=offset_array,
        source_cell_axis_rows=input_rows,
    )


def _require_hard_dataset(group: h5py.Group, name: str) -> h5py.Dataset:
    link = group.get(name, getlink=True)
    node = group.get(name)
    if not isinstance(link, h5py.HardLink) or not isinstance(node, h5py.Dataset):
        raise ParseEmpiricalScorerError(f"required /X/{name} dataset is absent")
    return node


def _read_disjoint_hyperslabs(
    dataset: h5py.Dataset,
    starts: np.ndarray,
    stops: np.ndarray,
) -> np.ndarray:
    if starts.ndim != 1 or starts.shape != stops.shape or starts.size < 1:
        raise ParseEmpiricalScorerError("CSR hyperslab request is malformed")
    lengths = stops - starts
    if np.any(lengths <= 0) or np.any(starts[1:] < stops[:-1]):
        raise ParseEmpiricalScorerError("CSR hyperslabs overlap or are empty")
    total = int(lengths.sum(dtype=np.int64))
    output = np.empty(total, dtype=dataset.dtype)
    file_space = dataset.id.get_space()
    file_space.select_none()
    for start, length in zip(starts.tolist(), lengths.tolist(), strict=True):
        file_space.select_hyperslab(
            start=(int(start),),
            count=(int(length),),
            op=h5py.h5s.SELECT_OR,
        )
    memory_space = h5py.h5s.create_simple((total,))
    dataset.id.read(memory_space, file_space, output)
    return output


def _batch_slices_from_lengths(
    lengths: np.ndarray,
    *,
    target_nnz: int,
) -> Iterable[slice]:
    if lengths.ndim != 1 or lengths.size < 1 or target_nnz < 1:
        raise ParseEmpiricalScorerError("selected-row batch geometry is invalid")
    cursor = 0
    while cursor < lengths.size:
        total = 0
        stop = cursor
        while stop < lengths.size and (total == 0 or total < target_nnz):
            total += int(lengths[stop])
            stop += 1
        yield slice(cursor, stop)
        cursor = stop


def _merge_consecutive_row_intervals(
    rows: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge adjacent selected physical rows into fewer HDF5 selections."""

    if rows.ndim != 1 or rows.shape != starts.shape or rows.shape != stops.shape:
        raise ParseEmpiricalScorerError("row-run geometry differs")
    run_start: list[int] = [int(starts[0])]
    run_stop: list[int] = []
    for index in range(1, rows.size):
        if int(rows[index]) != int(rows[index - 1]) + 1:
            run_stop.append(int(stops[index - 1]))
            run_start.append(int(starts[index]))
    run_stop.append(int(stops[-1]))
    result_start = np.asarray(run_start, dtype=np.int64)
    result_stop = np.asarray(run_stop, dtype=np.int64)
    if int(np.sum(result_stop - result_start, dtype=np.int64)) != int(
        np.sum(stops - starts, dtype=np.int64)
    ):
        raise ParseEmpiricalScorerError("merged CSR runs changed selected NNZ")
    return result_start, result_stop


def aggregate_registered_treated_means(
    h5ad_path: Path,
    *,
    expected_size_bytes: int,
    membership: TreatedMembership,
    panel_source_indices: np.ndarray,
    source_shape: tuple[int, int] = SOURCE_SHAPE,
    root_tasks: int = ROOT_TASKS,
    target_nnz: int = 64_000_000,
) -> TreatedMeans:
    """Read only registered treated rows and immediately aggregate by root-task."""

    if (
        Path(h5ad_path).is_symlink()
        or not Path(h5ad_path).is_file()
        or Path(h5ad_path).stat().st_size != expected_size_bytes
    ):
        raise ParseEmpiricalScorerError("H5AD identity or size differs")
    panel = np.asarray(panel_source_indices, dtype=np.int64)
    if (
        panel.shape != (FEATURES,)
        and source_shape == SOURCE_SHAPE
        or panel.ndim != 1
        or panel.size < 1
        or np.any(panel < 0)
        or np.any(panel[1:] <= panel[:-1])
        or int(panel[-1]) >= source_shape[1]
    ):
        raise ParseEmpiricalScorerError("panel source-index axis differs")
    rows = np.asarray(membership.row_indices, dtype=np.int64)
    codes = np.asarray(membership.root_task_offsets, dtype=np.int64)
    if (
        rows.ndim != 1
        or rows.shape != codes.shape
        or rows.size < 1
        or np.any(rows < 0)
        or np.any(rows[1:] <= rows[:-1])
        or int(rows[-1]) >= source_shape[0]
        or np.any(codes < 0)
        or np.any(codes >= root_tasks)
    ):
        raise ParseEmpiricalScorerError("treated membership geometry differs")

    sums = np.zeros((root_tasks, panel.size), dtype=np.float64)
    counts = np.zeros(root_tasks, dtype=np.int64)
    returned_nnz = 0
    requested_runs = 0
    batches = 0
    physical_row_runs = 1 + int(np.sum(np.diff(rows) != 1))
    with h5py.File(h5ad_path, "r") as handle:
        x_link = handle.get("/X", getlink=True)
        x_group = handle.get("/X")
        if not isinstance(x_link, h5py.HardLink) or not isinstance(
            x_group, h5py.Group
        ):
            raise ParseEmpiricalScorerError("H5AD /X is not a direct CSR group")
        shape = tuple(int(value) for value in np.asarray(x_group.attrs.get("shape")))
        if shape != tuple(source_shape):
            raise ParseEmpiricalScorerError("H5AD /X shape differs")
        data_ds = _require_hard_dataset(x_group, "data")
        indices_ds = _require_hard_dataset(x_group, "indices")
        indptr_ds = _require_hard_dataset(x_group, "indptr")
        if (
            data_ds.ndim != 1
            or data_ds.shape != indices_ds.shape
            or indptr_ds.shape != (shape[0] + 1,)
        ):
            raise ParseEmpiricalScorerError("H5AD CSR member geometry differs")
        endpoints = np.unique(np.concatenate((rows, rows + 1)))
        endpoint_values = np.asarray(indptr_ds[endpoints], dtype=np.int64)
        start_positions = np.searchsorted(endpoints, rows)
        stop_positions = np.searchsorted(endpoints, rows + 1)
        if (
            not np.array_equal(endpoints[start_positions], rows)
            or not np.array_equal(endpoints[stop_positions], rows + 1)
        ):
            raise ParseEmpiricalScorerError(
                "registered CSR endpoints are incomplete"
            )
        starts = np.asarray(endpoint_values[start_positions], dtype=np.int64)
        stops = np.asarray(endpoint_values[stop_positions], dtype=np.int64)
        lengths = stops - starts
        if (
            np.any(lengths <= 0)
            or np.any(starts < 0)
            or np.any(stops > data_ds.shape[0])
            or np.any(starts[1:] < stops[:-1])
        ):
            raise ParseEmpiricalScorerError("selected treated CSR intervals differ")

        for batch in _batch_slices_from_lengths(lengths, target_nnz=target_nnz):
            batch_rows = rows[batch]
            batch_codes = codes[batch]
            batch_starts = starts[batch]
            batch_stops = stops[batch]
            run_starts, run_stops = _merge_consecutive_row_intervals(
                batch_rows, batch_starts, batch_stops
            )
            data = np.asarray(
                _read_disjoint_hyperslabs(data_ds, run_starts, run_stops),
                dtype=np.float64,
            )
            indices = np.asarray(
                _read_disjoint_hyperslabs(indices_ds, run_starts, run_stops),
                dtype=np.int64,
            )
            batch_lengths = batch_stops - batch_starts
            indptr = np.empty(batch_lengths.size + 1, dtype=np.int64)
            indptr[0] = 0
            np.cumsum(batch_lengths, dtype=np.int64, out=indptr[1:])
            raw = sparse.csr_matrix(
                (data, indices, indptr),
                shape=(batch_lengths.size, shape[1]),
                copy=False,
            )
            if (
                not raw.has_canonical_format
                or not np.isfinite(raw.data).all()
                or np.any(raw.data < 0)
                or np.any(raw.data != np.floor(raw.data))
            ):
                raise ParseEmpiricalScorerError(
                    "registered treated counts violate numeric admission"
                )
            totals = np.asarray(raw.sum(axis=1), dtype=np.float64).reshape(-1)
            if np.any(~np.isfinite(totals)) or np.any(totals <= 0):
                raise ParseEmpiricalScorerError(
                    "registered treated full-axis total is invalid"
                )
            projected = raw[:, panel].astype(np.float64, copy=True)
            projected.data *= np.repeat(
                CP10K_TOTAL / totals, np.diff(projected.indptr)
            )
            np.log1p(projected.data, out=projected.data)
            assignment = sparse.csr_matrix(
                (
                    np.ones(batch_codes.size, dtype=np.float64),
                    (batch_codes, np.arange(batch_codes.size, dtype=np.int64)),
                ),
                shape=(root_tasks, batch_codes.size),
            )
            partial = (assignment @ projected).tocsr()
            active = np.unique(batch_codes)
            sums[active] += np.asarray(partial[active].toarray(), dtype=np.float64)
            counts += np.bincount(batch_codes, minlength=root_tasks)
            returned_nnz += int(data.size)
            requested_runs += int(run_starts.size)
            batches += 1
    if np.any(counts <= 0) or not np.isfinite(sums).all():
        raise ParseEmpiricalScorerError("treated root-task aggregation is incomplete")
    means = sums / counts[:, None]
    if not np.isfinite(means).all():
        raise ParseEmpiricalScorerError("treated root-task mean is nonfinite")
    means.setflags(write=False)
    counts.setflags(write=False)
    return TreatedMeans(
        values=means,
        counts=counts,
        access_audit={
            "counts_path_opened": "/X",
            "obs_opened": False,
            "source_shape": list(source_shape),
            "registered_rows_requested": int(rows.size),
            "registered_rows_returned": int(counts.sum()),
            "unregistered_numeric_rows_returned": 0,
            "registered_data_elements_returned": returned_nnz,
            "registered_index_elements_returned": returned_nnz,
            "disjoint_physical_row_runs": physical_row_runs,
            "hyperslab_run_selections": requested_runs,
            "hyperslab_batches": batches,
            "cell_level_outcomes_persisted": False,
            "aggregation_output": f"{root_tasks}_ROOT_TASK_MEANS",
        },
    )


def _load_npy_artifact(
    root: Path,
    record: Mapping[str, Any],
    *,
    label: str,
    expected_shape: tuple[int, ...],
    expected_dtype: str,
    mmap: bool = True,
) -> np.ndarray:
    path = _artifact_path(root, record, label=label)
    if record.get("shape") != list(expected_shape) or record.get(
        "dtype"
    ) != expected_dtype:
        raise ParseEmpiricalScorerError(f"{label} NPY record differs")
    value = np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
    if (
        value.shape != expected_shape
        or value.dtype.str != expected_dtype
        or not np.isfinite(value).all()
    ):
        raise ParseEmpiricalScorerError(f"{label} NPY payload differs")
    return value


def load_eval_bundle(
    report_path: Path,
    *,
    expected_sha256: str,
    static_scales_path: Path,
    static_scales_sha256: str,
    static_weights_path: Path,
    static_weights_sha256: str,
) -> EvalBundle:
    """Load exact axes, scales and C_obs means from the eligible PBS bundle."""

    report, report_sha = _load_json(
        report_path,
        expected_sha256=expected_sha256,
        label="EVAL-PBS bundle report",
    )
    if (
        report.get("record_type") != "PARSE_10M_EVALUATION_PBS_CONTROL_BUNDLE_V1"
        or report.get("status") != "PASS_EVALUATION_PBS_ONLY_NO_PREDICTION_OR_SCORE"
        or report.get("production_geometry") is not True
    ):
        raise ParseEmpiricalScorerError("EVAL-PBS bundle identity differs")
    root = Path(report_path).resolve().parent
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ParseEmpiricalScorerError("EVAL-PBS artifacts are absent")
    transformed = _load_npy_artifact(
        root,
        artifacts["transformed_panel"],
        label="EVAL control transformed panel",
        expected_shape=(12_672, FEATURES),
        expected_dtype="<f4",
    )
    panel_indices = _load_npy_artifact(
        root,
        artifacts["panel_source_indices"],
        label="panel source indices",
        expected_shape=(FEATURES,),
        expected_dtype="<i8",
        mmap=False,
    )
    root_task_path = _artifact_path(
        root, artifacts["root_task_axis"], label="root-task axis"
    )
    root_tasks = load_root_task_axis(root_task_path)
    panel_path = _artifact_path(
        root, artifacts["panel_features"], label="panel feature axis"
    )
    feature_ids: list[str] = []
    panel_weights: list[float] = []
    panel_scales: list[float] = []
    with _open_tsv(panel_path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected_header = (
            "panel_index",
            "source_feature_index",
            "feature_id",
            "variance_selection_rank",
            "feature_weight",
            "transformed_mean",
            "unfloored_variance",
            "unfloored_sd",
            "floored_scale",
        )
        if tuple(reader.fieldnames or ()) != expected_header:
            raise ParseEmpiricalScorerError("panel feature header differs")
        for index, row in enumerate(reader):
            if row["panel_index"] != str(index):
                raise ParseEmpiricalScorerError("panel feature order differs")
            feature_ids.append(_text(row["feature_id"], label="feature ID"))
            panel_weights.append(float(row["feature_weight"]))
            panel_scales.append(float(row["floored_scale"]))
    weights_from_axis = np.asarray(panel_weights, dtype=np.float64)
    scales_from_axis = np.asarray(panel_scales, dtype=np.float64)
    if (
        len(feature_ids) != FEATURES
        or len(set(feature_ids)) != FEATURES
        or not np.isfinite(weights_from_axis).all()
        or not np.isfinite(scales_from_axis).all()
    ):
        raise ParseEmpiricalScorerError("panel metric axis differs")
    if sha256_file(static_scales_path) != _sha256(
        static_scales_sha256, label="static scale SHA-256"
    ) or sha256_file(static_weights_path) != _sha256(
        static_weights_sha256, label="static weight SHA-256"
    ):
        raise ParseEmpiricalScorerError("static metric artifacts changed")
    static_scales = np.load(static_scales_path, allow_pickle=False)
    static_weights = np.load(static_weights_path, allow_pickle=False)
    if (
        static_scales.shape != (FEATURES,)
        or static_scales.dtype.str != "<f8"
        or static_weights.shape != (FEATURES,)
        or static_weights.dtype.str != "<f8"
        or not np.array_equal(static_scales, scales_from_axis)
        or not np.array_equal(static_weights, weights_from_axis)
        or np.any(static_scales < 0.1)
        or not np.all(static_weights == 1.0 / FEATURES)
        or not math.isclose(
            math.fsum(static_weights.tolist()), 1.0, rel_tol=0.0, abs_tol=2e-14
        )
    ):
        raise ParseEmpiricalScorerError("frozen scales or equal weights differ")

    role_path = _artifact_path(
        root, artifacts["role_memberships"], label="reference role memberships"
    )
    offsets: dict[tuple[str, str], list[tuple[int, int]]] = {}
    with _open_tsv(role_path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != ROLE_HEADER:
            raise ParseEmpiricalScorerError("role-membership header differs")
        for row in reader:
            if row["role"] != "C_obs":
                continue
            instance = _text(
                row["reference_instance_id"], label="reference instance"
            )
            group = _text(row["control_group_id"], label="control group")
            rank = _integer(row["role_cell_rank"], label="role-cell rank")
            control_offset = _integer(
                row["control_offset"], label="control offset"
            )
            offsets.setdefault((instance, group), []).append(
                (rank, control_offset)
            )
    groups = tuple(sorted(set(root_tasks.control_group_id)))
    c_obs: dict[str, np.ndarray] = {}
    for instance in REFERENCE_INSTANCE_IDS:
        role_cell_count = reference_role_cell_count(instance)
        values = np.empty((len(groups), FEATURES), dtype=np.float64)
        for group_index, group in enumerate(groups):
            members = offsets.get((instance, group))
            if (
                members is None
                or len(members) != role_cell_count
                or [rank for rank, _ in members]
                != list(range(role_cell_count))
            ):
                raise ParseEmpiricalScorerError(
                    f"C_obs membership differs: {instance}/{group}"
                )
            member_offsets = np.asarray(
                [offset for _, offset in members], dtype=np.int64
            )
            if np.any(member_offsets < 0) or np.any(member_offsets >= 12_672):
                raise ParseEmpiricalScorerError("C_obs offset is invalid")
            values[group_index] = np.mean(
                np.asarray(transformed[member_offsets], dtype=np.float64),
                axis=0,
                dtype=np.float64,
            )
        values.setflags(write=False)
        c_obs[instance] = values
    panel_indices = np.asarray(panel_indices, dtype=np.int64)
    panel_indices.setflags(write=False)
    static_weights = np.asarray(static_weights, dtype=np.float64)
    static_scales = np.asarray(static_scales, dtype=np.float64)
    static_weights.setflags(write=False)
    static_scales.setflags(write=False)
    return EvalBundle(
        report_sha256=report_sha,
        root=root,
        root_tasks=root_tasks,
        panel_source_indices=panel_indices,
        feature_ids=tuple(feature_ids),
        feature_weights=static_weights,
        feature_scales=static_scales,
        c_obs_by_instance=c_obs,
    )


def load_direct_predictions(
    report_path: Path,
    *,
    expected_sha256: str,
) -> Mapping[str, np.ndarray]:
    """Load the five exact DIRECT task-effect matrices."""

    report, _ = _load_json(
        report_path,
        expected_sha256=expected_sha256,
        label="DIRECT family report",
    )
    if (
        report.get("record_type")
        != "PARSE_10M_FIVE_DIRECT_STATE_PREDICTION_FAMILY_V1"
        or report.get("status")
        != "PASS_PARSE_10M_FIVE_DIRECT_STATE_PREDICTION_FAMILY_V1"
        or report.get("external_output") is None
    ):
        raise ParseEmpiricalScorerError("DIRECT family identity differs")
    root = Path(report["external_output"])
    records = report.get("predictions")
    if not isinstance(records, list) or tuple(
        row.get("method_id") for row in records if isinstance(row, Mapping)
    ) != tuple(DIRECT_METHOD_IDS):
        raise ParseEmpiricalScorerError("DIRECT prediction family differs")
    result: dict[str, np.ndarray] = {}
    for record in records:
        if (
            record.get("prediction_semantics") != "DIRECT"
            or record.get("output_type") != "PREDICTED_EFFECT"
            or record.get("C_pred_subtractions") != 0
            or record.get("uses_C_infer") is not False
        ):
            raise ParseEmpiricalScorerError("DIRECT prediction semantics differ")
        translated = {
            "path": record["file"],
            "sha256": record["file_sha256"],
            "size_bytes": record["size_bytes"],
            "dtype": record["dtype"],
            "shape": record["shape"],
        }
        result[record["method_id"]] = _load_npy_artifact(
            root,
            translated,
            label=f"DIRECT/{record['method_id']}",
            expected_shape=(TASKS, FEATURES),
            expected_dtype="<f4",
        )
    return result


def _load_completion_prediction(
    completion_path: Path,
    *,
    expected_completion_sha256: str,
    method_id: str,
    reference_instance_id: str,
) -> np.ndarray:
    completion, _ = _load_json(
        completion_path,
        expected_sha256=expected_completion_sha256,
        label=f"{method_id}/{reference_instance_id} PRIMARY completion",
    )
    semantic = completion.get("semantic_contract")
    artifacts = completion.get("artifacts")
    cellot = method_id == ABSOLUTE_METHOD_IDS[2]
    common_invalid = (
        completion.get("method_id") != method_id
        or completion.get("reference_instance_id") != reference_instance_id
        or completion.get("run_label") != "PRIMARY"
        or completion.get("production_geometry") is not True
        or completion.get("geometry", {}).get("root_tasks") != ROOT_TASKS
        or completion.get("geometry", {}).get("features") != FEATURES
        or not isinstance(semantic, Mapping)
        or not isinstance(artifacts, Mapping)
    )
    if common_invalid:
        raise ParseEmpiricalScorerError(
            f"{method_id}/{reference_instance_id} identity differs"
        )
    cpa_scgen_valid = (
        semantic.get("prediction_semantics") == "ABSOLUTE"
        and semantic.get("C_pred_subtractions") == 1
        and semantic.get("predicted_effect")
        == "ABSOLUTE_NATIVE_MINUS_C_PRED_MEAN"
    )
    cellot_valid = (
        completion.get("record_type")
        == "PARSE_CELLOT_REFERENCE_INSTANCE_PREDICTIONS_V1"
        and completion.get("status")
        == "PASS_CELLOT_SCORE_FREE_REFERENCE_INSTANCE_PREDICTIONS_V1"
        and semantic.get("input_role_exposed_to_predictor") == ["C_infer"]
        and semantic.get("C_pred_values_exposed_to_predictor") is False
        and semantic.get("native_output") == "ABSOLUTE_NATIVE_FROZEN_PANEL"
        and semantic.get("C_pred_subtractions") == 1
        and semantic.get("post_prediction_clipping") is False
    )
    cpa_scgen_valid = (
        completion.get("record_type")
        == "PARSE_10M_CPA_SCGEN_REFERENCE_INSTANCE_PREDICTIONS_V1"
        and completion.get("status")
        == "PASS_REFERENCE_INSTANCE_PREDICTIONS_NO_SCORING"
        and cpa_scgen_valid
    )
    if (cellot and not cellot_valid) or (
        not cellot and not cpa_scgen_valid
    ):
        raise ParseEmpiricalScorerError(
            f"{method_id}/{reference_instance_id} semantics differ"
        )
    return _load_npy_artifact(
        Path(completion_path).resolve().parent,
        artifacts["predicted_effect"],
        label=f"{method_id}/{reference_instance_id} predicted effect",
        expected_shape=(ROOT_TASKS, FEATURES),
        expected_dtype="<f4",
    )


def load_absolute_predictions(
    cpa_scgen_report_path: Path,
    *,
    cpa_scgen_sha256: str,
    cellot_report_path: Path,
    cellot_sha256: str,
) -> Mapping[tuple[str, str], np.ndarray]:
    """Load only eligible PRIMARY predicted-effect matrices."""

    cpa_report, _ = _load_json(
        cpa_scgen_report_path,
        expected_sha256=cpa_scgen_sha256,
        label="CPA/scGen prediction-family report",
    )
    if (
        cpa_report.get("record_type")
        != "PARSE_10M_CPA_SCGEN_COMPLETE_PREDICTION_FAMILY_V1"
        or cpa_report.get("status")
        != (
            "PASS_COMPLETE_CPA_SCGEN_30_METHOD_INSTANCE_"
            "PREDICTION_FAMILY_WITH_REPLAY_V1"
        )
        or cpa_report.get("replay", {}).get("passed") is not True
    ):
        raise ParseEmpiricalScorerError("CPA/scGen family identity differs")
    records = cpa_report.get("records")
    expected_pairs = tuple(
        (method, instance)
        for method in ABSOLUTE_METHOD_IDS[:2]
        for instance in REFERENCE_INSTANCE_IDS
    )
    if not isinstance(records, list) or tuple(
        (row.get("method_id"), row.get("reference_instance_id"))
        for row in records
        if isinstance(row, Mapping)
    ) != expected_pairs:
        raise ParseEmpiricalScorerError("CPA/scGen PRIMARY family differs")

    result: dict[tuple[str, str], np.ndarray] = {}
    for row in records:
        method = row["method_id"]
        instance = row["reference_instance_id"]
        result[(method, instance)] = _load_completion_prediction(
            Path(row["primary_completion_record"]),
            expected_completion_sha256=row["primary_completion_sha256"],
            method_id=method,
            reference_instance_id=instance,
        )

    cellot_report, _ = _load_json(
        cellot_report_path,
        expected_sha256=cellot_sha256,
        label="CellOT prediction-family report",
    )
    if (
        cellot_report.get("record_type")
        != "PARSE_CELLOT_COMPLETE_PREDICTION_FAMILY_V1"
        or cellot_report.get("status")
        != (
            "PASS_COMPLETE_CELLOT_15_INSTANCE_"
            "PREDICTION_FAMILY_WITH_REPLAY_V1"
        )
        or cellot_report.get("method_id") != ABSOLUTE_METHOD_IDS[2]
        or cellot_report.get("replay", {}).get("passed") is not True
    ):
        raise ParseEmpiricalScorerError("CellOT family identity differs")
    instances = cellot_report.get("instances")
    if not isinstance(instances, list) or tuple(
        row.get("reference_instance_id")
        for row in instances
        if isinstance(row, Mapping)
    ) != tuple(REFERENCE_INSTANCE_IDS):
        raise ParseEmpiricalScorerError("CellOT PRIMARY family differs")
    method = ABSOLUTE_METHOD_IDS[2]
    for row in instances:
        instance = row["reference_instance_id"]
        result[(method, instance)] = _load_completion_prediction(
            Path(row["primary_completion_record"]),
            expected_completion_sha256=row["primary_completion_sha256"],
            method_id=method,
            reference_instance_id=instance,
        )
    if tuple(result) != tuple(
        (method, instance)
        for method in ABSOLUTE_METHOD_IDS
        for instance in REFERENCE_INSTANCE_IDS
    ):
        raise ParseEmpiricalScorerError("absolute prediction family differs")
    return result


def standardized_squared_utility(
    observed_effect: np.ndarray,
    predicted_effect: np.ndarray,
    *,
    feature_weights: np.ndarray,
    feature_scales: np.ndarray,
    feature_chunk: int = 256,
) -> np.ndarray:
    """Negative equal-feature standardized squared error, row by row."""

    observed = np.asarray(observed_effect)
    predicted = np.asarray(predicted_effect)
    weights = np.asarray(feature_weights, dtype=np.float64)
    scales = np.asarray(feature_scales, dtype=np.float64)
    if (
        observed.ndim != 2
        or observed.shape != predicted.shape
        or observed.shape[1] != weights.size
        or weights.shape != scales.shape
        or feature_chunk < 1
        or not np.isfinite(observed).all()
        or not np.isfinite(predicted).all()
        or not np.isfinite(weights).all()
        or not np.isfinite(scales).all()
        or np.any(weights <= 0)
        or np.any(scales <= 0)
    ):
        raise ParseEmpiricalScorerError("metric input geometry differs")
    output = np.zeros(observed.shape[0], dtype=np.float64)
    for start in range(0, observed.shape[1], feature_chunk):
        stop = min(start + feature_chunk, observed.shape[1])
        residual = (
            np.asarray(observed[:, start:stop], dtype=np.float64)
            - np.asarray(predicted[:, start:stop], dtype=np.float64)
        ) / scales[start:stop]
        output -= np.sum(
            np.square(residual) * weights[start:stop],
            axis=1,
            dtype=np.float64,
        )
    if not np.isfinite(output).all():
        raise ParseEmpiricalScorerError("metric output is nonfinite")
    return output


def compute_instance_utilities(
    treated_means: np.ndarray,
    *,
    root_tasks: RootTaskAxis,
    c_obs_by_instance: Mapping[str, np.ndarray],
    direct_predictions: Mapping[str, np.ndarray],
    absolute_predictions: Mapping[tuple[str, str], np.ndarray],
    feature_weights: np.ndarray,
    feature_scales: np.ndarray,
) -> Mapping[str, np.ndarray]:
    """Calculate root-task utilities for all 15 eligible reference instances."""

    treated = np.asarray(treated_means, dtype=np.float64)
    if treated.shape != (ROOT_TASKS, FEATURES) or not np.isfinite(treated).all():
        raise ParseEmpiricalScorerError("treated mean tensor differs")
    if tuple(direct_predictions) != tuple(DIRECT_METHOD_IDS):
        raise ParseEmpiricalScorerError("DIRECT method order differs")
    if tuple(absolute_predictions) != tuple(
        (method, instance)
        for method in ABSOLUTE_METHOD_IDS
        for instance in REFERENCE_INSTANCE_IDS
    ):
        raise ParseEmpiricalScorerError("ABSOLUTE method-instance order differs")
    group_order = tuple(sorted(set(root_tasks.control_group_id)))
    group_index = {value: index for index, value in enumerate(group_order)}
    dispatch = np.asarray(
        [group_index[value] for value in root_tasks.control_group_id],
        dtype=np.int64,
    )
    result: dict[str, np.ndarray] = {}
    for instance in REFERENCE_INSTANCE_IDS:
        controls = c_obs_by_instance.get(instance)
        if controls is None or controls.shape != (len(group_order), FEATURES):
            raise ParseEmpiricalScorerError(f"C_obs tensor differs: {instance}")
        observed = treated - controls[dispatch]
        values = np.empty((ROOT_TASKS, METHODS), dtype=np.float64)
        for method_index, method in enumerate(DIRECT_METHOD_IDS):
            values[:, method_index] = standardized_squared_utility(
                observed,
                direct_predictions[method][root_tasks.task_index],
                feature_weights=feature_weights,
                feature_scales=feature_scales,
            )
        for absolute_index, method in enumerate(ABSOLUTE_METHOD_IDS):
            values[:, len(DIRECT_METHOD_IDS) + absolute_index] = (
                standardized_squared_utility(
                    observed,
                    absolute_predictions[(method, instance)],
                    feature_weights=feature_weights,
                    feature_scales=feature_scales,
                )
            )
        values.setflags(write=False)
        result[instance] = values
    return result


def aggregate_scheme_root_utilities(
    instance_utilities: Mapping[str, np.ndarray],
    *,
    root_index: np.ndarray,
    task_index: np.ndarray,
) -> Mapping[int, np.ndarray]:
    """Aggregate tasks and cross-fit rotations after nonlinear scoring."""

    if tuple(instance_utilities) != tuple(REFERENCE_INSTANCE_IDS):
        raise ParseEmpiricalScorerError("reference-instance utility order differs")
    root_axis = np.asarray(root_index, dtype=np.int64)
    task_axis = np.asarray(task_index, dtype=np.int64)
    if (
        root_axis.shape != (ROOT_TASKS,)
        or task_axis.shape != (ROOT_TASKS,)
        or len(set(zip(root_axis.tolist(), task_axis.tolist(), strict=True)))
        != ROOT_TASKS
    ):
        raise ParseEmpiricalScorerError("root-task dispatch differs")
    cubes: dict[str, np.ndarray] = {}
    for instance in REFERENCE_INSTANCE_IDS:
        rows = np.asarray(instance_utilities[instance], dtype=np.float64)
        if rows.shape != (ROOT_TASKS, METHODS) or not np.isfinite(rows).all():
            raise ParseEmpiricalScorerError(
                f"reference-instance utility tensor differs: {instance}"
            )
        cube = np.empty((ROOTS, TASKS, METHODS), dtype=np.float64)
        cube[root_axis, task_axis] = rows
        cubes[instance] = cube
    output: dict[int, np.ndarray] = {}
    for depth in DEPTHS:
        values = np.empty((ROOTS, len(SCHEMES), METHODS), dtype=np.float64)
        values[:, 0, :] = np.mean(
            cubes[f"D{depth}_NAIVE_SHARED"], axis=1, dtype=np.float64
        )
        values[:, 1, :] = np.mean(
            cubes[f"D{depth}_INDEPENDENT_SPLIT"], axis=1, dtype=np.float64
        )
        rotation_utilities = np.stack(
            [
                cubes[f"D{depth}_CROSSFIT_R{rotation}"]
                for rotation in range(3)
            ],
            axis=0,
        )
        # The arrays entering this mean are already nonlinear utilities.
        values[:, 2, :] = np.mean(
            np.mean(rotation_utilities, axis=2, dtype=np.float64),
            axis=0,
            dtype=np.float64,
        )
        values.setflags(write=False)
        output[depth] = values
    return output


def _finite_or_none(value: float) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def _point_ranking(values: np.ndarray) -> list[dict[str, Any]]:
    utilities = np.asarray(values, dtype=np.float64)
    if utilities.shape != (METHODS,) or not np.isfinite(utilities).all():
        raise ParseEmpiricalScorerError("point-ranking utility vector differs")
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
    *,
    root_ids: Sequence[str],
    generic: ModuleType,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Call the exact generic V1.0 84-contrast inference and decisions."""

    values = np.asarray(root_utilities, dtype=np.float64)
    if (
        values.shape != (ROOTS, len(SCHEMES), METHODS)
        or not np.isfinite(values).all()
        or tuple(root_ids) == ()
        or len(tuple(root_ids)) != ROOTS
    ):
        raise ParseEmpiricalScorerError("root utility tensor differs")
    family = generic.contrast_family(METHODS)
    signs = generic.sign_schedule(ROOTS)
    if family.size != CONTRASTS or signs.shape != (256, ROOTS):
        raise ParseEmpiricalScorerError("generic family or sign schedule differs")
    root_contrast = generic.root_contrasts(values[None, ...], family)
    inference = generic.infer_current_joint_maxt(
        root_contrast,
        signs,
        alpha=alpha,
    )
    decisions = generic.decision_sets(
        inference["lower"],
        inference["upper"],
        family,
        np.zeros((len(SCHEMES), METHODS), dtype=np.float64),
    )
    point_utilities = np.mean(values, axis=0, dtype=np.float64)
    utility_se = np.std(values, axis=0, ddof=1) / math.sqrt(ROOTS)
    intervals: list[dict[str, Any]] = []
    for index, (scheme_index, left, right) in enumerate(family.identities):
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
                "standard_error": float(
                    inference["standard_errors"][0, index]
                ),
                "lower": _finite_or_none(inference["lower"][0, index]),
                "upper": _finite_or_none(inference["upper"][0, index]),
                "resolved_direction": (
                    "A_GREATER"
                    if inference["lower"][0, index] > 0
                    else (
                        "B_GREATER"
                        if inference["upper"][0, index] < 0
                        else "UNRESOLVED"
                    )
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
    point_records = [
        {
            "scheme": scheme,
            "method_id": method,
            "point_utility": float(
                point_utilities[scheme_index, method_index]
            ),
            "root_standard_error": float(
                utility_se[scheme_index, method_index]
            ),
        }
        for scheme_index, scheme in enumerate(SCHEMES)
        for method_index, method in enumerate(METHOD_IDS)
    ]
    rankings = {
        scheme: _point_ranking(point_utilities[scheme_index])
        for scheme_index, scheme in enumerate(SCHEMES)
    }
    loo: list[dict[str, Any]] = []
    for held_out in range(ROOTS):
        keep = np.arange(ROOTS) != held_out
        loo_point = np.mean(values[keep], axis=0, dtype=np.float64)
        loo_contrasts = generic.root_contrasts(
            values[None, keep, :, :], family
        )[0].mean(axis=0, dtype=np.float64)
        loo.append(
            {
                "held_out_root_id": str(root_ids[held_out]),
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
                    scheme: _point_ranking(loo_point[scheme_index])
                    for scheme_index, scheme in enumerate(SCHEMES)
                },
            }
        )
    scheme_pairs = ((0, 1), (0, 2), (1, 2))
    reference_shifts = [
        {
            "method_id": method,
            "scheme_a": SCHEMES[left],
            "scheme_b": SCHEMES[right],
            "point_utility_difference": float(
                point_utilities[left, method_index]
                - point_utilities[right, method_index]
            ),
        }
        for method_index, method in enumerate(METHOD_IDS)
        for left, right in scheme_pairs
    ]
    return {
        "fallback": bool(inference["fallback"][0]),
        "critical_value": _finite_or_none(inference["critical_values"][0]),
        "sign_schedule_sha256": generic.sign_schedule_sha256(signs),
        "point_utility_records": point_records,
        "pairwise_interval_records": intervals,
        "best_model_confidence_sets": best_sets,
        "possible_rank_intervals": ranks,
        "deterministic_point_rankings": rankings,
        "leave_one_root_out": loo,
        "reference_scheme_point_shifts": reference_shifts,
        "reference_scheme_decision_changes": {
            "best_sets_not_all_equal": len(
                {tuple(best_sets[scheme]) for scheme in SCHEMES}
            )
            > 1,
            "point_rankings_not_all_equal": len(
                {
                    tuple(
                        row["method_id"]
                        for row in rankings[scheme]
                    )
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


def build_empirical_report(
    root_utilities_by_depth: Mapping[int, np.ndarray],
    *,
    eligibility_sha256: str,
    contract_sha256: str,
    generic_source_sha256: str,
    generic_qualification_sha256: str,
    root_ids: Sequence[str],
    treated_access_audit: Mapping[str, Any],
    input_bindings: Mapping[str, str],
    generic: ModuleType,
) -> dict[str, Any]:
    """Build the complete three-depth report without writing it."""

    if tuple(root_utilities_by_depth) != DEPTHS:
        raise ParseEmpiricalScorerError("depth order differs")
    depth_results = {
        str(depth): analyse_root_utilities(
            root_utilities_by_depth[depth],
            root_ids=root_ids,
            generic=generic,
        )
        for depth in DEPTHS
    }
    return {
        "schema_version": 1,
        "record_type": "PARSE_10M_EIGHT_MODEL_EMPIRICAL_ANALYSIS_V1",
        "status": PASS_STATUS,
        "scorer_id": SCORER_ID,
        "primary_depth": PRIMARY_DEPTH,
        "geometry": {
            "tasks": TASKS,
            "roots": ROOTS,
            "root_tasks": ROOT_TASKS,
            "methods": METHODS,
            "features": FEATURES,
            "schemes": len(SCHEMES),
            "joint_contrasts": CONTRASTS,
        },
        "method_order": list(METHOD_IDS),
        "scheme_order": list(SCHEMES),
        "root_order": list(root_ids),
        "eligibility_sha256": _sha256(
            eligibility_sha256, label="eligibility SHA-256"
        ),
        "contract_sha256": _sha256(contract_sha256, label="contract SHA-256"),
        "generic_inference": {
            "source_sha256": _sha256(
                generic_source_sha256, label="generic source SHA-256"
            ),
            "qualification_report_sha256": _sha256(
                generic_qualification_sha256,
                label="generic qualification SHA-256",
            ),
            "procedure": "CURRENT_JOINT_MAXT",
            "decision_sets": "EXACT_FROZEN_GENERIC_V1_0_FUNCTION",
        },
        "metric": {
            "larger_is_better": True,
            "utility": (
                "NEGATIVE_EQUAL_FEATURE_WEIGHTED_STANDARDIZED_"
                "SQUARED_PERTURBATION_EFFECT_ERROR"
            ),
            "crossfit": (
                "NONLINEAR_UTILITY_PER_ROTATION_THEN_EQUAL_"
                "AVERAGE_OF_THREE_UTILITIES"
            ),
        },
        "depths": depth_results,
        "primary_result": depth_results[str(PRIMARY_DEPTH)],
        "depth_sensitivity": {
            "secondary_depths": [16, 48],
            "formal_results_present": True,
        },
        "treated_access_audit": dict(treated_access_audit),
        "input_bindings": dict(input_bindings),
        "claim_boundary": (
            "CONDITIONAL_PARSE_EIGHT_DONOR_APPLICATION_NOT_"
            "DONOR_POPULATION_GENERALIZATION"
        ),
    }


__all__ = [
    "CONTRASTS",
    "DEPTHS",
    "EVALUATION_TREATED_CELLS",
    "EligibilityGate",
    "EvalBundle",
    "FEATURES",
    "METHODS",
    "PRIMARY_DEPTH",
    "PASS_STATUS",
    "ParseEmpiricalScorerError",
    "ROOTS",
    "ROOT_TASKS",
    "RootTaskAxis",
    "SCHEMES",
    "SCORER_ID",
    "TASKS",
    "TreatedMeans",
    "TreatedMembership",
    "aggregate_registered_treated_means",
    "aggregate_scheme_root_utilities",
    "analyse_root_utilities",
    "build_empirical_report",
    "canonical_json_bytes",
    "compute_instance_utilities",
    "gated_treated_read",
    "load_absolute_predictions",
    "load_direct_predictions",
    "load_eval_bundle",
    "load_frozen_generic",
    "load_root_task_axis",
    "load_treated_membership",
    "sha256_file",
    "standardized_squared_utility",
    "validate_contract",
    "validate_eligibility_gate",
    "validate_generic_qualification",
]
