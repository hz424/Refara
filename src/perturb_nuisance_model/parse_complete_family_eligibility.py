"""Check complete-family eligibility for the fixed Parse eight-model panel before scoring.

This module runs after prediction materialization and before empirical scoring.
It reads only:

* the published five-method DIRECT family;
* the EVALUATION-PBS control/axis bundle;
* the complete CPA/scGen PRIMARY and REPLAY prediction family; and
* the complete CellOT PRIMARY and REPLAY prediction family.

It has no H5AD reader and imports no metric or inference implementation.
Eligibility is structural: exact methods, axes, semantics, state and file
hashes, reference memberships, once-only C_pred subtraction, numerical
replay, and the frozen 84-contrast comparison axis.  A missing or extra
member is a complete-family NO-GO.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from perturb_nuisance_model.parse_10m_pipeline import (
    ABSOLUTE_METHOD_IDS,
    CELLOT,
    CPA,
    DIRECT_METHOD_IDS,
    METHOD_IDS,
    REFERENCE_INSTANCE_IDS,
    SCGEN,
    audit_numerical_replay,
    finalize_direct_prediction,
)


PASS_STATUS = "PASS_COMPLETE_EIGHT_MODEL_FAMILY_ELIGIBLE_NO_SCORING"
NO_GO_STATUS = "NO_GO_COMPLETE_EIGHT_MODEL_FAMILY_INELIGIBLE"
RECORD_TYPE = "PARSE_10M_COMPLETE_EIGHT_MODEL_ELIGIBILITY_V1"
EVAL_RECORD_TYPE = "PARSE_10M_EVALUATION_PBS_CONTROL_BUNDLE_V1"
EVAL_PASS_STATUS = "PASS_EVALUATION_PBS_ONLY_NO_PREDICTION_OR_SCORE"
DIRECT_RECORD_TYPE = "PARSE_10M_FIVE_DIRECT_STATE_PREDICTION_FAMILY_V1"
DIRECT_PASS_STATUS = "PASS_PARSE_10M_FIVE_DIRECT_STATE_PREDICTION_FAMILY_V1"
CPA_SCGEN_RECORD_TYPE = "PARSE_10M_CPA_SCGEN_COMPLETE_PREDICTION_FAMILY_V1"
CPA_SCGEN_PASS_STATUS = (
    "PASS_COMPLETE_CPA_SCGEN_30_METHOD_INSTANCE_PREDICTION_FAMILY_WITH_REPLAY_V1"
)
CELLOT_RECORD_TYPE = "PARSE_CELLOT_COMPLETE_PREDICTION_FAMILY_V1"
CELLOT_PASS_STATUS = (
    "PASS_COMPLETE_CELLOT_15_INSTANCE_PREDICTION_FAMILY_WITH_REPLAY_V1"
)
CPA_SCGEN_INSTANCE_RECORD_TYPE = (
    "PARSE_10M_CPA_SCGEN_REFERENCE_INSTANCE_PREDICTIONS_V1"
)
CPA_SCGEN_INSTANCE_PASS_STATUS = "PASS_REFERENCE_INSTANCE_PREDICTIONS_NO_SCORING"
CELLOT_INSTANCE_RECORD_TYPE = "PARSE_CELLOT_REFERENCE_INSTANCE_PREDICTIONS_V1"
CELLOT_INSTANCE_PASS_STATUS = (
    "PASS_CELLOT_SCORE_FREE_REFERENCE_INSTANCE_PREDICTIONS_V1"
)
PANEL_ID = "PARSE_10M_EIGHT_MODEL_PANEL_V1"
SCHEMES = ("NAIVE_SHARED", "INDEPENDENT_SPLIT", "UNIT_AWARE_CROSSFIT")
ARTIFACT_KEYS = ("absolute_native", "c_pred_mean", "predicted_effect")
RUN_LABELS = ("PRIMARY", "REPLAY")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ParseEligibilityError(RuntimeError):
    """An eligibility input violated the fixed pre-scoring contract."""


@dataclass(frozen=True)
class EligibilityGeometry:
    tasks: int
    roots: int
    root_tasks: int
    features: int

    @classmethod
    def production(cls) -> "EligibilityGeometry":
        return cls(tasks=722, roots=8, root_tasks=5_776, features=2_000)


@dataclass(frozen=True)
class AxisBundle:
    report_sha256: str
    task_ids: tuple[str, ...]
    root_task_ids: tuple[str, ...]
    root_task_control_groups: tuple[str, ...]
    feature_ids: tuple[str, ...]
    memberships: Mapping[tuple[str, str, str], tuple[str, ...]]
    artifact_sha256: Mapping[str, str]


@dataclass(frozen=True)
class FamilyScan:
    methods: tuple[str, ...]
    state_sha256_by_method: Mapping[str, str]
    method_instance_count: int
    primary_artifact_count: int
    replay_artifact_count: int
    replay_max_absolute_difference: float
    replay_max_scaled_difference: float


def sha256_file(path: Path, *, block_size: int = 16 * 1024 * 1024) -> str:
    """Return a whole-file SHA-256 without following symbolic links."""

    resolved = Path(path)
    if resolved.is_symlink() or not resolved.is_file():
        raise ParseEligibilityError(f"not a regular file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
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
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ParseEligibilityError(f"{label} is not lowercase SHA-256")
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
        raise ParseEligibilityError(f"{label} is not canonical text")
    return value


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise ParseEligibilityError(f"{label} is not an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ParseEligibilityError(f"{label} is not an integer") from error
    if result < 0 or str(result) != str(value):
        raise ParseEligibilityError(f"{label} is not canonical nonnegative integer")
    return result


def _read_json(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    resolved = Path(path)
    digest = sha256_file(resolved)
    if expected_sha256 is not None and digest != _sha256(
        expected_sha256, label=f"{label} expected SHA-256"
    ):
        raise ParseEligibilityError(f"{label} SHA-256 differs")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ParseEligibilityError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ParseEligibilityError(f"{label} must be a JSON object")
    return value, digest


def _artifact_path(
    root: Path,
    record: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    raw = record.get("path")
    if not isinstance(raw, str):
        raise ParseEligibilityError(f"{label} path is absent")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ParseEligibilityError(f"{label} path is unsafe")
    path = Path(root) / relative
    expected_size = _integer(record.get("size_bytes"), label=f"{label} size")
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != expected_size
        or sha256_file(path)
        != _sha256(record.get("sha256"), label=f"{label} SHA-256")
    ):
        raise ParseEligibilityError(f"{label} file identity differs")
    return path


def _open_tsv(path: Path):
    return (
        gzip.open(path, "rt", encoding="utf-8", newline="")
        if path.suffix == ".gz"
        else path.open("rt", encoding="utf-8", newline="")
    )


def _read_artifact_rows(
    root: Path,
    record: Mapping[str, Any],
    *,
    label: str,
    expected_header: Sequence[str],
) -> tuple[dict[str, str], ...]:
    path = _artifact_path(root, record, label=label)
    rows: list[dict[str, str]] = []
    with _open_tsv(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != tuple(expected_header):
            raise ParseEligibilityError(f"{label} header differs")
        rows.extend(reader)
    if len(rows) != _integer(record.get("rows"), label=f"{label} rows"):
        raise ParseEligibilityError(f"{label} row count differs")
    return tuple(rows)


def _membership_sha256(
    axis_ids: Sequence[str],
    memberships: Sequence[Sequence[str]],
    *,
    label: bytes,
) -> str:
    payload = [
        [axis_id, list(members)]
        for axis_id, members in zip(axis_ids, memberships, strict=True)
    ]
    digest = hashlib.sha256(label)
    digest.update(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _forbidden_boundary_true(boundary: Mapping[str, Any]) -> bool:
    """Recognize every currently published spelling of forbidden output."""

    forbidden_exact = {
        "evaluation_expression_read",
        "evaluation_perturbed_expression_read",
        "evaluation_treated_rows_read",
        "scores_utilities_intervals_winners_or_ranks_computed",
        "scores_utilities_contrasts_materialized",
        "winner_best_set_or_rank_materialized",
        "winner_best_set_rank_materialized",
    }
    for key, value in boundary.items():
        normalized = str(key)
        if normalized in forbidden_exact and value not in (False, 0):
            return True
        lowered = normalized.casefold()
        if any(
            token in lowered
            for token in ("score", "utility", "contrast", "winner", "best_set", "rank")
        ) and value not in (False, 0, None):
            return True
    return False


def load_axis_bundle(
    report_path: Path,
    *,
    expected_sha256: str,
    geometry: EligibilityGeometry,
    production_geometry: bool,
) -> AxisBundle:
    """Load only axes and reference memberships from the EVAL-PBS bundle."""

    report, report_sha = _read_json(
        report_path,
        label="EVAL-PBS bundle report",
        expected_sha256=expected_sha256,
    )
    control = report.get("control_geometry")
    boundary = report.get("access_boundary")
    artifacts = report.get("artifacts")
    if (
        report.get("record_type") != EVAL_RECORD_TYPE
        or report.get("status") != EVAL_PASS_STATUS
        or report.get("production_geometry") is not production_geometry
        or not isinstance(control, Mapping)
        or control.get("root_task_rows") != geometry.root_tasks
        or not isinstance(control.get("panel_shape"), list)
        or len(control["panel_shape"]) != 2
        or control["panel_shape"][1] != geometry.features
        or (production_geometry and control["panel_shape"][0] != 12_672)
        or control.get("reference_instances") != len(REFERENCE_INSTANCE_IDS)
        or not isinstance(boundary, Mapping)
        or boundary.get("evaluation_treated_rows_read") is not False
        or boundary.get("predictions_materialized") is not False
        or _forbidden_boundary_true(boundary)
        or not isinstance(artifacts, Mapping)
    ):
        raise ParseEligibilityError("EVAL-PBS bundle identity or boundary differs")
    root = Path(report_path).resolve().parent
    required = ("panel_features", "root_task_axis", "role_memberships")
    records: dict[str, Mapping[str, Any]] = {}
    for key in required:
        value = artifacts.get(key)
        if not isinstance(value, Mapping):
            raise ParseEligibilityError(f"EVAL-PBS bundle lacks {key}")
        records[key] = value

    panel_header = (
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
    panel_rows = _read_artifact_rows(
        root,
        records["panel_features"],
        label="panel feature axis",
        expected_header=panel_header,
    )
    feature_ids: list[str] = []
    for index, row in enumerate(panel_rows):
        if row["panel_index"] != str(index):
            raise ParseEligibilityError("panel feature order differs")
        feature_ids.append(_text(row["feature_id"], label="feature_id"))
    if len(feature_ids) != geometry.features or len(set(feature_ids)) != len(
        feature_ids
    ):
        raise ParseEligibilityError("panel feature count or uniqueness differs")

    root_header = (
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
    root_rows = _read_artifact_rows(
        root,
        records["root_task_axis"],
        label="root-task axis",
        expected_header=root_header,
    )
    root_task_ids: list[str] = []
    root_task_groups: list[str] = []
    task_by_index: dict[int, str] = {}
    roots_by_task: dict[int, set[str]] = {}
    for offset, row in enumerate(root_rows):
        if row["root_task_offset"] != str(offset):
            raise ParseEligibilityError("root-task order differs")
        task_index = _integer(row["task_index"], label="task_index")
        task_id = _text(row["task_id"], label="task_id")
        root_id = _text(row["root_id"], label="root_id")
        control_group = _text(row["control_group_id"], label="control_group_id")
        if task_index in task_by_index and task_by_index[task_index] != task_id:
            raise ParseEligibilityError("task identity differs across roots")
        task_by_index[task_index] = task_id
        roots_by_task.setdefault(task_index, set()).add(root_id)
        root_task_ids.append(f"{root_id}|{task_id}")
        root_task_groups.append(control_group)
    task_ids = tuple(task_by_index[index] for index in range(len(task_by_index)))
    if (
        len(root_task_ids) != geometry.root_tasks
        or len(set(root_task_ids)) != geometry.root_tasks
        or len(task_ids) != geometry.tasks
        or any(len(value) != geometry.roots for value in roots_by_task.values())
    ):
        raise ParseEligibilityError("root-task geometry differs")

    membership_header = (
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
    membership_rows = _read_artifact_rows(
        root,
        records["role_memberships"],
        label="reference memberships",
        expected_header=membership_header,
    )
    grouped: dict[tuple[str, str, str], list[str]] = {}
    for row in membership_rows:
        instance = _text(
            row["reference_instance_id"], label="reference_instance_id"
        )
        role = _text(row["role"], label="reference role")
        group = _text(row["control_group_id"], label="control_group_id")
        if instance not in REFERENCE_INSTANCE_IDS or role not in {
            "C_obs",
            "C_pred",
            "C_infer",
        }:
            raise ParseEligibilityError("reference membership identity differs")
        key = (instance, group, role)
        members = grouped.setdefault(key, [])
        if row["role_cell_rank"] != str(len(members)):
            raise ParseEligibilityError("reference membership rank differs")
        members.append(_text(row["obs_index"], label="reference obs_index"))
    memberships = {key: tuple(value) for key, value in grouped.items()}
    expected_keys = {
        (instance, group, role)
        for instance in REFERENCE_INSTANCE_IDS
        for group in set(root_task_groups)
        for role in ("C_obs", "C_pred", "C_infer")
    }
    if set(memberships) != expected_keys:
        raise ParseEligibilityError(
            "reference memberships are missing, extra, or duplicated"
        )
    return AxisBundle(
        report_sha256=report_sha,
        task_ids=task_ids,
        root_task_ids=tuple(root_task_ids),
        root_task_control_groups=tuple(root_task_groups),
        feature_ids=tuple(feature_ids),
        memberships=memberships,
        artifact_sha256={
            key: _sha256(records[key].get("sha256"), label=f"{key} SHA-256")
            for key in required
        },
    )


def _read_direct_axis(
    path: Path,
    *,
    expected_sha256: str,
    expected_header: Sequence[str],
    index_column: str,
    id_column: str,
) -> tuple[str, ...]:
    if sha256_file(path) != _sha256(expected_sha256, label="DIRECT axis SHA-256"):
        raise ParseEligibilityError("DIRECT axis SHA-256 differs")
    values: list[str] = []
    with _open_tsv(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != tuple(expected_header):
            raise ParseEligibilityError("DIRECT axis header differs")
        for index, row in enumerate(reader):
            if row[index_column] != str(index):
                raise ParseEligibilityError("DIRECT axis order differs")
            values.append(_text(row[id_column], label=id_column))
    if len(values) != len(set(values)):
        raise ParseEligibilityError("DIRECT axis is duplicated")
    return tuple(values)


def scan_direct_family(
    report_path: Path,
    *,
    expected_sha256: str,
    axes: AxisBundle,
    geometry: EligibilityGeometry,
) -> FamilyScan:
    """Verify the five DIRECT states and prediction files."""

    report, _ = _read_json(
        report_path,
        label="DIRECT family report",
        expected_sha256=expected_sha256,
    )
    boundary = report.get("access_boundary")
    if (
        report.get("record_type") != DIRECT_RECORD_TYPE
        or report.get("status") != DIRECT_PASS_STATUS
        or not isinstance(boundary, Mapping)
        or boundary.get("evaluation_expression_read") is not False
        or boundary.get("evaluation_treated_rows_read") != 0
        or _forbidden_boundary_true(boundary)
    ):
        raise ParseEligibilityError("DIRECT family identity or boundary differs")
    root = Path(_text(report.get("external_output"), label="DIRECT external root"))
    if root.is_symlink() or not root.is_dir():
        raise ParseEligibilityError("DIRECT external root is not a directory")
    axis = report.get("axes")
    if not isinstance(axis, Mapping):
        raise ParseEligibilityError("DIRECT axes are absent")
    task_path = root / _text(axis.get("task_axis_file"), label="task axis path")
    panel_path = root / _text(axis.get("panel_file"), label="panel axis path")
    task_ids = _read_direct_axis(
        task_path,
        expected_sha256=_sha256(
            axis.get("task_axis_file_sha256"), label="task axis SHA-256"
        ),
        expected_header=(
            "task_index",
            "task_id",
            "cytokine",
            "cell_type",
            "task_weight",
        ),
        index_column="task_index",
        id_column="task_id",
    )
    feature_ids = _read_direct_axis(
        panel_path,
        expected_sha256=_sha256(
            axis.get("panel_file_sha256"), label="panel axis SHA-256"
        ),
        expected_header=(
            "panel_index",
            "source_feature_index",
            "feature_id",
            "variance_selection_rank",
            "feature_weight",
            "transformed_mean",
            "unfloored_variance",
            "unfloored_sd",
            "floored_scale",
        ),
        index_column="panel_index",
        id_column="feature_id",
    )
    if task_ids != axes.task_ids or feature_ids != axes.feature_ids:
        raise ParseEligibilityError("DIRECT and shared bundle axes differ")

    states = report.get("states")
    predictions = report.get("predictions")
    if not isinstance(states, list) or not isinstance(predictions, list):
        raise ParseEligibilityError("DIRECT state or prediction records are absent")
    if [row.get("method_id") for row in states] != list(DIRECT_METHOD_IDS):
        raise ParseEligibilityError("DIRECT state family is incomplete or reordered")
    state_by_method: dict[str, str] = {}
    for row in states:
        if not isinstance(row, Mapping):
            raise ParseEligibilityError("DIRECT state record is malformed")
        digest = _sha256(
            row.get("canonical_state_sha256"), label="DIRECT state SHA-256"
        )
        if row.get("exact_replay") is not True:
            raise ParseEligibilityError("DIRECT state replay failed")
        state_by_method[str(row["method_id"])] = digest
    if [row.get("method_id") for row in predictions] != list(DIRECT_METHOD_IDS):
        raise ParseEligibilityError(
            "DIRECT prediction family is incomplete or reordered"
        )
    prediction_dir = root / "predictions"
    expected_files = {f"{method}.npy" for method in DIRECT_METHOD_IDS}
    if (
        prediction_dir.is_symlink()
        or not prediction_dir.is_dir()
        or {path.name for path in prediction_dir.iterdir() if path.is_file()}
        != expected_files
    ):
        raise ParseEligibilityError("DIRECT prediction file family differs")
    for row in predictions:
        if not isinstance(row, Mapping):
            raise ParseEligibilityError("DIRECT prediction record is malformed")
        method = str(row.get("method_id"))
        path = root / _text(row.get("file"), label="DIRECT prediction path")
        if (
            path.parent != prediction_dir
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size
            != _integer(row.get("size_bytes"), label="DIRECT prediction size")
            or sha256_file(path)
            != _sha256(row.get("file_sha256"), label="DIRECT prediction SHA-256")
            or row.get("shape") != [geometry.tasks, geometry.features]
            or row.get("dtype") != "<f4"
            or row.get("output_type") != "PREDICTED_EFFECT"
            or row.get("prediction_semantics") != "DIRECT"
            or row.get("reference_instance_id") is not None
            or row.get("uses_C_infer") is not False
            or row.get("C_pred_subtractions") != 0
            or row.get("exact_replay") is not True
            or row.get("state_sha256") != state_by_method.get(method)
        ):
            raise ParseEligibilityError(f"{method} DIRECT artifact differs")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            values.dtype != np.dtype("<f4")
            or values.shape != (geometry.tasks, geometry.features)
            or not np.isfinite(values).all()
        ):
            raise ParseEligibilityError(f"{method} DIRECT payload differs")
        rebuilt = finalize_direct_prediction(
            method_id=method,
            task_ids=axes.task_ids,
            feature_ids=axes.feature_ids,
            predicted_effect=values,
            state_sha256=state_by_method[method],
        )
        if rebuilt.artifact_sha256 != row.get("prediction_artifact_sha256"):
            raise ParseEligibilityError(
                f"{method} DIRECT logical artifact hash differs"
            )
    return FamilyScan(
        methods=DIRECT_METHOD_IDS,
        state_sha256_by_method=state_by_method,
        method_instance_count=len(DIRECT_METHOD_IDS),
        primary_artifact_count=len(DIRECT_METHOD_IDS),
        replay_artifact_count=len(DIRECT_METHOD_IDS),
        replay_max_absolute_difference=0.0,
        replay_max_scaled_difference=0.0,
    )


def _completion_paths(
    row: Mapping[str, Any],
) -> tuple[tuple[Path, str], tuple[Path, str]]:
    return (
        (
            Path(
                _text(
                    row.get("primary_completion_record"),
                    label="PRIMARY completion path",
                )
            ),
            _sha256(
                row.get("primary_completion_sha256"),
                label="PRIMARY completion SHA-256",
            ),
        ),
        (
            Path(
                _text(
                    row.get("replay_completion_record"),
                    label="REPLAY completion path",
                )
            ),
            _sha256(
                row.get("replay_completion_sha256"),
                label="REPLAY completion SHA-256",
            ),
        ),
    )


def _load_prediction_array(
    completion_dir: Path,
    record: Mapping[str, Any],
    *,
    key: str,
    geometry: EligibilityGeometry,
) -> np.ndarray:
    if record.get("path") != f"{key}.npy":
        raise ParseEligibilityError(f"{key} artifact path differs")
    path = completion_dir / f"{key}.npy"
    if (
        path.is_symlink()
        or not path.is_file()
        or sha256_file(path)
        != _sha256(record.get("sha256"), label=f"{key} SHA-256")
        or record.get("dtype") != "<f4"
        or record.get("shape") != [geometry.root_tasks, geometry.features]
    ):
        raise ParseEligibilityError(f"{key} artifact identity differs")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        values.dtype != np.dtype("<f4")
        or values.shape != (geometry.root_tasks, geometry.features)
        or not np.isfinite(values).all()
    ):
        raise ParseEligibilityError(f"{key} artifact payload differs")
    return values


def _validate_instance_semantics(
    completion: Mapping[str, Any],
    *,
    method: str,
) -> None:
    semantic = completion.get("semantic_contract")
    boundary = completion.get("access_boundary")
    if not isinstance(semantic, Mapping) or not isinstance(boundary, Mapping):
        raise ParseEligibilityError("prediction semantics or boundary are absent")
    if method in (CPA, SCGEN):
        valid_semantics = (
            semantic.get("prediction_semantics") == "ABSOLUTE"
            and semantic.get("C_obs_structurally_validated") is True
            and semantic.get("predictor_numeric_roles") == ["C_infer"]
            and semantic.get("effect_constructor_numeric_roles") == ["C_pred"]
            and semantic.get("C_pred_subtractions") == 1
        )
    else:
        valid_semantics = (
            semantic.get("input_role_exposed_to_predictor") == ["C_infer"]
            and semantic.get("C_obs_membership_structurally_validated") is True
            and semantic.get("C_obs_numeric_values_read_by_predictor") is False
            and semantic.get("C_pred_values_exposed_to_predictor") is False
            and semantic.get("native_output") == "ABSOLUTE_NATIVE_FROZEN_PANEL"
            and semantic.get("C_pred_subtractions") == 1
            and semantic.get("post_prediction_clipping") is False
        )
    if (
        not valid_semantics
        or boundary.get("evaluation_perturbed_expression_read") is not False
        or boundary.get("scores_utilities_contrasts_materialized") is not False
        or boundary.get("winner_best_set_or_rank_materialized") is not False
        or _forbidden_boundary_true(boundary)
    ):
        raise ParseEligibilityError(
            f"{method} prediction semantics or access boundary differs"
        )


def _expected_membership_hashes(
    axes: AxisBundle,
    instance: str,
) -> tuple[str, str]:
    infer = [
        axes.memberships[(instance, group, "C_infer")]
        for group in axes.root_task_control_groups
    ]
    pred = [
        axes.memberships[(instance, group, "C_pred")]
        for group in axes.root_task_control_groups
    ]
    return (
        _membership_sha256(
            axes.root_task_ids,
            infer,
            label=b"PARSE_10M_C_INFER_MEMBERSHIP_V1\0",
        ),
        _membership_sha256(
            axes.root_task_ids,
            pred,
            label=b"PARSE_10M_C_PRED_MEMBERSHIP_V1\0",
        ),
    )


def _scan_instance_pair(
    *,
    method: str,
    instance: str,
    row: Mapping[str, Any],
    axes: AxisBundle,
    geometry: EligibilityGeometry,
    production_geometry: bool,
) -> tuple[str, float, float]:
    completion_paths = _completion_paths(row)
    loaded: list[tuple[dict[str, Any], dict[str, np.ndarray]]] = []
    expected_infer, expected_pred = _expected_membership_hashes(axes, instance)
    for run_index, ((path, expected_sha), run_label) in enumerate(
        zip(completion_paths, RUN_LABELS, strict=True)
    ):
        completion, _ = _read_json(
            path,
            label=f"{method}/{instance}/{run_label} completion",
            expected_sha256=expected_sha,
        )
        if method in (CPA, SCGEN):
            expected_record = CPA_SCGEN_INSTANCE_RECORD_TYPE
            expected_status = CPA_SCGEN_INSTANCE_PASS_STATUS
        else:
            expected_record = CELLOT_INSTANCE_RECORD_TYPE
            expected_status = CELLOT_INSTANCE_PASS_STATUS
        completion_geometry = completion.get("geometry")
        bindings = completion.get("input_bindings")
        membership = completion.get("membership_bindings")
        artifacts = completion.get("artifacts")
        if (
            completion.get("record_type") != expected_record
            or completion.get("status") != expected_status
            or completion.get("method_id") != method
            or completion.get("run_label") != run_label
            or completion.get("reference_instance_id") != instance
            or completion.get("reference_instance_index")
            != REFERENCE_INSTANCE_IDS.index(instance)
            or completion.get("production_geometry") is not production_geometry
            or not isinstance(completion_geometry, Mapping)
            or completion_geometry.get("root_tasks") != geometry.root_tasks
            or completion_geometry.get("tasks") != geometry.tasks
            or completion_geometry.get("roots") != geometry.roots
            or completion_geometry.get("features") != geometry.features
            or not isinstance(bindings, Mapping)
            or bindings.get("eval_reference_bundle_report_sha256")
            != axes.report_sha256
            or not isinstance(membership, Mapping)
            or membership.get("C_infer_root_task_membership_sha256")
            != expected_infer
            or membership.get("C_pred_root_task_membership_sha256")
            != expected_pred
            or not isinstance(artifacts, Mapping)
            or set(artifacts) != set(ARTIFACT_KEYS)
        ):
            raise ParseEligibilityError(
                f"{method}/{instance}/{run_label} completion differs"
            )
        _validate_instance_semantics(completion, method=method)
        completion_dir = path.parent
        if (
            completion_dir.is_symlink()
            or {entry.name for entry in completion_dir.iterdir()}
            != {
                "completion_record.json",
                "absolute_native.npy",
                "c_pred_mean.npy",
                "predicted_effect.npy",
            }
        ):
            raise ParseEligibilityError(
                f"{method}/{instance}/{run_label} file tree differs"
            )
        arrays = {
            key: _load_prediction_array(
                completion_dir,
                artifacts[key],
                key=key,
                geometry=geometry,
            )
            for key in ARTIFACT_KEYS
        }
        reconstructed = np.subtract(
            arrays["absolute_native"], arrays["c_pred_mean"], dtype=np.float32
        )
        if not np.array_equal(arrays["predicted_effect"], reconstructed):
            raise ParseEligibilityError(
                f"{method}/{instance}/{run_label} is not "
                "ABSOLUTE_NATIVE minus exactly one C_pred"
            )
        loaded.append((completion, arrays))
        if run_index == 1 and (
            loaded[0][0].get("input_bindings") != bindings
            or loaded[0][0].get("membership_bindings") != membership
        ):
            raise ParseEligibilityError(
                f"{method}/{instance} PRIMARY/REPLAY lineage differs"
            )
    primary, replay = loaded
    max_abs = 0.0
    max_scaled = 0.0
    for key in ARTIFACT_KEYS:
        result = audit_numerical_replay(
            primary[1][key],
            replay[1][key],
            exact_required=False,
        )
        if not result.passed:
            raise ParseEligibilityError(
                f"{method}/{instance}/{key} numerical replay failed"
            )
        max_abs = max(max_abs, result.max_absolute_difference)
        max_scaled = max(max_scaled, result.max_scaled_difference)
    bindings = primary[0]["input_bindings"]
    state_key = (
        "cellot_state_family_sha256"
        if method == CELLOT
        else "state_logical_sha256"
    )
    return _sha256(
        bindings.get(state_key), label=f"{method} logical state SHA-256"
    ), max_abs, max_scaled


def scan_absolute_family(
    report_path: Path,
    *,
    expected_sha256: str,
    expected_methods: tuple[str, ...],
    axes: AxisBundle,
    geometry: EligibilityGeometry,
    production_geometry: bool,
) -> FamilyScan:
    """Stream and verify one complete ABSOLUTE prediction subfamily."""

    report, _ = _read_json(
        report_path,
        label="ABSOLUTE family report",
        expected_sha256=expected_sha256,
    )
    if expected_methods == (CPA, SCGEN):
        expected_record = CPA_SCGEN_RECORD_TYPE
        expected_status = CPA_SCGEN_PASS_STATUS
        rows = report.get("records")
        expected_pairs = tuple(
            (method, instance)
            for method in expected_methods
            for instance in REFERENCE_INSTANCE_IDS
        )
        geometry_record = report.get("geometry")
        if (
            not isinstance(geometry_record, Mapping)
            or geometry_record.get("methods") != 2
            or geometry_record.get("reference_instances") != 15
            or geometry_record.get("method_instance_pairs") != 30
            or geometry_record.get("root_task_rows") != geometry.root_tasks
            or geometry_record.get("features") != geometry.features
            or report.get("complete_eight_method_family_gate_open") is not False
            or report.get("scoring_authorized") is not False
        ):
            raise ParseEligibilityError("CPA/scGen family geometry differs")
    elif expected_methods == (CELLOT,):
        expected_record = CELLOT_RECORD_TYPE
        expected_status = CELLOT_PASS_STATUS
        rows = report.get("instances")
        expected_pairs = tuple((CELLOT, value) for value in REFERENCE_INSTANCE_IDS)
        geometry_record = report.get("geometry")
        if (
            report.get("method_id") != CELLOT
            or report.get("instance_count") != 15
            or report.get("complete_primary_and_replay_families") is not True
            or not isinstance(geometry_record, Mapping)
            or geometry_record.get("root_tasks") != geometry.root_tasks
            or geometry_record.get("features") != geometry.features
        ):
            raise ParseEligibilityError("CellOT family geometry differs")
    else:
        raise ParseEligibilityError("unknown ABSOLUTE subfamily")
    replay = report.get("replay")
    boundary = report.get("access_boundary", {})
    if (
        report.get("record_type") != expected_record
        or report.get("status") != expected_status
        or report.get("production_geometry") is not production_geometry
        or not isinstance(rows, list)
        or not isinstance(replay, Mapping)
        or replay.get("passed") is not True
        or (isinstance(boundary, Mapping) and _forbidden_boundary_true(boundary))
    ):
        raise ParseEligibilityError("ABSOLUTE family identity or boundary differs")
    observed_pairs = tuple(
        (
            str(row.get("method_id", CELLOT)),
            str(row.get("reference_instance_id")),
        )
        for row in rows
        if isinstance(row, Mapping)
    )
    if observed_pairs != expected_pairs or len(observed_pairs) != len(rows):
        raise ParseEligibilityError(
            "ABSOLUTE method-instance family is incomplete, extra, or reordered"
        )
    states: dict[str, str] = {}
    max_abs = 0.0
    max_scaled = 0.0
    for row, (method, instance) in zip(rows, expected_pairs, strict=True):
        assert isinstance(row, Mapping)
        state, observed_abs, observed_scaled = _scan_instance_pair(
            method=method,
            instance=instance,
            row=row,
            axes=axes,
            geometry=geometry,
            production_geometry=production_geometry,
        )
        if method in states and states[method] != state:
            raise ParseEligibilityError(f"{method} state differs across instances")
        states[method] = state
        max_abs = max(max_abs, observed_abs)
        max_scaled = max(max_scaled, observed_scaled)
    if tuple(states) != expected_methods:
        raise ParseEligibilityError("ABSOLUTE state family differs")
    return FamilyScan(
        methods=expected_methods,
        state_sha256_by_method=states,
        method_instance_count=len(expected_pairs),
        primary_artifact_count=len(expected_pairs) * len(ARTIFACT_KEYS),
        replay_artifact_count=len(expected_pairs) * len(ARTIFACT_KEYS),
        replay_max_absolute_difference=max_abs,
        replay_max_scaled_difference=max_scaled,
    )


def contrast_axis() -> tuple[dict[str, str], ...]:
    """Return the frozen 28 pairs x 3 schemes without any numeric endpoint."""

    return tuple(
        {
            "contrast_id": f"{scheme}::{left}__VS__{right}",
            "scheme": scheme,
            "method_a": left,
            "method_b": right,
        }
        for scheme in SCHEMES
        for left, right in itertools.combinations(METHOD_IDS, 2)
    )


def audit_from_paths(
    *,
    panel_path: Path,
    expected_panel_sha256: str,
    direct_report_path: Path,
    expected_direct_report_sha256: str,
    eval_bundle_report_path: Path,
    expected_eval_bundle_report_sha256: str,
    cpa_scgen_report_path: Path,
    expected_cpa_scgen_report_sha256: str,
    cellot_report_path: Path,
    expected_cellot_report_sha256: str,
    geometry: EligibilityGeometry | None = None,
    production_geometry: bool = True,
) -> dict[str, Any]:
    """Return PASS or a fail-closed NO-GO record; never calculate performance."""

    selected_geometry = geometry or EligibilityGeometry.production()
    bindings = {
        "panel": {
            "path": str(Path(panel_path)),
            "expected_sha256": expected_panel_sha256,
        },
        "direct_family": {
            "path": str(Path(direct_report_path)),
            "expected_sha256": expected_direct_report_sha256,
        },
        "eval_pbs_bundle": {
            "path": str(Path(eval_bundle_report_path)),
            "expected_sha256": expected_eval_bundle_report_sha256,
        },
        "cpa_scgen_family": {
            "path": str(Path(cpa_scgen_report_path)),
            "expected_sha256": expected_cpa_scgen_report_sha256,
        },
        "cellot_family": {
            "path": str(Path(cellot_report_path)),
            "expected_sha256": expected_cellot_report_sha256,
        },
    }
    failures: list[str] = []
    scans: tuple[FamilyScan, ...] = ()
    axes: AxisBundle | None = None
    try:
        panel, panel_sha = _read_json(
            panel_path,
            label="frozen Parse model panel",
            expected_sha256=expected_panel_sha256,
        )
        methods = panel.get("methods")
        if (
            panel.get("panel_id") != PANEL_ID
            or not isinstance(methods, list)
            or any(not isinstance(row, Mapping) for row in methods)
            or tuple(row.get("method_id") for row in methods) != METHOD_IDS
            or any(
                row.get("prediction_semantics")
                != ("DIRECT" if index < len(DIRECT_METHOD_IDS) else "ABSOLUTE")
                for index, row in enumerate(methods)
            )
        ):
            raise ParseEligibilityError("frozen model panel differs")
        bindings["panel"]["observed_sha256"] = panel_sha
        axes = load_axis_bundle(
            eval_bundle_report_path,
            expected_sha256=expected_eval_bundle_report_sha256,
            geometry=selected_geometry,
            production_geometry=production_geometry,
        )
        direct = scan_direct_family(
            direct_report_path,
            expected_sha256=expected_direct_report_sha256,
            axes=axes,
            geometry=selected_geometry,
        )
        cpa_scgen = scan_absolute_family(
            cpa_scgen_report_path,
            expected_sha256=expected_cpa_scgen_report_sha256,
            expected_methods=(CPA, SCGEN),
            axes=axes,
            geometry=selected_geometry,
            production_geometry=production_geometry,
        )
        cellot = scan_absolute_family(
            cellot_report_path,
            expected_sha256=expected_cellot_report_sha256,
            expected_methods=(CELLOT,),
            axes=axes,
            geometry=selected_geometry,
            production_geometry=production_geometry,
        )
        scans = (direct, cpa_scgen, cellot)
        observed_methods = tuple(
            method for scan in scans for method in scan.methods
        )
        if observed_methods != METHOD_IDS:
            raise ParseEligibilityError(
                "complete eight-method family is missing, extra, or reordered"
            )
        states = {
            method: digest
            for scan in scans
            for method, digest in scan.state_sha256_by_method.items()
        }
        if tuple(states) != METHOD_IDS:
            raise ParseEligibilityError("complete eight-state family differs")
        contrasts = contrast_axis()
        if (
            len(contrasts) != 84
            or len({row["contrast_id"] for row in contrasts}) != 84
            or {row["scheme"] for row in contrasts} != set(SCHEMES)
        ):
            raise ParseEligibilityError("frozen 84-contrast axis differs")
    except (
        ParseEligibilityError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
    ) as error:
        failures.append(str(error))

    eligible = not failures
    contrast_rows = contrast_axis()
    state_count = sum(len(scan.state_sha256_by_method) for scan in scans)
    return {
        "schema_version": 1,
        "record_type": RECORD_TYPE,
        "status": PASS_STATUS if eligible else NO_GO_STATUS,
        "complete_family_eligible": eligible,
        "scoring_performed": False,
        "failures": failures,
        "geometry": {
            "tasks": selected_geometry.tasks,
            "roots": selected_geometry.roots,
            "root_tasks": selected_geometry.root_tasks,
            "features": selected_geometry.features,
            "methods": len(METHOD_IDS),
            "states": state_count,
            "reference_instances": len(REFERENCE_INSTANCE_IDS),
            "absolute_method_instance_pairs": sum(
                scan.method_instance_count for scan in scans[1:]
            ),
            "within_scheme_method_pairs": 28,
            "schemes": len(SCHEMES),
            "joint_within_scheme_contrasts": len(contrast_rows),
        },
        "method_order": list(METHOD_IDS),
        "state_sha256_by_method": (
            {
                method: digest
                for scan in scans
                for method, digest in scan.state_sha256_by_method.items()
            }
            if eligible
            else None
        ),
        "reference_instance_order": list(REFERENCE_INSTANCE_IDS),
        "contrast_axis": list(contrast_rows),
        "artifact_audit": {
            "direct_effects": (
                scans[0].primary_artifact_count if len(scans) == 3 else 0
            ),
            "absolute_primary_artifacts": (
                sum(scan.primary_artifact_count for scan in scans[1:])
                if len(scans) == 3
                else 0
            ),
            "absolute_replay_artifacts": (
                sum(scan.replay_artifact_count for scan in scans[1:])
                if len(scans) == 3
                else 0
            ),
            "all_hashes_axes_semantics_and_once_only_subtractions_verified": (
                eligible
            ),
        },
        "replay": {
            "passed": eligible,
            "max_absolute_difference": (
                max(
                    (scan.replay_max_absolute_difference for scan in scans),
                    default=0.0,
                )
                if eligible
                else None
            ),
            "max_scaled_difference": (
                max(
                    (scan.replay_max_scaled_difference for scan in scans),
                    default=0.0,
                )
                if eligible
                else None
            ),
        },
        "input_bindings": bindings,
        "axis_bindings": (
            {
                "eval_bundle_report_sha256": axes.report_sha256,
                "artifact_sha256": dict(axes.artifact_sha256),
            }
            if axes is not None
            else None
        ),
        "access_boundary": {
            "input_surface": (
                "PUBLISHED_STATES_PREDICTIONS_AND_EVALUATION_PBS_AXES_ONLY"
            ),
            "h5ad_opened": False,
            "evaluation_treated_expression_read": False,
            "evaluation_perturbed_expression_read": False,
            "scores_utilities_contrasts_intervals_computed": False,
            "winner_best_set_or_rank_computed": False,
            "metric_or_scorer_imported": False,
        },
    }


__all__ = [
    "AxisBundle",
    "EligibilityGeometry",
    "NO_GO_STATUS",
    "PANEL_ID",
    "PASS_STATUS",
    "ParseEligibilityError",
    "RECORD_TYPE",
    "audit_from_paths",
    "canonical_json_bytes",
    "contrast_axis",
    "load_axis_bundle",
    "scan_absolute_family",
    "scan_direct_family",
    "sha256_file",
]
