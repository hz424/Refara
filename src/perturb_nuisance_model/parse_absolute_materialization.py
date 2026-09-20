"""Materialize Parse CPA/scGen absolute-state predictions without scoring.

This module is dataset-I/O agnostic.  It accepts a frozen EVALUATION-PBS
control bundle, exact reference-role memberships and injected predictors bound
to frozen terminal model states.  It never accepts perturbed EVALUATION
expression or a scorer.

The same compact donor-context reference membership is expanded over the
registered root-task axis.  ``C_obs`` is structurally validated but is never
returned through a numeric access method.  CPA and scGen numerically consume
only ``C_infer``; effect construction consumes only ``C_pred`` and subtracts
it exactly once through :func:`materialize_control_conditioned_absolute`.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy import sparse

from perturb_nuisance_model.parse_10m_pipeline import (
    CPA,
    SCGEN,
    REFERENCE_INSTANCE_IDS,
    PredictionArtifact,
    audit_numerical_replay,
    materialize_control_conditioned_absolute,
)


ROLE_ORDER = ("C_obs", "C_pred", "C_infer")
NUMERIC_ROLE_ORDER = ("C_infer", "C_pred")
BLOCK_ORDER = ("B1", "B2", "B3")
CONTROL_AXIS_COLUMNS = (
    "control_offset",
    "control_group_id",
    "donor",
    "cell_type",
    "block",
    "block_rank",
    "row_index",
    "obs_index",
    "selection_sha256",
)
ROOT_TASK_AXIS_COLUMNS = (
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
REFERENCE_MEMBERSHIP_COLUMNS = (
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

PRODUCTION_CONTROL_ROWS = 12_672
PRODUCTION_CONTROL_GROUPS = 88
PRODUCTION_ROOT_TASKS = 5_776
PRODUCTION_TASKS = 722
PRODUCTION_ROOTS = 8
PRODUCTION_PANEL_FEATURES = 2_000
PRODUCTION_SOURCE_FEATURES = 40_352
CP10K_TOTAL = 10_000.0
GPU_REPLAY_ATOL = 1.0e-6
GPU_REPLAY_RTOL = 1.0e-5

_INSTANCE = re.compile(
    r"^D(?P<depth>16|32|48)_(?P<scheme>"
    r"NAIVE_SHARED|INDEPENDENT_SPLIT|CROSSFIT_R[012])$"
)
_MISSING = frozenset({"", "na", "nan", "none", "null"})


class AbsoluteMaterializationError(RuntimeError):
    """A reference or ABSOLUTE prediction invariant failed."""


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, (str, np.str_)):
        raise AbsoluteMaterializationError(f"{name} must be text")
    result = str(value)
    if (
        result != result.strip()
        or result.casefold() in _MISSING
        or "\0" in result
    ):
        raise AbsoluteMaterializationError(f"{name} is not canonical text")
    try:
        result.encode("utf-8")
    except UnicodeEncodeError as error:
        raise AbsoluteMaterializationError(f"{name} is not valid UTF-8") from error
    return result


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise AbsoluteMaterializationError(f"{name} cannot be Boolean")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise AbsoluteMaterializationError(f"{name} must be an integer") from error
    if result < 0:
        raise AbsoluteMaterializationError(f"{name} cannot be negative")
    return result


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AbsoluteMaterializationError(f"{name} is not lowercase SHA-256")
    return value


def _dense(
    values: object,
    *,
    name: str,
    dtype: str = "<f4",
    shape: tuple[int, int] | None = None,
    nonnegative: bool = False,
) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as error:
        raise AbsoluteMaterializationError(f"{name} is not numeric") from error
    if result.ndim != 2 or min(result.shape) < 1:
        raise AbsoluteMaterializationError(f"{name} is not a nonempty matrix")
    if shape is not None and result.shape != shape:
        raise AbsoluteMaterializationError(
            f"{name} shape differs: {result.shape} != {shape}"
        )
    if not np.isfinite(result).all():
        raise AbsoluteMaterializationError(f"{name} contains a nonfinite value")
    if nonnegative and np.any(result < 0):
        raise AbsoluteMaterializationError(f"{name} contains a negative value")
    frozen = np.ascontiguousarray(result, dtype=dtype).copy()
    frozen.setflags(write=False)
    return frozen


@dataclass(frozen=True)
class ControlAxisRow:
    control_offset: int
    control_group_id: str
    donor: str
    cell_type: str
    block: str
    block_rank: int
    row_index: int
    obs_index: str
    selection_sha256: str

    def __post_init__(self) -> None:
        for name in ("control_offset", "block_rank", "row_index"):
            object.__setattr__(self, name, _integer(getattr(self, name), name=name))
        for name in (
            "control_group_id",
            "donor",
            "cell_type",
            "block",
            "obs_index",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "selection_sha256",
            _sha256(self.selection_sha256, name="selection_sha256"),
        )
        if self.block not in BLOCK_ORDER:
            raise AbsoluteMaterializationError("control block is not B1/B2/B3")


@dataclass(frozen=True)
class RootTaskRow:
    root_task_offset: int
    control_group_id: str
    task_index: int
    task_id: str
    source_root_index: int
    root_id: str
    donor: str
    cytokine: str
    cell_type: str
    perturbed_cell_count: int
    matched_pbs_cell_count: int
    task_weight: str
    analysis_role: str
    analysis_root_index: int
    analysis_root_weight: str

    def __post_init__(self) -> None:
        for name in (
            "root_task_offset",
            "task_index",
            "source_root_index",
            "perturbed_cell_count",
            "matched_pbs_cell_count",
            "analysis_root_index",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name=name))
        for name in (
            "control_group_id",
            "task_id",
            "root_id",
            "donor",
            "cytokine",
            "cell_type",
            "task_weight",
            "analysis_role",
            "analysis_root_weight",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        if self.analysis_role != "EVALUATION":
            raise AbsoluteMaterializationError(
                "root-task analysis role is not EVALUATION"
            )

    @property
    def root_task_index(self) -> int:
        return self.root_task_offset

    @property
    def root_task_id(self) -> str:
        return f"{self.root_id}|{self.task_id}"

    @property
    def target_cytokine(self) -> str:
        return self.cytokine


@dataclass(frozen=True)
class ReferenceMember:
    reference_instance_id: str
    depth: int
    scheme: str
    rotation: str
    role: str
    control_group_id: str
    donor: str
    cell_type: str
    role_cell_rank: int
    control_offset: int
    block: str
    block_rank: int
    row_index: int
    obs_index: str

    def __post_init__(self) -> None:
        for name in (
            "depth",
            "role_cell_rank",
            "control_offset",
            "block_rank",
            "row_index",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name=name))
        for name in (
            "reference_instance_id",
            "scheme",
            "control_group_id",
            "role",
            "donor",
            "cell_type",
            "block",
            "obs_index",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        if self.rotation not in ("NA", "R0", "R1", "R2"):
            raise AbsoluteMaterializationError(
                "reference rotation is not NA/R0/R1/R2"
            )
        if self.role not in ROLE_ORDER:
            raise AbsoluteMaterializationError("reference role is not frozen")
        if self.block not in BLOCK_ORDER:
            raise AbsoluteMaterializationError("reference block is not frozen")
        expected_depth, expected_scheme, expected_rotation = _reference_spec(
            self.reference_instance_id
        )
        if (
            self.depth != expected_depth
            or self.scheme != expected_scheme
            or self.rotation != expected_rotation
        ):
            raise AbsoluteMaterializationError(
                "reference instance metadata differs from its ID"
            )

    @property
    def reference_instance_index(self) -> int:
        return REFERENCE_INSTANCE_IDS.index(self.reference_instance_id)

    @property
    def member_rank(self) -> int:
        return self.role_cell_rank

    @property
    def control_row_offset(self) -> int:
        return self.control_offset


def _reference_spec(instance_id: str) -> tuple[int, str, str]:
    match = _INSTANCE.fullmatch(instance_id)
    if match is None:
        raise AbsoluteMaterializationError("reference instance ID is invalid")
    raw_scheme = match.group("scheme")
    if raw_scheme.startswith("CROSSFIT_R"):
        return (
            int(match.group("depth")),
            "UNIT_AWARE_CROSSFIT",
            raw_scheme.rsplit("_", 1)[1],
        )
    return int(match.group("depth")), raw_scheme, "NA"


def _expected_role_blocks(scheme: str) -> Mapping[str, tuple[str, ...]]:
    if scheme == "NAIVE_SHARED":
        return MappingProxyType(
            {role: BLOCK_ORDER for role in ROLE_ORDER}
        )
    if scheme == "INDEPENDENT_SPLIT":
        return MappingProxyType(
            {"C_obs": ("B1",), "C_pred": ("B2",), "C_infer": ("B3",)}
        )
    rotation = int(scheme[-1])
    rotations = (
        ("B1", "B2", "B3"),
        ("B2", "B3", "B1"),
        ("B3", "B1", "B2"),
    )
    return MappingProxyType(
        dict(zip(ROLE_ORDER, ((block,) for block in rotations[rotation]), strict=True))
    )


@dataclass(frozen=True)
class EvalReferenceBundle:
    """In-memory EVALUATION-PBS tensor and membership bundle used before scoring."""

    controls: tuple[ControlAxisRow, ...]
    root_tasks: tuple[RootTaskRow, ...]
    memberships: tuple[ReferenceMember, ...]
    transformed_panel: np.ndarray
    raw_counts: sparse.csr_matrix
    source_feature_ids: tuple[str, ...]
    panel_feature_ids: tuple[str, ...]
    panel_source_indices: np.ndarray
    report_sha256: str
    production_geometry: bool = True

    def __post_init__(self) -> None:
        controls = tuple(self.controls)
        root_tasks = tuple(self.root_tasks)
        memberships = tuple(self.memberships)
        if not controls or not root_tasks or not memberships:
            raise AbsoluteMaterializationError("reference bundle axis is empty")
        if tuple(item.control_offset for item in controls) != tuple(
            range(len(controls))
        ):
            raise AbsoluteMaterializationError("control axis is not contiguous")
        if len({item.obs_index for item in controls}) != len(controls):
            raise AbsoluteMaterializationError("control obs_index is duplicated")
        if tuple(item.root_task_index for item in root_tasks) != tuple(
            range(len(root_tasks))
        ):
            raise AbsoluteMaterializationError("root-task axis is not contiguous")
        if len({item.root_task_id for item in root_tasks}) != len(root_tasks):
            raise AbsoluteMaterializationError("root-task ID is duplicated")

        panel_features = tuple(
            _text(value, name="panel_feature_id")
            for value in self.panel_feature_ids
        )
        source_features = tuple(
            _text(value, name="source_feature_id")
            for value in self.source_feature_ids
        )
        if (
            len(panel_features) != len(set(panel_features))
            or len(source_features) != len(set(source_features))
        ):
            raise AbsoluteMaterializationError("feature axis is duplicated")
        panel_indices = np.asarray(self.panel_source_indices, dtype="<i8")
        if (
            panel_indices.ndim != 1
            or panel_indices.size != len(panel_features)
            or np.any(panel_indices < 0)
            or np.any(panel_indices >= len(source_features))
            or len(np.unique(panel_indices)) != panel_indices.size
            or np.any(np.diff(panel_indices) <= 0)
        ):
            raise AbsoluteMaterializationError(
                "panel source indices are invalid or not strictly increasing"
            )
        if tuple(source_features[int(index)] for index in panel_indices) != panel_features:
            raise AbsoluteMaterializationError(
                "panel features differ from source-axis projection"
            )
        panel = _dense(
            self.transformed_panel,
            name="EVALUATION PBS transformed panel",
            dtype="<f4",
            shape=(len(controls), len(panel_features)),
            nonnegative=True,
        )
        raw = sparse.csr_matrix(self.raw_counts, dtype=np.float32, copy=True)
        raw.sort_indices()
        if raw.shape != (len(controls), len(source_features)):
            raise AbsoluteMaterializationError("EVALUATION PBS raw-count shape differs")
        if (
            not np.isfinite(raw.data).all()
            or np.any(raw.data < 0)
            or np.any(raw.data != np.floor(raw.data))
        ):
            raise AbsoluteMaterializationError(
                "EVALUATION PBS raw counts are not finite nonnegative integers"
            )
        if np.any(np.asarray(raw.sum(axis=1)).ravel() <= 0):
            raise AbsoluteMaterializationError("EVALUATION PBS has an empty count row")

        control_by_offset = {
            item.control_offset: item for item in controls
        }
        groups: dict[str, list[ControlAxisRow]] = {}
        for item in controls:
            groups.setdefault(item.control_group_id, []).append(item)
        group_ids = tuple(sorted(groups, key=lambda value: value.encode("utf-8")))
        root_task_groups = {item.control_group_id for item in root_tasks}
        if set(group_ids) != root_task_groups:
            raise AbsoluteMaterializationError(
                "control groups and root-task groups differ"
            )
        for group_id, values in groups.items():
            expected_root = {
                (item.donor, item.cell_type) for item in values
            }
            if len(expected_root) != 1:
                raise AbsoluteMaterializationError(
                    f"control group has conflicting identity: {group_id}"
                )
            for block in BLOCK_ORDER:
                block_rows = sorted(
                    (item for item in values if item.block == block),
                    key=lambda item: item.block_rank,
                )
                if [item.block_rank for item in block_rows] != list(
                    range(len(block_rows))
                ):
                    raise AbsoluteMaterializationError(
                        "control block ranks are not contiguous"
                    )

        task_identity: dict[int, tuple[str, str, str]] = {}
        root_identity: dict[int, tuple[str, str]] = {}
        root_tasks_per_task: dict[int, set[int]] = {}
        tasks_per_root: dict[int, set[int]] = {}
        for item in root_tasks:
            identity = (item.task_id, item.target_cytokine, item.cell_type)
            previous = task_identity.setdefault(item.task_index, identity)
            if previous != identity:
                raise AbsoluteMaterializationError(
                    "task index has conflicting identity"
                )
            root_tasks_per_task.setdefault(item.task_index, set()).add(
                item.analysis_root_index
            )
            tasks_per_root.setdefault(item.analysis_root_index, set()).add(
                item.task_index
            )
            root_previous = root_identity.setdefault(
                item.analysis_root_index, (item.root_id, item.donor)
            )
            if root_previous != (item.root_id, item.donor):
                raise AbsoluteMaterializationError(
                    "analysis root has conflicting identity"
                )
            group = groups[item.control_group_id]
            if {
                (row.donor, row.cell_type) for row in group
            } != {(item.donor, item.cell_type)}:
                raise AbsoluteMaterializationError(
                    "root-task and control group identity differ"
                )

        canonical_memberships = sorted(
            memberships,
            key=lambda item: (
                item.reference_instance_index,
                item.control_group_id.encode("utf-8"),
                ROLE_ORDER.index(item.role),
                item.member_rank,
            ),
        )
        membership_groups: dict[
            tuple[str, str, str], list[ReferenceMember]
        ] = {}
        for item in canonical_memberships:
            if (
                item.reference_instance_index >= len(REFERENCE_INSTANCE_IDS)
                or REFERENCE_INSTANCE_IDS[item.reference_instance_index]
                != item.reference_instance_id
            ):
                raise AbsoluteMaterializationError(
                    "reference instance index and ID differ"
                )
            control = control_by_offset.get(item.control_row_offset)
            if (
                control is None
                or control.obs_index != item.obs_index
                or control.control_group_id != item.control_group_id
                or control.donor != item.donor
                or control.cell_type != item.cell_type
                or control.block != item.block
                or control.block_rank != item.block_rank
                or control.row_index != item.row_index
            ):
                raise AbsoluteMaterializationError(
                    "reference member differs from control axis"
                )
            membership_groups.setdefault(
                (item.reference_instance_id, item.control_group_id, item.role),
                [],
            ).append(item)

        expected_membership_keys = {
            (instance, group, role)
            for instance in REFERENCE_INSTANCE_IDS
            for group in group_ids
            for role in ROLE_ORDER
        }
        if set(membership_groups) != expected_membership_keys:
            raise AbsoluteMaterializationError(
                "reference role family is incomplete or contains an extra group"
            )
        for instance in REFERENCE_INSTANCE_IDS:
            depth, scheme, rotation = _reference_spec(instance)
            blocks_by_role = _expected_role_blocks(
                f"CROSSFIT_{rotation}"
                if scheme == "UNIT_AWARE_CROSSFIT"
                else scheme
            )
            for group_id in group_ids:
                offsets_by_role: dict[str, tuple[int, ...]] = {}
                for role in ROLE_ORDER:
                    members = membership_groups[(instance, group_id, role)]
                    if [item.member_rank for item in members] != list(
                        range(len(members))
                    ):
                        raise AbsoluteMaterializationError(
                            "reference member ranks are not contiguous"
                        )
                    expected_blocks = blocks_by_role[role]
                    if scheme == "NAIVE_SHARED":
                        selected_controls = (
                            row
                            for row in groups[group_id]
                            if row.block_rank < depth
                        )
                    else:
                        selected_controls = (
                            row
                            for block in expected_blocks
                            for row in sorted(
                                (
                                    candidate
                                    for candidate in groups[group_id]
                                    if candidate.block == block
                                    and candidate.block_rank < depth
                                ),
                                key=lambda candidate: candidate.block_rank,
                            )
                        )
                    expected_controls = tuple(
                        item.control_offset for item in selected_controls
                    )
                    observed = tuple(item.control_row_offset for item in members)
                    if observed != expected_controls:
                        raise AbsoluteMaterializationError(
                            "reference membership differs from frozen block/depth rule"
                        )
                    offsets_by_role[role] = observed
                if scheme == "NAIVE_SHARED":
                    if len(set(offsets_by_role.values())) != 1:
                        raise AbsoluteMaterializationError(
                            "NAIVE_SHARED roles do not share one exact pool"
                        )
                elif any(
                    set(offsets_by_role[left]).intersection(offsets_by_role[right])
                    for left, right in (
                        ("C_obs", "C_pred"),
                        ("C_obs", "C_infer"),
                        ("C_pred", "C_infer"),
                    )
                ):
                    raise AbsoluteMaterializationError(
                        "independent reference roles overlap"
                    )

        if self.production_geometry:
            if (
                len(controls) != PRODUCTION_CONTROL_ROWS
                or len(group_ids) != PRODUCTION_CONTROL_GROUPS
                or len(root_tasks) != PRODUCTION_ROOT_TASKS
                or len(task_identity) != PRODUCTION_TASKS
                or len(root_identity) != PRODUCTION_ROOTS
                or len(panel_features) != PRODUCTION_PANEL_FEATURES
                or len(source_features) != PRODUCTION_SOURCE_FEATURES
            ):
                raise AbsoluteMaterializationError(
                    "Parse production EVALUATION-PBS geometry differs"
                )
            if any(
                len(values) != 144 for values in groups.values()
            ) or any(
                len(values) != PRODUCTION_ROOTS
                for values in root_tasks_per_task.values()
            ) or any(
                len(values) != PRODUCTION_TASKS
                for values in tasks_per_root.values()
            ):
                raise AbsoluteMaterializationError(
                    "Parse production group or crossed support differs"
                )

        membership_index = {
            key: tuple(value) for key, value in membership_groups.items()
        }
        object.__setattr__(self, "controls", controls)
        object.__setattr__(self, "root_tasks", root_tasks)
        object.__setattr__(self, "memberships", tuple(canonical_memberships))
        object.__setattr__(self, "transformed_panel", panel)
        object.__setattr__(self, "raw_counts", raw)
        object.__setattr__(self, "source_feature_ids", source_features)
        object.__setattr__(self, "panel_feature_ids", panel_features)
        frozen_indices = np.ascontiguousarray(panel_indices, dtype="<i8")
        frozen_indices.setflags(write=False)
        object.__setattr__(self, "panel_source_indices", frozen_indices)
        object.__setattr__(
            self, "report_sha256", _sha256(self.report_sha256, name="report_sha256")
        )
        object.__setattr__(
            self, "_control_by_offset", MappingProxyType(control_by_offset)
        )
        object.__setattr__(self, "_groups", MappingProxyType({
            key: tuple(value) for key, value in groups.items()
        }))
        object.__setattr__(
            self, "_membership_index", MappingProxyType(membership_index)
        )
        object.__setattr__(
            self,
            "_root_task_by_id",
            MappingProxyType({item.root_task_id: item for item in root_tasks}),
        )

    @property
    def root_task_ids(self) -> tuple[str, ...]:
        return tuple(item.root_task_id for item in self.root_tasks)

    def members(
        self,
        reference_instance_id: str,
        control_group_id: str,
        role: str,
    ) -> tuple[ReferenceMember, ...]:
        if role not in ROLE_ORDER:
            raise AbsoluteMaterializationError("reference role is not frozen")
        try:
            return self._membership_index[
                (reference_instance_id, control_group_id, role)
            ]
        except KeyError as error:
            raise AbsoluteMaterializationError(
                "reference membership is not frozen"
            ) from error

    def member_ids(
        self,
        root_task_id: str,
        reference_instance_id: str,
        role: str,
    ) -> tuple[str, ...]:
        task = self._root_task_by_id[root_task_id]
        return tuple(
            item.obs_index
            for item in self.members(
                reference_instance_id, task.control_group_id, role
            )
        )

    def panel_pool(
        self,
        root_task_id: str,
        reference_instance_id: str,
        role: str,
    ) -> np.ndarray:
        if role not in NUMERIC_ROLE_ORDER:
            raise AbsoluteMaterializationError(
                "C_obs has structural validation only and no numeric access"
            )
        task = self._root_task_by_id[root_task_id]
        offsets = [
            item.control_row_offset
            for item in self.members(
                reference_instance_id, task.control_group_id, role
            )
        ]
        result = np.ascontiguousarray(
            self.transformed_panel[np.asarray(offsets, dtype=np.int64)],
            dtype="<f4",
        )
        result.setflags(write=False)
        return result

    def raw_pool(
        self,
        root_task_id: str,
        reference_instance_id: str,
        role: str,
    ) -> sparse.csr_matrix:
        if role != "C_infer":
            raise AbsoluteMaterializationError(
                "CPA raw-count access is restricted to C_infer"
            )
        task = self._root_task_by_id[root_task_id]
        offsets = [
            item.control_row_offset
            for item in self.members(
                reference_instance_id, task.control_group_id, role
            )
        ]
        return self.raw_counts[np.asarray(offsets, dtype=np.int64)].tocsr(
            copy=True
        )

    def root_task(self, root_task_id: str) -> RootTaskRow:
        try:
            return self._root_task_by_id[root_task_id]
        except KeyError as error:
            raise AbsoluteMaterializationError(
                "root-task ID is not frozen"
            ) from error


def full_axis_native_to_panel(
    native_mean: object,
    panel_source_indices: Sequence[int],
    *,
    source_feature_count: int,
) -> np.ndarray:
    """Full-total CP10K-log1p followed by the frozen panel projection."""

    native = _dense(
        native_mean,
        name="CPA native negative-binomial mean",
        dtype="<f4",
        nonnegative=True,
    )
    if native.shape[1] != source_feature_count:
        raise AbsoluteMaterializationError("CPA native source feature axis differs")
    totals = native.sum(axis=1, dtype=np.float64)
    if not np.isfinite(totals).all() or np.any(totals <= 0):
        raise AbsoluteMaterializationError(
            "CPA native mean has a nonpositive full-axis total"
        )
    indices = np.asarray(panel_source_indices, dtype=np.int64)
    if (
        indices.ndim != 1
        or indices.size < 1
        or np.any(indices < 0)
        or np.any(indices >= source_feature_count)
        or len(np.unique(indices)) != len(indices)
    ):
        raise AbsoluteMaterializationError("CPA panel indices are invalid")
    panel = np.log1p(
        native[:, indices].astype(np.float64)
        * (CP10K_TOTAL / totals[:, None])
    )
    return _dense(
        panel,
        name="CPA transformed ABSOLUTE_NATIVE panel",
        dtype="<f4",
        shape=(native.shape[0], indices.size),
        nonnegative=True,
    )


@dataclass(frozen=True)
class CPAFullAxisTaskAdapter:
    """Bind a full-source-axis CPA terminal state to the common predictor API."""

    bundle: EvalReferenceBundle
    counterfactual_native_mean: Callable[
        [str, str, str, float, sparse.csr_matrix], object
    ]

    def __post_init__(self) -> None:
        if not callable(self.counterfactual_native_mean):
            raise AbsoluteMaterializationError(
                "CPA counterfactual predictor must be callable"
            )

    def __call__(
        self,
        root_task_id: str,
        reference_instance_id: str,
        c_infer_panel: np.ndarray,
    ) -> np.ndarray:
        expected_panel = self.bundle.panel_pool(
            root_task_id, reference_instance_id, "C_infer"
        )
        observed = _dense(
            c_infer_panel,
            name="CPA C_infer transformed panel",
            dtype="<f4",
            shape=expected_panel.shape,
            nonnegative=True,
        )
        if not np.array_equal(observed, expected_panel):
            raise AbsoluteMaterializationError(
                "CPA C_infer panel differs from its frozen membership"
            )
        task = self.bundle.root_task(root_task_id)
        raw = self.bundle.raw_pool(
            root_task_id, reference_instance_id, "C_infer"
        )
        try:
            native = self.counterfactual_native_mean(
                root_task_id,
                reference_instance_id,
                task.target_cytokine,
                1.0,
                raw,
            )
        except Exception as error:
            raise AbsoluteMaterializationError(
                f"CPA terminal state failed for {root_task_id}"
            ) from error
        return full_axis_native_to_panel(
            native,
            self.bundle.panel_source_indices,
            source_feature_count=len(self.bundle.source_feature_ids),
        )


@dataclass(frozen=True)
class AbsoluteSubfamilyAudit:
    status: str
    eligible: bool
    failures: tuple[str, ...]
    artifact_count: int
    replay_passed: bool
    family_sha256: str
    c_obs_numeric_access: bool = False
    scoring_interface_present: bool = False
    complete_eight_method_family_gate_open: bool = False


@dataclass(frozen=True)
class AbsoluteSubfamilyResult:
    artifacts: tuple[PredictionArtifact, ...]
    audit: AbsoluteSubfamilyAudit


def materialize_absolute_method(
    bundle: EvalReferenceBundle,
    *,
    method_id: str,
    state_sha256: str,
    predictor: Callable[[str, str, np.ndarray], object],
) -> tuple[PredictionArtifact, ...]:
    """Materialize all 15 native/effect pairs for one frozen ABSOLUTE state."""

    if method_id not in (CPA, SCGEN, "CELLOT_522D2B9_ABSOLUTE_V1"):
        raise AbsoluteMaterializationError("method is not a frozen ABSOLUTE member")
    state = _sha256(state_sha256, name="state_sha256")
    if not callable(predictor):
        raise AbsoluteMaterializationError("ABSOLUTE predictor is not callable")
    axes = bundle.root_task_ids
    artifacts: list[PredictionArtifact] = []
    for instance_id in REFERENCE_INSTANCE_IDS:
        c_infer = tuple(
            bundle.panel_pool(axis_id, instance_id, "C_infer")
            for axis_id in axes
        )
        c_pred = tuple(
            bundle.panel_pool(axis_id, instance_id, "C_pred")
            for axis_id in axes
        )
        infer_memberships = tuple(
            bundle.member_ids(axis_id, instance_id, "C_infer")
            for axis_id in axes
        )
        pred_memberships = tuple(
            bundle.member_ids(axis_id, instance_id, "C_pred")
            for axis_id in axes
        )
        native, effect = materialize_control_conditioned_absolute(
            method_id=method_id,
            reference_instance_id=instance_id,
            root_task_ids=axes,
            feature_ids=bundle.panel_feature_ids,
            c_infer_by_root_task=c_infer,
            c_pred_by_root_task=c_pred,
            c_infer_memberships=infer_memberships,
            c_pred_memberships=pred_memberships,
            state_sha256=state,
            predictor=predictor,
        )
        artifacts.extend((native, effect))
    return tuple(artifacts)


def _audit_subfamily_artifacts(
    artifacts: Sequence[PredictionArtifact],
    replay_artifacts: Sequence[PredictionArtifact],
    *,
    bundle: EvalReferenceBundle,
    state_sha256_by_method: Mapping[str, str],
) -> AbsoluteSubfamilyAudit:
    failures: list[str] = []
    expected_methods = (CPA, SCGEN)
    if tuple(state_sha256_by_method) != expected_methods:
        failures.append("CPA/scGen state family order or membership differs")
    expected_keys = tuple(
        (method, instance, output)
        for method in expected_methods
        for instance in REFERENCE_INSTANCE_IDS
        for output in ("ABSOLUTE_NATIVE", "PREDICTED_EFFECT")
    )
    observed_keys = tuple(
        (
            item.method_id,
            item.reference_instance_id,
            item.output_type,
        )
        for item in artifacts
    )
    replay_keys = tuple(
        (
            item.method_id,
            item.reference_instance_id,
            item.output_type,
        )
        for item in replay_artifacts
    )
    if observed_keys != expected_keys or replay_keys != expected_keys:
        failures.append("CPA/scGen artifact family is incomplete or reordered")
    replay_passed = True
    for expected, observed in zip(artifacts, replay_artifacts, strict=False):
        if (
            expected.axis_ids != bundle.root_task_ids
            or expected.feature_ids != bundle.panel_feature_ids
            or expected.state_sha256
            != state_sha256_by_method.get(expected.method_id)
            or expected.values.dtype != np.dtype("<f4")
        ):
            failures.append("prediction axis, state or dtype differs")
        result = audit_numerical_replay(
            expected.values,
            observed.values,
            exact_required=False,
        )
        if not result.passed:
            replay_passed = False
            failures.append(
                f"{expected.method_id}/{expected.reference_instance_id}/"
                f"{expected.output_type} replay differs"
            )
    by_key = {
        (item.method_id, item.reference_instance_id, item.output_type): item
        for item in artifacts
    }
    for method in expected_methods:
        for instance in REFERENCE_INSTANCE_IDS:
            native = by_key.get((method, instance, "ABSOLUTE_NATIVE"))
            effect = by_key.get((method, instance, "PREDICTED_EFFECT"))
            if native is None or effect is None or effect.c_pred_mean is None:
                failures.append(f"{method}/{instance} lacks one semantic pair")
                continue
            expected_effect = np.subtract(
                native.values, effect.c_pred_mean, dtype=np.float32
            )
            if (
                native.c_pred_subtractions != 0
                or effect.c_pred_subtractions != 1
                or not np.array_equal(effect.values, expected_effect)
                or native.c_infer_membership_sha256
                != effect.c_infer_membership_sha256
                or native.c_pred_membership_sha256
                != effect.c_pred_membership_sha256
            ):
                failures.append(f"{method}/{instance} violates ABSOLUTE semantics")
    family = hashlib.sha256(b"PARSE_CPA_SCGEN_ABSOLUTE_FAMILY_V1\0")
    for item in artifacts:
        family.update(bytes.fromhex(item.artifact_sha256))
    unique = tuple(dict.fromkeys(failures))
    return AbsoluteSubfamilyAudit(
        status=(
            "PASS_CPA_SCGEN_ABSOLUTE_SUBFAMILY_COMPLETE_NO_SCORING"
            if not unique
            else "NO_GO_CPA_SCGEN_ABSOLUTE_SUBFAMILY"
        ),
        eligible=not unique,
        failures=unique,
        artifact_count=len(artifacts),
        replay_passed=replay_passed and not unique,
        family_sha256=family.hexdigest(),
        c_obs_numeric_access=False,
        scoring_interface_present=False,
        complete_eight_method_family_gate_open=False,
    )


def materialize_cpa_scgen_family(
    bundle: EvalReferenceBundle,
    *,
    state_sha256_by_method: Mapping[str, str],
    cpa_predictor: Callable[[str, str, np.ndarray], object],
    scgen_predictor: Callable[[str, str, np.ndarray], object],
) -> AbsoluteSubfamilyResult:
    """Materialize and replay the indivisible CPA/scGen ABSOLUTE subfamily."""

    if tuple(state_sha256_by_method) != (CPA, SCGEN):
        raise AbsoluteMaterializationError(
            "CPA and scGen states must be supplied together in frozen order"
        )
    predictors = {CPA: cpa_predictor, SCGEN: scgen_predictor}
    first: list[PredictionArtifact] = []
    replay: list[PredictionArtifact] = []
    for method in (CPA, SCGEN):
        first.extend(
            materialize_absolute_method(
                bundle,
                method_id=method,
                state_sha256=state_sha256_by_method[method],
                predictor=predictors[method],
            )
        )
    for method in (CPA, SCGEN):
        replay.extend(
            materialize_absolute_method(
                bundle,
                method_id=method,
                state_sha256=state_sha256_by_method[method],
                predictor=predictors[method],
            )
        )
    audit = _audit_subfamily_artifacts(
        first,
        replay,
        bundle=bundle,
        state_sha256_by_method=state_sha256_by_method,
    )
    if not audit.eligible:
        raise AbsoluteMaterializationError(
            "CPA/scGen complete ABSOLUTE subfamily failed: "
            + "; ".join(audit.failures)
        )
    return AbsoluteSubfamilyResult(tuple(first), audit)


def artifact_manifest(result: AbsoluteSubfamilyResult) -> dict[str, Any]:
    """Return a JSON-safe pre-scoring manifest for the 60 output matrices."""

    return {
        "record_type": "PARSE_10M_CPA_SCGEN_ABSOLUTE_PREDICTION_FAMILY_V1",
        "status": result.audit.status,
        "access_boundary": {
            "evaluation_pbs_controls_read": True,
            "evaluation_perturbed_expression_read": False,
            "C_obs_numeric_access": False,
            "numeric_reference_roles": list(NUMERIC_ROLE_ORDER),
            "scores_utilities_winners_or_ranks_computed": False,
        },
        "geometry": {
            "root_task_rows": len(result.artifacts[0].axis_ids),
            "reference_instances": len(REFERENCE_INSTANCE_IDS),
            "methods": 2,
            "native_matrices": 30,
            "effect_matrices": 30,
            "feature_count": len(result.artifacts[0].feature_ids),
        },
        "artifacts": [
            {
                "method_id": item.method_id,
                "reference_instance_id": item.reference_instance_id,
                "output_type": item.output_type,
                "shape": list(item.values.shape),
                "dtype": item.values.dtype.str,
                "state_sha256": item.state_sha256,
                "c_pred_subtractions": item.c_pred_subtractions,
                "c_infer_membership_sha256": item.c_infer_membership_sha256,
                "c_pred_membership_sha256": item.c_pred_membership_sha256,
                "artifact_sha256": item.artifact_sha256,
            }
            for item in result.artifacts
        ],
        "subfamily_audit": {
            "eligible": result.audit.eligible,
            "replay_passed": result.audit.replay_passed,
            "family_sha256": result.audit.family_sha256,
            "complete_eight_method_family_gate_open": (
                result.audit.complete_eight_method_family_gate_open
            ),
            "scoring_interface_present": result.audit.scoring_interface_present,
        },
    }


__all__ = [
    "BLOCK_ORDER",
    "CONTROL_AXIS_COLUMNS",
    "CP10K_TOTAL",
    "CPAFullAxisTaskAdapter",
    "EvalReferenceBundle",
    "AbsoluteMaterializationError",
    "AbsoluteSubfamilyAudit",
    "AbsoluteSubfamilyResult",
    "NUMERIC_ROLE_ORDER",
    "PRODUCTION_CONTROL_GROUPS",
    "PRODUCTION_CONTROL_ROWS",
    "PRODUCTION_PANEL_FEATURES",
    "PRODUCTION_ROOT_TASKS",
    "PRODUCTION_SOURCE_FEATURES",
    "REFERENCE_MEMBERSHIP_COLUMNS",
    "ROLE_ORDER",
    "ROOT_TASK_AXIS_COLUMNS",
    "ControlAxisRow",
    "ReferenceMember",
    "RootTaskRow",
    "artifact_manifest",
    "full_axis_native_to_panel",
    "materialize_absolute_method",
    "materialize_cpa_scgen_family",
]
