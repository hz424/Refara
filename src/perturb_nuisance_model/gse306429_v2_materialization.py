"""Contracts for materializing GSE306429 V2 predictions without scoring.

This module has no H5AD, model-loader, scorer, utility, interval, winner, or
rank interface.  It fixes the reference-role dispatch, converts frozen DIRECT
and CPA ABSOLUTE outputs to prediction-effect artifacts, and audits whether a
complete prediction family is eligible for a later, separate scoring stage.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Callable, Mapping, Sequence

import numpy as np


REFERENCE_INSTANCE_IDS = (
    "D16_NAIVE_SHARED",
    "D16_INDEPENDENT_SPLIT",
    "D16_CROSSFIT_R0",
    "D16_CROSSFIT_R1",
    "D16_CROSSFIT_R2",
    "D32_NAIVE_SHARED",
    "D32_INDEPENDENT_SPLIT",
    "D32_CROSSFIT_R0",
    "D32_CROSSFIT_R1",
    "D32_CROSSFIT_R2",
    "D48_NAIVE_SHARED",
    "D48_INDEPENDENT_SPLIT",
    "D48_CROSSFIT_R0",
    "D48_CROSSFIT_R1",
    "D48_CROSSFIT_R2",
)
DEPTHS = (16, 32, 48)
BLOCK_IDS = ("B1", "B2", "B3")
ROLE_IDS = ("C_obs", "C_pred", "C_infer")

NO_CHANGE = "NO_CHANGE_DIRECT_V1"
CONTEXT_MEAN = "CONTEXT_MEAN_EFFECT_DIRECT_V1"
TWO_WAY_RIDGE = "TWO_WAY_ADDITIVE_RIDGE_DIRECT_V1"
PCA64_RIDGE = "PCA64_ADDITIVE_RIDGE_DIRECT_V1"
CPA = "CPA_0_8_8_ABSOLUTE_V1"
RBF_RIDGE = "RBF_KERNEL_RIDGE_DIRECT_V1"

METHOD_IDS = (
    NO_CHANGE,
    CONTEXT_MEAN,
    TWO_WAY_RIDGE,
    PCA64_RIDGE,
    CPA,
    RBF_RIDGE,
)
DIRECT_METHOD_IDS = (
    NO_CHANGE,
    CONTEXT_MEAN,
    TWO_WAY_RIDGE,
    PCA64_RIDGE,
    RBF_RIDGE,
)
TRAINABLE_METHOD_IDS = (
    CONTEXT_MEAN,
    TWO_WAY_RIDGE,
    PCA64_RIDGE,
    CPA,
    RBF_RIDGE,
)
FIT_IDS = tuple(f"train_rep1__eval_rep2__task_fold{fold}" for fold in range(5)) + tuple(
    f"train_rep2__eval_rep1__task_fold{fold}" for fold in range(5)
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_INSTANCE_RE = re.compile(
    r"^D(?P<depth>16|32|48)_(?:(?P<single>NAIVE_SHARED|INDEPENDENT_SPLIT)"
    r"|CROSSFIT_R(?P<rotation>[012]))$"
)
_FIT_RE = re.compile(
    r"^train_rep(?P<train>[12])__eval_rep(?P<eval>[12])__task_fold(?P<fold>[0-4])$"
)
_MISSING = frozenset({"", "na", "nan", "none", "null"})
_FORBIDDEN_SCORING_TERMS = (
    "utility",
    "score",
    "contrast",
    "interval",
    "winner",
    "best_model",
    "rank",
)
_EMPTY_MEMBERSHIP_SHA256 = hashlib.sha256(
    b"GSE306429_V2_EMPTY_REFERENCE_MEMBERSHIP_V1\0"
).hexdigest()


class GSE306429MaterializationError(ValueError):
    """Raised when a frozen V2 materialization constraint is violated."""


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, (str, np.str_)):
        raise GSE306429MaterializationError(f"{name} must be a string")
    result = str(value)
    if result != result.strip() or result.casefold() in _MISSING or "\0" in result:
        raise GSE306429MaterializationError(
            f"{name} is not a canonical non-empty string"
        )
    try:
        result.encode("utf-8")
    except UnicodeEncodeError as error:
        raise GSE306429MaterializationError(f"{name} is not valid UTF-8") from error
    return result


def _labels(
    values: Sequence[object], *, name: str, unique: bool = True
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise GSE306429MaterializationError(f"{name} must be a sequence")
    result = tuple(_text(value, name=name) for value in values)
    if not result:
        raise GSE306429MaterializationError(f"{name} cannot be empty")
    if unique and len(set(result)) != len(result):
        raise GSE306429MaterializationError(f"{name} must be unique")
    return result


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise GSE306429MaterializationError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise GSE306429MaterializationError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise GSE306429MaterializationError(f"{name} must be an integer") from error
    if result < 1 or (
        not isinstance(value, (int, np.integer)) and str(result) != str(value)
    ):
        raise GSE306429MaterializationError(
            f"{name} must be a canonical positive integer"
        )
    return result


def _float32_matrix(
    value: object, *, name: str, shape: tuple[int, int] | None = None
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype="<f4")
    except (TypeError, ValueError, OverflowError) as error:
        raise GSE306429MaterializationError(f"{name} must be numeric") from error
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise GSE306429MaterializationError(
            f"{name} must be a non-empty two-dimensional matrix"
        )
    if shape is not None and array.shape != shape:
        raise GSE306429MaterializationError(
            f"{name} shape {array.shape} differs from frozen shape {shape}"
        )
    if not np.isfinite(array).all():
        raise GSE306429MaterializationError(f"{name} contains a nonfinite value")
    result = np.ascontiguousarray(array, dtype="<f4")
    result.setflags(write=False)
    return result


def _float64_control_matrix(
    value: object, *, name: str, feature_count: int
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype="<f8")
    except (TypeError, ValueError, OverflowError) as error:
        raise GSE306429MaterializationError(f"{name} must be numeric") from error
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != feature_count:
        raise GSE306429MaterializationError(
            f"{name} must have shape (positive, {feature_count})"
        )
    if not np.isfinite(array).all():
        raise GSE306429MaterializationError(f"{name} contains a nonfinite value")
    result = np.ascontiguousarray(array, dtype="<f8")
    result.setflags(write=False)
    return result


def _canonical_json_sha256(label: bytes, value: object) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(label)
    digest.update(rendered)
    return digest.hexdigest()


def _reference_membership_sha256(
    atom_ids: Sequence[str], memberships: Sequence[Sequence[str]]
) -> str:
    return _canonical_json_sha256(
        b"GSE306429_V2_REFERENCE_MEMBERSHIP_V1\0",
        [
            [atom, list(cell_ids)]
            for atom, cell_ids in zip(atom_ids, memberships, strict=True)
        ],
    )


def reference_membership_sha256(
    atom_ids: Sequence[object], memberships: Sequence[Sequence[object]]
) -> str:
    """Hash one exact atom-by-reference membership table.

    This public wrapper is used by file-aware production runners.  It binds the
    explicit atom order and ordered cell IDs and never relies on mapping-key
    order after canonical JSON serialization.
    """

    atoms = _labels(atom_ids, name="atom_ids")
    if len(memberships) != len(atoms):
        raise GSE306429MaterializationError(
            "reference memberships must have one row per atom"
        )
    parsed = tuple(
        _labels(value, name="reference membership") for value in memberships
    )
    return _reference_membership_sha256(atoms, parsed)


@dataclass(frozen=True)
class FrozenReferenceCell:
    """One metadata-frozen DMSO cell and its block/rank assignment."""

    control_id: str
    library_id: str
    block_id: str
    block_rank: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "control_id", _text(self.control_id, name="control_id")
        )
        object.__setattr__(
            self, "library_id", _text(self.library_id, name="library_id")
        )
        if self.block_id not in BLOCK_IDS:
            raise GSE306429MaterializationError(f"block_id must be one of {BLOCK_IDS}")
        object.__setattr__(
            self,
            "block_rank",
            _positive_integer(self.block_rank, name="block_rank"),
        )


@dataclass(frozen=True)
class ReferenceMembership:
    """Exact C_obs/C_pred/C_infer membership for one registered instance."""

    reference_instance_id: str
    depth: int
    scheme: str
    rotation: int | None
    c_obs_ids: tuple[str, ...]
    c_pred_ids: tuple[str, ...]
    c_infer_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.reference_instance_id not in REFERENCE_INSTANCE_IDS:
            raise GSE306429MaterializationError("reference_instance_id is not frozen")
        match = _INSTANCE_RE.fullmatch(self.reference_instance_id)
        if match is None:
            raise GSE306429MaterializationError(
                "reference_instance_id cannot be parsed"
            )
        expected_depth = int(match.group("depth"))
        expected_scheme = match.group("single") or "UNIT_AWARE_CROSSFIT"
        expected_rotation = (
            None if match.group("rotation") is None else int(match.group("rotation"))
        )
        if (
            self.depth != expected_depth
            or self.scheme != expected_scheme
            or self.rotation != expected_rotation
        ):
            raise GSE306429MaterializationError(
                "reference metadata differs from reference_instance_id"
            )
        for name in ("c_obs_ids", "c_pred_ids", "c_infer_ids"):
            object.__setattr__(
                self,
                name,
                _labels(getattr(self, name), name=name),
            )

    def ids_for_role(self, role_id: str) -> tuple[str, ...]:
        """Return the frozen membership for one active reference role."""

        if role_id not in ROLE_IDS:
            raise GSE306429MaterializationError(f"role_id must be one of {ROLE_IDS}")
        return getattr(self, role_id.lower() + "_ids")


def build_reference_memberships(
    cells: Sequence[FrozenReferenceCell],
) -> tuple[ReferenceMembership, ...]:
    """Build all 15 memberships from one library's frozen B1/B2/B3 ranks."""

    parsed = tuple(
        cell if isinstance(cell, FrozenReferenceCell) else FrozenReferenceCell(**cell)
        for cell in cells
    )
    if not parsed:
        raise GSE306429MaterializationError("reference cell registry is empty")
    libraries = {cell.library_id for cell in parsed}
    if len(libraries) != 1:
        raise GSE306429MaterializationError(
            "one membership build must contain exactly one library"
        )
    control_ids = tuple(cell.control_id for cell in parsed)
    if len(control_ids) != len(set(control_ids)):
        raise GSE306429MaterializationError("control_id is not unique")

    blocks: dict[str, tuple[str, ...]] = {}
    for block_id in BLOCK_IDS:
        block_cells = sorted(
            (cell for cell in parsed if cell.block_id == block_id),
            key=lambda cell: cell.block_rank,
        )
        ranks = tuple(cell.block_rank for cell in block_cells)
        if ranks != tuple(range(1, len(ranks) + 1)):
            raise GSE306429MaterializationError(
                f"{block_id} ranks must be unique and contiguous from one"
            )
        if len(block_cells) < max(DEPTHS):
            raise GSE306429MaterializationError(
                f"{block_id} has fewer than {max(DEPTHS)} frozen controls"
            )
        blocks[block_id] = tuple(cell.control_id for cell in block_cells)

    memberships: list[ReferenceMembership] = []
    rotations = (
        ("B1", "B2", "B3"),
        ("B2", "B3", "B1"),
        ("B3", "B1", "B2"),
    )
    for instance_id in REFERENCE_INSTANCE_IDS:
        match = _INSTANCE_RE.fullmatch(instance_id)
        if match is None:  # pragma: no cover - constant self-check
            raise RuntimeError("invalid frozen reference instance constant")
        depth = int(match.group("depth"))
        prefix = {block: blocks[block][:depth] for block in BLOCK_IDS}
        if match.group("single") == "NAIVE_SHARED":
            shared = tuple(
                cell_id for block_id in BLOCK_IDS for cell_id in prefix[block_id]
            )
            role_ids = (shared, shared, shared)
            scheme = "NAIVE_SHARED"
            rotation = None
        elif match.group("single") == "INDEPENDENT_SPLIT":
            role_ids = (prefix["B1"], prefix["B2"], prefix["B3"])
            scheme = "INDEPENDENT_SPLIT"
            rotation = None
        else:
            rotation = int(match.group("rotation"))
            role_blocks = rotations[rotation]
            role_ids = tuple(prefix[block] for block in role_blocks)
            scheme = "UNIT_AWARE_CROSSFIT"
        memberships.append(
            ReferenceMembership(
                reference_instance_id=instance_id,
                depth=depth,
                scheme=scheme,
                rotation=rotation,
                c_obs_ids=role_ids[0],
                c_pred_ids=role_ids[1],
                c_infer_ids=role_ids[2],
            )
        )
    result = tuple(memberships)
    if tuple(item.reference_instance_id for item in result) != REFERENCE_INSTANCE_IDS:
        raise RuntimeError("reference membership order differs from frozen order")
    return result


@dataclass(frozen=True)
class PredictionArtifact:
    """One fixed-order, scorer-free prediction matrix and semantic trace."""

    method_id: str
    fit_id: str
    reference_instance_id: str
    atom_ids: tuple[str, ...]
    feature_ids: tuple[str, ...]
    values: np.ndarray
    state_sha256: str
    output_type: str
    source_output_type: str
    c_pred_subtractions: int
    uses_c_infer: bool
    c_infer_membership_sha256: str
    c_pred_membership_sha256: str
    c_pred_mean: np.ndarray | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise GSE306429MaterializationError("prediction schema_version must be one")
        if self.method_id not in METHOD_IDS:
            raise GSE306429MaterializationError("method_id is not in panel V2")
        if self.fit_id not in FIT_IDS:
            raise GSE306429MaterializationError("fit_id is not frozen")
        if self.reference_instance_id not in REFERENCE_INSTANCE_IDS:
            raise GSE306429MaterializationError("reference_instance_id is not frozen")
        atoms = _labels(self.atom_ids, name="atom_ids")
        features = _labels(self.feature_ids, name="feature_ids")
        values = _float32_matrix(
            self.values,
            name="prediction values",
            shape=(len(atoms), len(features)),
        )
        object.__setattr__(self, "atom_ids", atoms)
        object.__setattr__(self, "feature_ids", features)
        object.__setattr__(self, "values", values)
        _sha256(self.state_sha256, name="state_sha256")
        if self.output_type not in {"ABSOLUTE_NATIVE", "PREDICTED_EFFECT"}:
            raise GSE306429MaterializationError("output_type is not registered")
        if self.source_output_type not in {
            "ABSOLUTE_NATIVE",
            "PREDICTED_EFFECT",
        }:
            raise GSE306429MaterializationError("source_output_type is not registered")
        if self.c_pred_subtractions not in {0, 1}:
            raise GSE306429MaterializationError(
                "c_pred_subtractions must be zero or one"
            )
        _sha256(
            self.c_infer_membership_sha256,
            name="c_infer_membership_sha256",
        )
        _sha256(
            self.c_pred_membership_sha256,
            name="c_pred_membership_sha256",
        )
        if self.c_pred_mean is None:
            reference_mean = None
        else:
            reference_mean = _float32_matrix(
                self.c_pred_mean,
                name="C_pred mean",
                shape=(len(atoms), len(features)),
            )
        object.__setattr__(self, "c_pred_mean", reference_mean)

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (
            self.method_id,
            self.fit_id,
            self.reference_instance_id,
            self.output_type,
        )

    @property
    def artifact_sha256(self) -> str:
        header = {
            "schema_version": self.schema_version,
            "method_id": self.method_id,
            "fit_id": self.fit_id,
            "reference_instance_id": self.reference_instance_id,
            "atom_ids": self.atom_ids,
            "feature_ids": self.feature_ids,
            "dtype": "<f4",
            "shape": self.values.shape,
            "state_sha256": self.state_sha256,
            "output_type": self.output_type,
            "source_output_type": self.source_output_type,
            "c_pred_subtractions": self.c_pred_subtractions,
            "uses_c_infer": self.uses_c_infer,
            "c_infer_membership_sha256": self.c_infer_membership_sha256,
            "c_pred_membership_sha256": self.c_pred_membership_sha256,
            "has_c_pred_mean": self.c_pred_mean is not None,
        }
        digest = hashlib.sha256(b"GSE306429_V2_PREDICTION_ARTIFACT_V1\0")
        digest.update(
            json.dumps(
                header,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(self.values.tobytes(order="C"))
        if self.c_pred_mean is not None:
            digest.update(b"\0")
            digest.update(self.c_pred_mean.tobytes(order="C"))
        return digest.hexdigest()


def materialize_direct_effects(
    *,
    method_id: str,
    fit_id: str,
    atom_ids: Sequence[object],
    feature_ids: Sequence[object],
    predicted_effect: object,
    state_sha256: str,
) -> tuple[PredictionArtifact, ...]:
    """Tag a DIRECT effect and reproduce it unchanged for all 15 instances.

    The interface deliberately has no C_pred, C_infer, control, or reference
    matrix argument.  The exact same float32 bytes are retained for every
    registered instance; no reference subtraction is possible here.
    """

    if method_id not in DIRECT_METHOD_IDS:
        raise GSE306429MaterializationError(
            "materialize_direct_effects requires a frozen DIRECT method"
        )
    if fit_id not in FIT_IDS:
        raise GSE306429MaterializationError("fit_id is not frozen")
    atoms = _labels(atom_ids, name="atom_ids")
    features = _labels(feature_ids, name="feature_ids")
    values = _float32_matrix(
        predicted_effect,
        name="predicted_effect",
        shape=(len(atoms), len(features)),
    )
    state_digest = _sha256(state_sha256, name="state_sha256")
    return tuple(
        PredictionArtifact(
            method_id=method_id,
            fit_id=fit_id,
            reference_instance_id=instance_id,
            atom_ids=atoms,
            feature_ids=features,
            values=values,
            state_sha256=state_digest,
            output_type="PREDICTED_EFFECT",
            source_output_type="PREDICTED_EFFECT",
            c_pred_subtractions=0,
            uses_c_infer=False,
            c_infer_membership_sha256=_EMPTY_MEMBERSHIP_SHA256,
            c_pred_membership_sha256=_EMPTY_MEMBERSHIP_SHA256,
            c_pred_mean=None,
        )
        for instance_id in REFERENCE_INSTANCE_IDS
    )


CPAPredictor = Callable[[str, str, tuple[str, ...], np.ndarray], np.ndarray]


def materialize_cpa_effects(
    *,
    fit_id: str,
    atom_ids: Sequence[object],
    feature_ids: Sequence[object],
    controls_by_atom: Mapping[str, Mapping[str, object]],
    memberships_by_atom: Mapping[str, Sequence[ReferenceMembership]],
    absolute_predictor: CPAPredictor,
    state_sha256: str,
) -> tuple[tuple[PredictionArtifact, ...], tuple[PredictionArtifact, ...]]:
    """Dispatch every CPA C_infer pool and subtract one transformed C_pred.

    ``absolute_predictor`` is called once per atom and reference instance.  It
    receives the exact ordered C_infer cell prefix, never C_pred, and must
    return one already-transformed ABSOLUTE_NATIVE profile per control cell.
    Those profiles are averaged before exactly one mean of the separately
    transformed C_pred cells is subtracted.  All rotations remain separate.

    Returns ``(absolute_native_artifacts, predicted_effect_artifacts)`` in the
    exact registered reference-instance order.
    """

    if not callable(absolute_predictor):
        raise GSE306429MaterializationError("absolute_predictor must be callable")
    if fit_id not in FIT_IDS:
        raise GSE306429MaterializationError("fit_id is not frozen")
    atoms = _labels(atom_ids, name="atom_ids")
    features = _labels(feature_ids, name="feature_ids")
    state_digest = _sha256(state_sha256, name="state_sha256")
    if tuple(controls_by_atom) != atoms:
        raise GSE306429MaterializationError(
            "controls_by_atom keys must equal the frozen atom order"
        )
    if tuple(memberships_by_atom) != atoms:
        raise GSE306429MaterializationError(
            "memberships_by_atom keys must equal the frozen atom order"
        )

    canonical_memberships: dict[str, tuple[ReferenceMembership, ...]] = {}
    canonical_controls: dict[str, dict[str, np.ndarray]] = {}
    for atom_id in atoms:
        memberships = tuple(memberships_by_atom[atom_id])
        if tuple(item.reference_instance_id for item in memberships) != (
            REFERENCE_INSTANCE_IDS
        ):
            raise GSE306429MaterializationError(
                f"{atom_id} membership order is not the frozen 15-instance order"
            )
        canonical_memberships[atom_id] = memberships
        control_mapping = controls_by_atom[atom_id]
        if not isinstance(control_mapping, Mapping) or not control_mapping:
            raise GSE306429MaterializationError(
                f"{atom_id} control profile mapping is empty"
            )
        parsed_controls: dict[str, np.ndarray] = {}
        for raw_id, raw_profile in control_mapping.items():
            control_id = _text(raw_id, name="control_id")
            if control_id in parsed_controls:
                raise GSE306429MaterializationError(
                    f"{atom_id} contains duplicate control IDs"
                )
            profile = _float64_control_matrix(
                np.asarray(raw_profile).reshape(1, -1),
                name=f"{atom_id} control profile",
                feature_count=len(features),
            )[0].copy()
            profile.setflags(write=False)
            parsed_controls[control_id] = profile
        required = {
            cell_id
            for membership in memberships
            for role_id in ROLE_IDS
            for cell_id in membership.ids_for_role(role_id)
        }
        missing = sorted(required.difference(parsed_controls))
        if missing:
            raise GSE306429MaterializationError(
                f"{atom_id} lacks registered control profiles: {missing[:3]}"
            )
        canonical_controls[atom_id] = parsed_controls

    native_artifacts: list[PredictionArtifact] = []
    effect_artifacts: list[PredictionArtifact] = []
    for instance_index, instance_id in enumerate(REFERENCE_INSTANCE_IDS):
        native_rows: list[np.ndarray] = []
        c_pred_rows: list[np.ndarray] = []
        infer_memberships: list[tuple[str, ...]] = []
        pred_memberships: list[tuple[str, ...]] = []
        for atom_id in atoms:
            membership = canonical_memberships[atom_id][instance_index]
            c_infer_ids = membership.c_infer_ids
            c_pred_ids = membership.c_pred_ids
            controls = canonical_controls[atom_id]
            c_infer = np.ascontiguousarray(
                np.vstack([controls[cell_id] for cell_id in c_infer_ids]),
                dtype="<f8",
            )
            c_infer.setflags(write=False)
            try:
                predicted_cells = absolute_predictor(
                    atom_id,
                    instance_id,
                    c_infer_ids,
                    c_infer,
                )
            except Exception as error:
                raise GSE306429MaterializationError(
                    f"CPA ABSOLUTE_NATIVE prediction failed for {atom_id} {instance_id}"
                ) from error
            absolute_cells = _float64_control_matrix(
                predicted_cells,
                name="CPA per-control ABSOLUTE_NATIVE predictions",
                feature_count=len(features),
            )
            if absolute_cells.shape[0] != len(c_infer_ids):
                raise GSE306429MaterializationError(
                    "CPA must return one absolute profile per C_infer cell"
                )
            c_pred = np.ascontiguousarray(
                np.vstack([controls[cell_id] for cell_id in c_pred_ids]),
                dtype="<f8",
            )
            native_rows.append(absolute_cells.mean(axis=0, dtype=np.float64))
            c_pred_rows.append(c_pred.mean(axis=0, dtype=np.float64))
            infer_memberships.append(c_infer_ids)
            pred_memberships.append(c_pred_ids)

        native_values = _float32_matrix(
            np.vstack(native_rows),
            name="CPA mean ABSOLUTE_NATIVE",
            shape=(len(atoms), len(features)),
        )
        c_pred_mean = _float32_matrix(
            np.vstack(c_pred_rows),
            name="CPA transformed C_pred mean",
            shape=(len(atoms), len(features)),
        )
        effect_values = _float32_matrix(
            np.subtract(native_values, c_pred_mean, dtype=np.float32),
            name="CPA PREDICTED_EFFECT",
            shape=(len(atoms), len(features)),
        )
        infer_digest = _reference_membership_sha256(atoms, infer_memberships)
        pred_digest = _reference_membership_sha256(atoms, pred_memberships)
        native_artifacts.append(
            PredictionArtifact(
                method_id=CPA,
                fit_id=fit_id,
                reference_instance_id=instance_id,
                atom_ids=atoms,
                feature_ids=features,
                values=native_values,
                state_sha256=state_digest,
                output_type="ABSOLUTE_NATIVE",
                source_output_type="ABSOLUTE_NATIVE",
                c_pred_subtractions=0,
                uses_c_infer=True,
                c_infer_membership_sha256=infer_digest,
                c_pred_membership_sha256=pred_digest,
                c_pred_mean=None,
            )
        )
        effect_artifacts.append(
            PredictionArtifact(
                method_id=CPA,
                fit_id=fit_id,
                reference_instance_id=instance_id,
                atom_ids=atoms,
                feature_ids=features,
                values=effect_values,
                state_sha256=state_digest,
                output_type="PREDICTED_EFFECT",
                source_output_type="ABSOLUTE_NATIVE",
                c_pred_subtractions=1,
                uses_c_infer=True,
                c_infer_membership_sha256=infer_digest,
                c_pred_membership_sha256=pred_digest,
                c_pred_mean=c_pred_mean,
            )
        )
    return tuple(native_artifacts), tuple(effect_artifacts)


def finalize_cpa_reference_matrices(
    *,
    fit_id: str,
    reference_instance_id: str,
    atom_ids: Sequence[object],
    feature_ids: Sequence[object],
    absolute_native: object,
    c_pred_mean: object,
    state_sha256: str,
    c_infer_memberships: Sequence[Sequence[object]],
    c_pred_memberships: Sequence[Sequence[object]],
) -> tuple[PredictionArtifact, PredictionArtifact]:
    """Freeze one batched CPA reference result and subtract C_pred once.

    The predictor is not implemented in this module. The function accepts an
    already transformed ABSOLUTE_NATIVE matrix, converts both inputs to the
    registered float32 representation, performs exactly one float32 subtraction,
    and returns the native/effect semantic artifacts.
    """

    if fit_id not in FIT_IDS:
        raise GSE306429MaterializationError("fit_id is not frozen")
    if reference_instance_id not in REFERENCE_INSTANCE_IDS:
        raise GSE306429MaterializationError(
            "reference_instance_id is not frozen"
        )
    atoms = _labels(atom_ids, name="atom_ids")
    features = _labels(feature_ids, name="feature_ids")
    shape = (len(atoms), len(features))
    native = _float32_matrix(
        absolute_native, name="CPA mean ABSOLUTE_NATIVE", shape=shape
    )
    pred_mean = _float32_matrix(
        c_pred_mean, name="CPA transformed C_pred mean", shape=shape
    )
    effect = _float32_matrix(
        np.subtract(native, pred_mean, dtype=np.float32),
        name="CPA PREDICTED_EFFECT",
        shape=shape,
    )
    infer_digest = reference_membership_sha256(atoms, c_infer_memberships)
    pred_digest = reference_membership_sha256(atoms, c_pred_memberships)
    state_digest = _sha256(state_sha256, name="state_sha256")
    native_artifact = PredictionArtifact(
        method_id=CPA,
        fit_id=fit_id,
        reference_instance_id=reference_instance_id,
        atom_ids=atoms,
        feature_ids=features,
        values=native,
        state_sha256=state_digest,
        output_type="ABSOLUTE_NATIVE",
        source_output_type="ABSOLUTE_NATIVE",
        c_pred_subtractions=0,
        uses_c_infer=True,
        c_infer_membership_sha256=infer_digest,
        c_pred_membership_sha256=pred_digest,
        c_pred_mean=None,
    )
    effect_artifact = PredictionArtifact(
        method_id=CPA,
        fit_id=fit_id,
        reference_instance_id=reference_instance_id,
        atom_ids=atoms,
        feature_ids=features,
        values=effect,
        state_sha256=state_digest,
        output_type="PREDICTED_EFFECT",
        source_output_type="ABSOLUTE_NATIVE",
        c_pred_subtractions=1,
        uses_c_infer=True,
        c_infer_membership_sha256=infer_digest,
        c_pred_membership_sha256=pred_digest,
        c_pred_mean=pred_mean,
    )
    return native_artifact, effect_artifact


@dataclass(frozen=True)
class NoFitBinding:
    """Replay-bound state identity for the sole NO_FIT family member."""

    method_id: str
    feature_ids: tuple[str, ...]
    source_sha256: str
    environment_digest: str
    binding_sha256: str
    replay_binding_sha256: str

    def __post_init__(self) -> None:
        if self.method_id != NO_CHANGE:
            raise GSE306429MaterializationError(
                "the only NO_FIT binding is NO_CHANGE_DIRECT_V1"
            )
        object.__setattr__(
            self, "feature_ids", _labels(self.feature_ids, name="feature_ids")
        )
        for name in (
            "source_sha256",
            "environment_digest",
            "binding_sha256",
            "replay_binding_sha256",
        ):
            _sha256(getattr(self, name), name=name)


@dataclass(frozen=True)
class StateArtifact:
    """One registered fitted-state identity and its leakage/replay evidence."""

    method_id: str
    fit_id: str
    training_row_manifest_sha256: str
    source_sha256: str
    environment_digest: str
    state_sha256: str
    replay_state_sha256: str
    seed: int
    terminal_epoch: int
    training_root_ids: tuple[str, ...]
    evaluation_root_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.method_id not in TRAINABLE_METHOD_IDS:
            raise GSE306429MaterializationError(
                "state method_id is not a trainable panel member"
            )
        if self.fit_id not in FIT_IDS:
            raise GSE306429MaterializationError("fit_id is not frozen")
        for name in (
            "training_row_manifest_sha256",
            "source_sha256",
            "environment_digest",
            "state_sha256",
            "replay_state_sha256",
        ):
            _sha256(getattr(self, name), name=name)
        if isinstance(self.seed, bool) or int(self.seed) != self.seed:
            raise GSE306429MaterializationError("seed must be an integer")
        if (
            isinstance(self.terminal_epoch, bool)
            or int(self.terminal_epoch) != self.terminal_epoch
            or self.terminal_epoch < 0
        ):
            raise GSE306429MaterializationError(
                "terminal_epoch must be a nonnegative integer"
            )
        training = _labels(self.training_root_ids, name="training_root_ids")
        evaluation = _labels(self.evaluation_root_ids, name="evaluation_root_ids")
        object.__setattr__(self, "training_root_ids", training)
        object.__setattr__(self, "evaluation_root_ids", evaluation)

    @property
    def key(self) -> tuple[str, str]:
        return (self.method_id, self.fit_id)


@dataclass(frozen=True)
class PredictionReplay:
    """Independent replay digest for one materialized prediction artifact."""

    method_id: str
    fit_id: str
    reference_instance_id: str
    output_type: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        if self.method_id not in METHOD_IDS:
            raise GSE306429MaterializationError("replay method_id is not frozen")
        if self.fit_id not in FIT_IDS:
            raise GSE306429MaterializationError("replay fit_id is not frozen")
        if self.reference_instance_id not in REFERENCE_INSTANCE_IDS:
            raise GSE306429MaterializationError(
                "replay reference_instance_id is not frozen"
            )
        if self.output_type not in {"ABSOLUTE_NATIVE", "PREDICTED_EFFECT"}:
            raise GSE306429MaterializationError("replay output_type is not registered")
        _sha256(self.artifact_sha256, name="artifact_sha256")

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (
            self.method_id,
            self.fit_id,
            self.reference_instance_id,
            self.output_type,
        )


@dataclass(frozen=True)
class CompleteFamilyEligibilityReport:
    """Machine-readable eligibility result without a performance evaluation."""

    status: str
    complete_family_eligible: bool
    scoring_performed: bool
    failures: tuple[str, ...]
    no_fit_binding_count: int
    fitted_state_count: int
    predicted_effect_count: int
    absolute_native_count: int
    replay_count: int


def _expected_state_keys() -> tuple[tuple[str, str], ...]:
    return tuple(
        (method_id, fit_id) for method_id in TRAINABLE_METHOD_IDS for fit_id in FIT_IDS
    )


def _expected_effect_keys() -> tuple[tuple[str, str, str, str], ...]:
    return tuple(
        (method_id, fit_id, reference_id, "PREDICTED_EFFECT")
        for method_id in METHOD_IDS
        for fit_id in FIT_IDS
        for reference_id in REFERENCE_INSTANCE_IDS
    )


def _expected_native_keys() -> tuple[tuple[str, str, str, str], ...]:
    return tuple(
        (CPA, fit_id, reference_id, "ABSOLUTE_NATIVE")
        for fit_id in FIT_IDS
        for reference_id in REFERENCE_INSTANCE_IDS
    )


def audit_complete_family_eligibility(
    *,
    no_fit_bindings: Sequence[NoFitBinding],
    fitted_states: Sequence[StateArtifact],
    predicted_effects: Sequence[PredictionArtifact],
    absolute_native_predictions: Sequence[PredictionArtifact],
    prediction_replays: Sequence[PredictionReplay],
    expected_atom_ids_by_fit: Mapping[str, Sequence[object]],
    expected_feature_ids: Sequence[object],
    expected_training_roots_by_fit: Mapping[str, Sequence[object]],
    expected_evaluation_roots_by_fit: Mapping[str, Sequence[object]],
    produced_artifact_names: Sequence[object] = (),
) -> CompleteFamilyEligibilityReport:
    """Audit the frozen 1-NO_FIT + 50-state complete prediction family.

    This function does not read prediction performance and cannot score. A
    PASS only establishes eligibility for later scoring with the separately
    specified scorer.
    """

    failures: list[str] = []
    features = _labels(expected_feature_ids, name="expected_feature_ids")
    try:
        atom_map = {
            fit_id: _labels(expected_atom_ids_by_fit[fit_id], name="atom_ids")
            for fit_id in FIT_IDS
        }
    except KeyError as error:
        raise GSE306429MaterializationError(
            f"expected_atom_ids_by_fit lacks {error.args[0]}"
        ) from error
    if tuple(expected_atom_ids_by_fit) != FIT_IDS:
        failures.append("expected atom map does not use the exact frozen fit order")

    try:
        training_roots = {
            fit_id: _labels(
                expected_training_roots_by_fit[fit_id],
                name="expected_training_roots",
            )
            for fit_id in FIT_IDS
        }
        evaluation_roots = {
            fit_id: _labels(
                expected_evaluation_roots_by_fit[fit_id],
                name="expected_evaluation_roots",
            )
            for fit_id in FIT_IDS
        }
    except KeyError as error:
        raise GSE306429MaterializationError(
            f"expected root map lacks {error.args[0]}"
        ) from error
    if tuple(expected_training_roots_by_fit) != FIT_IDS:
        failures.append("training-root map does not use the exact frozen fit order")
    if tuple(expected_evaluation_roots_by_fit) != FIT_IDS:
        failures.append("evaluation-root map does not use the exact frozen fit order")
    for fit_id in FIT_IDS:
        if set(training_roots[fit_id]).intersection(evaluation_roots[fit_id]):
            failures.append(f"{fit_id} expected training/evaluation roots overlap")

    no_fit = tuple(no_fit_bindings)
    if len(no_fit) != 1:
        failures.append("complete family requires exactly one NO_FIT binding")
    elif (
        no_fit[0].method_id != NO_CHANGE
        or no_fit[0].feature_ids != features
        or no_fit[0].binding_sha256 != no_fit[0].replay_binding_sha256
    ):
        failures.append("NO_FIT binding identity, feature order, or replay failed")

    states = tuple(fitted_states)
    expected_state_keys = _expected_state_keys()
    observed_state_keys = tuple(state.key for state in states)
    if observed_state_keys != expected_state_keys:
        failures.append(
            "fitted states are incomplete, duplicated, extra, or out of order"
        )
    state_by_key = {state.key: state for state in states}
    if len(state_by_key) != len(states):
        failures.append("fitted state keys are duplicated")
    terminal_epochs = {
        CONTEXT_MEAN: 0,
        TWO_WAY_RIDGE: 0,
        PCA64_RIDGE: 0,
        CPA: 24,
        RBF_RIDGE: 0,
    }
    for key in expected_state_keys:
        state = state_by_key.get(key)
        if state is None:
            continue
        method_id, fit_id = key
        if state.state_sha256 != state.replay_state_sha256:
            failures.append(f"{method_id}/{fit_id} state replay differs")
        if state.seed != 17 or state.terminal_epoch != terminal_epochs[method_id]:
            failures.append(f"{method_id}/{fit_id} frozen schedule differs")
        if state.training_root_ids != training_roots[fit_id]:
            failures.append(f"{method_id}/{fit_id} training roots differ")
        if state.evaluation_root_ids != evaluation_roots[fit_id]:
            failures.append(f"{method_id}/{fit_id} evaluation roots differ")
        if set(state.training_root_ids).intersection(state.evaluation_root_ids):
            failures.append(f"{method_id}/{fit_id} has root leakage")

    effects = tuple(predicted_effects)
    natives = tuple(absolute_native_predictions)
    expected_effect_keys = _expected_effect_keys()
    expected_native_keys = _expected_native_keys()
    observed_effect_keys = tuple(artifact.key for artifact in effects)
    observed_native_keys = tuple(artifact.key for artifact in natives)
    if observed_effect_keys != expected_effect_keys:
        failures.append(
            "PREDICTED_EFFECT family is incomplete, extra, duplicated, or out of order"
        )
    if observed_native_keys != expected_native_keys:
        failures.append(
            "CPA ABSOLUTE_NATIVE family is incomplete, extra, duplicated, or out of order"
        )
    effect_by_key = {artifact.key: artifact for artifact in effects}
    native_by_key = {artifact.key: artifact for artifact in natives}
    if len(effect_by_key) != len(effects):
        failures.append("PREDICTED_EFFECT keys are duplicated")
    if len(native_by_key) != len(natives):
        failures.append("ABSOLUTE_NATIVE keys are duplicated")

    no_fit_sha = no_fit[0].binding_sha256 if len(no_fit) == 1 else None
    for key in expected_effect_keys:
        artifact = effect_by_key.get(key)
        if artifact is None:
            continue
        method_id, fit_id, _, _ = key
        if artifact.atom_ids != atom_map[fit_id]:
            failures.append(f"{method_id}/{fit_id} atom order differs")
        if artifact.feature_ids != features:
            failures.append(f"{method_id}/{fit_id} feature order differs")
        if artifact.values.dtype != np.dtype("<f4"):
            failures.append(f"{method_id}/{fit_id} prediction dtype differs")
        if not np.isfinite(artifact.values).all():
            failures.append(f"{method_id}/{fit_id} prediction is nonfinite")
        expected_state_sha = (
            no_fit_sha
            if method_id == NO_CHANGE
            else (
                state_by_key[(method_id, fit_id)].state_sha256
                if (method_id, fit_id) in state_by_key
                else None
            )
        )
        if artifact.state_sha256 != expected_state_sha:
            failures.append(f"{method_id}/{fit_id} prediction state binding differs")
        if method_id in DIRECT_METHOD_IDS:
            if (
                artifact.output_type != "PREDICTED_EFFECT"
                or artifact.source_output_type != "PREDICTED_EFFECT"
                or artifact.c_pred_subtractions != 0
                or artifact.uses_c_infer
                or artifact.c_pred_mean is not None
                or artifact.c_pred_membership_sha256 != _EMPTY_MEMBERSHIP_SHA256
                or artifact.c_infer_membership_sha256 != _EMPTY_MEMBERSHIP_SHA256
            ):
                failures.append(
                    f"{method_id}/{fit_id} violates DIRECT reference invariance"
                )
        else:
            if (
                artifact.source_output_type != "ABSOLUTE_NATIVE"
                or artifact.c_pred_subtractions != 1
                or not artifact.uses_c_infer
                or artifact.c_pred_mean is None
            ):
                failures.append(f"{method_id}/{fit_id} violates CPA effect semantics")

    for method_id in DIRECT_METHOD_IDS:
        for fit_id in FIT_IDS:
            artifacts = [
                effect_by_key.get((method_id, fit_id, reference_id, "PREDICTED_EFFECT"))
                for reference_id in REFERENCE_INSTANCE_IDS
            ]
            if any(artifact is None for artifact in artifacts):
                continue
            first = artifacts[0]
            assert first is not None
            if any(
                not np.array_equal(first.values, artifact.values)
                for artifact in artifacts[1:]
                if artifact is not None
            ):
                failures.append(
                    f"{method_id}/{fit_id} DIRECT values vary by reference instance"
                )

    for key in expected_native_keys:
        native = native_by_key.get(key)
        effect_key = (key[0], key[1], key[2], "PREDICTED_EFFECT")
        effect = effect_by_key.get(effect_key)
        if native is None or effect is None:
            continue
        method_id, fit_id, reference_id, _ = key
        if (
            native.atom_ids != atom_map[fit_id]
            or native.feature_ids != features
            or native.values.dtype != np.dtype("<f4")
            or not np.isfinite(native.values).all()
        ):
            failures.append(
                f"{method_id}/{fit_id}/{reference_id} native axis/dtype/finiteness differs"
            )
        if (
            native.state_sha256 != effect.state_sha256
            or native.source_output_type != "ABSOLUTE_NATIVE"
            or native.c_pred_subtractions != 0
            or not native.uses_c_infer
            or native.c_pred_mean is not None
            or native.c_infer_membership_sha256 != effect.c_infer_membership_sha256
            or native.c_pred_membership_sha256 != effect.c_pred_membership_sha256
        ):
            failures.append(
                f"{method_id}/{fit_id}/{reference_id} CPA native trace differs"
            )
        if effect.c_pred_mean is not None:
            reconstructed = np.subtract(
                native.values,
                effect.c_pred_mean,
                dtype=np.float32,
            )
            if not np.array_equal(reconstructed, effect.values):
                failures.append(
                    f"{method_id}/{fit_id}/{reference_id} is not exactly "
                    "ABSOLUTE_NATIVE minus one C_pred mean"
                )

    replays = tuple(prediction_replays)
    expected_replay_keys = expected_effect_keys + expected_native_keys
    observed_replay_keys = tuple(replay.key for replay in replays)
    if observed_replay_keys != expected_replay_keys:
        failures.append(
            "prediction replays are incomplete, extra, duplicated, or out of order"
        )
    replay_by_key = {replay.key: replay for replay in replays}
    if len(replay_by_key) != len(replays):
        failures.append("prediction replay keys are duplicated")
    all_artifacts = {**effect_by_key, **native_by_key}
    for key in expected_replay_keys:
        replay = replay_by_key.get(key)
        artifact = all_artifacts.get(key)
        if (
            replay is not None
            and artifact is not None
            and replay.artifact_sha256 != artifact.artifact_sha256
        ):
            failures.append("/".join(key) + " replay hash differs")

    for raw_name in produced_artifact_names:
        name = _text(raw_name, name="produced_artifact_name")
        normalized = name.casefold().replace("-", "_").replace(" ", "_")
        if any(term in normalized for term in _FORBIDDEN_SCORING_TERMS):
            failures.append(f"scoring-stage artifact was produced: {name}")

    unique_failures = tuple(dict.fromkeys(failures))
    passed = not unique_failures
    return CompleteFamilyEligibilityReport(
        status="PASS_COMPLETE_FAMILY_ELIGIBLE" if passed else "NO_GO",
        complete_family_eligible=passed,
        scoring_performed=False,
        failures=unique_failures,
        no_fit_binding_count=len(no_fit),
        fitted_state_count=len(states),
        predicted_effect_count=len(effects),
        absolute_native_count=len(natives),
        replay_count=len(replays),
    )


__all__ = [
    "BLOCK_IDS",
    "CONTEXT_MEAN",
    "CPA",
    "CompleteFamilyEligibilityReport",
    "DEPTHS",
    "DIRECT_METHOD_IDS",
    "FIT_IDS",
    "FrozenReferenceCell",
    "GSE306429MaterializationError",
    "METHOD_IDS",
    "NO_CHANGE",
    "NoFitBinding",
    "PCA64_RIDGE",
    "PredictionArtifact",
    "PredictionReplay",
    "RBF_RIDGE",
    "REFERENCE_INSTANCE_IDS",
    "ROLE_IDS",
    "ReferenceMembership",
    "StateArtifact",
    "TRAINABLE_METHOD_IDS",
    "TWO_WAY_RIDGE",
    "audit_complete_family_eligibility",
    "build_reference_memberships",
    "finalize_cpa_reference_matrices",
    "materialize_cpa_effects",
    "materialize_direct_effects",
    "reference_membership_sha256",
]
