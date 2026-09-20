"""Adapter for the fixed-perturbation eight-method family.

This module contains no dataset reader, model runtime, or scoring interface.
It makes the training/evaluation geometry explicit while preserving the
frozen M8 order and DIRECT-versus-ABSOLUTE reference semantics used by the
public conditional applications.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np

from perturb_nuisance_model.gse306429_v2_baselines import (
    CONTEXT_MEAN,
    NO_CHANGE,
    PCA64_RIDGE,
    RBF_RIDGE,
    TWO_WAY_RIDGE,
    DirectBaselineState,
    canonical_state_sha256,
    fit_internal_direct_method,
    make_no_change_direct_state,
    predict_internal_direct,
)


CPA = "CPA_0_8_8_ABSOLUTE_V1"
SCGEN = "SCGEN_2_1_1_ABSOLUTE_V1"
CELLOT = "CELLOT_522D2B9_ABSOLUTE_V1"
DIRECT_METHOD_IDS = (NO_CHANGE, CONTEXT_MEAN, TWO_WAY_RIDGE, PCA64_RIDGE, RBF_RIDGE)
ABSOLUTE_METHOD_IDS = (CPA, SCGEN, CELLOT)
METHOD_IDS = DIRECT_METHOD_IDS + ABSOLUTE_METHOD_IDS
REFERENCE_INSTANCE_IDS = (
    "D8_NAIVE_SHARED",
    "D8_INDEPENDENT_SPLIT",
    "D8_CROSSFIT_R0",
    "D8_CROSSFIT_R1",
    "D8_CROSSFIT_R2",
)
SCHEME_IDS = ("NAIVE_SHARED", "INDEPENDENT_SPLIT", "UNIT_AWARE_CROSSFIT")
REFERENCE_POOL_SIZE = MappingProxyType(
    {instance: (24 if instance == "D8_NAIVE_SHARED" else 8) for instance in REFERENCE_INSTANCE_IDS}
)
_MISSING = frozenset({"", "na", "nan", "none", "null"})


class FixedPerturbationM8Error(ValueError):
    """A frozen geometry or prediction-semantics invariant was violated."""


def _label(value: object, *, name: str) -> str:
    if not isinstance(value, (str, np.str_)):
        raise FixedPerturbationM8Error(f"{name} must be text")
    result = str(value)
    if result != result.strip() or result.casefold() in _MISSING or "\0" in result:
        raise FixedPerturbationM8Error(f"{name} is not canonical text")
    result.encode("utf-8")
    return result


def _labels(values: Sequence[object], *, name: str, unique: bool = True) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise FixedPerturbationM8Error(f"{name} must be a sequence")
    result = tuple(_label(value, name=name) for value in values)
    if not result or (unique and len(result) != len(set(result))):
        raise FixedPerturbationM8Error(f"{name} is empty or duplicated")
    return result


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise FixedPerturbationM8Error(f"{name} is not lowercase SHA-256")
    return value


def _matrix(values: object, *, name: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    try:
        result = np.asarray(values, dtype="<f4")
    except (TypeError, ValueError, OverflowError) as error:
        raise FixedPerturbationM8Error(f"{name} is not numeric") from error
    if result.ndim != 2 or min(result.shape) < 1:
        raise FixedPerturbationM8Error(f"{name} must be a nonempty matrix")
    if shape is not None and result.shape != shape:
        raise FixedPerturbationM8Error(f"{name} shape differs: {result.shape} != {shape}")
    if not np.isfinite(result).all():
        raise FixedPerturbationM8Error(f"{name} contains a nonfinite value")
    frozen = np.ascontiguousarray(result, dtype="<f4").copy()
    frozen.setflags(write=False)
    return frozen


@dataclass(frozen=True)
class GeometrySpec:
    dataset_id: str
    artifact_domain: str
    perturbation_id: str
    train_root_ids: tuple[str, ...]
    evaluation_root_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    feature_count: int = 2_000
    train_cells_per_arm: int = 8

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _label(self.dataset_id, name="dataset_id"))
        object.__setattr__(self, "artifact_domain", _label(self.artifact_domain, name="artifact_domain"))
        object.__setattr__(self, "perturbation_id", _label(self.perturbation_id, name="perturbation_id"))
        object.__setattr__(self, "train_root_ids", _labels(self.train_root_ids, name="train_root_ids"))
        object.__setattr__(self, "evaluation_root_ids", _labels(self.evaluation_root_ids, name="evaluation_root_ids"))
        object.__setattr__(self, "task_ids", _labels(self.task_ids, name="task_ids"))
        if set(self.train_root_ids).intersection(self.evaluation_root_ids):
            raise FixedPerturbationM8Error("TRAIN and EVALUATION roots overlap")
        if self.feature_count < 1 or self.train_cells_per_arm < 1:
            raise FixedPerturbationM8Error("feature or cell count is invalid")

    @property
    def train_root_task_count(self) -> int:
        return len(self.train_root_ids) * len(self.task_ids)

    @property
    def evaluation_root_task_count(self) -> int:
        return len(self.evaluation_root_ids) * len(self.task_ids)


def _validate_cartesian(
    spec: GeometrySpec,
    *,
    row_ids: Sequence[object],
    root_ids: Sequence[object],
    task_ids: Sequence[object],
    role: str,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    rows = _labels(row_ids, name=f"{role}_row_ids")
    roots = _labels(root_ids, name=f"{role}_root_ids", unique=False)
    tasks = _labels(task_ids, name=f"{role}_task_ids", unique=False)
    expected_roots = spec.train_root_ids if role == "TRAIN" else spec.evaluation_root_ids
    expected = {(root, task) for root in expected_roots for task in spec.task_ids}
    observed = tuple(zip(roots, tasks, strict=True)) if len(roots) == len(tasks) else ()
    if len(rows) != len(observed) or len(observed) != len(expected) or set(observed) != expected:
        raise FixedPerturbationM8Error(f"{role} root-task Cartesian geometry differs")
    if len(observed) != len(set(observed)):
        raise FixedPerturbationM8Error(f"{role} root-task pairs are duplicated")
    return rows, roots, tasks


def _membership_sha256(
    spec: GeometrySpec,
    axis_ids: tuple[str, ...],
    memberships: Sequence[Sequence[object]],
    *,
    role: str,
) -> str:
    if len(axis_ids) != len(memberships):
        raise FixedPerturbationM8Error(f"{role} membership rows differ from axis")
    digest = hashlib.sha256(f"{spec.artifact_domain}|{role}|".encode())
    for axis, raw_members in zip(axis_ids, memberships, strict=True):
        members = _labels(raw_members, name=f"{role}_members")
        digest.update(axis.encode()); digest.update(b"\0")
        for member in members:
            digest.update(member.encode()); digest.update(b"\0")
    return digest.hexdigest()


def _artifact_sha256(
    spec: GeometrySpec,
    *,
    method_id: str,
    reference_instance_id: str | None,
    output_type: str,
    axis_ids: tuple[str, ...],
    feature_ids: tuple[str, ...],
    values: np.ndarray,
    state_sha256: str,
    c_infer_membership_sha256: str | None,
    c_pred_membership_sha256: str | None,
) -> str:
    digest = hashlib.sha256(f"{spec.artifact_domain}|PREDICTION_ARTIFACT|".encode())
    for value in (
        method_id,
        reference_instance_id or "SCHEME_INVARIANT",
        output_type,
        state_sha256,
        c_infer_membership_sha256 or "NONE",
        c_pred_membership_sha256 or "NONE",
    ):
        digest.update(value.encode()); digest.update(b"\0")
    for value in axis_ids + feature_ids:
        digest.update(value.encode()); digest.update(b"\0")
    digest.update(np.asarray(values, dtype="<f4", order="C").tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class PredictionArtifact:
    dataset_id: str
    artifact_domain: str
    method_id: str
    reference_instance_id: str | None
    output_type: str
    source_output_type: str
    c_pred_subtractions: int
    axis_ids: tuple[str, ...]
    feature_ids: tuple[str, ...]
    values: np.ndarray
    state_sha256: str
    c_infer_membership_sha256: str | None
    c_pred_membership_sha256: str | None
    c_pred_mean: np.ndarray | None
    artifact_sha256: str


def fit_direct_family(
    spec: GeometrySpec,
    *,
    training_row_ids: Sequence[object],
    training_root_ids: Sequence[object],
    training_task_ids: Sequence[object],
    effects: object,
    feature_ids: Sequence[object],
) -> Mapping[str, DirectBaselineState]:
    rows, _roots, tasks = _validate_cartesian(
        spec,
        row_ids=training_row_ids,
        root_ids=training_root_ids,
        task_ids=training_task_ids,
        role="TRAIN",
    )
    features = _labels(feature_ids, name="feature_ids")
    if len(features) != spec.feature_count:
        raise FixedPerturbationM8Error("feature count differs")
    effect_matrix = _matrix(effects, name="root-task effects", shape=(len(rows), len(features)))
    compounds = (spec.perturbation_id,) * len(rows)
    doses = np.ones(len(rows), dtype=np.float64)
    states: dict[str, DirectBaselineState] = {
        NO_CHANGE: make_no_change_direct_state(
            feature_ids=features,
            context_ids=tuple(sorted(set(tasks), key=str.encode)),
            compound_ids=(spec.perturbation_id,),
        )
    }
    for method_id in DIRECT_METHOD_IDS[1:]:
        states[method_id] = fit_internal_direct_method(
            method_id,
            task_ids=rows,
            contexts=tasks,
            compounds=compounds,
            doses_um=doses,
            effects=effect_matrix,
            feature_ids=features,
        )
    return MappingProxyType(states)


def materialize_direct_family(
    spec: GeometrySpec,
    states: Mapping[str, DirectBaselineState],
    *,
    evaluation_row_ids: Sequence[object],
    evaluation_root_ids: Sequence[object],
    evaluation_task_ids: Sequence[object],
    feature_ids: Sequence[object],
) -> tuple[PredictionArtifact, ...]:
    if tuple(states) != DIRECT_METHOD_IDS:
        raise FixedPerturbationM8Error("DIRECT state order or membership differs")
    rows, _roots, tasks = _validate_cartesian(
        spec,
        row_ids=evaluation_row_ids,
        root_ids=evaluation_root_ids,
        task_ids=evaluation_task_ids,
        role="EVALUATION",
    )
    features = _labels(feature_ids, name="feature_ids")
    if len(features) != spec.feature_count:
        raise FixedPerturbationM8Error("feature count differs")
    compounds = (spec.perturbation_id,) * len(rows)
    doses = np.ones(len(rows), dtype=np.float64)
    result = []
    for method_id in DIRECT_METHOD_IDS:
        state_sha = canonical_state_sha256(states[method_id])
        values = _matrix(
            predict_internal_direct(
                states[method_id],
                contexts=tasks,
                compounds=compounds,
                doses_um=doses,
                feature_ids=features,
            ),
            name="DIRECT predicted effect",
            shape=(len(rows), len(features)),
        )
        digest = _artifact_sha256(
            spec,
            method_id=method_id,
            reference_instance_id=None,
            output_type="PREDICTED_EFFECT",
            axis_ids=rows,
            feature_ids=features,
            values=values,
            state_sha256=state_sha,
            c_infer_membership_sha256=None,
            c_pred_membership_sha256=None,
        )
        result.append(PredictionArtifact(
            dataset_id=spec.dataset_id,
            artifact_domain=spec.artifact_domain,
            method_id=method_id,
            reference_instance_id=None,
            output_type="PREDICTED_EFFECT",
            source_output_type="DIRECT",
            c_pred_subtractions=0,
            axis_ids=rows,
            feature_ids=features,
            values=values,
            state_sha256=state_sha,
            c_infer_membership_sha256=None,
            c_pred_membership_sha256=None,
            c_pred_mean=None,
            artifact_sha256=digest,
        ))
    return tuple(result)


def finalize_absolute_prediction(
    spec: GeometrySpec,
    *,
    method_id: str,
    reference_instance_id: str,
    root_task_ids: Sequence[object],
    feature_ids: Sequence[object],
    absolute_native: object,
    c_pred_mean: object,
    state_sha256: str,
    c_infer_memberships: Sequence[Sequence[object]],
    c_pred_memberships: Sequence[Sequence[object]],
) -> tuple[PredictionArtifact, PredictionArtifact]:
    if method_id not in ABSOLUTE_METHOD_IDS or reference_instance_id not in REFERENCE_INSTANCE_IDS:
        raise FixedPerturbationM8Error("ABSOLUTE method or reference instance differs")
    axes = _labels(root_task_ids, name="root_task_ids")
    features = _labels(feature_ids, name="feature_ids")
    shape = (len(axes), len(features))
    native = _matrix(absolute_native, name="ABSOLUTE_NATIVE", shape=shape)
    pred = _matrix(c_pred_mean, name="C_pred mean", shape=shape)
    effect = _matrix(np.subtract(native, pred, dtype=np.float32), name="PREDICTED_EFFECT", shape=shape)
    state = _sha256(state_sha256, name="state_sha256")
    infer_sha = _membership_sha256(spec, axes, c_infer_memberships, role="C_INFER")
    pred_sha = _membership_sha256(spec, axes, c_pred_memberships, role="C_PRED")
    common = dict(
        dataset_id=spec.dataset_id,
        artifact_domain=spec.artifact_domain,
        method_id=method_id,
        reference_instance_id=reference_instance_id,
        source_output_type="ABSOLUTE_NATIVE",
        axis_ids=axes,
        feature_ids=features,
        state_sha256=state,
        c_infer_membership_sha256=infer_sha,
        c_pred_membership_sha256=pred_sha,
    )
    native_sha = _artifact_sha256(
        spec,
        method_id=method_id,
        reference_instance_id=reference_instance_id,
        output_type="ABSOLUTE_NATIVE",
        axis_ids=axes,
        feature_ids=features,
        values=native,
        state_sha256=state,
        c_infer_membership_sha256=infer_sha,
        c_pred_membership_sha256=pred_sha,
    )
    effect_sha = _artifact_sha256(
        spec,
        method_id=method_id,
        reference_instance_id=reference_instance_id,
        output_type="PREDICTED_EFFECT",
        axis_ids=axes,
        feature_ids=features,
        values=effect,
        state_sha256=state,
        c_infer_membership_sha256=infer_sha,
        c_pred_membership_sha256=pred_sha,
    )
    return (
        PredictionArtifact(**common, output_type="ABSOLUTE_NATIVE", c_pred_subtractions=0,
                           values=native, c_pred_mean=None, artifact_sha256=native_sha),
        PredictionArtifact(**common, output_type="PREDICTED_EFFECT", c_pred_subtractions=1,
                           values=effect, c_pred_mean=pred, artifact_sha256=effect_sha),
    )


def materialize_control_conditioned_absolute(
    spec: GeometrySpec,
    *,
    method_id: str,
    reference_instance_id: str,
    root_task_ids: Sequence[object],
    feature_ids: Sequence[object],
    c_infer_by_root_task: Sequence[object],
    c_pred_by_root_task: Sequence[object],
    c_infer_memberships: Sequence[Sequence[object]],
    c_pred_memberships: Sequence[Sequence[object]],
    state_sha256: str,
    predictor: Callable[[str, str, np.ndarray], object],
) -> tuple[PredictionArtifact, PredictionArtifact]:
    axes = _labels(root_task_ids, name="root_task_ids")
    features = _labels(feature_ids, name="feature_ids")
    if reference_instance_id not in REFERENCE_INSTANCE_IDS:
        raise FixedPerturbationM8Error("reference instance differs")
    if not callable(predictor) or not all(len(value) == len(axes) for value in (
        c_infer_by_root_task, c_pred_by_root_task, c_infer_memberships, c_pred_memberships
    )):
        raise FixedPerturbationM8Error("control-conditioned inputs differ from root-task axis")
    expected_pool_size = REFERENCE_POOL_SIZE[reference_instance_id]
    native_rows = []
    pred_rows = []
    for axis, infer_raw, pred_raw, infer_ids_raw, pred_ids_raw in zip(
        axes,
        c_infer_by_root_task,
        c_pred_by_root_task,
        c_infer_memberships,
        c_pred_memberships,
        strict=True,
    ):
        infer_ids = _labels(infer_ids_raw, name="C_infer_members")
        pred_ids = _labels(pred_ids_raw, name="C_pred_members")
        if len(infer_ids) != expected_pool_size or len(pred_ids) != expected_pool_size:
            raise FixedPerturbationM8Error("reference instance pool size differs")
        if reference_instance_id == "D8_NAIVE_SHARED":
            if infer_ids != pred_ids:
                raise FixedPerturbationM8Error("NAIVE_SHARED roles do not share the registered 24-cell union")
        elif set(infer_ids).intersection(pred_ids):
            raise FixedPerturbationM8Error("independent reference roles overlap")
        infer = _matrix(infer_raw, name="C_infer cells", shape=(expected_pool_size, len(features)))
        pred = _matrix(pred_raw, name="C_pred cells", shape=(expected_pool_size, len(features)))
        predicted = _matrix(
            predictor(axis, reference_instance_id, infer),
            name="per-C_infer ABSOLUTE_NATIVE",
            shape=infer.shape,
        )
        native_rows.append(predicted.mean(axis=0, dtype=np.float64))
        pred_rows.append(pred.mean(axis=0, dtype=np.float64))
    return finalize_absolute_prediction(
        spec,
        method_id=method_id,
        reference_instance_id=reference_instance_id,
        root_task_ids=axes,
        feature_ids=features,
        absolute_native=np.vstack(native_rows),
        c_pred_mean=np.vstack(pred_rows),
        state_sha256=state_sha256,
        c_infer_memberships=c_infer_memberships,
        c_pred_memberships=c_pred_memberships,
    )


@dataclass(frozen=True)
class SCGenTaskAdapter:
    root_task_to_task: Mapping[str, str]
    latent_shift_by_task: Mapping[str, np.ndarray]
    encode: Callable[[np.ndarray], object]
    decode: Callable[[np.ndarray], object]
    feature_count: int

    def __call__(self, root_task_id: str, reference_instance_id: str, controls: np.ndarray) -> np.ndarray:
        if reference_instance_id not in REFERENCE_INSTANCE_IDS:
            raise FixedPerturbationM8Error("scGen reference instance differs")
        task = self.root_task_to_task[root_task_id]
        x = _matrix(controls, name="scGen C_infer")
        encoded = _matrix(self.encode(x), name="scGen latent")
        shift = np.asarray(self.latent_shift_by_task[task], dtype="<f4")
        if x.shape[1] != self.feature_count or shift.ndim != 1 or encoded.shape[1] != shift.size or not np.isfinite(shift).all():
            raise FixedPerturbationM8Error("scGen dimensions differ")
        return _matrix(self.decode(encoded + shift[None, :]), name="scGen ABSOLUTE_NATIVE", shape=x.shape)


@dataclass(frozen=True)
class CellOTTaskAdapter:
    root_task_to_perturbation: Mapping[str, str]
    transport_by_perturbation: Mapping[str, Callable[[np.ndarray], object]]
    feature_count: int

    def __call__(self, root_task_id: str, reference_instance_id: str, controls: np.ndarray) -> np.ndarray:
        if reference_instance_id not in REFERENCE_INSTANCE_IDS:
            raise FixedPerturbationM8Error("CellOT reference instance differs")
        perturbation = self.root_task_to_perturbation[root_task_id]
        x = _matrix(controls, name="CellOT C_infer")
        if x.shape[1] != self.feature_count:
            raise FixedPerturbationM8Error("CellOT dimensions differ")
        return _matrix(self.transport_by_perturbation[perturbation](x), name="CellOT ABSOLUTE_NATIVE", shape=x.shape)


@dataclass(frozen=True)
class CompleteFamilyReport:
    status: str
    eligible: bool
    failures: tuple[str, ...]
    state_count: int
    direct_artifact_count: int
    absolute_artifact_count: int
    joint_contrast_count: int
    scoring_interface_present: bool = False


def joint_contrast_ids() -> tuple[str, ...]:
    return tuple(
        f"{scheme}|{left}|{right}"
        for scheme in SCHEME_IDS
        for left, right in combinations(METHOD_IDS, 2)
    )


def audit_complete_family(
    spec: GeometrySpec,
    *,
    state_sha256_by_method: Mapping[str, str],
    replay_state_sha256_by_method: Mapping[str, str],
    direct_artifacts: Sequence[PredictionArtifact],
    absolute_artifacts: Sequence[PredictionArtifact],
    replay_direct_artifacts: Sequence[PredictionArtifact],
    replay_absolute_artifacts: Sequence[PredictionArtifact],
    expected_root_task_ids: Sequence[object],
    expected_feature_ids: Sequence[object],
) -> CompleteFamilyReport:
    axes = _labels(expected_root_task_ids, name="expected_root_task_ids")
    features = _labels(expected_feature_ids, name="expected_feature_ids")
    failures: list[str] = []
    if len(axes) != spec.evaluation_root_task_count or len(features) != spec.feature_count:
        failures.append("production axis dimensions differ")
    if tuple(state_sha256_by_method) != METHOD_IDS or tuple(replay_state_sha256_by_method) != METHOD_IDS:
        failures.append("state family order or membership differs")
    for method in METHOD_IDS:
        try:
            if _sha256(state_sha256_by_method[method], name="state SHA") != _sha256(replay_state_sha256_by_method[method], name="replay state SHA"):
                failures.append(f"{method} state replay differs")
        except (KeyError, FixedPerturbationM8Error):
            failures.append(f"{method} state is absent or invalid")
    expected_direct = tuple((method, None, "PREDICTED_EFFECT") for method in DIRECT_METHOD_IDS)
    observed_direct = tuple((a.method_id, a.reference_instance_id, a.output_type) for a in direct_artifacts)
    expected_absolute = tuple(
        (method, instance, output)
        for method in ABSOLUTE_METHOD_IDS
        for instance in REFERENCE_INSTANCE_IDS
        for output in ("ABSOLUTE_NATIVE", "PREDICTED_EFFECT")
    )
    observed_absolute = tuple((a.method_id, a.reference_instance_id, a.output_type) for a in absolute_artifacts)
    if observed_direct != expected_direct:
        failures.append("DIRECT artifact family differs")
    if observed_absolute != expected_absolute:
        failures.append("ABSOLUTE artifact family differs")
    for artifact in tuple(direct_artifacts) + tuple(absolute_artifacts):
        if artifact.dataset_id != spec.dataset_id or artifact.artifact_domain != spec.artifact_domain:
            failures.append(f"{artifact.method_id} dataset binding differs")
        if artifact.axis_ids != axes or artifact.feature_ids != features:
            failures.append(f"{artifact.method_id} axis differs")
        if state_sha256_by_method.get(artifact.method_id) != artifact.state_sha256:
            failures.append(f"{artifact.method_id} state binding differs")
        try:
            expected_digest = _artifact_sha256(
                spec,
                method_id=artifact.method_id,
                reference_instance_id=artifact.reference_instance_id,
                output_type=artifact.output_type,
                axis_ids=artifact.axis_ids,
                feature_ids=artifact.feature_ids,
                values=artifact.values,
                state_sha256=artifact.state_sha256,
                c_infer_membership_sha256=artifact.c_infer_membership_sha256,
                c_pred_membership_sha256=artifact.c_pred_membership_sha256,
            )
            if artifact.artifact_sha256 != expected_digest:
                failures.append(f"{artifact.method_id} artifact digest differs")
        except (FixedPerturbationM8Error, KeyError, TypeError):
            failures.append(f"{artifact.method_id} artifact is malformed")
        if artifact.method_id in DIRECT_METHOD_IDS:
            if any((
                artifact.reference_instance_id is not None,
                artifact.output_type != "PREDICTED_EFFECT",
                artifact.source_output_type != "DIRECT",
                artifact.c_pred_subtractions != 0,
                artifact.c_infer_membership_sha256 is not None,
                artifact.c_pred_membership_sha256 is not None,
                artifact.c_pred_mean is not None,
            )):
                failures.append(f"{artifact.method_id} DIRECT semantics differ")
        elif any((
            artifact.reference_instance_id not in REFERENCE_INSTANCE_IDS,
            artifact.source_output_type != "ABSOLUTE_NATIVE",
            artifact.c_infer_membership_sha256 is None,
            artifact.c_pred_membership_sha256 is None,
        )):
            failures.append(f"{artifact.method_id} ABSOLUTE semantics differ")
    primary = tuple(direct_artifacts) + tuple(absolute_artifacts)
    replay = tuple(replay_direct_artifacts) + tuple(replay_absolute_artifacts)
    if len(primary) != len(replay) or any(
        left.artifact_sha256 != right.artifact_sha256 or not np.array_equal(left.values, right.values)
        for left, right in zip(primary, replay, strict=False)
    ):
        failures.append("prediction replay differs")
    by_key = {(a.method_id, a.reference_instance_id, a.output_type): a for a in absolute_artifacts}
    for method in ABSOLUTE_METHOD_IDS:
        for instance in REFERENCE_INSTANCE_IDS:
            native = by_key.get((method, instance, "ABSOLUTE_NATIVE"))
            effect = by_key.get((method, instance, "PREDICTED_EFFECT"))
            if native is None or effect is None:
                continue
            if effect.c_pred_subtractions != 1 or effect.c_pred_mean is None:
                failures.append(f"{method}/{instance} C_pred subtraction count differs")
            elif not np.array_equal(np.subtract(native.values, effect.c_pred_mean, dtype=np.float32), effect.values):
                failures.append(f"{method}/{instance} is not exactly once-centered")
    contrasts = joint_contrast_ids()
    if len(contrasts) != 84 or len(set(contrasts)) != 84:
        failures.append("joint contrast family differs")
    return CompleteFamilyReport(
        status="PASS_COMPLETE_EIGHT_MODEL_SOURCE_API_NO_SCORING" if not failures else "NO_GO_COMPLETE_EIGHT_MODEL_SOURCE_API",
        eligible=not failures,
        failures=tuple(failures),
        state_count=len(state_sha256_by_method),
        direct_artifact_count=len(direct_artifacts),
        absolute_artifact_count=len(absolute_artifacts),
        joint_contrast_count=len(contrasts),
    )


__all__ = [
    "ABSOLUTE_METHOD_IDS",
    "CELLOT",
    "CPA",
    "CellOTTaskAdapter",
    "CompleteFamilyReport",
    "DIRECT_METHOD_IDS",
    "FixedPerturbationM8Error",
    "GeometrySpec",
    "METHOD_IDS",
    "PredictionArtifact",
    "REFERENCE_INSTANCE_IDS",
    "REFERENCE_POOL_SIZE",
    "SCGEN",
    "SCHEME_IDS",
    "SCGenTaskAdapter",
    "audit_complete_family",
    "finalize_absolute_prediction",
    "fit_direct_family",
    "joint_contrast_ids",
    "materialize_control_conditioned_absolute",
    "materialize_direct_family",
]
