"""Parse 10M state and prediction contracts that do not evaluate model performance.

This module deliberately has no H5AD reader and no scoring interface.  It
provides three narrow pieces needed by the frozen eight-method Parse panel:

* reuse of the existing deterministic DIRECT baseline implementation;
* one common control-conditioned ABSOLUTE materialization boundary for CPA,
  scGen and CellOT; and
* a complete-family structural audit before a separate scorer can run.

DIRECT methods emit perturbation effects and cannot accept ``C_pred``.
ABSOLUTE methods consume the registered ``C_infer`` cells, retain their
``ABSOLUTE_NATIVE`` output, and subtract exactly one arithmetic-mean
``C_pred`` profile.  Negative finite values are valid in the frozen
CP10K-log1p feature space and are never clipped here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
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


SCGEN = "SCGEN_2_1_1_ABSOLUTE_V1"
CELLOT = "CELLOT_522D2B9_ABSOLUTE_V1"
CPA = "CPA_0_8_8_ABSOLUTE_V1"

DIRECT_METHOD_IDS = (
    NO_CHANGE,
    CONTEXT_MEAN,
    TWO_WAY_RIDGE,
    PCA64_RIDGE,
    RBF_RIDGE,
)
ABSOLUTE_METHOD_IDS = (CPA, SCGEN, CELLOT)
METHOD_IDS = DIRECT_METHOD_IDS + ABSOLUTE_METHOD_IDS

DEPTHS = (16, 32, 48)
REFERENCE_INSTANCE_IDS = tuple(
    instance
    for depth in DEPTHS
    for instance in (
        f"D{depth}_NAIVE_SHARED",
        f"D{depth}_INDEPENDENT_SPLIT",
        f"D{depth}_CROSSFIT_R0",
        f"D{depth}_CROSSFIT_R1",
        f"D{depth}_CROSSFIT_R2",
    )
)

TRAIN_ROOT_IDS = ("Donor5", "Donor6", "Donor11", "Donor12")
EVALUATION_ROOT_IDS = (
    "Donor1",
    "Donor9",
    "Donor3",
    "Donor4",
    "Donor2",
    "Donor8",
    "Donor10",
    "Donor7",
)
PRODUCTION_TASK_COUNT = 722
PRODUCTION_ROOT_TASK_COUNT = 5_776
PRODUCTION_FEATURE_COUNT = 2_000
GPU_REPLAY_ATOL = 1.0e-6
GPU_REPLAY_RTOL = 1.0e-5

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MISSING = frozenset({"", "na", "nan", "none", "null"})


class ParsePipelineError(ValueError):
    """A frozen Parse state or prediction invariant was violated."""


@dataclass(frozen=True)
class ReplayResult:
    exact_required: bool
    passed: bool
    max_absolute_difference: float
    max_scaled_difference: float


def audit_numerical_replay(
    expected: object,
    observed: object,
    *,
    exact_required: bool,
) -> ReplayResult:
    """Apply the frozen replay rule to one final float32 prediction matrix.

    DIRECT CPU artifacts require exact byte-equivalent values. GPU model
    outputs use ``abs(a-b) <= 1e-6 + 1e-5 * max(abs(a), abs(b))`` elementwise.
    Axes, shape, dtype, state and input hashes remain exact outside this
    numerical comparison.
    """

    left = _matrix(expected, name="expected replay values")
    right = _matrix(observed, name="observed replay values", shape=left.shape)
    difference = np.abs(
        np.subtract(left, right, dtype=np.float32)
    ).astype(np.float64)
    scale = np.maximum(np.abs(left), np.abs(right)).astype(np.float64)
    tolerance = GPU_REPLAY_ATOL + GPU_REPLAY_RTOL * scale
    if exact_required:
        passed = bool(np.array_equal(left, right))
    else:
        passed = bool(np.all(difference <= tolerance))
    scaled = np.divide(
        difference,
        tolerance,
        out=np.zeros_like(difference),
        where=tolerance > 0,
    )
    return ReplayResult(
        exact_required=exact_required,
        passed=passed,
        max_absolute_difference=float(difference.max(initial=0.0)),
        max_scaled_difference=float(scaled.max(initial=0.0)),
    )


def _label(value: object, *, name: str) -> str:
    if not isinstance(value, (str, np.str_)):
        raise ParsePipelineError(f"{name} must be a string")
    result = str(value)
    if (
        result != result.strip()
        or result.casefold() in _MISSING
        or "\0" in result
    ):
        raise ParsePipelineError(f"{name} is not a canonical non-empty label")
    try:
        result.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ParsePipelineError(f"{name} is not valid UTF-8") from error
    return result


def _labels(
    values: Sequence[object],
    *,
    name: str,
    unique: bool = True,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ParsePipelineError(f"{name} must be a sequence")
    result = tuple(_label(value, name=name) for value in values)
    if not result:
        raise ParsePipelineError(f"{name} cannot be empty")
    if unique and len(result) != len(set(result)):
        raise ParsePipelineError(f"{name} must be unique")
    return result


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ParsePipelineError(f"{name} must be a lowercase SHA-256")
    return value


def _matrix(
    values: object,
    *,
    name: str,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    try:
        result = np.asarray(values, dtype="<f4")
    except (TypeError, ValueError, OverflowError) as error:
        raise ParsePipelineError(f"{name} must be numeric") from error
    if result.ndim != 2 or min(result.shape) < 1:
        raise ParsePipelineError(f"{name} must be a non-empty matrix")
    if shape is not None and result.shape != shape:
        raise ParsePipelineError(
            f"{name} shape {result.shape} differs from {shape}"
        )
    if not np.isfinite(result).all():
        raise ParsePipelineError(f"{name} contains a nonfinite value")
    frozen = np.ascontiguousarray(result, dtype="<f4").copy()
    frozen.setflags(write=False)
    return frozen


def _membership_sha256(
    axis_ids: Sequence[str],
    memberships: Sequence[Sequence[object]],
    *,
    label: bytes,
) -> str:
    if len(axis_ids) != len(memberships):
        raise ParsePipelineError("membership rows differ from the prediction axis")
    canonical = []
    for axis_id, values in zip(axis_ids, memberships, strict=True):
        members = _labels(values, name="reference membership")
        canonical.append([axis_id, list(members)])
    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(label)
    digest.update(payload)
    return digest.hexdigest()


def _prediction_sha256(
    *,
    method_id: str,
    reference_instance_id: str | None,
    axis_ids: Sequence[str],
    feature_ids: Sequence[str],
    output_type: str,
    values: np.ndarray,
    state_sha256: str,
    c_infer_membership_sha256: str | None,
    c_pred_membership_sha256: str | None,
    c_pred_mean: np.ndarray | None,
) -> str:
    header = {
        "axis_ids": list(axis_ids),
        "c_infer_membership_sha256": c_infer_membership_sha256,
        "c_pred_membership_sha256": c_pred_membership_sha256,
        "dtype": values.dtype.str,
        "feature_ids": list(feature_ids),
        "method_id": method_id,
        "output_type": output_type,
        "reference_instance_id": reference_instance_id,
        "shape": list(values.shape),
        "state_sha256": state_sha256,
    }
    digest = hashlib.sha256(b"PARSE_10M_PREDICTION_ARTIFACT_V1\0")
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
    digest.update(values.tobytes(order="C"))
    if c_pred_mean is not None:
        digest.update(b"\0C_PRED_MEAN\0")
        digest.update(c_pred_mean.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class StateRecord:
    """Logical identity for one complete panel member.

    CellOT's 90 cytokine maps are one logical state family and must share one
    family digest.  The record contains lineage only; it cannot load a model.
    """

    method_id: str
    state_sha256: str
    replay_state_sha256: str
    source_sha256: str
    environment_sha256: str
    training_manifest_sha256: str
    training_root_ids: tuple[str, ...] = TRAIN_ROOT_IDS
    evaluation_root_ids: tuple[str, ...] = EVALUATION_ROOT_IDS

    def __post_init__(self) -> None:
        if self.method_id not in METHOD_IDS:
            raise ParsePipelineError("state method is not in the frozen panel")
        for name in (
            "state_sha256",
            "replay_state_sha256",
            "source_sha256",
            "environment_sha256",
            "training_manifest_sha256",
        ):
            _digest(getattr(self, name), name=name)
        training = _labels(self.training_root_ids, name="training_root_ids")
        evaluation = _labels(
            self.evaluation_root_ids, name="evaluation_root_ids"
        )
        if training != TRAIN_ROOT_IDS or evaluation != EVALUATION_ROOT_IDS:
            raise ParsePipelineError("state donor split differs from frozen Parse split")
        if set(training).intersection(evaluation):
            raise ParsePipelineError("training and evaluation roots overlap")
        object.__setattr__(self, "training_root_ids", training)
        object.__setattr__(self, "evaluation_root_ids", evaluation)


@dataclass(frozen=True)
class PredictionArtifact:
    """One DIRECT or ABSOLUTE prediction artifact produced before scoring."""

    method_id: str
    reference_instance_id: str | None
    axis_ids: tuple[str, ...]
    feature_ids: tuple[str, ...]
    values: np.ndarray
    state_sha256: str
    output_type: str
    source_output_type: str
    c_pred_subtractions: int
    uses_c_infer: bool
    c_infer_membership_sha256: str | None
    c_pred_membership_sha256: str | None
    c_pred_mean: np.ndarray | None
    artifact_sha256: str

    def __post_init__(self) -> None:
        if self.method_id not in METHOD_IDS:
            raise ParsePipelineError("prediction method is not frozen")
        axes = _labels(self.axis_ids, name="axis_ids")
        features = _labels(self.feature_ids, name="feature_ids")
        values = _matrix(
            self.values,
            name="prediction values",
            shape=(len(axes), len(features)),
        )
        state_digest = _digest(self.state_sha256, name="state_sha256")
        if self.output_type not in {"ABSOLUTE_NATIVE", "PREDICTED_EFFECT"}:
            raise ParsePipelineError("prediction output type is not registered")
        if self.source_output_type not in {"DIRECT", "ABSOLUTE_NATIVE"}:
            raise ParsePipelineError("prediction source semantics are not registered")

        if self.method_id in DIRECT_METHOD_IDS:
            if (
                self.reference_instance_id is not None
                or self.output_type != "PREDICTED_EFFECT"
                or self.source_output_type != "DIRECT"
                or self.c_pred_subtractions != 0
                or self.uses_c_infer
                or self.c_infer_membership_sha256 is not None
                or self.c_pred_membership_sha256 is not None
                or self.c_pred_mean is not None
            ):
                raise ParsePipelineError("DIRECT artifact violates reference semantics")
        else:
            if self.reference_instance_id not in REFERENCE_INSTANCE_IDS:
                raise ParsePipelineError("ABSOLUTE artifact lacks a frozen instance")
            if (
                self.source_output_type != "ABSOLUTE_NATIVE"
                or not self.uses_c_infer
            ):
                raise ParsePipelineError("ABSOLUTE artifact must consume C_infer")
            infer_digest = _digest(
                self.c_infer_membership_sha256,
                name="c_infer_membership_sha256",
            )
            pred_digest = _digest(
                self.c_pred_membership_sha256,
                name="c_pred_membership_sha256",
            )
            if self.output_type == "ABSOLUTE_NATIVE":
                if self.c_pred_subtractions != 0 or self.c_pred_mean is not None:
                    raise ParsePipelineError(
                        "ABSOLUTE_NATIVE cannot contain a C_pred subtraction"
                    )
            else:
                if self.c_pred_subtractions != 1 or self.c_pred_mean is None:
                    raise ParsePipelineError(
                        "PREDICTED_EFFECT must subtract exactly one C_pred"
                    )
                pred_mean = _matrix(
                    self.c_pred_mean,
                    name="C_pred mean",
                    shape=values.shape,
                )
                object.__setattr__(self, "c_pred_mean", pred_mean)
            object.__setattr__(self, "c_infer_membership_sha256", infer_digest)
            object.__setattr__(self, "c_pred_membership_sha256", pred_digest)

        expected_digest = _prediction_sha256(
            method_id=self.method_id,
            reference_instance_id=self.reference_instance_id,
            axis_ids=axes,
            feature_ids=features,
            output_type=self.output_type,
            values=values,
            state_sha256=state_digest,
            c_infer_membership_sha256=self.c_infer_membership_sha256,
            c_pred_membership_sha256=self.c_pred_membership_sha256,
            c_pred_mean=self.c_pred_mean,
        )
        if self.artifact_sha256 != expected_digest:
            raise ParsePipelineError("prediction artifact digest differs")
        object.__setattr__(self, "axis_ids", axes)
        object.__setattr__(self, "feature_ids", features)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "state_sha256", state_digest)


def fit_direct_family(
    *,
    task_ids: Sequence[object],
    cell_types: Sequence[object],
    cytokines: Sequence[object],
    effects: object,
    feature_ids: Sequence[object],
) -> Mapping[str, DirectBaselineState]:
    """Fit all five frozen DIRECT states from TRAIN-only task profiles."""

    tasks = _labels(task_ids, name="task_ids")
    contexts = _labels(cell_types, name="cell_types", unique=False)
    compounds = _labels(cytokines, name="cytokines", unique=False)
    if len(contexts) != len(tasks) or len(compounds) != len(tasks):
        raise ParsePipelineError("task metadata lengths differ")
    features = _labels(feature_ids, name="feature_ids")
    effect_matrix = np.asarray(effects, dtype=np.float64)
    if effect_matrix.shape != (len(tasks), len(features)):
        raise ParsePipelineError("TRAIN direct profile shape differs")
    if not np.isfinite(effect_matrix).all():
        raise ParsePipelineError("TRAIN direct profiles contain nonfinite values")
    doses = np.ones(len(tasks), dtype=np.float64)

    states: dict[str, DirectBaselineState] = {
        NO_CHANGE: make_no_change_direct_state(
            feature_ids=features,
            context_ids=tuple(sorted(set(contexts), key=str.encode)),
            compound_ids=tuple(sorted(set(compounds), key=str.encode)),
        )
    }
    for method_id in DIRECT_METHOD_IDS[1:]:
        states[method_id] = fit_internal_direct_method(
            method_id,
            task_ids=tasks,
            contexts=contexts,
            compounds=compounds,
            doses_um=doses,
            effects=effect_matrix,
            feature_ids=features,
        )
    return states


def materialize_direct_family(
    states: Mapping[str, DirectBaselineState],
    *,
    task_ids: Sequence[object],
    cell_types: Sequence[object],
    cytokines: Sequence[object],
    feature_ids: Sequence[object],
) -> tuple[PredictionArtifact, ...]:
    """Materialize scheme-invariant DIRECT effects without a reference API."""

    if tuple(states) != DIRECT_METHOD_IDS:
        raise ParsePipelineError("DIRECT state mapping order/family differs")
    tasks = _labels(task_ids, name="task_ids")
    contexts = _labels(cell_types, name="cell_types", unique=False)
    compounds = _labels(cytokines, name="cytokines", unique=False)
    features = _labels(feature_ids, name="feature_ids")
    if len(contexts) != len(tasks) or len(compounds) != len(tasks):
        raise ParsePipelineError("prediction metadata lengths differ")
    doses = np.ones(len(tasks), dtype=np.float64)
    artifacts = []
    for method_id in DIRECT_METHOD_IDS:
        state = states[method_id]
        values = predict_internal_direct(
            state,
            contexts=contexts,
            compounds=compounds,
            doses_um=doses,
            feature_ids=features,
        )
        artifacts.append(
            finalize_direct_prediction(
                method_id=method_id,
                task_ids=tasks,
                feature_ids=features,
                predicted_effect=values,
                state_sha256=canonical_state_sha256(state),
            )
        )
    return tuple(artifacts)


def finalize_direct_prediction(
    *,
    method_id: str,
    task_ids: Sequence[object],
    feature_ids: Sequence[object],
    predicted_effect: object,
    state_sha256: str,
) -> PredictionArtifact:
    """Freeze one DIRECT task matrix; there is intentionally no C_pred input."""

    if method_id not in DIRECT_METHOD_IDS:
        raise ParsePipelineError("method is not DIRECT")
    tasks = _labels(task_ids, name="task_ids")
    features = _labels(feature_ids, name="feature_ids")
    values = _matrix(
        predicted_effect,
        name="DIRECT predicted effect",
        shape=(len(tasks), len(features)),
    )
    state_digest = _digest(state_sha256, name="state_sha256")
    artifact_digest = _prediction_sha256(
        method_id=method_id,
        reference_instance_id=None,
        axis_ids=tasks,
        feature_ids=features,
        output_type="PREDICTED_EFFECT",
        values=values,
        state_sha256=state_digest,
        c_infer_membership_sha256=None,
        c_pred_membership_sha256=None,
        c_pred_mean=None,
    )
    return PredictionArtifact(
        method_id=method_id,
        reference_instance_id=None,
        axis_ids=tasks,
        feature_ids=features,
        values=values,
        state_sha256=state_digest,
        output_type="PREDICTED_EFFECT",
        source_output_type="DIRECT",
        c_pred_subtractions=0,
        uses_c_infer=False,
        c_infer_membership_sha256=None,
        c_pred_membership_sha256=None,
        c_pred_mean=None,
        artifact_sha256=artifact_digest,
    )


def finalize_absolute_prediction(
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
    """Freeze ABSOLUTE_NATIVE and subtract exactly one C_pred profile."""

    if method_id not in ABSOLUTE_METHOD_IDS:
        raise ParsePipelineError("method is not an ABSOLUTE panel member")
    if reference_instance_id not in REFERENCE_INSTANCE_IDS:
        raise ParsePipelineError("reference instance is not frozen")
    axes = _labels(root_task_ids, name="root_task_ids")
    features = _labels(feature_ids, name="feature_ids")
    shape = (len(axes), len(features))
    native = _matrix(absolute_native, name="ABSOLUTE_NATIVE", shape=shape)
    pred_mean = _matrix(c_pred_mean, name="C_pred mean", shape=shape)
    effect = _matrix(
        np.subtract(native, pred_mean, dtype=np.float32),
        name="PREDICTED_EFFECT",
        shape=shape,
    )
    state_digest = _digest(state_sha256, name="state_sha256")
    infer_digest = _membership_sha256(
        axes,
        c_infer_memberships,
        label=b"PARSE_10M_C_INFER_MEMBERSHIP_V1\0",
    )
    pred_digest = _membership_sha256(
        axes,
        c_pred_memberships,
        label=b"PARSE_10M_C_PRED_MEMBERSHIP_V1\0",
    )

    native_digest = _prediction_sha256(
        method_id=method_id,
        reference_instance_id=reference_instance_id,
        axis_ids=axes,
        feature_ids=features,
        output_type="ABSOLUTE_NATIVE",
        values=native,
        state_sha256=state_digest,
        c_infer_membership_sha256=infer_digest,
        c_pred_membership_sha256=pred_digest,
        c_pred_mean=None,
    )
    effect_digest = _prediction_sha256(
        method_id=method_id,
        reference_instance_id=reference_instance_id,
        axis_ids=axes,
        feature_ids=features,
        output_type="PREDICTED_EFFECT",
        values=effect,
        state_sha256=state_digest,
        c_infer_membership_sha256=infer_digest,
        c_pred_membership_sha256=pred_digest,
        c_pred_mean=pred_mean,
    )
    common = {
        "method_id": method_id,
        "reference_instance_id": reference_instance_id,
        "axis_ids": axes,
        "feature_ids": features,
        "state_sha256": state_digest,
        "source_output_type": "ABSOLUTE_NATIVE",
        "uses_c_infer": True,
        "c_infer_membership_sha256": infer_digest,
        "c_pred_membership_sha256": pred_digest,
    }
    return (
        PredictionArtifact(
            **common,
            values=native,
            output_type="ABSOLUTE_NATIVE",
            c_pred_subtractions=0,
            c_pred_mean=None,
            artifact_sha256=native_digest,
        ),
        PredictionArtifact(
            **common,
            values=effect,
            output_type="PREDICTED_EFFECT",
            c_pred_subtractions=1,
            c_pred_mean=pred_mean,
            artifact_sha256=effect_digest,
        ),
    )


def materialize_control_conditioned_absolute(
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
    """Apply a frozen CPA/scGen/CellOT state to registered C_infer pools.

    The injected predictor receives ``(root_task_id, reference_instance_id,
    C_infer_cells)`` and must return one ABSOLUTE_NATIVE row per C_infer cell.
    The adapter averages those rows, independently averages C_pred, and then
    delegates to :func:`finalize_absolute_prediction`.
    """

    if method_id not in ABSOLUTE_METHOD_IDS:
        raise ParsePipelineError("method is not control-conditioned ABSOLUTE")
    if not callable(predictor):
        raise ParsePipelineError("predictor must be callable")
    axes = _labels(root_task_ids, name="root_task_ids")
    features = _labels(feature_ids, name="feature_ids")
    if not (
        len(c_infer_by_root_task)
        == len(c_pred_by_root_task)
        == len(c_infer_memberships)
        == len(c_pred_memberships)
        == len(axes)
    ):
        raise ParsePipelineError("reference pool rows differ from root-task axis")

    native_rows: list[np.ndarray] = []
    pred_rows: list[np.ndarray] = []
    for axis_id, infer_raw, pred_raw, infer_ids, pred_ids in zip(
        axes,
        c_infer_by_root_task,
        c_pred_by_root_task,
        c_infer_memberships,
        c_pred_memberships,
        strict=True,
    ):
        infer = _matrix(infer_raw, name="C_infer cells")
        pred = _matrix(pred_raw, name="C_pred cells")
        if infer.shape[1] != len(features) or pred.shape[1] != len(features):
            raise ParsePipelineError("reference pool feature axis differs")
        if infer.shape[0] != len(infer_ids) or pred.shape[0] != len(pred_ids):
            raise ParsePipelineError(
                "reference pool row count differs from frozen membership"
            )
        try:
            predicted = _matrix(
                predictor(axis_id, reference_instance_id, infer),
                name="per-C_infer ABSOLUTE_NATIVE",
                shape=infer.shape,
            )
        except Exception as error:
            raise ParsePipelineError(
                f"{method_id} prediction failed for {axis_id}"
            ) from error
        native_rows.append(predicted.mean(axis=0, dtype=np.float64))
        pred_rows.append(pred.mean(axis=0, dtype=np.float64))
    return finalize_absolute_prediction(
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
    """Bind a frozen global scGen state to TRAIN-only task latent shifts.

    ``encode`` and ``decode`` are injected so this module does not import
    torch, scvi-tools, scGen or an H5AD reader. A production runner must bind
    the frozen package and state.
    """

    root_task_to_task: Mapping[str, str]
    latent_shift_by_task: Mapping[str, np.ndarray]
    encode: Callable[[np.ndarray], object]
    decode: Callable[[np.ndarray], object]
    feature_count: int

    def __post_init__(self) -> None:
        if not callable(self.encode) or not callable(self.decode):
            raise ParsePipelineError("scGen encode/decode must be callable")
        if self.feature_count < 1:
            raise ParsePipelineError("scGen feature_count must be positive")
        root_map = {
            _label(root_task, name="root_task_id"): _label(task, name="task_id")
            for root_task, task in self.root_task_to_task.items()
        }
        if not root_map or len(root_map) != len(self.root_task_to_task):
            raise ParsePipelineError("scGen root-task mapping is empty or duplicated")
        shifts: dict[str, np.ndarray] = {}
        latent_count: int | None = None
        for raw_task, raw_shift in self.latent_shift_by_task.items():
            task = _label(raw_task, name="task_id")
            shift = np.asarray(raw_shift, dtype="<f4")
            if shift.ndim != 1 or shift.size < 1 or not np.isfinite(shift).all():
                raise ParsePipelineError("scGen latent shift is invalid")
            if latent_count is None:
                latent_count = int(shift.size)
            elif shift.size != latent_count:
                raise ParsePipelineError("scGen latent shift dimensions differ")
            frozen = np.ascontiguousarray(shift, dtype="<f4").copy()
            frozen.setflags(write=False)
            shifts[task] = frozen
        if set(root_map.values()) != set(shifts):
            raise ParsePipelineError("scGen task mapping and latent shifts differ")
        object.__setattr__(self, "root_task_to_task", MappingProxyType(root_map))
        object.__setattr__(
            self, "latent_shift_by_task", MappingProxyType(shifts)
        )

    def __call__(
        self,
        root_task_id: str,
        reference_instance_id: str,
        c_infer: np.ndarray,
    ) -> np.ndarray:
        if reference_instance_id not in REFERENCE_INSTANCE_IDS:
            raise ParsePipelineError("scGen reference instance is not frozen")
        try:
            task_id = self.root_task_to_task[root_task_id]
        except KeyError as error:
            raise ParsePipelineError("scGen root-task is not frozen") from error
        controls = _matrix(c_infer, name="scGen C_infer")
        if controls.shape[1] != self.feature_count:
            raise ParsePipelineError("scGen C_infer feature dimension differs")
        encoded = _matrix(self.encode(controls), name="scGen latent encoding")
        shift = self.latent_shift_by_task[task_id]
        if encoded.shape[1] != shift.size:
            raise ParsePipelineError("scGen encoded and shift dimensions differ")
        shifted = np.add(encoded, shift[None, :], dtype=np.float32)
        decoded = _matrix(
            self.decode(shifted),
            name="scGen ABSOLUTE_NATIVE",
            shape=controls.shape,
        )
        return decoded


@dataclass(frozen=True)
class CellOTTaskAdapter:
    """Bind each Parse root-task to one frozen cytokine-specific CellOT map."""

    root_task_to_cytokine: Mapping[str, str]
    transport_by_cytokine: Mapping[str, Callable[[np.ndarray], object]]
    feature_count: int

    def __post_init__(self) -> None:
        if self.feature_count < 1:
            raise ParsePipelineError("CellOT feature_count must be positive")
        root_map = {
            _label(root_task, name="root_task_id"): _label(
                cytokine, name="cytokine"
            )
            for root_task, cytokine in self.root_task_to_cytokine.items()
        }
        transports: dict[str, Callable[[np.ndarray], object]] = {}
        for raw_cytokine, transport in self.transport_by_cytokine.items():
            cytokine = _label(raw_cytokine, name="cytokine")
            if not callable(transport):
                raise ParsePipelineError("CellOT transport must be callable")
            transports[cytokine] = transport
        if not root_map or set(root_map.values()) != set(transports):
            raise ParsePipelineError("CellOT root-task and map families differ")
        object.__setattr__(self, "root_task_to_cytokine", MappingProxyType(root_map))
        object.__setattr__(
            self, "transport_by_cytokine", MappingProxyType(transports)
        )

    def __call__(
        self,
        root_task_id: str,
        reference_instance_id: str,
        c_infer: np.ndarray,
    ) -> np.ndarray:
        if reference_instance_id not in REFERENCE_INSTANCE_IDS:
            raise ParsePipelineError("CellOT reference instance is not frozen")
        try:
            cytokine = self.root_task_to_cytokine[root_task_id]
        except KeyError as error:
            raise ParsePipelineError("CellOT root-task is not frozen") from error
        controls = _matrix(c_infer, name="CellOT C_infer")
        if controls.shape[1] != self.feature_count:
            raise ParsePipelineError("CellOT C_infer feature dimension differs")
        return _matrix(
            self.transport_by_cytokine[cytokine](controls),
            name="CellOT ABSOLUTE_NATIVE",
            shape=controls.shape,
        )


@dataclass(frozen=True)
class CompleteFamilyReport:
    status: str
    eligible: bool
    failures: tuple[str, ...]
    state_count: int
    direct_effect_count: int
    absolute_native_count: int
    absolute_effect_count: int
    scoring_interface_present: bool = False


def audit_complete_family(
    *,
    states: Sequence[StateRecord],
    predictions: Sequence[PredictionArtifact],
    expected_task_ids: Sequence[object],
    expected_root_task_ids: Sequence[object],
    expected_feature_ids: Sequence[object],
    production_geometry: bool = True,
) -> CompleteFamilyReport:
    """Audit the eight-method family before scoring and reject incomplete inputs."""

    tasks = _labels(expected_task_ids, name="expected_task_ids")
    root_tasks = _labels(
        expected_root_task_ids, name="expected_root_task_ids"
    )
    features = _labels(expected_feature_ids, name="expected_feature_ids")
    failures: list[str] = []
    if production_geometry and (
        len(tasks) != PRODUCTION_TASK_COUNT
        or len(root_tasks) != PRODUCTION_ROOT_TASK_COUNT
        or len(features) != PRODUCTION_FEATURE_COUNT
    ):
        failures.append("production axis dimensions differ")

    state_by_method = {state.method_id: state for state in states}
    if len(state_by_method) != len(states):
        failures.append("duplicate state method")
    if tuple(state_by_method) != METHOD_IDS:
        failures.append("state family order or membership differs")
    for state in states:
        if state.state_sha256 != state.replay_state_sha256:
            failures.append(f"{state.method_id} state replay differs")

    direct = [
        artifact
        for artifact in predictions
        if artifact.method_id in DIRECT_METHOD_IDS
    ]
    native = [
        artifact
        for artifact in predictions
        if artifact.method_id in ABSOLUTE_METHOD_IDS
        and artifact.output_type == "ABSOLUTE_NATIVE"
    ]
    absolute_effect = [
        artifact
        for artifact in predictions
        if artifact.method_id in ABSOLUTE_METHOD_IDS
        and artifact.output_type == "PREDICTED_EFFECT"
    ]
    expected_direct_keys = tuple(
        (method_id, None, "PREDICTED_EFFECT")
        for method_id in DIRECT_METHOD_IDS
    )
    expected_absolute_keys = tuple(
        (method_id, instance_id, output_type)
        for method_id in ABSOLUTE_METHOD_IDS
        for instance_id in REFERENCE_INSTANCE_IDS
        for output_type in ("ABSOLUTE_NATIVE", "PREDICTED_EFFECT")
    )
    observed_direct_keys = tuple(
        (
            artifact.method_id,
            artifact.reference_instance_id,
            artifact.output_type,
        )
        for artifact in direct
    )
    observed_absolute_keys = tuple(
        (
            artifact.method_id,
            artifact.reference_instance_id,
            artifact.output_type,
        )
        for artifact in predictions
        if artifact.method_id in ABSOLUTE_METHOD_IDS
    )
    if observed_direct_keys != expected_direct_keys:
        failures.append("DIRECT prediction family order or membership differs")
    if observed_absolute_keys != expected_absolute_keys:
        failures.append("ABSOLUTE prediction family order or membership differs")

    for artifact in direct:
        if artifact.axis_ids != tasks or artifact.feature_ids != features:
            failures.append(f"{artifact.method_id} DIRECT axis differs")
        state = state_by_method.get(artifact.method_id)
        if state is None or artifact.state_sha256 != state.state_sha256:
            failures.append(f"{artifact.method_id} DIRECT state binding differs")
    for artifact in native + absolute_effect:
        if artifact.axis_ids != root_tasks or artifact.feature_ids != features:
            failures.append(
                f"{artifact.method_id} {artifact.reference_instance_id} axis differs"
            )
        state = state_by_method.get(artifact.method_id)
        if state is None or artifact.state_sha256 != state.state_sha256:
            failures.append(
                f"{artifact.method_id} {artifact.reference_instance_id} state differs"
            )

    native_by_key = {
        (artifact.method_id, artifact.reference_instance_id): artifact
        for artifact in native
    }
    for effect in absolute_effect:
        key = (effect.method_id, effect.reference_instance_id)
        paired = native_by_key.get(key)
        if paired is None or effect.c_pred_mean is None:
            failures.append(f"{key} lacks paired ABSOLUTE_NATIVE")
            continue
        expected = np.subtract(
            paired.values, effect.c_pred_mean, dtype=np.float32
        )
        if not np.array_equal(effect.values, expected):
            failures.append(f"{key} is not exactly ABSOLUTE_NATIVE minus C_pred")
        if (
            paired.c_infer_membership_sha256
            != effect.c_infer_membership_sha256
            or paired.c_pred_membership_sha256
            != effect.c_pred_membership_sha256
        ):
            failures.append(f"{key} reference membership differs within pair")

    # Preserve first occurrence while keeping the audit deterministic.
    unique_failures = tuple(dict.fromkeys(failures))
    eligible = not unique_failures
    return CompleteFamilyReport(
        status=(
            "PASS_COMPLETE_EIGHT_MODEL_FAMILY_NO_SCORING"
            if eligible
            else "NO_GO_INCOMPLETE_EIGHT_MODEL_FAMILY"
        ),
        eligible=eligible,
        failures=unique_failures,
        state_count=len(states),
        direct_effect_count=len(direct),
        absolute_native_count=len(native),
        absolute_effect_count=len(absolute_effect),
        scoring_interface_present=False,
    )


__all__ = [
    "ABSOLUTE_METHOD_IDS",
    "CELLOT",
    "CPA",
    "DIRECT_METHOD_IDS",
    "EVALUATION_ROOT_IDS",
    "METHOD_IDS",
    "PRODUCTION_FEATURE_COUNT",
    "PRODUCTION_ROOT_TASK_COUNT",
    "PRODUCTION_TASK_COUNT",
    "REFERENCE_INSTANCE_IDS",
    "GPU_REPLAY_ATOL",
    "GPU_REPLAY_RTOL",
    "SCGEN",
    "TRAIN_ROOT_IDS",
    "CellOTTaskAdapter",
    "CompleteFamilyReport",
    "ParsePipelineError",
    "PredictionArtifact",
    "ReplayResult",
    "SCGenTaskAdapter",
    "StateRecord",
    "audit_complete_family",
    "audit_numerical_replay",
    "finalize_absolute_prediction",
    "finalize_direct_prediction",
    "fit_direct_family",
    "materialize_control_conditioned_absolute",
    "materialize_direct_family",
]
