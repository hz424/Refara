"""Deterministic internal DIRECT baselines for the GSE306429 V2 panel.

The functions in this module operate only on already constructed, task-level
training perturbation effects.  They do not read files, controls, expression
matrices, predictions, scores, or reference allocations.  Every fitted state
has ``DIRECT`` prediction semantics: its output is a perturbation-effect
vector and must not be reference-centred downstream.

Training rows are always placed in bytewise UTF-8 task-ID order before any
floating-point reduction.  Feature order is never changed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import ClassVar, Mapping, Sequence, TypeAlias

import numpy as np

from perturb_nuisance_contracts import BenchmarkContractError


NO_CHANGE = "NO_CHANGE_DIRECT_V1"
CONTEXT_MEAN = "CONTEXT_MEAN_EFFECT_DIRECT_V1"
TWO_WAY_RIDGE = "TWO_WAY_ADDITIVE_RIDGE_DIRECT_V1"
PCA64_RIDGE = "PCA64_ADDITIVE_RIDGE_DIRECT_V1"
RBF_RIDGE = "RBF_KERNEL_RIDGE_DIRECT_V1"

SUPPORTED_METHODS = (
    NO_CHANGE,
    CONTEXT_MEAN,
    TWO_WAY_RIDGE,
    PCA64_RIDGE,
    RBF_RIDGE,
)

RIDGE_LAMBDA = 1.0
PCA_MAX_RANK = 64
RBF_GAMMA_FACTOR = 0.5
STATE_SCHEMA = "gse306429_internal_direct_state_v1"
_MISSING_LABELS = frozenset({"", "na", "nan", "none", "null"})


class GSE306429BaselineError(BenchmarkContractError):
    """Raised when a frozen internal baseline cannot execute exactly."""


def _utf8_key(value: str) -> bytes:
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise GSE306429BaselineError("label is not valid UTF-8") from error


def _labels(
    values: Sequence[object],
    *,
    name: str,
    expected_length: int | None = None,
    require_unique: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise GSE306429BaselineError(f"{name} must be a sequence of labels")
    result: list[str] = []
    for value in values:
        if not isinstance(value, (str, np.str_)):
            raise GSE306429BaselineError(f"{name} contains a non-string label")
        text = str(value)
        if (
            text != text.strip()
            or text.casefold() in _MISSING_LABELS
            or "\0" in text
        ):
            raise GSE306429BaselineError(
                f"{name} contains a missing or non-canonical label"
            )
        _utf8_key(text)
        result.append(text)
    labels = tuple(result)
    if expected_length is not None and len(labels) != expected_length:
        raise GSE306429BaselineError(
            f"{name} has {len(labels)} entries; expected {expected_length}"
        )
    if not labels:
        raise GSE306429BaselineError(f"{name} must be non-empty")
    if require_unique and len(labels) != len(set(labels)):
        raise GSE306429BaselineError(f"{name} must be unique")
    return labels


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values), key=_utf8_key))


def _require_ordered_unique(values: Sequence[object], *, name: str) -> tuple[str, ...]:
    labels = _labels(values, name=name, require_unique=True)
    if labels != _ordered_unique(labels):
        raise GSE306429BaselineError(
            f"{name} must be unique and in bytewise UTF-8 order"
        )
    return labels


def _feature_ids(values: Sequence[object]) -> tuple[str, ...]:
    # Feature order is a frozen external axis, not an axis this module sorts.
    return _labels(values, name="feature_ids", require_unique=True)


def _finite_matrix(
    values: object,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
    ndim: int = 2,
) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise GSE306429BaselineError(f"{name} must be numeric") from error
    if array.ndim != ndim or 0 in array.shape:
        raise GSE306429BaselineError(
            f"{name} must be a non-empty {ndim}-dimensional array"
        )
    if shape is not None and array.shape != shape:
        raise GSE306429BaselineError(
            f"{name} has shape {array.shape}; expected {shape}"
        )
    if not np.isfinite(array).all():
        raise GSE306429BaselineError(f"{name} contains a nonfinite value")
    result = np.ascontiguousarray(array, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def _finite_vector(
    values: object,
    *,
    name: str,
    length: int | None = None,
) -> np.ndarray:
    array = _finite_matrix(values, name=name, ndim=1)
    if length is not None and array.shape != (length,):
        raise GSE306429BaselineError(
            f"{name} has length {array.shape[0]}; expected {length}"
        )
    return array


def _positive_doses(values: Sequence[object], *, expected_length: int) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        raise GSE306429BaselineError("doses_um must be a sequence")
    parsed: list[float] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or value is None:
            raise GSE306429BaselineError("doses_um contains an invalid value")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise GSE306429BaselineError(
                "doses_um contains a non-numeric value"
            ) from error
        if not math.isfinite(number) or number <= 0.0:
            raise GSE306429BaselineError(
                "doses_um must contain only finite positive values"
            )
        parsed.append(number)
    if len(parsed) != expected_length:
        raise GSE306429BaselineError(
            f"doses_um has {len(parsed)} entries; expected {expected_length}"
        )
    result = np.asarray(parsed, dtype=np.float64)
    result.setflags(write=False)
    return result


def normalized_dose(value: object) -> str:
    """Return the frozen geometry-compatible dose spelling."""

    dose = _positive_doses([value], expected_length=1)[0]
    return format(float(dose), ".15g")


def compound_dose_token(compound: str, dose_um: object) -> str:
    """Return the exact categorical compound-dose predictor token."""

    checked = _labels([compound], name="compound", expected_length=1)[0]
    return f"{checked}|dose_uM={normalized_dose(dose_um)}"


@dataclass(frozen=True)
class _TrainingTable:
    task_ids: tuple[str, ...]
    contexts: tuple[str, ...]
    compounds: tuple[str, ...]
    doses_um: np.ndarray
    effects: np.ndarray
    feature_ids: tuple[str, ...]


def _training_table(
    *,
    task_ids: Sequence[object],
    contexts: Sequence[object],
    compounds: Sequence[object],
    doses_um: Sequence[object],
    effects: object,
    feature_ids: Sequence[object],
) -> _TrainingTable:
    tasks = _labels(task_ids, name="task_ids", require_unique=True)
    n_tasks = len(tasks)
    context_labels = _labels(contexts, name="contexts", expected_length=n_tasks)
    compound_labels = _labels(compounds, name="compounds", expected_length=n_tasks)
    doses = _positive_doses(doses_um, expected_length=n_tasks)
    features = _feature_ids(feature_ids)
    effect_matrix = _finite_matrix(
        effects,
        name="effects",
        shape=(n_tasks, len(features)),
    )
    order = np.asarray(
        sorted(range(n_tasks), key=lambda index: _utf8_key(tasks[index])),
        dtype=np.int64,
    )
    ordered_doses = np.ascontiguousarray(doses[order], dtype=np.float64)
    ordered_effects = np.ascontiguousarray(effect_matrix[order], dtype=np.float64)
    ordered_doses.setflags(write=False)
    ordered_effects.setflags(write=False)
    return _TrainingTable(
        task_ids=tuple(tasks[index] for index in order),
        contexts=tuple(context_labels[index] for index in order),
        compounds=tuple(compound_labels[index] for index in order),
        doses_um=ordered_doses,
        effects=ordered_effects,
        feature_ids=features,
    )


@dataclass(frozen=True)
class _PredictionRows:
    contexts: tuple[str, ...]
    compounds: tuple[str, ...]
    doses_um: np.ndarray


def _prediction_rows(
    *,
    contexts: Sequence[object],
    compounds: Sequence[object],
    doses_um: Sequence[object],
    feature_ids: Sequence[object],
    expected_features: tuple[str, ...],
) -> _PredictionRows:
    context_labels = _labels(contexts, name="contexts")
    n_rows = len(context_labels)
    compound_labels = _labels(
        compounds, name="compounds", expected_length=n_rows
    )
    doses = _positive_doses(doses_um, expected_length=n_rows)
    observed_features = _feature_ids(feature_ids)
    if observed_features != expected_features:
        raise GSE306429BaselineError(
            "feature_ids differ in identity or order from the fitted state"
        )
    return _PredictionRows(context_labels, compound_labels, doses)


def _reject_unseen(
    requested: Sequence[str],
    supported: tuple[str, ...],
    *,
    name: str,
) -> None:
    unseen = _ordered_unique(set(requested).difference(supported))
    if unseen:
        raise GSE306429BaselineError(f"unseen {name}: {list(unseen)!r}")


def _freeze_state_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
    ndim: int = 2,
) -> np.ndarray:
    return _finite_matrix(value, name=name, shape=shape, ndim=ndim)


class _StateMixin:
    prediction_semantics: ClassVar[str] = "DIRECT"
    uses_c_infer: ClassVar[bool] = False
    c_pred_subtractions: ClassVar[int] = 0

    def state_manifest(self) -> dict[str, object]:
        return canonical_state_manifest(self)  # type: ignore[arg-type]

    def state_sha256(self) -> str:
        return canonical_state_sha256(self)  # type: ignore[arg-type]


@dataclass(frozen=True)
class NoChangeDirectState(_StateMixin):
    """Zero-effect DIRECT state with metadata-only support categories."""

    feature_ids: tuple[str, ...]
    context_ids: tuple[str, ...]
    compound_ids: tuple[str, ...]
    method_id: ClassVar[str] = NO_CHANGE

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_ids", _feature_ids(self.feature_ids))
        object.__setattr__(
            self,
            "context_ids",
            _require_ordered_unique(self.context_ids, name="context_ids"),
        )
        object.__setattr__(
            self,
            "compound_ids",
            _require_ordered_unique(self.compound_ids, name="compound_ids"),
        )

    def predict(
        self,
        *,
        contexts: Sequence[object],
        compounds: Sequence[object],
        doses_um: Sequence[object],
        feature_ids: Sequence[object],
    ) -> np.ndarray:
        rows = _prediction_rows(
            contexts=contexts,
            compounds=compounds,
            doses_um=doses_um,
            feature_ids=feature_ids,
            expected_features=self.feature_ids,
        )
        _reject_unseen(rows.contexts, self.context_ids, name="context")
        _reject_unseen(rows.compounds, self.compound_ids, name="compound")
        return np.zeros((len(rows.contexts), len(self.feature_ids)), dtype=np.float64)


@dataclass(frozen=True)
class ContextMeanEffectDirectState(_StateMixin):
    """Equal-task context mean on canonical task ordering."""

    feature_ids: tuple[str, ...]
    training_task_ids: tuple[str, ...]
    context_ids: tuple[str, ...]
    compound_ids: tuple[str, ...]
    context_effects: np.ndarray
    method_id: ClassVar[str] = CONTEXT_MEAN

    def __post_init__(self) -> None:
        features = _feature_ids(self.feature_ids)
        tasks = _require_ordered_unique(
            self.training_task_ids, name="training_task_ids"
        )
        contexts = _require_ordered_unique(self.context_ids, name="context_ids")
        compounds = _require_ordered_unique(self.compound_ids, name="compound_ids")
        effects = _freeze_state_array(
            self.context_effects,
            name="context_effects",
            shape=(len(contexts), len(features)),
        )
        object.__setattr__(self, "feature_ids", features)
        object.__setattr__(self, "training_task_ids", tasks)
        object.__setattr__(self, "context_ids", contexts)
        object.__setattr__(self, "compound_ids", compounds)
        object.__setattr__(self, "context_effects", effects)

    def predict(
        self,
        *,
        contexts: Sequence[object],
        compounds: Sequence[object],
        doses_um: Sequence[object],
        feature_ids: Sequence[object],
    ) -> np.ndarray:
        rows = _prediction_rows(
            contexts=contexts,
            compounds=compounds,
            doses_um=doses_um,
            feature_ids=feature_ids,
            expected_features=self.feature_ids,
        )
        _reject_unseen(rows.contexts, self.context_ids, name="context")
        _reject_unseen(rows.compounds, self.compound_ids, name="compound")
        lookup = {value: index for index, value in enumerate(self.context_ids)}
        indices = np.asarray([lookup[value] for value in rows.contexts], dtype=np.int64)
        return np.ascontiguousarray(self.context_effects[indices], dtype=np.float64)


@dataclass(frozen=True)
class TwoWayAdditiveRidgeDirectState(_StateMixin):
    """Unpenalized intercept plus penalized full categorical indicators."""

    feature_ids: tuple[str, ...]
    training_task_ids: tuple[str, ...]
    context_ids: tuple[str, ...]
    compound_ids: tuple[str, ...]
    compound_dose_ids: tuple[str, ...]
    intercept: np.ndarray
    coefficients: np.ndarray
    ridge_lambda: float = RIDGE_LAMBDA
    method_id: ClassVar[str] = TWO_WAY_RIDGE

    def __post_init__(self) -> None:
        _validate_additive_state(self)

    def predict(
        self,
        *,
        contexts: Sequence[object],
        compounds: Sequence[object],
        doses_um: Sequence[object],
        feature_ids: Sequence[object],
    ) -> np.ndarray:
        rows = _prediction_rows(
            contexts=contexts,
            compounds=compounds,
            doses_um=doses_um,
            feature_ids=feature_ids,
            expected_features=self.feature_ids,
        )
        design = _prediction_additive_design(
            rows, self.context_ids, self.compound_ids, self.compound_dose_ids
        )
        prediction = self.intercept + design @ self.coefficients
        return _finite_prediction(prediction, len(rows.contexts), len(self.feature_ids))


@dataclass(frozen=True)
class PCA64AdditiveRidgeDirectState(_StateMixin):
    """PCA effect representation with the frozen additive latent predictor."""

    feature_ids: tuple[str, ...]
    training_task_ids: tuple[str, ...]
    context_ids: tuple[str, ...]
    compound_ids: tuple[str, ...]
    compound_dose_ids: tuple[str, ...]
    feature_mean: np.ndarray
    components: np.ndarray
    latent_intercept: np.ndarray
    latent_coefficients: np.ndarray
    ridge_lambda: float = RIDGE_LAMBDA
    max_rank: int = PCA_MAX_RANK
    method_id: ClassVar[str] = PCA64_RIDGE

    def __post_init__(self) -> None:
        features, tasks, contexts, compounds, compound_doses = (
            _validate_common_additive_fields(self)
        )
        if (
            isinstance(self.max_rank, bool)
            or not isinstance(self.max_rank, int)
            or self.max_rank != PCA_MAX_RANK
        ):
            raise GSE306429BaselineError("PCA max_rank must be frozen at 64")
        if float(self.ridge_lambda) != RIDGE_LAMBDA:
            raise GSE306429BaselineError("ridge_lambda must be frozen at 1.0")
        rank = min(PCA_MAX_RANK, len(tasks) - 1, len(features))
        feature_mean = _freeze_state_array(
            self.feature_mean,
            name="feature_mean",
            shape=(len(features),),
            ndim=1,
        )
        components = np.asarray(self.components, dtype=np.float64)
        if components.shape != (rank, len(features)):
            raise GSE306429BaselineError(
                f"components has shape {components.shape}; "
                f"expected {(rank, len(features))}"
            )
        if not np.isfinite(components).all():
            raise GSE306429BaselineError("components contains a nonfinite value")
        frozen_components = np.ascontiguousarray(components).copy()
        for row in frozen_components:
            pivot = int(np.argmax(np.abs(row)))
            if row[pivot] < 0.0:
                raise GSE306429BaselineError(
                    "PCA component violates the frozen sign rule"
                )
        frozen_components.setflags(write=False)
        latent_intercept = _freeze_state_array(
            self.latent_intercept,
            name="latent_intercept",
            shape=(rank,),
            ndim=1,
        )
        latent_coefficients = _freeze_state_array(
            self.latent_coefficients,
            name="latent_coefficients",
            shape=(len(contexts) + len(compound_doses), rank),
        )
        object.__setattr__(self, "feature_ids", features)
        object.__setattr__(self, "training_task_ids", tasks)
        object.__setattr__(self, "context_ids", contexts)
        object.__setattr__(self, "compound_ids", compounds)
        object.__setattr__(self, "compound_dose_ids", compound_doses)
        object.__setattr__(self, "feature_mean", feature_mean)
        object.__setattr__(self, "components", frozen_components)
        object.__setattr__(self, "latent_intercept", latent_intercept)
        object.__setattr__(self, "latent_coefficients", latent_coefficients)

    @property
    def rank(self) -> int:
        return int(self.components.shape[0])

    def predict(
        self,
        *,
        contexts: Sequence[object],
        compounds: Sequence[object],
        doses_um: Sequence[object],
        feature_ids: Sequence[object],
    ) -> np.ndarray:
        rows = _prediction_rows(
            contexts=contexts,
            compounds=compounds,
            doses_um=doses_um,
            feature_ids=feature_ids,
            expected_features=self.feature_ids,
        )
        design = _prediction_additive_design(
            rows, self.context_ids, self.compound_ids, self.compound_dose_ids
        )
        latent = self.latent_intercept + design @ self.latent_coefficients
        prediction = self.feature_mean + latent @ self.components
        return _finite_prediction(prediction, len(rows.contexts), len(self.feature_ids))


@dataclass(frozen=True)
class RBFKernelRidgeDirectState(_StateMixin):
    """Frozen unit-bandwidth RBF kernel ridge on metadata predictors."""

    feature_ids: tuple[str, ...]
    training_task_ids: tuple[str, ...]
    context_ids: tuple[str, ...]
    compound_ids: tuple[str, ...]
    log10_dose_mean: float
    log10_dose_sample_sd: float
    training_inputs: np.ndarray
    dual_coefficients: np.ndarray
    ridge_lambda: float = RIDGE_LAMBDA
    gamma_factor: float = RBF_GAMMA_FACTOR
    method_id: ClassVar[str] = RBF_RIDGE

    def __post_init__(self) -> None:
        features = _feature_ids(self.feature_ids)
        tasks = _require_ordered_unique(
            self.training_task_ids, name="training_task_ids"
        )
        contexts = _require_ordered_unique(self.context_ids, name="context_ids")
        compounds = _require_ordered_unique(self.compound_ids, name="compound_ids")
        if float(self.ridge_lambda) != RIDGE_LAMBDA:
            raise GSE306429BaselineError("ridge_lambda must be frozen at 1.0")
        if float(self.gamma_factor) != RBF_GAMMA_FACTOR:
            raise GSE306429BaselineError("RBF gamma factor must be frozen at 0.5")
        dose_mean = float(self.log10_dose_mean)
        dose_sd = float(self.log10_dose_sample_sd)
        if not math.isfinite(dose_mean) or not math.isfinite(dose_sd) or dose_sd < 0:
            raise GSE306429BaselineError("RBF dose state is invalid")
        width = len(contexts) + len(compounds) + 1
        inputs = _freeze_state_array(
            self.training_inputs,
            name="training_inputs",
            shape=(len(tasks), width),
        )
        coefficients = _freeze_state_array(
            self.dual_coefficients,
            name="dual_coefficients",
            shape=(len(tasks), len(features)),
        )
        object.__setattr__(self, "feature_ids", features)
        object.__setattr__(self, "training_task_ids", tasks)
        object.__setattr__(self, "context_ids", contexts)
        object.__setattr__(self, "compound_ids", compounds)
        object.__setattr__(self, "log10_dose_mean", dose_mean)
        object.__setattr__(self, "log10_dose_sample_sd", dose_sd)
        object.__setattr__(self, "training_inputs", inputs)
        object.__setattr__(self, "dual_coefficients", coefficients)

    def predict(
        self,
        *,
        contexts: Sequence[object],
        compounds: Sequence[object],
        doses_um: Sequence[object],
        feature_ids: Sequence[object],
    ) -> np.ndarray:
        rows = _prediction_rows(
            contexts=contexts,
            compounds=compounds,
            doses_um=doses_um,
            feature_ids=feature_ids,
            expected_features=self.feature_ids,
        )
        inputs = _rbf_inputs(
            rows.contexts,
            rows.compounds,
            rows.doses_um,
            context_ids=self.context_ids,
            compound_ids=self.compound_ids,
            log10_dose_mean=self.log10_dose_mean,
            log10_dose_sample_sd=self.log10_dose_sample_sd,
        )
        kernel = _rbf_kernel(inputs, self.training_inputs)
        prediction = kernel @ self.dual_coefficients
        return _finite_prediction(prediction, len(rows.contexts), len(self.feature_ids))


DirectBaselineState: TypeAlias = (
    NoChangeDirectState
    | ContextMeanEffectDirectState
    | TwoWayAdditiveRidgeDirectState
    | PCA64AdditiveRidgeDirectState
    | RBFKernelRidgeDirectState
)


def _validate_common_additive_fields(
    state: TwoWayAdditiveRidgeDirectState | PCA64AdditiveRidgeDirectState,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    features = _feature_ids(state.feature_ids)
    tasks = _require_ordered_unique(
        state.training_task_ids, name="training_task_ids"
    )
    contexts = _require_ordered_unique(state.context_ids, name="context_ids")
    compounds = _require_ordered_unique(state.compound_ids, name="compound_ids")
    compound_doses = _require_ordered_unique(
        state.compound_dose_ids, name="compound_dose_ids"
    )
    return features, tasks, contexts, compounds, compound_doses


def _validate_additive_state(state: TwoWayAdditiveRidgeDirectState) -> None:
    features, tasks, contexts, compounds, compound_doses = (
        _validate_common_additive_fields(state)
    )
    if float(state.ridge_lambda) != RIDGE_LAMBDA:
        raise GSE306429BaselineError("ridge_lambda must be frozen at 1.0")
    intercept = _freeze_state_array(
        state.intercept,
        name="intercept",
        shape=(len(features),),
        ndim=1,
    )
    coefficients = _freeze_state_array(
        state.coefficients,
        name="coefficients",
        shape=(len(contexts) + len(compound_doses), len(features)),
    )
    object.__setattr__(state, "feature_ids", features)
    object.__setattr__(state, "training_task_ids", tasks)
    object.__setattr__(state, "context_ids", contexts)
    object.__setattr__(state, "compound_ids", compounds)
    object.__setattr__(state, "compound_dose_ids", compound_doses)
    object.__setattr__(state, "intercept", intercept)
    object.__setattr__(state, "coefficients", coefficients)


def _finite_prediction(values: object, n_rows: int, n_features: int) -> np.ndarray:
    prediction = np.asarray(values, dtype=np.float64)
    if prediction.shape != (n_rows, n_features):
        raise GSE306429BaselineError("internal prediction shape is invalid")
    if not np.isfinite(prediction).all():
        raise GSE306429BaselineError("internal prediction contains a nonfinite value")
    return np.ascontiguousarray(prediction, dtype=np.float64)


def make_no_change_direct_state(
    *,
    feature_ids: Sequence[object],
    context_ids: Sequence[object],
    compound_ids: Sequence[object],
) -> NoChangeDirectState:
    """Create the metadata-bound, zero-parameter NO_CHANGE state."""

    return NoChangeDirectState(
        feature_ids=_feature_ids(feature_ids),
        context_ids=_ordered_unique(
            _labels(context_ids, name="context_ids", require_unique=True)
        ),
        compound_ids=_ordered_unique(
            _labels(compound_ids, name="compound_ids", require_unique=True)
        ),
    )


def fit_context_mean_effect_direct(
    *,
    task_ids: Sequence[object],
    contexts: Sequence[object],
    compounds: Sequence[object],
    doses_um: Sequence[object],
    effects: object,
    feature_ids: Sequence[object],
) -> ContextMeanEffectDirectState:
    table = _training_table(
        task_ids=task_ids,
        contexts=contexts,
        compounds=compounds,
        doses_um=doses_um,
        effects=effects,
        feature_ids=feature_ids,
    )
    context_ids = _ordered_unique(table.contexts)
    means = np.vstack(
        [
            table.effects[
                np.asarray(
                    [value == context for value in table.contexts], dtype=bool
                )
            ].mean(axis=0)
            for context in context_ids
        ]
    )
    return ContextMeanEffectDirectState(
        feature_ids=table.feature_ids,
        training_task_ids=table.task_ids,
        context_ids=context_ids,
        compound_ids=_ordered_unique(table.compounds),
        context_effects=means,
    )


def _training_additive_design(
    table: _TrainingTable,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    context_ids = _ordered_unique(table.contexts)
    compound_ids = _ordered_unique(table.compounds)
    tokens = tuple(
        compound_dose_token(compound, dose)
        for compound, dose in zip(table.compounds, table.doses_um)
    )
    compound_dose_ids = _ordered_unique(tokens)
    context_lookup = {value: index for index, value in enumerate(context_ids)}
    token_lookup = {
        value: len(context_ids) + index
        for index, value in enumerate(compound_dose_ids)
    }
    design = np.zeros(
        (len(table.task_ids), len(context_ids) + len(compound_dose_ids)),
        dtype=np.float64,
    )
    for row, (context, token) in enumerate(zip(table.contexts, tokens)):
        design[row, context_lookup[context]] = 1.0
        design[row, token_lookup[token]] = 1.0
    return design, context_ids, compound_ids, compound_dose_ids


def _prediction_additive_design(
    rows: _PredictionRows,
    context_ids: tuple[str, ...],
    compound_ids: tuple[str, ...],
    compound_dose_ids: tuple[str, ...],
) -> np.ndarray:
    _reject_unseen(rows.contexts, context_ids, name="context")
    # A new dose of a seen compound is explicitly executable; a wholly unseen
    # compound is not an eligible categorical request.
    _reject_unseen(rows.compounds, compound_ids, name="compound")
    context_lookup = {value: index for index, value in enumerate(context_ids)}
    token_lookup = {
        value: len(context_ids) + index
        for index, value in enumerate(compound_dose_ids)
    }
    design = np.zeros(
        (len(rows.contexts), len(context_ids) + len(compound_dose_ids)),
        dtype=np.float64,
    )
    for row, (context, compound, dose) in enumerate(
        zip(rows.contexts, rows.compounds, rows.doses_um)
    ):
        design[row, context_lookup[context]] = 1.0
        token = compound_dose_token(compound, dose)
        if token in token_lookup:
            design[row, token_lookup[token]] = 1.0
    return design


def _ridge_with_unpenalized_intercept(
    design: np.ndarray,
    response: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the frozen ridge objective without penalizing the intercept."""

    x_mean = design.mean(axis=0)
    y_mean = response.mean(axis=0)
    centered_x = design - x_mean
    centered_y = response - y_mean
    gram = centered_x.T @ centered_x
    gram.flat[:: gram.shape[0] + 1] += RIDGE_LAMBDA
    try:
        coefficients = np.linalg.solve(gram, centered_x.T @ centered_y)
    except np.linalg.LinAlgError as error:
        raise GSE306429BaselineError("ridge solve failed") from error
    intercept = y_mean - x_mean @ coefficients
    if not np.isfinite(intercept).all() or not np.isfinite(coefficients).all():
        raise GSE306429BaselineError("ridge state is nonfinite")
    return intercept, coefficients


def fit_two_way_additive_ridge_direct(
    *,
    task_ids: Sequence[object],
    contexts: Sequence[object],
    compounds: Sequence[object],
    doses_um: Sequence[object],
    effects: object,
    feature_ids: Sequence[object],
) -> TwoWayAdditiveRidgeDirectState:
    table = _training_table(
        task_ids=task_ids,
        contexts=contexts,
        compounds=compounds,
        doses_um=doses_um,
        effects=effects,
        feature_ids=feature_ids,
    )
    design, context_ids, compound_ids, compound_dose_ids = (
        _training_additive_design(table)
    )
    intercept, coefficients = _ridge_with_unpenalized_intercept(
        design, table.effects
    )
    return TwoWayAdditiveRidgeDirectState(
        feature_ids=table.feature_ids,
        training_task_ids=table.task_ids,
        context_ids=context_ids,
        compound_ids=compound_ids,
        compound_dose_ids=compound_dose_ids,
        intercept=intercept,
        coefficients=coefficients,
    )


def _canonicalize_pca_signs(components: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(components, dtype=np.float64).copy()
    for row in result:
        # np.argmax returns the earliest feature in an exact absolute tie.
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            row *= -1.0
    return result


def fit_pca64_additive_ridge_direct(
    *,
    task_ids: Sequence[object],
    contexts: Sequence[object],
    compounds: Sequence[object],
    doses_um: Sequence[object],
    effects: object,
    feature_ids: Sequence[object],
) -> PCA64AdditiveRidgeDirectState:
    table = _training_table(
        task_ids=task_ids,
        contexts=contexts,
        compounds=compounds,
        doses_um=doses_um,
        effects=effects,
        feature_ids=feature_ids,
    )
    design, context_ids, compound_ids, compound_dose_ids = (
        _training_additive_design(table)
    )
    feature_mean = table.effects.mean(axis=0)
    centered = table.effects - feature_mean
    rank = min(PCA_MAX_RANK, len(table.task_ids) - 1, len(table.feature_ids))
    if rank:
        try:
            _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError as error:
            raise GSE306429BaselineError("PCA SVD failed") from error
        components = _canonicalize_pca_signs(right_vectors[:rank])
        scores = centered @ components.T
    else:
        components = np.empty((0, len(table.feature_ids)), dtype=np.float64)
        scores = np.empty((len(table.task_ids), 0), dtype=np.float64)
    latent_intercept, latent_coefficients = _ridge_with_unpenalized_intercept(
        design, scores
    )
    return PCA64AdditiveRidgeDirectState(
        feature_ids=table.feature_ids,
        training_task_ids=table.task_ids,
        context_ids=context_ids,
        compound_ids=compound_ids,
        compound_dose_ids=compound_dose_ids,
        feature_mean=feature_mean,
        components=components,
        latent_intercept=latent_intercept,
        latent_coefficients=latent_coefficients,
    )


def _rbf_inputs(
    contexts: tuple[str, ...],
    compounds: tuple[str, ...],
    doses_um: np.ndarray,
    *,
    context_ids: tuple[str, ...],
    compound_ids: tuple[str, ...],
    log10_dose_mean: float,
    log10_dose_sample_sd: float,
) -> np.ndarray:
    _reject_unseen(contexts, context_ids, name="context")
    _reject_unseen(compounds, compound_ids, name="compound")
    context_lookup = {value: index for index, value in enumerate(context_ids)}
    compound_lookup = {
        value: len(context_ids) + index
        for index, value in enumerate(compound_ids)
    }
    result = np.zeros(
        (len(contexts), len(context_ids) + len(compound_ids) + 1),
        dtype=np.float64,
    )
    for row, (context, compound) in enumerate(zip(contexts, compounds)):
        result[row, context_lookup[context]] = 1.0
        result[row, compound_lookup[compound]] = 1.0
    log_dose = np.log10(doses_um)
    if log10_dose_sample_sd == 0.0:
        result[:, -1] = 0.0
    else:
        result[:, -1] = (
            log_dose - log10_dose_mean
        ) / log10_dose_sample_sd
    if not np.isfinite(result).all():
        raise GSE306429BaselineError("RBF predictor matrix is nonfinite")
    return result


def _rbf_kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_squared = np.einsum("ij,ij->i", left, left)[:, None]
    right_squared = np.einsum("ij,ij->i", right, right)[None, :]
    squared_distance = left_squared + right_squared - 2.0 * (left @ right.T)
    # Roundoff can make an algebraic squared distance slightly negative.
    np.maximum(squared_distance, 0.0, out=squared_distance)
    kernel = np.exp(-RBF_GAMMA_FACTOR * squared_distance)
    if not np.isfinite(kernel).all():
        raise GSE306429BaselineError("RBF kernel is nonfinite")
    return kernel


def fit_rbf_kernel_ridge_direct(
    *,
    task_ids: Sequence[object],
    contexts: Sequence[object],
    compounds: Sequence[object],
    doses_um: Sequence[object],
    effects: object,
    feature_ids: Sequence[object],
) -> RBFKernelRidgeDirectState:
    table = _training_table(
        task_ids=task_ids,
        contexts=contexts,
        compounds=compounds,
        doses_um=doses_um,
        effects=effects,
        feature_ids=feature_ids,
    )
    if len(table.task_ids) < 2:
        raise GSE306429BaselineError(
            "RBF sample dose standardization requires at least two training tasks"
        )
    context_ids = _ordered_unique(table.contexts)
    compound_ids = _ordered_unique(table.compounds)
    log_doses = np.log10(table.doses_um)
    dose_mean = float(log_doses.mean())
    dose_sd = float(log_doses.std(ddof=1))
    if not math.isfinite(dose_mean) or not math.isfinite(dose_sd):
        raise GSE306429BaselineError("RBF dose standardization failed")
    # Treat exact floating zero as the registered zero-variance case.
    if dose_sd == 0.0:
        dose_sd = 0.0
    inputs = _rbf_inputs(
        table.contexts,
        table.compounds,
        table.doses_um,
        context_ids=context_ids,
        compound_ids=compound_ids,
        log10_dose_mean=dose_mean,
        log10_dose_sample_sd=dose_sd,
    )
    kernel = _rbf_kernel(inputs, inputs)
    # Enforce the algebraic symmetry before the positive ridge shift.
    kernel = 0.5 * (kernel + kernel.T)
    kernel.flat[:: len(table.task_ids) + 1] += RIDGE_LAMBDA
    try:
        dual_coefficients = np.linalg.solve(kernel, table.effects)
    except np.linalg.LinAlgError as error:
        raise GSE306429BaselineError("RBF ridge solve failed") from error
    if not np.isfinite(dual_coefficients).all():
        raise GSE306429BaselineError("RBF state is nonfinite")
    return RBFKernelRidgeDirectState(
        feature_ids=table.feature_ids,
        training_task_ids=table.task_ids,
        context_ids=context_ids,
        compound_ids=compound_ids,
        log10_dose_mean=dose_mean,
        log10_dose_sample_sd=dose_sd,
        training_inputs=inputs,
        dual_coefficients=dual_coefficients,
    )


def fit_internal_direct_method(
    method_id: str,
    *,
    task_ids: Sequence[object],
    contexts: Sequence[object],
    compounds: Sequence[object],
    doses_um: Sequence[object],
    effects: object,
    feature_ids: Sequence[object],
) -> DirectBaselineState:
    """Fit one registered trainable internal DIRECT method.

    ``NO_CHANGE_DIRECT_V1`` is deliberately excluded because its constructor
    accepts metadata support but no outcome-bearing ``effects`` argument.
    """

    fitters = {
        CONTEXT_MEAN: fit_context_mean_effect_direct,
        TWO_WAY_RIDGE: fit_two_way_additive_ridge_direct,
        PCA64_RIDGE: fit_pca64_additive_ridge_direct,
        RBF_RIDGE: fit_rbf_kernel_ridge_direct,
    }
    if method_id == NO_CHANGE:
        raise GSE306429BaselineError(
            "NO_CHANGE_DIRECT_V1 must use make_no_change_direct_state"
        )
    try:
        fitter = fitters[method_id]
    except KeyError as error:
        raise GSE306429BaselineError(
            f"unsupported internal DIRECT method: {method_id!r}"
        ) from error
    return fitter(
        task_ids=task_ids,
        contexts=contexts,
        compounds=compounds,
        doses_um=doses_um,
        effects=effects,
        feature_ids=feature_ids,
    )


def predict_internal_direct(
    state: DirectBaselineState,
    *,
    contexts: Sequence[object],
    compounds: Sequence[object],
    doses_um: Sequence[object],
    feature_ids: Sequence[object],
) -> np.ndarray:
    """Materialize a DIRECT effect without accepting any reference argument."""

    if not isinstance(
        state,
        (
            NoChangeDirectState,
            ContextMeanEffectDirectState,
            TwoWayAdditiveRidgeDirectState,
            PCA64AdditiveRidgeDirectState,
            RBFKernelRidgeDirectState,
        ),
    ):
        raise GSE306429BaselineError("unsupported internal DIRECT state")
    return state.predict(
        contexts=contexts,
        compounds=compounds,
        doses_um=doses_um,
        feature_ids=feature_ids,
    )


def _array_descriptor(name: str, value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(b"GSE306429_DIRECT_STATE_ARRAY_V1\0")
    digest.update(name.encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(item) for item in array.shape).encode("ascii"))
    digest.update(b"\0<f8\0")
    digest.update(array.tobytes(order="C"))
    return {
        "dtype": "<f8",
        "name": name,
        "sha256": digest.hexdigest(),
        "shape": list(array.shape),
    }


def _state_parts(
    state: DirectBaselineState,
) -> tuple[dict[str, object], Mapping[str, np.ndarray]]:
    common: dict[str, object] = {
        "c_pred_subtractions": 0,
        "compound_ids": list(state.compound_ids),
        "context_ids": list(state.context_ids),
        "feature_ids": list(state.feature_ids),
        "method_id": state.method_id,
        "prediction_semantics": "DIRECT",
        "uses_c_infer": False,
    }
    arrays: dict[str, np.ndarray] = {}
    if isinstance(state, NoChangeDirectState):
        common["training"] = "NO_FIT"
    else:
        common["training_task_ids"] = list(state.training_task_ids)
        if isinstance(state, ContextMeanEffectDirectState):
            arrays["context_effects"] = state.context_effects
        elif isinstance(state, TwoWayAdditiveRidgeDirectState):
            common["compound_dose_ids"] = list(state.compound_dose_ids)
            common["intercept_penalized"] = False
            common["ridge_lambda"] = state.ridge_lambda
            arrays.update(
                intercept=state.intercept,
                coefficients=state.coefficients,
            )
        elif isinstance(state, PCA64AdditiveRidgeDirectState):
            common["compound_dose_ids"] = list(state.compound_dose_ids)
            common["intercept_penalized"] = False
            common["max_rank"] = state.max_rank
            common["rank"] = state.rank
            common["ridge_lambda"] = state.ridge_lambda
            common["svd_sign_rule"] = (
                "largest_absolute_loading_positive_with_earliest_feature_tie_break"
            )
            arrays.update(
                feature_mean=state.feature_mean,
                components=state.components,
                latent_intercept=state.latent_intercept,
                latent_coefficients=state.latent_coefficients,
            )
        elif isinstance(state, RBFKernelRidgeDirectState):
            common["dose_standardization"] = "sample_sd_ddof_1"
            common["gamma_factor"] = state.gamma_factor
            common["kernel"] = "exp(-0.5*squared_euclidean_distance)"
            common["log10_dose_mean"] = state.log10_dose_mean
            common["log10_dose_sample_sd"] = state.log10_dose_sample_sd
            common["ridge_lambda"] = state.ridge_lambda
            arrays.update(
                training_inputs=state.training_inputs,
                dual_coefficients=state.dual_coefficients,
            )
        else:  # pragma: no cover - guarded by the public type and caller check
            raise GSE306429BaselineError("unsupported state")
    return common, arrays


def canonical_state_manifest(state: DirectBaselineState) -> dict[str, object]:
    """Return a canonical JSON-compatible manifest binding every state array."""

    if not isinstance(
        state,
        (
            NoChangeDirectState,
            ContextMeanEffectDirectState,
            TwoWayAdditiveRidgeDirectState,
            PCA64AdditiveRidgeDirectState,
            RBFKernelRidgeDirectState,
        ),
    ):
        raise GSE306429BaselineError("unsupported internal DIRECT state")
    metadata, arrays = _state_parts(state)
    return {
        "arrays": [
            _array_descriptor(name, arrays[name])
            for name in sorted(arrays, key=lambda value: value.encode("ascii"))
        ],
        "metadata": metadata,
        "schema": STATE_SCHEMA,
    }


def canonical_state_bytes(state: DirectBaselineState) -> bytes:
    """Serialize the hash manifest deterministically as strict ASCII JSON."""

    return json.dumps(
        canonical_state_manifest(state),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def canonical_state_sha256(state: DirectBaselineState) -> str:
    return hashlib.sha256(canonical_state_bytes(state)).hexdigest()


__all__ = [
    "CONTEXT_MEAN",
    "NO_CHANGE",
    "PCA64_RIDGE",
    "RBF_RIDGE",
    "RIDGE_LAMBDA",
    "SUPPORTED_METHODS",
    "TWO_WAY_RIDGE",
    "ContextMeanEffectDirectState",
    "DirectBaselineState",
    "GSE306429BaselineError",
    "NoChangeDirectState",
    "PCA64AdditiveRidgeDirectState",
    "RBFKernelRidgeDirectState",
    "TwoWayAdditiveRidgeDirectState",
    "canonical_state_bytes",
    "canonical_state_manifest",
    "canonical_state_sha256",
    "compound_dose_token",
    "fit_context_mean_effect_direct",
    "fit_internal_direct_method",
    "fit_pca64_additive_ridge_direct",
    "fit_rbf_kernel_ridge_direct",
    "fit_two_way_additive_ridge_direct",
    "make_no_change_direct_state",
    "normalized_dose",
    "predict_internal_direct",
]
