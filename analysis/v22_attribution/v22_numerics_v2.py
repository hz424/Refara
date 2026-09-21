"""Numerical helpers for the V22 attribution analysis."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray


class V22CoreError(ValueError):
    """Raised when numerical inputs do not meet the analysis requirements."""


def _finite_float(name: str, value: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise V22CoreError(f"{name} contains a non-finite value")
    return array


def _svd_sign_fix(components: NDArray[np.float64]) -> NDArray[np.float64]:
    """Use a deterministic sign for each component."""

    fixed = components.copy()
    for row in fixed:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            row *= -1.0
    return fixed


@dataclass(frozen=True)
class LowRankRidge:
    """A low-rank multivariate ridge fit with an unpenalized intercept."""

    control_center: NDArray[np.float64]
    components: NDArray[np.float64]
    coefficients: NDArray[np.float64]
    ridge_lambda: float

    def predict(self, control_profiles: ArrayLike) -> NDArray[np.float64]:
        controls = _finite_float("control_profiles", control_profiles)
        if controls.ndim != 2 or controls.shape[1] != self.control_center.shape[0]:
            raise V22CoreError(
                "query control profiles do not match the fitted gene axis"
            )
        scores = (controls - self.control_center) @ self.components.T
        design = np.column_stack((np.ones(controls.shape[0]), scores))
        return design @ self.coefficients


@dataclass(frozen=True)
class MatchedRidgeFits:
    """State- and effect-target ridge fits sharing exactly one design basis."""

    effect: LowRankRidge
    state: LowRankRidge


def fit_matched_rank4_ridge(
    control_profiles: ArrayLike,
    effect_targets: ArrayLike,
    state_targets: ArrayLike,
) -> MatchedRidgeFits:
    """Fit a matched rank-4 ridge pair with lambda 1.

    PCA and both regressions use only the arrays supplied by the caller.  The
    same centered rank-4 scores and the same unpenalized-intercept design are
    used for the state and effect targets; only the target matrix changes.
    """

    controls = _finite_float("control_profiles", control_profiles)
    effect = _finite_float("effect_targets", effect_targets)
    state = _finite_float("state_targets", state_targets)
    if controls.ndim != 2 or effect.ndim != 2 or state.ndim != 2:
        raise V22CoreError("matched-ridge inputs must all be two-dimensional")
    if effect.shape != state.shape or effect.shape != controls.shape:
        raise V22CoreError(
            "controls, effect targets, and state targets must share shape"
        )
    if controls.shape[0] < 5:
        raise V22CoreError("rank-4 ridge requires at least five training profiles")
    if controls.shape[1] == 0:
        raise V22CoreError("rank-4 ridge requires a nonempty gene axis")

    center = controls.mean(axis=0)
    centered = controls - center
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    tolerance = np.finfo(np.float64).eps * max(centered.shape) * singular_values[0]
    positive_rank = int(np.count_nonzero(singular_values > tolerance))
    if positive_rank < 4:
        raise V22CoreError(
            "training controls have fewer than four positive PCA directions"
        )
    components = _svd_sign_fix(vt[:4])
    scores = centered @ components.T
    design = np.column_stack((np.ones(controls.shape[0]), scores))
    penalty = np.diag((0.0, 1.0, 1.0, 1.0, 1.0))
    normal = design.T @ design + penalty

    def _fit(target: NDArray[np.float64]) -> LowRankRidge:
        coefficients = np.linalg.solve(normal, design.T @ target)
        return LowRankRidge(center.copy(), components.copy(), coefficients, 1.0)

    return MatchedRidgeFits(effect=_fit(effect), state=_fit(state))


def effect_to_state(
    effect_output: ArrayLike, reference: ArrayLike
) -> NDArray[np.float64]:
    """Convert a direct-effect output to state without any hidden reference."""

    effect = _finite_float("effect_output", effect_output)
    ref = _finite_float("reference", reference)
    if effect.shape != ref.shape:
        raise V22CoreError("effect output and reference must share shape")
    return effect + ref


def state_to_effect(
    state_output: ArrayLike, c_pred: ArrayLike
) -> NDArray[np.float64]:
    """Convert an absolute-state output to effect by subtracting Cpred once."""

    state = _finite_float("state_output", state_output)
    ref = _finite_float("c_pred", c_pred)
    if state.shape != ref.shape:
        raise V22CoreError("state output and Cpred must share shape")
    return state - ref


def adapt_output(
    output: ArrayLike,
    emitted_semantics: str,
    requested_semantics: str,
    reference: ArrayLike,
) -> NDArray[np.float64]:
    """Apply exactly one declared state/effect adapter."""

    emitted = emitted_semantics.upper()
    requested = requested_semantics.upper()
    if emitted not in {"EFFECT", "STATE"} or requested not in {"EFFECT", "STATE"}:
        raise V22CoreError("output semantics must be EFFECT or STATE")
    array = _finite_float("output", output)
    if emitted == requested:
        return array.copy()
    if emitted == "EFFECT":
        return effect_to_state(array, reference)
    return state_to_effect(array, reference)


def standardized_negative_mse(
    predicted_effect: ArrayLike,
    observed_effect: ArrayLike,
    gene_scale: ArrayLike,
    *,
    scale_floor: float = 0.1,
) -> float:
    """Return equal-gene negative standardized effect-scale MSE."""

    prediction = _finite_float("predicted_effect", predicted_effect)
    observed = _finite_float("observed_effect", observed_effect)
    scale = _finite_float("gene_scale", gene_scale)
    if prediction.shape != observed.shape:
        raise V22CoreError("predicted and observed effects must share shape")
    if prediction.ndim != 1 or not prediction.size or scale.shape != prediction.shape:
        raise V22CoreError(
            "utility inputs must be one-dimensional on the same gene axis"
        )
    if not np.isfinite(scale_floor) or scale_floor <= 0.0:
        raise V22CoreError("scale_floor must be finite and positive")
    if np.any(scale < 0.0):
        raise V22CoreError("gene_scale must contain nonnegative standard deviations")
    denominator = np.maximum(scale, scale_floor)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            utility = -float(np.mean(np.square((prediction - observed) / denominator)))
    except FloatingPointError as error:
        raise V22CoreError("utility exceeds floating-point range") from error
    if not np.isfinite(utility):
        raise V22CoreError("utility is nonfinite")
    return utility
