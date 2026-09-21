"""Numerical routines for the V22 factor-isolation analysis.

Inputs are precomputed fold-level arrays; data loading and model fitting are
handled elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


ENDPOINT_NAMES = ("E1", "E2", "E3", "E4")
ENDPOINT_DIRECTIONS = np.asarray((1.0, 1.0, -1.0, 1.0))
EXPECTED_DONORS = 8
EXPECTED_TASKS = 4
EXPECTED_SEEDS = 5
EXPECTED_PERMUTATIONS = 6


class V22CoreError(ValueError):
    """Base class for a violated frozen-core invariant."""


class IncompleteFormalFamily(V22CoreError):
    """Raised when the all-or-none formal completeness rule is violated."""


def _finite_float(name: str, value: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise V22CoreError(f"{name} contains a non-finite value")
    return array


def _require_shape(name: str, value: NDArray[np.float64], shape: tuple[int, ...]) -> None:
    if value.shape != shape:
        raise V22CoreError(f"{name} has shape {value.shape}; expected {shape}")


def _svd_sign_fix(components: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply the frozen largest-absolute-loading/earliest-feature sign rule."""

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
            raise V22CoreError("query control profiles do not match the fitted gene axis")
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
    """Fit the frozen rank-4, lambda-1 matched ridge pair.

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
        raise V22CoreError("controls, effect targets, and state targets must share shape")
    if controls.shape[0] < 5:
        raise V22CoreError("rank-4 ridge requires at least five training profiles")

    center = controls.mean(axis=0)
    centered = controls - center
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    tolerance = np.finfo(np.float64).eps * max(centered.shape) * singular_values[0]
    positive_rank = int(np.count_nonzero(singular_values > tolerance))
    if positive_rank < 4:
        raise V22CoreError("training controls have fewer than four positive PCA directions")
    components = _svd_sign_fix(vt[:4])
    scores = centered @ components.T
    design = np.column_stack((np.ones(controls.shape[0]), scores))
    penalty = np.diag((0.0, 1.0, 1.0, 1.0, 1.0))
    normal = design.T @ design + penalty

    def _fit(target: NDArray[np.float64]) -> LowRankRidge:
        coefficients = np.linalg.solve(normal, design.T @ target)
        return LowRankRidge(center.copy(), components.copy(), coefficients, 1.0)

    return MatchedRidgeFits(effect=_fit(effect), state=_fit(state))


def effect_to_state(effect_output: ArrayLike, reference: ArrayLike) -> NDArray[np.float64]:
    """Convert a direct-effect output to state without any hidden reference."""

    effect = _finite_float("effect_output", effect_output)
    ref = _finite_float("reference", reference)
    if effect.shape != ref.shape:
        raise V22CoreError("effect output and reference must share shape")
    return effect + ref


def state_to_effect(state_output: ArrayLike, c_pred: ArrayLike) -> NDArray[np.float64]:
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
    if prediction.ndim != 1 or scale.shape != prediction.shape:
        raise V22CoreError("utility inputs must be one-dimensional on the same gene axis")
    if not np.isfinite(scale_floor) or scale_floor <= 0.0:
        raise V22CoreError("scale_floor must be finite and positive")
    if np.any(scale < 0.0):
        raise V22CoreError("gene_scale must contain nonnegative standard deviations")
    denominator = np.maximum(scale, scale_floor)
    return -float(np.mean(np.square((prediction - observed) / denominator)))


@dataclass(frozen=True)
class EndpointResult:
    donor_values: NDArray[np.float64]
    estimate: float


def _finish_endpoint(name: str, donor_values: NDArray[np.float64]) -> EndpointResult:
    donor_values = _finite_float(f"{name} donor values", donor_values)
    _require_shape(f"{name} donor values", donor_values, (EXPECTED_DONORS,))
    return EndpointResult(donor_values=donor_values, estimate=float(donor_values.mean()))


def endpoint_e1(
    ridge_utility: ArrayLike,
    cellflow_utility: ArrayLike,
) -> EndpointResult:
    """Frozen-adapter output-mismatch by Cpred interaction.

    Arrays have axes donor, task, seed, permutation, Cpred(A/B), and
    representation(EFFECT/STATE).  For k in {A,B}, R_k^{ab}=U_R^a-U_F^b.
    E1 averages
      [R_B^{ES}-(R_B^{EE}+R_B^{SS})/2]
      -[R_A^{ES}-(R_A^{EE}+R_A^{SS})/2].
    The fixed order is permutation -> task -> seed -> donor.
    """

    expected = (EXPECTED_DONORS, EXPECTED_TASKS, EXPECTED_SEEDS, EXPECTED_PERMUTATIONS, 2, 2)
    ridge = _finite_float("ridge_utility", ridge_utility)
    flow = _finite_float("cellflow_utility", cellflow_utility)
    _require_shape("ridge_utility", ridge, expected)
    _require_shape("cellflow_utility", flow, expected)

    def mismatch(k: int) -> NDArray[np.float64]:
        r_es = ridge[..., k, 0] - flow[..., k, 1]
        r_ee = ridge[..., k, 0] - flow[..., k, 0]
        r_ss = ridge[..., k, 1] - flow[..., k, 1]
        return r_es - 0.5 * (r_ee + r_ss)

    contrast = mismatch(1) - mismatch(0)
    by_permutation = contrast.mean(axis=3)
    by_task = by_permutation.mean(axis=1)
    by_seed = by_task.mean(axis=1)
    return _finish_endpoint("E1", by_seed)


def endpoint_e2(
    effect_trained_effect_view_utility: ArrayLike,
    state_trained_state_view_utility: ArrayLike,
) -> EndpointResult:
    """Matched-ridge refit interaction, with no seed axis.

    Arrays have axes donor, task, permutation, Cpred(A/B).  The contrast is
    (U_effect,B-U_state,B) - (U_effect,A-U_state,A), followed by the fixed
    permutation -> task -> donor aggregation.
    """

    expected = (EXPECTED_DONORS, EXPECTED_TASKS, EXPECTED_PERMUTATIONS, 2)
    effect = _finite_float("effect-trained utility", effect_trained_effect_view_utility)
    state = _finite_float("state-trained utility", state_trained_state_view_utility)
    _require_shape("effect-trained utility", effect, expected)
    _require_shape("state-trained utility", state, expected)
    contrast = (effect[..., 1] - state[..., 1]) - (effect[..., 0] - state[..., 0])
    by_permutation = contrast.mean(axis=2)
    by_task = by_permutation.mean(axis=1)
    return _finish_endpoint("E2", by_task)


def endpoints_e3_e4(cellflow_q: ArrayLike) -> tuple[EndpointResult, EndpointResult]:
    """Directional Cmodel bundle contrasts Q(A)-Q(C) and Q(B)-Q(C).

    ``cellflow_q`` has axes donor, task, seed, permutation, Cmodel(A/B/C).
    Each utility must already use the same physical block for the source cells
    and its context embedding.  Aggregation is permutation -> task -> seed ->
    donor; label balance is therefore resolved before any donor pooling.
    """

    expected = (EXPECTED_DONORS, EXPECTED_TASKS, EXPECTED_SEEDS, EXPECTED_PERMUTATIONS, 3)
    q = _finite_float("cellflow_q", cellflow_q)
    _require_shape("cellflow_q", q, expected)

    def aggregate(contrast: NDArray[np.float64], name: str) -> EndpointResult:
        by_permutation = contrast.mean(axis=3)
        by_task = by_permutation.mean(axis=1)
        by_seed = by_task.mean(axis=1)
        return _finish_endpoint(name, by_seed)

    return aggregate(q[..., 0] - q[..., 2], "E3"), aggregate(q[..., 1] - q[..., 2], "E4")


def stack_endpoint_results(results: Sequence[EndpointResult]) -> NDArray[np.float64]:
    """Return the frozen donor-by-endpoint matrix in E1,E2,E3,E4 order."""

    if len(results) != 4:
        raise V22CoreError("exactly four endpoint results are required")
    matrix = np.column_stack([result.donor_values for result in results])
    _require_shape("endpoint matrix", matrix, (EXPECTED_DONORS, 4))
    return _finite_float("endpoint matrix", matrix)


@dataclass(frozen=True)
class SimultaneousInference:
    endpoint_names: tuple[str, ...]
    directions: NDArray[np.float64]
    estimates: NDArray[np.float64]
    standard_errors: NDArray[np.float64]
    critical_value: float
    native_lower: NDArray[np.float64]
    native_upper: NDArray[np.float64]
    oriented_lower: NDArray[np.float64]
    oriented_upper: NDArray[np.float64]
    endpoint_supported: NDArray[np.bool_]
    uniform_counts: NDArray[np.int64]
    uniform_8_of_8: NDArray[np.bool_]
    sign_vector_count: int
    critical_order_one_indexed: int


def exact_shared_sign_max_abs_t(
    donor_by_endpoint: ArrayLike,
    *,
    alpha: float = 0.05,
    directions: ArrayLike = ENDPOINT_DIRECTIONS,
) -> SimultaneousInference:
    """Compute the exact 2^8 shared-sign max-|t| working-law intervals.

    A single sign vector is applied to the centered four-endpoint residual row
    of each donor.  Every one of the 256 sign vectors is enumerated.  The
    critical value is the ceil((1-alpha)*256)-th ordered maximum absolute
    recentered, restudentized statistic.  Endpoint support requires the simultaneous
    lower bound on the predeclared oriented scale to be strictly greater than
    zero; equality is not support.
    """

    values = _finite_float("donor_by_endpoint", donor_by_endpoint)
    _require_shape("donor_by_endpoint", values, (EXPECTED_DONORS, 4))
    direction = _finite_float("directions", directions)
    _require_shape("directions", direction, (4,))
    if not np.isin(direction, (-1.0, 1.0)).all():
        raise V22CoreError("each endpoint direction must be +1 or -1")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise V22CoreError("alpha must lie strictly between zero and one")

    estimates = values.mean(axis=0)
    standard_errors = values.std(axis=0, ddof=1) / np.sqrt(EXPECTED_DONORS)
    if np.any(standard_errors <= 0.0):
        raise V22CoreError("working-law inference requires positive endpoint standard errors")
    residuals = values - estimates
    influence = residuals / EXPECTED_DONORS
    influence_sum_squares = np.sum(np.square(influence), axis=0)

    maxima: list[float] = []
    unbounded = False
    for signs_tuple in product((-1.0, 1.0), repeat=EXPECTED_DONORS):
        signs = np.asarray(signs_tuple)[:, None]
        signed_sum = np.sum(signs * influence, axis=0)
        denominator_squared = (EXPECTED_DONORS / (EXPECTED_DONORS - 1.0)) * (
            influence_sum_squares - np.square(signed_sum) / EXPECTED_DONORS
        )
        # Roundoff can make an algebraically nonnegative value very slightly
        # negative. A genuinely zero denominator makes the frozen working law
        # unbounded and therefore cannot support any endpoint.
        tolerance = np.finfo(np.float64).eps * np.maximum(1.0, influence_sum_squares)
        denominator_squared = np.where(
            (denominator_squared < 0.0) & (denominator_squared >= -tolerance),
            0.0,
            denominator_squared,
        )
        if np.any(denominator_squared <= 0.0):
            unbounded = True
            maxima.append(float("inf"))
            continue
        statistic = np.abs(signed_sum) / np.sqrt(denominator_squared)
        maxima.append(float(np.max(statistic)))

    sign_count = len(maxima)
    if sign_count != 256:
        raise AssertionError("the eight-donor sign schedule must contain exactly 256 vectors")
    critical_order = int(np.ceil((1.0 - alpha) * sign_count))
    critical = (
        float("inf")
        if unbounded
        else float(np.sort(np.asarray(maxima))[critical_order - 1])
    )
    native_lower = estimates - critical * standard_errors
    native_upper = estimates + critical * standard_errors
    oriented_estimate = direction * estimates
    oriented_half_width = critical * standard_errors
    oriented_lower = oriented_estimate - oriented_half_width
    oriented_upper = oriented_estimate + oriented_half_width
    supported = oriented_lower > 0.0
    oriented_donor_values = values * direction[None, :]
    uniform_counts = np.count_nonzero(oriented_donor_values > 0.0, axis=0).astype(np.int64)

    return SimultaneousInference(
        endpoint_names=ENDPOINT_NAMES,
        directions=direction.copy(),
        estimates=estimates,
        standard_errors=standard_errors,
        critical_value=critical,
        native_lower=native_lower,
        native_upper=native_upper,
        oriented_lower=oriented_lower,
        oriented_upper=oriented_upper,
        endpoint_supported=supported,
        uniform_counts=uniform_counts,
        uniform_8_of_8=uniform_counts == EXPECTED_DONORS,
        sign_vector_count=sign_count,
        critical_order_one_indexed=critical_order,
    )


@dataclass(frozen=True)
class ClaimMapping:
    representation_pair_supported: bool
    cmodel_pair_supported: bool
    all_four_supported: bool
    all_four_uniform_8_of_8: bool
    strict_reversal_secondary: bool
    claim: str


def map_claim(
    endpoint_supported: ArrayLike,
    uniform_8_of_8: ArrayLike,
    *,
    strict_reversal: bool,
    complete: bool,
) -> ClaimMapping:
    """Map predeclared pair/joint support to the frozen claim ladder.

    Strict reversal is a separately reported secondary result.  It cannot
    create or strengthen a primary mechanism claim.
    """

    supported = np.asarray(endpoint_supported, dtype=bool)
    uniform = np.asarray(uniform_8_of_8, dtype=bool)
    if supported.shape != (4,) or uniform.shape != (4,):
        raise V22CoreError("claim mapping requires four support and four uniform flags")
    representation_pair = bool(supported[0] and supported[1])
    cmodel_pair = bool(supported[2] and supported[3])
    joint = bool(representation_pair and cmodel_pair)
    joint_uniform = bool(joint and uniform.all())

    if not complete:
        claim = "INCONCLUSIVE_INCOMPLETE"
    elif joint:
        claim = "JOINT_ATTRIBUTION_SUPPORTED"
    elif representation_pair:
        claim = "OUTPUT_REPRESENTATION_ATTRIBUTION_SUPPORTED"
    elif cmodel_pair:
        claim = "MODEL_CONDITIONING_ALLOCATION_ATTRIBUTION_SUPPORTED"
    else:
        claim = "NO_PAIRED_PRIMARY_ATTRIBUTION_SUPPORTED"

    return ClaimMapping(
        representation_pair,
        cmodel_pair,
        joint,
        joint_uniform,
        bool(strict_reversal),
        claim,
    )


def is_strict_reversal(contrast_under_a: float, contrast_under_b: float) -> bool:
    """Secondary strict reversal: both finite, nonzero, and opposite signs."""

    pair = _finite_float("reversal contrasts", (contrast_under_a, contrast_under_b))
    return bool(pair[0] != 0.0 and pair[1] != 0.0 and pair[0] * pair[1] < 0.0)


def require_complete_formal_family(
    cellflow_fit_complete: ArrayLike,
    matched_ridge_fit_complete: ArrayLike,
    donor_by_endpoint: ArrayLike,
) -> bool:
    """Enforce 40/40 CellFlow, 16/16 ridge, and 8x4 finite endpoints."""

    cellflow = np.asarray(cellflow_fit_complete)
    ridge = np.asarray(matched_ridge_fit_complete)
    endpoints = np.asarray(donor_by_endpoint, dtype=np.float64)
    if cellflow.shape != (EXPECTED_DONORS, EXPECTED_SEEDS) or not np.all(cellflow == True):  # noqa: E712
        raise IncompleteFormalFamily("CellFlow formal family is not complete 40/40")
    if ridge.shape != (EXPECTED_DONORS, 2) or not np.all(ridge == True):  # noqa: E712
        raise IncompleteFormalFamily("matched-ridge formal family is not complete 16/16")
    if endpoints.shape != (EXPECTED_DONORS, 4) or not np.isfinite(endpoints).all():
        raise IncompleteFormalFamily("the four primary endpoints are not complete for all eight donors")
    return True


def endpoint_dict(donor_by_endpoint: ArrayLike) -> Mapping[str, NDArray[np.float64]]:
    """Expose the four frozen donor vectors without changing their order."""

    values = _finite_float("donor_by_endpoint", donor_by_endpoint)
    _require_shape("donor_by_endpoint", values, (EXPECTED_DONORS, 4))
    return {name: values[:, index].copy() for index, name in enumerate(ENDPOINT_NAMES)}
