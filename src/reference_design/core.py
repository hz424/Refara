"""Balanced reference allocation and squared-error comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations, product
from numbers import Integral, Real
from typing import Any, Sequence

import numpy as np


PATTERNS = ("S", "M", "P", "O", "D")
ROLE_TUPLES = tuple(product(range(3), repeat=3))


def pattern_for(observation: int, prediction: int, model: int) -> str:
    """Classify a tuple in observation, prediction, model order."""
    if observation == prediction == model:
        return "S"
    if observation == prediction:
        return "M"
    if observation == model:
        return "P"
    if prediction == model:
        return "O"
    return "D"


_PATTERN_INDICES = tuple(
    tuple(i for i, roles in enumerate(ROLE_TUPLES) if pattern_for(*roles) == pattern)
    for pattern in PATTERNS
)


def _array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numbers")
    array = array.astype(np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values")
    return array


def _vector(values: Any, name: str) -> np.ndarray:
    array = _array(values, name)
    if array.ndim != 1 or not array.size:
        raise ValueError(f"{name} must be a nonempty gene vector")
    return array


def _name(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _ids(values: Any, length: int, name: str, unique: bool) -> tuple[str, ...]:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1 or len(array) != length:
        raise ValueError(f"{name} must have one entry per control cell")
    result = tuple(_name(value, name) for value in array)
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate identifiers")
    return result


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _tolerance(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative number")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return value


@dataclass(frozen=True)
class Prediction:
    """A native effect or state, fixed or supplied for three input blocks.

    ``kind='effect'`` declares an output already on the response scale; it uses
    no prediction-centring reference. Either output kind may depend on the input
    block, represented by a [3, gene] array in the same block order as controls.
    A state stored after baseline subtraction remains a state predictor: restore
    its baseline before constructing this object.
    """

    name: str
    values: Any
    kind: str = "state"

    def __post_init__(self) -> None:
        _name(self.name, "Model name")
        if self.kind not in ("state", "effect"):
            raise ValueError("Prediction kind must be 'state' or 'effect'")
        values = _array(self.values, f"Prediction {self.name}")
        if values.ndim == 1:
            if not values.size:
                raise ValueError("A prediction must contain at least one gene")
        elif values.ndim != 2 or values.shape[0] != 3 or values.shape[1] == 0:
            raise ValueError("A prediction must have shape [gene] or [3, gene]")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class Allocation:
    """Three disjoint blocks with the same sampled depth in every stratum."""

    means: np.ndarray
    membership: list[dict[str, Any]]
    depth: int
    seed: int
    strata: tuple[str, ...]

    @property
    def cells_per_block(self) -> int:
        return self.depth * len(self.strata)


@dataclass(frozen=True)
class ScoreResult:
    """MSEs have model then tuple/pattern axes; margins use negative MSE."""

    model_names: tuple[str, ...]
    model_kinds: tuple[str, ...]
    atomic_mse: np.ndarray
    mse: np.ndarray
    V: float
    pairwise: list[dict[str, Any]]
    rankings: list[dict[str, Any]]
    checks: dict[str, Any]
    patterns: tuple[str, ...] = PATTERNS
    role_tuples: tuple[tuple[int, int, int], ...] = ROLE_TUPLES

    @property
    def utility(self) -> np.ndarray:
        return -self.mse

    @property
    def atomic_utility(self) -> np.ndarray:
        return -self.atomic_mse


def allocate_controls(
    values: Any,
    cell_ids: Sequence[str],
    depth: int,
    seed: int,
    strata: Sequence[str] | None = None,
    blocks: int = 3,
) -> Allocation:
    """Sample two or three equal-depth blocks without replacement within each stratum.

    Each block mean weights strata equally. Reusing the seed gives nested blocks
    at increasing depths. Input row order does not change the selected cells.
    """
    cells = _array(values, "Control cells")
    if cells.ndim != 2 or not all(cells.shape):
        raise ValueError("Control cells must have shape [cell, gene]")
    identifiers = _ids(cell_ids, len(cells), "Cell IDs", unique=True)
    depth = _integer(depth, "Depth", 1)
    seed = _integer(seed, "Seed", 0)
    labels = ("all",) * len(cells) if strata is None else _ids(
        strata, len(cells), "Strata", unique=False
    )
    blocks = _integer(blocks, "Blocks", 2)
    if blocks not in (2, 3):
        raise ValueError("Blocks must be 2 or 3")
    groups = tuple(sorted(set(labels)))
    rows_by_group = {
        group: sorted((i for i, label in enumerate(labels) if label == group),
                      key=lambda i: identifiers[i])
        for group in groups
    }
    for group, rows in rows_by_group.items():
        if len(rows) < blocks * depth:
            raise ValueError(
                f"Stratum {group!r} has {len(rows)} controls; {blocks} blocks need {blocks * depth}"
            )
    rng = np.random.default_rng(seed)
    block_rows: list[list[int]] = [[] for _ in range(blocks)]
    membership: list[dict[str, Any]] = []
    for group, rows in rows_by_group.items():
        chosen = rng.permutation(rows)[:blocks * depth].reshape(depth, blocks)
        for block in range(blocks):
            for within_block_index, cell_index in enumerate(chosen[:, block]):
                index = int(cell_index)
                block_rows[block].append(index)
                membership.append(dict(
                    cell_id=identifiers[index], cell_index=index, stratum=group,
                    block=block, within_block_index=within_block_index,
                ))
    with np.errstate(over="raise", invalid="raise"):
        means = np.stack([cells[rows].mean(axis=0) for rows in block_rows])
    return Allocation(means, membership, depth, seed, groups)


def _conversion(values: Any, baseline: Any, sign: int) -> np.ndarray:
    values = _array(values, "Prediction")
    baseline = _array(baseline, "Baseline")
    if values.ndim not in (1, 2) or not all(values.shape):
        raise ValueError("Prediction must have shape [gene] or [row, gene]")
    if baseline.shape != values.shape and baseline.shape != (values.shape[-1],):
        raise ValueError("Baseline must match the prediction or its gene axis")
    with np.errstate(over="raise", invalid="raise"):
        return values + sign * baseline


def state_to_effect(state: Any, baseline: Any) -> np.ndarray:
    """Subtract an explicit baseline; the predictor retains its state semantics."""
    return _conversion(state, baseline, -1)


def effect_to_state(effect: Any, baseline: Any) -> np.ndarray:
    """Add back the same baseline used to represent a state as an effect."""
    return _conversion(effect, baseline, 1)


def _classification(d_s: float, d_d: float, tolerance: float, mixed: bool,
                    identity_applicable: bool = True) -> str:
    if abs(d_s) <= tolerance or abs(d_d) <= tolerance:
        return "numerical_tie"
    if d_s < 0 < d_d:
        return "state_to_effect" if mixed and identity_applicable else "ranking_reversal"
    if d_d < 0 < d_s:
        return "unexpected_reversal" if mixed and identity_applicable else "ranking_reversal"
    if d_s > 0:
        return "effect_preferred_both" if mixed else "a_preferred_both"
    return "state_preferred_both" if mixed else "b_preferred_both"


def score_references(
    treated: Any,
    controls: Any,
    predictions: Sequence[Prediction],
    scales: Any | None = None,
    tie_tolerance: float = 1e-12,
    identity_atol: float = 1e-10,
    identity_rtol: float = 1e-10,
) -> ScoreResult:
    """Score all 27 balanced assignments before averaging within each pattern.

    Arrays share a fixed gene order and expression scale. ``controls`` has shape
    [3, gene]; either conditioned output kind follows these same three blocks.
    Scales, when supplied, are fixed positive gene divisors. Positive pair margins
    favor model A; mixed comparisons put the native effect in position A.
    Fixed-effect margin identities are evaluated only for pairs without an
    input-conditioned native effect. All other pairs retain descriptive scores.
    """
    target = _vector(treated, "Treated mean")
    blocks = _array(controls, "Control means")
    if blocks.shape != (3, len(target)):
        raise ValueError("Control means must have shape [3, gene] matching the target")
    divisors = np.ones_like(target) if scales is None else _vector(scales, "Scales")
    if divisors.shape != target.shape or np.any(divisors <= 0):
        raise ValueError("Scales must be positive and match the target gene axis")
    tie_tolerance = _tolerance(tie_tolerance, "Tie tolerance")
    identity_atol = _tolerance(identity_atol, "Identity absolute tolerance")
    identity_rtol = _tolerance(identity_rtol, "Identity relative tolerance")
    predictions = tuple(predictions)
    if not predictions or any(not isinstance(p, Prediction) for p in predictions):
        raise ValueError("Supply at least one Prediction")
    names = tuple(p.name for p in predictions)
    kinds = tuple(p.kind for p in predictions)
    if len(set(names)) != len(names):
        raise ValueError("Model names must be unique")
    if any(p.values.shape[-1] != len(target) for p in predictions):
        raise ValueError("Every prediction must match the target gene axis")

    atomic = np.empty((len(predictions), len(ROLE_TUPLES)), dtype=np.float64)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for mi, prediction in enumerate(predictions):
            for ti, (observation, prediction_block, model_block) in enumerate(ROLE_TUPLES):
                if prediction.kind == "effect":
                    effect = prediction.values if prediction.values.ndim == 1 else prediction.values[model_block]
                    # Form the observed effect first, so a large shared
                    # target/reference cannot erase a small native effect.
                    residual = effect - (target - blocks[observation])
                else:
                    state = prediction.values if prediction.values.ndim == 1 else prediction.values[model_block]
                    residual = (state - target) + (blocks[observation] - blocks[prediction_block])
                atomic[mi, ti] = np.mean((residual / divisors) ** 2)
        mse = np.stack([atomic[:, ids].mean(axis=1) for ids in _PATTERN_INDICES], axis=1)
        distance = float(np.mean([
            np.mean(((blocks[i] - blocks[j]) / divisors) ** 2)
            for i, j in permutations(range(3), 2)
        ]))
    if not np.isfinite(atomic).all() or not np.isfinite(distance):
        raise FloatingPointError("Scores exceed floating-point range")

    pairwise: list[dict[str, Any]] = []
    for first, second in combinations(sorted(range(len(names)), key=lambda i: names[i]), 2):
        mixed = kinds[first] != kinds[second]
        a, b = (second, first) if mixed and kinds[first] == "state" else (first, second)
        d_s = float(mse[b, 0] - mse[a, 0])
        d_d = float(mse[b, 4] - mse[a, 4])
        applicable = not any(predictions[index].kind == "effect" and predictions[index].values.ndim == 2
                             for index in (a, b))
        if applicable:
            displacement = distance if mixed else 0.0
            predicted = d_s + displacement
            residual = d_d - predicted
            identity_bound = identity_atol + identity_rtol * max(abs(d_d), abs(d_s), displacement)
            verified = bool(abs(residual) <= identity_bound)
            expected = bool((d_s < -tie_tolerance and predicted > tie_tolerance)
                            or (d_s > tie_tolerance and predicted < -tie_tolerance))
        else:
            predicted = residual = identity_bound = verified = expected = None
        observed = ((d_s < -tie_tolerance and d_d > tie_tolerance)
                    or (d_s > tie_tolerance and d_d < -tie_tolerance))
        pairwise.append(dict(
            model_a=names[a], model_b=names[b], kind_a=kinds[a], kind_b=kinds[b],
            d_S=d_s, V=distance, d_D=d_d, predicted_d_D=predicted,
            identity_residual=residual, identity_applicable=applicable,
            identity_verified=verified, identity_tolerance=float(identity_bound) if applicable else None,
            theory=("reference_penalty" if mixed else "zero_displacement") if applicable else "not_applicable_input_conditioned_effect",
            condition_strict=bool(-distance < d_s < 0) if mixed and applicable else None,
            expected_crossing=expected, observed_crossing=bool(observed),
            S_tie=bool(abs(d_s) <= tie_tolerance), D_tie=bool(abs(d_d) <= tie_tolerance),
            classification=_classification(d_s, d_d, tie_tolerance, mixed, applicable),
        ))

    rankings: list[dict[str, Any]] = []
    for pi, pattern in enumerate(PATTERNS):
        for mi in sorted(range(len(names)), key=lambda i: (mse[i, pi], names[i])):
            losses = mse[:, pi]
            close = np.abs(losses - losses[mi]) <= tie_tolerance
            rankings.append(dict(
                pattern=pattern, model=names[mi], kind=kinds[mi], mse=float(losses[mi]),
                rank=1 + int(np.count_nonzero(losses < losses[mi] - tie_tolerance)),
                tied=bool(np.count_nonzero(close) > 1),
            ))
    applicable_pairs = [pair for pair in pairwise if pair["identity_applicable"]]
    checks = dict(
        all_identities_verified=all(pair["identity_applicable"] and pair["identity_verified"] for pair in pairwise),
        all_applicable_identities_verified=all(pair["identity_verified"] for pair in applicable_pairs),
        applicable_identity_count=len(applicable_pairs),
        inapplicable_identity_count=len(pairwise) - len(applicable_pairs),
        identity_failures=sum(not pair["identity_verified"] for pair in applicable_pairs),
        max_identity_residual=max((abs(pair["identity_residual"]) for pair in applicable_pairs), default=0.0),
        identity_atol=identity_atol, identity_rtol=identity_rtol, tie_tolerance=tie_tolerance,
    )
    return ScoreResult(names, kinds, atomic, mse, distance, pairwise, rankings, checks)
