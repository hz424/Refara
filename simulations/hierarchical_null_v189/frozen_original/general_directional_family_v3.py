#!/usr/bin/env python3
"""Exact root-level directional inference for a declared complete family.

The resolver consumes one finite utility for every
``scheme x independent-root x model`` combination.  It does not construct
utilities or certify that upstream rows are genuinely independent roots.

V3 generalizes the frozen M8/three-scheme predecessor to arbitrary declared
model and scheme families.  It deliberately distinguishes independent roots
with a common conditional sign probability, where rejected directions may
populate a reported graph, from an independent heterogeneous intersection-null
regime, where graph/rank summaries are withheld because a rejection need not
describe a common population direction.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


METHOD_ID = "GENERAL_DIRECTIONAL_FAMILY_V3"
PROTOCOL_ID = "GENERAL_DIRECTIONAL_FAMILY_METHOD_V3_PROTOCOL"
SCHEMA_VERSION = "3.0.0"

REGIME_COMMON_SIGN = "INDEPENDENT_ROOTS_COMMON_SIGN_PROBABILITY"
REGIME_HETEROGENEOUS = "INDEPENDENT_HETEROGENEOUS_ROOTWISE_INTERSECTION_NULL"
ALLOWED_REGIMES = (REGIME_COMMON_SIGN, REGIME_HETEROGENEOUS)

ALPHA_NUMERATOR = 1
ALPHA_DENOMINATOR = 20
CONTIGUOUS_INTEGER_LIMIT = 1 << 53
MAX_IDENTIFIER_UTF8_BYTES = 256
MAX_DIRECTIONAL_HYPOTHESES = 10_000
MAX_DIRECTIONAL_ROOT_WORK = 2_000_000
MAX_MODELS = 64
MAX_ROOTS = 4096


class DirectionalFamilyError(RuntimeError):
    """Fail-closed input, arithmetic, or family-contract error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DirectionalFamilyError(message)


def _plain_int(value: object, name: str) -> int:
    require(type(value) is int, f"{name} must be a plain Python int")
    return value


def _validate_alpha(numerator: object, denominator: object) -> tuple[int, int]:
    alpha_num = _plain_int(numerator, "alpha_numerator")
    alpha_den = _plain_int(denominator, "alpha_denominator")
    require(alpha_num > 0 and alpha_den > 0, "alpha components must be positive")
    require(2 * alpha_num <= alpha_den,
            "alpha must be at most one half for the directional family")
    return alpha_num, alpha_den


def _validate_identifiers(
    values: Sequence[str], *, name: str, minimum: int
) -> tuple[str, ...]:
    require(type(values) in (list, tuple), f"{name} must be a list or tuple")
    require(len(values) >= minimum, f"{name} must contain at least {minimum} values")
    result: list[str] = []
    for index, value in enumerate(values):
        require(type(value) is str and bool(value),
                f"{name}[{index}] must be a nonempty plain string")
        require(unicodedata.normalize("NFC", value) == value,
                f"{name}[{index}] must already be NFC-normalized")
        try:
            encoded = value.encode("utf-8")
        except UnicodeEncodeError as error:
            raise DirectionalFamilyError(
                f"{name}[{index}] is not valid UTF-8 text"
            ) from error
        require(len(encoded) <= MAX_IDENTIFIER_UTF8_BYTES,
                f"{name}[{index}] is too long")
        require(not any(ord(character) < 32 or ord(character) == 127 for character in value),
                f"{name}[{index}] contains a control character")
        result.append(value)
    require(len(set(result)) == len(result), f"{name} contains duplicate values")
    return tuple(result)


def _registry_sha256(namespace: str, identifiers: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(namespace.encode("ascii") + b"\x00")
    for value in identifiers:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _lossless_float64_copy(utilities: np.ndarray) -> np.ndarray:
    require(type(utilities) is np.ndarray,
            "utilities must be an exact base numpy.ndarray; subclasses are forbidden")
    source = utilities
    require(source.ndim == 3, "utilities must have scheme x root x model shape")
    require(source.dtype.kind in "fiu", "utilities must have a real float or integer dtype")
    require(bool(np.isfinite(source).all()), "utilities must be finite")

    if source.dtype.kind == "i":
        require(bool((source >= -CONTIGUOUS_INTEGER_LIMIT).all()) and
                bool((source <= CONTIGUOUS_INTEGER_LIMIT).all()),
                "signed integer lies outside the contiguous exact float64 interval")
    elif source.dtype.kind == "u":
        require(bool((source <= CONTIGUOUS_INTEGER_LIMIT).all()),
                "unsigned integer lies outside the contiguous exact float64 interval")

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        converted = np.array(source, dtype=np.float64, order="C", copy=True)
    require(type(converted) is np.ndarray and converted.dtype == np.dtype("float64"),
            "conversion did not produce a base float64 ndarray")
    require(converted.flags.owndata and converted.flags.c_contiguous and
            converted.dtype.isnative,
            "conversion did not produce an owned native C-contiguous array")
    require(bool(np.isfinite(converted).all()),
            "float64 conversion produced nonfinite values")

    if source.dtype.kind == "f" and source.dtype.itemsize > np.dtype("float64").itemsize:
        roundtrip = converted.astype(source.dtype, copy=True)
        require(bool(np.array_equal(roundtrip, source)),
                "wider float values do not round-trip exactly through float64")
        zeros = source == 0
        require(bool(np.array_equal(np.signbit(roundtrip[zeros]), np.signbit(source[zeros]))),
                "wider float zero signs do not round-trip through float64")
    return converted


def exact_upper_tail(positive_count: int, nonzero_count: int) -> tuple[int, int]:
    """Return exact ``P[Binomial(K, 1/2) >= W]`` as integers."""

    wins = _plain_int(positive_count, "positive_count")
    total = _plain_int(nonzero_count, "nonzero_count")
    require(0 <= wins <= total, "invalid sign counts")
    require(total <= MAX_ROOTS, "nonzero_count exceeds the implementation safety limit")
    return _exact_upper_tail_validated(wins, total)


@lru_cache(maxsize=4096)
def _exact_upper_tail_validated(wins: int, total: int) -> tuple[int, int]:
    denominator = 1 << total

    # Sum the shorter symmetric side with a recurrence.  This is exact, uses
    # O(min(W, K-W+1)) integer operations, and avoids repeated large `comb`
    # calls near the advertised root-count limit.
    if wins <= total // 2:
        stop = wins - 1
        complement = True
    else:
        stop = total - wins
        complement = False
    coefficient = 1
    partial = 0
    for value in range(stop + 1):
        if value:
            coefficient = coefficient * (total - value + 1) // value
        partial += coefficient
    numerator = denominator - partial if complement else partial
    return numerator, denominator


def _safe_fraction_float(value: Fraction) -> tuple[float | None, bool]:
    """Return a non-authoritative float without mapping positive values to 0."""

    converted = float(value)
    if value > 0 and converted == 0.0:
        return None, True
    return converted, False


def _fraction_payload(value: Fraction) -> dict[str, Any]:
    clipped = min(Fraction(1, 1), max(Fraction(0, 1), value))
    value_float, underflow = _safe_fraction_float(clipped)
    return {
        "numerator": clipped.numerator,
        "denominator": clipped.denominator,
        "value_float": value_float,
        "value_float_underflow": underflow,
        "exact_fraction_is_authoritative": True,
    }


def _exact_ratio_payload(numerator: int, denominator: int) -> dict[str, Any]:
    value_float, underflow = _safe_fraction_float(Fraction(numerator, denominator))
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value_float": value_float,
        "value_float_underflow": underflow,
        "exact_fraction_is_authoritative": True,
    }


def _exact_reject(
    numerator: int,
    denominator: int,
    *,
    remaining: int,
    alpha_numerator: int,
    alpha_denominator: int,
) -> bool:
    require(0 <= numerator <= denominator and denominator > 0,
            "p-value fraction lies outside [0,1]")
    require(remaining >= 1, "remaining multiplicity must be positive")
    return numerator * alpha_denominator * remaining <= denominator * alpha_numerator


def adjust_complete_family(
    hypotheses: Sequence[Mapping[str, Any]],
    *,
    alpha_numerator: int = ALPHA_NUMERATOR,
    alpha_denominator: int = ALPHA_DENOMINATOR,
) -> dict[str, dict[str, Any]]:
    """Apply exact Holm and Bonferroni adjustment to one declared family."""

    alpha_num, alpha_den = _validate_alpha(alpha_numerator, alpha_denominator)
    require(type(hypotheses) in (list, tuple) and bool(hypotheses),
            "hypotheses must be a nonempty list or tuple")
    require(len(hypotheses) <= MAX_DIRECTIONAL_HYPOTHESES,
            "directional family exceeds the implementation safety limit")
    normalized: list[dict[str, Any]] = []
    identifiers: list[str] = []
    for index, source in enumerate(hypotheses):
        require(type(source) is dict, f"hypothesis {index} must be a plain dict")
        identifier = source.get("hypothesis_id")
        require(type(identifier) is str and bool(identifier),
                f"hypothesis {index} has an invalid ID")
        numerator = _plain_int(source.get("p_numerator"), "p_numerator")
        denominator = _plain_int(source.get("p_denominator"), "p_denominator")
        require(0 <= numerator <= denominator and denominator > 0,
                "p-value fraction lies outside [0,1]")
        row = dict(source)
        row["hypothesis_id"] = identifier
        row["p_fraction"] = Fraction(numerator, denominator)
        normalized.append(row)
        identifiers.append(identifier)
    _validate_identifiers(identifiers, name="hypothesis_ids", minimum=1)

    ordered = sorted(normalized, key=lambda row: (row["p_fraction"], row["hypothesis_id"]))
    family_size = len(ordered)
    holm_open = True
    running_adjusted = Fraction(0, 1)
    decisions: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(ordered, 1):
        remaining = family_size - position + 1
        passes = _exact_reject(
            row["p_numerator"], row["p_denominator"], remaining=remaining,
            alpha_numerator=alpha_num, alpha_denominator=alpha_den,
        )
        holm_reject = bool(holm_open and passes)
        if not passes:
            holm_open = False
        running_adjusted = max(running_adjusted, row["p_fraction"] * remaining)
        bonferroni_adjusted = row["p_fraction"] * family_size
        decisions[row["hypothesis_id"]] = {
            "holm_order": position,
            "holm_remaining": remaining,
            "holm_reject": holm_reject,
            "holm_adjusted": _fraction_payload(running_adjusted),
            "bonferroni_reject": _exact_reject(
                row["p_numerator"], row["p_denominator"], remaining=family_size,
                alpha_numerator=alpha_num, alpha_denominator=alpha_den,
            ),
            "bonferroni_adjusted": _fraction_payload(bonferroni_adjusted),
        }
    return decisions


def _strongly_connected_components(
    models: Sequence[str], edges: Sequence[tuple[str, str]]
) -> list[list[str]]:
    adjacency = {model: [] for model in models}
    for winner, loser in edges:
        require(winner in adjacency and loser in adjacency, "edge model outside family")
        require(winner != loser, "self edge is forbidden")
        adjacency[winner].append(loser)
    for values in adjacency.values():
        values.sort()

    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[list[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for neighbor in adjacency[node]:
            if neighbor not in indices:
                visit(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif neighbor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[neighbor])
        if lowlinks[node] == indices[node]:
            component: list[str] = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == node:
                    break
            components.append(sorted(component))

    for model in models:
        if model not in indices:
            visit(model)
    return sorted(components, key=lambda values: values[0])


def reported_graph_decision(
    models: Sequence[str], edges: Iterable[tuple[str, str]]
) -> dict[str, Any]:
    """Summarize only the graph formed by rejected directional nulls."""

    model_tuple = _validate_identifiers(models, name="models", minimum=2)
    require(len(model_tuple) <= MAX_MODELS,
            "model registry exceeds the implementation safety limit")
    raw_edges = list(edges)
    edge_list: list[tuple[str, str]] = []
    for index, edge in enumerate(raw_edges):
        require(type(edge) is tuple and len(edge) == 2,
                f"edge {index} must be a two-string tuple")
        winner, loser = edge
        require(type(winner) is str and type(loser) is str,
                f"edge {index} must be a two-string tuple")
        require(winner in model_tuple and loser in model_tuple,
                f"edge {index} references a model outside the family")
        require(winner != loser, f"edge {index} is a self edge")
        edge_list.append((winner, loser))
    require(len(set(edge_list)) == len(edge_list), "duplicate graph edge")
    edge_set = set(edge_list)
    require(not any((loser, winner) in edge_set for winner, loser in edge_set),
            "opposite graph edges for one model pair are forbidden")
    edge_list.sort()

    adjacency = {model: set() for model in model_tuple}
    reverse = {model: set() for model in model_tuple}
    for winner, loser in edge_list:
        adjacency[winner].add(loser)
        reverse[loser].add(winner)
    source_set = [model for model in model_tuple if not reverse[model]]
    condorcet = [
        model for model in model_tuple
        if adjacency[model] == set(model_tuple) - {model}
    ]
    require(len(condorcet) <= 1, "multiple direct supported Condorcet winners")

    components = _strongly_connected_components(model_tuple, edge_list)
    cyclic_components = [component for component in components if len(component) > 1]
    if cyclic_components:
        return {
            "reported_graph_acyclic": False,
            "decision_status": "ABSTAIN_CYCLIC_REPORTED_GRAPH",
            "reported_graph_source_set": source_set,
            "cycle_components": cyclic_components,
            "edges": [[winner, loser] for winner, loser in edge_list],
            "reported_graph_position_bounds": None,
            "supported_condorcet_winner": None if not condorcet else condorcet[0],
            "true_best_confidence_set_claim": False,
            "population_rank_confidence_claim": False,
        }

    def reachable(start: str, graph: Mapping[str, set[str]]) -> set[str]:
        seen: set[str] = set()
        pending = list(graph[start])
        while pending:
            node = pending.pop()
            if node in seen:
                continue
            seen.add(node)
            pending.extend(graph[node] - seen)
        return seen

    require(bool(source_set), "acyclic finite graph unexpectedly has no source")
    bounds: dict[str, dict[str, int]] = {}
    for model in model_tuple:
        predecessors = reachable(model, reverse)
        successors = reachable(model, adjacency)
        lower = 1 + len(predecessors)
        upper = len(model_tuple) - len(successors)
        require(1 <= lower <= upper <= len(model_tuple), "invalid graph position bounds")
        bounds[model] = {"lower": lower, "upper": upper}
    status = (
        "SUPPORTED_SINGLETON_REPORTED_GRAPH_SOURCE"
        if len(source_set) == 1
        else "ABSTAIN_NON_SINGLETON_REPORTED_GRAPH_SOURCE"
    )
    return {
        "reported_graph_acyclic": True,
        "decision_status": status,
        "reported_graph_source_set": source_set,
        "cycle_components": [],
        "edges": [[winner, loser] for winner, loser in edge_list],
        "reported_graph_position_bounds": bounds,
        "supported_condorcet_winner": None if not condorcet else condorcet[0],
        "true_best_confidence_set_claim": False,
        "population_rank_confidence_claim": False,
    }


def _minimum_unanimous_nonzero_count(
    family_size: int, *, alpha_numerator: int, alpha_denominator: int
) -> int:
    count = 0
    while (1 << count) * alpha_numerator < alpha_denominator * family_size:
        count += 1
    return count


def resolve_directional_family(
    utilities: np.ndarray,
    *,
    model_ids: Sequence[str],
    scheme_ids: Sequence[str],
    root_ids: Sequence[str],
    regime: str,
    alpha_numerator: int = ALPHA_NUMERATOR,
    alpha_denominator: int = ALPHA_DENOMINATOR,
) -> dict[str, Any]:
    """Resolve a complete declared directional family.

    ``regime`` is mandatory.  Software cannot verify either regime; it records
    the caller's scientific assertion and changes graph authority accordingly.
    """

    models = _validate_identifiers(model_ids, name="model_ids", minimum=2)
    schemes = _validate_identifiers(scheme_ids, name="scheme_ids", minimum=1)
    roots = _validate_identifiers(root_ids, name="root_ids", minimum=1)
    require(len(models) <= MAX_MODELS,
            "model registry exceeds the implementation safety limit")
    require(len(roots) <= MAX_ROOTS,
            "root registry exceeds the implementation safety limit")
    require(type(regime) is str and regime in ALLOWED_REGIMES,
            "regime is not a supported exact contract")
    alpha_num, alpha_den = _validate_alpha(alpha_numerator, alpha_denominator)

    array = _lossless_float64_copy(utilities)
    require(array.shape == (len(schemes), len(roots), len(models)),
            "utilities shape differs from declared scheme/root/model registries")
    family_size = 2 * len(schemes) * math.comb(len(models), 2)
    require(family_size <= MAX_DIRECTIONAL_HYPOTHESES,
            "directional family exceeds the implementation safety limit")
    require(family_size * len(roots) <= MAX_DIRECTIONAL_ROOT_WORK,
            "directional family by root work exceeds the implementation safety limit")

    # Root order has no inferential meaning.  Canonicalizing it makes complete
    # output invariant to a joint registry/array permutation.
    root_order = sorted(range(len(roots)), key=lambda index: roots[index])
    canonical_roots = tuple(roots[index] for index in root_order)
    array = np.array(array[:, root_order, :], dtype=np.float64, order="C", copy=True)

    hypotheses: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    with np.errstate(over="raise", invalid="raise"):
        try:
            for scheme_index, scheme in enumerate(schemes):
                for left in range(len(models)):
                    for right in range(left + 1, len(models)):
                        model_a, model_b = models[left], models[right]
                        difference = array[scheme_index, :, left] - array[scheme_index, :, right]
                        require(bool(np.isfinite(difference).all()),
                                "pairwise subtraction produced nonfinite values")
                        wins = int(np.count_nonzero(difference > 0.0))
                        losses = int(np.count_nonzero(difference < 0.0))
                        ties = int(np.count_nonzero(difference == 0.0))
                        require(wins + losses + ties == len(canonical_roots),
                                "sign partition is incomplete")
                        nonzero = wins + losses
                        forward_num, denominator = exact_upper_tail(wins, nonzero)
                        reverse_num, reverse_den = exact_upper_tail(losses, nonzero)
                        require(denominator == reverse_den,
                                "opposite tails use different denominators")
                        pair_index = len(pair_rows)
                        forward_id = f"H{2 * pair_index:08d}"
                        reverse_id = f"H{2 * pair_index + 1:08d}"
                        hypotheses.extend([
                            {
                                "hypothesis_id": forward_id,
                                "scheme_id": scheme,
                                "winner": model_a,
                                "loser": model_b,
                                "p_numerator": forward_num,
                                "p_denominator": denominator,
                            },
                            {
                                "hypothesis_id": reverse_id,
                                "scheme_id": scheme,
                                "winner": model_b,
                                "loser": model_a,
                                "p_numerator": reverse_num,
                                "p_denominator": denominator,
                            },
                        ])
                        pair_rows.append({
                            "pair_id": f"P{pair_index:08d}",
                            "scheme_id": scheme,
                            "model_a": model_a,
                            "model_b": model_b,
                            "positive_roots": wins,
                            "negative_roots": losses,
                            "tie_roots": ties,
                            "nonzero_roots": nonzero,
                            "observed_positive_fraction_among_nonzero_roots": (
                                None if nonzero == 0 else wins / nonzero
                            ),
                            "theta_hat_conditional_nonzero": (
                                None
                                if regime != REGIME_COMMON_SIGN or nonzero == 0
                                else wins / nonzero
                            ),
                            "tie_fraction": ties / len(canonical_roots),
                            "forward_hypothesis_id": forward_id,
                            "reverse_hypothesis_id": reverse_id,
                            "forward_p_value": _exact_ratio_payload(
                                forward_num, denominator
                            ),
                            "reverse_p_value": _exact_ratio_payload(
                                reverse_num, denominator
                            ),
                        })
        except FloatingPointError as error:
            raise DirectionalFamilyError("pairwise float64 subtraction overflowed") from error

    require(len(hypotheses) == family_size, "constructed directional family is incomplete")
    decisions = adjust_complete_family(
        hypotheses, alpha_numerator=alpha_num, alpha_denominator=alpha_den
    )
    rejected_by_correction: dict[str, dict[str, list[tuple[str, str]]]] = {
        "holm": {scheme: [] for scheme in schemes},
        "bonferroni": {scheme: [] for scheme in schemes},
    }
    final_pairs: list[dict[str, Any]] = []
    for source in pair_rows:
        row = dict(source)
        forward_id = row.pop("forward_hypothesis_id")
        reverse_id = row.pop("reverse_hypothesis_id")
        forward_decision = decisions[forward_id]
        reverse_decision = decisions[reverse_id]
        row["forward_adjustment"] = forward_decision
        row["reverse_adjustment"] = reverse_decision
        for correction, key in (("holm", "holm_reject"),
                                ("bonferroni", "bonferroni_reject")):
            forward_reject = bool(forward_decision[key])
            reverse_reject = bool(reverse_decision[key])
            require(not (forward_reject and reverse_reject),
                    "both directions rejected for one model pair")
            rejected_label = (
                f"{row['model_a']}>{row['model_b']}" if forward_reject
                else f"{row['model_b']}>{row['model_a']}" if reverse_reject
                else None
            )
            row[f"{correction}_rejected_ordered_null"] = (
                None if rejected_label is None else {
                    "winner": (
                        row["model_a"] if forward_reject else row["model_b"]
                    ),
                    "loser": (
                        row["model_b"] if forward_reject else row["model_a"]
                    ),
                }
            )
            row[f"{correction}_supported_direction"] = (
                row[f"{correction}_rejected_ordered_null"]
                if regime == REGIME_COMMON_SIGN else None
            )
            if forward_reject:
                rejected_by_correction[correction][row["scheme_id"]].append(
                    (row["model_a"], row["model_b"])
                )
            if reverse_reject:
                rejected_by_correction[correction][row["scheme_id"]].append(
                    (row["model_b"], row["model_a"])
                )
        final_pairs.append(row)

    scheme_decisions: dict[str, dict[str, Any]] = {}
    for correction in ("holm", "bonferroni"):
        scheme_decisions[correction] = {}
        for scheme in schemes:
            rejected = sorted(rejected_by_correction[correction][scheme])
            if regime == REGIME_COMMON_SIGN:
                decision = reported_graph_decision(models, rejected)
                decision["graph_authority"] = "AUTHORIZED_FOR_REPORTED_GRAPH_ONLY"
            else:
                decision = {
                    "decision_status": "GRAPH_WITHHELD_HETEROGENEOUS_INTERSECTION_REGIME",
                    "graph_authority": "NONE",
                    "rejected_rootwise_intersection_null_labels": [
                        [winner, loser] for winner, loser in rejected
                    ],
                    "direction_claim_authorized": False,
                    "true_best_confidence_set_claim": False,
                    "population_rank_confidence_claim": False,
                }
            scheme_decisions[correction][scheme] = decision

    minimum_nonzero = _minimum_unanimous_nonzero_count(
        family_size, alpha_numerator=alpha_num, alpha_denominator=alpha_den
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "protocol_id": PROTOCOL_ID,
        "regime": regime,
        "software_verified_root_assumptions": False,
        "estimand": (
            "COMMON P(U_A>U_B | U_A!=U_B) ACROSS INDEPENDENT ROOTS"
            if regime == REGIME_COMMON_SIGN
            else "ROOTWISE CONDITIONAL SIGN-PROBABILITY INTERSECTION NULL"
        ),
        "mean_utility_estimand": False,
        "model_ids": list(models),
        "model_registry_sha256": _registry_sha256("V3_MODEL_REGISTRY", models),
        "scheme_ids": list(schemes),
        "scheme_registry_sha256": _registry_sha256("V3_SCHEME_REGISTRY", schemes),
        "root_count": len(canonical_roots),
        "root_registry_sha256": _registry_sha256("V3_ROOT_REGISTRY", canonical_roots),
        "utility_array_binding": "REQUIRED_UPSTREAM_NOT_EMITTED_BY_UNLABELLED_CORE",
        "family": {
            "unordered_pair_count": len(final_pairs),
            "directional_hypothesis_count": family_size,
            "alpha_numerator": alpha_num,
            "alpha_denominator": alpha_den,
            "primary_adjustment": "HOLM_STEP_DOWN_EXACT_RATIONAL",
            "secondary_adjustment": "BONFERRONI_EXACT_RATIONAL",
            "family_pruned_after_observation": False,
        },
        "structural_resolvability": {
            "minimum_unanimous_nonzero_roots_for_first_holm_rejection": minimum_nonzero,
            "maximum_observed_nonzero_roots": max(row["nonzero_roots"] for row in final_pairs),
            "first_holm_rejection_arithmetically_possible": any(
                row["nonzero_roots"] >= minimum_nonzero for row in final_pairs
            ),
        },
        "pairwise": final_pairs,
        "scheme_decisions": scheme_decisions,
        "interpretation_boundaries": {
            "graph_output_authorized": regime == REGIME_COMMON_SIGN,
            "reported_graph_only": regime == REGIME_COMMON_SIGN,
            "rootwise_intersection_tests_only": regime == REGIME_HETEROGENEOUS,
            "true_best_confidence_set": False,
            "population_rank_confidence_intervals": False,
            "population_transport": False,
            "root_independence_must_be_certified_upstream": True,
            "descendant_cells_tasks_folds_are_not_additional_roots": True,
        },
    }


def canonical_resolution_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
