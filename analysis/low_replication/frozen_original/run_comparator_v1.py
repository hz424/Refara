#!/usr/bin/env python3
"""Run exactly one frozen V3 comparator V1 cell and persist counts only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import stat
import sys
import types
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
RUNNER_PATH = Path(__file__).resolve()
AMENDMENT_PATH = HERE / "COMPARATOR_EXECUTABLE_AMENDMENT_V1.md"
SPEC_PATH = HERE / "COMPARATOR_EXECUTABLE_SPEC_V1.json"
LEDGER_PATH = HERE / "COMPARATOR_CELL_LEDGER_V1.tsv"
NUMERICS_PATH = HERE / "comparator_numerics_v1.py"
STATIC_VALIDATOR_PATH = HERE / "validate_comparator_executable_amendment_v1.py"
FROZEN_TEST_PATH = HERE / "tests" / "test_comparator_executable_amendment_v1.py"
V3_ROOT = HERE.parents[1]
CORE_PATH = (
    V3_ROOT
    / "01_general_directional_family_method_v3"
    / "general_directional_family_v3.py"
)
RUNTIME_LOCK_PATH = (
    V3_ROOT / "01_general_directional_family_method_v3" / "RUNTIME_LOCK_V3.json"
)

PINNED_AMENDMENT_SHA256 = (
    "13f37d8c0ef20088ce6861cf1207f6a2ec9b7f7c0b2a9010cece406726798ac7"
)
PINNED_SPEC_SHA256 = (
    "083c38a23329afb6ecd31bfb088684214272d6fa9bdb18c86813a0b404657373"
)
PINNED_LEDGER_SHA256 = (
    "ffa08beb00fd9ff36ee4bc0a48b1ff75d589388e72d043dbf8040992c13590f0"
)
PINNED_NUMERICS_SHA256 = (
    "a7b7e4c4034d0d96ee50db32ece37f7b00571d9d99bfdddfa57b0b3870541cd8"
)
PINNED_STATIC_VALIDATOR_SHA256 = (
    "d5634d0442be7859ac96ecfa0316203661e576fdcf80065931e79c563d951464"
)
PINNED_FROZEN_TEST_SHA256 = (
    "ac69924c7cdcc1abe4e294b1f40203198fe5cc3d2953ac2b47a24119a0e729ea"
)
PINNED_CORE_SHA256 = (
    "4da8f63a1f0b14d2101074becea0f9d719878e971b61a89dda5815ab97bd9b0e"
)
PINNED_RUNTIME_LOCK_SHA256 = (
    "4de9c8f39a57bfc92c98d0abf734e55348aa1c117da35ce25a36dfd94aeeddb6"
)

COMMON_SIGN_REGIME = "INDEPENDENT_ROOTS_COMMON_SIGN_PROBABILITY"
MODEL_IDS = ("M01", "M02", "M03", "M04")
SCHEME_IDS = ("S01",)
STREAM_LABELS = ("DATA", "BOOT_ROOT", "BOOT_TASK", "BOOT_CELL")
PRODUCTION_REPLICATES = 2_500
BOOTSTRAP_RESAMPLES = 499
SHARD_PREFIX = "COMPARATOR_CELL_"
SHARD_SUFFIX = "_COUNTS_V1.json"

SIGN_PROCEDURE = "EXACT_ROOT_SIGN_HOLM"
ROOT_BOOTSTRAP_PROCEDURE = "ROOT_NONPARAMETRIC_BOOTSTRAP_MEAN_CONTRAST_HOLM"
ROOT_CR1_PROCEDURE = "ROOT_CR1_T_MEAN_CONTRAST_HOLM"
TASK_NEGATIVE_PROCEDURE = "TASK_NEGATIVE_CONTROL_BOOTSTRAP_MEAN_CONTRAST_HOLM"
CELL_NEGATIVE_PROCEDURE = "CELL_NEGATIVE_CONTROL_BOOTSTRAP_MEAN_CONTRAST_HOLM"
POINT_PROCEDURE = "POINT_LEADERBOARD_NO_UNCERTAINTY"
INFERENTIAL_PROCEDURES = (
    SIGN_PROCEDURE,
    ROOT_BOOTSTRAP_PROCEDURE,
    ROOT_CR1_PROCEDURE,
    TASK_NEGATIVE_PROCEDURE,
    CELL_NEGATIVE_PROCEDURE,
)

PROCEDURE_INTERPRETATION = {
    SIGN_PROCEDURE: {
        "inferential_object": "ROOT_SIGN",
        "error_measure_class": "SIGN_FWER_AND_SIGN_DIRECTIONAL_RECOVERY",
        "graph_authority": "V3_REPORTED_COMMON_SIGN_GRAPH_DESCRIPTOR_ONLY",
        "procedure_authority": (
            "VALID_SIGN_INFERENCE_CHARACTERIZATION_NOT_A_QUALIFICATION_RECEIPT"
        ),
    },
    ROOT_BOOTSTRAP_PROCEDURE: {
        "inferential_object": "ROOT_MEAN",
        "error_measure_class": (
            "ROOT_MEAN_COMPARATOR_ERROR_NOT_SIGN_FWER_AND_MEAN_DIRECTIONAL_RECOVERY"
        ),
        "graph_authority": "DESCRIPTIVE_MEAN_REJECTION_GRAPH_NOT_A_V3_GRAPH",
        "procedure_authority": "DESCRIPTIVE_ROOT_MEAN_COMPARATOR",
    },
    ROOT_CR1_PROCEDURE: {
        "inferential_object": "ROOT_MEAN",
        "error_measure_class": (
            "ROOT_MEAN_COMPARATOR_ERROR_NOT_SIGN_FWER_AND_MEAN_DIRECTIONAL_RECOVERY"
        ),
        "graph_authority": "DESCRIPTIVE_MEAN_REJECTION_GRAPH_NOT_A_V3_GRAPH",
        "procedure_authority": "DESCRIPTIVE_ROOT_MEAN_COMPARATOR",
    },
    TASK_NEGATIVE_PROCEDURE: {
        "inferential_object": "INVALID_DESCENDANT_TASK_PSEUDOREPLICATION",
        "error_measure_class": (
            "INVALID_DESCENDANT_NEGATIVE_CONTROL_REJECTION_BEHAVIOR_NOT_VALID_FWER"
        ),
        "graph_authority": "INVALID_PSEUDOREPLICATION_NEGATIVE_CONTROL_GRAPH",
        "procedure_authority": "NEGATIVE_CONTROL_NO_VALID_INFERENCE",
    },
    CELL_NEGATIVE_PROCEDURE: {
        "inferential_object": "INVALID_DESCENDANT_CELL_PSEUDOREPLICATION",
        "error_measure_class": (
            "INVALID_DESCENDANT_NEGATIVE_CONTROL_REJECTION_BEHAVIOR_NOT_VALID_FWER"
        ),
        "graph_authority": "INVALID_PSEUDOREPLICATION_NEGATIVE_CONTROL_GRAPH",
        "procedure_authority": "NEGATIVE_CONTROL_NO_VALID_INFERENCE",
    },
}

FAILURE_REASON_KEYS = (
    "nonfinite_generated_or_derived_value",
    "numerical_algorithm_failure",
    "opposite_rejected_directions",
    "method_or_artifact_contract_failure",
    "unexpected_exception",
)


class ComparatorRunError(RuntimeError):
    """Raised when the runner or frozen execution contract is violated."""


class NonfiniteComparatorValue(ComparatorRunError):
    """Raised when an outer replicate generates or derives a nonfinite value."""


class OppositeRejectedDirections(ComparatorRunError):
    """Raised if both directions for one model pair survive correction."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ComparatorRunError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _read_pinned_bytes(path: Path, expected_sha256: str) -> bytes:
    metadata = os.lstat(path)
    require(
        stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1,
        f"{path.name} is not one singly linked regular file",
    )
    raw = path.read_bytes()
    require(
        sha256_bytes(raw) == expected_sha256,
        f"{path.name} differs from its frozen byte hash",
    )
    return raw


def _exec_pinned_module(path: Path, expected_sha256: str, name: str) -> Any:
    raw = _read_pinned_bytes(path, expected_sha256)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    except Exception:
        sys.modules.pop(name, None)
        raise
    require(
        sha256_path(path) == expected_sha256,
        f"{path.name} changed during verified execution",
    )
    return module


def _runtime_identity() -> dict[str, str]:
    executable = Path(sys.executable).resolve()
    numpy_init = Path(np.__file__).resolve()
    return {
        "implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_executable_sha256": sha256_path(executable),
        "numpy_version": np.__version__,
        "numpy_init_sha256": sha256_path(numpy_init),
    }


def validate_static_artifacts_and_runtime() -> dict[str, Any]:
    """Validate the exact frozen bundle and locked local runtime."""

    frozen_paths = {
        "executable_amendment": (AMENDMENT_PATH, PINNED_AMENDMENT_SHA256),
        "executable_spec": (SPEC_PATH, PINNED_SPEC_SHA256),
        "cell_ledger": (LEDGER_PATH, PINNED_LEDGER_SHA256),
        "comparator_numerics": (NUMERICS_PATH, PINNED_NUMERICS_SHA256),
        "static_validator": (
            STATIC_VALIDATOR_PATH,
            PINNED_STATIC_VALIDATOR_SHA256,
        ),
        "frozen_tests": (FROZEN_TEST_PATH, PINNED_FROZEN_TEST_SHA256),
        "method_core": (CORE_PATH, PINNED_CORE_SHA256),
        "runtime_lock": (RUNTIME_LOCK_PATH, PINNED_RUNTIME_LOCK_SHA256),
    }
    frozen_bytes = {
        name: _read_pinned_bytes(path, expected)
        for name, (path, expected) in frozen_paths.items()
    }
    validator = _exec_pinned_module(
        STATIC_VALIDATOR_PATH,
        PINNED_STATIC_VALIDATOR_SHA256,
        "comparator_static_validator_runner_bound_v1",
    )
    static_receipt = validator.validate_paths()
    require(
        static_receipt["state"]
        == "PASS_COMPARATOR_EXECUTABLE_AMENDMENT_VALIDATION"
        and static_receipt["qualification_authority"] == "NONE"
        and static_receipt["empirical_execution_authority"] == "NONE"
        and static_receipt["outcome_access_authority"] == "NONE",
        "static validator returned the wrong state or authority",
    )
    spec = validator.parse_strict_json_bytes(
        frozen_bytes["executable_spec"], "runner-bound comparator spec"
    )
    rows = validator.parse_ledger_bytes(frozen_bytes["cell_ledger"])

    lock = json.loads(frozen_bytes["runtime_lock"].decode("utf-8"))
    identity = _runtime_identity()
    runtime = lock["runtime"]
    dependencies = lock["dependencies"]
    require(identity["implementation"] == runtime["implementation"],
            "Python implementation differs from runtime lock")
    require(identity["python_version"] == runtime["python_version"],
            "Python version differs from runtime lock")
    require(Path(sys.executable).resolve() == Path(runtime["executable"]),
            "Python executable differs from runtime lock")
    require(identity["python_executable_sha256"] == runtime["executable_sha256"],
            "Python executable hash differs from runtime lock")
    require(identity["numpy_version"] == dependencies["numpy_version"],
            "NumPy version differs from runtime lock")
    require(Path(np.__file__).resolve() == Path(dependencies["numpy_init_path"]),
            "NumPy path differs from runtime lock")
    require(identity["numpy_init_sha256"] == dependencies["numpy_init_sha256"],
            "NumPy hash differs from runtime lock")
    require(
        lock["execution_contract"]["network_required"] is False
        and lock["execution_contract"][
            "empirical_or_outcome_inputs_authorized"
        ] is False
        and lock["authority"]["outcome_access_authority"] == "NONE",
        "runtime lock grants forbidden empirical or outcome authority",
    )
    require(spec["scope"]["outcome_access_authority"] == "NONE",
            "comparator spec grants outcome authority")
    require(spec["scope"]["network_required"] is False,
            "comparator spec requires network access")
    for name, (path, expected) in frozen_paths.items():
        require(sha256_path(path) == expected,
                f"{name} changed during verified validation")
    return {
        "spec": spec,
        "rows": rows,
        "runtime_identity": identity,
        "static_receipt": static_receipt,
        "artifact_sha256": {
            name: expected for name, (_path, expected) in frozen_paths.items()
        },
    }


def direction_ids() -> tuple[str, ...]:
    return tuple(
        f"S01|{winner}>{loser}"
        for winner in MODEL_IDS
        for loser in MODEL_IDS
        if winner != loser
    )


def shard_filename(cell_id: str) -> str:
    require(
        cell_id in {f"CMP{index:03d}" for index in range(1, 17)},
        "invalid comparator cell ID",
    )
    return f"{SHARD_PREFIX}{cell_id}{SHARD_SUFFIX}"


def _outer_seed(row: Mapping[str, str], outer_index: int, label: str) -> int:
    require(label in STREAM_LABELS, "unknown comparator stream label")
    require(type(outer_index) is int and 0 <= outer_index <= 2_499,
            "outer index is outside the frozen range")
    preimage = (
        row["seed_namespace"].encode("utf-8")
        + b"\0"
        + row["cell_id"].encode("utf-8")
        + b"\0OUTER\0"
        + f"{outer_index:06d}".encode("ascii")
        + b"\0"
        + label.encode("utf-8")
    )
    return int.from_bytes(hashlib.sha256(preimage).digest(), "big", signed=False)


def _finite(array: np.ndarray, name: str) -> None:
    if not bool(np.isfinite(array).all()):
        raise NonfiniteComparatorValue(f"{name} contains a nonfinite value")


def _analysis_rows(
    row: Mapping[str, str],
    law: Mapping[str, Any],
    data_rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root_count = int(row["root_count"])
    means = np.asarray(law["mean_vector_M01_to_M04"], dtype=np.float64)
    components = law["variance_components"]
    root_effect = data_rng.standard_normal((root_count, 4))
    task_effect = data_rng.standard_normal((root_count, 8, 4))
    cell_effect = data_rng.standard_normal((root_count, 8, 20, 4))
    atomic = (
        means[None, None, None, :]
        + math.sqrt(components["v_root"])
        * root_effect[:, None, None, :]
        + math.sqrt(components["v_task"])
        * task_effect[:, :, None, :]
        + math.sqrt(components["v_cell"]) * cell_effect
    )
    atomic = np.asarray(atomic, dtype=np.float64, order="C")
    _finite(atomic, "generated atomic utilities")

    support = law["observed_cell_counts_by_task_T01_to_T08"]
    require(
        type(support) is list
        and len(support) == 8
        and all(type(value) is int and 1 <= value <= 20 for value in support),
        "law support vector differs from the executable spec",
    )
    task_rows_3d = np.empty((root_count, 8, 4), dtype=np.float64)
    cell_blocks: list[np.ndarray] = []
    observed_count = root_count * sum(support)
    for root_index in range(root_count):
        for task_index, retained_count in enumerate(support):
            retained = atomic[root_index, task_index, :retained_count, :]
            task_rows_3d[root_index, task_index, :] = np.mean(
                retained, axis=0, dtype=np.float64
            )
            scale = observed_count / (root_count * 8 * retained_count)
            cell_blocks.append(
                np.multiply(retained, np.float64(scale), dtype=np.float64)
            )
    root_rows = np.empty((root_count, 4), dtype=np.float64)
    for root_index in range(root_count):
        root_rows[root_index, :] = np.mean(
            task_rows_3d[root_index, :, :], axis=0, dtype=np.float64
        )
    point_means = np.mean(root_rows, axis=0, dtype=np.float64)
    task_rows = np.asarray(
        task_rows_3d.reshape(root_count * 8, 4),
        dtype=np.float64,
        order="C",
    )
    cell_rows = np.asarray(
        np.concatenate(cell_blocks, axis=0), dtype=np.float64, order="C"
    )
    require(cell_rows.shape == (observed_count, 4),
            "scaled cell-row geometry differs")
    for name, value in (
        ("task rows", task_rows),
        ("root rows", root_rows),
        ("scaled cell rows", cell_rows),
        ("point means", point_means),
    ):
        _finite(value, name)
    task_point = np.mean(task_rows, axis=0, dtype=np.float64)
    cell_point = np.mean(cell_rows, axis=0, dtype=np.float64)
    tolerance = 64.0 * np.finfo(np.float64).eps * max(
        1.0, float(np.max(np.abs(point_means)))
    )
    require(
        bool(np.all(np.abs(task_point - point_means) <= tolerance))
        and bool(np.all(np.abs(cell_point - point_means) <= tolerance)),
        "analysis-row point estimands are not equal-root/equal-task means",
    )
    return root_rows, task_rows, cell_rows, point_means


def _bootstrap_rejections(
    rows: np.ndarray,
    point_means: np.ndarray,
    rng: np.random.Generator,
    numerics: Any,
) -> tuple[str, ...]:
    row_count = rows.shape[0]
    observed_column_means = np.mean(rows, axis=0, dtype=np.float64)
    centered = np.subtract(rows, observed_column_means[None, :])
    indices = rng.integers(
        0,
        row_count,
        size=(BOOTSTRAP_RESAMPLES, row_count),
        dtype=np.int64,
        endpoint=False,
    )
    resampled_means = np.empty((BOOTSTRAP_RESAMPLES, 4), dtype=np.float64)
    for model_index in range(4):
        selected = np.take(centered[:, model_index], indices)
        resampled_means[:, model_index] = np.mean(
            selected, axis=1, dtype=np.float64
        )
        del selected
    _finite(resampled_means, "bootstrap resampled means")
    numerators: dict[str, int] = {}
    for winner_index, winner in enumerate(MODEL_IDS):
        for loser_index, loser in enumerate(MODEL_IDS):
            if winner_index == loser_index:
                continue
            observed_contrast = float(
                point_means[winner_index] - point_means[loser_index]
            )
            resampled_contrast = np.subtract(
                resampled_means[:, winner_index],
                resampled_means[:, loser_index],
            )
            if not math.isfinite(observed_contrast):
                raise NonfiniteComparatorValue(
                    "observed bootstrap contrast is nonfinite"
                )
            numerators[f"S01|{winner}>{loser}"] = 1 + int(
                np.count_nonzero(resampled_contrast >= observed_contrast)
            )
    require(set(numerators) == set(direction_ids()),
            "bootstrap family is not the complete H=12 family")
    return tuple(numerics.holm_common_denominator(numerators, 500))


def _cr1_rejections(
    root_rows: np.ndarray, numerics: Any
) -> tuple[tuple[str, ...], int]:
    p_values: dict[str, float] = {}
    zero_standard_errors = 0
    for winner_index, winner in enumerate(MODEL_IDS):
        for loser_index, loser in enumerate(MODEL_IDS):
            if winner_index == loser_index:
                continue
            contrasts = np.subtract(
                root_rows[:, winner_index], root_rows[:, loser_index]
            )
            result = numerics.one_sample_root_t_positive(contrasts)
            identifier = f"S01|{winner}>{loser}"
            p_values[identifier] = result.one_sided_p_value
            zero_standard_errors += int(result.zero_standard_error_fail_closed)
    require(set(p_values) == set(direction_ids()),
            "CR1 family is not the complete H=12 family")
    return tuple(numerics.holm_float64(p_values)), zero_standard_errors


def _ensure_no_opposites(rejected: Sequence[str]) -> None:
    rejected_set = set(rejected)
    require(len(rejected_set) == len(rejected), "duplicate rejected direction")
    require(rejected_set <= set(direction_ids()),
            "rejected direction lies outside H=12")
    for identifier in rejected_set:
        pair = identifier.split("|", 1)[1]
        winner, loser = pair.split(">", 1)
        if f"S01|{loser}>{winner}" in rejected_set:
            raise OppositeRejectedDirections(
                f"both {winner}>{loser} and {loser}>{winner} were rejected"
            )


def _graph_summary(rejected: Sequence[str], core: Any) -> dict[str, Any]:
    _ensure_no_opposites(rejected)
    edges = []
    for identifier in rejected:
        pair = identifier.split("|", 1)[1]
        winner, loser = pair.split(">", 1)
        edges.append((winner, loser))
    graph = core.reported_graph_decision(MODEL_IDS, edges)
    condorcet = graph["supported_condorcet_winner"]
    return {
        "rejected": tuple(rejected),
        "graph_cycle": not graph["reported_graph_acyclic"],
        "source_set_size": len(graph["reported_graph_source_set"]),
        "direct_condorcet": condorcet,
    }


def _sign_summary(root_rows: np.ndarray, core: Any) -> dict[str, Any]:
    root_count = root_rows.shape[0]
    result = core.resolve_directional_family(
        root_rows[None, :, :],
        model_ids=MODEL_IDS,
        scheme_ids=SCHEME_IDS,
        root_ids=tuple(f"R{index:03d}" for index in range(1, root_count + 1)),
        regime=COMMON_SIGN_REGIME,
    )
    require(result["family"]["directional_hypothesis_count"] == 12,
            "V3 sign resolver returned the wrong family size")
    require(result["mean_utility_estimand"] is False,
            "V3 sign resolver returned a mean estimand")
    rejected: list[str] = []
    unordered_ties = 0
    unordered_non_ties = 0
    for pair in result["pairwise"]:
        unordered_ties += int(pair["tie_roots"])
        unordered_non_ties += int(pair["nonzero_roots"])
        decision = pair["holm_rejected_ordered_null"]
        if decision is not None:
            rejected.append(
                f"{pair['scheme_id']}|{decision['winner']}>{decision['loser']}"
            )
    summary = _graph_summary(rejected, core)
    frozen_graph = result["scheme_decisions"]["holm"]["S01"]
    require(
        frozen_graph["graph_authority"] == "AUTHORIZED_FOR_REPORTED_GRAPH_ONLY"
        and summary["graph_cycle"]
        == (not frozen_graph["reported_graph_acyclic"])
        and summary["source_set_size"]
        == len(frozen_graph["reported_graph_source_set"])
        and summary["direct_condorcet"]
        == frozen_graph["supported_condorcet_winner"],
        "V3 sign graph summary differs from the frozen resolver",
    )
    summary["ordered_root_ties"] = 2 * unordered_ties
    summary["ordered_root_non_ties"] = 2 * unordered_non_ties
    require(
        summary["ordered_root_ties"] + summary["ordered_root_non_ties"]
        == 12 * root_count,
        "V3 ordered root tie accounting differs",
    )
    return summary


def _empty_procedure_counts(*, include_v3: bool) -> dict[str, Any]:
    counts: dict[str, Any] = {
        "outer_replicates_no_rejections": 0,
        "outer_replicates_any_null_direction_rejected": 0,
        "null_direction_rejections_total": 0,
        "outer_replicates_any_true_direction_rejected": 0,
        "true_direction_rejections_total": 0,
        "outer_replicates_all_true_directions_rejected_nonvacuous": 0,
        "outer_replicates_graph_cycle": 0,
        "outer_replicates_any_direct_condorcet": 0,
        "outer_replicates_target_direct_condorcet": 0,
        "outer_replicates_nontarget_direct_condorcet": 0,
        "reported_graph_source_set_size_counts_0_to_4": {
            str(value): 0 for value in range(5)
        },
        "zero_standard_error_ordered_tests_total": 0,
        "outer_replicates_any_zero_standard_error": 0,
    }
    if include_v3:
        counts["ordered_root_non_ties_total"] = 0
        counts["ordered_root_ties_total"] = 0
    return counts


def _update_procedure_counts(
    counts: dict[str, Any],
    summary: Mapping[str, Any],
    *,
    null_directions: set[str],
    true_directions: set[str],
    target_model: str | None,
    zero_standard_errors: int,
) -> None:
    rejected = set(summary["rejected"])
    null_rejected = rejected & null_directions
    true_rejected = rejected & true_directions
    counts["outer_replicates_no_rejections"] += int(not rejected)
    counts["outer_replicates_any_null_direction_rejected"] += int(
        bool(null_rejected)
    )
    counts["null_direction_rejections_total"] += len(null_rejected)
    counts["outer_replicates_any_true_direction_rejected"] += int(
        bool(true_rejected)
    )
    counts["true_direction_rejections_total"] += len(true_rejected)
    counts["outer_replicates_all_true_directions_rejected_nonvacuous"] += int(
        bool(true_directions) and true_directions <= rejected
    )
    counts["outer_replicates_graph_cycle"] += int(summary["graph_cycle"])
    condorcet = summary["direct_condorcet"]
    counts["outer_replicates_any_direct_condorcet"] += int(condorcet is not None)
    counts["outer_replicates_target_direct_condorcet"] += int(
        target_model is not None and condorcet == target_model
    )
    counts["outer_replicates_nontarget_direct_condorcet"] += int(
        condorcet is not None and (target_model is None or condorcet != target_model)
    )
    counts["reported_graph_source_set_size_counts_0_to_4"][
        str(summary["source_set_size"])
    ] += 1
    counts["zero_standard_error_ordered_tests_total"] += zero_standard_errors
    counts["outer_replicates_any_zero_standard_error"] += int(
        zero_standard_errors > 0
    )
    if "ordered_root_non_ties_total" in counts:
        counts["ordered_root_non_ties_total"] += summary[
            "ordered_root_non_ties"
        ]
        counts["ordered_root_ties_total"] += summary["ordered_root_ties"]


def _failure_reason(error: BaseException, numerics: Any, core: Any) -> str:
    if isinstance(error, NonfiniteComparatorValue):
        return "nonfinite_generated_or_derived_value"
    if isinstance(error, OppositeRejectedDirections):
        return "opposite_rejected_directions"
    if isinstance(error, (numerics.ComparatorNumericalError, FloatingPointError)):
        return "numerical_algorithm_failure"
    if isinstance(error, (ComparatorRunError, core.DirectionalFamilyError)):
        return "method_or_artifact_contract_failure"
    return "unexpected_exception"


def run_cell(
    row: Mapping[str, str],
    spec: Mapping[str, Any],
    numerics: Any,
    core: Any,
    *,
    test_outer_replicates: int | None = None,
) -> dict[str, Any]:
    """Execute one cell; failed outer replicates are counted and never replaced."""

    if test_outer_replicates is None:
        attempted = PRODUCTION_REPLICATES
    else:
        require(
            type(test_outer_replicates) is int
            and 1 <= test_outer_replicates <= 4,
            "explicit test override must be a plain integer from 1 to 4",
        )
        attempted = test_outer_replicates
    require(int(row["outer_attempted_replicates"]) == PRODUCTION_REPLICATES,
            "ledger production replicate count differs")
    require(int(row["bootstrap_resamples"]) == BOOTSTRAP_RESAMPLES,
            "ledger bootstrap count differs")
    law_id = row["law_id"]
    law = spec["hierarchical_gaussian_law"]["laws"][law_id]
    if law_id in ("C01", "C02"):
        null_directions = set(direction_ids())
        true_directions: set[str] = set()
        target_model = None
    else:
        scoring = spec["truth_scoring"]["C03_and_C04"]
        null_directions = set(scoring["null_ordered_hypotheses"])
        true_directions = set(scoring["true_alternative_ordered_hypotheses"])
        target_model = scoring["target_model"]
    require(
        null_directions | true_directions == set(direction_ids())
        and not (null_directions & true_directions),
        "law truth registry does not partition H=12",
    )

    procedures = {
        procedure: {
            **PROCEDURE_INTERPRETATION[procedure],
            "counts": _empty_procedure_counts(include_v3=procedure == SIGN_PROCEDURE),
        }
        for procedure in INFERENTIAL_PROCEDURES
    }
    point_counts = {
        "selection_counts_M01_to_M04": {model: 0 for model in MODEL_IDS},
        "target_model_selections": 0,
        "exact_maximum_tie_break_events": 0,
    }
    failures = {key: 0 for key in FAILURE_REASON_KEYS}
    successful = 0
    failed = 0

    for outer_index in range(attempted):
        try:
            streams = {
                label: np.random.Generator(
                    np.random.PCG64(_outer_seed(row, outer_index, label))
                )
                for label in STREAM_LABELS
            }
            root_rows, task_rows, cell_rows, point_means = _analysis_rows(
                row, law, streams["DATA"]
            )
            sign = _sign_summary(root_rows, core)
            root_bootstrap = _graph_summary(
                _bootstrap_rejections(
                    root_rows, point_means, streams["BOOT_ROOT"], numerics
                ),
                core,
            )
            task_negative = _graph_summary(
                _bootstrap_rejections(
                    task_rows, point_means, streams["BOOT_TASK"], numerics
                ),
                core,
            )
            cell_negative = _graph_summary(
                _bootstrap_rejections(
                    cell_rows, point_means, streams["BOOT_CELL"], numerics
                ),
                core,
            )
            cr1_rejected, zero_standard_errors = _cr1_rejections(
                root_rows, numerics
            )
            cr1 = _graph_summary(cr1_rejected, core)
            maximum = float(np.max(point_means))
            maximum_indices = np.flatnonzero(point_means == maximum)
            require(maximum_indices.size >= 1, "point maximum is missing")
            selected_model = MODEL_IDS[int(maximum_indices[0])]
            replicate_results = {
                SIGN_PROCEDURE: (sign, 0),
                ROOT_BOOTSTRAP_PROCEDURE: (root_bootstrap, 0),
                ROOT_CR1_PROCEDURE: (cr1, zero_standard_errors),
                TASK_NEGATIVE_PROCEDURE: (task_negative, 0),
                CELL_NEGATIVE_PROCEDURE: (cell_negative, 0),
            }
        except Exception as error:  # one failed attempt, counted once, never replaced
            failed += 1
            failures[_failure_reason(error, numerics, core)] += 1
            continue

        successful += 1
        for procedure, (summary, zero_standard_errors) in replicate_results.items():
            _update_procedure_counts(
                procedures[procedure]["counts"],
                summary,
                null_directions=null_directions,
                true_directions=true_directions,
                target_model=target_model,
                zero_standard_errors=zero_standard_errors,
            )
        point_counts["selection_counts_M01_to_M04"][selected_model] += 1
        point_counts["target_model_selections"] += int(
            target_model is not None and selected_model == target_model
        )
        point_counts["exact_maximum_tie_break_events"] += int(
            maximum_indices.size > 1
        )

    require(successful + failed == attempted, "outer replicate accounting differs")
    require(sum(failures.values()) == failed, "failure-reason accounting differs")
    return {
        "cell_id": row["cell_id"],
        "array_index": int(row["array_index"]),
        "family_id": row["family_id"],
        "root_count": int(row["root_count"]),
        "law_id": law_id,
        "cell_master_seed_sha256": row["seed_sha256_hex"],
        "production_outer_replicates": PRODUCTION_REPLICATES,
        "test_outer_replicate_override_used": test_outer_replicates is not None,
        "outer_counts": {
            "attempted_replicates": attempted,
            "successful_replicates": successful,
            "failed_replicates": failed,
        },
        "failure_reason_counts": failures,
        "inferential_procedures": procedures,
        "point_procedure": {
            "procedure_id": POINT_PROCEDURE,
            "inferential_object": "POINT_SELECTION_ONLY",
            "error_measure_class": "POINT_SELECTION_NO_FWER_NO_INFERENCE",
            "graph_authority": "NO_GRAPH",
            "procedure_authority": "NO_TEST_NO_INTERVAL_NO_FWER_NO_GRAPH",
            "counts": point_counts,
        },
    }


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_create_only(path: Path, payload: bytes) -> None:
    destination = path.absolute()
    parent_metadata = os.lstat(destination.parent)
    require(stat.S_ISDIR(parent_metadata.st_mode),
            "output parent must be an existing real directory")
    try:
        os.lstat(destination)
    except FileNotFoundError:
        pass
    else:
        raise ComparatorRunError("output already exists or is a symlink")
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def build_payload(
    row: Mapping[str, str],
    artifacts: Mapping[str, Any],
    numerics: Any,
    core: Any,
    *,
    test_outer_replicates: int | None = None,
    execution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if test_outer_replicates is None:
        require(type(execution) is dict, "production scheduler provenance is missing")
        execution_mode = "SLURM_ARRAY_PRODUCTION"
    else:
        require(execution is None, "test override cannot inject production provenance")
        execution_mode = "EXPLICIT_TEST_OVERRIDE_NO_PRODUCTION_AUTHORITY"
        execution = {
            "mode": execution_mode,
            "slurm_job_id": None,
            "slurm_array_job_id": None,
            "slurm_array_task_id": None,
            "slurm_restart_count": None,
        }
    cell = run_cell(
        row,
        artifacts["spec"],
        numerics,
        core,
        test_outer_replicates=test_outer_replicates,
    )
    hashes = dict(artifacts["artifact_sha256"])
    hashes["runner"] = sha256_path(RUNNER_PATH)
    return {
        "schema_version": "1.0.0",
        "artifact_type": "V3_COMPARATOR_CELL_AGGREGATE_INTEGER_COUNTS_V1",
        "spec_id": "V3_COMPARATOR_EXECUTABLE_SPEC_V1",
        "method_id": "GENERAL_DIRECTIONAL_FAMILY_V3",
        "source_free": True,
        "aggregate_integer_counts_only": True,
        "raw_synthetic_arrays_persisted": False,
        "bootstrap_indices_persisted": False,
        "p_value_arrays_persisted": False,
        "replicate_level_rows_persisted": False,
        "network_access_authority": "NONE",
        "qualification_authority": "NONE",
        "empirical_execution_authority": "NONE",
        "outcome_access_authority": "NONE",
        "claim_authority": "DESCRIPTIVE_COMPARATOR_CHARACTERIZATION_ONLY",
        "execution_mode": execution_mode,
        "execution": dict(execution),
        "artifact_sha256": hashes,
        "runtime_identity": artifacts["runtime_identity"],
        "cell": cell,
    }


def _production_execution(row: Mapping[str, str]) -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    restart_count = os.environ.get("SLURM_RESTART_COUNT", "0")
    require(
        type(job_id) is str
        and job_id.isascii()
        and job_id.isdigit()
        and type(array_job_id) is str
        and array_job_id.isascii()
        and array_job_id.isdigit()
        and array_task_id == row["array_index"]
        and restart_count == "0",
        "production Slurm provenance differs from the array cell",
    )
    return {
        "mode": "SLURM_ARRAY_PRODUCTION",
        "slurm_job_id": job_id,
        "slurm_array_job_id": array_job_id,
        "slurm_array_task_id": int(array_task_id),
        "slurm_restart_count": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one frozen source-free comparator cell, counts only."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--test-only-array-index", type=int)
    parser.add_argument("--test-only-outer-replicates", type=int)
    arguments = parser.parse_args(argv)
    test_mode = (
        arguments.test_only_array_index is not None
        or arguments.test_only_outer_replicates is not None
    )
    require(
        (arguments.test_only_array_index is None)
        == (arguments.test_only_outer_replicates is None),
        "both explicit test-only arguments must be supplied together",
    )
    require(sys.dont_write_bytecode, "Python bytecode writing must be disabled")
    artifacts = validate_static_artifacts_and_runtime()
    numerics = _exec_pinned_module(
        NUMERICS_PATH,
        PINNED_NUMERICS_SHA256,
        "comparator_numerics_runner_bound_v1",
    )
    core = _exec_pinned_module(
        CORE_PATH,
        PINNED_CORE_SHA256,
        "general_directional_family_v3_comparator_runner_bound_v1",
    )
    require(core.METHOD_ID == "GENERAL_DIRECTIONAL_FAMILY_V3",
            "verified V3 core method ID differs")
    if test_mode:
        require(
            type(arguments.test_only_array_index) is int
            and 0 <= arguments.test_only_array_index <= 15,
            "test-only array index must be 0 through 15",
        )
        array_index = arguments.test_only_array_index
        execution = None
        test_replicates = arguments.test_only_outer_replicates
    else:
        raw_index = os.environ.get("SLURM_ARRAY_TASK_ID")
        require(
            raw_index is not None
            and raw_index in {str(value) for value in range(16)},
            "SLURM_ARRAY_TASK_ID must be exactly 0 through 15",
        )
        array_index = int(raw_index)
        test_replicates = None
        execution = _production_execution(artifacts["rows"][array_index])
    row = artifacts["rows"][array_index]
    payload = build_payload(
        row,
        artifacts,
        numerics,
        core,
        test_outer_replicates=test_replicates,
        execution=execution,
    )
    production_complete = (
        test_mode
        or payload["cell"]["outer_counts"]
        == {
            "attempted_replicates": PRODUCTION_REPLICATES,
            "successful_replicates": PRODUCTION_REPLICATES,
            "failed_replicates": 0,
        }
    )
    output = arguments.output_dir / shard_filename(row["cell_id"])
    write_create_only(output, canonical_json_bytes(payload))
    if not production_complete:
        print(json.dumps({
            "state": "WROTE_FAILED_COMPARATOR_COUNTS_ONLY_CELL_SHARD",
            "cell_id": row["cell_id"],
            "output_sha256": sha256_path(output.absolute()),
            "qualification_authority": "NONE",
            "outcome_access_authority": "NONE",
        }, sort_keys=True, separators=(",", ":")))
        return 1
    print(json.dumps({
        "state": "WROTE_COMPARATOR_COUNTS_ONLY_CELL_SHARD",
        "cell_id": row["cell_id"],
        "execution_mode": payload["execution_mode"],
        "output_sha256": sha256_path(output.absolute()),
        "qualification_authority": "NONE",
        "outcome_access_authority": "NONE",
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
