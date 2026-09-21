#!/usr/bin/env python3
"""Evaluate a generic reference-aware inference procedure with source-free simulations.

The simulations assess two properties:

* validity: simultaneous coverage, familywise/directional error,
  best-model-set coverage, and possible-rank-set coverage; and
* informativeness: interval width, resolved fraction, conservative fallback
  rate, best-set size, and possible-rank width.

The program reads no dataset, expression matrix, prediction, model score or
empirical outcome. It takes a formal JSON configuration. The built-in
``--self-test`` checks implementation invariants on a deliberately
small grid and is not a scientific qualification.

The primary procedure is ``CURRENT_JOINT_MAXT``:

* all within-reference-scheme model pairs form one joint family;
* root contributions are recentered and restudentized under sign flips;
* 8- and 12-root designs use every sign vector;
* 20- and 58-root designs use a fixed, antithetic, outcome-independent
  4,096-vector Monte Carlo sign schedule; and
* designs with fewer than eight roots, or invalid numerical structures,
  return unbounded intervals, full best-model sets, and full rank sets.

This conservative fallback sacrifices informativeness to preserve validity.
``NAIVE_CELL_LEVEL`` and ``ROOT_AWARE_MARGINAL`` are required comparators.
``ROOT_BONFERRONI_T`` is reported only as a standard sensitivity comparator and
does not replace the max-|T| procedure.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import beta, t


FRAMEWORK_ID = "GENERIC_REFERENCE_AWARE_ROOT_INFERENCE_V1_0"
SCHEMA_VERSION = "1.0.0"
REFERENCE_SCHEMES = (
    "NAIVE_SHARED",
    "INDEPENDENT_SPLIT",
    "UNIT_AWARE_CROSSFIT",
)
PROCEDURES = (
    "NAIVE_CELL_LEVEL",
    "ROOT_AWARE_MARGINAL",
    "ROOT_BONFERRONI_T",
    "CURRENT_JOINT_MAXT",
)
PRIMARY_PROCEDURE = "CURRENT_JOINT_MAXT"

REQUIRED_ROOT_COUNTS = (4, 6, 8, 12, 20, 58)
REQUIRED_METHOD_COUNTS = (6, 8, 10)
REQUIRED_SUPPORT_PATTERNS = ("BALANCED", "UNEQUAL")
REQUIRED_VARIANCE_PATTERNS = ("HOMOSKEDASTIC", "HETEROSKEDASTIC")
REQUIRED_ROOT_LAWS = (
    "GAUSSIAN",
    "STUDENT_T_DF4",
    "CENTERED_GAMMA_SHAPE2",
)
REQUIRED_TRUTH_PATTERNS = ("NULL", "CLEAR_WINNER", "TIED_BEST")
REQUIRED_REFERENCE_STRENGTHS = ("WEAK", "MODERATE", "STRONG")


class GenericFrameworkError(RuntimeError):
    """Raised when the outcome-free contract or simulation is invalid."""


@dataclass(frozen=True)
class Scenario:
    """One point in the generic design-geometry grid."""

    root_count: int
    method_count: int
    profile_id: str
    support_pattern: str
    variance_pattern: str
    reference_strength_name: str
    reference_strength: float
    root_law: str
    truth_pattern: str
    exact_degenerate: bool = False

    @property
    def scenario_id(self) -> str:
        return f"R{self.root_count}__M{self.method_count}__{self.profile_id}"


@dataclass(frozen=True)
class Family:
    """Within-scheme all-pair contrast family."""

    method_count: int
    identities: tuple[tuple[int, int, int], ...]
    labels: tuple[str, ...]

    @property
    def pair_count_per_scheme(self) -> int:
        return self.method_count * (self.method_count - 1) // 2

    @property
    def size(self) -> int:
        return len(self.identities)


def default_config() -> dict[str, Any]:
    """Return the bounded generic V1.0 profile grid.

    The 11 explicit profiles cover every required axis without taking the
    1,944-row Cartesian product.  Combined with six root counts and three
    method counts, the formal grid has 198 scenarios.
    """

    return {
        "schema_version": SCHEMA_VERSION,
        "framework_id": FRAMEWORK_ID,
        "outcome_free": True,
        "require_full_grid": True,
        "grid": {
            "root_counts": list(REQUIRED_ROOT_COUNTS),
            "method_counts": list(REQUIRED_METHOD_COUNTS),
            "profiles": [
                {
                    "profile_id": "P01_NULL_BALANCED_WEAK_GAUSSIAN",
                    "support_pattern": "BALANCED",
                    "variance_pattern": "HOMOSKEDASTIC",
                    "reference_strength_name": "WEAK",
                    "reference_strength": 0.15,
                    "root_law": "GAUSSIAN",
                    "truth_pattern": "NULL",
                    "exact_degenerate": False,
                },
                {
                    "profile_id": "P02_NULL_UNEQUAL_MODERATE_GAUSSIAN",
                    "support_pattern": "UNEQUAL",
                    "variance_pattern": "HOMOSKEDASTIC",
                    "reference_strength_name": "MODERATE",
                    "reference_strength": 0.60,
                    "root_law": "GAUSSIAN",
                    "truth_pattern": "NULL",
                    "exact_degenerate": False,
                },
                {
                    "profile_id": "P03_NULL_UNEQUAL_HETERO_STRONG_GAUSSIAN",
                    "support_pattern": "UNEQUAL",
                    "variance_pattern": "HETEROSKEDASTIC",
                    "reference_strength_name": "STRONG",
                    "reference_strength": 1.20,
                    "root_law": "GAUSSIAN",
                    "truth_pattern": "NULL",
                    "exact_degenerate": False,
                },
                {
                    "profile_id": "P04_NULL_UNEQUAL_HETERO_STRONG_T4",
                    "support_pattern": "UNEQUAL",
                    "variance_pattern": "HETEROSKEDASTIC",
                    "reference_strength_name": "STRONG",
                    "reference_strength": 1.20,
                    "root_law": "STUDENT_T_DF4",
                    "truth_pattern": "NULL",
                    "exact_degenerate": False,
                },
                {
                    "profile_id": "P05_NULL_UNEQUAL_HETERO_STRONG_SKEW",
                    "support_pattern": "UNEQUAL",
                    "variance_pattern": "HETEROSKEDASTIC",
                    "reference_strength_name": "STRONG",
                    "reference_strength": 1.20,
                    "root_law": "CENTERED_GAMMA_SHAPE2",
                    "truth_pattern": "NULL",
                    "exact_degenerate": False,
                },
                {
                    "profile_id": "P06_CLEAR_BALANCED_WEAK_GAUSSIAN",
                    "support_pattern": "BALANCED",
                    "variance_pattern": "HOMOSKEDASTIC",
                    "reference_strength_name": "WEAK",
                    "reference_strength": 0.15,
                    "root_law": "GAUSSIAN",
                    "truth_pattern": "CLEAR_WINNER",
                    "exact_degenerate": False,
                },
                {
                    "profile_id": "P07_CLEAR_UNEQUAL_HETERO_MODERATE_GAUSSIAN",
                    "support_pattern": "UNEQUAL",
                    "variance_pattern": "HETEROSKEDASTIC",
                    "reference_strength_name": "MODERATE",
                    "reference_strength": 0.60,
                    "root_law": "GAUSSIAN",
                    "truth_pattern": "CLEAR_WINNER",
                    "exact_degenerate": False,
                },
                {
                    "profile_id": "P08_CLEAR_UNEQUAL_HETERO_STRONG_T4",
                    "support_pattern": "UNEQUAL",
                    "variance_pattern": "HETEROSKEDASTIC",
                    "reference_strength_name": "STRONG",
                    "reference_strength": 1.20,
                    "root_law": "STUDENT_T_DF4",
                    "truth_pattern": "CLEAR_WINNER",
                    "exact_degenerate": False,
                },
                {
                    "profile_id": "P09_TIED_BALANCED_MODERATE_GAUSSIAN",
                    "support_pattern": "BALANCED",
                    "variance_pattern": "HOMOSKEDASTIC",
                    "reference_strength_name": "MODERATE",
                    "reference_strength": 0.60,
                    "root_law": "GAUSSIAN",
                    "truth_pattern": "TIED_BEST",
                    "exact_degenerate": False,
                },
                {
                    "profile_id": "P10_TIED_UNEQUAL_HETERO_STRONG_SKEW",
                    "support_pattern": "UNEQUAL",
                    "variance_pattern": "HETEROSKEDASTIC",
                    "reference_strength_name": "STRONG",
                    "reference_strength": 1.20,
                    "root_law": "CENTERED_GAMMA_SHAPE2",
                    "truth_pattern": "TIED_BEST",
                    "exact_degenerate": False,
                },
                {
                    "profile_id": "P11_EXACT_DEGENERATE_FALLBACK",
                    "support_pattern": "BALANCED",
                    "variance_pattern": "HOMOSKEDASTIC",
                    "reference_strength_name": "WEAK",
                    "reference_strength": 0.15,
                    "root_law": "GAUSSIAN",
                    "truth_pattern": "NULL",
                    "exact_degenerate": True,
                },
            ],
        },
        "simulation": {
            "alpha": 0.05,
            "repetitions": 10_000,
            "batch_size": 8,
            "seed": 2_026_072_901,
            "sign_schedule_seed": 1_046_527,
            "monte_carlo_sign_count": 4096,
            "minimum_roots_for_joint_inference": 8,
            "balanced_descendants_per_root": 48,
            "unequal_descendants_minimum": 12,
            "unequal_descendants_maximum": 96,
            "clear_winner_effect": 0.75,
            "tied_best_effect": 0.75,
            "finite_cell_standard_deviation": 0.80,
        },
        "validity_gates": {
            "simultaneous_coverage_lower_95_min": 0.94,
            "familywise_or_wrong_direction_error_upper_95_max": 0.06,
            "best_model_set_coverage_lower_95_min": 0.94,
            "possible_rank_set_coverage_lower_95_min": 0.94,
        },
    }


def self_test_config() -> dict[str, Any]:
    """Return a tiny implementation-only grid."""

    config = default_config()
    config["require_full_grid"] = False
    config["grid"] = {
        "root_counts": [4, 8],
        "method_counts": [6],
        "profiles": [
            {
                "profile_id": "SELF_TEST_NULL_BALANCED",
                "support_pattern": "BALANCED",
                "variance_pattern": "HOMOSKEDASTIC",
                "reference_strength_name": "MODERATE",
                "reference_strength": 0.60,
                "root_law": "GAUSSIAN",
                "truth_pattern": "NULL",
                "exact_degenerate": False,
            }
        ],
    }
    config["simulation"]["repetitions"] = 8
    config["simulation"]["batch_size"] = 4
    return config


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise GenericFrameworkError(
            f"{label} keys differ; missing={missing}, extra={extra}"
        )


def _unique_sequence(
    value: Any, *, label: str, cast: type[int] | type[str]
) -> tuple[int, ...] | tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise GenericFrameworkError(f"{label} must be a non-empty list")
    try:
        converted = tuple(cast(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise GenericFrameworkError(f"{label} contains an invalid value") from exc
    if len(set(converted)) != len(converted):
        raise GenericFrameworkError(f"{label} contains duplicates")
    return converted


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a formal outcome-free configuration."""

    if not isinstance(config, Mapping):
        raise GenericFrameworkError("configuration root must be an object")
    _require_exact_keys(
        config,
        {
            "schema_version",
            "framework_id",
            "outcome_free",
            "require_full_grid",
            "grid",
            "simulation",
            "validity_gates",
        },
        "configuration",
    )
    if (
        config["schema_version"] != SCHEMA_VERSION
        or config["framework_id"] != FRAMEWORK_ID
        or config["outcome_free"] is not True
        or not isinstance(config["require_full_grid"], bool)
    ):
        raise GenericFrameworkError(
            "framework identity or outcome-free declaration differs"
        )

    grid = config["grid"]
    if not isinstance(grid, Mapping):
        raise GenericFrameworkError("grid must be an object")
    _require_exact_keys(
        grid,
        {
            "root_counts",
            "method_counts",
            "profiles",
        },
        "grid",
    )
    roots = _unique_sequence(
        grid["root_counts"], label="root_counts", cast=int
    )
    methods = _unique_sequence(
        grid["method_counts"], label="method_counts", cast=int
    )
    profiles_raw = grid["profiles"]
    if not isinstance(profiles_raw, list) or not profiles_raw:
        raise GenericFrameworkError("profiles must be a non-empty list")
    profile_keys = {
        "profile_id",
        "support_pattern",
        "variance_pattern",
        "reference_strength_name",
        "reference_strength",
        "root_law",
        "truth_pattern",
        "exact_degenerate",
    }
    profiles: list[dict[str, Any]] = []
    for index, raw in enumerate(profiles_raw):
        if not isinstance(raw, Mapping):
            raise GenericFrameworkError(f"profile {index} must be an object")
        _require_exact_keys(raw, profile_keys, f"profile {index}")
        profile = {
            "profile_id": str(raw["profile_id"]),
            "support_pattern": str(raw["support_pattern"]),
            "variance_pattern": str(raw["variance_pattern"]),
            "reference_strength_name": str(
                raw["reference_strength_name"]
            ),
            "reference_strength": float(raw["reference_strength"]),
            "root_law": str(raw["root_law"]),
            "truth_pattern": str(raw["truth_pattern"]),
            "exact_degenerate": raw["exact_degenerate"],
        }
        if (
            not profile["profile_id"]
            or not isinstance(profile["exact_degenerate"], bool)
            or profile["support_pattern"] not in REQUIRED_SUPPORT_PATTERNS
            or profile["variance_pattern"] not in REQUIRED_VARIANCE_PATTERNS
            or profile["reference_strength_name"]
            not in REQUIRED_REFERENCE_STRENGTHS
            or profile["root_law"] not in REQUIRED_ROOT_LAWS
            or profile["truth_pattern"] not in REQUIRED_TRUTH_PATTERNS
            or not math.isfinite(profile["reference_strength"])
            or profile["reference_strength"] < 0.0
        ):
            raise GenericFrameworkError(f"profile {index} is invalid")
        if (
            profile["exact_degenerate"]
            and profile["truth_pattern"] != "NULL"
        ):
            raise GenericFrameworkError(
                "exact-degenerate profile must have null truth"
            )
        profiles.append(profile)
    profile_ids = [profile["profile_id"] for profile in profiles]
    if len(set(profile_ids)) != len(profile_ids):
        raise GenericFrameworkError("profile identifiers contain duplicates")
    if any(root not in REQUIRED_ROOT_COUNTS for root in roots):
        raise GenericFrameworkError("unsupported root count")
    if any(method not in REQUIRED_METHOD_COUNTS for method in methods):
        raise GenericFrameworkError("unsupported method count")

    if config["require_full_grid"]:
        required = (
            (set(roots), set(REQUIRED_ROOT_COUNTS), "root counts"),
            (set(methods), set(REQUIRED_METHOD_COUNTS), "method counts"),
            (
                {profile["support_pattern"] for profile in profiles},
                set(REQUIRED_SUPPORT_PATTERNS),
                "support patterns",
            ),
            (
                {profile["variance_pattern"] for profile in profiles},
                set(REQUIRED_VARIANCE_PATTERNS),
                "variance patterns",
            ),
            (
                {profile["root_law"] for profile in profiles},
                set(REQUIRED_ROOT_LAWS),
                "root laws",
            ),
            (
                {profile["truth_pattern"] for profile in profiles},
                set(REQUIRED_TRUTH_PATTERNS),
                "truth patterns",
            ),
            (
                {
                    profile["reference_strength_name"]
                    for profile in profiles
                },
                set(REQUIRED_REFERENCE_STRENGTHS),
                "reference strengths",
            ),
        )
        for observed, expected, label in required:
            if observed != expected:
                raise GenericFrameworkError(
                    f"full generic grid does not cover all required {label}"
                )
        if not 10 <= len(profiles) <= 11:
            raise GenericFrameworkError(
                "formal generic grid must contain 10 or 11 bounded profiles"
            )

        def covered(**requirements: Any) -> bool:
            return any(
                all(profile[key] == value for key, value in requirements.items())
                for profile in profiles
            )

        required_profiles = {
            "null balanced weak Gaussian": covered(
                truth_pattern="NULL",
                support_pattern="BALANCED",
                reference_strength_name="WEAK",
                root_law="GAUSSIAN",
            ),
            "null unequal moderate": covered(
                truth_pattern="NULL",
                support_pattern="UNEQUAL",
                reference_strength_name="MODERATE",
            ),
            "null unequal heteroskedastic strong Gaussian": covered(
                truth_pattern="NULL",
                support_pattern="UNEQUAL",
                variance_pattern="HETEROSKEDASTIC",
                reference_strength_name="STRONG",
                root_law="GAUSSIAN",
            ),
            "null unequal heteroskedastic strong t4": covered(
                truth_pattern="NULL",
                support_pattern="UNEQUAL",
                variance_pattern="HETEROSKEDASTIC",
                reference_strength_name="STRONG",
                root_law="STUDENT_T_DF4",
            ),
            "null unequal heteroskedastic strong skew": covered(
                truth_pattern="NULL",
                support_pattern="UNEQUAL",
                variance_pattern="HETEROSKEDASTIC",
                reference_strength_name="STRONG",
                root_law="CENTERED_GAMMA_SHAPE2",
            ),
            "clear winner balanced": covered(
                truth_pattern="CLEAR_WINNER",
                support_pattern="BALANCED",
            ),
            "clear winner unequal heteroskedastic": covered(
                truth_pattern="CLEAR_WINNER",
                support_pattern="UNEQUAL",
                variance_pattern="HETEROSKEDASTIC",
            ),
            "tied best balanced": covered(
                truth_pattern="TIED_BEST",
                support_pattern="BALANCED",
            ),
            "tied best unequal heteroskedastic": covered(
                truth_pattern="TIED_BEST",
                support_pattern="UNEQUAL",
                variance_pattern="HETEROSKEDASTIC",
            ),
            "exact-degenerate fallback": any(
                profile["exact_degenerate"] for profile in profiles
            ),
        }
        missing_profiles = [
            label for label, present in required_profiles.items() if not present
        ]
        if missing_profiles:
            raise GenericFrameworkError(
                "formal profile grid lacks: " + ", ".join(missing_profiles)
            )

    simulation = config["simulation"]
    if not isinstance(simulation, Mapping):
        raise GenericFrameworkError("simulation must be an object")
    _require_exact_keys(
        simulation,
        {
            "alpha",
            "repetitions",
            "batch_size",
            "seed",
            "sign_schedule_seed",
            "monte_carlo_sign_count",
            "minimum_roots_for_joint_inference",
            "balanced_descendants_per_root",
            "unequal_descendants_minimum",
            "unequal_descendants_maximum",
            "clear_winner_effect",
            "tied_best_effect",
            "finite_cell_standard_deviation",
        },
        "simulation",
    )
    alpha = float(simulation["alpha"])
    repetitions = int(simulation["repetitions"])
    batch_size = int(simulation["batch_size"])
    sign_count = int(simulation["monte_carlo_sign_count"])
    minimum_roots = int(simulation["minimum_roots_for_joint_inference"])
    balanced_n = int(simulation["balanced_descendants_per_root"])
    unequal_min = int(simulation["unequal_descendants_minimum"])
    unequal_max = int(simulation["unequal_descendants_maximum"])
    if (
        not 0.0 < alpha < 0.5
        or repetitions < 2
        or batch_size < 1
        or sign_count != 4096
        or sign_count % 2
        or minimum_roots != 8
        or balanced_n < 2
        or unequal_min < 2
        or unequal_max < unequal_min
    ):
        raise GenericFrameworkError("simulation constants are invalid")
    for key in (
        "clear_winner_effect",
        "tied_best_effect",
        "finite_cell_standard_deviation",
    ):
        number = float(simulation[key])
        if not math.isfinite(number) or number <= 0.0:
            raise GenericFrameworkError(f"{key} must be finite and positive")

    gates = config["validity_gates"]
    if not isinstance(gates, Mapping):
        raise GenericFrameworkError("validity_gates must be an object")
    _require_exact_keys(
        gates,
        {
            "simultaneous_coverage_lower_95_min",
            "familywise_or_wrong_direction_error_upper_95_max",
            "best_model_set_coverage_lower_95_min",
            "possible_rank_set_coverage_lower_95_min",
        },
        "validity_gates",
    )
    if any(
        not 0.0 <= float(value) <= 1.0 for value in gates.values()
    ):
        raise GenericFrameworkError("validity gates must lie in [0, 1]")

    normalized = deepcopy(dict(config))
    normalized["grid"]["root_counts"] = list(roots)
    normalized["grid"]["method_counts"] = list(methods)
    normalized["grid"]["profiles"] = profiles
    return normalized


def iter_scenarios(config: Mapping[str, Any]) -> Iterable[Scenario]:
    """Cross root/method counts with the bounded explicit profile rows."""

    grid = config["grid"]
    for root_count, method_count, profile in itertools.product(
        grid["root_counts"],
        grid["method_counts"],
        grid["profiles"],
    ):
        yield Scenario(
            int(root_count),
            int(method_count),
            str(profile["profile_id"]),
            str(profile["support_pattern"]),
            str(profile["variance_pattern"]),
            str(profile["reference_strength_name"]),
            float(profile["reference_strength"]),
            str(profile["root_law"]),
            str(profile["truth_pattern"]),
            bool(profile["exact_degenerate"]),
        )


def contrast_family(method_count: int) -> Family:
    """Build all method pairs within each of the three reference schemes."""

    if method_count not in REQUIRED_METHOD_COUNTS:
        raise GenericFrameworkError("method count is outside the generic grid")
    identities: list[tuple[int, int, int]] = []
    labels: list[str] = []
    for scheme_index, scheme in enumerate(REFERENCE_SCHEMES):
        for left in range(method_count):
            for right in range(left + 1, method_count):
                identities.append((scheme_index, left, right))
                labels.append(f"{scheme}::M{left + 1}_MINUS_M{right + 1}")
    family = Family(method_count, tuple(identities), tuple(labels))
    expected = len(REFERENCE_SCHEMES) * method_count * (method_count - 1) // 2
    if family.size != expected or len(set(family.labels)) != expected:
        raise GenericFrameworkError("contrast family construction failed")
    return family


def sign_schedule(
    root_count: int, *, monte_carlo_count: int = 4096, seed: int = 1_046_527
) -> np.ndarray | None:
    """Return the frozen exhaustive or antithetic sign schedule.

    Designs below eight roots deliberately abstain and therefore have no sign
    schedule.  The Monte Carlo schedule seed is a method constant, not the
    simulation random seed.
    """

    if root_count not in REQUIRED_ROOT_COUNTS:
        raise GenericFrameworkError("root count is outside the generic grid")
    if root_count < 8:
        return None
    if root_count in (8, 12):
        masks = np.arange(2**root_count, dtype=np.uint64)[:, None]
        shifts = np.arange(root_count, dtype=np.uint64)[None, :]
        bits = (masks >> shifts) & np.uint64(1)
        signs = bits.astype(np.int8) * np.int8(2) - np.int8(1)
    else:
        if monte_carlo_count != 4096 or monte_carlo_count % 2:
            raise GenericFrameworkError(
                "Monte Carlo schedule must contain 4096 antithetic signs"
            )
        rng = np.random.default_rng(
            np.random.SeedSequence((seed, root_count, 0x5349474E))
        )
        half = monte_carlo_count // 2
        rows: list[np.ndarray] = []
        seen: set[bytes] = set()
        while len(rows) < half:
            row = (
                rng.integers(0, 2, size=root_count, dtype=np.int8) * 2 - 1
            )
            key = row.tobytes()
            opposite_key = (-row).tobytes()
            if key in seen or opposite_key in seen:
                continue
            rows.append(row)
            seen.add(key)
            seen.add(opposite_key)
        first = np.stack(rows)
        signs = np.concatenate((first, -first[::-1]), axis=0)
    if (
        signs.ndim != 2
        or signs.shape[1] != root_count
        or not np.all((signs == -1) | (signs == 1))
        or not np.array_equal(signs[::-1], -signs)
        or np.unique(signs, axis=0).shape[0] != signs.shape[0]
    ):
        raise GenericFrameworkError("sign schedule is not unique and antithetic")
    signs.setflags(write=False)
    return signs


def sign_schedule_sha256(signs: np.ndarray | None) -> str | None:
    if signs is None:
        return None
    digest = hashlib.sha256()
    digest.update(np.asarray(signs.shape, dtype="<i8").tobytes())
    digest.update(signs.astype(np.int8, copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _truth(
    scenario: Scenario, simulation: Mapping[str, Any]
) -> np.ndarray:
    values = np.zeros(
        (len(REFERENCE_SCHEMES), scenario.method_count), dtype=np.float64
    )
    if scenario.truth_pattern == "CLEAR_WINNER":
        values[:, 0] = float(simulation["clear_winner_effect"])
    elif scenario.truth_pattern == "TIED_BEST":
        values[:, :2] = float(simulation["tied_best_effect"])
    elif scenario.truth_pattern != "NULL":
        raise GenericFrameworkError("unknown truth pattern")
    return values


def _standard_draw(
    rng: np.random.Generator, shape: tuple[int, ...], law: str
) -> np.ndarray:
    if law == "GAUSSIAN":
        return rng.normal(size=shape)
    if law == "STUDENT_T_DF4":
        return rng.standard_t(df=4, size=shape) / math.sqrt(2.0)
    if law == "CENTERED_GAMMA_SHAPE2":
        return (rng.gamma(shape=2.0, scale=1.0, size=shape) - 2.0) / math.sqrt(
            2.0
        )
    raise GenericFrameworkError("unknown root law")


def _root_support(
    scenario: Scenario, simulation: Mapping[str, Any]
) -> np.ndarray:
    if scenario.support_pattern == "BALANCED":
        support = np.full(
            scenario.root_count,
            int(simulation["balanced_descendants_per_root"]),
            dtype=np.int64,
        )
    elif scenario.support_pattern == "UNEQUAL":
        support = np.rint(
            np.geomspace(
                int(simulation["unequal_descendants_minimum"]),
                int(simulation["unequal_descendants_maximum"]),
                scenario.root_count,
            )
        ).astype(np.int64)
        # A deterministic rotation avoids always assigning minimum support to
        # the first root while remaining outcome-independent.
        support = np.roll(support, scenario.root_count // 3)
    else:
        raise GenericFrameworkError("unknown support pattern")
    if np.any(support < 2):
        raise GenericFrameworkError("every root needs at least two descendants")
    return support


def simulate_root_utilities(
    rng: np.random.Generator,
    repetitions: int,
    scenario: Scenario,
    simulation: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate observed root-level utilities and descendant counts."""

    roots = scenario.root_count
    methods = scenario.method_count
    schemes = len(REFERENCE_SCHEMES)
    truth = _truth(scenario, simulation)
    method_loading = np.linspace(0.55, 1.35, methods)
    reference_loading = np.linspace(0.35, 1.30, methods)

    common = _standard_draw(rng, (repetitions, roots, 1), scenario.root_law)
    method_specific = _standard_draw(
        rng, (repetitions, roots, methods), scenario.root_law
    )
    root_method = (
        0.35 * common * method_loading[None, None, :]
        + math.sqrt(1.0 - 0.35**2) * method_specific
    )
    scheme_method = _standard_draw(
        rng, (repetitions, roots, schemes, methods), scenario.root_law
    )
    reference_common = _standard_draw(
        rng, (repetitions, roots, schemes, 1), scenario.root_law
    )
    reference_specific = _standard_draw(
        rng, (repetitions, roots, schemes, methods), scenario.root_law
    )

    root_scale = np.ones(roots, dtype=np.float64)
    if scenario.variance_pattern == "HETEROSKEDASTIC":
        root_scale = np.linspace(0.45, 1.75, roots)
        root_scale /= math.sqrt(float(np.mean(np.square(root_scale))))
    elif scenario.variance_pattern != "HOMOSKEDASTIC":
        raise GenericFrameworkError("unknown variance pattern")
    scale = root_scale[None, :, None, None]
    scheme_reference_factor = np.asarray((1.00, 1.15, 0.75))[
        None, None, :, None
    ]
    reference = scenario.reference_strength * scheme_reference_factor * (
        0.75
        * reference_common
        * reference_loading[None, None, None, :]
        + 0.35 * reference_specific
    )
    latent = (
        truth[None, None, :, :]
        + scale
        * (
            0.60 * root_method[:, :, None, :]
            + 0.20 * scheme_method
            + reference
        )
    )
    support = _root_support(scenario, simulation)
    cell_sd = float(simulation["finite_cell_standard_deviation"])
    finite_cell_noise = rng.normal(
        size=(repetitions, roots, schemes, methods)
    ) * (cell_sd / np.sqrt(support))[None, :, None, None]
    observed = latent + finite_cell_noise
    if scenario.exact_degenerate:
        # Every method has exactly the same root contribution.  This profile
        # tests the declared conservative fallback; it is not a near-zero
        # tuning scenario.
        observed = np.repeat(observed[:, :, :, :1], methods, axis=3)
    if (
        observed.shape != (repetitions, roots, schemes, methods)
        or not np.isfinite(observed).all()
    ):
        raise GenericFrameworkError("simulated root utilities are invalid")
    return observed, support


def root_contrasts(root_utilities: np.ndarray, family: Family) -> np.ndarray:
    values = np.asarray(root_utilities, dtype=np.float64)
    if (
        values.ndim != 4
        or values.shape[2:] != (
            len(REFERENCE_SCHEMES),
            family.method_count,
        )
    ):
        raise GenericFrameworkError("root utility tensor shape differs")
    output = np.empty(
        (values.shape[0], values.shape[1], family.size), dtype=np.float64
    )
    for index, (scheme, left, right) in enumerate(family.identities):
        output[:, :, index] = (
            values[:, :, scheme, left] - values[:, :, scheme, right]
        )
    return output


def true_contrasts(
    truth: np.ndarray, family: Family
) -> np.ndarray:
    output = np.empty(family.size, dtype=np.float64)
    for index, (scheme, left, right) in enumerate(family.identities):
        output[index] = truth[scheme, left] - truth[scheme, right]
    return output


def _interval_result(
    point: np.ndarray,
    standard_errors: np.ndarray,
    critical: float | np.ndarray,
    *,
    force_fallback: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    point = np.asarray(point, dtype=np.float64)
    se = np.asarray(standard_errors, dtype=np.float64)
    if point.shape != se.shape or point.ndim != 2:
        raise GenericFrameworkError("point and standard-error shapes differ")
    repetitions, family_size = point.shape
    invalid = ~np.all(np.isfinite(point) & np.isfinite(se) & (se > 0.0), axis=1)
    if force_fallback is not None:
        forced = np.asarray(force_fallback, dtype=bool)
        if forced.shape != (repetitions,):
            raise GenericFrameworkError("fallback mask shape differs")
        invalid |= forced
    critical_array = np.asarray(critical, dtype=np.float64)
    if critical_array.ndim == 0:
        critical_array = np.full(repetitions, float(critical_array))
    if critical_array.shape != (repetitions,):
        raise GenericFrameworkError("critical-value shape differs")
    invalid |= ~np.isfinite(critical_array) | (critical_array <= 0.0)
    lower = np.full((repetitions, family_size), -np.inf)
    upper = np.full((repetitions, family_size), np.inf)
    valid = ~invalid
    lower[valid] = point[valid] - critical_array[valid, None] * se[valid]
    upper[valid] = point[valid] + critical_array[valid, None] * se[valid]
    return {
        "point": point,
        "standard_errors": se,
        "critical_values": critical_array,
        "lower": lower,
        "upper": upper,
        "fallback": invalid,
    }


def infer_naive_cell_level(
    contrasts: np.ndarray,
    support: np.ndarray,
    *,
    alpha: float,
    finite_cell_sd: float,
) -> dict[str, np.ndarray]:
    """Treat descendant cells as independent (deliberately naive)."""

    values = np.asarray(contrasts, dtype=np.float64)
    counts = np.asarray(support, dtype=np.float64)
    if (
        values.ndim != 3
        or counts.shape != (values.shape[1],)
        or np.any(counts < 2)
    ):
        raise GenericFrameworkError("naive comparator input shape differs")
    total = float(np.sum(counts))
    point = np.einsum("brf,r->bf", values, counts, optimize=True) / total
    between = np.einsum(
        "brf,r->bf",
        np.square(values - point[:, None, :]),
        counts,
        optimize=True,
    )
    within_pair_variance = 2.0 * finite_cell_sd**2
    within = float(np.sum((counts - 1.0) * within_pair_variance))
    sample_variance = (between + within) / (total - 1.0)
    se = np.sqrt(np.maximum(sample_variance, 0.0) / total)
    critical = float(t.ppf(1.0 - alpha / 2.0, df=int(total - 1)))
    return _interval_result(point, se, critical)


def infer_root_t(
    contrasts: np.ndarray,
    *,
    alpha: float,
    bonferroni: bool,
) -> dict[str, np.ndarray]:
    """Root-aware marginal or Bonferroni-t intervals."""

    values = np.asarray(contrasts, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] < 2:
        raise GenericFrameworkError("root comparator input shape differs")
    roots = values.shape[1]
    family_size = values.shape[2]
    point = np.mean(values, axis=1)
    se = np.std(values, axis=1, ddof=1) / math.sqrt(roots)
    tail = alpha / (2.0 * family_size) if bonferroni else alpha / 2.0
    critical = float(t.ppf(1.0 - tail, df=roots - 1))
    return _interval_result(point, se, critical)


def infer_current_joint_maxt(
    contrasts: np.ndarray,
    signs: np.ndarray | None,
    *,
    alpha: float,
    minimum_roots: int = 8,
    sign_chunk_size: int = 256,
) -> dict[str, np.ndarray]:
    """Recentered/restudentized root sign-flip max-|T| inference."""

    values = np.asarray(contrasts, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] < 2:
        raise GenericFrameworkError("joint comparator input shape differs")
    repetitions, roots, family_size = values.shape
    point = np.mean(values, axis=1)
    se = np.std(values, axis=1, ddof=1) / math.sqrt(roots)
    if roots < minimum_roots:
        return _interval_result(
            point,
            se,
            np.ones(repetitions),
            force_fallback=np.ones(repetitions, dtype=bool),
        )
    if signs is None or signs.shape[1] != roots:
        raise GenericFrameworkError("eligible design lacks its sign schedule")

    influences = (values - point[:, None, :]) / roots
    centering_error = np.max(np.abs(np.sum(influences, axis=1)), axis=1)
    centering_scale = np.maximum(
        np.max(np.sum(np.abs(influences), axis=1), axis=1),
        np.finfo(np.float64).tiny,
    )
    centered = (
        centering_error
        <= 512.0 * np.finfo(np.float64).eps * centering_scale
    )
    base_ss = np.sum(np.square(influences), axis=1)
    maxima = np.empty((repetitions, signs.shape[0]), dtype=np.float64)
    pseudo_valid = np.ones(repetitions, dtype=bool)
    multiplier = roots / (roots - 1.0)
    sign_float = signs.astype(np.float64, copy=False)
    for start in range(0, signs.shape[0], sign_chunk_size):
        stop = min(start + sign_chunk_size, signs.shape[0])
        signed = np.einsum(
            "qr,brf->bqf",
            sign_float[start:stop],
            influences,
            optimize=True,
        )
        pseudo_ss = base_ss[:, None, :] - np.square(signed) / roots
        tolerance = (
            2048.0
            * np.finfo(np.float64).eps
            * np.maximum(base_ss[:, None, :], np.finfo(np.float64).tiny)
        )
        pseudo_valid &= np.all(
            np.isfinite(pseudo_ss) & (pseudo_ss > tolerance), axis=(1, 2)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            studentized = signed / np.sqrt(multiplier * pseudo_ss)
        maxima[:, start:stop] = np.max(np.abs(studentized), axis=2)

    complement_scale = np.maximum(
        np.maximum(np.abs(maxima), np.abs(maxima[:, ::-1])), 1.0
    )
    complement_valid = np.all(
        np.abs(maxima - maxima[:, ::-1])
        <= 512.0 * np.finfo(np.float64).eps * complement_scale,
        axis=1,
    )
    order = int(math.ceil((1.0 - alpha) * signs.shape[0])) - 1
    critical = np.partition(maxima, order, axis=1)[:, order]
    forced = (
        ~centered
        | ~pseudo_valid
        | ~complement_valid
        | ~np.all(np.isfinite(maxima), axis=1)
    )
    return _interval_result(
        point, se, critical, force_fallback=forced
    )


def decision_sets(
    lower: np.ndarray,
    upper: np.ndarray,
    family: Family,
    truth: np.ndarray,
) -> dict[str, np.ndarray]:
    """Derive best-model and possible-rank confidence sets."""

    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if lower.shape != upper.shape or lower.shape[1] != family.size:
        raise GenericFrameworkError("interval family shape differs")
    repetitions = lower.shape[0]
    schemes = len(REFERENCE_SCHEMES)
    methods = family.method_count
    lower_matrix = np.zeros(
        (repetitions, schemes, methods, methods), dtype=np.float64
    )
    upper_matrix = np.zeros_like(lower_matrix)
    for index, (scheme, left, right) in enumerate(family.identities):
        lower_matrix[:, scheme, left, right] = lower[:, index]
        upper_matrix[:, scheme, left, right] = upper[:, index]
        lower_matrix[:, scheme, right, left] = -upper[:, index]
        upper_matrix[:, scheme, right, left] = -lower[:, index]

    reach = lower_matrix > 0.0
    for intermediate in range(methods):
        reach |= (
            reach[:, :, :, intermediate, None]
            & reach[:, :, None, intermediate, :]
        )
    cycles = np.any(np.diagonal(reach, axis1=2, axis2=3), axis=(1, 2))
    predecessors = np.sum(reach, axis=2)
    successors = np.sum(reach, axis=3)
    best_set = predecessors == 0
    rank_lower = 1 + predecessors
    rank_upper = methods - successors
    if np.any(cycles):
        best_set[cycles] = True
        rank_lower[cycles] = 1
        rank_upper[cycles] = methods

    tolerance = 1.0e-14
    true_best = np.isclose(
        truth,
        np.max(truth, axis=1, keepdims=True),
        rtol=0.0,
        atol=tolerance,
    )
    best_covered = np.all(
        (~true_best[None, :, :]) | best_set, axis=(1, 2)
    )
    true_rank_lower = np.empty((schemes, methods), dtype=np.int64)
    true_rank_upper = np.empty_like(true_rank_lower)
    for scheme in range(schemes):
        for method in range(methods):
            greater = int(
                np.sum(truth[scheme] > truth[scheme, method] + tolerance)
            )
            equal = int(
                np.sum(
                    np.isclose(
                        truth[scheme],
                        truth[scheme, method],
                        rtol=0.0,
                        atol=tolerance,
                    )
                )
            )
            true_rank_lower[scheme, method] = 1 + greater
            true_rank_upper[scheme, method] = greater + equal
    rank_covered = np.all(
        (rank_lower <= true_rank_lower[None, :, :])
        & (rank_upper >= true_rank_upper[None, :, :]),
        axis=(1, 2),
    )

    singleton = np.sum(best_set, axis=2) == 1
    selected = np.argmax(best_set, axis=2)
    false_singleton_by_scheme = np.zeros(
        (repetitions, schemes), dtype=bool
    )
    correct_singleton_by_scheme = np.zeros_like(false_singleton_by_scheme)
    for scheme in range(schemes):
        true_best_count = int(np.sum(true_best[scheme]))
        false_singleton_by_scheme[:, scheme] = singleton[:, scheme] & (
            (true_best_count != 1)
            | (~true_best[scheme, selected[:, scheme]])
        )
        if true_best_count == 1:
            true_winner = int(np.argmax(true_best[scheme]))
            correct_singleton_by_scheme[:, scheme] = singleton[:, scheme] & (
                selected[:, scheme] == true_winner
            )
    return {
        "best_set": best_set,
        "rank_lower": rank_lower,
        "rank_upper": rank_upper,
        "best_covered": best_covered,
        "rank_covered": rank_covered,
        "false_singleton": np.any(false_singleton_by_scheme, axis=1),
        "correct_singleton_by_scheme": correct_singleton_by_scheme,
        "cycle_fallback": cycles,
    }


def _binomial_bound(successes: int, trials: int, *, lower: bool) -> float:
    if trials < 1 or not 0 <= successes <= trials:
        raise GenericFrameworkError("invalid binomial count")
    if lower:
        return (
            0.0
            if successes == 0
            else float(beta.ppf(0.05, successes, trials - successes + 1))
        )
    return (
        1.0
        if successes == trials
        else float(beta.ppf(0.95, successes + 1, trials - successes))
    )


def _new_accumulator(family_size: int, scheme_count: int) -> dict[str, Any]:
    return {
        "repetitions": 0,
        "simultaneous_successes": 0,
        "error_events": 0,
        "best_successes": 0,
        "rank_successes": 0,
        "false_singletons": 0,
        "fallbacks": 0,
        "cycles": 0,
        "finite_width_sum": 0.0,
        "finite_width_count": 0,
        "resolved_intervals": 0,
        "interval_count": 0,
        "best_set_size_sum": 0.0,
        "rank_size_sum": 0.0,
        "decision_cell_count": 0,
        "correct_singletons_by_scheme": np.zeros(
            scheme_count, dtype=np.int64
        ),
        "point_sum": np.zeros(family_size, dtype=np.float64),
        "point_square_sum": np.zeros(family_size, dtype=np.float64),
        "se_square_sum": np.zeros(family_size, dtype=np.float64),
        "se_count": np.zeros(family_size, dtype=np.int64),
        "cross_scheme_shift_sum": 0.0,
        "cross_scheme_shift_count": 0,
    }


def _update_accumulator(
    accumulator: dict[str, Any],
    result: Mapping[str, np.ndarray],
    *,
    family: Family,
    truth: np.ndarray,
    truth_contrasts_vector: np.ndarray,
) -> None:
    lower = result["lower"]
    upper = result["upper"]
    point = result["point"]
    se = result["standard_errors"]
    fallback = result["fallback"]
    repetitions = lower.shape[0]
    null = np.isclose(
        truth_contrasts_vector, 0.0, rtol=0.0, atol=1.0e-14
    )
    covered = np.all(
        (lower <= truth_contrasts_vector[None, :])
        & (upper >= truth_contrasts_vector[None, :]),
        axis=1,
    )
    null_error = np.any(
        ((lower > 0.0) | (upper < 0.0)) & null[None, :], axis=1
    )
    wrong_direction = np.any(
        (
            (truth_contrasts_vector[None, :] > 0.0)
            & (upper < 0.0)
            & (~null[None, :])
        )
        | (
            (truth_contrasts_vector[None, :] < 0.0)
            & (lower > 0.0)
            & (~null[None, :])
        ),
        axis=1,
    )
    decisions = decision_sets(lower, upper, family, truth)
    widths = upper - lower
    finite_width = np.isfinite(widths)
    resolved = (lower > 0.0) | (upper < 0.0)

    accumulator["repetitions"] += repetitions
    accumulator["simultaneous_successes"] += int(np.sum(covered))
    accumulator["error_events"] += int(np.sum(null_error | wrong_direction))
    accumulator["best_successes"] += int(
        np.sum(decisions["best_covered"])
    )
    accumulator["rank_successes"] += int(
        np.sum(decisions["rank_covered"])
    )
    accumulator["false_singletons"] += int(
        np.sum(decisions["false_singleton"])
    )
    accumulator["fallbacks"] += int(np.sum(fallback))
    accumulator["cycles"] += int(np.sum(decisions["cycle_fallback"]))
    accumulator["finite_width_sum"] += float(np.sum(widths[finite_width]))
    accumulator["finite_width_count"] += int(np.sum(finite_width))
    accumulator["resolved_intervals"] += int(np.sum(resolved))
    accumulator["interval_count"] += int(widths.size)
    accumulator["best_set_size_sum"] += float(
        np.sum(decisions["best_set"])
    )
    rank_sizes = decisions["rank_upper"] - decisions["rank_lower"] + 1
    accumulator["rank_size_sum"] += float(np.sum(rank_sizes))
    accumulator["decision_cell_count"] += int(rank_sizes.size)
    accumulator["correct_singletons_by_scheme"] += np.sum(
        decisions["correct_singleton_by_scheme"], axis=0
    )
    accumulator["point_sum"] += np.sum(point, axis=0)
    accumulator["point_square_sum"] += np.sum(np.square(point), axis=0)
    finite_se = np.isfinite(se) & (se > 0.0)
    accumulator["se_square_sum"] += np.sum(
        np.where(finite_se, np.square(se), 0.0), axis=0
    )
    accumulator["se_count"] += np.sum(finite_se, axis=0)

    pairs = family.pair_count_per_scheme
    reshaped = point.reshape(
        repetitions, len(REFERENCE_SCHEMES), pairs
    )
    for scheme in range(1, len(REFERENCE_SCHEMES)):
        accumulator["cross_scheme_shift_sum"] += float(
            np.sum(np.abs(reshaped[:, scheme] - reshaped[:, 0]))
        )
        accumulator["cross_scheme_shift_count"] += repetitions * pairs


def _finalize_accumulator(
    accumulator: Mapping[str, Any],
    *,
    truth: np.ndarray,
) -> dict[str, Any]:
    repetitions = int(accumulator["repetitions"])
    simultaneous = int(accumulator["simultaneous_successes"])
    errors = int(accumulator["error_events"])
    best = int(accumulator["best_successes"])
    ranks = int(accumulator["rank_successes"])
    empirical_variance = (
        accumulator["point_square_sum"]
        - np.square(accumulator["point_sum"]) / repetitions
    ) / (repetitions - 1)
    empirical_se = np.sqrt(np.maximum(empirical_variance, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        estimated_se_rms = np.sqrt(
            accumulator["se_square_sum"] / accumulator["se_count"]
        )
        ratios = empirical_se / estimated_se_rms
    finite_ratio = np.isfinite(ratios) & (ratios > 0.0)
    finite_width_count = int(accumulator["finite_width_count"])
    interval_count = int(accumulator["interval_count"])
    decision_count = int(accumulator["decision_cell_count"])
    true_singleton = np.sum(
        np.isclose(
            truth,
            np.max(truth, axis=1, keepdims=True),
            rtol=0.0,
            atol=1.0e-14,
        ),
        axis=1,
    ) == 1
    correct_probabilities: dict[str, float | None] = {}
    for index, scheme in enumerate(REFERENCE_SCHEMES):
        correct_probabilities[scheme] = (
            float(
                accumulator["correct_singletons_by_scheme"][index]
                / repetitions
            )
            if true_singleton[index]
            else None
        )
    return {
        "validity": {
            "simultaneous_coverage": simultaneous / repetitions,
            "simultaneous_coverage_lower_95": _binomial_bound(
                simultaneous, repetitions, lower=True
            ),
            "familywise_or_wrong_direction_error": errors / repetitions,
            "familywise_or_wrong_direction_error_upper_95": _binomial_bound(
                errors, repetitions, lower=False
            ),
            "best_model_set_coverage": best / repetitions,
            "best_model_set_coverage_lower_95": _binomial_bound(
                best, repetitions, lower=True
            ),
            "possible_rank_set_coverage": ranks / repetitions,
            "possible_rank_set_coverage_lower_95": _binomial_bound(
                ranks, repetitions, lower=True
            ),
            "false_singleton_winner_probability": (
                int(accumulator["false_singletons"]) / repetitions
            ),
            "standard_error_ratio_min": (
                float(np.min(ratios[finite_ratio]))
                if np.any(finite_ratio)
                else None
            ),
            "standard_error_ratio_max": (
                float(np.max(ratios[finite_ratio]))
                if np.any(finite_ratio)
                else None
            ),
        },
        "informativeness": {
            "mean_finite_interval_width": (
                float(accumulator["finite_width_sum"] / finite_width_count)
                if finite_width_count
                else None
            ),
            "finite_interval_fraction": finite_width_count / interval_count,
            "resolved_contrast_fraction": (
                int(accumulator["resolved_intervals"]) / interval_count
            ),
            "fallback_or_abstention_rate": (
                int(accumulator["fallbacks"]) / repetitions
            ),
            "cycle_abstention_rate": (
                int(accumulator["cycles"]) / repetitions
            ),
            "mean_best_model_set_size": (
                float(accumulator["best_set_size_sum"])
                / (repetitions * len(REFERENCE_SCHEMES))
            ),
            "mean_possible_rank_set_size": (
                float(accumulator["rank_size_sum"]) / decision_count
            ),
            "correct_singleton_probability_by_scheme": correct_probabilities,
            "mean_absolute_cross_scheme_pair_shift": (
                float(
                    accumulator["cross_scheme_shift_sum"]
                    / accumulator["cross_scheme_shift_count"]
                )
                if accumulator["cross_scheme_shift_count"]
                else None
            ),
        },
    }


def qualify_scenario(
    scenario: Scenario,
    *,
    config: Mapping[str, Any],
    repetitions: int,
    seed: int,
    signs: np.ndarray | None,
) -> dict[str, Any]:
    """Run all four procedures for one outcome-free geometry scenario."""

    simulation = config["simulation"]
    alpha = float(simulation["alpha"])
    family = contrast_family(scenario.method_count)
    truth = _truth(scenario, simulation)
    truth_vector = true_contrasts(truth, family)
    accumulators = {
        procedure: _new_accumulator(
            family.size, len(REFERENCE_SCHEMES)
        )
        for procedure in PROCEDURES
    }
    scenario_seed = int.from_bytes(
        hashlib.sha256(
            f"{seed}\0{scenario.scenario_id}".encode("utf-8")
        ).digest()[:8],
        "little",
    )
    rng = np.random.default_rng(scenario_seed)
    completed = 0
    batch_size = int(simulation["batch_size"])
    while completed < repetitions:
        current = min(batch_size, repetitions - completed)
        utilities, support = simulate_root_utilities(
            rng, current, scenario, simulation
        )
        contrasts = root_contrasts(utilities, family)
        results = {
            "NAIVE_CELL_LEVEL": infer_naive_cell_level(
                contrasts,
                support,
                alpha=alpha,
                finite_cell_sd=(
                    0.0
                    if scenario.exact_degenerate
                    else float(
                        simulation["finite_cell_standard_deviation"]
                    )
                ),
            ),
            "ROOT_AWARE_MARGINAL": infer_root_t(
                contrasts, alpha=alpha, bonferroni=False
            ),
            "ROOT_BONFERRONI_T": infer_root_t(
                contrasts, alpha=alpha, bonferroni=True
            ),
            "CURRENT_JOINT_MAXT": infer_current_joint_maxt(
                contrasts,
                signs,
                alpha=alpha,
                minimum_roots=int(
                    simulation["minimum_roots_for_joint_inference"]
                ),
            ),
        }
        for procedure, result in results.items():
            _update_accumulator(
                accumulators[procedure],
                result,
                family=family,
                truth=truth,
                truth_contrasts_vector=truth_vector,
            )
        completed += current

    procedures = {
        procedure: _finalize_accumulator(
            accumulators[procedure], truth=truth
        )
        for procedure in PROCEDURES
    }
    return {
        "scenario_id": scenario.scenario_id,
        "geometry": {
            "root_count": scenario.root_count,
            "method_count": scenario.method_count,
            "reference_scheme_count": len(REFERENCE_SCHEMES),
            "pair_count_per_scheme": family.pair_count_per_scheme,
            "joint_family_size": family.size,
            "support_pattern": scenario.support_pattern,
            "variance_pattern": scenario.variance_pattern,
            "reference_strength": scenario.reference_strength_name,
            "reference_strength_value": scenario.reference_strength,
            "root_law": scenario.root_law,
            "truth_pattern": scenario.truth_pattern,
            "profile_id": scenario.profile_id,
            "exact_degenerate": scenario.exact_degenerate,
        },
        "sign_schedule": {
            "mode": (
                "CONSERVATIVE_ABSTENTION"
                if signs is None
                else (
                    "EXHAUSTIVE"
                    if scenario.root_count in (8, 12)
                    else "FIXED_ANTITHETIC_MONTE_CARLO"
                )
            ),
            "count": 0 if signs is None else int(signs.shape[0]),
            "sha256": sign_schedule_sha256(signs),
        },
        "procedures": procedures,
    }


def _joint_gate_results(
    scenario_result: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, bool]:
    validity = scenario_result["procedures"][PRIMARY_PROCEDURE]["validity"]
    return {
        "simultaneous_coverage": (
            float(validity["simultaneous_coverage_lower_95"])
            >= float(gates["simultaneous_coverage_lower_95_min"])
        ),
        "familywise_or_wrong_direction_error": (
            float(
                validity[
                    "familywise_or_wrong_direction_error_upper_95"
                ]
            )
            <= float(
                gates[
                    "familywise_or_wrong_direction_error_upper_95_max"
                ]
            )
        ),
        "best_model_set_coverage": (
            float(validity["best_model_set_coverage_lower_95"])
            >= float(gates["best_model_set_coverage_lower_95_min"])
        ),
        "possible_rank_set_coverage": (
            float(validity["possible_rank_set_coverage_lower_95"])
            >= float(
                gates["possible_rank_set_coverage_lower_95_min"]
            )
        ),
    }


def _qualify_scenario_worker(
    payload: tuple[Scenario, dict[str, Any], int, int]
) -> dict[str, Any]:
    """Process-pool entry point; each scenario owns a deterministic stream."""

    scenario, config, repetitions, seed = payload
    simulation = config["simulation"]
    signs = sign_schedule(
        scenario.root_count,
        monte_carlo_count=int(simulation["monte_carlo_sign_count"]),
        seed=int(simulation["sign_schedule_seed"]),
    )
    return qualify_scenario(
        scenario,
        config=config,
        repetitions=repetitions,
        seed=seed,
        signs=signs,
    )


def run_framework(
    config: Mapping[str, Any],
    *,
    repetitions: int | None = None,
    seed: int | None = None,
    workers: int = 1,
    self_test: bool = False,
) -> dict[str, Any]:
    """Run the generic outcome-free grid and return a JSON-safe report."""

    frozen = validate_config(config)
    simulation = frozen["simulation"]
    actual_repetitions = (
        int(simulation["repetitions"])
        if repetitions is None
        else int(repetitions)
    )
    actual_seed = int(simulation["seed"]) if seed is None else int(seed)
    actual_workers = int(workers)
    if actual_repetitions < 2 or actual_workers < 1:
        raise GenericFrameworkError(
            "at least two repetitions and one worker are required"
        )
    scenarios = list(iter_scenarios(frozen))
    sign_cache = {
        root_count: sign_schedule(
            root_count,
            monte_carlo_count=int(simulation["monte_carlo_sign_count"]),
            seed=int(simulation["sign_schedule_seed"]),
        )
        for root_count in frozen["grid"]["root_counts"]
    }
    payloads = [
        (scenario, frozen, actual_repetitions, actual_seed)
        for scenario in scenarios
    ]
    if actual_workers == 1:
        results = [
            _qualify_scenario_worker(payload) for payload in payloads
        ]
    else:
        # executor.map preserves the preregistered scenario order.  Scenario
        # seeds depend only on the master seed and scenario ID, so worker count
        # and completion order cannot change simulation draws.
        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            results = list(executor.map(_qualify_scenario_worker, payloads))

    all_gates_pass = True
    for result in results:
        gates = _joint_gate_results(result, frozen["validity_gates"])
        result["joint_validity_gates"] = gates
        result["joint_validity_pass"] = all(gates.values())
        all_gates_pass &= result["joint_validity_pass"]

    if self_test:
        status = "PASS_GENERIC_FRAMEWORK_V1_IMPLEMENTATION_SELF_TEST"
        scientific_qualification = False
        scientific_exit_code = 0
    elif all_gates_pass:
        status = "PASS_GENERIC_FRAMEWORK_V1_OUTCOME_FREE_VALIDITY"
        scientific_qualification = True
        scientific_exit_code = 0
    else:
        status = "NO_GO_GENERIC_FRAMEWORK_V1_OUTCOME_FREE_VALIDITY"
        scientific_qualification = True
        scientific_exit_code = 2
    return {
        "record_type": "GENERIC_REFERENCE_FRAMEWORK_V1_QUALIFICATION",
        "schema_version": SCHEMA_VERSION,
        "framework_id": FRAMEWORK_ID,
        "status": status,
        "scientific_exit_code": scientific_exit_code,
        "scientific_qualification": scientific_qualification,
        "outcome_free": True,
        "access_boundary": {
            "configuration_json_read": True,
            "dataset_opened": False,
            "expression_or_counts_opened": False,
            "predictions_opened": False,
            "scores_winners_or_ranks_opened": False,
        },
        "separation_of_evidence": {
            "hard_validity_dimensions": [
                "simultaneous_coverage",
                "familywise_or_wrong_direction_error",
                "best_model_set_coverage",
                "possible_rank_set_coverage",
            ],
            "informativeness_dimensions": [
                "interval_width",
                "resolved_contrast_fraction",
                "fallback_or_abstention_rate",
                "best_model_set_size",
                "possible_rank_set_size",
                "singleton_winner_probability",
            ],
            "conservative_fallback_rule": (
                "UNBOUNDED_INTERVALS_FULL_BEST_SET_FULL_RANK_SET; "
                "COUNTS_AGAINST_INFORMATIVENESS_ONLY"
            ),
        },
        "procedures": {
            "mandatory_comparators": [
                "NAIVE_CELL_LEVEL",
                "ROOT_AWARE_MARGINAL",
            ],
            "standard_sensitivity": "ROOT_BONFERRONI_T",
            "primary_proposed": PRIMARY_PROCEDURE,
            "primary_family": "ALL_WITHIN_SCHEME_METHOD_PAIRS_JOINTLY",
            "cross_scheme_reference_effects": (
                "DESCRIPTIVE_ONLY_NOT_A_PRIMARY_VALIDITY_GATE"
            ),
        },
        "execution": {
            "repetitions_per_scenario": actual_repetitions,
            "seed": actual_seed,
            "workers": actual_workers,
            "scenario_count": len(scenarios),
            "reference_schemes": list(REFERENCE_SCHEMES),
        },
        "sign_schedules": {
            str(root_count): {
                "mode": (
                    "CONSERVATIVE_ABSTENTION"
                    if signs is None
                    else (
                        "EXHAUSTIVE"
                        if root_count in (8, 12)
                        else "FIXED_ANTITHETIC_MONTE_CARLO"
                    )
                ),
                "count": 0 if signs is None else int(signs.shape[0]),
                "sha256": sign_schedule_sha256(signs),
            }
            for root_count, signs in sign_cache.items()
        },
        "validity_gates": dict(frozen["validity_gates"]),
        "all_joint_validity_gates_pass": all_gates_pass,
        "scenarios": results,
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GenericFrameworkError("configuration JSON root is not an object")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        config = self_test_config() if args.config is None else _load_json(args.config)
    else:
        if args.config is None:
            parser.error("--config is required for a formal qualification")
        if args.output is None:
            parser.error("--output is required for a formal qualification")
        config = _load_json(args.config)
    report = run_framework(
        config,
        repetitions=args.repetitions,
        seed=args.seed,
        workers=(
            args.workers
            if args.workers is not None
            else (1 if args.self_test else 52)
        ),
        self_test=args.self_test,
    )
    if args.output is None:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    else:
        _write_json_atomic(args.output, report)
        print(args.output)
    return int(report["scientific_exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
