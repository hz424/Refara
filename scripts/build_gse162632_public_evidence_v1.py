#!/usr/bin/env python3
"""Build and verify the public GSE162632 held-out evidence tables.

The distributed root utilities are algebraic reconstructions from the frozen
leave-one-root point summaries that already drive Supplementary Figure 2:

    u_r = R * U - (R - 1) * U_{-r},  R = 8.

The full 84-contrast table is then recomputed from those distributed root
utilities with the outcome-independent root-family and generic inference
implementations. Bonferroni-t intervals and Holm-adjusted marginal
t-test p-values are included as conventional working-law sensitivities.

This program never reads expression matrices, cell-level outcomes,
predictions, checkpoints or the non-redistributed empirical score bundle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import t


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from perturb_nuisance_contracts.source_data_root import (  # noqa: E402
    SourceDataRootError,
    SourceDataSnapshot,
    validate_source_data_root,
)

SOURCE_LOO_RUNTIME_PATH = Path(
    "data/derived/supplementary_frozen_source/"
    "gse162632_leave_one_root_point_stability.tsv"
)
SELECTED_INTERVAL_ANCHOR_RUNTIME_PATH = Path(
    "data/derived/figure3/"
    "GSE162632_HELD_OUT_PCA64_VS_SCGEN_SIMULTANEOUS_INTERVALS_V1.tsv"
)
SOURCE_LOO = REPO / SOURCE_LOO_RUNTIME_PATH
SELECTED_INTERVAL_ANCHOR = REPO / SELECTED_INTERVAL_ANCHOR_RUNTIME_PATH
DEFAULT_OUTPUT_DIR = REPO / "data/derived/public_held_out_evidence"
ROOT_INFERENCE_SOURCE = (
    REPO / "src/perturb_nuisance_contracts/root_family_inference.py"
)
GENERIC_SOURCE = REPO / "scripts/qualify_generic_reference_framework_v1.py"
SOURCE_DATA_SNAPSHOT: SourceDataSnapshot | None = None

ROOT_TABLE_NAME = "GSE162632_ROOT_UTILITIES_PUBLIC_V1.tsv"
CONTRAST_TABLE_NAME = "GSE162632_FULL_84_CONTRASTS_PUBLIC_V1.tsv"
MANIFEST_NAME = "GSE162632_PUBLIC_EVIDENCE_MANIFEST_V1.json"

SOURCE_LOO_SHA256 = (
    "5091a2f5351e9b3d020421d0c75abbe47b7f9fb510a0aa921904e1d2d277659d"
)
SELECTED_INTERVAL_ANCHOR_SHA256 = (
    "cd5701732cf66239e5ae2c3c7c6d547ddaab779ea78fbef57d2a18380d1e12b8"
)
GENERIC_SOURCE_SHA256 = (
    "976b3d82c24b5153ac8b3499ec4d3d313d0853541c4f43d850acca38bfde496c"
)
SOURCE_REPORT_SHA256 = (
    "1a297a99c03985493aa4156eaa89fafe0cbd89a32950745399ef427bd9ae923c"
)
FROZEN_GENERATOR_SOURCE_SHA256 = (
    "618a103db9548b0e1564c27f8ddb87ba391d7cd2d778e079c722732459bec633"
)
# High-precision inversion of the df=7 Student-t survival function at
# 0.05/(2*84).  SciPy releases can differ by a few ulps in this extreme
# quantile, so the public serialization uses this fixed mathematical value
# after checking that the installed generic implementation agrees closely.
BONFERRONI_T_CRITICAL = 5.907038874082924

DATASET_ID = "GSE162632_FIXED_IAV_6H"
DATASET_ROLE = "FORMAL_HELD_OUT_WITHIN_STUDY_CONDITIONAL_APPLICATION"
UTILITY_METRIC = "oriented_standardized_squared_perturbation_effect_error"
ROOT_ANALYSIS_ROLE = (
    "ALGEBRAIC_RECONSTRUCTION_FROM_FROZEN_LEAVE_ONE_ROOT_POINTS__"
    "NOT_NEW_INFERENCE"
)
EVIDENCE_ROLE = "HELD_OUT_FIXED_BEFORE_OUTCOME_INSPECTION_CONDITIONAL"

ROOT_HEADER = (
    "dataset_id",
    "dataset_role",
    "source_report_sha256",
    "public_root_code",
    "root_order",
    "scheme_id",
    "scheme_order",
    "method_id",
    "panel_order",
    "root_count",
    "task_count",
    "root_utility",
    "utility_metric",
    "larger_is_better",
    "analysis_role",
)

CONTRAST_HEADER = (
    "dataset_id",
    "dataset_role",
    "source_report_sha256",
    "contrast_index",
    "contrast_id",
    "scheme_id",
    "scheme_order",
    "method_a",
    "method_a_order",
    "method_b",
    "method_b_order",
    "point_difference",
    "root_standard_error",
    "max_t_lower",
    "max_t_upper",
    "max_t_resolved_direction",
    "max_t_critical_value",
    "max_t_alpha",
    "max_t_family_size",
    "max_t_sign_count",
    "max_t_sign_schedule_sha256",
    "max_t_fallback",
    "bonferroni_t_lower",
    "bonferroni_t_upper",
    "bonferroni_t_resolved_direction",
    "bonferroni_t_critical_value",
    "bonferroni_t_alpha",
    "bonferroni_t_family_size",
    "bonferroni_t_df",
    "marginal_t_statistic",
    "marginal_t_two_sided_p_value",
    "holm_adjusted_two_sided_p_value",
    "holm_reject_0_05",
    "holm_resolved_direction",
    "evidence_role",
)

LOO_HEADER = (
    "dataset_id",
    "source_report_sha256",
    "held_out_root_id",
    "held_out_root_order",
    "scheme_id",
    "scheme_order",
    "method_id",
    "panel_order",
    "leave_one_root_point_utility",
    "leave_one_root_point_rank",
    "analysis_role",
)


class PublicEvidenceError(RuntimeError):
    """Raised when a public evidence input or result differs."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicEvidenceError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, module_name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(module_name, path)
    require(
        specification is not None and specification.loader is not None,
        f"cannot import {path}",
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def load_implementations() -> tuple[ModuleType, ModuleType]:
    require(
        sha256_file(GENERIC_SOURCE) == GENERIC_SOURCE_SHA256,
        "generic inference source SHA-256 differs",
    )
    root_inference = load_module(
        ROOT_INFERENCE_SOURCE,
        "_public_root_family_inference_v1",
    )
    generic = load_module(GENERIC_SOURCE, "_public_gse162632_generic_v1")
    require(
        root_inference.ROOTS == 8
        and root_inference.TASKS == 5
        and root_inference.METHODS == 8
        and root_inference.CONTRASTS == 84
        and root_inference.ALPHA == 0.05
        and root_inference.SIGN_COUNT == 256
        and tuple(root_inference.ROOT_IDS)
        == tuple(f"ROOT_{index:02d}" for index in range(1, 9))
        and callable(root_inference.analyse_root_utilities),
        "root-family inference geometry or API differs",
    )
    for name in (
        "contrast_family",
        "root_contrasts",
        "sign_schedule",
        "sign_schedule_sha256",
        "infer_current_joint_maxt",
        "infer_root_t",
    ):
        require(callable(getattr(generic, name, None)), f"generic API missing: {name}")
    return root_inference, generic


def read_tsv(path: Path, expected_header: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require(tuple(reader.fieldnames or ()) == expected_header, f"header differs: {path}")
        rows = list(reader)
    require(rows, f"table is empty: {path}")
    return rows


def float_text(value: float) -> str:
    number = float(value)
    require(math.isfinite(number), "nonfinite public numeric value")
    return format(number, ".17g")


def p_value_text(value: float) -> str:
    number = float(value)
    require(math.isfinite(number) and 0.0 <= number <= 1.0, "invalid public p-value")
    # Twelve significant digits are far beyond the inferential resolution of
    # eight roots and remove irrelevant special-function ulp differences
    # across supported SciPy releases.
    return format(number, ".12g")


def bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def direction(lower: float, upper: float) -> str:
    return "A_GREATER" if lower > 0.0 else "B_GREATER" if upper < 0.0 else "UNRESOLVED"


def direction_from_rejection(estimate: float, rejected: bool) -> str:
    if not rejected:
        return "UNRESOLVED"
    require(estimate != 0.0, "rejected zero estimate has no direction")
    return "A_GREATER" if estimate > 0.0 else "B_GREATER"


def render_tsv(rows: Sequence[Mapping[str, Any]], header: tuple[str, ...]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=header,
        delimiter="\t",
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    for row in rows:
        require(tuple(row) == header, "row fields or field order differ")
        writer.writerow(row)
    return output.getvalue()


def build_root_rows(root_inference: ModuleType) -> list[dict[str, str]]:
    require(sha256_file(SOURCE_LOO) == SOURCE_LOO_SHA256, "frozen LOO source differs")
    source = read_tsv(SOURCE_LOO, LOO_HEADER)
    require(len(source) == 192, "frozen LOO source must contain 192 rows")

    roots = tuple(root_inference.ROOT_IDS)
    schemes = tuple(root_inference.SCHEMES)
    methods = tuple(root_inference.METHOD_IDS)
    expected = {
        (root_index, scheme_index, method_index)
        for root_index in range(root_inference.ROOTS)
        for scheme_index in range(len(schemes))
        for method_index in range(root_inference.METHODS)
    }
    observed: dict[tuple[int, int, int], float] = {}
    for row in source:
        root_index = int(row["held_out_root_order"]) - 1
        scheme_index = int(row["scheme_order"]) - 1
        method_index = int(row["panel_order"]) - 1
        key = (root_index, scheme_index, method_index)
        require(key in expected and key not in observed, "LOO key is invalid or duplicated")
        require(
            row["dataset_id"] == DATASET_ID
            and row["source_report_sha256"] == SOURCE_REPORT_SHA256
            and row["held_out_root_id"] == roots[root_index]
            and row["scheme_id"] == schemes[scheme_index]
            and row["method_id"] == methods[method_index]
            and 1 <= int(row["leave_one_root_point_rank"]) <= root_inference.METHODS
            and row["analysis_role"] == "DESCRIPTIVE_ROOT_INFLUENCE_NOT_NEW_INFERENCE",
            "LOO identity, order or role differs",
        )
        value = float(row["leave_one_root_point_utility"])
        require(math.isfinite(value), "LOO point utility is nonfinite")
        observed[key] = value
    require(set(observed) == expected, "LOO family is incomplete")

    loo = np.empty(
        (root_inference.ROOTS, len(schemes), root_inference.METHODS),
        dtype=np.float64,
    )
    for key, value in observed.items():
        loo[key] = value
    full = np.mean(loo, axis=0, dtype=np.float64)
    root_utilities = (
        root_inference.ROOTS * full[None, :, :]
        - (root_inference.ROOTS - 1) * loo
    )
    require(np.isfinite(root_utilities).all(), "reconstructed root utilities are nonfinite")

    inverse = (
        np.sum(root_utilities, axis=0, dtype=np.float64)[None, :, :]
        - root_utilities
    ) / (root_inference.ROOTS - 1)
    require(
        float(np.max(np.abs(inverse - loo))) <= 2.0e-15,
        "root reconstruction does not reproduce every LOO point",
    )

    rows: list[dict[str, str]] = []
    for root_index, root_id in enumerate(roots):
        for scheme_index, scheme in enumerate(schemes):
            for method_index, method in enumerate(methods):
                rows.append(
                    {
                        "dataset_id": DATASET_ID,
                        "dataset_role": DATASET_ROLE,
                        "source_report_sha256": SOURCE_REPORT_SHA256,
                        "public_root_code": root_id,
                        "root_order": str(root_index + 1),
                        "scheme_id": scheme,
                        "scheme_order": str(scheme_index + 1),
                        "method_id": method,
                        "panel_order": str(method_index + 1),
                        "root_count": str(root_inference.ROOTS),
                        "task_count": str(root_inference.TASKS),
                        "root_utility": float_text(
                            root_utilities[root_index, scheme_index, method_index]
                        ),
                        "utility_metric": UTILITY_METRIC,
                        "larger_is_better": "true",
                        "analysis_role": ROOT_ANALYSIS_ROLE,
                    }
                )
    require(len(rows) == 192, "public root table geometry differs")
    return rows


def root_array_from_rows(
    rows: Sequence[Mapping[str, str]],
    root_inference: ModuleType,
) -> np.ndarray:
    require(len(rows) == 192, "public root row count differs")
    roots = tuple(root_inference.ROOT_IDS)
    schemes = tuple(root_inference.SCHEMES)
    methods = tuple(root_inference.METHOD_IDS)
    values = np.full(
        (root_inference.ROOTS, len(schemes), root_inference.METHODS),
        np.nan,
        dtype=np.float64,
    )
    seen: set[tuple[int, int, int]] = set()
    for row in rows:
        root_index = int(row["root_order"]) - 1
        scheme_index = int(row["scheme_order"]) - 1
        method_index = int(row["panel_order"]) - 1
        key = (root_index, scheme_index, method_index)
        require(
            0 <= root_index < root_inference.ROOTS
            and 0 <= scheme_index < len(schemes)
            and 0 <= method_index < root_inference.METHODS
            and key not in seen,
            "public root key is invalid or duplicated",
        )
        require(
            row["dataset_id"] == DATASET_ID
            and row["dataset_role"] == DATASET_ROLE
            and row["source_report_sha256"] == SOURCE_REPORT_SHA256
            and row["public_root_code"] == roots[root_index]
            and row["scheme_id"] == schemes[scheme_index]
            and row["method_id"] == methods[method_index]
            and row["root_count"] == str(root_inference.ROOTS)
            and row["task_count"] == str(root_inference.TASKS)
            and row["utility_metric"] == UTILITY_METRIC
            and row["larger_is_better"] == "true"
            and row["analysis_role"] == ROOT_ANALYSIS_ROLE,
            "public root metadata differs",
        )
        values[key] = float(row["root_utility"])
        seen.add(key)
    require(len(seen) == 192 and np.isfinite(values).all(), "public root tensor is incomplete")
    return values


def holm_adjusted(raw_p: np.ndarray) -> np.ndarray:
    values = np.asarray(raw_p, dtype=np.float64)
    require(
        values.shape == (84,)
        and np.all((values >= 0.0) & (values <= 1.0)),
        "Holm input differs",
    )
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (values.size - rank) * float(values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted


def build_contrast_rows(
    root_utilities: np.ndarray,
    root_inference: ModuleType,
    generic: ModuleType,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    values = np.asarray(root_utilities, dtype=np.float64)
    require(
        values.shape
        == (
            root_inference.ROOTS,
            len(root_inference.SCHEMES),
            root_inference.METHODS,
        )
        and np.isfinite(values).all(),
        "public root utility tensor differs",
    )
    analysis = root_inference.analyse_root_utilities(values, generic)
    require(
        analysis["fallback"] is False
        and len(analysis["pairwise_interval_records"])
        == root_inference.CONTRASTS,
        "current max-T analysis fell back or changed geometry",
    )

    family = generic.contrast_family(root_inference.METHODS)
    root_contrasts = generic.root_contrasts(values[None, ...], family)
    bonferroni = generic.infer_root_t(
        root_contrasts,
        alpha=root_inference.ALPHA,
        bonferroni=True,
    )
    require(not bool(bonferroni["fallback"][0]), "Bonferroni-t sensitivity fell back")
    point = np.asarray(bonferroni["point"][0], dtype=np.float64)
    standard_error = np.asarray(
        bonferroni["standard_errors"][0], dtype=np.float64
    )
    t_statistic = point / standard_error
    raw_p = 2.0 * t.sf(
        np.abs(t_statistic),
        df=root_inference.ROOTS - 1,
    )
    holm_p = holm_adjusted(raw_p)

    signs = generic.sign_schedule(
        root_inference.ROOTS,
        monte_carlo_count=4_096,
        seed=root_inference.SIGN_SEED,
    )
    require(
        signs is not None
        and signs.shape
        == (root_inference.SIGN_COUNT, root_inference.ROOTS)
        and generic.sign_schedule_sha256(signs)
        == root_inference.SIGN_SCHEDULE_SHA256,
        "exhaustive R8 sign schedule differs",
    )
    max_t_critical = float(analysis["critical_value"])
    observed_bonferroni_critical = float(bonferroni["critical_values"][0])
    require(
        math.isclose(
            observed_bonferroni_critical,
            BONFERRONI_T_CRITICAL,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ),
        "generic Bonferroni-t critical value differs materially",
    )
    bonferroni_critical = BONFERRONI_T_CRITICAL

    rows: list[dict[str, str]] = []
    for index, ((scheme_index, left, right), record) in enumerate(
        zip(family.identities, analysis["pairwise_interval_records"])
    ):
        require(
            record["contrast_index"] == index
            and record["scheme"] == root_inference.SCHEMES[scheme_index]
            and record["method_a"] == root_inference.METHOD_IDS[left]
            and record["method_b"] == root_inference.METHOD_IDS[right]
            and math.isclose(
                float(record["point_difference"]),
                float(point[index]),
                rel_tol=0.0,
                abs_tol=2.0e-15,
            )
            and math.isclose(
                float(record["standard_error"]),
                float(standard_error[index]),
                rel_tol=0.0,
                abs_tol=2.0e-15,
            ),
            "max-T and Bonferroni contrast identities or moments differ",
        )
        max_lower = float(record["lower"])
        max_upper = float(record["upper"])
        bonf_lower = float(point[index] - bonferroni_critical * standard_error[index])
        bonf_upper = float(point[index] + bonferroni_critical * standard_error[index])
        rejected = bool(holm_p[index] <= root_inference.ALPHA)
        rows.append(
            {
                "dataset_id": DATASET_ID,
                "dataset_role": DATASET_ROLE,
                "source_report_sha256": SOURCE_REPORT_SHA256,
                "contrast_index": str(index),
                "contrast_id": record["contrast_id"],
                "scheme_id": root_inference.SCHEMES[scheme_index],
                "scheme_order": str(scheme_index + 1),
                "method_a": root_inference.METHOD_IDS[left],
                "method_a_order": str(left + 1),
                "method_b": root_inference.METHOD_IDS[right],
                "method_b_order": str(right + 1),
                "point_difference": float_text(point[index]),
                "root_standard_error": float_text(standard_error[index]),
                "max_t_lower": float_text(max_lower),
                "max_t_upper": float_text(max_upper),
                "max_t_resolved_direction": direction(max_lower, max_upper),
                "max_t_critical_value": float_text(max_t_critical),
                "max_t_alpha": "0.05",
                "max_t_family_size": str(root_inference.CONTRASTS),
                "max_t_sign_count": str(root_inference.SIGN_COUNT),
                "max_t_sign_schedule_sha256": (
                    root_inference.SIGN_SCHEDULE_SHA256
                ),
                "max_t_fallback": "false",
                "bonferroni_t_lower": float_text(bonf_lower),
                "bonferroni_t_upper": float_text(bonf_upper),
                "bonferroni_t_resolved_direction": direction(bonf_lower, bonf_upper),
                "bonferroni_t_critical_value": float_text(bonferroni_critical),
                "bonferroni_t_alpha": "0.05",
                "bonferroni_t_family_size": str(root_inference.CONTRASTS),
                "bonferroni_t_df": str(root_inference.ROOTS - 1),
                "marginal_t_statistic": float_text(t_statistic[index]),
                "marginal_t_two_sided_p_value": p_value_text(raw_p[index]),
                "holm_adjusted_two_sided_p_value": p_value_text(holm_p[index]),
                "holm_reject_0_05": bool_text(rejected),
                "holm_resolved_direction": direction_from_rejection(point[index], rejected),
                "evidence_role": EVIDENCE_ROLE,
            }
        )
    require(len(rows) == 84, "public full-family table geometry differs")

    max_directions = [row["max_t_resolved_direction"] for row in rows]
    bonf_directions = [row["bonferroni_t_resolved_direction"] for row in rows]
    holm_directions = [row["holm_resolved_direction"] for row in rows]
    require(
        sum(value != "UNRESOLVED" for value in max_directions) == 68
        and sum(value != "UNRESOLVED" for value in bonf_directions) == 52
        and holm_directions == max_directions
        and not any(
            max_direction != "UNRESOLVED"
            and bonf_direction != "UNRESOLVED"
            and max_direction != bonf_direction
            for max_direction, bonf_direction in zip(max_directions, bonf_directions)
        ),
        "full-family sensitivity resolution differs",
    )
    summary = {
        "max_t_critical_value": max_t_critical,
        "bonferroni_t_critical_value": bonferroni_critical,
        "max_t_resolved": 68,
        "bonferroni_t_resolved": 52,
        "holm_resolved": 68,
        "max_t_bonferroni_opposite_directions": 0,
    }
    return rows, summary


def validate_selected_anchor(rows: Sequence[Mapping[str, str]]) -> None:
    require(
        sha256_file(SELECTED_INTERVAL_ANCHOR) == SELECTED_INTERVAL_ANCHOR_SHA256,
        "selected PCA64-scGen interval anchor differs",
    )
    with SELECTED_INTERVAL_ANCHOR.open("r", encoding="utf-8", newline="") as handle:
        selected = list(csv.DictReader(handle, delimiter="\t"))
    require(len(selected) == 3, "selected interval anchor geometry differs")
    index = {
        (row["scheme_id"], row["method_a"], row["method_b"]): row for row in rows
    }
    for anchor in selected:
        key = (anchor["scheme_id"], anchor["method_a"], anchor["method_b"])
        require(key in index, "selected interval is absent from the full family")
        full = index[key]
        require(
            anchor["resolved_direction"] == full["max_t_resolved_direction"]
            and anchor["source_report_sha256"] == full["source_report_sha256"],
            "selected interval direction or provenance differs",
        )
        for selected_field, full_field in (
            ("point_difference", "point_difference"),
            ("lower", "max_t_lower"),
            ("upper", "max_t_upper"),
        ):
            require(
                math.isclose(
                    float(anchor[selected_field]),
                    float(full[full_field]),
                    rel_tol=0.0,
                    abs_tol=3.0e-15,
                ),
                f"selected interval numeric field differs: {selected_field}",
            )


def manifest_text(
    root_text: str,
    contrast_text: str,
    summary: Mapping[str, Any],
) -> str:
    manifest = {
        "schema_version": "1.0.0",
        "record_type": "GSE162632_PUBLIC_HELD_OUT_EVIDENCE_V1",
        "status": "PASS_SELF_CONTAINED_ROOT_TO_FULL_84_RECALCULATION",
        "dataset_id": DATASET_ID,
        "scientific_scope": {
            "estimand": (
                "equal-root conditional method-utility differences over the five "
                "specified tasks and eight realized donor-processing-batch roots"
            ),
            "working_law": (
                "equal-root root-level inference conditional on the eight realized roots"
            ),
            "population_generalization_allowed": False,
            "cross_scheme_interaction_in_family": False,
        },
        "geometry": {
            "roots": 8,
            "tasks": 5,
            "methods": 8,
            "schemes": 3,
            "within_scheme_pairs": 28,
            "joint_contrasts": 84,
        },
        "inference": {
            "max_t": {
                "definition": (
                    "recentered and re-studentized exhaustive root sign-flip "
                    "max-absolute-T over all 84 within-scheme contrasts"
                ),
                "alpha": 0.05,
                "sign_count": 256,
                "critical_order_one_based": 244,
                "critical_value": summary["max_t_critical_value"],
                "resolved_contrasts": summary["max_t_resolved"],
            },
            "bonferroni_t": {
                "definition": (
                    "two-sided root t intervals using alpha/(2*84) in each tail"
                ),
                "alpha": 0.05,
                "degrees_of_freedom": 7,
                "critical_value": summary["bonferroni_t_critical_value"],
                "resolved_contrasts": summary["bonferroni_t_resolved"],
            },
            "holm": {
                "definition": (
                    "Holm step-down adjustment of 84 two-sided marginal root t-test "
                    "p-values with stable contrast-index tie breaking"
                ),
                "alpha": 0.05,
                "degrees_of_freedom": 7,
                "resolved_contrasts": summary["holm_resolved"],
            },
        },
        "privacy_and_access_boundary": {
            "root_identifiers_anonymous": False,
            "root_identifiers_pseudonymized": True,
            "root_identifier_mapping_included": False,
            "public_root_code_column": "public_root_code",
            "identifier_description": (
                "release-local ROOT_01 through ROOT_08 codes preserve only the "
                "within-release root order; no study-identifier mapping is included"
            ),
            "cell_level_values_included": False,
            "expression_or_prediction_arrays_included": False,
            "raw_to_prediction_replay_from_public_checkout": False,
            "root_to_84_contrast_recalculation_from_public_checkout": True,
        },
        "source_bindings": {
            "frozen_leave_one_root_table": {
                "path": SOURCE_LOO_RUNTIME_PATH.as_posix(),
                "sha256": SOURCE_LOO_SHA256,
            },
            "selected_interval_anchor": {
                "path": SELECTED_INTERVAL_ANCHOR_RUNTIME_PATH.as_posix(),
                "sha256": SELECTED_INTERVAL_ANCHOR_SHA256,
            },
            "source_report_sha256": SOURCE_REPORT_SHA256,
            "root_family_inference_source_sha256": sha256_file(
                ROOT_INFERENCE_SOURCE
            ),
            "generic_source_sha256": GENERIC_SOURCE_SHA256,
            # Preserve the identity of the generator that created the frozen
            # release artifact. The v1.6 wrapper identity is bound separately
            # by the top-level replay receipt.
            "generator_source_sha256": FROZEN_GENERATOR_SOURCE_SHA256,
        },
        "artifacts": {
            ROOT_TABLE_NAME: {
                "rows": 192,
                "sha256": sha256_bytes(root_text.encode("utf-8")),
            },
            CONTRAST_TABLE_NAME: {
                "rows": 84,
                "sha256": sha256_bytes(contrast_text.encode("utf-8")),
            },
        },
    }
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def materialize() -> tuple[dict[str, str], dict[str, Any]]:
    root_inference, generic = load_implementations()
    root_rows = build_root_rows(root_inference)
    root_text = render_tsv(root_rows, ROOT_HEADER)

    # Round-trip through the public serialization before forming contrasts.
    # This makes the 84-row table exactly reproducible from the distributed
    # root table without relying on unreported higher-precision values.
    root_reader = csv.DictReader(io.StringIO(root_text), delimiter="\t")
    require(tuple(root_reader.fieldnames or ()) == ROOT_HEADER, "rendered root header differs")
    serialized_root_rows = list(root_reader)
    root_values = root_array_from_rows(serialized_root_rows, root_inference)
    contrast_rows, summary = build_contrast_rows(
        root_values,
        root_inference,
        generic,
    )
    validate_selected_anchor(contrast_rows)
    contrast_text = render_tsv(contrast_rows, CONTRAST_HEADER)
    outputs = {
        ROOT_TABLE_NAME: root_text,
        CONTRAST_TABLE_NAME: contrast_text,
    }
    outputs[MANIFEST_NAME] = manifest_text(root_text, contrast_text, summary)
    return outputs, summary


def configure_source_data_root(source_data_root: Path) -> SourceDataSnapshot:
    """Bind empirical inputs to a manifest-verified external Source Data tree."""

    global SOURCE_LOO, SELECTED_INTERVAL_ANCHOR, DEFAULT_OUTPUT_DIR
    global SOURCE_DATA_SNAPSHOT
    snapshot = validate_source_data_root(source_data_root)
    derived = snapshot.root / "data/derived"
    SOURCE_LOO = snapshot.root / SOURCE_LOO_RUNTIME_PATH
    SELECTED_INTERVAL_ANCHOR = snapshot.root / SELECTED_INTERVAL_ANCHOR_RUNTIME_PATH
    DEFAULT_OUTPUT_DIR = derived / "public_held_out_evidence"
    SOURCE_DATA_SNAPSHOT = snapshot
    return snapshot


def check_release(output_dir: Path | None = None) -> dict[str, Any]:
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    expected, summary = materialize()
    for name, text in expected.items():
        path = output_dir / name
        require(path.is_file() and not path.is_symlink(), f"public artifact missing: {path}")
        require(path.read_text(encoding="utf-8") == text, f"public artifact differs: {path}")

    # Independently parse the checked-in root table and recompute every
    # checked-in contrast field from it.
    root_inference, generic = load_implementations()
    checked_root_rows = read_tsv(output_dir / ROOT_TABLE_NAME, ROOT_HEADER)
    checked_values = root_array_from_rows(checked_root_rows, root_inference)
    checked_contrasts, _checked_summary = build_contrast_rows(
        checked_values,
        root_inference,
        generic,
    )
    validate_selected_anchor(checked_contrasts)
    recomputed_contrast_text = render_tsv(checked_contrasts, CONTRAST_HEADER)
    require(
        (output_dir / CONTRAST_TABLE_NAME).read_text(encoding="utf-8")
        == recomputed_contrast_text,
        "full 84-contrast table is not exactly reproducible from public roots",
    )
    return {
        "status": "PASS_EXACT_PUBLIC_ROOT_TO_FULL_84_VERIFICATION",
        "root_rows": 192,
        "contrast_rows": 84,
        **summary,
    }


def write_release(output_dir: Path) -> dict[str, Any]:
    outputs, summary = materialize()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in outputs:
        path = output_dir / name
        require(not path.exists() and not path.is_symlink(), f"refusing to overwrite: {path}")
    for name, text in outputs.items():
        (output_dir / name).write_text(text, encoding="utf-8", newline="")
    return {
        "status": "PASS_WROTE_PUBLIC_HELD_OUT_EVIDENCE",
        "output_dir": str(output_dir.resolve()),
        "root_rows": 192,
        "contrast_rows": 84,
        **summary,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the public GSE162632 held-out evidence tables or verify "
            "supplied copies byte for byte."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        action="store_true",
        help="recompute the supplied files and verify them byte for byte",
    )
    mode.add_argument(
        "--output-dir",
        type=Path,
        help="write a new release into an empty target directory",
    )
    parser.add_argument(
        "--source-data-root",
        type=Path,
        required=True,
        help="extracted 05_SOURCE_DATA directory matching the bundled manifest",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    snapshot = configure_source_data_root(args.source_data_root)
    if args.output_dir is not None:
        if args.output_dir.expanduser().is_symlink():
            raise PublicEvidenceError("output directory cannot be a symlink")
        output = args.output_dir.expanduser().resolve()
        if output == snapshot.root or snapshot.root in output.parents:
            raise PublicEvidenceError("output directory cannot be inside Source Data root")
    report = check_release() if args.check else write_release(output)
    snapshot.assert_unchanged()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PublicEvidenceError, SourceDataRootError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
