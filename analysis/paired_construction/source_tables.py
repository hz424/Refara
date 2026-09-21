"""Validate the retained 56 paired changes and 448 root contrasts.

Numerical functions and constants extracted unchanged from the historical renderer.
"""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
from typing import Any
import numpy as np

PAIRED_ROOT_TABLE = "GSE162632_PAIRED_CONSTRUCTION_CHANGES_ROOTS_PUBLIC_V1.tsv"
PAIRED_SUMMARY_TABLE = "GSE162632_PAIRED_CONSTRUCTION_CHANGES_FULL_56_PUBLIC_V1.tsv"
PAIRED_MANIFEST = "GSE162632_PAIRED_CONSTRUCTION_CHANGES_MANIFEST_V1.json"
PAIRED_ROOT_TABLE_SHA256 = "afafe030a28c46fda828cb34da5ecd318db452d028d021cc75b4e1bb3ccc2ff4"
PAIRED_SUMMARY_TABLE_SHA256 = "b26caa9829f74c633a155237a232b8684f4be16c483248ed1d9085d389162050"
PAIRED_MANIFEST_SHA256 = "0ab513f3b8a9db70321d7295effee07f1be88747d7395fe070e264836b7324cc"
SOURCE_REPORT_SHA256 = "1a297a99c03985493aa4156eaa89fafe0cbd89a32950745399ef427bd9ae923c"
PCA64_METHOD_ID = "PCA64_ADDITIVE_RIDGE_DIRECT_V1"
SCGEN_METHOD_ID = "SCGEN_2_1_1_ABSOLUTE_V1"

PAIRED_TARGET_ORDER = ("INDEPENDENT_SPLIT", "UNIT_AWARE_CROSSFIT")
PAIRED_DATASET_ID = "GSE162632_FIXED_IAV_6H"
PAIRED_DATASET_ROLE = "POST_HOC_HELD_OUT_WITHIN_STUDY_PAIRED_CONSTRUCTION_CHANGE"
PAIRED_CHANGE_DEFINITION = "TARGET_MINUS_SOURCE_OF_METHOD_A_MINUS_METHOD_B"
PAIRED_ANALYSIS_TIMING = "POST_HOC_AFTER_OUTCOME_ACCESS"
PAIRED_EVIDENCE_ROLE = (
    "POST_HOC_PAIRED_COMPOSITE_CONSTRUCTION_CHANGE_WORKING_SENSITIVITY"
)
PAIRED_UTILITY_METRIC = "oriented_standardized_squared_perturbation_effect_error"
PAIRED_SIGN_SCHEDULE_SHA256 = (
    "12d6e0393ce9981498e8d9f37f0a04b3a7ce50584c200d9bc99658a26c98f3b7"
)
PAIRED_MAX_T_CRITICAL_VALUE = 5.000133078920623

PAIRED_ROOT_HEADER = (
    "dataset_id",
    "dataset_role",
    "source_report_sha256",
    "paired_change_index",
    "paired_change_id",
    "transition_id",
    "transition_order",
    "source_scheme_id",
    "source_scheme_order",
    "target_scheme_id",
    "target_scheme_order",
    "method_a",
    "method_a_order",
    "method_b",
    "method_b_order",
    "public_root_code",
    "root_order",
    "root_source_contrast",
    "root_target_contrast",
    "root_paired_change",
    "root_count",
    "task_count",
    "utility_metric",
    "change_definition",
    "analysis_timing",
    "evidence_role",
)

PAIRED_SUMMARY_HEADER = (
    "dataset_id",
    "dataset_role",
    "source_report_sha256",
    "paired_change_index",
    "paired_change_id",
    "transition_id",
    "transition_order",
    "source_scheme_id",
    "source_scheme_order",
    "target_scheme_id",
    "target_scheme_order",
    "method_a",
    "method_a_order",
    "method_b",
    "method_b_order",
    "point_paired_change",
    "root_standard_error",
    "max_t_lower",
    "max_t_upper",
    "max_t_resolved_change_direction",
    "max_t_critical_value",
    "max_t_alpha",
    "max_t_family_size",
    "max_t_sign_count",
    "max_t_sign_schedule_sha256",
    "max_t_fallback",
    "change_definition",
    "analysis_timing",
    "evidence_role",
)


class Figure3V10Error(RuntimeError):
    """Raised when a package-local evidence or display invariant fails."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_exact_tsv(path: Path, expected_header: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != expected_header:
            raise Figure3V10Error(
                f"paired-change header changed for {path.name}: {reader.fieldnames}"
            )
        return list(reader)


def _false_token(value: str) -> bool:
    return value.strip().lower() in {"false", "0", "no"}


def read_paired_changes(package: Path) -> dict[str, Any]:
    """Load the post hoc 56-member paired-construction working-law family."""

    source_dir = package / "data" / "derived" / "public_held_out_evidence"
    root_path = source_dir / PAIRED_ROOT_TABLE
    summary_path = source_dir / PAIRED_SUMMARY_TABLE
    manifest_path = source_dir / PAIRED_MANIFEST
    if sha256(root_path) != PAIRED_ROOT_TABLE_SHA256:
        raise Figure3V10Error("paired-change root table digest changed")
    if sha256(summary_path) != PAIRED_SUMMARY_TABLE_SHA256:
        raise Figure3V10Error("paired-change summary table digest changed")
    if sha256(manifest_path) != PAIRED_MANIFEST_SHA256:
        raise Figure3V10Error("paired-change manifest digest changed")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest["record_type"]
        != "GSE162632_PAIRED_CONSTRUCTION_CHANGES_PUBLIC_V1"
        or manifest["status"]
        != "PASS_EXACT_PUBLIC_ROOT_TO_SEPARATE_POST_HOC_56_RECALCULATION"
        or manifest["dataset_id"] != PAIRED_DATASET_ID
        or manifest["analysis_timing"] != PAIRED_ANALYSIS_TIMING
        or manifest["geometry"]
        != {
            "methods": 8,
            "paired_change_family_size": 56,
            "root_change_rows": 448,
            "roots": 8,
            "shared_referenced_transitions": 2,
            "tasks": 5,
            "unordered_method_pairs": 28,
        }
        or manifest["artifacts"][PAIRED_ROOT_TABLE]
        != {"rows": 448, "sha256": PAIRED_ROOT_TABLE_SHA256}
        or manifest["artifacts"][PAIRED_SUMMARY_TABLE]
        != {"rows": 56, "sha256": PAIRED_SUMMARY_TABLE_SHA256}
        or manifest["inference"]["alpha"] != 0.05
        or manifest["inference"]["critical_value"]
        != PAIRED_MAX_T_CRITICAL_VALUE
        or manifest["inference"]["sign_count"] != 256
        or manifest["inference"]["sign_schedule_sha256"]
        != PAIRED_SIGN_SCHEDULE_SHA256
        or manifest["scientific_scope"]["separate_from_preexisting_84_member_family"]
        is not True
        or manifest["scientific_scope"]["preexisting_84_member_family_modified"]
        is not False
        or manifest["scientific_scope"]["working_law_sensitivity_only"]
        is not True
        or manifest["scientific_scope"][
            "isolated_construction_component_attribution_allowed"
        ]
        is not False
        or manifest["scientific_scope"]["population_generalization_allowed"]
        is not False
    ):
        raise Figure3V10Error("paired-change manifest scope or geometry changed")

    raw_roots = _read_exact_tsv(root_path, PAIRED_ROOT_HEADER)
    raw_summaries = _read_exact_tsv(summary_path, PAIRED_SUMMARY_HEADER)
    if len(raw_roots) != 448 or len(raw_summaries) != 56:
        raise Figure3V10Error(
            "paired-change tables must contain 448 root rows and 56 summary rows"
        )

    root_rows: list[dict[str, Any]] = []
    for row in raw_roots:
        parsed = {
            **row,
            "paired_change_index": int(row["paired_change_index"]),
            "transition_order": int(row["transition_order"]),
            "source_scheme_order": int(row["source_scheme_order"]),
            "target_scheme_order": int(row["target_scheme_order"]),
            "method_a_order": int(row["method_a_order"]),
            "method_b_order": int(row["method_b_order"]),
            "root_order": int(row["root_order"]),
            "root_source_contrast": float(row["root_source_contrast"]),
            "root_target_contrast": float(row["root_target_contrast"]),
            "root_paired_change": float(row["root_paired_change"]),
            "root_count": int(row["root_count"]),
            "task_count": int(row["task_count"]),
        }
        if not np.isclose(
            parsed["root_target_contrast"] - parsed["root_source_contrast"],
            parsed["root_paired_change"],
            atol=2e-15,
            rtol=0.0,
        ):
            raise Figure3V10Error("root paired-change arithmetic changed")
        root_rows.append(parsed)

    summary_rows: list[dict[str, Any]] = []
    for row in raw_summaries:
        parsed = {
            **row,
            "paired_change_index": int(row["paired_change_index"]),
            "transition_order": int(row["transition_order"]),
            "source_scheme_order": int(row["source_scheme_order"]),
            "target_scheme_order": int(row["target_scheme_order"]),
            "method_a_order": int(row["method_a_order"]),
            "method_b_order": int(row["method_b_order"]),
            "point_paired_change": float(row["point_paired_change"]),
            "root_standard_error": float(row["root_standard_error"]),
            "max_t_lower": float(row["max_t_lower"]),
            "max_t_upper": float(row["max_t_upper"]),
            "max_t_critical_value": float(row["max_t_critical_value"]),
            "max_t_alpha": float(row["max_t_alpha"]),
            "max_t_family_size": int(row["max_t_family_size"]),
            "max_t_sign_count": int(row["max_t_sign_count"]),
        }
        summary_rows.append(parsed)

    summary_indices = [row["paired_change_index"] for row in summary_rows]
    if summary_indices != list(range(56)):
        raise Figure3V10Error("paired-change family index must be ordered 0 through 55")
    if len({row["paired_change_id"] for row in summary_rows}) != 56:
        raise Figure3V10Error("paired-change IDs must be unique")

    expected_roots = [f"ROOT_{index:02d}" for index in range(1, 9)]
    expected_targets = {1: "INDEPENDENT_SPLIT", 2: "UNIT_AWARE_CROSSFIT"}
    expected_transitions = {
        1: "INDEPENDENT_SPLIT_MINUS_NAIVE_SHARED",
        2: "UNIT_AWARE_CROSSFIT_MINUS_NAIVE_SHARED",
    }
    transition_ids: dict[int, str] = {}
    pair_keys_by_transition: dict[int, set[tuple[str, str]]] = {1: set(), 2: set()}
    roots_by_index: dict[int, list[dict[str, Any]]] = {}
    for root in root_rows:
        roots_by_index.setdefault(root["paired_change_index"], []).append(root)
    if set(roots_by_index) != set(range(56)):
        raise Figure3V10Error("paired-change root rows do not cover all 56 family members")

    for summary in summary_rows:
        index = summary["paired_change_index"]
        order = summary["transition_order"]
        if order not in expected_targets:
            raise Figure3V10Error("paired-change transition order changed")
        if order != index // 28 + 1:
            raise Figure3V10Error("paired-change transition blocks changed")
        transition_ids.setdefault(order, summary["transition_id"])
        if (
            transition_ids[order] != summary["transition_id"]
            or summary["transition_id"] != expected_transitions[order]
        ):
            raise Figure3V10Error("paired-change transition identity changed within order")
        if (
            summary["dataset_id"] != PAIRED_DATASET_ID
            or summary["dataset_role"] != PAIRED_DATASET_ROLE
            or summary["source_report_sha256"] != SOURCE_REPORT_SHA256
            or summary["source_scheme_id"] != "NAIVE_SHARED"
            or summary["source_scheme_order"] != 1
            or summary["target_scheme_id"] != expected_targets[order]
            or summary["target_scheme_order"] != order + 1
            or summary["method_a_order"] >= summary["method_b_order"]
            or summary["max_t_alpha"] != 0.05
            or summary["max_t_family_size"] != 56
            or summary["max_t_sign_count"] != 256
            or summary["max_t_sign_schedule_sha256"]
            != PAIRED_SIGN_SCHEDULE_SHA256
            or not np.isclose(
                summary["max_t_critical_value"],
                PAIRED_MAX_T_CRITICAL_VALUE,
                atol=2e-15,
                rtol=0.0,
            )
            or not _false_token(summary["max_t_fallback"])
            or summary["change_definition"] != PAIRED_CHANGE_DEFINITION
            or summary["analysis_timing"] != PAIRED_ANALYSIS_TIMING
            or summary["evidence_role"] != PAIRED_EVIDENCE_ROLE
        ):
            raise Figure3V10Error("paired-change family scope or post hoc metadata changed")
        if not (
            summary["max_t_lower"]
            <= summary["point_paired_change"]
            <= summary["max_t_upper"]
        ):
            raise Figure3V10Error("paired-change interval does not contain its point")
        if not np.isclose(
            summary["point_paired_change"] - summary["max_t_lower"],
            summary["max_t_critical_value"] * summary["root_standard_error"],
            atol=2e-15,
            rtol=0.0,
        ) or not np.isclose(
            summary["max_t_upper"] - summary["point_paired_change"],
            summary["max_t_critical_value"] * summary["root_standard_error"],
            atol=2e-15,
            rtol=0.0,
        ):
            raise Figure3V10Error("paired-change max-T interval arithmetic changed")
        expected_direction = (
            "POSITIVE"
            if summary["max_t_lower"] > 0.0
            else "NEGATIVE"
            if summary["max_t_upper"] < 0.0
            else "UNRESOLVED"
        )
        if summary["max_t_resolved_change_direction"] != expected_direction:
            raise Figure3V10Error("paired-change resolved-direction rule changed")

        family_roots = sorted(roots_by_index[index], key=lambda row: row["root_order"])
        if [row["public_root_code"] for row in family_roots] != expected_roots:
            raise Figure3V10Error("paired-change root order changed")
        for root in family_roots:
            shared_fields = (
                "dataset_id",
                "dataset_role",
                "source_report_sha256",
                "paired_change_id",
                "transition_id",
                "transition_order",
                "source_scheme_id",
                "source_scheme_order",
                "target_scheme_id",
                "target_scheme_order",
                "method_a",
                "method_a_order",
                "method_b",
                "method_b_order",
                "change_definition",
                "analysis_timing",
                "evidence_role",
            )
            if any(root[field] != summary[field] for field in shared_fields):
                raise Figure3V10Error("paired-change root/summary metadata mismatch")
            if root["root_count"] != 8 or root["task_count"] != 5:
                raise Figure3V10Error("paired-change realized-root or task count changed")
            if root["utility_metric"] != PAIRED_UTILITY_METRIC:
                raise Figure3V10Error("paired-change utility metric changed")
        values = np.array([row["root_paired_change"] for row in family_roots], dtype=float)
        mean = float(np.mean(values))
        standard_error = float(np.std(values, ddof=1) / np.sqrt(len(values)))
        if not np.isclose(mean, summary["point_paired_change"], atol=2e-15, rtol=0.0):
            raise Figure3V10Error("paired-change roots do not reproduce the family point")
        if not np.isclose(
            standard_error,
            summary["root_standard_error"],
            atol=2e-15,
            rtol=0.0,
        ):
            raise Figure3V10Error("paired-change roots do not reproduce the standard error")
        pair_keys_by_transition[order].add((summary["method_a"], summary["method_b"]))

    if any(len(pairs) != 28 for pairs in pair_keys_by_transition.values()):
        raise Figure3V10Error("each paired transition must contain all 28 method pairs")
    if pair_keys_by_transition[1] != pair_keys_by_transition[2]:
        raise Figure3V10Error("paired transitions must use the same 28 method pairs")

    selected_summaries: dict[str, dict[str, Any]] = {}
    selected_roots: dict[str, list[dict[str, Any]]] = {}
    for target in PAIRED_TARGET_ORDER:
        matches = [
            row
            for row in summary_rows
            if row["target_scheme_id"] == target
            and row["method_a"] == PCA64_METHOD_ID
            and row["method_b"] == SCGEN_METHOD_ID
        ]
        if len(matches) != 1:
            raise Figure3V10Error(f"expected one focal paired change for {target}")
        selected = matches[0]
        selected_summaries[target] = selected
        selected_roots[target] = sorted(
            roots_by_index[selected["paired_change_index"]],
            key=lambda row: row["root_order"],
        )
        if (
            selected["point_paired_change"] <= 0.0
            or selected["max_t_lower"] <= 0.0
            or any(row["root_paired_change"] <= 0.0 for row in selected_roots[target])
        ):
            raise Figure3V10Error(
                f"focal paired change is not positive in all roots for {target}"
            )

    return {
        "root_rows": root_rows,
        "summary_rows": summary_rows,
        "selected_summaries": selected_summaries,
        "selected_roots": selected_roots,
        "manifest": manifest,
    }


