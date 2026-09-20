#!/usr/bin/env python3
"""Build the separate post hoc 56-member paired-construction-change replay.

The input is the separately distributed 192-row root-utility table. For each unordered
method pair and each of two target constructions, the root contribution is

    (target method-A-minus-method-B) - (shared method-A-minus-method-B).

The 56 paired changes enter one recentered, re-studentized max-|T| working-law
family.  This family is separate from, and does not modify, the pre-existing
84-member within-construction family.
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


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from perturb_nuisance_contracts.source_data_root import (  # noqa: E402
    SourceDataRootError,
    SourceDataSnapshot,
    validate_source_data_root,
)

EVIDENCE_RUNTIME_DIR = Path("data/derived/public_held_out_evidence")
INPUT_ROOT_RUNTIME_PATH = (
    EVIDENCE_RUNTIME_DIR / "GSE162632_ROOT_UTILITIES_PUBLIC_V1.tsv"
)
EVIDENCE_DIR = REPO / EVIDENCE_RUNTIME_DIR
INPUT_ROOT_TABLE = REPO / INPUT_ROOT_RUNTIME_PATH
BASE_GENERATOR = REPO / "scripts/build_gse162632_public_evidence_v1.py"
SOURCE_DATA_SNAPSHOT: SourceDataSnapshot | None = None

ROOT_OUTPUT_NAME = "GSE162632_PAIRED_CONSTRUCTION_CHANGES_ROOTS_PUBLIC_V1.tsv"
SUMMARY_OUTPUT_NAME = "GSE162632_PAIRED_CONSTRUCTION_CHANGES_FULL_56_PUBLIC_V1.tsv"
MANIFEST_OUTPUT_NAME = "GSE162632_PAIRED_CONSTRUCTION_CHANGES_MANIFEST_V1.json"

INPUT_ROOT_SHA256 = "802766638793ab1d374a0f0198a905acde7739819f5b96626967435eb17584e8"
REPLAY_BASE_GENERATOR_SHA256 = "c771fc81d0c587b1fafc6c3dbde79bdec5a2b11246c3484937907e55af8b0251"
FROZEN_BASE_GENERATOR_SHA256 = "618a103db9548b0e1564c27f8ddb87ba391d7cd2d778e079c722732459bec633"
FROZEN_PAIRED_CHANGE_GENERATOR_SHA256 = "af8964a9e5ef16aad162cfea769ee026ac5bbf2b6c36dacdef92c7bb47b35d1c"
EXPECTED_CRITICAL = 5.000133078920623
EXPECTED_SIGN_SHA256 = "12d6e0393ce9981498e8d9f37f0a04b3a7ce50584c200d9bc99658a26c98f3b7"
EXPECTED_RESOLVED = 33
EXPECTED_BY_TRANSITION = (16, 17)

DATASET_ID = "GSE162632_FIXED_IAV_6H"
DATASET_ROLE = "POST_HOC_HELD_OUT_WITHIN_STUDY_PAIRED_CONSTRUCTION_CHANGE"
SOURCE_REPORT_SHA256 = "1a297a99c03985493aa4156eaa89fafe0cbd89a32950745399ef427bd9ae923c"
UTILITY_METRIC = "oriented_standardized_squared_perturbation_effect_error"
CHANGE_DEFINITION = "TARGET_MINUS_SOURCE_OF_METHOD_A_MINUS_METHOD_B"
ANALYSIS_TIMING = "POST_HOC_AFTER_OUTCOME_ACCESS"
EVIDENCE_ROLE = "POST_HOC_PAIRED_COMPOSITE_CONSTRUCTION_CHANGE_WORKING_SENSITIVITY"

TRANSITIONS = (
    ("INDEPENDENT_SPLIT_MINUS_NAIVE_SHARED", 0, 1),
    ("UNIT_AWARE_CROSSFIT_MINUS_NAIVE_SHARED", 0, 2),
)

ROOT_HEADER = (
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

SUMMARY_HEADER = (
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


class PairedChangeError(RuntimeError):
    """Raised when a paired-change input, result, or release differs."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PairedChangeError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    require(
        specification is not None and specification.loader is not None,
        f"cannot import {path}",
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def load_implementations() -> tuple[ModuleType, ModuleType, ModuleType]:
    require(
        sha256_file(BASE_GENERATOR) == REPLAY_BASE_GENERATOR_SHA256,
        "base public-evidence generator SHA-256 differs",
    )
    base = load_module(BASE_GENERATOR, "_paired_change_base_public_v1")
    root_inference, generic = base.load_implementations()
    require(
        tuple(root_inference.SCHEMES)
        == ("NAIVE_SHARED", "INDEPENDENT_SPLIT", "UNIT_AWARE_CROSSFIT")
        and root_inference.ROOTS == 8
        and root_inference.TASKS == 5
        and root_inference.METHODS == 8
        and root_inference.SIGN_COUNT == 256
        and root_inference.SIGN_SCHEDULE_SHA256 == EXPECTED_SIGN_SHA256,
        "root-family geometry or sign schedule differs",
    )
    return base, root_inference, generic


def float_text(value: float) -> str:
    number = float(value)
    require(math.isfinite(number), "nonfinite public numeric value")
    return format(number, ".17g")


def bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def resolved_direction(lower: float, upper: float) -> str:
    return "POSITIVE" if lower > 0.0 else "NEGATIVE" if upper < 0.0 else "UNRESOLVED"


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


def pair_geometry(root_inference: ModuleType, generic: ModuleType) -> tuple[Any, tuple[tuple[int, int], ...]]:
    family = generic.contrast_family(root_inference.METHODS)
    pairs = tuple((left, right) for scheme, left, right in family.identities[:28])
    require(
        len(pairs) == 28
        and all(scheme == 0 for scheme, _left, _right in family.identities[:28])
        and tuple(family.identities[28:56])
        == tuple((1, left, right) for left, right in pairs)
        and tuple(family.identities[56:84])
        == tuple((2, left, right) for left, right in pairs),
        "within-construction pair geometry differs",
    )
    return family, pairs


def build_root_rows(
    root_utilities: np.ndarray,
    root_inference: ModuleType,
    generic: ModuleType,
) -> list[dict[str, str]]:
    values = np.asarray(root_utilities, dtype=np.float64)
    require(values.shape == (8, 3, 8) and np.isfinite(values).all(), "root tensor differs")
    family, pairs = pair_geometry(root_inference, generic)
    within = generic.root_contrasts(values[None, ...], family)[0]
    rows: list[dict[str, str]] = []
    for transition_index, (transition_id, source_index, target_index) in enumerate(TRANSITIONS):
        for pair_index, (left, right) in enumerate(pairs):
            change_index = transition_index * len(pairs) + pair_index
            source_column = source_index * len(pairs) + pair_index
            target_column = target_index * len(pairs) + pair_index
            method_a = root_inference.METHOD_IDS[left]
            method_b = root_inference.METHOD_IDS[right]
            paired_change_id = f"{transition_id}::{method_a}__VS__{method_b}"
            for root_index, root_code in enumerate(root_inference.ROOT_IDS):
                source_value = float(within[root_index, source_column])
                target_value = float(within[root_index, target_column])
                rows.append(
                    {
                        "dataset_id": DATASET_ID,
                        "dataset_role": DATASET_ROLE,
                        "source_report_sha256": SOURCE_REPORT_SHA256,
                        "paired_change_index": str(change_index),
                        "paired_change_id": paired_change_id,
                        "transition_id": transition_id,
                        "transition_order": str(transition_index + 1),
                        "source_scheme_id": root_inference.SCHEMES[source_index],
                        "source_scheme_order": str(source_index + 1),
                        "target_scheme_id": root_inference.SCHEMES[target_index],
                        "target_scheme_order": str(target_index + 1),
                        "method_a": method_a,
                        "method_a_order": str(left + 1),
                        "method_b": method_b,
                        "method_b_order": str(right + 1),
                        "public_root_code": root_code,
                        "root_order": str(root_index + 1),
                        "root_source_contrast": float_text(source_value),
                        "root_target_contrast": float_text(target_value),
                        "root_paired_change": float_text(target_value - source_value),
                        "root_count": "8",
                        "task_count": "5",
                        "utility_metric": UTILITY_METRIC,
                        "change_definition": CHANGE_DEFINITION,
                        "analysis_timing": ANALYSIS_TIMING,
                        "evidence_role": EVIDENCE_ROLE,
                    }
                )
    require(len(rows) == 448, "paired root table geometry differs")
    return rows


def change_array_from_rows(
    rows: Sequence[Mapping[str, str]],
    root_inference: ModuleType,
    generic: ModuleType,
) -> np.ndarray:
    require(len(rows) == 448, "paired root row count differs")
    _family, pairs = pair_geometry(root_inference, generic)
    values = np.full((8, 56), np.nan, dtype=np.float64)
    seen: set[tuple[int, int]] = set()
    for row in rows:
        change_index = int(row["paired_change_index"])
        root_index = int(row["root_order"]) - 1
        key = (root_index, change_index)
        require(0 <= change_index < 56 and 0 <= root_index < 8 and key not in seen, "paired root key differs")
        transition_index = change_index // 28
        pair_index = change_index % 28
        transition_id, source_index, target_index = TRANSITIONS[transition_index]
        left, right = pairs[pair_index]
        method_a = root_inference.METHOD_IDS[left]
        method_b = root_inference.METHOD_IDS[right]
        source_value = float(row["root_source_contrast"])
        target_value = float(row["root_target_contrast"])
        change_value = float(row["root_paired_change"])
        require(
            row["dataset_id"] == DATASET_ID
            and row["dataset_role"] == DATASET_ROLE
            and row["source_report_sha256"] == SOURCE_REPORT_SHA256
            and row["paired_change_id"] == f"{transition_id}::{method_a}__VS__{method_b}"
            and row["transition_id"] == transition_id
            and row["transition_order"] == str(transition_index + 1)
            and row["source_scheme_id"] == root_inference.SCHEMES[source_index]
            and row["source_scheme_order"] == str(source_index + 1)
            and row["target_scheme_id"] == root_inference.SCHEMES[target_index]
            and row["target_scheme_order"] == str(target_index + 1)
            and row["method_a"] == method_a
            and row["method_a_order"] == str(left + 1)
            and row["method_b"] == method_b
            and row["method_b_order"] == str(right + 1)
            and row["public_root_code"] == root_inference.ROOT_IDS[root_index]
            and row["root_count"] == "8"
            and row["task_count"] == "5"
            and row["utility_metric"] == UTILITY_METRIC
            and row["change_definition"] == CHANGE_DEFINITION
            and row["analysis_timing"] == ANALYSIS_TIMING
            and row["evidence_role"] == EVIDENCE_ROLE,
            "paired root metadata differs",
        )
        require(
            math.isclose(target_value - source_value, change_value, rel_tol=0.0, abs_tol=2.0e-15),
            "paired root arithmetic differs",
        )
        values[root_index, change_index] = change_value
        seen.add(key)
    require(len(seen) == 448 and np.isfinite(values).all(), "paired root matrix is incomplete")
    return values


def build_summary_rows(
    changes: np.ndarray,
    root_inference: ModuleType,
    generic: ModuleType,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    values = np.asarray(changes, dtype=np.float64)
    require(values.shape == (8, 56) and np.isfinite(values).all(), "paired change matrix differs")
    signs = generic.sign_schedule(8, monte_carlo_count=4_096, seed=root_inference.SIGN_SEED)
    require(
        signs is not None
        and signs.shape == (256, 8)
        and generic.sign_schedule_sha256(signs) == EXPECTED_SIGN_SHA256,
        "exhaustive sign schedule differs",
    )
    inference = generic.infer_current_joint_maxt(values[None, ...], signs, alpha=0.05)
    require(not bool(inference["fallback"][0]), "paired 56-family inference fell back")
    critical = float(inference["critical_values"][0])
    require(math.isclose(critical, EXPECTED_CRITICAL, rel_tol=0.0, abs_tol=2.0e-15), "paired critical value differs")
    _family, pairs = pair_geometry(root_inference, generic)
    rows: list[dict[str, str]] = []
    for change_index in range(56):
        transition_index = change_index // 28
        pair_index = change_index % 28
        transition_id, source_index, target_index = TRANSITIONS[transition_index]
        left, right = pairs[pair_index]
        method_a = root_inference.METHOD_IDS[left]
        method_b = root_inference.METHOD_IDS[right]
        lower = float(inference["lower"][0, change_index])
        upper = float(inference["upper"][0, change_index])
        rows.append(
            {
                "dataset_id": DATASET_ID,
                "dataset_role": DATASET_ROLE,
                "source_report_sha256": SOURCE_REPORT_SHA256,
                "paired_change_index": str(change_index),
                "paired_change_id": f"{transition_id}::{method_a}__VS__{method_b}",
                "transition_id": transition_id,
                "transition_order": str(transition_index + 1),
                "source_scheme_id": root_inference.SCHEMES[source_index],
                "source_scheme_order": str(source_index + 1),
                "target_scheme_id": root_inference.SCHEMES[target_index],
                "target_scheme_order": str(target_index + 1),
                "method_a": method_a,
                "method_a_order": str(left + 1),
                "method_b": method_b,
                "method_b_order": str(right + 1),
                "point_paired_change": float_text(inference["point"][0, change_index]),
                "root_standard_error": float_text(inference["standard_errors"][0, change_index]),
                "max_t_lower": float_text(lower),
                "max_t_upper": float_text(upper),
                "max_t_resolved_change_direction": resolved_direction(lower, upper),
                "max_t_critical_value": float_text(critical),
                "max_t_alpha": "0.05",
                "max_t_family_size": "56",
                "max_t_sign_count": "256",
                "max_t_sign_schedule_sha256": EXPECTED_SIGN_SHA256,
                "max_t_fallback": bool_text(False),
                "change_definition": CHANGE_DEFINITION,
                "analysis_timing": ANALYSIS_TIMING,
                "evidence_role": EVIDENCE_ROLE,
            }
        )
    directions = [row["max_t_resolved_change_direction"] for row in rows]
    by_transition = tuple(sum(value != "UNRESOLVED" for value in directions[start : start + 28]) for start in (0, 28))
    require(sum(value != "UNRESOLVED" for value in directions) == EXPECTED_RESOLVED, "paired resolution count differs")
    require(by_transition == EXPECTED_BY_TRANSITION, "transition resolution counts differ")
    focal = [
        row
        for row in rows
        if row["method_a"] == "PCA64_ADDITIVE_RIDGE_DIRECT_V1"
        and row["method_b"] == "SCGEN_2_1_1_ABSOLUTE_V1"
    ]
    expected_focal = (
        (0.15431756271199337, 0.0032756208969892367, 0.13793902231095384, 0.1706961031130329),
        (0.15866483570994305, 0.0034034459775929139, 0.14164715289506138, 0.17568251852482472),
    )
    require(len(focal) == 2, "focal paired changes are incomplete")
    for row, expected in zip(focal, expected_focal):
        observed = tuple(float(row[field]) for field in ("point_paired_change", "root_standard_error", "max_t_lower", "max_t_upper"))
        require(all(math.isclose(a, b, rel_tol=0.0, abs_tol=3.0e-15) for a, b in zip(observed, expected)), "focal paired change differs")
    return rows, {
        "max_t_critical_value": critical,
        "resolved_changes": EXPECTED_RESOLVED,
        "resolved_split_minus_shared": by_transition[0],
        "resolved_rotation_minus_shared": by_transition[1],
    }


def manifest_text(root_text: str, summary_text: str, report: Mapping[str, Any]) -> str:
    manifest = {
        "schema_version": "1.0.0",
        "record_type": "GSE162632_PAIRED_CONSTRUCTION_CHANGES_PUBLIC_V1",
        "status": "PASS_EXACT_PUBLIC_ROOT_TO_SEPARATE_POST_HOC_56_RECALCULATION",
        "dataset_id": DATASET_ID,
        "analysis_timing": ANALYSIS_TIMING,
        "scientific_scope": {
            "estimand": "equal-root target-minus-shared change in each fixed method-pair contrast",
            "finite_set_target": True,
            "working_law_sensitivity_only": True,
            "separate_from_preexisting_84_member_family": True,
            "preexisting_84_member_family_modified": False,
            "joint_140_member_family_claim": False,
            "isolated_construction_component_attribution_allowed": False,
            "population_generalization_allowed": False,
        },
        "geometry": {
            "roots": 8,
            "tasks": 5,
            "methods": 8,
            "unordered_method_pairs": 28,
            "shared_referenced_transitions": 2,
            "paired_change_family_size": 56,
            "root_change_rows": 448,
        },
        "inference": {
            "definition": "recentered and re-studentized exhaustive root sign-flip max-absolute-T over 56 paired construction changes",
            "alpha": 0.05,
            "sign_count": 256,
            "sign_schedule_sha256": EXPECTED_SIGN_SHA256,
            "critical_order_one_based": 244,
            "critical_value": report["max_t_critical_value"],
            "resolved_changes": report["resolved_changes"],
            "resolved_split_minus_shared": report["resolved_split_minus_shared"],
            "resolved_rotation_minus_shared": report["resolved_rotation_minus_shared"],
        },
        "source_bindings": {
            "root_utility_table": {
                "path": INPUT_ROOT_RUNTIME_PATH.as_posix(),
                "rows": 192,
                "sha256": INPUT_ROOT_SHA256,
            },
            "base_public_evidence_generator_sha256": FROZEN_BASE_GENERATOR_SHA256,
            "paired_change_generator_sha256": FROZEN_PAIRED_CHANGE_GENERATOR_SHA256,
        },
        "privacy_and_access_boundary": {
            "root_identifiers_pseudonymized": True,
            "root_identifier_mapping_included": False,
            "cell_level_values_included": False,
            "expression_or_prediction_arrays_included": False,
        },
        "artifacts": {
            ROOT_OUTPUT_NAME: {"rows": 448, "sha256": sha256_bytes(root_text.encode("utf-8"))},
            SUMMARY_OUTPUT_NAME: {"rows": 56, "sha256": sha256_bytes(summary_text.encode("utf-8"))},
        },
    }
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def materialize() -> tuple[dict[str, str], dict[str, Any]]:
    require(sha256_file(INPUT_ROOT_TABLE) == INPUT_ROOT_SHA256, "input root table SHA-256 differs")
    base, root_inference, generic = load_implementations()
    root_rows = base.read_tsv(INPUT_ROOT_TABLE, base.ROOT_HEADER)
    root_utilities = base.root_array_from_rows(root_rows, root_inference)
    paired_root_rows = build_root_rows(root_utilities, root_inference, generic)
    root_text = render_tsv(paired_root_rows, ROOT_HEADER)
    reader = csv.DictReader(io.StringIO(root_text), delimiter="\t")
    require(tuple(reader.fieldnames or ()) == ROOT_HEADER, "rendered paired root header differs")
    serialized_changes = change_array_from_rows(list(reader), root_inference, generic)
    summary_rows, report = build_summary_rows(serialized_changes, root_inference, generic)
    summary_text = render_tsv(summary_rows, SUMMARY_HEADER)
    outputs = {ROOT_OUTPUT_NAME: root_text, SUMMARY_OUTPUT_NAME: summary_text}
    outputs[MANIFEST_OUTPUT_NAME] = manifest_text(root_text, summary_text, report)
    return outputs, report


def configure_source_data_root(source_data_root: Path) -> SourceDataSnapshot:
    """Bind the paired replay to manifest-verified external evidence."""

    global EVIDENCE_DIR, INPUT_ROOT_TABLE, SOURCE_DATA_SNAPSHOT
    snapshot = validate_source_data_root(source_data_root)
    EVIDENCE_DIR = snapshot.root / EVIDENCE_RUNTIME_DIR
    INPUT_ROOT_TABLE = snapshot.root / INPUT_ROOT_RUNTIME_PATH
    SOURCE_DATA_SNAPSHOT = snapshot
    return snapshot


def check_release(output_dir: Path | None = None) -> dict[str, Any]:
    if output_dir is None:
        output_dir = EVIDENCE_DIR
    outputs, report = materialize()
    for name, text in outputs.items():
        path = output_dir / name
        require(path.is_file() and not path.is_symlink(), f"public artifact missing: {path}")
        require(path.read_text(encoding="utf-8") == text, f"public artifact differs: {path}")
    return {
        "status": "PASS_EXACT_PAIRED_CONSTRUCTION_CHANGE_56_VERIFICATION",
        "root_change_rows": 448,
        "summary_rows": 56,
        **report,
    }


def write_release(output_dir: Path) -> dict[str, Any]:
    outputs, report = materialize()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in outputs:
        path = output_dir / name
        require(not path.exists() and not path.is_symlink(), f"refusing to overwrite: {path}")
    for name, text in outputs.items():
        (output_dir / name).write_text(text, encoding="utf-8", newline="")
    return {
        "status": "PASS_WROTE_PAIRED_CONSTRUCTION_CHANGE_56_EVIDENCE",
        "output_dir": str(output_dir.resolve()),
        "root_change_rows": 448,
        "summary_rows": 56,
        **report,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or verify the separate post hoc paired-construction-change replay")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check", action="store_true", help="verify the supplied 56-comparison files"
    )
    mode.add_argument(
        "--output-dir", type=Path, help="new directory for reconstructed files"
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
            raise PairedChangeError("output directory cannot be a symlink")
        output = args.output_dir.expanduser().resolve()
        if output == snapshot.root or snapshot.root in output.parents:
            raise PairedChangeError("output directory cannot be inside Source Data root")
    report = check_release() if args.check else write_release(output)
    snapshot.assert_unchanged()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PairedChangeError, SourceDataRootError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
