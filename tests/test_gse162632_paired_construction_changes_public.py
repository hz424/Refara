from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/build_gse162632_paired_construction_changes_public_v1.py"
SOURCE_DATA_ROOT = Path(os.environ.get("REFERENCE_CELL_SOURCE_DATA_ROOT", REPO))
EVIDENCE = SOURCE_DATA_ROOT / "data/derived/public_held_out_evidence"
INPUT_ROOTS = EVIDENCE / "GSE162632_ROOT_UTILITIES_PUBLIC_V1.tsv"
CHANGE_ROOTS = EVIDENCE / "GSE162632_PAIRED_CONSTRUCTION_CHANGES_ROOTS_PUBLIC_V1.tsv"
SUMMARY = EVIDENCE / "GSE162632_PAIRED_CONSTRUCTION_CHANGES_FULL_56_PUBLIC_V1.tsv"
MANIFEST = EVIDENCE / "GSE162632_PAIRED_CONSTRUCTION_CHANGES_MANIFEST_V1.json"


def load_generator():
    specification = importlib.util.spec_from_file_location(
        "_test_paired_construction_changes_public_v1", SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    if "REFERENCE_CELL_SOURCE_DATA_ROOT" in os.environ:
        module.configure_source_data_root(SOURCE_DATA_ROOT)
    return module


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_separate_post_hoc_56_release_is_exact(source_data_root: Path) -> None:
    report = load_generator().check_release()
    assert report == {
        "status": "PASS_EXACT_PAIRED_CONSTRUCTION_CHANGE_56_VERIFICATION",
        "root_change_rows": 448,
        "summary_rows": 56,
        "max_t_critical_value": pytest.approx(5.000133078920623, abs=2.0e-15),
        "resolved_changes": 33,
        "resolved_split_minus_shared": 16,
        "resolved_rotation_minus_shared": 17,
    }


def test_paired_change_round_trip_is_byte_identical(
    tmp_path: Path,
    source_data_root: Path,
) -> None:
    report = load_generator().write_release(tmp_path)
    assert report["status"] == "PASS_WROTE_PAIRED_CONSTRUCTION_CHANGE_56_EVIDENCE"
    for source in (CHANGE_ROOTS, SUMMARY, MANIFEST):
        assert (tmp_path / source.name).read_bytes() == source.read_bytes()


def test_root_rows_are_complete_and_algebraically_paired(source_data_root: Path) -> None:
    utility_rows = read_tsv(INPUT_ROOTS)
    utilities = {
        (row["public_root_code"], row["scheme_id"], row["method_id"]): float(
            row["root_utility"]
        )
        for row in utility_rows
    }
    rows = read_tsv(CHANGE_ROOTS)
    assert len(rows) == 448
    assert len(
        {(row["paired_change_index"], row["public_root_code"]) for row in rows}
    ) == 448
    assert {row["public_root_code"] for row in rows} == {
        f"ROOT_{index:02d}" for index in range(1, 9)
    }
    assert {
        sum(row["paired_change_index"] == str(index) for row in rows)
        for index in range(56)
    } == {8}
    assert {row["source_scheme_id"] for row in rows} == {"NAIVE_SHARED"}
    assert {row["target_scheme_id"] for row in rows} == {
        "INDEPENDENT_SPLIT",
        "UNIT_AWARE_CROSSFIT",
    }
    assert {row["root_count"] for row in rows} == {"8"}
    assert {row["task_count"] for row in rows} == {"5"}
    assert {row["analysis_timing"] for row in rows} == {
        "POST_HOC_AFTER_OUTCOME_ACCESS"
    }

    for row in rows:
        root = row["public_root_code"]
        method_a = row["method_a"]
        method_b = row["method_b"]
        source = (
            utilities[(root, row["source_scheme_id"], method_a)]
            - utilities[(root, row["source_scheme_id"], method_b)]
        )
        target = (
            utilities[(root, row["target_scheme_id"], method_a)]
            - utilities[(root, row["target_scheme_id"], method_b)]
        )
        assert float(row["root_source_contrast"]) == pytest.approx(
            source, abs=2.0e-15
        )
        assert float(row["root_target_contrast"]) == pytest.approx(
            target, abs=2.0e-15
        )
        assert float(row["root_paired_change"]) == pytest.approx(
            target - source, abs=2.0e-15
        )


def test_summary_schema_focal_values_and_family_boundary(source_data_root: Path) -> None:
    rows = read_tsv(SUMMARY)
    assert len(rows) == 56
    assert [int(row["paired_change_index"]) for row in rows] == list(range(56))
    assert {row["max_t_family_size"] for row in rows} == {"56"}
    assert {row["max_t_sign_count"] for row in rows} == {"256"}
    assert {row["max_t_fallback"] for row in rows} == {"false"}
    assert {
        row["max_t_resolved_change_direction"] for row in rows
    } <= {"POSITIVE", "NEGATIVE", "UNRESOLVED"}
    assert sum(
        row["max_t_resolved_change_direction"] != "UNRESOLVED" for row in rows
    ) == 33
    assert all(
        float(row["max_t_critical_value"])
        == pytest.approx(5.000133078920623, abs=2.0e-15)
        for row in rows
    )

    focal = [
        row
        for row in rows
        if row["method_a"] == "PCA64_ADDITIVE_RIDGE_DIRECT_V1"
        and row["method_b"] == "SCGEN_2_1_1_ABSOLUTE_V1"
    ]
    assert [row["transition_id"] for row in focal] == [
        "INDEPENDENT_SPLIT_MINUS_NAIVE_SHARED",
        "UNIT_AWARE_CROSSFIT_MINUS_NAIVE_SHARED",
    ]
    expected = (
        (0.15431756271199337, 0.0032756208969892367, 0.13793902231095384, 0.1706961031130329),
        (0.15866483570994305, 0.0034034459775929139, 0.14164715289506138, 0.17568251852482472),
    )
    for row, values in zip(focal, expected):
        observed = tuple(
            float(row[field])
            for field in (
                "point_paired_change",
                "root_standard_error",
                "max_t_lower",
                "max_t_upper",
            )
        )
        assert observed == pytest.approx(values, abs=3.0e-15)
        assert row["max_t_resolved_change_direction"] == "POSITIVE"

    root_rows = read_tsv(CHANGE_ROOTS)
    direct_methods = {
        "NO_CHANGE_DIRECT_V1",
        "CONTEXT_MEAN_EFFECT_DIRECT_V1",
        "TWO_WAY_ADDITIVE_RIDGE_DIRECT_V1",
        "PCA64_ADDITIVE_RIDGE_DIRECT_V1",
        "RBF_KERNEL_RIDGE_DIRECT_V1",
    }
    reversal_rows = [
        row
        for row in rows
        if row["method_a"] in direct_methods
        and row["method_b"] == "SCGEN_2_1_1_ABSOLUTE_V1"
    ]
    assert len(reversal_rows) == 10
    assert all(
        row["max_t_resolved_change_direction"] == "POSITIVE"
        for row in reversal_rows
    )
    for summary_row in reversal_rows:
        matching = [
            row
            for row in root_rows
            if row["paired_change_index"] == summary_row["paired_change_index"]
        ]
        assert len(matching) == 8
        assert sum(float(row["root_source_contrast"]) for row in matching) / 8 < 0
        assert sum(float(row["root_target_contrast"]) for row in matching) / 8 > 0

    focal_indices = {row["paired_change_index"] for row in focal}
    for change_index in focal_indices:
        matching = [
            row for row in root_rows if row["paired_change_index"] == change_index
        ]
        assert len(matching) == 8
        assert all(float(row["root_paired_change"]) > 0 for row in matching)


def test_manifest_binds_artifacts_and_limits_interpretation(source_data_root: Path) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    scope = manifest["scientific_scope"]
    assert manifest["analysis_timing"] == "POST_HOC_AFTER_OUTCOME_ACCESS"
    assert scope["separate_from_preexisting_84_member_family"] is True
    assert scope["preexisting_84_member_family_modified"] is False
    assert scope["joint_140_member_family_claim"] is False
    assert scope["isolated_construction_component_attribution_allowed"] is False
    assert scope["population_generalization_allowed"] is False
    assert manifest["geometry"] == {
        "methods": 8,
        "paired_change_family_size": 56,
        "root_change_rows": 448,
        "roots": 8,
        "shared_referenced_transitions": 2,
        "tasks": 5,
        "unordered_method_pairs": 28,
    }
    for path, rows in ((CHANGE_ROOTS, 448), (SUMMARY, 56)):
        bound = manifest["artifacts"][path.name]
        assert bound["rows"] == rows
        assert bound["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
