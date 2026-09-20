from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/build_gse162632_public_evidence_v1.py"
SOURCE_DATA_ROOT = Path(os.environ.get("REFERENCE_CELL_SOURCE_DATA_ROOT", REPO))
EVIDENCE = SOURCE_DATA_ROOT / "data/derived/public_held_out_evidence"
ROOT_TABLE = EVIDENCE / "GSE162632_ROOT_UTILITIES_PUBLIC_V1.tsv"
CONTRAST_TABLE = EVIDENCE / "GSE162632_FULL_84_CONTRASTS_PUBLIC_V1.tsv"
MANIFEST = EVIDENCE / "GSE162632_PUBLIC_EVIDENCE_MANIFEST_V1.json"
ROOT_INFERENCE = (
    REPO / "src/perturb_nuisance_contracts/root_family_inference.py"
)


def load_generator():
    specification = importlib.util.spec_from_file_location(
        "_test_gse162632_public_evidence_v1", SCRIPT
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


def test_public_root_to_full_84_release_is_exact(source_data_root: Path) -> None:
    report = load_generator().check_release()
    assert report == {
        "status": "PASS_EXACT_PUBLIC_ROOT_TO_FULL_84_VERIFICATION",
        "root_rows": 192,
        "contrast_rows": 84,
        "max_t_critical_value": pytest.approx(4.578236227495731, abs=1.0e-14),
        "bonferroni_t_critical_value": pytest.approx(
            5.907038874082924, abs=1.0e-12
        ),
        "max_t_resolved": 68,
        "bonferroni_t_resolved": 52,
        "holm_resolved": 68,
        "max_t_bonferroni_opposite_directions": 0,
    }


def test_public_evidence_round_trip_is_byte_identical(
    tmp_path: Path,
    source_data_root: Path,
) -> None:
    report = load_generator().write_release(tmp_path)
    assert report["status"] == "PASS_WROTE_PUBLIC_HELD_OUT_EVIDENCE"
    for source in (ROOT_TABLE, CONTRAST_TABLE, MANIFEST):
        assert (tmp_path / source.name).read_bytes() == source.read_bytes()


def test_full_family_sensitivities_preserve_selected_directions(
    source_data_root: Path,
) -> None:
    rows = read_tsv(CONTRAST_TABLE)
    assert len(rows) == 84
    assert sum(row["max_t_resolved_direction"] != "UNRESOLVED" for row in rows) == 68
    assert (
        sum(row["bonferroni_t_resolved_direction"] != "UNRESOLVED" for row in rows)
        == 52
    )
    assert sum(row["holm_resolved_direction"] != "UNRESOLVED" for row in rows) == 68
    assert [row["holm_resolved_direction"] for row in rows] == [
        row["max_t_resolved_direction"] for row in rows
    ]

    selected = {
        row["scheme_id"]: row
        for row in rows
        if row["method_a"] == "PCA64_ADDITIVE_RIDGE_DIRECT_V1"
        and row["method_b"] == "SCGEN_2_1_1_ABSOLUTE_V1"
    }
    assert set(selected) == {
        "NAIVE_SHARED",
        "INDEPENDENT_SPLIT",
        "UNIT_AWARE_CROSSFIT",
    }
    expected = {
        "NAIVE_SHARED": ("B_GREATER", -0.0510423639, -0.0169557437),
        "INDEPENDENT_SPLIT": ("A_GREATER", 0.0891262334, 0.1515107845),
        "UNIT_AWARE_CROSSFIT": ("A_GREATER", 0.0924171199, 0.1569144439),
    }
    for scheme, (resolved, lower, upper) in expected.items():
        row = selected[scheme]
        assert row["max_t_resolved_direction"] == resolved
        assert row["bonferroni_t_resolved_direction"] == resolved
        assert row["holm_resolved_direction"] == resolved
        assert float(row["bonferroni_t_lower"]) == pytest.approx(lower, abs=5.0e-10)
        assert float(row["bonferroni_t_upper"]) == pytest.approx(upper, abs=5.0e-10)


def test_root_codes_are_release_local_pseudonyms(source_data_root: Path) -> None:
    rows = read_tsv(ROOT_TABLE)
    assert len(rows) == 192
    assert "public_root_code" in rows[0]
    assert "public_study_root_id" not in rows[0]
    assert "source_report_path" not in rows[0]
    assert {row["public_root_code"] for row in rows} == {
        f"ROOT_{index:02d}" for index in range(1, 9)
    }
    assert all(row["larger_is_better"] == "true" for row in rows)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    boundary = manifest["privacy_and_access_boundary"]
    assert boundary["root_identifiers_anonymous"] is False
    assert boundary["root_identifiers_pseudonymized"] is True
    assert boundary["root_identifier_mapping_included"] is False
    assert boundary["public_root_code_column"] == "public_root_code"
    assert boundary["cell_level_values_included"] is False
    assert boundary["expression_or_prediction_arrays_included"] is False
    assert boundary["root_to_84_contrast_recalculation_from_public_checkout"] is True
    assert boundary["raw_to_prediction_replay_from_public_checkout"] is False
    assert "ROOT_01 through ROOT_08" in boundary["identifier_description"]


def test_root_family_inference_module_is_neutral_and_complete() -> None:
    source = ROOT_INFERENCE.read_text(encoding="utf-8")
    for forbidden in (
        "HMN",
        "A" + "FR",
        "GSE162632",
        "public_score_input_" + "bundle_v1",
        "cell_selection" + "_salt",
        "evaluation_iav_selection" + "_salt",
        "control_membership" + "_salt",
    ):
        assert forbidden not in source

    root_inference, generic = load_generator().load_implementations()
    rng = np.random.default_rng(20260825)
    values = rng.normal(
        size=(
            root_inference.ROOTS,
            len(root_inference.SCHEMES),
            root_inference.METHODS,
        )
    )
    report = root_inference.analyse_root_utilities(values, generic)
    assert len(report["point_utility_records"]) == 24
    assert len(report["pairwise_interval_records"]) == 84
    assert len(report["leave_one_root_out"]) == 8
    assert report["sign_schedule"]["count"] == 256
    assert {
        row["held_out_root_id"] for row in report["leave_one_root_out"]
    } == {f"ROOT_{index:02d}" for index in range(1, 9)}

    degenerate = root_inference.analyse_root_utilities(
        np.zeros((8, 3, 8), dtype=np.float64),
        generic,
    )
    assert degenerate["fallback"] is True
    assert all(
        row["lower"] is None and row["upper"] is None
        for row in degenerate["pairwise_interval_records"]
    )
