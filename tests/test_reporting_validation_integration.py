"""Synthetic integration checks; not sampling-calibration experiments."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
from scipy.stats import t


import reference_design.reporting_validation.integration_driver as driver
import reference_design.reporting_validation.reporting_candidate as reporting


def prepare(tmp_path, modify=None):
    bundle, protocol, admission, plan = driver.integration_fixture()
    frozen = tmp_path / "publication"
    seal = driver.freeze_for_matrix(bundle, protocol, admission, plan, frozen)
    matrix = driver.matrix_fixture(frozen, seal)
    if modify:
        modify(matrix)
    path = tmp_path / "invented_matrix.json"
    path.write_bytes(reporting.canonical(matrix))
    return frozen, seal, path


def execute(tmp_path, modify=None):
    frozen, seal, path = prepare(tmp_path, modify)
    receipt = driver.assess_matrix(frozen, seal, path, tmp_path / "assessment")
    load = lambda name: json.loads((tmp_path / "assessment" / name).read_text())
    return receipt, load("paired_unit_intervals.json"), load("reporting_result.json"), load("finite_panel_bounds.json")


def test_native_unit_matrix_to_full_N_intervals_to_fixed_reporting_pipeline(tmp_path):
    receipt, intervals, report, bounds = execute(tmp_path)
    assert receipt["scientific_validation"] is receipt["interval_calibration_performed"] is False
    assert receipt["multiplicity_N"] == intervals["multiplicity_N"] == 4
    assert intervals["independent_unit_count"] == 8
    assert intervals["t_critical"] == pytest.approx(t.isf(.05 / 8, 7))
    assert intervals["t_critical"] > t.isf(.05 / 4, 7)
    assert intervals["records"]["pair_0"]["interval"][0] > 0
    assert intervals["records"]["pair_1"]["interval"][1] < 0
    assert intervals["records"]["pair_2"]["reason"] == "missing_unit_margin"
    assert intervals["records"]["pair_2"]["observed_units"] == 7
    assert intervals["records"]["pair_3"]["reason"] == "zero_original_SE_without_structural_identity"
    for row in report["curves"]:
        if row["published_M"] == 2:
            assert row["fixed_roster_N"] == 4
            assert row["supported"] == row["reversed"] == 1 and row["unresolved"] == 0
        if row["published_M"] == 3:
            assert row["unresolved"] == row["unassessable_published"] == 1
            assert row["unresolved_fraction"] == pytest.approx(1 / 3)
    assert all(row["bounds"]["true_reversal_fraction_difference_bounds"] == [0., 0.]
               for row in bounds["paired_rule_bounds"] if row["M"])
    assert receipt["raw_matrix_file_sha256"] == reporting.file_hash(tmp_path / "invented_matrix.json")
    for name, digest in receipt["output_sha256"].items():
        assert reporting.file_hash(tmp_path / "assessment" / name) == digest


def test_matrix_not_read_until_publication_and_one_time_open_are_verified(tmp_path, monkeypatch):
    frozen, seal, path = prepare(tmp_path)
    original = Path.read_bytes
    observed = []
    def guarded(p):
        if p == path:
            observed.append(True)
            assert (frozen / "MATRIX_OPENED.json").exists()
            assert not (frozen / "ASSESSMENT_OPENED.json").exists()
            assert reporting.file_hash(frozen / "PUBLICATION_FROZEN.json") == seal
        return original(p)
    monkeypatch.setattr(Path, "read_bytes", guarded)
    driver.assess_matrix(frozen, seal, path, tmp_path / "assessment")
    assert observed == [True]
    with pytest.raises(ValueError, match="already opened"):
        driver.assess_matrix(frozen, seal, path, tmp_path / "retry")
    assert observed == [True]


def test_bad_freeze_prevents_any_matrix_read(tmp_path, monkeypatch):
    frozen, seal, path = prepare(tmp_path)
    original = Path.read_bytes
    def guarded(p):
        assert p != path, "Assessment read before freeze verification"
        return original(p)
    monkeypatch.setattr(Path, "read_bytes", guarded)
    with pytest.raises(ValueError, match="freeze was modified"):
        driver.assess_matrix(frozen, "0" * 64, path, tmp_path / "assessment")
    assert not (frozen / "MATRIX_OPENED.json").exists()


def test_interval_implementation_hash_locked_before_A_read(tmp_path, monkeypatch):
    frozen, seal, path = prepare(tmp_path)
    hashes = driver.implementation_hashes()
    hashes["assessment_intervals.py"] = "0" * 64
    monkeypatch.setattr(driver, "implementation_hashes", lambda: hashes)
    with pytest.raises(ValueError, match="code changed"):
        driver.assess_matrix(frozen, seal, path, tmp_path / "assessment")
    assert not (frozen / "MATRIX_OPENED.json").exists()


def test_constant_published_margin_needs_identity_not_zero_width_CI(tmp_path):
    def constant(matrix):
        for row in matrix["margins"]:
            row[0] = 0.
    _, intervals, report, _ = execute(tmp_path, constant)
    assert intervals["records"]["pair_0"]["status"] == "unassessable"
    assert intervals["records"]["pair_0"]["interval"] is None
    assert intervals["multiplicity_N"] == 4
    for row in report["curves"]:
        if row["published_M"] == 2:
            assert row["reversed"] == row["unresolved"] == row["unassessable_published"] == 1


@pytest.mark.parametrize("change,match", [
    (lambda m: m.update(mode="real_data"), "remain disabled"),
    (lambda m: m.update(structural_zero_evidence={"pair_3": {"basis": "equal_observed_values"}}), "not authenticated"),
    (lambda m: m["comparison_ids"].pop(), "all-N"),
    (lambda m: m["unit_ids"].pop(), "independent-unit roster"),
    (lambda m: m["margins"][0].__setitem__(0, True), "real JSON numbers"),
    (lambda m: m["margins"][0].__setitem__(0, 2**53+1), "represented exactly"),
    (lambda m: m["margins"][0].__setitem__(0, -(2**53+1)), "represented exactly"),
    (lambda m: m.update(membership_sha256="0" * 64), "membership differs"),
    (lambda m: m.update(margin_input_contract_sha256="0" * 64), "aggregation contract differs"),
])
def test_invalid_matrix_contracts_cannot_shrink_or_fake_input(tmp_path, change, match):
    frozen, seal, path = prepare(tmp_path, change)
    with pytest.raises(ValueError, match=match):
        driver.assess_matrix(frozen, seal, path, tmp_path / "assessment")
    assert (frozen / "MATRIX_OPENED.json").exists()
    assert not (frozen / "ASSESSMENT_OPENED.json").exists()
    with pytest.raises(FileExistsError):
        driver.assess_matrix(frozen, seal, path, tmp_path / "retry")


def test_axis_reordering_preserves_canonical_matrix_and_intervals(tmp_path):
    original = execute(tmp_path / "first")
    def permute(matrix):
        matrix["unit_ids"].reverse()
        matrix["comparison_ids"].reverse()
        matrix["margins"] = [list(reversed(row)) for row in reversed(matrix["margins"])]
    reordered = execute(tmp_path / "second", permute)
    assert original[0]["canonical_matrix_sha256"] == reordered[0]["canonical_matrix_sha256"]
    assert original[1] == reordered[1]
    assert original[2]["curves"] == reordered[2]["curves"]


def test_real_protocol_fails_before_any_publication_output(tmp_path):
    bundle, protocol, admission, plan = driver.integration_fixture()
    plan["mode"] = "real"
    with pytest.raises(ValueError, match="remain disabled"):
        driver.freeze_for_matrix(bundle, protocol, admission, plan, tmp_path / "publication")
    assert not (tmp_path / "publication").exists()


def test_unbounded_joint_critical_value_never_falls_back_to_t_only(tmp_path):
    bundle, protocol, admission, plan = driver.integration_fixture()
    bundle["units"] = [u for u in bundle["units"] if u["partition"] != "assessment" or u["unit_id"] in ("A0", "A1")]
    bundle["controls"] = {u: cells for u, cells in bundle["controls"].items() if u in {r["unit_id"] for r in bundle["units"]}}
    protocol["minimum_assessment_units"] = plan["minimum_units"] = 2
    frozen = tmp_path / "publication"
    seal = driver.freeze_for_matrix(bundle, protocol, admission, plan, frozen)
    matrix = driver.matrix_fixture(frozen, seal)
    matrix["margins"] = [[3., -4., None, 0.], [5., -2., 1., 0.]]
    path = tmp_path / "matrix.json"
    path.write_bytes(reporting.canonical(matrix))
    driver.assess_matrix(frozen, seal, path, tmp_path / "assessment")
    intervals = json.loads((tmp_path / "assessment/paired_unit_intervals.json").read_text())
    report = json.loads((tmp_path / "assessment/reporting_result.json").read_text())
    assert intervals["bootstrap_status"] == "unbounded_positive_infinite_critical_quantile"
    assert intervals["records"]["pair_0"]["t_interval"] is not None
    assert intervals["records"]["pair_0"]["interval"] is None
    for row in report["curves"]:
        assert row["supported"] == row["reversed"] == 0
        assert row["unresolved"] == row["published_M"]
