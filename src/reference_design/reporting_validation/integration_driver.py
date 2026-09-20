"""Synthetic-only bridge from frozen publication decisions to unit-matrix CIs.

All matrix entries in the CLI fixture are invented. The interface provides no
real-study admission, predictor-identity authentication, or calibration claim.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import assessment_intervals as intervals
from . import reporting_candidate as reporting
from .synthetic_fixture import fixture

MODE = reporting.MODE
MULTIPLICITY = "all_frozen_comparisons_bonferroni_t_and_joint_max_t_envelope"


def _save(path, value):
    with Path(path).open("xb") as stream:
        stream.write(reporting.canonical(value))


def _now():
    return datetime.now(timezone.utc).isoformat()


def implementation_hashes():
    return {"integration_driver.py": reporting.file_hash(__file__),
            "reporting_candidate.py": reporting.file_hash(reporting.__file__),
            "assessment_intervals.py": reporting.file_hash(intervals.__file__)}


def validate_interval_plan(plan, protocol):
    reporting.require(set(plan) == {"mode", "method", "confidence", "resamples", "seed", "minimum_units", "multiplicity", "structural_zero_policy"}, "Interval plan schema differs")
    reporting.require(plan["mode"] == MODE, "Real matrix comparisons remain disabled")
    reporting.require(plan["method"] == intervals.METHOD == protocol["assessment_interval_method"], "Interval method differs")
    reporting.require(plan["confidence"] == protocol["confidence_level"], "Interval confidence differs")
    reporting.require(plan["multiplicity"] == MULTIPLICITY == protocol["multiplicity_policy"], "All-N multiplicity contract differs")
    reporting.require(plan["structural_zero_policy"] == "no_unauthenticated_identity_claims", "Structural-zero authentication is not implemented")
    for key, minimum in (("resamples", 2), ("seed", 0), ("minimum_units", 2)):
        reporting.require(type(plan[key]) is int and plan[key] >= minimum, "Invalid interval " + key)
    reporting.require(plan["minimum_units"] == protocol["minimum_assessment_units"], "Unit floor differs from publication protocol")


def freeze_for_matrix(bundle, protocol, admission, interval_plan, destination):
    """Only publication data and assessment metadata are accepted at this stage."""
    protocol = deepcopy(protocol)
    validate_interval_plan(interval_plan, protocol)
    reporting.validate_protocol(protocol, admission)
    protocol["matrix_integration"] = {"interval_plan": deepcopy(interval_plan),
        "implementation_sha256": implementation_hashes(),
        "margin_input_contract": "one_equal_weight_native_unit_by_complete_frozen_task_model_pair; loss_b_minus_loss_a; seeds_and_folds_scored_then_averaged",
        "freeze_time_utc": _now()}
    return reporting.freeze_publication(bundle, protocol, admission, destination)


def _load_matrix_after_open(path, frozen, freeze_sha):
    # One byte buffer is both parsed and hashed: no path reread can change input.
    raw = Path(path).read_bytes()
    def invalid_constant(value):
        raise ValueError("Matrix JSON must use null for missing data, not " + value)
    data = json.loads(raw, parse_constant=invalid_constant)
    reporting.require(data.get("mode") == MODE and data.get("source_kind") == "hand_constructed_fixture", "Real matrix comparisons remain disabled")
    expected_fields = {"mode", "source_kind", "publication_freeze_sha256", "membership_sha256", "margin_input_contract_sha256", "unit_ids", "comparison_ids", "margins"}
    reporting.require(set(data) == expected_fields, "Matrix schema differs; structural-zero claims are not authenticated")
    reporting.require(data["publication_freeze_sha256"] == freeze_sha and data["membership_sha256"] == frozen["membership_sha256"], "Matrix freeze or unit membership differs")
    contract = frozen["protocol"]["matrix_integration"]["margin_input_contract"]
    reporting.require(data["margin_input_contract_sha256"] == reporting.digest(contract), "Matrix aggregation contract differs")
    expected_units = sorted(u["unit_id"] for u in frozen["units"] if u["partition"] == "assessment")
    expected_comparisons = [r["pair_id"] for r in frozen["roster"]]
    supplied_units, supplied_comparisons = data["unit_ids"], data["comparison_ids"]
    reporting.require(isinstance(supplied_units, list) and isinstance(supplied_comparisons, list), "Matrix axes must be explicit lists")
    reporting.require(len(supplied_units) == len(set(supplied_units)) and set(supplied_units) == set(expected_units), "Complete frozen independent-unit roster required")
    reporting.require(len(supplied_comparisons) == len(set(supplied_comparisons)) and set(supplied_comparisons) == set(expected_comparisons), "Complete frozen all-N comparison roster required")
    values = data["margins"]
    reporting.require(isinstance(values, list) and len(values) == len(expected_units), "Matrix row count differs")
    converted = []
    for row in values:
        reporting.require(isinstance(row, list) and len(row) == len(expected_comparisons), "Matrix column count differs")
        output = []
        for value in row:
            if value is None:
                output.append(np.nan)
            else:
                reporting.require(type(value) in (int, float), "Matrix entries must be real JSON numbers or null")
                try:
                    numeric = float(value)
                except (ValueError, OverflowError) as error:
                    raise ValueError("Matrix numeric overflow") from error
                reporting.require(math.isfinite(numeric), "Matrix contains infinite values")
                reporting.require(type(value) is not int or int(numeric) == value,
                                  "Matrix integer cannot be represented exactly as float64")
                output.append(numeric)
        converted.append(output)
    matrix = np.asarray(converted, dtype=float)
    matrix = matrix[[supplied_units.index(u) for u in expected_units]][:, [supplied_comparisons.index(c) for c in expected_comparisons]]
    return matrix, expected_units, expected_comparisons, hashlib.sha256(raw).hexdigest()


def assess_matrix(frozen_directory, expected_freeze_sha256, matrix_path, output_directory):
    """Claim one matrix opening after checking the P freeze; compute and classify.

    A failed attempt remains marked. It must not be retried with tuned choices.
    The existing classifier adds its separate interval-read receipt afterward.
    """
    directory, output = Path(frozen_directory), Path(output_directory)
    freeze_path = directory / "PUBLICATION_FROZEN.json"
    freeze_bytes = freeze_path.read_bytes()
    reporting.require(hashlib.sha256(freeze_bytes).hexdigest() == expected_freeze_sha256, "Publication freeze was modified")
    frozen = json.loads(freeze_bytes)
    protocol = frozen["protocol"]
    reporting.validate_protocol(protocol, frozen["admission"])
    integration = protocol["matrix_integration"]
    validate_interval_plan(integration["interval_plan"], protocol)
    reporting.require(integration["implementation_sha256"] == implementation_hashes(), "Integrated code changed after publication freeze")
    reporting.require(reporting.digest(protocol) == frozen["protocol_sha256"] and reporting.digest(frozen["units"]) == frozen["membership_sha256"], "Frozen protocol or membership hash differs")
    reporting.require(not (directory / "ASSESSMENT_OPENED.json").exists(), "Intervals were already opened by another route")
    reporting.require(not output.exists(), "Integration output already exists")
    _save(directory / "MATRIX_OPENED.json", {"publication_freeze_sha256": expected_freeze_sha256,
        "opened_at_utc": _now(), "attempt": 1, "mode": MODE})
    output.mkdir(parents=True, exist_ok=False)
    matrix, units, ids, source_sha = _load_matrix_after_open(matrix_path, frozen, expected_freeze_sha256)
    plan = integration["interval_plan"]
    result = intervals.paired_matrix_intervals(matrix, unit_ids=units, comparison_ids=ids,
        mode=MODE, confidence=plan["confidence"], resamples=plan["resamples"], seed=plan["seed"],
        minimum_units=plan["minimum_units"], structural_zero_evidence=None)
    reporting.require(result["multiplicity_N"] == len(frozen["roster"]) and set(result["records"]) == set(ids), "Interval module changed the all-N family")
    _save(output / "paired_unit_intervals.json", result)
    supplied = {"mode": MODE, "source_kind": "hand_constructed_fixture", "publication_freeze_sha256": expected_freeze_sha256,
        "membership_sha256": frozen["membership_sha256"], "unit_ids": units,
        "interval_method": result["method"], "confidence_level": result["confidence"],
        "multiplicity_policy": protocol["multiplicity_policy"], "intervals": []}
    for identity in ids:
        row = result["records"][identity]
        if row["status"] == "unassessable":
            supplied["intervals"].append({"pair_id": identity, "status": "unassessable", "reason": row["reason"]})
        else:
            reporting.require(row["status"] == "assessable", "Unverified structural zero must not enter the workflow")
            supplied["intervals"].append({"pair_id": identity, "status": "interval", "lower": row["interval"][0], "upper": row["interval"][1]})
    supplied_path = output / "intervals_for_classifier.json"
    _save(supplied_path, supplied)
    report = reporting.assess(directory, expected_freeze_sha256, supplied_path, output / "reporting_result.json")
    panel_rows, paired_rows = [], []
    mechanical = {(r["published_M"], r["rule_id"]): r for r in report["curves"]}
    for point in frozen["plans"]["curve"]:
        directions = {rule: {r["pair_id"]: r["direction"] for r in point["decisions"] if r["rule_id"] == rule} for rule in reporting.RULES}
        for rule in reporting.RULES:
            bounds = intervals.finite_panel_risk_bounds(ids, directions[rule], result["records"])
            row = mechanical[(point["published_M"], rule)]
            reporting.require(bounds["counts"]["candidates"] == row["fixed_roster_N"] and bounds["counts"]["published"] == row["published_M"] and all(bounds["counts"][k] == row[k] for k in ("supported", "reversed", "unresolved")), "Independent interval and reporting classifiers disagree")
            panel_rows.append({"rule_id": rule, "M": point["published_M"], "bounds": bounds})
        for index, rule_j in enumerate(reporting.RULES):
            for rule_k in reporting.RULES[index + 1:]:
                bounds = intervals.paired_rule_difference_bounds(ids, directions[rule_j], directions[rule_k], result["records"])
                paired_rows.append({"rule_j": rule_j, "rule_k": rule_k, "M": point["published_M"], "bounds": bounds})
    _save(output / "finite_panel_bounds.json", {"mode": MODE, "scientific_validation": False,
        "panel_bounds": panel_rows, "paired_rule_bounds": paired_rows})
    receipt = {"schema": "refara.reporting_integration.engineering.v1", "status": "PASS", "mode": MODE,
        "scientific_validation": False, "real_comparisons_allowed": False, "interval_calibration_performed": False,
        "publication_freeze_sha256": expected_freeze_sha256, "raw_matrix_file_sha256": source_sha,
        "canonical_matrix_sha256": result["matrix_sha256"], "multiplicity_N": result["multiplicity_N"],
        "native_units": len(units), "primary_M": report["primary_M"], "primary_status": report["primary_status"],
        "implementation_sha256": implementation_hashes(),
        "output_sha256": {p.name: reporting.file_hash(p) for p in sorted(output.glob("*.json"))},
        "open_receipts": {name: reporting.file_hash(directory / name) for name in ("MATRIX_OPENED.json", "ASSESSMENT_OPENED.json")},
        "scope": "Hand-constructed equal-native-unit margins test an approximate interval construction and frozen reporting workflow. This does not establish interval calibration, unit independence, empirical rule advantage or admitted scientific validation."}
    _save(output / "ENGINEERING_RECEIPT.json", receipt)
    return receipt


def integration_fixture():
    bundle, protocol, admission = fixture()
    for index in range(2, 8):
        unit = f"A{index}"
        bundle["units"].append({"unit_id": unit, "partition": "assessment", "native_experiment_id": "experiment_" + unit, "independence_block_id": "block_" + unit})
        bundle["controls"][unit] = [unit + f"_cell_{j}" for j in range(4)]
    protocol["assessment_interval_method"] = intervals.METHOD
    protocol["multiplicity_policy"] = MULTIPLICITY
    protocol["minimum_assessment_units"] = 8
    plan = {"mode": MODE, "method": intervals.METHOD, "confidence": protocol["confidence_level"],
        "resamples": 199, "seed": 17, "minimum_units": 8, "multiplicity": MULTIPLICITY,
        "structural_zero_policy": "no_unauthenticated_identity_claims"}
    return bundle, protocol, admission, plan


def matrix_fixture(frozen_directory, seal):
    frozen = json.loads((Path(frozen_directory) / "PUBLICATION_FROZEN.json").read_text())
    units = sorted(u["unit_id"] for u in frozen["units"] if u["partition"] == "assessment")
    offsets = [-.35, -.25, -.15, -.05, .05, .15, .25, .35]
    return {"mode": MODE, "source_kind": "hand_constructed_fixture", "publication_freeze_sha256": seal,
        "membership_sha256": frozen["membership_sha256"],
        "margin_input_contract_sha256": reporting.digest(frozen["protocol"]["matrix_integration"]["margin_input_contract"]),
        "unit_ids": units, "comparison_ids": [r["pair_id"] for r in frozen["roster"]],
        "margins": [[4 + v, -3 + v, None if i == 7 else v, 0.] for i, v in enumerate(offsets)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    bundle, protocol, admission, plan = integration_fixture()
    seal = freeze_for_matrix(bundle, protocol, admission, plan, args.output / "publication")
    matrix_path = args.output / "invented_native_unit_margins.json"
    _save(matrix_path, matrix_fixture(args.output / "publication", seal))
    receipt = assess_matrix(args.output / "publication", seal, matrix_path, args.output / "assessment")
    print(json.dumps({"status": receipt["status"], "mode": MODE, "scientific_validation": False, "multiplicity_N": receipt["multiplicity_N"]}))


if __name__ == "__main__":
    main()
