#!/usr/bin/env python3
"""Compare reporting rules on Norman and the held-out influenza donors."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import compare
from export import ROOT, digest, export_inputs as export_norman, require, write_json
from transfer import export_inputs as export_transfer


def read_table(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_table(path, rows):
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def check_table(actual_path, expected_path):
    actual, expected = read_table(actual_path), read_table(expected_path)
    require(len(actual) == len(expected), "Result row count differs: " + actual_path.name)
    for row_number, (observed, recorded) in enumerate(zip(actual, expected), 1):
        require(observed.keys() == recorded.keys(), "Result columns differ: " + actual_path.name)
        for key, wanted in recorded.items():
            value = observed[key]
            if value == wanted:
                continue
            try:
                equal = math.isclose(float(value), float(wanted), rel_tol=1e-10, abs_tol=5e-12)
            except ValueError:
                equal = False
            require(equal, f"Result differs in {actual_path.name}, row {row_number}, {key}")


def evaluate(output, *, verify=True):
    output = Path(output)
    require(not output.exists(), "Output already exists; choose a fresh directory")
    output.mkdir(parents=True)
    protocol_path = ROOT / "comparison_protocol.json"
    protocol_hash = digest(protocol_path)
    datasets = ("norman", "original8", "cap48")
    frozen = []
    for name in datasets:
        folder = output / "datasets" / name
        inputs = folder / "inputs"
        if name == "norman":
            export_norman(inputs, stage="design")
        else:
            export_transfer(inputs, name, stage="design")
        compare.freeze(inputs / "manifest.json", protocol_path, folder / "freeze")
        require(not (inputs / "assessment_scores.tsv").exists(),
                "Assessment scores were exported before planning")
        plan = folder / "freeze" / "plan.json"
        frozen.append({"resource": name, "plan": plan.relative_to(output).as_posix(),
                       "sha256": digest(plan)})
    freeze_record = output / "plans_frozen.json"
    write_json(freeze_record, {"protocol_sha256": protocol_hash, "plans": frozen,
                              "assessment_tables_exported": False})
    freeze_hash = digest(freeze_record)

    summaries, rules, areas = {}, [], []
    for item in frozen:
        name = item["resource"]
        folder = output / "datasets" / name
        inputs = folder / "inputs"
        plan = output / item["plan"]
        require(digest(plan) == item["sha256"], "Frozen plan changed before assessment")
        if name == "norman":
            export_norman(inputs, stage="assessment")
        else:
            export_transfer(inputs, name, stage="assessment")
        summary = compare.check(plan, inputs / "assessment_scores.tsv", folder / "check")
        summaries[name] = {key: summary[key] for key in (
            "development_cases", "development_units", "assessment_cases", "assessment_units", "scope")}
        rules.extend(dict(resource=name, **row) for row in summary["rule_summary"])
        areas.extend(dict(resource=name, **row) for row in summary["aurc"])
    require(digest(protocol_path) == protocol_hash and digest(freeze_record) == freeze_hash,
            "Analysis protocol or freeze record changed during assessment")
    require(all(digest(output / item["plan"]) == item["sha256"] for item in frozen),
            "A frozen plan changed during assessment")
    results = output / "results"
    results.mkdir()
    tables = {"comparison_rules.tsv": rules, "comparison_aurc.tsv": areas}
    for name, rows in tables.items():
        destination = results / name
        write_table(destination, rows)
        if verify:
            check_table(destination, ROOT / "expected" / name)
    receipt = {
        "schema_version": 1,
        "status": "pass" if verify else "computed_without_expected_result_check",
        "analysis": "Retrospective reporting-rule comparison",
        "protocol_sha256": protocol_hash,
        "comparison_code_sha256": digest(ROOT / "compare.py"),
        "evaluation_code_sha256": digest(Path(__file__)),
        "plans_frozen_sha256": freeze_hash,
        "all_plans_frozen_before_assessment_export": True,
        "datasets": summaries,
        "results": {name: {"rows": len(rows), "sha256": digest(results / name)}
                    for name, rows in tables.items()},
        "expected_results_checked": verify,
    }
    write_json(output / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate(args.output)
    print(json.dumps({"status": result["status"], "datasets": result["datasets"]}, indent=2))


if __name__ == "__main__":
    main()
