#!/usr/bin/env python3
"""Replay the Norman organizer decisions across all four reporting gaps."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from export import ROOT, authenticate, digest, export_inputs, require, write_json
from reference_design.organizer import POLICIES, check_design, design


def read_table(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_table(path, rows, columns):
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(checks):
    policy_rows, budget_rows, geometry_rows = [], [], []
    for gap, result in checks:
        prefix = {"resource": "Norman", "minimum_gap": gap}
        policy_rows.extend(dict(**prefix, **row) for row in result["summaries"])
        geometry_rows.extend(dict(**prefix, **row) for row in result["geometry_vs_repeated"])
        for policy in POLICIES:
            selected = [row for row in result["budget_assessments"] if row["policy"] == policy]
            released = [row for row in selected if row["decision"] == "release"]
            budget_rows.append(dict(**prefix, policy=policy, assessment_cases=len(selected),
                released=len(released), held=len(selected) - len(released),
                supported=sum(row["supported"] for row in released),
                unsupported=sum(row["unsupported_recommendation"] for row in released),
                selected_budgets=json.dumps(dict(Counter(row["budget"] for row in released)), sort_keys=True)))
    return {"policy_summary.tsv": policy_rows, "budget_summary.tsv": budget_rows,
            "geometry_comparison.tsv": geometry_rows}


def replay(output, root=ROOT):
    root, output = Path(root).resolve(), Path(output)
    require(not output.exists(), "Output already exists; choose a fresh directory")
    output.mkdir(parents=True)
    inputs = output / "inputs"
    study = export_inputs(inputs, stage="design", root=root)
    frozen = []
    for gap in study["minimum_gaps"]:
        label = "gap_" + str(gap).replace(".", "p")
        manifest = dict(study["configuration"], minimum_gap=gap)
        manifest_path = inputs / (label + ".json")
        write_json(manifest_path, manifest)
        design_path = output / "designs" / label
        design(manifest_path, design_path)
        frozen.append({"minimum_gap": gap, "label": label,
                       "path": (design_path / "design.json").relative_to(output).as_posix(),
                       "sha256": digest(design_path / "design.json")})
    require(not (inputs / "assessment_scores.tsv").exists(),
            "Full assessment scores were exported before designs were frozen")
    freeze_path = output / "designs_frozen.json"
    write_json(freeze_path, {"schema_version": 1,
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "designs": frozen,
        "full_assessment_scores_exported": False})
    freeze_digest = digest(freeze_path)

    export_inputs(inputs, stage="assessment", root=root)
    checks = []
    for item in frozen:
        design_file = output / item["path"]
        require(digest(design_file) == item["sha256"], "Frozen design changed before checking")
        result = check_design(design_file, inputs / "assessment_scores.tsv",
                              output / "checks" / item["label"])
        checks.append((item["minimum_gap"], result))

    # Reference outcomes enter only after every design has been checked.
    authenticate(root, expected=True)
    tables = summarize(checks)
    results = output / "results"
    results.mkdir()
    for name, rows in tables.items():
        expected = read_table(root / "expected" / name)
        write_table(results / name, rows, list(expected[0]))
        require(read_table(results / name) == expected, "Replay differs from recorded results: " + name)
    require(digest(freeze_path) == freeze_digest, "Design freeze changed during assessment")
    require(all(digest(output / item["path"]) == item["sha256"] for item in frozen),
            "A frozen design changed during assessment")
    primary = [row for row in tables["policy_summary.tsv"]
               if row["minimum_gap"] == study["primary_minimum_gap"] and row["budget"] == "d8"]
    primary_budgets = [row for row in tables["budget_summary.tsv"]
                      if row["minimum_gap"] == study["primary_minimum_gap"]]
    receipt = {"schema_version": 1, "status": "pass", "resource": "Norman",
        "source_manifest_sha256": digest(root / "SOURCE_MANIFEST.json"),
        "designs_frozen_sha256": freeze_digest, "configurations": len(frozen),
        "all_designs_frozen_before_assessment_export": True,
        "expected_results_used_for_planning": False,
        "primary_control_cells_available": 192, "primary_minimum_gap": study["primary_minimum_gap"],
        "primary_pair_results": primary, "primary_budget_results": primary_budgets,
        "strict_geometry_gains": sum(row["outcome"] == "strict_gain"
                                     for row in tables["geometry_comparison.tsv"]),
        "results": {name: {"rows": len(rows), "sha256": digest(results / name)}
                    for name, rows in tables.items()},
        "scope": study["evaluation_scope"]}
    write_json(output / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    receipt = replay(args.output)
    print(json.dumps({"status": receipt["status"], "configurations": receipt["configurations"],
                      "primary_pair_results": receipt["primary_pair_results"],
                      "strict_geometry_gains": receipt["strict_geometry_gains"]}))


if __name__ == "__main__":
    main()
