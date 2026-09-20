#!/usr/bin/env python3
"""Export the recorded Norman scores into organizer input tables."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
DESIGN_FILES = ("development_scores.tsv", "assessment_anchors.tsv", "geometry.tsv")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        result = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def authenticate(root=ROOT, expected=False):
    root = Path(root).resolve()
    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text())
    require(manifest["schema_version"] == 1, "Unsupported source manifest")
    for name, record in manifest["files"].items():
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe source path")
        if name.startswith("expected/") and not expected:
            continue
        path = root / relative
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root),
                "Source file is missing or outside the example: " + name)
        require(path.stat().st_size == record["bytes"] and digest(path) == record["sha256"],
                "Source binding differs: " + name)
    return manifest


def read_scores(path, study, cases, anchor_only=False):
    config = study["configuration"]
    realizations = ([config["anchor_realization"]] if anchor_only
                    else [item["id"] for item in study["realizations"]])
    budgets = [item["id"] for item in config["budgets"]]
    expected = set(itertools.product(cases, budgets, realizations, config["models"]))
    values = {}
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        require(reader.fieldnames == study["score_columns"], "Unexpected score columns")
        for row in reader:
            require(row["unit"] == study["unit"], "Unexpected score unit")
            key = tuple(row[field] for field in ("case", "budget", "realization", "model"))
            require(key in expected and key not in values, "Duplicate or undeclared score row")
            value = float(row["score"])
            require(math.isfinite(value) and value >= 0, "Invalid standardized MSE")
            values[key] = value
    require(set(values) == expected, "Incomplete score support")
    return values


def export_inputs(output, stage="all", root=ROOT):
    """Keep full assessment scores separate until the design stage is complete."""
    require(stage in ("design", "assessment", "all"), "Unsupported export stage")
    root, output = Path(root).resolve(), Path(output)
    manifest = authenticate(root)
    study = json.loads((root / "study.json").read_text())
    development, assessment = study["development_cases"], study["assessment_cases"]
    require(len(set(development)) == len(development) and len(set(assessment)) == len(assessment),
            "Duplicate cases in the split")
    require(not set(development) & set(assessment), "Development and assessment cases overlap")
    records = {record["export_name"]: (name, record)
               for name, record in manifest["files"].items() if "export_name" in record}
    if stage == "assessment":
        require(output.is_dir(), "Export design inputs first")
        for name in DESIGN_FILES:
            require(digest(output / name) == records[name][1]["export_sha256"],
                    "Previously exported design input changed: " + name)
        names = ("assessment_scores.tsv",)
    else:
        require(not output.exists(), "Output already exists; choose a fresh directory")
        output.mkdir(parents=True)
        names = (*DESIGN_FILES, "assessment_scores.tsv") if stage == "all" else DESIGN_FILES
    for name in names:
        source_name, record = records[name]
        source = root / source_name
        payload = gzip.decompress(source.read_bytes()) if source.suffix == ".gz" else source.read_bytes()
        require(len(payload) == record["export_bytes"]
                and hashlib.sha256(payload).hexdigest() == record["export_sha256"],
                "Expanded source binding differs: " + name)
        with (output / name).open("xb") as stream:
            stream.write(payload)
    if stage != "assessment":
        read_scores(output / "development_scores.tsv", study, development)
        read_scores(output / "assessment_anchors.tsv", study, assessment, anchor_only=True)
        write_json(output / "manifest.json", study["configuration"])
    if stage != "design":
        scores = read_scores(output / "assessment_scores.tsv", study, assessment)
        anchors = read_scores(output / "assessment_anchors.tsv", study, assessment, anchor_only=True)
        require(all(scores[key] == value for key, value in anchors.items()),
                "Assessment scores differ from the recorded anchors")
    write_json(output / ("assessment_export.json" if stage == "assessment" else "export.json"), {
        "schema_version": 1, "stage": stage,
        "source_manifest_sha256": digest(root / "SOURCE_MANIFEST.json"),
        "files": {name: {"sha256": digest(output / name), "rows": records[name][1]["rows"]}
                  for name in names},
    })
    return study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", choices=("design", "assessment", "all"), default="all")
    args = parser.parse_args()
    export_inputs(args.output, args.stage)
    print(json.dumps({"status": "exported", "stage": args.stage}))


if __name__ == "__main__":
    main()
