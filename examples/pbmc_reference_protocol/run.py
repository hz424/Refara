#!/usr/bin/env python3
"""Score bundled PBMC predictions with the reference protocol and check the replay."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from reference_design.protocol import run_plan


REPOSITORY = Path(__file__).resolve().parents[2]
BUNDLE = REPOSITORY / "capsules/gse162632_scgen/replay"
TOLERANCE = 5e-13


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, columns, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def run(output: Path) -> dict:
    manifest = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    names = {"root_tasks", "root_task_metrics", "root_metrics", "focal_contrasts",
             "treated_means", "c_obs", "c_pred", "pca64_effect", "scgen_native_q1",
             "scales", "weights"}
    records = {entry["name"]: entry for entry in manifest["files"] if entry["name"] in names}
    if set(records) != names:
        raise ValueError("The focal capsule is missing a required input")
    arrays = {}
    for name, entry in records.items():
        path = BUNDLE / entry["path"]
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError(f"Capsule file differs from its manifest: {entry['path']}")
        if path.suffix == ".npy":
            values = np.load(path, allow_pickle=False)
            if (list(values.shape) != entry["shape"] or str(values.dtype.str) != entry["dtype"]
                    or not np.isfinite(values).all()):
                raise ValueError(f"Unexpected array layout or values: {entry['path']}")
            arrays[name] = values
    tasks = read_tsv(BUNDLE / records["root_tasks"]["path"])
    genes = [f"G{i+1:04d}" for i in range(arrays["treated_means"].shape[1])]
    if not np.all(arrays["weights"] == 1 / len(genes)):
        raise ValueError("This example requires the capsule's equal feature weights")
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir()
    write_json(inputs / "source.json", {
        "capsule": "capsules/gse162632_scgen/replay",
        "manifest_sha256": digest(BUNDLE / "manifest.json"),
        "files": [records[name] for name in sorted(records)],
        "realization": "q1", "construction": "split", "reference_instance": 1,
        "features": "G0001 to G2000 label the capsule's fixed pseudonymized feature axis.",
        "prediction_handling": "Cached predictions remain fixed. scGen was generated upstream "
        "using its original model-input block. B1 below is the prediction-centring baseline; "
        "this example does not regenerate the model output from B1.",
        "arithmetic": "scGen is exported as the capsule's float32 state-minus-baseline result, "
        "with that original baseline supplied for restoration before scoring.",
    })
    write_tsv(inputs / "scales.tsv", ["gene", "value"], zip(genes, arrays["scales"]))
    # Preserve the emitted effect's original arithmetic before writing lossless
    # decimal float values. The model remains a state model in the plan.
    scgen_effect = arrays["scgen_native_q1"][1] - arrays["c_pred"][1]
    plan_tasks = []
    for task in tasks:
        index = int(task["root_task_index"])
        task_id = f"{task['root_id']}_{task['task_id']}"
        folder = inputs / task_id
        folder.mkdir()
        vectors = {"treated": arrays["treated_means"][index],
                   "ridge": arrays["pca64_effect"][index],
                   "scgen": scgen_effect[index], "baseline": arrays["c_pred"][1, index]}
        for name, values in vectors.items():
            write_tsv(folder / f"{name}.tsv", ["gene", "value"],
                      ((gene, float(value)) for gene, value in zip(genes, values)))
        write_tsv(folder / "references.tsv", ["allocation", "depth", "block", *genes], [
            [0, 8, "B1", *map(float, arrays["c_pred"][1, index])],
            [0, 8, "B2", *map(float, arrays["c_obs"][1, index])],
        ])
        for model, filenames in [("ridge", ["ridge.tsv", "../source.json"]),
                                 ("scgen", ["scgen.tsv", "baseline.tsv", "../source.json"])]:
            write_json(folder / f"{model}_provenance.json", {
                "description": "Fixed supplied prediction from the bundled q1 split replay; "
                "source.json records the original input hashes and export semantics.",
                "files": [{"path": name, "sha256": digest(folder / name)} for name in filenames],
            })
        relative = f"inputs/{task_id}"
        plan_tasks.append({
            "id": task_id, "unit": task["root_id"],
            "treated": f"{relative}/treated.tsv", "references": f"{relative}/references.tsv",
            "models": [
                {"name": "PCA64_ridge", "kind": "effect", "prediction": f"{relative}/ridge.tsv",
                 "provenance": f"{relative}/ridge_provenance.json"},
                {"name": "scGen", "kind": "state", "conditioning": "fixed",
                 "representation": "effect", "prediction": f"{relative}/scgen.tsv",
                 "baseline": f"{relative}/baseline.tsv", "provenance": f"{relative}/scgen_provenance.json"},
            ],
        })
    write_json(output / "plan.json", {
        "target": "heldout_effect", "prediction_controls": "available", "diagnostics": True,
        "scales": "inputs/scales.tsv", "tasks": plan_tasks,
    })
    receipt = run_plan(output / "plan.json", output / "results")
    expected_task = {(f"{r['root_id']}_{r['task_id']}", model): float(r[field])
                     for r in read_tsv(BUNDLE / "root_task_metrics.tsv")
                     if r["realization_id"] == "q1" and r["construction"] == "split"
                     for model, field in [("PCA64_ridge", "pca64_standardized_mse"),
                                          ("scGen", "scgen_standardized_mse")]}
    expected_unit = {(r["root_id"], model): float(r[field])
                     for r in read_tsv(BUNDLE / "root_metrics.tsv")
                     if r["realization_id"] == "q1" and r["construction"] == "split"
                     for model, field in [("PCA64_ridge", "pca64_mean_standardized_mse"),
                                          ("scGen", "scgen_mean_standardized_mse")]}
    errors = []
    for level, key, expected in [("task", "task", expected_task), ("unit", "unit", expected_unit)]:
        rows = read_tsv(output / "results" / f"primary_{level}_scores.tsv")
        actual = {(r[key], r["model"]): float(r["MSE"]) for r in rows}
        if set(actual) != set(expected) or len(actual) != len(rows):
            raise ValueError(f"The protocol's {level} score support differs from the capsule")
        errors.extend(abs(actual[item] - expected[item]) for item in expected)
    pair = read_tsv(output / "results/primary_summary_pairs.tsv")
    expected_margin = next(float(r["split_pca64_minus_scgen"])
                           for r in read_tsv(BUNDLE / "focal_contrasts.tsv")
                           if r["realization_id"] == "q1")
    if len(pair) != 1 or (pair[0]["model_a"], pair[0]["model_b"]) != ("PCA64_ridge", "scGen"):
        raise ValueError("Unexpected model-pair orientation")
    margin = float(pair[0]["margin"])
    errors.append(abs(margin - expected_margin))
    result = {"tasks": len(tasks), "units": len({t["unit"] for t in plan_tasks}),
              "features": len(genes), "checked_scores": len(expected_task) + len(expected_unit),
              "maximum_absolute_error": max(errors), "tolerance": TOLERANCE,
              "model_a": "PCA64_ridge", "model_b": "scGen", "margin": margin,
              "expected_margin": expected_margin, "diagnostics_emitted": receipt["diagnostics_emitted"],
              "software_version": receipt["software_version"]}
    if result["maximum_absolute_error"] > TOLERANCE or receipt["diagnostics_emitted"]:
        raise ValueError(f"Replay validation failed: {result}")
    write_json(output / "verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory for inputs, plan and results")
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))
