#!/usr/bin/env python3
"""Run a public-data example with frozen didactic predictions and official scoring."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from urllib.request import urlopen

from reference_design.vcc import UPSTREAM_COMMIT, sha256_file
from reference_design.vcc_cli import run_audit

SOURCES = {
    "adamson10x005": {
        "sha256": "6c6eca0f53f8887b86597e2a4ff512ff2b2d3d9c78ee7deec9a6e7d6ae859d01",
        "url": "https://zenodo.org/api/records/13350497/files/AdamsonWeissman2016_GSM2406677_10X005.h5ad/content",
        "generator": "prepare_public_example.py",
        "generator_args": ["--dataset", "adamson10x005"],
    },
    "dixit": {
        "sha256": "f7c483fa2a3cb79740ee816850ad7b8432f2c4c17fb4ac6fdd2cbe32a29f11a1",
        "url": "https://zenodo.org/api/records/13350497/files/DixitRegev2016_K562_TFs_13_days.h5ad/content",
        "generator": "prepare_public_example.py",
        "generator_args": ["--dataset", "dixit"],
    },
    "h1-tutorial": {
        "sha256": "eb36c766cbf76353f9981cb3a3aa32137622d1de53b29d861c483742bcd4dec7",
        "url": ("https://raw.githubusercontent.com/ArcInstitute/cell-eval2/"
                + UPSTREAM_COMMIT + "/docs/data/H1-VCC-2025-training.h5ad"),
        "generator": "prepare_example.py",
    },
}


def _artifact(root: Path, path: Path) -> dict:
    return {"path": os.path.relpath(path, root), "sha256": sha256_file(path)}


def build_manifest(root: Path, upstream: Path) -> dict:
    inputs = root / "inputs"
    source = json.loads((inputs / "SOURCE.json").read_text())
    dataset_id = source["identity"]["dataset_id"]
    has_intervention = "input_alt_control_ids" in source["split"]
    allocations = [{"allocation_id": "original", "kind": "original", "reference_policy": "shared"}]
    for depth in source.get("alloc_depths", [20, 30]):
        for seed in (0, 1):
            for policy in ("shared", "separated"):
                allocations.append({"allocation_id": f"{policy}_n{depth}_s{seed}",
                                    "kind": "equal_depth", "reference_policy": policy,
                                    "depth_per_stratum": depth, "seed": seed})
    models = []
    names = ["input_mean", "truth_informed"]
    if has_intervention:
        names.insert(1, "input_mean_alternative")
    for name in names:
        intervention = name == "input_mean_alternative"
        conditioning = {
            "mode": "input_intervention" if intervention else "fixed_submission",
            "input_pool_id": "alternative_public" if intervention else "primary_public",
            "uses_input_controls": name != "truth_informed",
            "checkpoint_sha256": source["predictions"][name]["checkpoint_sha256"],
            "provenance": ("Didactic resampling of the named public input controls; generation-rule hash, no trained model."
                           if name != "truth_informed" else
                           "Didactic truth-informed prediction made from observed perturbed means; input controls are not used. No trained model."),
        }
        if intervention:
            conditioning["parent_model_id"] = "input_mean"
        models.append({
            "model_id": name, "prediction": _artifact(root, inputs / f"{name}.h5ad"),
            "conditioning": conditioning,
        })
    input_pools = [{
        "input_pool_id": "primary_public", "dataset_id": dataset_id, "split": "public_input",
        "artifact": _artifact(root, inputs / "input_controls.h5ad"),
        "control_ids": source["split"]["input_control_ids"],
        "provenance": "First declared segment of sorted source control IDs; selection and physical identities are recorded in inputs/SOURCE.json.",
    }]
    if has_intervention:
        input_pools.append({
            "input_pool_id": "alternative_public", "dataset_id": dataset_id, "split": "public_input",
            "artifact": _artifact(root, inputs / "input_controls_alternative.h5ad"),
            "control_ids": source["split"]["input_alt_control_ids"],
            "provenance": "Second declared segment of sorted source control IDs; disjoint from the primary input and scoring controls.",
        })
    context = {
        "context_id": ("Adamson_K562_public_example" if source.get("dataset") == "adamson10x005"
                       else "Dixit_K562_public_example" if has_intervention else "H1_public_example"),
        "panel_id": "selected_targets_all_genes" if has_intervention else "five_perturbations_1000_genes",
        "reference": _artifact(root, inputs / "real.h5ad"),
        "pert_col": source.get("pert_col", "target_gene"), "control_label": source.get("control_label", "non-targeting"),
        "scoring_pool": {
            "dataset_id": dataset_id, "split": "public_validation",
            "control_ids": source["split"]["scoring_control_ids"],
            "provenance": "Example scoring holdout within public data, using the declared sorted-ID segment after both input pools; see inputs/SOURCE.json.",
        },
        "input_pools": input_pools,
        "models": models, "allocations": allocations,
        "calibration": {"mode": "build", "bundle_id": "public_example",
                        "baseline_prediction": _artifact(root, inputs / "calibration_baseline.h5ad")},
        "unit_declaration": {
            "status": "NOT_ESTABLISHED", "unit_ids": [],
            "basis": "One public dataset subset; control allocations are not independent experimental repeats.",
            "authority": "Public example description, not a biological-independence certification.",
        },
    }
    return {"schema": "reference_design.vcc_audit.v1", "audit_id": "public_reference_example",
            "description": "Public Perturb-seq data and frozen didactic predictions scored with VCC2026 metrics; software demonstration only.",
            "data_scope": "public_data_demo", "upstream": {"path": str(upstream.absolute()), "commit": UPSTREAM_COMMIT},
            "runtime": {"de_backend": "scanpy", "device": "cpu", "num_threads": 2},
            "contexts": [context]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-python", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", choices=sorted(SOURCES), default="adamson10x005",
                        help="Default: full-gene Adamson example. Dixit and h1-tutorial retain calibration-failure cases.")
    parser.add_argument("--source", type=Path, help="Optional local copy of the pinned public H5AD")
    parser.add_argument("--upstream", type=Path, help="Optional existing official checkout at the pinned commit")
    parser.add_argument("--prepare-only", action="store_true", help="Create frozen inputs and manifest without scoring")
    args = parser.parse_args()
    source_spec = SOURCES[args.dataset]
    root = args.output.absolute()
    if root.exists():
        parser.error(f"Use a new output directory: {root}")
    root.mkdir(parents=True)
    source = args.source.absolute() if args.source else root / "public_source.h5ad"
    if args.source is None:
        with urlopen(source_spec["url"], timeout=120) as response, source.open("wb") as stream:
            shutil.copyfileobj(response, stream)
    if sha256_file(source) != source_spec["sha256"]:
        raise ValueError("Public source SHA256 does not match the pinned dataset")
    if args.upstream:
        upstream = args.upstream.absolute()
    else:
        upstream = root / "official_source"
        subprocess.run(["git", "clone", "--no-checkout", "https://github.com/ArcInstitute/cell-eval2.git", str(upstream)], check=True)
        subprocess.run(["git", "-C", str(upstream), "checkout", "--detach", UPSTREAM_COMMIT], check=True)
    script = Path(__file__).with_name(source_spec["generator"])
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
        environment[name] = "2"
    subprocess.run([args.official_python, str(script), *source_spec.get("generator_args", []),
                    "--source", str(source), "--output", str(root / "inputs")],
                   env=environment, check=True)
    manifest = build_manifest(root, upstream)
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    execution = None
    if not args.prepare_only:
        execution = run_audit(path, root / "audit", args.official_python)
    print(json.dumps({"manifest": str(path), "prepared_only": args.prepare_only,
                      "audit_status": execution["status"] if execution else None,
                      "scientific_scope": manifest["description"]}, indent=2))
    return 2 if execution and execution["status"] == "COMPLETE_WITH_UNAVAILABLE_ANALYSES" else 0


if __name__ == "__main__":
    raise SystemExit(main())
