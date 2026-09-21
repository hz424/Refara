"""Run the frozen Parse PCA predictor and score four control allocations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
from scipy import sparse

from reference_design.fixed_budget import rule_partitions, score_reference_design
from reference_design.real_model import FrozenPCA, normalize_full_counts

HERE = Path(__file__).resolve().parent


def captured(path):
    payload = Path(path).read_bytes()
    return payload, {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def table(payload):
    return list(csv.DictReader(io.StringIO(payload.decode("utf-8")), delimiter="\t"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(output, input_path=None):
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output must be a new or empty directory")
    started = time.perf_counter()
    manifest = json.loads((HERE / "MANIFEST.json").read_text())
    for relative, record in manifest["assets"].items():
        _, actual = captured(HERE / relative)
        if actual != record:
            raise ValueError(f"Bundled asset differs: {relative}")
    bundle = HERE / "data"
    model = FrozenPCA.from_npz(bundle / "model.npz",
                               expected_sha256=manifest["assets"]["data/model.npz"]["sha256"])
    source_features = [r["feature_id"] for r in table((bundle / "source_features.tsv").read_bytes())]
    panel_indices = np.load(bundle / "feature_axis.npy", allow_pickle=False)
    if tuple(map(str, panel_indices)) != model.feature_ids:
        raise ValueError("Checkpoint and training panel axes differ")
    scales = np.load(bundle / "scales.npy", allow_pickle=False)
    input_path = Path(input_path) if input_path is not None else bundle / "input.json"
    config_bytes, config_record = captured(input_path)
    config = json.loads(config_bytes)
    if config.get("schema") != "REFARA_REAL_MODEL_INPUT_V1" or config.get("normalization") != "full_source_CP10K_log1p":
        raise ValueError("Unknown input schema or normalization")
    if not isinstance(config.get("unit"), str) or not config["unit"].strip():
        raise ValueError("Input requires a nonempty unit label")
    buffers, input_records = {}, {"input_json": config_record}
    for key in ("counts", "features", "cells"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f"Input requires {key}")
        buffers[key], input_records[key] = captured(input_path.parent / config[key])
    features = table(buffers["features"])
    if not features or any(set(row) != {"source_index", "feature_id"} for row in features):
        raise ValueError("Features table requires source_index and feature_id")
    if [row["source_index"] for row in features] != list(map(str, range(len(features)))):
        raise ValueError("Features source_index must describe the input column order")
    counts = sparse.load_npz(io.BytesIO(buffers["counts"]))
    expression = normalize_full_counts(counts, [row["feature_id"] for row in features],
                                       source_features, panel_indices).astype(np.float64)
    cells = table(buffers["cells"])
    if len(cells) != len(expression) or any(set(row) != {"cell_id", "role"} for row in cells):
        raise ValueError("Cells table requires one cell_id/role row per count row")
    ids = [row["cell_id"] for row in cells]
    if len(set(ids)) != len(ids) or any(not item.strip() for item in ids):
        raise ValueError("Cell IDs must be nonempty and unique")
    if any(row["role"] not in ("control", "treated") for row in cells):
        raise ValueError("Roles must be control or treated")
    control_rows = np.array([i for i, row in enumerate(cells) if row["role"] == "control"])
    treated_rows = np.array([i for i, row in enumerate(cells) if row["role"] == "treated"])
    if len(control_rows) < 16 or len(treated_rows) != 8:
        raise ValueError("This B16/T8 example needs at least16 controls and exactly8 treated cells")
    target = expression[treated_rows].mean(axis=0)
    context, condition = config.get("context"), config.get("condition")
    rng = np.random.Generator(np.random.PCG64(20260920))
    scores, membership, effects = [], [], []
    calls = 0
    for allocation in range(4):
        chosen = rng.permutation(control_rows)[:16]
        controls = expression[chosen]
        for label, rule in (("S", "shared_all_B"), ("O", "O_half")):
            # Real inference on original fitted weights, once per allocation/rule.
            # This native direct-effect model has no control-cell dependency.
            effect = model.predict(context, condition)
            calls += 1
            effects.append(effect)
            native_loss = score_reference_design(controls, effect, target, scales, rule)
            # A representation conversion of the same effect, not a new model.
            states = controls + effect
            state_loss = score_reference_design(controls, states, target, scales, rule)
            zero_loss = score_reference_design(controls, np.zeros_like(effect), target, scales, rule)
            if not np.isclose(native_loss, state_loss, atol=1e-12, rtol=1e-12):
                raise ValueError("Effect/state representation parity failed")
            scores.append({"allocation": allocation, "protocol": label, "design": rule,
                           "PCA_native_effect_loss": native_loss,
                           "PCA_equivalent_state_loss": state_loss, "NC_zero_effect_loss": zero_loss})
            input_rows, observation_rows = rule_partitions(16, rule)[0]
            for role, rows in (("input", input_rows), ("observation", observation_rows)):
                membership.extend({"allocation": allocation, "protocol": label, "role": role,
                                   "cell_id": ids[int(chosen[i])]} for i in rows)
            membership.extend({"allocation": allocation, "protocol": label, "role": "treated",
                               "cell_id": ids[int(i)]} for i in treated_rows)
    effects = np.stack(effects)
    if not np.array_equal(effects, np.broadcast_to(effects[0], effects.shape)):
        raise ValueError("Native effect unexpectedly depends on allocation")
    result = {
        "schema": "REFARA_REAL_MODEL_DEMONSTRATION_V1",
        "purpose": "Method-use demonstration; no independent validation or performance claim",
        "input_scope": config.get("scope", "User-supplied cells; sampling independence not assessed"),
        "unit": config["unit"], "context": context, "condition": condition,
        "features": {"full_library": len(source_features), "scored": len(panel_indices),
                     "scales": "Fixed training scales; uniform weights over2000 scored genes"},
        "allocation": {"count": 4, "physical_control_budget": 16, "treated_cells": 8,
                       "available_controls": len(control_rows), "rng": "PCG64", "seed": 20260920,
                       "S": "input16, observed16; same cells", "O": "input8, observed8; disjoint cells"},
        "inference": {"model": "PCA_a100_r128_g0", "native_output": "effect",
                      "uses_control_cells": False, "actual_predictor_calls": calls,
                      "prediction_bank_loaded": False, "control_cell_forwards": 0,
                      "prediction_identical_across_control_allocations": True,
                      "equivalent_state": "control + native effect; arithmetic re-encoding only",
                      "zero_baseline": "NC has an all-zero effect and no fitted weights"},
        "representation_max_loss_difference": max(abs(r["PCA_native_effect_loss"] - r["PCA_equivalent_state_loss"]) for r in scores),
        "input_binding": input_records, "bundled_assets": manifest,
        "scores": scores, "elapsed_seconds": time.perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".real-model-", dir=output.parent))
    try:
        for filename, rows in (("scores.tsv", scores), ("cell_membership.tsv", membership)):
            with (staging / filename).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
        np.save(staging / "inferred_effects.npy", effects, allow_pickle=False)
        write_json(staging / "RESULTS.json", result)
        # replace() can replace an empty directory, never a populated result.
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New or empty output directory")
    parser.add_argument("--input", type=Path, help="Input JSON; defaults to32 bundled real training cells")
    args = parser.parse_args()
    try:
        result = run(args.output, args.input)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"Input/output error: {exc}\n")
    print(json.dumps({"output": str(args.output), "predictor_calls": result["inference"]["actual_predictor_calls"],
                      "native_output": "effect; control-independent", "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
