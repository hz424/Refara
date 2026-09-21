"""Audit a real frozen scGen predictor on four prespecified control allocations."""
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

from reference_design import Prediction, score_references
from reference_design.conditioned_model import FrozenSCGen
from reference_design.real_model import normalize_full_counts

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "real_model_reference"


def capture(path):
    data = Path(path).read_bytes()
    return data, {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def table(data):
    return list(csv.DictReader(io.StringIO(data.decode()), delimiter="\t"))


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_input(path, model):
    config_bytes, binding = capture(path)
    config = json.loads(config_bytes)
    if config.get("schema") != "REFARA_REAL_MODEL_INPUT_V1" or config.get("normalization") != "full_source_CP10K_log1p":
        raise ValueError("Unknown input schema or normalization")
    if not isinstance(config.get("unit"), str) or not config["unit"].strip():
        raise ValueError("Input needs a nonempty unit label")
    records, buffers = {"input_json": binding}, {}
    for key in ("counts", "features", "cells"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f"Input requires {key}")
        buffers[key], records[key] = capture(path.parent / config[key])
    features = table(buffers["features"])
    if not features or any(set(row) != {"source_index", "feature_id"} for row in features):
        raise ValueError("Feature table needs source_index and feature_id")
    if [r["source_index"] for r in features] != list(map(str, range(len(features)))):
        raise ValueError("Source indices must describe input column order")
    shared_manifest = json.loads((SHARED / "MANIFEST.json").read_text())
    for name in ("source_features.tsv", "feature_axis.npy", "scales.npy"):
        _, actual = capture(SHARED / "data" / name)
        if actual != shared_manifest["assets"]["data/" + name]:
            raise ValueError("Shared training metadata differs")
    fitted_features = [r["feature_id"] for r in table((SHARED / "data/source_features.tsv").read_bytes())]
    axis = np.load(SHARED / "data/feature_axis.npy", allow_pickle=False)
    scales = np.load(SHARED / "data/scales.npy", allow_pickle=False)
    if tuple(map(str, axis)) != model.feature_ids or not np.array_equal(scales, model.scales):
        raise ValueError("Model and input feature/scale contracts disagree")
    counts = sparse.load_npz(io.BytesIO(buffers["counts"]))
    expression = normalize_full_counts(counts, [r["feature_id"] for r in features], fitted_features, axis)
    cells = table(buffers["cells"])
    if len(cells) != len(expression) or any(set(row) != {"cell_id", "role"} for row in cells):
        raise ValueError("Cells table must match count rows with cell_id and role")
    identifiers = [r["cell_id"] for r in cells]
    if len(set(identifiers)) != len(identifiers) or any(not x.strip() for x in identifiers):
        raise ValueError("Cell IDs must be nonempty and unique")
    if any(r["role"] not in ("control", "treated") for r in cells):
        raise ValueError("Unknown cell role")
    control_rows = np.array([i for i, r in enumerate(cells) if r["role"] == "control"])
    treated_rows = np.array([i for i, r in enumerate(cells) if r["role"] == "treated"])
    if len(control_rows) != 24 or len(treated_rows) != 8:
        raise ValueError("This audit requires exactly 24 controls and 8 treated cells")
    expected = {"input_json": "input.json", "counts": "counts.npz",
                "features": "source_features.tsv", "cells": "cells.tsv"}
    is_training_fixture = all(records[key] == shared_manifest["assets"]["data/" + name]
                              for key, name in expected.items())
    if path.resolve() == (SHARED / "data/input.json").resolve() and not is_training_fixture:
        raise ValueError("Bundled training input differs from its frozen manifest")
    return config, expression, control_rows, treated_rows, identifiers, records, is_training_fixture


def run(output, input_path=None):
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output must be a new or empty directory")
    started = time.perf_counter()
    manifest = json.loads((HERE / "MANIFEST.json").read_text())
    weight_record = manifest["assets"]["data/scgen_seed17.npz"]
    model = FrozenSCGen(HERE / "data/scgen_seed17.npz", expected_sha256=weight_record["sha256"])
    input_path = Path(input_path) if input_path else SHARED / "data/input.json"
    config, expression, controls, treated, cell_ids, binding, is_training_fixture = load_input(input_path, model)
    target = expression[treated].mean(axis=0, dtype=np.float64)
    scales = model.scales
    context, condition = config.get("context"), config.get("condition")
    rng = np.random.Generator(np.random.PCG64(20260920))
    orders = np.stack([rng.permutation(24) for _ in range(4)])
    points, control_means, state_means, diagnostics, atomic, membership = [], [], [], [], [], []
    operation_contract = (
        ("anchor_O", 0, 0, 2),
        ("input_only", 1, 0, 2),
        ("rescore_only", 0, 0, 1),
        ("matched_O_new_input", 1, 1, 2),
        ("shared_S", 0, 0, 0),
    )  # Name, model-input block, prediction-centering block, observation block.
    maximum_O_difference = 0.
    for allocation, order in enumerate(orders):
        block_rows = controls[order.reshape(3, 8)]
        predicted, means = [], []
        for block, rows in enumerate(block_rows):
            # Every block is genuinely re-encoded and decoded on its own input.
            predicted.append(model.predict_points(expression[rows], context=context, condition=condition,
                                                   feature_ids=model.feature_ids))
            means.append(expression[rows].mean(axis=0, dtype=np.float64))
            membership.extend(dict(allocation=allocation, block="ABC"[block], cell_id=cell_ids[int(i)],
                                   count_row=int(i)) for i in rows)
        predicted = np.stack(predicted)
        means = np.stack(means)
        states = predicted.mean(axis=1, dtype=np.float64)
        points.append(predicted); control_means.append(means); state_means.append(states)
        encoded_effects = states - means
        scored = score_references(target, means, [Prediction("scGen_native_state", states, "state"),
            Prediction("same_scGen_O_effect_encoding", encoded_effects, "effect")], scales=scales)
        baseline_loss = None
        for name, model_block, prediction_block, observation_block in operation_contract:
            index = scored.role_tuples.index((observation_block, prediction_block, model_block))
            loss = float(scored.atomic_mse[0, index])
            if name == "anchor_O":
                baseline_loss = loss
            state_distance = float(np.sqrt(np.mean(((states[model_block] - states[0]) / scales) ** 2)))
            diagnostics.append(dict(allocation=allocation, operation=name, model_input="ABC"[model_block],
                prediction_center="ABC"[prediction_block], observation_reference="ABC"[observation_block],
                state_prediction_key=f"allocation{allocation}:block{'ABC'[model_block]}",
                scaled_mean_prediction_RMS_from_anchor=state_distance,
                prediction_mean_sha256=array_hash(states[model_block]),
                loss=loss, loss_change_from_anchor=loss-baseline_loss,
                requires_changed_input_prediction=model_block != 0,
                rescore_only_additional_forwards=0 if name == "rescore_only" else None,
                interpretation="Diagnostic input change with both scoring references held fixed" if name == "input_only" else
                    "Same native prediction; observation-reference change only" if name == "rescore_only" else
                    "Matched model input and prediction-centering reference"))
        for index, (observation, prediction, input_block) in enumerate(scored.role_tuples):
            row = dict(allocation=allocation, observation_block="ABC"[observation],
                prediction_center_block="ABC"[prediction], model_input_block="ABC"[input_block],
                native_state_loss=float(scored.atomic_mse[0, index]),
                O_effect_encoding_loss=float(scored.atomic_mse[1, index]) if prediction == input_block else None,
                same_encoding_contract=prediction == input_block)
            if prediction == input_block:
                difference = abs(row["native_state_loss"] - row["O_effect_encoding_loss"])
                maximum_O_difference = max(maximum_O_difference, difference)
                if not np.isclose(row["native_state_loss"], row["O_effect_encoding_loss"], atol=1e-12, rtol=1e-12):
                    raise ValueError("Matched-baseline state/effect equivalence failed")
            atomic.append(row)
    # A separate call verifies the fixed-point caching contract numerically.
    whole_pool = model.predict_points(expression[controls], context=context, condition=condition,
                                      feature_ids=model.feature_ids)
    points = np.stack(points)
    bank_difference = max(float(np.max(np.abs(points[i] - whole_pool[order.reshape(3, 8)])))
                          for i, order in enumerate(orders))
    if not all(np.allclose(points[i], whole_pool[order.reshape(3, 8)], atol=2e-5, rtol=2e-5)
               for i, order in enumerate(orders)):
        raise ValueError("Point-bank/batch-shape parity failed")
    arrays = dict(native_state_points=points, input_control_means=np.stack(control_means),
                  native_state_means=np.stack(state_means), normalized_input_cells=expression,
                  control_rows=controls, treated_rows=treated, allocation_control_offsets=orders,
                  whole_pool_state_points=whole_pool, treated_mean=target, scales=scales)
    result = dict(schema="REFARA_REAL_CONDITIONED_AUDIT_V1", status="PASS_EXECUTED_MECHANISM_AUDIT",
        scientific_confirmation=False, purpose="Mechanism and usability on previously used data; no generalization or ranking claim",
        input_scope="Bundled previously used training cells" if is_training_fixture else
                    "User-supplied input; training provenance and biological independence not authenticated",
        is_bundled_training_fixture=is_training_fixture,
        unit=config["unit"], context=context, condition=condition,
        model=dict(family="scGen", seed=17, native_output="state", uses_control_expression=True,
                   source_checkpoint_sha256=model.metadata["source_checkpoint_sha256"],
                   exported_checkpoint_sha256=model.checkpoint_sha256),
        input_binding=binding, allocation=dict(count=4, controls=24, block_size=8, treated=8, seed=20260920,
                   rng="PCG64", selection="All four predeclared permutations retained; no selection for reversals"),
        diagnostics=diagnostics,
        O_encoding=dict(maximum_loss_difference=maximum_O_difference,
                        scope="Same scGen prediction converted to an effect using its matched input baseline; not a separately trained effect model. Equivalence is checked only when prediction center equals model input."),
        inference=dict(actual_calls=model.calls,actual_control_cell_forwards=model.control_cell_forwards,
                       unique_physical_controls=24, allocation_inference_calls=12,
                       allocation_control_cell_forwards=96, cache_contract_check_forwards=24,
                       rescore_only_additional_forwards=0, bank_max_absolute_difference=bank_difference,
                       cache_contract="Eval point model; same cell prediction is subset-independent up to measured float32 rounding. General set-dependent models require separate inference."),
        arrays={name:dict(shape=list(value.shape),dtype=str(value.dtype),sha256=array_hash(value)) for name,value in arrays.items()},
        elapsed_seconds=time.perf_counter()-started)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".conditioned-model-", dir=output.parent))
    try:
        for filename, rows in (("operation_scores.tsv",diagnostics),("role_scores.tsv",atomic),("control_membership.tsv",membership)):
            with (staging / filename).open("w",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]),delimiter="\t",lineterminator="\n")
                writer.writeheader();writer.writerows(rows)
        np.savez_compressed(staging / "inference_arrays.npz", **arrays)
        result["array_archive"] = capture(staging / "inference_arrays.npz")[1]
        write_json(staging / "RESULTS.json", result)
        os.replace(staging, output)
    finally:
        if staging.exists():shutil.rmtree(staging)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--input",type=Path,help="Same full-count JSON schema as real_model_reference; requires 24 controls and 8 treated")
    args=parser.parse_args()
    try:result=run(args.output,args.input)
    except (ValueError,KeyError,OSError) as exc:parser.exit(2,f"Input/output error: {exc}\n")
    print(json.dumps(dict(status=result["status"],actual_calls=result["inference"]["actual_calls"],
        control_cell_forwards=result["inference"]["actual_control_cell_forwards"],
        O_encoding_max_loss_difference=result["O_encoding"]["maximum_loss_difference"],
        elapsed_seconds=result["elapsed_seconds"])))


if __name__ == "__main__":
    main()
