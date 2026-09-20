"""Plan reference changes and replay a small, genuine benchmark example.

The planner uses declared dependencies. Numeric replay starts from cached model
outputs, decodes their storage representation, and scores canonical coordinates.
It does not establish that an external predictor used its declared input cells.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parents[1]
PROTOCOL_SHA256 = "e65613d2f393663cb324535fac93efbc01ebc818865c6f063626a61d02461689"
PBMC_ADDENDUM_SHA256 = "bf1c1bc927321ea6abb62ccf3d0ca103d8e5fceb5e9236af0537f424f2455fef"
PRIOR_BINDING_SHA256 = "36db8b187b9cdfdc7c9fc434e855cd2d78bbad90123906e39b6fef1236b166ab"
METRICS = ("standardized_MSE", "raw_RMSE", "perturbation_centred_Pearson", "centroid_accuracy")
TOLERANCE = 1e-10


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_hash(array: np.ndarray) -> str:
    a = np.ascontiguousarray(array, dtype="<f8")
    h = hashlib.sha256()
    h.update(json.dumps(list(a.shape), separators=(",", ":")).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _native_encoding(contract: dict) -> str:
    return "state" if contract["output_kind"] == "state" else "effect"


def _active_storage(contract: dict) -> bool:
    return contract["storage_encoding"] != _native_encoding(contract)


def validate_contract(contract: dict) -> dict:
    """Validate role requirements without looking at prediction or target values."""
    c = dict(contract)
    require(c.get("task") in ("treated_state", "reference_effect"), "Unknown or missing task")
    require(c.get("output_kind") in ("state", "native_effect"), "Unknown or missing output_kind")
    require(type(c.get("input_conditioned")) is bool, "input_conditioned must be boolean")
    require(c.get("storage_encoding") in ("state", "effect"), "Unknown or missing storage_encoding")
    for field in ("model_input_id", "prediction_reference_id", "observation_reference_id", "storage_baseline_id", "native_state_baseline_id"):
        c.setdefault(field, None)
        require(c[field] is None or (isinstance(c[field], str) and bool(c[field])), f"Invalid {field}")
    active = []
    if c["input_conditioned"]:
        active.append("model_input_id")
    if c["task"] == "reference_effect":
        active.append("observation_reference_id")
        if c["output_kind"] == "state":
            active.append("prediction_reference_id")
    elif c["output_kind"] == "native_effect":
        active.append("native_state_baseline_id")
    if _active_storage(c):
        active.append("storage_baseline_id")
    for field in active:
        require(c[field] is not None, f"Missing active declaration: {field}")
    # The replay decodes exports before scoring. Its metric artifacts always
    # describe the aligned state coordinates used by the matched Systema panel.
    for field in ("metric_coordinate_id", "candidate_coordinate_id"):
        c.setdefault(field, "aligned_state")
        require(c[field] == "aligned_state", f"Unsupported {field}: decode to aligned_state before scoring")
    return c


def _dependencies(c: dict) -> dict:
    effect = c["task"] == "reference_effect"
    state = c["output_kind"] == "state"
    return {
        "model_input": "prediction_output" if c["input_conditioned"] else "unused",
        "prediction_reference": "prediction_effect" if effect and state else "unused",
        "observation_reference": "observed_effect_and_metric_rendering" if effect else "unused",
        "storage_baseline": "decode_export" if _active_storage(c) else "unused",
        "native_state_baseline": "native_effect_state_lift" if not effect and not state else "unused",
        "metric_reference": "reference_centred_Pearson",
        "candidate_collection": "centroid_retrieval",
    }


def plan_change(before: dict, after: dict) -> dict:
    """Return executable consequences of a declared reference/encoding change.

    Changes of fitted pipeline or its dependence contract are outside this API.
    Feature coordinates, scales, metric centre and retrieval collection stay
    fixed; this function does not plan edits to those artifact values. Missing
    inactive role IDs are valid. The plan never reads scores or arrays.
    """
    b, a = validate_contract(before), validate_contract(after)
    for field in ("output_kind", "input_conditioned"):
        require(b[field] == a[field], f"A change of {field} requires a separate pipeline contract")
    changed = sorted(key for key in set(b) | set(a) if b.get(key) != a.get(key))
    reload_prediction = a["input_conditioned"] and b["model_input_id"] != a["model_input_id"]
    task_change = b["task"] != a["task"]
    after_effect = a["task"] == "reference_effect"
    state = a["output_kind"] == "state"
    cp_change = after_effect and state and b["prediction_reference_id"] != a["prediction_reference_id"]
    co_change = after_effect and b["observation_reference_id"] != a["observation_reference_id"]
    lift_change = (not after_effect and not state and b["native_state_baseline_id"] != a["native_state_baseline_id"])
    scope = ("new_task" if task_change else "new_observed_realization" if co_change
             else "new_prediction_effect" if cp_change else "new_prediction_state" if lift_change else "same_declared_quantity")
    effect_action = ("unused" if not after_effect else "recentre" if state and (task_change or cp_change or reload_prediction) else "preserve")
    target_action = "rebuild" if task_change or co_change else "preserve"
    recipe = ["Load the cached prediction for the new model input, or generate it if absent."
              if reload_prediction else "Reuse the declared model-native prediction."]
    if _active_storage(a):
        recipe.append("Decode the stored array with its declared storage baseline before constructing the scored prediction.")
    if after_effect:
        recipe.append("Construct the predicted effect by subtracting the declared prediction reference from the state output."
                      if state else "Preserve the emitted native effect; do not subtract a prediction reference.")
        recipe.append("Construct the observed effect from treated expression and the declared observation reference."
                      if target_action == "rebuild" else "Retain the declared observed-effect realization.")
        recipe.append("For these Systema metrics, lift the effect by the observation reference and score in the declared state coordinates.")
    elif not state:
        recipe.append("Lift the native effect with native_state_baseline_id; Cpred and Cobs remain unused.")
    else:
        recipe.append("Compare the decoded state prediction with treated expression; Cpred and Cobs remain unused.")
    if scope != "same_declared_quantity":
        recipe.append("Record the changed prediction or target declaration and keep its result identifiable.")
    return {
        "prediction_action": "load_or_generate_for_new_input" if reload_prediction else "reuse",
        "prediction_effect_action": effect_action,
        "observed_target_action": target_action,
        "comparison_scope": scope,
        "requires_task_declaration": bool(task_change),
        "requires_reference_record_update": bool(cp_change or co_change),
        "requires_prediction_definition_update": bool(task_change or cp_change or lift_change),
        "requires_score_recompute": bool(reload_prediction or task_change or cp_change or co_change or lift_change),
        "change_flags": {
            "task_type_changed": bool(task_change),
            "model_input_changed": bool(reload_prediction),
            "prediction_reference_changed": bool(cp_change),
            "observation_reference_changed": bool(co_change),
            "native_state_baseline_changed": bool(lift_change),
            "prediction_definition_changed": bool(task_change or cp_change or lift_change),
            "observed_realization_changed": bool(task_change or co_change),
            "storage_encoding_changed": b["storage_encoding"] != a["storage_encoding"],
            "storage_baseline_changed": bool((_active_storage(b) or _active_storage(a)) and b["storage_baseline_id"] != a["storage_baseline_id"]),
        },
        "changed_fields": changed,
        "role_dependencies": {"before": _dependencies(b), "after": _dependencies(a)},
        "recipe": recipe,
    }


def _reference(references: dict, reference_id: str | None, genes: int) -> np.ndarray:
    require(reference_id is not None and reference_id in references, f"Unknown or missing reference: {reference_id}")
    ref = np.asarray(references[reference_id], dtype=np.float64)
    require(ref.shape == (genes,) and bool(np.isfinite(ref).all()), f"Invalid reference array: {reference_id}")
    return ref


def _rows(array: np.ndarray) -> np.ndarray:
    a = np.asarray(array, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    require(a.ndim == 2 and a.shape[0] > 0 and a.shape[1] > 0 and bool(np.isfinite(a).all()), "Prediction must be finite fits × genes")
    return a


def encode_prediction(native: np.ndarray, contract: dict, references: dict) -> tuple[np.ndarray, dict]:
    """Encode a model-native output and return its consistency-check metadata."""
    c, values = validate_contract(contract), _rows(native)
    stored = values.copy()
    baseline_id = c["storage_baseline_id"] if _active_storage(c) else None
    if _active_storage(c):
        baseline = _reference(references, baseline_id, values.shape[1])
        stored += baseline if c["output_kind"] == "native_effect" else -baseline
    payload = {"model_input_id": c["model_input_id"], "output_kind": c["output_kind"],
               "storage_encoding": c["storage_encoding"], "storage_baseline_id": baseline_id,
               "array_sha256": array_hash(stored)}
    return stored, payload


def decode_prediction(stored: np.ndarray, contract: dict, references: dict, payload: dict) -> np.ndarray:
    """Check supplied provenance consistency and recover the model-native output."""
    c, values = validate_contract(contract), _rows(stored)
    require(payload.get("output_kind") == c["output_kind"], "Payload output kind differs from contract")
    require(payload.get("storage_encoding") == c["storage_encoding"], "Payload storage encoding differs from contract")
    if c["input_conditioned"]:
        require(payload.get("model_input_id") == c["model_input_id"], "Cached prediction has a stale or missing input ID")
    baseline_id = c["storage_baseline_id"] if _active_storage(c) else None
    require(payload.get("storage_baseline_id") == baseline_id, "Payload storage baseline differs from contract")
    require(payload.get("array_sha256") == array_hash(values), "Stored prediction digest mismatch")
    native = values.copy()
    if _active_storage(c):
        baseline = _reference(references, baseline_id, values.shape[1])
        native += -baseline if c["output_kind"] == "native_effect" else baseline
    return native


def canonical_components(native: np.ndarray, contract: dict, references: dict,
                         observed: np.ndarray, scales: np.ndarray, *, candidates: np.ndarray | None = None,
                         metric_reference: np.ndarray | None = None, correct_index: int = 0) -> dict:
    """Build the declared prediction/target and matched Systema metric vectors.

    Reference effects have a canonical effect prediction/target, while the
    matched Systema metric implementation uses their explicitly aligned state
    coordinates. Metric centre and retrieval candidates are supplied in those
    coordinates. Kang's capsule has neither artifact, so those two scores are NA.
    """
    c, values = validate_contract(contract), _rows(native)
    genes = values.shape[1]
    y, scale = np.asarray(observed, dtype=np.float64), np.asarray(scales, dtype=np.float64)
    require(y.shape == scale.shape == (genes,) and bool(np.isfinite(y).all()), "Invalid observed target or scales")
    require(bool(np.isfinite(scale).all() and (scale > 0).all()), "Scales must be finite and positive")
    if c["task"] == "reference_effect":
        observation_ref = _reference(references, c["observation_reference_id"], genes)
        prediction = values - _reference(references, c["prediction_reference_id"], genes) if c["output_kind"] == "state" else values.copy()
        target = y - observation_ref
        metric_prediction = prediction + observation_ref
    else:
        prediction = values.copy() if c["output_kind"] == "state" else values + _reference(references, c["native_state_baseline_id"], genes)
        target, metric_prediction = y.copy(), prediction.copy()
    residual = prediction - target
    scores = np.full((values.shape[0], 4), np.nan)
    scores[:, 0] = np.mean(np.square(residual / scale), axis=1)
    scores[:, 1] = np.sqrt(np.mean(np.square(residual), axis=1))
    if metric_reference is not None:
        centre = np.asarray(metric_reference, dtype=np.float64)
        require(centre.shape == (genes,) and bool(np.isfinite(centre).all()), "Invalid metric reference")
        p, truth = metric_prediction - centre, y - centre
        p, truth = p - p.mean(axis=1, keepdims=True), truth - truth.mean()
        denominator = np.linalg.norm(p, axis=1) * np.linalg.norm(truth)
        np.divide(np.sum(p * truth, axis=1), denominator, out=scores[:, 2], where=denominator > 0)
        scores[:, 2] = np.clip(scores[:, 2], -1.0, 1.0)
    if candidates is not None:
        collection = np.asarray(candidates, dtype=np.float64)
        require(collection.ndim == 2 and collection.shape[1] == genes and len(collection) > 1, "Invalid retrieval candidates")
        require(bool(np.isfinite(collection).all()), "Retrieval candidates must all be finite")
        require(0 <= correct_index < len(collection) and np.array_equal(collection[correct_index], y), "Correct retrieval target differs from observed target")
        distances = np.linalg.norm(metric_prediction[:, None, :] - collection[None, :, :], axis=2)
        scores[:, 3] = np.sum(distances > distances[:, correct_index, None], axis=1) / (len(collection) - 1)
    return {"native_prediction": values.copy(), "prediction": prediction, "target": target,
            "residual": residual, "metric_prediction": metric_prediction, "metric_target": y.copy(), "scores": scores}


def _load_adapter(path: Path):
    spec = importlib.util.spec_from_file_location("role_demo_input_adapter", path)
    require(spec is not None and spec.loader is not None, "Cannot load source adapter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_demo(out: Path) -> None:
    """Extract metadata-selected task zero from hash-verified historical caches."""
    require(not out.exists(), "Demo output directory already exists")
    require(file_hash(HERE / "PROTOCOL.json") == PROTOCOL_SHA256, "New protocol hash mismatch")
    require(file_hash(HERE / "PBMC_COVERAGE_ADDENDUM.json") == PBMC_ADDENDUM_SHA256, "PBMC coverage addendum hash mismatch")
    prior = HERE.parent / "reference_coupling_validation_v1"
    adapter = _load_adapter(prior / "code/input_adapter.py")
    require(file_hash(prior / "FORECAST_BINDING.json") == PRIOR_BINDING_SHA256, "Prior forecast binding mismatch")
    arrays, resource_meta = {}, {}
    out.mkdir(parents=True)
    _json(out / "BUILD_BINDING.json", {"protocol_sha256": PROTOCOL_SHA256, "pbmc_coverage_addendum_sha256": PBMC_ADDENDUM_SHA256,
          "prior_binding_sha256": PRIOR_BINDING_SHA256,
          "code_sha256": file_hash(Path(__file__)), "input_adapter_sha256": file_hash(prior / "code/input_adapter.py"),
          "selection": "Metadata task 0, allocation 0; Norman depth 8 all six models; Kang both models/all fits; PBMC depth 8 all eight models under its coverage addendum.",
          "access": "Retrospective existing-data extraction; protocol fixed before this script decoded targets."})
    for resource, prefix in (("Norman", "norman"), ("Kang", "kang"), ("PBMC", "pbmc")):
        planning = adapter.load_planning(resource)
        record = next(planning.iter_records())
        targets = adapter.load_targets(resource, freeze_path=prior / "FORECAST_BINDING.json", freeze_sha256=PRIOR_BINDING_SHA256)
        arrays[prefix + "_controls"] = np.asarray(record.controls).copy()
        arrays[prefix + "_scales"] = np.asarray(record.scales).copy()
        arrays[prefix + "_observed"] = targets[record.task].copy()
        models = []
        for i, model in enumerate(record.models):
            key = f"{prefix}_model_{i}"
            arrays[key] = np.asarray(model.values).copy()
            models.append({"name": model.name, "kind": model.kind, "input_conditioned": bool(model.input_conditioned),
                           "array_key": key, "fits": model.values.shape[0]})
        if resource == "Norman":
            arrays[prefix + "_candidates"] = np.stack([targets[task] for task in planning.metadata["task_labels"]])
            for name in ("TRAIN_control", "TRAIN_perturbation", "native_effect_training_baseline"):
                arrays[prefix + "_" + name] = np.asarray(planning.metadata[name]).copy()
        resource_meta[resource] = {"prefix": prefix, "task": record.task, "task_id": record.task_id, "unit": record.unit,
                                  "allocation": record.allocation, "depth": record.depth, "models": models,
                                  "correct_candidate_index": 0 if resource == "Norman" else None,
                                  "source_bindings": planning.metadata["source_bindings"],
                                  "metrics_available": list(METRICS if resource == "Norman" else METRICS[:2])}
    np.savez_compressed(out / "profiles.npz", **arrays)
    _json(out / "manifest.json", {"schema_version": 1, "data_file": "profiles.npz", "data_sha256": file_hash(out / "profiles.npz"),
          "protocol_sha256": PROTOCOL_SHA256, "pbmc_coverage_addendum_sha256": PBMC_ADDENDUM_SHA256, "resources": resource_meta,
          "array_manifest": {key: {"shape": list(value.shape), "sha256": array_hash(value)} for key, value in arrays.items()},
          "scope": "Genuine cached profiles selected by metadata, not known benchmark errors or a human usability study.",
          "native_effect": "Native ridge removes its saved TRAIN lift in the original adapter; direct-state rendering uses TRAIN_control.",
          "metric_coordinates": "Aligned state; fixed TRAIN_perturbation and all 55 observed Norman targets. Kang and PBMC have only the two distance metrics.",
          "cellflow": "Calculate each of five fit scores, then average scores."})


def _environment(manifest: dict, arrays: dict, resource: str) -> dict:
    meta = manifest["resources"][resource]
    prefix = meta["prefix"]
    refs = {f"{resource}.B{i+1}": arrays[prefix + "_controls"][i] for i in range(3)}
    if resource == "Norman":
        refs["Norman.TRAIN_control"] = arrays[prefix + "_TRAIN_control"]
        refs["Norman.native_effect_training_baseline"] = arrays[prefix + "_native_effect_training_baseline"]
    return {"references": refs, "observed": arrays[prefix + "_observed"], "scales": arrays[prefix + "_scales"],
            "candidates": arrays.get(prefix + "_candidates"), "metric_reference": arrays.get(prefix + "_TRAIN_perturbation"),
            "correct_index": 0}


def _base_contract(resource: str, model: dict) -> dict:
    return {"task": "treated_state" if resource == "Kang" else "reference_effect", "output_kind": model["kind"],
            "input_conditioned": model["input_conditioned"], "model_input_id": f"{resource}.B1",
            "prediction_reference_id": f"{resource}.B1", "observation_reference_id": f"{resource}.B2",
            "storage_encoding": "state" if model["kind"] == "state" else "effect", "storage_baseline_id": None,
            "native_state_baseline_id": "Norman.TRAIN_control" if resource == "Norman" else None,
            "metric_coordinate_id": "aligned_state", "candidate_coordinate_id": "aligned_state"}


def _native(arrays: dict, model: dict, contract: dict) -> np.ndarray:
    values = arrays[model["array_key"]]
    if model["kind"] == "native_effect":
        return values.copy()
    block = int(contract["model_input_id"].rsplit("B", 1)[1]) - 1 if model["input_conditioned"] else 0
    return values[:, block, :].copy()


def _cases(manifest: dict) -> list[dict]:
    cases = []
    for resource, meta in manifest["resources"].items():
        for model_i, model in enumerate(meta["models"]):
            base = _base_contract(resource, model)
            changes = ({"unused_roles_Kang": {"prediction_reference_id": "Kang.B2", "observation_reference_id": "Kang.B3"}}
                       if resource == "Kang" else {
                           "input_replacement": {"model_input_id": f"{resource}.B2"},
                           "paired_input_effect": {"model_input_id": f"{resource}.B2", "prediction_reference_id": f"{resource}.B2"},
                           "prediction_baseline": {"prediction_reference_id": f"{resource}.B3"},
                           "observation_baseline": {"observation_reference_id": f"{resource}.B3"},
                           "common_scoring_reference": {"prediction_reference_id": f"{resource}.B2", "observation_reference_id": f"{resource}.B2"},
                           "same_target_storage_encoding": {"storage_encoding": "effect" if model["kind"] == "state" else "state", "storage_baseline_id": f"{resource}.B3"},
                       })
            if resource == "Norman" or (resource == "PBMC" and model["kind"] == "state"):
                changes["task_switch"] = {"task": "treated_state"}
            for operation, delta in changes.items():
                before = dict(base)
                if operation == "common_scoring_reference":
                    before["observation_reference_id"] = f"{resource}.B1"
                after = {**before, **delta}
                cases.append({"case_id": f"{resource}_{model_i}_{operation}", "resource": resource, "model": model["name"],
                              "model_index": model_i, "operation": operation, "before": before, "after": after,
                              "case_origin": "metadata_fixed_operation_on_genuine_cached_profiles",
                              "coverage": "supplemental_PBMC_addendum" if resource == "PBMC" else "primary_protocol"})
    return cases


def _components(native: np.ndarray, contract: dict, env: dict) -> dict:
    return canonical_components(native, contract, env["references"], env["observed"], env["scales"],
                                candidates=env["candidates"], metric_reference=env["metric_reference"], correct_index=env["correct_index"])


def _difference(a: np.ndarray, b: np.ndarray) -> float:
    require(a.shape == b.shape, "Component shape mismatch")
    require(np.array_equal(np.isnan(a), np.isnan(b)), "Undefined score support changed")
    use = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[use] - b[use]))) if bool(use.any()) else 0.0


def _scores_json(scores: np.ndarray) -> dict:
    average = scores.mean(axis=0)
    return {name: float(value) if np.isfinite(value) else None for name, value in zip(METRICS, average)}


def _stress_tests(manifest: dict, arrays: dict) -> list[dict]:
    """Injected integrity failures; semantic faults need an external reference."""
    model = manifest["resources"]["Norman"]["models"][0]
    env = _environment(manifest, arrays, "Norman")
    base = _base_contract("Norman", model)
    converted = {**base, "storage_encoding": "effect", "storage_baseline_id": "Norman.B1"}
    native = _native(arrays, model, base)
    stored, payload = encode_prediction(native, converted, env["references"])
    tests = []
    for name, contract in (("wrong_storage_baseline", {**converted, "storage_baseline_id": "Norman.B2"}),
                           ("missing_storage_baseline", {**converted, "storage_baseline_id": None}),
                           ("stale_model_input_id", {**converted, "model_input_id": "Norman.B2"}),
                           ("inconsistent_metric_coordinates", {**converted, "metric_coordinate_id": "unconverted_effect"})):
        try:
            decode_prediction(stored, contract, env["references"], payload)
        except ValueError as error:
            tests.append({"test": name, "origin": "injected_software_fault", "result": "rejected_inconsistent_contract_or_payload", "message": str(error)})
        else:
            raise AssertionError(f"Injected inconsistency accepted: {name}")
    native_model = manifest["resources"]["Norman"]["models"][5]
    native_contract = _base_contract("Norman", native_model)
    correct = _native(arrays, native_model, native_contract)
    wrong = correct - env["references"]["Norman.B1"]
    wrong_stored, wrong_payload = encode_prediction(wrong, native_contract, env["references"])
    decoded = decode_prediction(wrong_stored, native_contract, env["references"], wrong_payload)
    difference = _difference(decoded, correct)
    require(difference > TOLERANCE, "Native double-centring stress test has no realized vector change")
    tests.append({"test": "native_double_centring_with_self_consistent_payload", "origin": "injected_software_fault",
                  "result": "accepted_provenance_consistency_check_but_failed_external_canonical_check",
                  "canonical_prediction_max_abs_error": difference,
                  "meaning": "Declarations and self-consistent hashes cannot identify arbitrary scientifically wrong values without a trusted expected output."})
    return tests


def _role_omissions(cases: list[dict]) -> list[dict]:
    rows = []
    for case in cases:
        for field in ("model_input_id", "prediction_reference_id", "observation_reference_id"):
            contract = {**case["after"], field: None}
            try:
                validate_contract(contract)
            except ValueError as error:
                rows.append({"case_id": case["case_id"], "field": field, "result": "missing_active_information", "message": str(error)})
            else:
                rows.append({"case_id": case["case_id"], "field": field, "result": "valid_unused_role_omission", "message": None})
    return rows


def _reader_report(out: Path, outputs: list[dict], stresses: list[dict], omissions: list[dict]) -> None:
    max_error = max(max(row["canonical_max_abs_errors"].values()) for row in outputs)
    lines = [
        "# Reference changes: replay results", "",
        f"The replay executed {len(outputs)} declared benchmark changes using saved, real predictions. Each action recovered the expected canonical prediction, observed target, residual and available scores; the largest component discrepancy was {max_error:.3g}.", "",
        "The example covers six Norman pipelines, CPA, scGen, CellOT and five native-effect pipelines in PBMC, and ridge plus the five saved CellFlow fits in Kang. The original protocol specifies 44 Norman/Kang cases; a separate coverage addendum specifies the 51 PBMC cases. These are organizer operations on cached data, not errors found in a published benchmark.", "",
        "## What each operation required", "",
        "A new-input selection loads an already saved prediction for the changed input. Reuse preserves the model-native output; effect construction and scoring can still change. The final column compares full canonical residual vectors at the fixed numerical tolerance.", "",
        "| Operation | Cases | Reuse prediction | Select new-input prediction | Residual changed |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "input_replacement": "Replace model input",
        "paired_input_effect": "Replace model input and paired effect baseline",
        "prediction_baseline": "Replace prediction baseline",
        "observation_baseline": "Replace observation baseline",
        "common_scoring_reference": "Move both scoring references together",
        "same_target_storage_encoding": "Re-encode the same delivered prediction",
        "task_switch": "Declare a state task in place of an effect task",
        "unused_roles_Kang": "Change unused references in Kang state scoring",
    }
    for operation, label in labels.items():
        rows = [row for row in outputs if row["operation"] == operation]
        reuse = sum(row["plan"]["prediction_action"] == "reuse" for row in rows)
        changed = sum(row["before_after_max_abs_changes"]["residual"] > TOLERANCE for row in rows)
        lines.append(f"| {label} | {len(rows)} | {reuse} | {len(rows) - reuse} | {changed} |")
    lines += ["",
        "For a state-output pipeline, moving the same scoring reference on both sides preserves the residual. Native effects retain their emitted meaning and do not acquire a prediction-reference subtraction. Changing an observation reference creates a new realized effect target; it does not require a new predictor output. Changes of task type are recorded separately from changes within a reference-effect task.", "",
        "All 16 storage-conversion or unused-role cases preserved their full canonical components and scores. Norman uses all four matched Systema metric formulas. PBMC and Kang report standardized MSE and raw RMSE because this capsule does not supply their Systema training-centroid and candidate artifacts. CellFlow scores are computed for each fit before averaging.", "",
        "## Inspect an individual change", "",
        "Open [action_summary.tsv](action_summary.tsv) for one row per resource, pipeline and operation. [actions.json](actions.json) contains the executable plan, before/after scores, declared input and storage provenance, and component discrepancies. [cases.json](cases.json) records both contracts; [components.npz](components.npz) retains the canonical vectors and fit-level scores.", "",
        "The planner covers the listed reference, storage and native-state-lift dependencies. Feature coordinates, scales, metric centre and retrieval collection are held fixed. Storage exports are decoded before metrics are calculated. An organizer changing metric artifacts must specify that change separately.", "",
        "## Missing declarations and injected checks", "",
        f"Deleting each role ID from each case produced {sum(row['result'] == 'missing_active_information' for row in omissions)} explicit missing-information results and {sum(row['result'] == 'valid_unused_role_omission' for row in omissions)} valid omissions of unused roles. See [role_omission_checks.json](role_omission_checks.json). These statuses test the declared dependencies; they are not a diagnostic-accuracy score for Systema.", "",
        "Four injected inconsistencies were rejected: an incompatible storage baseline, a missing active storage baseline, a stale model-input declaration and unsupported metric coordinates. A native effect deliberately centred twice passed its self-consistent metadata check, but failed comparison with the trusted canonical output. That case shows why consistent declarations alone cannot verify arbitrary prediction values. See [injected_stress_tests.json](injected_stress_tests.json).", "",
        "This is a cached-array replay: it runs no model inference or training and does not measure analyst time. Repeated evaluation supplied with the same role definitions and arrays can recover the same numerical results. The demonstrated value is an explicit, executable route from a benchmark change to the operations it requires.", "",
    ]
    (out / "report.md").write_text("\n".join(lines))


def run_demo(data: Path, out: Path) -> None:
    """Replay the bundled example without its large historical source caches."""
    started = time.monotonic()
    require(not out.exists(), "Choose an output directory that does not already exist")
    manifest = json.loads((data / "manifest.json").read_text())
    require(manifest["protocol_sha256"] == PROTOCOL_SHA256, "Demo protocol binding differs")
    archive_path = data / manifest["data_file"]
    require(file_hash(archive_path) == manifest["data_sha256"], "Demo archive digest mismatch")
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    require(set(arrays) == set(manifest["array_manifest"]), "Demo array inventory differs")
    for key, expected in manifest["array_manifest"].items():
        require(list(arrays[key].shape) == expected["shape"] and array_hash(arrays[key]) == expected["sha256"], f"Demo array changed: {key}")
    cases = _cases(manifest)
    out.mkdir(parents=True)
    _json(out / "EXECUTION_BINDING.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
          "protocol_sha256": PROTOCOL_SHA256, "code_sha256": file_hash(Path(__file__)),
          "demo_manifest_sha256": file_hash(data / "manifest.json"), "data_sha256": manifest["data_sha256"],
          "python": sys.version, "numpy": np.__version__, "selected_valid_cases": len(cases)})
    _json(out / "cases.json", cases)
    outputs, components, summary_rows = [], {}, []
    for case in cases:
        resource = case["resource"]
        model = manifest["resources"][resource]["models"][case["model_index"]]
        env = _environment(manifest, arrays, resource)
        before, after = case["before"], case["after"]
        plan = plan_change(before, after)
        native_before = _native(arrays, model, before)
        stored_before, payload_before = encode_prediction(native_before, before, env["references"])
        decoded_before = decode_prediction(stored_before, before, env["references"], payload_before)
        # This is execution of the requested dependency action. Loading a cached
        # new-input slice is never counted as a newly run model inference.
        selected = _native(arrays, model, after) if plan["prediction_action"] != "reuse" else decoded_before.copy()
        stored_after, payload_after = encode_prediction(selected, after, env["references"])
        decoded_after = decode_prediction(stored_after, after, env["references"], payload_after)
        cb, ca = _components(decoded_before, before, env), _components(decoded_after, after, env)
        expected = _components(_native(arrays, model, after), after, env)
        errors = {key: _difference(ca[key], expected[key]) for key in ca}
        require(max(errors.values()) <= TOLERANCE, f"Action failed canonical reconstruction: {case['case_id']}")
        shifts = {key: _difference(cb[key], ca[key]) for key in ("native_prediction", "prediction", "target", "residual", "metric_prediction", "scores")}
        expected_invariance = case["operation"] in ("same_target_storage_encoding", "unused_roles_Kang")
        if expected_invariance:
            require(max(shifts.values()) <= TOLERANCE, f"Valid no-change operation changed canonical components: {case['case_id']}")
        if case["operation"] == "common_scoring_reference" and model["kind"] == "state":
            require(shifts["residual"] <= TOLERANCE, "Common scoring centre changed state residual")
        for phase, values in (("before", cb), ("after", ca)):
            for key, value in values.items():
                components[f"{case['case_id']}__{phase}__{key}"] = value
        components[f"{case['case_id']}__before__stored"] = stored_before
        components[f"{case['case_id']}__after__stored"] = stored_after
        row = {"case_id": case["case_id"], "resource": resource, "model": model["name"], "fits": model["fits"],
               "operation": case["operation"], "case_origin": case["case_origin"], "plan": plan,
               "before_payload": payload_before, "after_payload": payload_after,
               "before_scores": _scores_json(cb["scores"]), "after_scores": _scores_json(ca["scores"]),
               "canonical_max_abs_errors": errors, "before_after_max_abs_changes": shifts,
               "target_declaration_unchanged": before["task"] == after["task"] and (after["task"] == "treated_state" or before["observation_reference_id"] == after["observation_reference_id"]),
               "expected_full_invariance": expected_invariance, "action_verified": True,
               "manual_role_reasoning_parity": "same canonical arrays and definitions available for independent QA"}
        outputs.append(row)
        summary_rows.append({"case_id": case["case_id"], "resource": resource, "model": model["name"], "fits": model["fits"],
             "operation": case["operation"], "prediction_action": plan["prediction_action"],
             "prediction_effect_action": plan["prediction_effect_action"], "observed_target_action": plan["observed_target_action"],
             "comparison_scope": plan["comparison_scope"], "requires_score_recompute": plan["requires_score_recompute"],
             "action_max_abs_error": max(errors.values()), **{f"change_{key}": value for key, value in shifts.items()}})
    _json(out / "actions.json", outputs)
    np.savez_compressed(out / "components.npz", **components)
    with (out / "action_summary.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)
    stresses, omissions = _stress_tests(manifest, arrays), _role_omissions(cases)
    _json(out / "injected_stress_tests.json", stresses)
    _json(out / "role_omission_checks.json", omissions)
    _reader_report(out, outputs, stresses, omissions)
    receipt = {"status": "PASS", "valid_operation_cases": len(outputs), "operation_families": sorted({r["operation"] for r in outputs}),
               "models": {r: len(manifest["resources"][r]["models"]) for r in manifest["resources"]},
               "cached_new_input_selections": sum(r["plan"]["prediction_action"] != "reuse" for r in outputs),
               "reused_prediction_cases": sum(r["plan"]["prediction_action"] == "reuse" for r in outputs),
               "new_model_inferences": 0, "max_action_component_error": max(max(r["canonical_max_abs_errors"].values()) for r in outputs),
               "full_invariance_cases": sum(r["expected_full_invariance"] for r in outputs),
               "injected_stress_tests": len(stresses), "injected_integrity_rejections": 4,
               "injected_semantic_faults_needing_external_canonical_output": 1,
               "role_omission_checks": len(omissions),
               "missing_active_information": sum(r["result"] == "missing_active_information" for r in omissions),
               "valid_unused_role_omissions": sum(r["result"] == "valid_unused_role_omission" for r in omissions),
               "elapsed_seconds_from_cached_profiles": time.monotonic() - started,
               "files": {p.name: {"sha256": file_hash(p), "bytes": p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()},
               "scope": "Agent-executed cached-array demonstration. No human time comparison, new training, external benchmark bug prevalence or Systema diagnostic accuracy is measured."}
    _json(out / "RECEIPT.json", receipt)
    print(json.dumps(receipt, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-demo", help="Extract the fixed example from original hash-verified caches")
    build.add_argument("--out", type=Path, required=True)
    demo = sub.add_parser("demo", help="Replay the bundled example; no original caches or training required")
    demo.add_argument("--data", type=Path, default=HERE / "reader_demo")
    demo.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build-demo":
        build_demo(args.out)
    else:
        run_demo(args.data, args.out)


if __name__ == "__main__":
    main()
