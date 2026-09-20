"""Reference-operation planner and coordinate scorer.

The planner and numerical functions derive from the released
reference-role-value-v1/code/role_actions.py. This installed interface also
rejects unknown contract fields, names native-effect selection explicitly, and
binds storage-baseline values. It does not attest historical generation.
"""

from __future__ import annotations

import hashlib

import json

import numpy as np



def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def array_hash(array: np.ndarray) -> str:
    a = np.ascontiguousarray(array, dtype="<f8")
    h = hashlib.sha256()
    h.update(json.dumps(list(a.shape), separators=(",", ":")).encode())
    h.update(a.tobytes())
    return h.hexdigest()

def _native_encoding(contract: dict) -> str:
    return "state" if contract["output_kind"] == "state" else "effect"

def _active_storage(contract: dict) -> bool:
    return contract["storage_encoding"] != _native_encoding(contract)

def validate_contract(contract: dict) -> dict:
    """Validate role requirements without looking at prediction or target values."""
    require(isinstance(contract, dict), "Contract must be an object")
    allowed = {"task", "output_kind", "input_conditioned", "storage_encoding",
               "model_input_id", "prediction_reference_id", "observation_reference_id",
               "storage_baseline_id", "native_state_baseline_id",
               "metric_coordinate_id", "candidate_coordinate_id"}
    require(not set(contract) - allowed, f"Unknown contract fields: {sorted(set(contract) - allowed)}")
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
    effect_action = ("unused" if not after_effect else
                     "select_new_native_effect_without_centring" if not state and reload_prediction else
                     "recentre" if state and (task_change or cp_change or reload_prediction) else "preserve")
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
               "storage_baseline_sha256": array_hash(_reference(references, baseline_id, values.shape[1])) if baseline_id else None,
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
    baseline_hash = array_hash(_reference(references, baseline_id, values.shape[1])) if baseline_id else None
    require(payload.get("storage_baseline_sha256") == baseline_hash, "Payload storage baseline values changed")
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
