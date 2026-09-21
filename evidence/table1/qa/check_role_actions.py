"""Independent action-contract truth table and direct numerical reconstruction."""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "2"

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import importlib.util
import itertools
import json
from pathlib import Path
import time

import numpy as np

from check_role_metrics import sha256, scalar_metrics


HERE = Path(__file__).resolve().parents[1]
PLAN_SHA256 = "1614c18e9b27332edfc78c6d5559e4a8a2332303504275956850b5278ff48690"
EDIT_FIELDS = ("model_input_id", "prediction_reference_id", "observation_reference_id", "storage_encoding", "storage_baseline_id", "native_state_baseline_id")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def base_contracts():
    for task in ("treated_state", "reference_effect"):
        for kind, conditioned in (("state", True), ("state", False), ("native_effect", False)):
            yield {"task": task, "output_kind": kind, "input_conditioned": conditioned,
                   "model_input_id": "B0", "prediction_reference_id": "B0", "observation_reference_id": "B0",
                   "storage_encoding": "state" if kind == "state" else "effect", "storage_baseline_id": "B0",
                   "native_state_baseline_id": "T0"}


def expected_plan(before, after):
    changed = {name for name in before if before[name] != after[name]}
    task_changed = "task" in changed
    reload = before["input_conditioned"] and "model_input_id" in changed
    effect_task = after["task"] == "reference_effect"
    state_output = after["output_kind"] == "state"
    active_lift_change = not effect_task and not state_output and "native_state_baseline_id" in changed
    observed_change = task_changed or (effect_task and "observation_reference_id" in changed)
    prediction_baseline_change = effect_task and state_output and "prediction_reference_id" in changed
    if task_changed:
        scope = "new_task"
    elif observed_change:
        scope = "new_observed_realization"
    elif prediction_baseline_change:
        scope = "new_prediction_effect"
    elif active_lift_change:
        scope = "new_prediction_state"
    else:
        scope = "same_declared_quantity"
    return {"prediction_action": "load_or_generate_for_new_input" if reload else "reuse",
            "observed_target_action": "rebuild" if observed_change else "preserve",
            "comparison_scope": scope,
            "requires_task_declaration": task_changed,
            "requires_reference_record_update": bool(prediction_baseline_change or (effect_task and "observation_reference_id" in changed)),
            "requires_prediction_definition_update": bool(task_changed or prediction_baseline_change or active_lift_change),
            "requires_score_recompute": bool(task_changed or active_lift_change or reload or observed_change or prediction_baseline_change)}


def independent_components(native, contract, references, observed, scales, candidates, metric_reference):
    if contract["task"] == "treated_state":
        prediction = native if contract["output_kind"] == "state" else native + references[contract["native_state_baseline_id"]]
        target = observed
        metric_prediction = prediction
    else:
        prediction = native - references[contract["prediction_reference_id"]] if contract["output_kind"] == "state" else native
        target = observed - references[contract["observation_reference_id"]]
        metric_prediction = prediction + references[contract["observation_reference_id"]]
    scores = scalar_metrics(metric_prediction, 0, candidates, scales, metric_reference)
    return {"prediction": prediction, "target": target, "residual": prediction - target,
            "metric_prediction": metric_prediction, "metric_target": observed, "scores": scores}


def run(args):
    started = time.monotonic()
    require(sha256(HERE / "qa/ACTION_CHECK_PLAN.json") == PLAN_SHA256, "Frozen QA plan changed")
    require(sha256(args.protocol) == args.protocol_sha256, "Protocol changed")
    specification = importlib.util.spec_from_file_location("tested_role_actions", args.module)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    counts, maxima = Counter(), defaultdict(float)
    case_rows = []

    def close(name, actual, expected):
        actual, expected = np.asarray(actual), np.asarray(expected)
        require(actual.shape == expected.shape, (name, actual.shape, expected.shape))
        require(np.array_equal(np.isnan(actual), np.isnan(expected)), name + " undefined mismatch")
        finite = np.isfinite(actual) & np.isfinite(expected)
        errors = np.abs(actual[finite] - expected[finite])
        maximum = float(errors.max()) if errors.size else 0.
        require(bool(np.all(errors <= 1e-10 + 1e-9 * np.abs(expected[finite]))), (name, maximum))
        counts[name] += actual.size
        maxima[name] = max(maxima[name], maximum)

    references = {
        "B0": np.array([0.2, -0.3, 0.7, 1.2, 0.0, 0.8, 0.1]),
        "B1": np.array([-0.6, 0.4, 0.1, 0.3, 1.4, -0.1, 0.6]),
        "T0": np.array([0.1, 0.2, 0.4, -0.1, 0.7, 0.1, 0.3]),
        "T1": np.array([0.4, 0.8, 0.1, 0.5, 0.2, -0.2, 0.9]),
    }
    candidates = np.array([[1.3, 0.1, 0.8, -0.2, 0.4, 1.1, 0.6],
                           [0.4, 0.9, -0.4, 0.1, 0.7, 0.3, 1.2],
                           [0.1, -0.2, 0.5, 1.3, -0.2, 0.9, -0.1]])
    observed, scales = candidates[0], np.array([0.5, 1.2, 0.8, 1.4, 0.7, 0.9, 1.1])
    metric_reference = np.array([0.1, 0.3, 0.2, 0.4, -0.2, 0.1, 0.5])
    raw_state = np.array([0.8, 0.3, 1.1, -0.4, 0.9, 0.7, 0.2])
    raw_effect = np.array([0.2, 0.1, -0.3, 0.4, 0.0, 0.6, -0.2])
    input_change = np.array([0.1, -0.3, 0.2, 0.0, 0.5, -0.2, 0.4])

    for base_index, before in enumerate(base_contracts()):
        for flags in itertools.product((False, True), repeat=len(EDIT_FIELDS)):
            after = dict(before)
            for field, flag in zip(EDIT_FIELDS, flags):
                if not flag:
                    continue
                if field == "storage_encoding":
                    after[field] = "effect" if before[field] == "state" else "state"
                else:
                    after[field] = "T1" if field == "native_state_baseline_id" else "B1"
            actual_plan = module.plan_change(before, after)
            expected = expected_plan(before, after)
            for name, wanted in expected.items():
                require(actual_plan[name] == wanted, (base_index, dict(zip(EDIT_FIELDS, flags)), name, actual_plan[name], wanted))
                counts["planner_field_checks"] += 1
            # Verify role-effect arithmetic separately from cache invalidation.
            if after["task"] == "treated_state":
                wanted_effect_action = "unused"
            elif after["output_kind"] == "native_effect":
                wanted_effect_action = "preserve"
            elif flags[1] or (after["input_conditioned"] and flags[0]):
                wanted_effect_action = "recentre"
            else:
                wanted_effect_action = "preserve"
            require(actual_plan["prediction_effect_action"] == wanted_effect_action,
                    ("prediction_effect_action", actual_plan, wanted_effect_action))
            counts["planner_field_checks"] += 1
            task_is_effect = after["task"] == "reference_effect"
            output_is_state = after["output_kind"] == "state"
            active_cp = task_is_effect and output_is_state and flags[1]
            active_co = task_is_effect and flags[2]
            active_lift = not task_is_effect and not output_is_state and flags[5]
            before_storage_active = before["storage_encoding"] != ("state" if output_is_state else "effect")
            after_storage_active = after["storage_encoding"] != ("state" if output_is_state else "effect")
            expected_flags = {"task_type_changed": False, "model_input_changed": bool(after["input_conditioned"] and flags[0]),
                              "prediction_reference_changed": bool(active_cp), "observation_reference_changed": bool(active_co),
                              "native_state_baseline_changed": bool(active_lift), "prediction_definition_changed": bool(active_cp or active_lift),
                              "observed_realization_changed": bool(active_co), "storage_encoding_changed": flags[3],
                              "storage_baseline_changed": bool(flags[4] and (before_storage_active or after_storage_active))}
            for name, wanted_flag in expected_flags.items():
                require(actual_plan["change_flags"][name] == wanted_flag, ("change_flags", name, actual_plan["change_flags"], wanted_flag))
                counts["independent_change_flags"] += 1

            native = raw_state.copy() if after["output_kind"] == "state" else raw_effect.copy()
            if after["input_conditioned"] and after["model_input_id"] == "B1":
                native += input_change
            if after["output_kind"] == "state" and after["storage_encoding"] == "effect":
                expected_stored = native - references[after["storage_baseline_id"]]
            elif after["output_kind"] == "native_effect" and after["storage_encoding"] == "state":
                expected_stored = native + references[after["storage_baseline_id"]]
            else:
                expected_stored = native
            stored, payload = module.encode_prediction(native, after, references)
            close("encoded_values", stored, expected_stored[None, :])
            decoded = module.decode_prediction(stored, after, references, payload)
            close("decoded_native", decoded, native[None, :])
            actual = module.canonical_components(decoded, after, references, observed, scales,
                                                  candidates=candidates, metric_reference=metric_reference, correct_index=0)
            wanted = independent_components(native, after, references, observed, scales, candidates, metric_reference)
            for name in ("prediction", "residual", "metric_prediction", "scores"):
                wanted[name] = wanted[name][None, :]
            for name in wanted:
                close("canonical_" + name, actual[name], wanted[name])
            case_rows.append({"base_contract": base_index, "edited_fields": [name for name, flag in zip(EDIT_FIELDS, flags) if flag],
                              "expected_plan": expected, "status": "PASS"})

    require(len(case_rows) == 384, len(case_rows))
    for before in base_contracts():
        after = dict(before, task="reference_effect" if before["task"] == "treated_state" else "treated_state")
        actual = module.plan_change(before, after)
        require(actual["comparison_scope"] == "new_task" and actual["requires_task_declaration"] is True,
                ("task switch", actual))
        require(actual["prediction_action"] == "reuse" and actual["requires_score_recompute"] is True,
                ("task switch", actual))
        counts["task_switch_cases"] += 1
        for field, value in (("output_kind", "native_effect" if before["output_kind"] == "state" else "state"),
                             ("input_conditioned", not before["input_conditioned"])):
            try:
                module.plan_change(before, dict(before, **{field: value}))
            except ValueError:
                counts["contract_changes_rejected"] += 1
            else:
                raise AssertionError("Changed pipeline contract accepted: " + field)

    # Equal residuals do not make the two finite observed effects identical.
    before = next(c for c in base_contracts() if c["task"] == "reference_effect" and c["output_kind"] == "state" and not c["input_conditioned"])
    after = dict(before, prediction_reference_id="B1", observation_reference_id="B1")
    old = independent_components(raw_state, before, references, observed, scales, candidates, metric_reference)
    new = independent_components(raw_state, after, references, observed, scales, candidates, metric_reference)
    close("equal_scores_changed_target", old["scores"], new["scores"])
    require(not np.array_equal(old["prediction"], new["prediction"]) and not np.array_equal(old["target"], new["target"]),
            "Independent target-identity control accidentally unchanged")
    declaration = module.plan_change(before, after)
    require(declaration["observed_target_action"] == "rebuild" and declaration["comparison_scope"] == "new_observed_realization",
            "Equal scores concealed changed observed target")
    counts["equal_score_distinct_target_control"] += 1

    invalid_candidates = candidates.copy()
    invalid_candidates[1, 0] = np.nan
    try:
        module.canonical_components(raw_state, before, references, observed, scales,
                                    candidates=invalid_candidates, metric_reference=metric_reference, correct_index=0)
    except ValueError:
        counts["nonfinite_candidates_rejected"] += 1
    else:
        raise AssertionError("An undefined candidate distance was silently scored")

    receipt = {"status": "PASS", "created_utc": datetime.now(timezone.utc).isoformat(),
               "protocol_sha256": args.protocol_sha256, "plan_sha256": PLAN_SHA256,
               "api_refinement_sha256": sha256(HERE / "qa/ACTION_API_REFINEMENT.json"),
               "factorial_cases": len(case_rows), "comparison_counts": dict(counts), "maximum_absolute_errors": dict(maxima),
               "checker_sha256": sha256(__file__), "tested_module_sha256": sha256(args.module),
               "elapsed_seconds": time.monotonic() - started,
               "scope": "Independent software contract and numerical checks; synthetic values do not establish empirical error prevalence or method superiority."}
    (HERE / "qa/INDEPENDENT_ACTION_CASES.json").write_text(json.dumps(case_rows, indent=2) + "\n")
    (HERE / "qa/INDEPENDENT_ACTION_CHECK.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--module", type=Path, required=True)
    run(parser.parse_args())
