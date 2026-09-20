"""Check the complete reader demonstration against raw sources and direct algebra.

No production action-planner, input-adapter or metric implementation is imported.
Every saved before/after component is reconstructed, including each CellFlow fit.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "2"

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import time

import numpy as np

from check_role_metrics import INPUT_INVENTORY_SHA256, MODELS, PREVIOUS, ROOT, scalar_metrics, sha256
from check_role_actions import expected_plan


HERE = Path(__file__).resolve().parents[1]
PBMC_ADDENDUM_SHA256 = "bf1c1bc927321ea6abb62ccf3d0ca103d8e5fceb5e9236af0537f424f2455fef"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def raw_demo_arrays():
    inventory_path = PREVIOUS / "INPUT_INVENTORY.json"
    require(sha256(inventory_path) == INPUT_INVENTORY_SHA256, "Source inventory changed")
    inventory = json.loads(inventory_path.read_text())
    bindings = {}

    def source(suffix):
        matches = [row for row in inventory["files"] if row["path"].endswith(suffix)]
        require(len(matches) == 1, suffix)
        row = matches[0]
        path = ROOT / row["path"]
        require(sha256(path) == row["sha256"], "Changed raw input " + str(path))
        bindings[row["path"]] = row["sha256"]
        return path

    out = {}
    c = np.load(source("norman_training_review/source_data/reference_results/control_block_means.npy"), mmap_mode="r")
    selected = np.load(source("systema_matched_norman_v1/results/selected_CPA_block_profiles.npy"), mmap_mode="r")
    with np.load(source("systema_matched_norman_v1/results/fixed_profiles_and_references.npz"), allow_pickle=False) as raw:
        out["norman_controls"] = np.asarray(c[0, 0], dtype=float)
        out["norman_scales"] = np.asarray(raw["scales"], dtype=float)
        out["norman_observed"] = np.asarray(raw["observed_states"][0], dtype=float)
        out["norman_candidates"] = np.asarray(raw["observed_states"], dtype=float)
        for name in ("TRAIN_control", "TRAIN_perturbation", "native_effect_training_baseline"):
            out["norman_" + name] = np.asarray(raw[name], dtype=float)
        profiles = np.asarray(raw["profiles"], dtype=float)
        out["norman_model_0"] = np.asarray(selected[0, 0, 0][None, :, :], dtype=float)
        for mi in (1, 2, 3):
            out[f"norman_model_{mi}"] = np.broadcast_to(profiles[mi, 0], (1, 3, 2000))
        out["norman_model_4"] = np.zeros((1, 2000))
        out["norman_model_5"] = (profiles[5, 0] - out["norman_native_effect_training_baseline"])[None, :]
    with np.load(source("Kang_Reference_Mean_Arrays_for_Review/mean_arrays.npz"), allow_pickle=False) as raw:
        out["kang_controls"] = np.asarray(raw["control_means"][0, 0], dtype=float)
        out["kang_scales"] = np.asarray(raw["gene_scales"][0], dtype=float)
        out["kang_observed"] = np.asarray(raw["treated_means"][0, 0], dtype=float)
        out["kang_model_0"] = np.asarray(raw["ridge_state_terminal"][0, 0][None, :, :], dtype=float)
        out["kang_model_1"] = np.asarray(raw["cellflow_terminal"][0, 0], dtype=float)
    axes = json.loads(source("biological_consequence_design/inputs/axes.json").read_text())
    with np.load(source("biological_consequence_design/inputs/benchmark_inputs.npz"), allow_pickle=False) as raw:
        out["pbmc_controls"] = np.asarray(raw["control_means"][0, :, 0], dtype=float)
        out["pbmc_scales"] = np.asarray(raw["scales"], dtype=float)
        out["pbmc_observed"] = np.asarray(raw["treated_means"][0], dtype=float)
        native, states = raw["direct_effects"], raw["state_predictions"]
        for mi in range(5):
            out[f"pbmc_model_{mi}"] = np.asarray(native[0, mi][None, :], dtype=float)
        for mi in range(5, 8):
            out[f"pbmc_model_{mi}"] = np.asarray(states[0, :, 0, mi - 5][None, :, :], dtype=float)
    return out, bindings, axes


def component_oracle(native, contract, refs, observed, scales, candidates=None, centre=None):
    if contract["task"] == "reference_effect":
        pred = native - refs[contract["prediction_reference_id"]] if contract["output_kind"] == "state" else native
        target = observed - refs[contract["observation_reference_id"]]
        rendered = pred + refs[contract["observation_reference_id"]]
    else:
        pred = native if contract["output_kind"] == "state" else native + refs[contract["native_state_baseline_id"]]
        target, rendered = observed, pred
    residual = pred - target
    scores = np.full((len(native), 4), np.nan)
    for fit, vector in enumerate(rendered):
        if candidates is not None:
            scores[fit] = scalar_metrics(vector, 0, candidates, scales, centre)
        else:
            error = residual[fit]
            scores[fit, 0] = np.dot(error / scales, error / scales) / len(scales)
            scores[fit, 1] = np.sqrt(np.dot(error, error) / len(scales))
    return {"native_prediction": native, "prediction": pred, "target": target, "residual": residual,
            "metric_prediction": rendered, "metric_target": observed, "scores": scores}


def run(args):
    started = time.monotonic()
    require(sha256(args.protocol) == args.protocol_sha256, "Protocol changed")
    require(sha256(HERE / "PBMC_COVERAGE_ADDENDUM.json") == PBMC_ADDENDUM_SHA256, "PBMC addendum changed")
    counts, maxima = Counter(), defaultdict(float)

    def close(name, actual, wanted, exact=False):
        actual, wanted = np.asarray(actual, dtype=float), np.asarray(wanted, dtype=float)
        require(actual.shape == wanted.shape, (name, actual.shape, wanted.shape))
        require(np.array_equal(np.isnan(actual), np.isnan(wanted)), (name, "undefined support"))
        use = np.isfinite(actual) & np.isfinite(wanted)
        require(bool(np.all(use | (np.isnan(actual) & np.isnan(wanted)))), (name, "infinite value"))
        errors = np.abs(actual[use] - wanted[use])
        maximum = float(errors.max()) if len(errors) else 0.
        require(bool(np.all(errors <= (0 if exact else 1e-10 + 1e-9 * np.abs(wanted[use])))), (name, maximum))
        maxima[name] = max(maxima[name], maximum)
        counts[name] += actual.size

    manifest = json.loads((args.data / "manifest.json").read_text())
    require(manifest["protocol_sha256"] == args.protocol_sha256 and manifest["pbmc_coverage_addendum_sha256"] == PBMC_ADDENDUM_SHA256, "Demo binding differs")
    require(sha256(args.data / manifest["data_file"]) == manifest["data_sha256"], "Demo data digest differs")
    with np.load(args.data / manifest["data_file"], allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    raw, source_bindings, pbmc_axes = raw_demo_arrays()
    require(set(raw) == set(arrays), "Reader demo contains unexpected/missing arrays")
    for name in raw:
        close("raw_demo_" + name, arrays[name], raw[name], exact=True)
    require([model["name"] for model in manifest["resources"]["Norman"]["models"]] == list(MODELS), "Norman model identity changed")
    require([model["name"] for model in manifest["resources"]["PBMC"]["models"]] == pbmc_axes["model_ids"], "PBMC model identity changed")
    require(manifest["resources"]["Kang"]["models"][1]["fits"] == 5, "CellFlow fits missing")

    cases = json.loads((args.results / "cases.json").read_text())
    actions = json.loads((args.results / "actions.json").read_text())
    action_map = {row["case_id"]: row for row in actions}
    require(len(cases) == len(actions) == len(action_map) == 95, "Case count or identities changed")
    summaries = {}
    with (args.results / "action_summary.tsv").open(newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            require(row["case_id"] not in summaries, "Duplicate summary")
            summaries[row["case_id"]] = row
    with np.load(args.results / "components.npz", allow_pickle=False) as archive:
        stored_components = {name: archive[name] for name in archive.files}
    expected_case_ids = set()
    for resource, meta in manifest["resources"].items():
        for mi, model in enumerate(meta["models"]):
            operations = ["unused_roles_Kang"] if resource == "Kang" else ["input_replacement", "paired_input_effect", "prediction_baseline", "observation_baseline", "common_scoring_reference", "same_target_storage_encoding"]
            if resource == "Norman" or (resource == "PBMC" and model["kind"] == "state"):
                operations.append("task_switch")
            expected_case_ids.update(f"{resource}_{mi}_{op}" for op in operations)
    require({case["case_id"] for case in cases} == expected_case_ids, "Metadata-fixed case selection changed")
    require(set(action_map) == set(summaries) == expected_case_ids, "Output case support changed")
    expected_component_keys = set()
    resource_counts = Counter()
    for case in cases:
        resource, cid = case["resource"], case["case_id"]
        meta = manifest["resources"][resource]
        prefix, model = meta["prefix"], meta["models"][case["model_index"]]
        expected_conditioned = (case["model_index"] == 0 if resource == "Norman" else True if resource == "Kang" else case["model_index"] >= 5)
        expected_kind = "state" if (case["model_index"] < 4 if resource == "Norman" else True if resource == "Kang" else case["model_index"] >= 5) else "native_effect"
        require(model["input_conditioned"] == expected_conditioned and model["kind"] == expected_kind, ("Pipeline contract", cid))
        refs = {resource + f".B{i+1}": arrays[prefix + "_controls"][i] for i in range(3)}
        if resource == "Norman":
            refs["Norman.TRAIN_control"] = arrays["norman_TRAIN_control"]
            refs["Norman.native_effect_training_baseline"] = arrays["norman_native_effect_training_baseline"]
        wanted_phases = {}
        for phase in ("before", "after"):
            contract = case[phase]
            emitted = arrays[model["array_key"]]
            if expected_kind == "state":
                block = int(contract["model_input_id"].split(".B")[1]) - 1 if expected_conditioned else 0
                native = emitted[:, block]
            else:
                native = emitted
            wanted = component_oracle(native, contract, refs, arrays[prefix + "_observed"], arrays[prefix + "_scales"],
                                      arrays.get(prefix + "_candidates"), arrays.get(prefix + "_TRAIN_perturbation"))
            encoded = native
            if expected_kind == "state" and contract["storage_encoding"] == "effect":
                encoded = native - refs[contract["storage_baseline_id"]]
            elif expected_kind == "native_effect" and contract["storage_encoding"] == "state":
                encoded = native + refs[contract["storage_baseline_id"]]
            wanted["stored"] = encoded
            for name, value in wanted.items():
                key = f"{cid}__{phase}__{name}"
                expected_component_keys.add(key)
                close("real_action_" + name, stored_components[key], value)
            for ki, metric in enumerate(("standardized_MSE", "raw_RMSE", "perturbation_centred_Pearson", "centroid_accuracy")):
                expected_mean = wanted["scores"][:, ki].mean()
                actual_mean = action_map[cid][phase + "_scores"][metric]
                if np.isnan(expected_mean):
                    require(actual_mean is None, ("Unsupported metric invented", cid, metric))
                else:
                    close("real_action_fit_mean_scores", actual_mean, expected_mean)
            wanted_phases[phase] = wanted
        plan = expected_plan(case["before"], case["after"])
        for name, wanted in plan.items():
            require(action_map[cid]["plan"][name] == wanted, (cid, name, action_map[cid]["plan"][name], wanted))
            counts["real_action_plan_fields"] += 1
        for name in ("native_prediction", "prediction", "target", "residual", "metric_prediction", "scores"):
            before, after = wanted_phases["before"][name], wanted_phases["after"][name]
            use = np.isfinite(before) & np.isfinite(after)
            shift = float(np.max(np.abs(before[use] - after[use])))
            close("real_action_reported_shift", action_map[cid]["before_after_max_abs_changes"][name], shift)
            close("real_action_summary_shift", summaries[cid]["change_" + name], shift)
        old, new = case["before"], case["after"]
        same_target_declaration = old["task"] == new["task"] and (new["task"] == "treated_state" or old["observation_reference_id"] == new["observation_reference_id"])
        require(action_map[cid]["target_declaration_unchanged"] == same_target_declaration, ("Target identity", cid))
        resource_counts[resource] += 1
    require(expected_component_keys == set(stored_components), "Unexpected/missing saved components")

    omission_rows = json.loads((args.results / "role_omission_checks.json").read_text())
    case_map = {case["case_id"]: case for case in cases}
    seen_omissions = set()
    for row in omission_rows:
        key = row["case_id"], row["field"]
        require(key not in seen_omissions, "Duplicate omission check")
        seen_omissions.add(key)
        contract = case_map[row["case_id"]]["after"]
        active = {"model_input_id": contract["input_conditioned"], "prediction_reference_id": contract["task"] == "reference_effect" and contract["output_kind"] == "state",
                  "observation_reference_id": contract["task"] == "reference_effect"}[row["field"]]
        wanted = "missing_active_information" if active else "valid_unused_role_omission"
        require(row["result"] == wanted, ("Omission classification", row))
        counts["role_omission_" + wanted] += 1
    require(len(seen_omissions) == 285, "Incomplete omission coverage")
    receipt = {"status": "PASS", "protocol_sha256": args.protocol_sha256, "pbmc_addendum_sha256": PBMC_ADDENDUM_SHA256,
               "production_code_imported": False, "raw_demo_arrays": len(raw), "valid_cases": len(cases),
               "cases_by_resource": dict(resource_counts), "saved_components_checked": len(expected_component_keys),
               "comparison_counts": dict(counts), "maximum_absolute_errors": dict(maxima),
               "source_bindings": source_bindings, "demo_manifest_sha256": sha256(args.data / "manifest.json"),
               "data_sha256": manifest["data_sha256"], "actions_sha256": sha256(args.results / "actions.json"),
               "case_sha256": sha256(args.results / "cases.json"), "components_sha256": sha256(args.results / "components.npz"),
               "checker_sha256": sha256(__file__), "independent_contract_oracle_sha256": sha256(Path(__file__).with_name("check_role_actions.py")),
               "elapsed_seconds": time.monotonic() - started,
               "scope": "All bundled operations checked against their original cached inputs and independent formulas. These are retrospective action replays, not observed external organizer errors or human usability measurements."}
    (HERE / "qa/INDEPENDENT_REAL_ACTION_CHECK.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in receipt.items() if key not in ("source_bindings", "comparison_counts", "maximum_absolute_errors")}), flush=True)
    print(json.dumps({"maximum_absolute_error": max(maxima.values()), "omission_counts": {key: value for key, value in counts.items() if key.startswith("role_omission_")}}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    run(parser.parse_args())
