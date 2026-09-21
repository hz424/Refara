"""Independent raw-array oracle and complete interface checks for role metrics.

This checker imports no production study module. Scalar predictions and metrics
are reconstructed from the original bound arrays. Full-table checks connect the
new 27-role cube to the earlier, separately computed coupling and Systema panels.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "2"

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np


HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
PREVIOUS = ROOT / "submission/work/reference_coupling_validation_v1"
SAMPLE_SHA256 = "4f5ec1881c4758662838696f8e240f200411f9aa625276114891fa35a049ed0a"
INPUT_INVENTORY_SHA256 = "2eb15dfb0a474ad52db41f96eb761ddbb48d4978525314546695f440c0672a08"
MODELS = ("selected_CPA", "original_CPA_fixed", "additive", "compositional_ridge", "NC_native_effect", "native_effect_ridge")
METRICS = ("standardized_MSE", "raw_RMSE", "perturbation_centred_Pearson", "centroid_accuracy")
DEPTHS = (8, 16, 32, 64, 128, 135)
ROLES = tuple(itertools.product(range(3), repeat=3))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def pattern(role):
    co, cp, cm = role
    if co == cp == cm:
        return "S"
    if co == cp:
        return "M"
    if co == cm:
        return "P"
    if cp == cm:
        return "O"
    return "D"


def scalar_metrics(prediction, task, targets, scales, reference):
    """One prediction at a time; no production cdist or metric implementation."""
    residual = np.asarray(prediction, dtype=np.float64) - targets[task]
    mse = float(np.dot(residual / scales, residual / scales) / len(scales))
    rmse = float(np.sqrt(np.dot(residual, residual) / len(scales)))
    predicted_delta = prediction - reference
    observed_delta = targets[task] - reference
    if np.std(predicted_delta) == 0 or np.std(observed_delta) == 0:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(predicted_delta, observed_delta)[0, 1])
    correct_distance = float(np.linalg.norm(residual))
    farther = sum(float(np.linalg.norm(prediction - candidate)) > correct_distance
                  for index, candidate in enumerate(targets) if index != task)
    centroid = farther / (len(targets) - 1)
    return np.array([mse, rmse, correlation, centroid])


def authorize(protocol, protocol_sha256):
    require(protocol.is_file() and sha256(protocol) == protocol_sha256,
            "The new protocol must exist and match its supplied SHA256 before target loading")
    sample_path = HERE / "qa/EMPIRICAL_SAMPLING.json"
    require(sha256(sample_path) == SAMPLE_SHA256, "Frozen QA sample changed")
    inventory_path = PREVIOUS / "INPUT_INVENTORY.json"
    require(sha256(inventory_path) == INPUT_INVENTORY_SHA256, "Bound source inventory changed")
    inventory = json.loads(inventory_path.read_text())
    source_bindings = {}

    def source(suffix):
        entries = [entry for entry in inventory["files"] if entry["path"].endswith(suffix)]
        require(len(entries) == 1, (suffix, len(entries)))
        entry = entries[0]
        path = ROOT / entry["path"]
        require(path.stat().st_size == entry["bytes"] and sha256(path) == entry["sha256"],
                "Changed source " + str(path))
        source_bindings[entry["path"]] = entry["sha256"]
        return path

    paths = {
        "selected": source("systema_matched_norman_v1/results/selected_CPA_block_profiles.npy"),
        "controls": source("norman_training_review/source_data/reference_results/control_block_means.npy"),
        "fixed": source("systema_matched_norman_v1/results/fixed_profiles_and_references.npz"),
    }
    return paths, {"protocol_sha256": protocol_sha256, "sample_sha256": SAMPLE_SHA256,
                   "input_inventory_sha256": INPUT_INVENTORY_SHA256, "source_bindings": source_bindings}


def run(args):
    started = time.monotonic()
    paths, binding = authorize(args.protocol, args.protocol_sha256)
    counts, maxima = Counter(), defaultdict(float)
    atol, rtol = 1e-10, 1e-9

    def close(name, actual, expected, *, exact=False):
        actual, expected = np.asarray(actual), np.asarray(expected)
        require(actual.shape == expected.shape, (name, actual.shape, expected.shape))
        require(np.array_equal(np.isnan(actual), np.isnan(expected)), name + " undefined support differs")
        finite = np.isfinite(actual) & np.isfinite(expected)
        require(bool(np.all(finite | (np.isnan(actual) & np.isnan(expected)))), name + " contains infinity")
        errors = np.abs(actual[finite] - expected[finite])
        maximum = float(errors.max()) if errors.size else 0.0
        tolerance = 0 if exact else atol + rtol * np.maximum(np.abs(actual[finite]), np.abs(expected[finite]))
        require(bool(np.all(errors <= tolerance)), (name, maximum))
        maxima[name] = max(maxima[name], maximum)
        counts[name] += actual.size

    with np.load(args.cube, allow_pickle=False) as archive:
        tasks = tuple(archive[args.tasks_key].tolist())
        models = tuple(archive[args.models_key].tolist())
        depths = tuple(int(x) for x in archive[args.depths_key])
        allocations = tuple(int(x) for x in archive[args.allocations_key])
        metrics = tuple(archive[args.metrics_key].tolist())
        roles = tuple(tuple(int(x) for x in row) for row in archive[args.roles_key])
        role_scores = archive[args.role_scores_key]
        direct_scores = archive[args.direct_scores_key]
    require(models == MODELS and depths == DEPTHS and allocations == tuple(range(30)), "Cube axes changed")
    require(metrics == METRICS and set(roles) == set(ROLES) and len(roles) == 27, "Metric/role axes changed")
    require(role_scores.shape == (55, 6, 30, 27, 6, 4), role_scores.shape)
    require(direct_scores.shape == (55, 6, 30, 3, 6, 4), direct_scores.shape)
    require(bool(np.all(np.isfinite(role_scores[..., [0, 1, 3]]))), "Undefined non-Pearson metric")
    role_index = {role: index for index, role in enumerate(roles)}
    sample = json.loads((HERE / "qa/EMPIRICAL_SAMPLING.json").read_text())

    # Source targets are first decoded in this checker only after authorize().
    selected = np.load(paths["selected"], mmap_mode="r", allow_pickle=False)
    controls = np.load(paths["controls"], mmap_mode="r", allow_pickle=False)
    with np.load(paths["fixed"], allow_pickle=False) as archive:
        targets = np.asarray(archive["observed_states"], dtype=float)
        profiles = np.asarray(archive["profiles"], dtype=float)
        scales = np.asarray(archive["scales"], dtype=float)
        training_control = np.asarray(archive["TRAIN_control"], dtype=float)
        training_perturbation = np.asarray(archive["TRAIN_perturbation"], dtype=float)
        training_lift = np.asarray(archive["native_effect_training_baseline"], dtype=float)
        require(tuple(archive["conditions"].tolist()) == tasks, "Raw target task axis changed")
    require(targets.shape == (55, 2000) and profiles.shape == (6, 55, 2000), "Raw shapes changed")
    expected_rows = []
    for ti in sample["task_indices_zero_based"]:
        for di, depth in enumerate(depths):
            for ai in sample["allocation_indices_zero_based"]:
                blocks = np.asarray(controls[ai, di], dtype=float)
                for mi, model in enumerate(models):
                    if mi == 0:
                        states = np.asarray(selected[ti, ai, di], dtype=float)
                        effect = None
                    elif mi < 4:
                        states = np.repeat(profiles[mi, ti][None, :], 3, axis=0)
                        effect = None
                    else:
                        effect = np.zeros_like(scales) if mi == 4 else profiles[5, ti] - training_lift
                        states = np.repeat((effect + training_control)[None, :], 3, axis=0)
                    for ri, (co, cp, cm) in enumerate(roles):
                        prediction = states[cm] - blocks[cp] + blocks[co] if effect is None else effect + blocks[co]
                        expected = scalar_metrics(prediction, ti, targets, scales, training_perturbation)
                        close("raw_role_metrics", role_scores[ti, di, ai, ri, mi], expected)
                        close("raw_role_centroid_exact", role_scores[ti, di, ai, ri, mi, 3:4], expected[3:4], exact=True)
                        expected_rows.append([ti, tasks[ti], depth, ai, model, "roles", co, cp, cm, *map(float, expected)])
                    for cm in range(3):
                        expected = scalar_metrics(states[cm], ti, targets, scales, training_perturbation)
                        close("raw_direct_metrics", direct_scores[ti, di, ai, cm, mi], expected)
                        close("raw_direct_centroid_exact", direct_scores[ti, di, ai, cm, mi, 3:4], expected[3:4], exact=True)
                        expected_rows.append([ti, tasks[ti], depth, ai, model, "direct_state", "unused", "unused", cm, *map(float, expected)])
        print(json.dumps({"raw_task_checked": tasks[ti], "score_rows": len(expected_rows), "elapsed_seconds": round(time.monotonic() - started, 2)}), flush=True)
    require(len(expected_rows) == sample["scalar_score_rows_expected"], "Raw sample row count changed")

    # Complete existing-panel agreement is independent of the raw sample.
    with np.load(PREVIOUS / "checks/systema/task_scores.npz", allow_pickle=False) as old:
        for axis, wanted in (("tasks", tasks), ("models", models), ("depths", depths), ("allocations", allocations), ("metrics", metrics)):
            require(tuple(old[axis].tolist()) == wanted, "Earlier panel axis differs: " + axis)
        close("full_prior_direct_state_panel", direct_scores, old["scores_direct_state"])
        old_pairs = old["O_aligned_effect_realizations"]
        aligned = np.stack([role_scores[:, :, :, role_index[(int(co), int(cm), int(cm))]] for cm, co in old_pairs], axis=3)
        close("full_prior_O_panel", aligned, old["scores_O_aligned_effect"])

    balanced = {label: role_scores[:, :, :, [i for i, role in enumerate(roles) if pattern(role) == label]].mean(axis=3)
                for label in ("S", "M", "P", "O", "D")}
    task_index = {task: index for index, task in enumerate(tasks)}
    depth_index = {depth: index for index, depth in enumerate(depths)}
    model_index = {model: index for index, model in enumerate(models)}
    pattern_rows = 0
    with (PREVIOUS / "checks/pattern_scores.tsv").open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["resource"] != "Norman":
                continue
            ti, di, ai, mi = task_index[row["task"]], depth_index[int(row["depth"])], int(row["allocation"]), model_index[row["model"]]
            expected = float(row["mse"])
            close("full_prior_balanced_MSE", balanced[row["pattern"]][ti, di, ai, mi, 0], expected)
            pattern_rows += 1
    require(pattern_rows == 297000, pattern_rows)

    # All block permutations should preserve balanced pattern means. Each check
    # remaps labels, not predictions or observations to a different target.
    for permutation in itertools.permutations(range(3)):
        for label in balanced:
            remapped = [role_index[tuple(permutation[block] for block in role)] for role in roles if pattern(role) == label]
            close("all_label_permutation_balanced_means", role_scores[:, :, :, remapped].mean(axis=3), balanced[label])
    for co, cp, cm in roles:
        index = role_index[(co, cp, cm)]
        close("all_native_unused_roles", role_scores[:, :, :, index, 4:], role_scores[:, :, :, role_index[(co, 0, 0)], 4:], exact=True)
        close("all_fixed_state_unused_Cmodel", role_scores[:, :, :, index, 1:4], role_scores[:, :, :, role_index[(co, cp, 0)], 1:4], exact=True)
        if co == cp:
            close("all_common_centre_state_matches_direct", role_scores[:, :, :, index, :4], direct_scores[:, :, :, cm, :4])

    # Small metric controls verify interpretation, not one copy of the engine.
    synthetic_targets = np.array([[0., 1., 3.], [0., 1., 3.], [4., 1., 0.]])
    tied = scalar_metrics(synthetic_targets[0], 0, synthetic_targets, np.ones(3), np.zeros(3))
    close("scalar_tie_control", tied, np.array([0., 0., 1., 0.5]))
    constant = scalar_metrics(np.ones(3), 0, synthetic_targets, np.ones(3), np.zeros(3))
    require(np.isnan(constant[2]), "Constant-vector Pearson should be undefined")

    expected_path = HERE / "qa/EXPECTED_METRIC_SAMPLE.tsv"
    with expected_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["task_index", "task", "depth", "allocation", "model", "view", "Cobs", "Cpred", "Cmodel", *metrics])
        writer.writerows(expected_rows)
    receipt = {"status": "PASS", "created_utc": datetime.now(timezone.utc).isoformat(), **binding,
               "production_code_imported": False, "raw_score_rows": len(expected_rows),
               "full_prior_pattern_rows": pattern_rows, "comparison_counts": dict(counts),
               "maximum_absolute_errors": dict(maxima), "elapsed_seconds": time.monotonic() - started,
               "checker_sha256": sha256(__file__), "cube_sha256": sha256(args.cube),
               "expected_sample_sha256": sha256(expected_path),
               "previous_systema_cube_sha256": sha256(PREVIOUS / "checks/systema/task_scores.npz"),
               "previous_pattern_scores_sha256": sha256(PREVIOUS / "checks/pattern_scores.tsv"),
               "checks_do_not_establish": "Scientific novelty, prospective validation, or superiority over a correctly paired exhaustive Systema evaluator."}
    (HERE / "qa/INDEPENDENT_METRIC_CHECK.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "PASS", "raw_score_rows": len(expected_rows), "maximum_absolute_error": max(maxima.values()), "elapsed_seconds": receipt["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--cube", type=Path, required=True)
    parser.add_argument("--tasks-key", default="tasks")
    parser.add_argument("--models-key", default="models")
    parser.add_argument("--depths-key", default="depths")
    parser.add_argument("--allocations-key", default="allocations")
    parser.add_argument("--metrics-key", default="metrics")
    parser.add_argument("--roles-key", default="role_tuples")
    parser.add_argument("--role-scores-key", default="scores_roles")
    parser.add_argument("--direct-scores-key", default="scores_direct_state")
    run(parser.parse_args())
