#!/usr/bin/env python3
"""Fit the prospectively fixed Norman native effect ridge using training means.

Only three expression/feature inputs are accepted: training_means.npz,
gene_panel.tsv, and scales.npy. Test task identifiers are opened after the
hyperparameter-selection receipt is written. No test expression or evaluation
control file is opened. Run with OPENBLAS_NUM_THREADS=1 for a compact CPU run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import zipfile

import numpy as np
import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_tsv(path, records):
    pd.DataFrame(records).to_csv(path, sep="\t", index=False, float_format="%.17g")


def deterministic_npz(path, **arrays):
    """Store ordinary NumPy arrays without wall-clock metadata or pickle."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, array in arrays.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(array), allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, buffer.getvalue(), compresslevel=6)


def components(condition):
    if not isinstance(condition, str) or not condition or condition.strip() != condition:
        raise ValueError("Condition identifiers must be nonempty, unpadded strings")
    if condition == "ctrl":
        return ()
    parts = condition.split("+")
    if len(parts) not in (1, 2) or any(not p or p == "ctrl" for p in parts):
        raise ValueError(f"Unsupported condition {condition!r}")
    if len(set(parts)) != len(parts):
        raise ValueError(f"Repeated constituent in {condition!r}")
    return tuple(parts)


def make_design(conditions, features):
    if len(features) != len(set(features)):
        raise ValueError("Duplicate feature identifiers")
    axis = {name: index for index, name in enumerate(features)}
    result = np.zeros((len(conditions), len(features)), dtype=np.float64)
    for row, condition in enumerate(conditions):
        for part in components(str(condition)):
            if part not in axis:
                raise ValueError(f"Untrained constituent {part!r}")
            result[row, axis[part]] = 1.0
    return result


def double_folds(conditions, namespace, n_folds):
    if n_folds < 2:
        raise ValueError("At least two folds are required")
    doubles = [str(c) for c in conditions if len(components(str(c))) == 2]
    if len(set(doubles)) != len(doubles) or len(doubles) < n_folds:
        raise ValueError("Validation doubles must be unique and cover every fold")
    digest = {c: hashlib.sha256((namespace + "\0" + c).encode("utf-8")).hexdigest() for c in doubles}
    ordered = sorted(doubles, key=lambda c: (digest[c], c))
    return {c: (rank % n_folds, rank, digest[c]) for rank, c in enumerate(ordered)}


def ridge_coefficients(design, target, alpha):
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or not x.shape[0]:
        raise ValueError("Design and target must be nonempty aligned matrices")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(alpha) or alpha < 0:
        raise ValueError("Ridge inputs must be finite with nonnegative alpha")
    gram = x.T @ x
    gram.flat[:: gram.shape[0] + 1] += float(alpha)
    return np.linalg.solve(gram, x.T @ y)


def fold_indices(conditions, assignments, fold):
    held_out = np.array([i for i, c in enumerate(conditions) if c in assignments and assignments[c][0] == fold], dtype=int)
    if not len(held_out):
        raise ValueError("Empty validation fold")
    in_training = np.ones(len(conditions), dtype=bool)
    in_training[held_out] = False
    return np.flatnonzero(in_training), held_out


def select_alpha(alphas, errors, tolerance):
    alphas, errors = np.asarray(alphas, dtype=float), np.asarray(errors, dtype=float)
    if alphas.ndim != 1 or alphas.shape != errors.shape or not len(alphas):
        raise ValueError("Aligned nonempty alpha and error vectors are required")
    if not np.isfinite(errors).all() or not np.isfinite(alphas).all() or tolerance < 0:
        raise ValueError("Nonfinite selection inputs or negative tolerance")
    minimum = float(errors.min())
    eligible = alphas[np.abs(errors - minimum) <= tolerance]
    return float(eligible.max()), minimum


def cross_validate(design, target, conditions, alphas, namespace, n_folds):
    assignments = double_folds(conditions, namespace, n_folds)
    ordered_doubles = sorted(assignments)
    oof_axis = {condition: row for row, condition in enumerate(ordered_doubles)}
    predictions = np.full((len(alphas), len(ordered_doubles), target.shape[1]), np.nan)
    condition_records, train_records, fold_records, alpha_records = [], [], [], []
    for alpha_index, alpha in enumerate(alphas):
        for fold in range(n_folds):
            train, held = fold_indices(conditions, assignments, fold)
            coefficients = ridge_coefficients(design[train], target[train], alpha)
            held_predictions = design[held] @ coefficients
            held_errors = np.mean((held_predictions - target[held]) ** 2, axis=1)
            train_errors = np.mean((design[train] @ coefficients - target[train]) ** 2, axis=1)
            for row, error, prediction in zip(held, held_errors, held_predictions):
                condition = conditions[row]
                condition_records.append({"alpha": alpha, "fold": fold, "condition": condition, "standardized_mse": error})
                predictions[alpha_index, oof_axis[condition]] = prediction
            for row, error in zip(train, train_errors):
                train_records.append({"alpha": alpha, "fold": fold, "condition": conditions[row], "condition_order": len(components(conditions[row])), "standardized_mse": error})
            fold_records.append({"alpha": alpha, "fold": fold, "n_train_conditions": len(train), "n_validation_conditions": len(held), "training_mse": float(train_errors.mean()), "validation_mse": float(held_errors.mean())})
        selected = [record["standardized_mse"] for record in condition_records if record["alpha"] == alpha]
        if len(selected) != len(ordered_doubles):
            raise AssertionError("Each training double must have one out-of-fold prediction")
        alpha_records.append({"alpha": alpha, "validation_mse": float(np.mean(selected)), "n_validation_conditions": len(selected)})
    if not np.isfinite(predictions).all():
        raise AssertionError("Incomplete out-of-fold predictions")
    return assignments, ordered_doubles, predictions, condition_records, train_records, fold_records, alpha_records


def fit(prepared_dir, protocol_path, output_dir):
    prepared_dir, protocol_path, output_dir = map(Path, (prepared_dir, protocol_path, output_dir))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory must be empty; a frozen fit is never overwritten")
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_bytes = protocol_path.read_bytes()
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    sidecar = protocol_path.with_name(protocol_path.name + ".sha256").read_text().split()[0]
    if sidecar != protocol_hash:
        raise ValueError("Protocol hash sidecar does not match")
    protocol = json.loads(protocol_bytes)
    model, validation = protocol["model"], protocol["validation"]
    if model["id"] != "direct_effect_ridge" or model["native_output"] != "effect":
        raise ValueError("This implementation requires the frozen direct-effect ridge protocol")
    if model["alpha_grid"] != [0, 0.01, 0.1, 1, 10, 100] or validation["fold_count"] != 5 or validation["tie_tolerance"] != 1e-12:
        raise ValueError("Unexpected hyperparameter grid or validation rule")
    input_hashes, access_events = {}, []

    def checked_input(name, stage):
        path = prepared_dir / name
        actual = sha256(path)
        if actual != protocol["frozen_input_sha256"][name]:
            raise ValueError(f"Frozen input hash mismatch: {name}")
        input_hashes[name] = actual
        access_events.append({"file": name, "stage": stage, "sha256": actual})
        return path

    training_path = checked_input("training_means.npz", "training_and_validation")
    with np.load(training_path, allow_pickle=False) as arrays:
        required = {"conditions", "gemgroups", "mean_by_gemgroup", "mean_equal_gemgroup", "n_cells_by_gemgroup"}
        if set(arrays.files) != required:
            raise ValueError("Unexpected training mean archive fields")
        conditions = [str(c) for c in arrays["conditions"].tolist()]
        means = arrays["mean_equal_gemgroup"].copy()
        gemgroups = arrays["gemgroups"].copy()
        group_means = arrays["mean_by_gemgroup"]
        counts = arrays["n_cells_by_gemgroup"]
        if means.shape != (161, 2000) or len(conditions) != 161 or len(set(conditions)) != 161:
            raise ValueError("The frozen training design requires 161 unique conditions and 2,000 genes")
        if gemgroups.shape != (8,) or len(set(gemgroups.tolist())) != 8 or group_means.shape != (161, 8, 2000) or counts.shape != (161, 8):
            raise ValueError("Training means must cover eight unique gemgroups")
        if not np.isfinite(means).all() or not np.isfinite(group_means).all() or not np.isfinite(counts).all() or not np.all(counts > 0):
            raise ValueError("Training expression must be finite with positive support")
        if not np.array_equal(means, group_means.mean(axis=1)):
            raise ValueError("Training means must use equal gemgroup weights")
    features = sorted(c for c in conditions if len(components(c)) == 1)
    if len(features) != 105 or sum(len(components(c)) == 2 for c in conditions) != 55 or conditions.count("ctrl") != 1:
        raise ValueError("Training must contain 105 singles, 55 doubles, and one ctrl")
    panel_path = checked_input("gene_panel.tsv", "training_and_validation")
    gene_panel = pd.read_csv(panel_path, sep="\t")
    scale_path = checked_input("scales.npy", "training_and_validation")
    scales = np.load(scale_path, allow_pickle=False)
    if scales.shape != (2000,) or not np.isfinite(scales).all() or not np.all(scales > 0):
        raise ValueError("Gene scales must be 2,000 finite positive values")
    needed = {"panel_index", "ensembl_id", "gene_symbol", "scale"}
    if not needed.issubset(gene_panel.columns) or len(gene_panel) != 2000:
        raise ValueError("Invalid frozen gene panel")
    if not np.array_equal(gene_panel["panel_index"].to_numpy(), np.arange(2000)) or gene_panel["ensembl_id"].duplicated().any():
        raise ValueError("Gene panel axis must be ordered and have unique Ensembl identifiers")
    if gene_panel[["gene_symbol", "ensembl_id"]].isna().any().any() or not np.allclose(gene_panel["scale"].to_numpy(), scales, rtol=2e-15, atol=0):
        raise ValueError("Frozen gene identifiers/scales are inconsistent")
    design = make_design(conditions, features)
    baseline = means[conditions.index("ctrl")].copy()
    target = (means - baseline) / scales
    if np.any(target[conditions.index("ctrl")]):
        raise AssertionError("Training control must have exactly zero native effect")

    alphas = model["alpha_grid"]
    assignments, doubles, oof_predictions, cv_rows, train_rows, fold_rows, alpha_rows = cross_validate(
        design, target, conditions, alphas, validation["hash_namespace"], validation["fold_count"]
    )
    alpha, minimum = select_alpha(alphas, [r["validation_mse"] for r in alpha_rows], validation["tie_tolerance"])
    for row in alpha_rows:
        row["selected"] = row["alpha"] == alpha
        row["within_tie_tolerance"] = abs(row["validation_mse"] - minimum) <= validation["tie_tolerance"]
    membership = [{"condition": c, "condition_order": len(components(c)), "validation_fold": assignments[c][0] if c in assignments else -1,
                   "hash_rank": assignments[c][1] if c in assignments else -1,
                   "assignment_sha256": assignments[c][2] if c in assignments else "",
                   "fold_role": "out_of_fold_double" if c in assignments else "training_in_every_fold"} for c in conditions]
    write_tsv(output_dir / "cv_condition_errors.tsv", cv_rows)
    write_tsv(output_dir / "cv_training_errors.tsv", train_rows)
    write_tsv(output_dir / "cv_fold_summary.tsv", fold_rows)
    write_tsv(output_dir / "cv_alpha_summary.tsv", alpha_rows)
    write_tsv(output_dir / "fold_membership.tsv", membership)
    deterministic_npz(output_dir / "cv_oof_predictions.npz", alphas=np.asarray(alphas, dtype=float), conditions=np.asarray(doubles), standardized_effects=oof_predictions)
    selection = {
        "status": "PASS_FROZEN_TRAINING_ONLY_SELECTION", "protocol_sha256": protocol_hash,
        "selected_alpha": alpha, "minimum_validation_mse": minimum,
        "selected_validation_mse": next(r["validation_mse"] for r in alpha_rows if r["selected"]),
        "selected_alpha_on_grid_boundary": alpha in (min(alphas), max(alphas)),
        "alpha_grid_expanded": False, "tie_tolerance": validation["tie_tolerance"],
        "tie_rule": validation["tie_rule"], "n_training_conditions": 161,
        "n_validation_doubles": 55, "n_folds": 5, "n_genes": 2000,
        "input_sha256": dict(input_hashes), "test_task_labels_read": False,
        "test_expression_used": False, "evaluation_controls_used": False,
        "fit_code_sha256": sha256(__file__), "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "output_sha256": {p.name: sha256(p) for p in sorted(output_dir.iterdir()) if p.is_file()},
    }
    selection_path = output_dir / "selection_receipt.json"
    write_json(selection_path, selection)
    selection_hash = sha256(selection_path)
    access_events.append({"event": "selection_receipt_written", "sha256": selection_hash})

    # Refit on all original training conditions before opening test identifiers.
    coefficients = ridge_coefficients(design, target, alpha)
    raw_coefficients = coefficients * scales[None, :]
    train_prediction = design @ coefficients
    train_errors = np.mean((train_prediction - target) ** 2, axis=1)
    if np.any(train_prediction[conditions.index("ctrl")]):
        raise AssertionError("The fitted predictor must have exactly zero effect at ctrl")
    write_tsv(output_dir / "refit_training_errors.tsv", [{"condition": c, "condition_order": len(components(c)), "standardized_mse": e} for c, e in zip(conditions, train_errors)])
    training_summary = [{"condition_group": name, "n_conditions": int(mask.sum()), "standardized_mse": float(train_errors[mask].mean())}
                        for name, mask in [("all", np.ones(161, dtype=bool)), ("ctrl", np.array([c == "ctrl" for c in conditions])),
                                           ("single", np.array([len(components(c)) == 1 for c in conditions])),
                                           ("double", np.array([len(components(c)) == 2 for c in conditions]))]]
    write_tsv(output_dir / "refit_training_summary.tsv", training_summary)
    np.save(output_dir / "standardized_coefficients.npy", coefficients, allow_pickle=False)
    np.save(output_dir / "effect_coefficients.npy", raw_coefficients, allow_pickle=False)
    np.save(output_dir / "training_control_baseline.npy", baseline, allow_pickle=False)
    np.save(output_dir / "scales.npy", scales, allow_pickle=False)
    (output_dir / "gene_panel.tsv").write_bytes(panel_path.read_bytes())
    write_tsv(output_dir / "feature_axis.tsv", [{"feature_index": i, "perturbed_gene": f} for i, f in enumerate(features)])
    checkpoint_path = output_dir / "direct_effect_checkpoint.npz"
    deterministic_npz(checkpoint_path, standardized_coefficients=coefficients, effect_coefficients=raw_coefficients,
                      training_control_baseline=baseline, scales=scales, feature_names=np.asarray(features),
                      gene_symbols=gene_panel["gene_symbol"].to_numpy(dtype=str), ensembl_ids=gene_panel["ensembl_id"].to_numpy(dtype=str),
                      training_conditions=np.asarray(conditions), training_design=design, selected_alpha=np.asarray(alpha),
                      protocol_sha256=np.asarray(protocol_hash), selection_receipt_sha256=np.asarray(selection_hash))
    checkpoint_hash = sha256(checkpoint_path)
    access_events.append({"event": "all_training_refit_checkpoint_written", "sha256": checkpoint_hash})

    # This is deliberately the first access to held-out task identifiers.
    task_path = checked_input("test_tasks.tsv", "prediction_after_selection_and_refit")
    tasks = pd.read_csv(task_path, sep="\t")
    if "condition" not in tasks.columns:
        raise ValueError("Test task table must identify conditions in a condition column")
    test_conditions = tasks["condition"].tolist()
    if len(test_conditions) != protocol["evaluation"]["test_conditions"] or len(set(test_conditions)) != len(test_conditions):
        raise ValueError("Expected 55 unique held-out test conditions")
    if any(not isinstance(c, str) or len(components(c)) != 2 for c in test_conditions) or set(test_conditions) & set(conditions):
        raise ValueError("Test conditions must be unseen doubles of the trained constituents")
    test_design = make_design(test_conditions, features)
    test_standardized = test_design @ coefficients
    test_effects = test_standardized * scales[None, :]
    if not np.isfinite(test_effects).all():
        raise ValueError("Nonfinite fixed effect prediction")
    prediction_path = output_dir / "test_effect_predictions.npz"
    deterministic_npz(prediction_path, conditions=np.asarray(test_conditions), effects=test_effects,
                      standardized_effects=test_standardized, input_design=test_design, feature_names=np.asarray(features),
                      gene_symbols=gene_panel["gene_symbol"].to_numpy(dtype=str), ensembl_ids=gene_panel["ensembl_id"].to_numpy(dtype=str))
    if sha256(selection_path) != selection_hash:
        raise AssertionError("Selection receipt changed after test identifiers were opened")
    receipt = {
        "status": "PASS_FROZEN_DIRECT_EFFECT_FIT", "protocol_sha256": protocol_hash,
        "model_id": model["id"], "native_output": "effect", "has_intercept": False,
        "selected_alpha": alpha, "validation_mse": selection["selected_validation_mse"],
        "training_mse": float(train_errors.mean()), "selected_alpha_on_grid_boundary": selection["selected_alpha_on_grid_boundary"],
        "alpha_grid_expanded": False, "n_train_conditions": len(conditions), "n_test_conditions": len(test_conditions),
        "n_input_features": len(features), "n_output_genes": len(scales),
        "selection_receipt_sha256": selection_hash, "checkpoint_sha256": checkpoint_hash,
        "test_expression_used": False, "evaluation_controls_used": False,
        "test_task_labels_read_only_after_selection": True, "test_task_labels_read_only_after_refit": True,
        "reference_input_arguments": [], "evaluation_control_recentring": False,
        "input_access_events": access_events, "input_sha256": input_hashes,
        "fit_code_sha256": sha256(__file__), "python_version": platform.python_version(),
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "output_sha256": {p.name: sha256(p) for p in sorted(output_dir.iterdir()) if p.is_file()},
    }
    write_json(output_dir / "fit_receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "selected_alpha": alpha, "validation_mse": receipt["validation_mse"],
                      "training_mse": receipt["training_mse"], "prediction_sha256": sha256(prediction_path)}, sort_keys=True))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    fit(args.prepared_dir, args.protocol, args.output_dir)


if __name__ == "__main__":
    main()
