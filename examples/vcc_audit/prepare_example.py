#!/usr/bin/env python3
"""Prepare a public H1 tutorial fixture; execute with Python >=3.11 on a CPU job.

Creates data and didactic predictions, not trained models or VCC leaderboard
results. Downloads and scoring are deliberately owned by the example runner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


UPSTREAM_COMMIT = "5e64833518a6603a0301cbe28185d49c30f4a986"
SOURCE_URL = (
    "https://raw.githubusercontent.com/ArcInstitute/cell-eval2/"
    f"{UPSTREAM_COMMIT}/docs/data/H1-VCC-2025-training.h5ad"
)
SOURCE_SHA256 = "eb36c766cbf76353f9981cb3a3aa32137622d1de53b29d861c483742bcd4dec7"
BASELINE_SOURCE_SHA256 = "dac2f014cf41ac36e8160bfaa2a65508778adb8a5b1e3522d25d3f16642f3abf"
TUTORIAL_URL = (
    "https://github.com/ArcInstitute/cell-eval2/blob/"
    f"{UPSTREAM_COMMIT}/docs/tutorial.md"
)
DATASET_URL = "https://huggingface.co/datasets/arcinstitute/VCC_train"
PERT_COL = "target_gene"
CONTROL_LABEL = "non-targeting"
ID_PREFIX = "h1-vcc2025:"
PERTURBATIONS = ("MED12", "SRC", "STAT1", "TET1", "TMSB4X")
INPUT_SEED = 1729
TRUTH_SEED = 2718
BASELINE_SEED = 31415


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def write_ids(path: Path, values: list[str]) -> None:
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def artifact(path: Path) -> dict:
    return {"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}


def prepare(source: Path, output: Path) -> dict:
    if sys.version_info < (3, 11):
        raise RuntimeError("Use the separate cell-eval2 Python >=3.11 environment")
    source = source.resolve(strict=True)
    output = output.resolve()
    examples_root = Path(__file__).resolve().parent
    if output == examples_root or examples_root in output.parents:
        raise ValueError("Generated outputs must be outside examples/vcc_audit")
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    if sha256(source) != SOURCE_SHA256:
        raise ValueError("Source SHA256 differs from the pinned public tutorial fixture")

    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    import cell_eval2.baseline as official_baseline

    baseline_source = Path(official_baseline.__file__).resolve()
    if sha256(baseline_source) != BASELINE_SOURCE_SHA256:
        raise ValueError("Imported official baseline.py differs from the pinned revision")
    generator_sha = sha256(Path(__file__).resolve())
    original = ad.read_h5ad(source)
    if original.shape != (600, 1000):
        raise ValueError(f"Unexpected public tutorial shape: {original.shape}")
    if PERT_COL not in original.obs or original.obs[PERT_COL].isna().any():
        raise ValueError("Missing perturbation labels")
    if not original.obs_names.is_unique or not original.var_names.is_unique:
        raise ValueError("Source cell and gene identifiers must be unique")
    original_ids = [str(value) for value in original.obs_names]
    if any(not value or "\n" in value or "\r" in value for value in original_ids):
        raise ValueError("Cell IDs must be nonempty single-line strings")
    if len(set(original_ids)) != original.n_obs:
        raise ValueError("String conversion makes source cell IDs ambiguous")

    def count_checks(matrix, *, integral: bool) -> dict:
        values = matrix.data if sp.issparse(matrix) else np.asarray(matrix).ravel()
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Count matrix contains nonfinite or negative values")
        fractional = int(np.count_nonzero(values != np.rint(values)))
        if integral and fractional:
            raise ValueError("Expected raw integer counts")
        totals = np.asarray(matrix.sum(axis=1), dtype=np.float64).ravel()
        if not np.isfinite(totals).all() or (totals > 1_000_000).any():
            raise ValueError("Cell total exceeds the official count limit")
        return {
            "finite": True, "nonnegative": True,
            "integer_valued": fractional == 0,
            "fractional_stored_values": fractional,
            "max_cell_total": float(totals.max()) if totals.size else 0.0,
        }

    source_checks = count_checks(original.X, integral=True)
    labels = original.obs[PERT_COL].astype(str).to_numpy()
    counts = {label: int(np.count_nonzero(labels == label)) for label in sorted(set(labels))}
    expected = {label: 100 for label in (*PERTURBATIONS, CONTROL_LABEL)}
    if counts != expected:
        raise ValueError(f"Expected five 100-cell perturbations and 100 controls, got {counts}")

    # One physical-identity namespace for ALL observed cells, before splitting.
    obs = original.obs.copy()
    obs["source_cell_id"] = original_ids
    obs["source_dataset_id"] = "h1-vcc2025"
    obs.index = pd.Index([ID_PREFIX + value for value in original_ids], name="cell_id")
    full = ad.AnnData(X=original.X.copy(), obs=obs, var=original.var.copy())
    control_rows = sorted(np.flatnonzero(labels == CONTROL_LABEL).tolist(),
                          key=lambda i: original_ids[i])
    input_rows, scoring_rows = control_rows[:40], control_rows[40:]
    input_ids = [str(full.obs_names[i]) for i in input_rows]
    scoring_ids = [str(full.obs_names[i]) for i in scoring_rows]
    if set(input_ids) & set(scoring_ids):
        raise AssertionError("Input and scoring controls overlap")
    pert_rows = np.flatnonzero(labels != CONTROL_LABEL)
    real_rows = np.concatenate([pert_rows, np.asarray(scoring_rows, dtype=np.intp)])
    real = full[real_rows].copy()
    inputs = full[input_rows].copy()
    real.obs["example_role"] = np.where(
        real.obs[PERT_COL].astype(str).to_numpy() == CONTROL_LABEL,
        "scoring_control", "observed_perturbation")
    inputs.obs["example_role"] = "public_input_control"

    prediction_labels = [label for label in PERTURBATIONS for _ in range(100)]

    def prediction(matrix, name: str):
        pred_obs = pd.DataFrame({PERT_COL: prediction_labels,
                                 "example_arm": name},
                                index=pd.Index([
                                    f"didactic:{name}:{label}:{i:03d}"
                                    for label in PERTURBATIONS for i in range(100)
                                ], name="prediction_id"))
        return ad.AnnData(X=sp.csr_matrix(matrix), obs=pred_obs, var=full.var.copy())

    input_rng = np.random.default_rng(INPUT_SEED)
    sampled_rows = input_rng.integers(0, inputs.n_obs, size=500)
    input_prediction = prediction(inputs.X[sampled_rows].copy(), "input_mean")

    truth_rng = np.random.default_rng(TRUTH_SEED)
    truth_blocks = []
    for label in PERTURBATIONS:
        values = full[labels == label].X
        mean = np.asarray(values.astype(np.float64).mean(axis=0)).ravel()
        truth_blocks.append(truth_rng.poisson(mean, size=(100, full.n_vars)))
    truth_prediction = prediction(np.vstack(truth_blocks), "truth_informed")

    # Calibration is an organizer-side oracle. Its profile and emission use the
    # full original tutorial panel (all 100 controls), as explicitly recorded.
    # Keep the exact official floating-count emission; do not round it to integers.
    profile = official_baseline.generic_response_profile(
        full, pert_col=PERT_COL, control=CONTROL_LABEL, exclude_target_gene=True)
    baseline_full = official_baseline.build_baseline_prediction(
        profile, full, pert_col=PERT_COL, control=CONTROL_LABEL,
        emit="dispersed", seed=BASELINE_SEED)
    baseline_prediction = baseline_full[
        baseline_full.obs[PERT_COL].astype(str).to_numpy() != CONTROL_LABEL].copy()
    baseline_prediction.obs_names = pd.Index([
        f"calibration:{i:04d}" for i in range(baseline_prediction.n_obs)
    ], name="prediction_id")

    generated_checks = {
        "input_mean": count_checks(input_prediction.X, integral=True),
        "truth_informed": count_checks(truth_prediction.X, integral=True),
        "calibration_baseline": count_checks(baseline_prediction.X, integral=False),
    }
    rules = {
        "input_mean": {
            "schema": "vcc_audit.example_generation_rule.v1", "trained_model": False,
            "rule": "resample public input-control cells with replacement; 100 per perturbation",
            "seed": INPUT_SEED, "bit_generator": "PCG64", "source_sha256": SOURCE_SHA256,
            "input_control_ids": input_ids, "perturbation_order": list(PERTURBATIONS),
            "uses_observed_perturbed_expression": False,
        },
        "truth_informed": {
            "schema": "vcc_audit.example_generation_rule.v1", "trained_model": False,
            "rule": "independent Poisson draws from the observed mean of each perturbation",
            "seed": TRUTH_SEED, "bit_generator": "PCG64", "source_sha256": SOURCE_SHA256,
            "cells_per_perturbation": 100, "perturbation_order": list(PERTURBATIONS),
            "uses_observed_perturbed_expression": True,
            "interpretation": "truth-informed interface prop; not a deployable predictor",
        },
        "calibration_baseline": {
            "schema": "vcc_audit.example_generation_rule.v1", "trained_model": False,
            "rule": "official generic_response_profile then build_baseline_prediction",
            "emit": "dispersed", "seed": BASELINE_SEED,
            "exclude_target_gene": True, "rounded_after_emission": False,
            "source_sha256": SOURCE_SHA256, "upstream_commit": UPSTREAM_COMMIT,
            "profile_source": "all 500 observed perturbed cells; equal weight per perturbation",
            "emission_template": "full original 600-cell panel; resampling all 100 controls",
            "interpretation": "organizer calibration oracle; not a model candidate",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.prepare-", dir=output.parent))
    try:
        products = {
            "real": real, "input_controls": inputs, "input_mean": input_prediction,
            "truth_informed": truth_prediction, "calibration_baseline": baseline_prediction,
        }
        for name, data in products.items():
            data.write_h5ad(stage / f"{name}.h5ad")
        write_ids(stage / "control_ids.txt", scoring_ids)
        write_ids(stage / "input_control_ids.txt", input_ids)
        write_ids(stage / "all_control_ids.txt", input_ids + scoring_ids)
        with (stage / "cell_identity_map.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["source_cell_id", "cell_id", "target_gene", "example_role"])
            for i, old_id in enumerate(original_ids):
                cell_id = str(full.obs_names[i])
                role = ("public_input_control" if cell_id in set(input_ids) else
                        "scoring_control" if cell_id in set(scoring_ids) else
                        "observed_perturbation")
                writer.writerow([old_id, cell_id, labels[i], role])
        predictions = {}
        for name, rule in rules.items():
            path = stage / f"{name}_generation_rule.json"
            write_json(path, rule)
            predictions[name] = {
                "artifact": artifact(stage / f"{name}.h5ad"),
                "generation_rule": artifact(path),
                "checkpoint_sha256": sha256(path),
                "checkpoint_semantics": "generation-rule file hash; no trained checkpoint",
                "trained_model": False,
                "count_checks": generated_checks[name],
            }
        with (stage / "input_mean_resampling.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["prediction_id", "input_cell_id"])
            writer.writerows(zip(input_prediction.obs_names,
                                [input_ids[int(i)] for i in sampled_rows]))
        if sha256(source) != SOURCE_SHA256 or sha256(Path(__file__).resolve()) != generator_sha:
            raise RuntimeError("Input or generator changed during fixture preparation")
        if sha256(baseline_source) != BASELINE_SOURCE_SHA256:
            raise RuntimeError("Official baseline implementation changed during preparation")
        receipt = {
            "schema": "reference_design.vcc_public_example_source.v1",
            "status": "COMPLETE_PUBLIC_EXAMPLE_FIXTURE",
            "scientific_scope": "software demonstration on a public VCC2025 training subset",
            "vcc2026_empirical_evidence": False,
            "source": {"url": SOURCE_URL, "sha256": SOURCE_SHA256,
                       "dataset_url": DATASET_URL, "tutorial_url": TUTORIAL_URL,
                       "upstream_commit": UPSTREAM_COMMIT,
                       "shape": list(original.shape), "perturbation_counts": counts,
                       "count_checks": source_checks,
                       "source_terms": "Public VCC_train dataset and upstream tutorial; retain their source terms"},
            "identity": {"dataset_id": "h1-vcc2025", "prefix": ID_PREFIX,
                         "mapping": "cell_identity_map.tsv", "one_namespace_before_split": True},
            "split": {"rule": "sort original control cell IDs; first 40 input, remaining 60 scoring",
                      "input_controls": 40, "scoring_controls": 60,
                      "observed_perturbed_cells": 500, "observed_genes": 1000,
                      "overlap": 0, "input_control_ids": input_ids,
                      "scoring_control_ids": scoring_ids,
                      "meaning": "example-specific control holdout within public training data"},
            "seeds": {"input_mean": INPUT_SEED, "truth_informed": TRUTH_SEED,
                      "calibration_baseline": BASELINE_SEED},
            "predictions": predictions,
            "calibration": {"profile_observed_cells": 500, "profile_perturbations": 5,
                            "profile_target_gene_exclusions": int(profile.n_excluded),
                            "emission_template_cells": 600, "emission_control_cells": 100,
                            "retained_prediction_cells": 500, "retained_control_rows": 0,
                            "baseline_source_sha256": BASELINE_SOURCE_SHA256,
                            "control_injection": "worker assigns each scoring view's C_pred controls",
                            "fractional_counts": "preserved exactly from official dispersed emission"},
            "generator_sha256": generator_sha,
            "environment": {"python": sys.version.split()[0],
                            **{name: importlib.metadata.version(name) for name in
                               ("anndata", "numpy", "scipy", "pandas", "cell-eval2")}},
            "files": [artifact(path) for path in sorted(stage.iterdir()) if path.is_file()],
        }
        write_json(stage / "SOURCE.json", receipt)
        if output.exists():
            raise FileExistsError(f"Output appeared during preparation: {output}")
        os.rename(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = prepare(args.source, args.output)
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()),
                      "source_sha256": SOURCE_SHA256}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
