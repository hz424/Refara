#!/usr/bin/env python3
"""Prepare a pinned public count-panel example (Python >=3.11).

Selection uses source metadata and sorted cell IDs, never performance scores.
Three didactic prediction arms and an official calibration oracle are emitted;
none is a trained model. Run numerical preparation inside a scheduled CPU job.
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
SOURCE_URL = "https://zenodo.org/api/records/13350497/files/DixitRegev2016_K562_TFs_13_days.h5ad/content"
SOURCE_SHA256 = "f7c483fa2a3cb79740ee816850ad7b8432f2c4c17fb4ac6fdd2cbe32a29f11a1"
SOURCE_MD5 = "b594d224cf17b268cd7e02efc976cea7"
BASELINE_SOURCE_SHA256 = "dac2f014cf41ac36e8160bfaa2a65508778adb8a5b1e3522d25d3f16642f3abf"
DATASET_ID = "scperturb-dixit2016-k562-tfs-13days"
ID_PREFIX = DATASET_ID + ":"
PERT_COL, CONTROL_LABEL = "target_gene", "non-targeting"
CELLS_PER_PERTURBATION, INPUT_COUNT, SCORING_COUNT = 200, 200, 400
MAX_TARGETS = 20
INPUT_SEED, TRUTH_SEED, BASELINE_SEED = 1729, 2718, 31415

DATASET_PROFILES = {
    "dixit": {
        "source_url": SOURCE_URL, "source_sha256": SOURCE_SHA256, "source_md5": SOURCE_MD5,
        "shape": [19268, 21713], "dataset_id": DATASET_ID,
        "citation": "Dixit et al. (2016), Perturb-Seq; scPerturb standardized dataset",
        "publication_doi": "10.1016/j.cell.2016.11.038",
        "metadata_kind": "dixit_single_target",
        "scientific_scope": "public Dixit K562 software demonstration with didactic predictions",
    },
    "adamson10x005": {
        "source_url": "https://zenodo.org/api/records/13350497/files/AdamsonWeissman2016_GSM2406677_10X005.h5ad/content",
        "source_sha256": "6c6eca0f53f8887b86597e2a4ff512ff2b2d3d9c78ee7deec9a6e7d6ae859d01",
        "source_md5": "8657391920e7f8b3e6fd52745777002a",
        "shape": [15006, 32738], "dataset_id": "scperturb-adamson2016-k562-10x005",
        "citation": "Adamson et al. (2016), K562 10X005; scPerturb standardized dataset",
        "publication_doi": "10.1016/j.cell.2016.11.048",
        "metadata_kind": "adamson_explicit_single_target_constructs",
        "scientific_scope": "public Adamson K562 software demonstration with didactic predictions",
        "construct_to_gene": {"ATF6_only_pMJ145": "ATF6", "IRE1_only_pMJ148": "ERN1",
                              "PERK_only_pMJ146": "EIF2AK3"},
        "control_constructs": ["3x_neg_ctrl_pMJ144-1", "3x_neg_ctrl_pMJ144-2"],
        "mapping_sources": [
            {"url": "https://rest.genenames.org/fetch/symbol/ERN1", "hgnc_id": "HGNC:3449",
             "source_alias": "IRE1", "symbol": "ERN1",
             "snapshot_sha256": "1c019eb9a54c1b5fbb5c31a179d91cc4e9b1d1be8216e7b7b74156dbc981f6b3"},
            {"url": "https://rest.genenames.org/fetch/symbol/EIF2AK3", "hgnc_id": "HGNC:3255",
             "source_alias": "PERK", "symbol": "EIF2AK3",
             "snapshot_sha256": "54d13b6afd458af387bc8d0aca2c3f92b8d08171686163a8ede1c11ed62f08a8"},
        ],
        "nperts_interpretation": "not used for biological target count: scPerturb computes this field as underscore-separated label segments",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def artifact(path: Path) -> dict:
    return {"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}


def prepare(source: Path, output: Path, dataset: str = "dixit") -> dict:
    if sys.version_info < (3, 11):
        raise RuntimeError("Use the separate cell-eval2 Python >=3.11 environment")
    if dataset not in DATASET_PROFILES:
        raise ValueError(f"Unknown pinned dataset profile: {dataset}")
    profile_spec = DATASET_PROFILES[dataset]
    SOURCE_SHA256, SOURCE_URL = profile_spec["source_sha256"], profile_spec["source_url"]
    SOURCE_MD5, DATASET_ID = profile_spec["source_md5"], profile_spec["dataset_id"]
    ID_PREFIX = DATASET_ID + ":"
    source, output = source.resolve(strict=True), output.resolve()
    examples_root = Path(__file__).resolve().parent
    if output == examples_root or examples_root in output.parents:
        raise ValueError("Generated outputs must be outside examples/vcc_audit")
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    if sha256(source) != SOURCE_SHA256:
        raise ValueError("Source SHA256 differs from the pinned Zenodo artifact")

    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    import cell_eval2.baseline as official_baseline

    baseline_source = Path(official_baseline.__file__).resolve()
    if sha256(baseline_source) != BASELINE_SOURCE_SHA256:
        raise ValueError("Official baseline implementation differs from the pinned revision")
    generator_sha = sha256(Path(__file__).resolve())
    original = ad.read_h5ad(source)
    if list(original.shape) != profile_spec["shape"]:
        raise ValueError(f"Unexpected pinned {dataset} shape: {original.shape}")
    if not original.obs_names.is_unique or not original.var_names.is_unique:
        raise ValueError("Unique source cell and gene identifiers are required")
    required_columns = (("perturbation", "target", "guide_id", "nperts")
                        if dataset == "dixit" else ("perturbation",))
    for column in required_columns:
        if column not in original.obs:
            raise ValueError(f"Missing source metadata column: {column}")
    original_ids = [str(value) for value in original.obs_names]
    if (len(set(original_ids)) != original.n_obs or
            any(not value or any(c in value for c in "\t\r\n") for value in original_ids)):
        raise ValueError("Cell IDs must be unique, nonempty, single-line identifiers")

    def count_checks(matrix, *, integral: bool) -> dict:
        values = matrix.data if sp.issparse(matrix) else np.asarray(matrix).ravel()
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Count matrix contains nonfinite or negative values")
        fractional = int(np.count_nonzero(values != np.rint(values)))
        if integral and fractional:
            raise ValueError("Expected raw integer counts; no conversion is performed")
        totals = np.asarray(matrix.sum(axis=1), dtype=np.float64).ravel()
        if not np.isfinite(totals).all() or (totals > 1_000_000).any():
            raise ValueError("Cell total exceeds the official count limit")
        return {"finite": True, "nonnegative": True, "integer_valued": fractional == 0,
                "fractional_stored_values": fractional,
                "max_cell_total": float(totals.max()) if totals.size else 0.0}

    source_checks = count_checks(original.X, integral=True)
    gene_names = set(original.var_names.astype(str))
    source_perturbation = original.obs["perturbation"]
    # pandas 3 may retain a float NaN in astype(str).to_numpy(); use an
    # explicit sentinel while testing validity against the original metadata.
    raw_construct = np.asarray(source_perturbation.astype(object).where(
        source_perturbation.notna(), ""), dtype=str)
    if dataset == "dixit":
        source_target = np.asarray(original.obs["target"].astype(object).where(
            original.obs["target"].notna(), ""), dtype=str)
        target = source_target.copy()
        is_control = source_perturbation.eq("control").fillna(False).to_numpy(dtype=bool)
        metadata_valid = (source_perturbation.notna().to_numpy() &
                          original.obs["target"].notna().to_numpy() &
                          original.obs["guide_id"].notna().to_numpy() &
                          original.obs["nperts"].eq(1).to_numpy())
        selection_rule = "nonmissing perturbation/target/guide_id and nperts=1; noncontrol target exactly on gene axis; >=200 cells; lexicographic first20; first200 sorted original cell IDs per target"
        declared_mapping = {}
    else:
        source_target = raw_construct.copy()
        declared_mapping = profile_spec["construct_to_gene"]
        if not set(declared_mapping.values()) <= gene_names:
            raise ValueError("Declared canonical target is absent from the complete source gene axis")
        target = np.asarray([declared_mapping.get(value, value) for value in raw_construct], dtype=str)
        is_control = np.isin(raw_construct, profile_spec["control_constructs"])
        metadata_valid = (source_perturbation.notna().to_numpy() &
                          (is_control | np.isin(raw_construct, list(declared_mapping))))
        selection_rule = "explicit named single-target constructs mapped to canonical gene symbols; exactly the two declared negative-control constructs; exclude combinations/unassigned constructs; target present on full gene axis and >=200 cells; lexicographic first20; first200 sorted original cell IDs per target; nperts is not a biological-target-count filter"
    control_rows = sorted(np.flatnonzero(is_control & metadata_valid).tolist(),
                          key=lambda i: original_ids[i])
    if len(control_rows) < 2 * INPUT_COUNT + SCORING_COUNT:
        raise ValueError("Insufficient metadata-valid controls for the fixed three-way split")
    target_rows, selection_rows = {}, []
    for label in sorted(set(target[~is_control])):
        raw_rows = np.flatnonzero((target == label) & ~is_control)
        valid_rows = np.flatnonzero((target == label) & ~is_control & metadata_valid)
        present = label in gene_names
        eligible = present and len(valid_rows) >= CELLS_PER_PERTURBATION
        selection_rows.append({"source_target": label, "raw_noncontrol_cells": len(raw_rows),
                               "metadata_valid_noncontrol_cells": len(valid_rows),
                               "target_present_on_gene_axis": present, "eligible": eligible})
        if eligible:
            target_rows[label] = sorted(valid_rows.tolist(), key=lambda i: original_ids[i])
    perturbations = sorted(target_rows)[:MAX_TARGETS]
    if len(perturbations) < 2:
        raise ValueError("Fewer than two eligible single-target perturbations")
    for row in selection_rows:
        row["selected"] = row["source_target"] in perturbations
    pert_rows = [i for label in perturbations for i in target_rows[label][:CELLS_PER_PERTURBATION]]
    primary_rows = control_rows[:INPUT_COUNT]
    alternative_rows = control_rows[INPUT_COUNT:2 * INPUT_COUNT]
    scoring_rows = control_rows[2 * INPUT_COUNT:2 * INPUT_COUNT + SCORING_COUNT]
    # One physical-cell namespace is assigned before any partition. No cell/gene
    # matrix transformation is made: in particular the full gene axis is retained.
    obs = original.obs.copy()
    obs["source_cell_id"] = original_ids
    obs["source_dataset_id"] = DATASET_ID
    obs["source_target"] = source_target
    obs[PERT_COL] = np.where(is_control, CONTROL_LABEL, target)
    obs.index = pd.Index([ID_PREFIX + value for value in original_ids], name="cell_id")
    full = ad.AnnData(X=original.X.copy(), obs=obs, var=original.var.copy())
    inputs = full[primary_rows].copy()
    alternatives = full[alternative_rows].copy()
    real = full[pert_rows + scoring_rows].copy()
    primary_ids = inputs.obs_names.astype(str).tolist()
    alternative_ids = alternatives.obs_names.astype(str).tolist()
    scoring_ids = full.obs_names[scoring_rows].astype(str).tolist()
    memberships = [set(primary_ids), set(alternative_ids), set(scoring_ids)]
    if any(memberships[i] & memberships[j] for i in range(3) for j in range(i)):
        raise AssertionError("Public input and scoring partitions overlap")
    inputs.obs["example_role"] = "public_input_control"
    alternatives.obs["example_role"] = "alternative_public_input_control"
    real.obs["example_role"] = np.where(real.obs[PERT_COL].eq(CONTROL_LABEL),
                                           "scoring_control", "observed_perturbation")
    prediction_labels = [label for label in perturbations for _ in range(CELLS_PER_PERTURBATION)]
    prediction_size = len(prediction_labels)

    def prediction(matrix, name):
        prediction_obs = pd.DataFrame({PERT_COL: prediction_labels, "example_arm": name},
                                      index=pd.Index([
                                          f"didactic:{name}:{label}:{i:03d}"
                                          for label in perturbations
                                          for i in range(CELLS_PER_PERTURBATION)],
                                          name="prediction_id"))
        return ad.AnnData(X=sp.csr_matrix(matrix), obs=prediction_obs, var=full.var.copy())

    # Both input arms apply the identical algorithm/seed to distinct source cells.
    sampled_rows = np.random.default_rng(INPUT_SEED).integers(0, INPUT_COUNT, size=prediction_size)
    input_prediction = prediction(inputs.X[sampled_rows].copy(), "input_mean")
    alternative_prediction = prediction(alternatives.X[sampled_rows].copy(), "input_mean_alternative")
    truth_rng = np.random.default_rng(TRUTH_SEED)
    truth_blocks = []
    for label in perturbations:
        values = full[target_rows[label][:CELLS_PER_PERTURBATION]].X
        mean = np.asarray(values.astype(np.float64).mean(axis=0)).ravel()
        truth_blocks.append(sp.csr_matrix(truth_rng.poisson(mean, size=(CELLS_PER_PERTURBATION, full.n_vars))))
    truth_prediction = prediction(sp.vstack(truth_blocks, format="csr"), "truth_informed")

    # This transductive official comparator is separate from candidate input pools.
    # Its profile uses the selected observed panel; its emission resamples all
    # metadata-valid source controls, including the declared input/scoring pools.
    # The worker replaces these original controls with each scoring view's C_pred.
    template = full[pert_rows + control_rows].copy()
    profile = official_baseline.generic_response_profile(
        template, pert_col=PERT_COL, control=CONTROL_LABEL, exclude_target_gene=True)
    if int(profile.n_excluded) != len(perturbations):
        raise ValueError("Official baseline did not resolve every selected target gene")
    baseline_full = official_baseline.build_baseline_prediction(
        profile, template, pert_col=PERT_COL, control=CONTROL_LABEL,
        emit="dispersed", seed=BASELINE_SEED)
    baseline_prediction = baseline_full[~baseline_full.obs[PERT_COL].eq(CONTROL_LABEL).to_numpy()].copy()
    baseline_prediction.obs_names = pd.Index([f"calibration:{i:05d}" for i in range(prediction_size)],
                                             name="prediction_id")
    products = {"real": real, "input_controls": inputs,
                "input_controls_alternative": alternatives,
                "input_mean": input_prediction, "input_mean_alternative": alternative_prediction,
                "truth_informed": truth_prediction, "calibration_baseline": baseline_prediction}
    generated_checks = {name: count_checks(data.X, integral=name != "calibration_baseline")
                        for name, data in products.items()}
    input_rule = {
        "schema": "vcc_audit.example_generation_rule.v1", "trained_model": False,
        "rule": "resample cells with replacement from the supplied public input pool",
        "seed": INPUT_SEED, "bit_generator": "PCG64", "source_sha256": SOURCE_SHA256,
        "input_pool_size": INPUT_COUNT, "cells_per_perturbation": CELLS_PER_PERTURBATION,
        "perturbation_order": perturbations, "uses_observed_perturbed_expression": False,
        "input_binding": "Input membership is a separate runtime argument, not a fitted parameter",
    }
    truth_rule = {
        "schema": "vcc_audit.example_generation_rule.v1", "trained_model": False,
        "rule": "independent Poisson draws from each selected observed perturbation mean",
        "seed": TRUTH_SEED, "bit_generator": "PCG64", "source_sha256": SOURCE_SHA256,
        "cells_per_perturbation": CELLS_PER_PERTURBATION, "perturbation_order": perturbations,
        "uses_observed_perturbed_expression": True,
        "interpretation": "truth-informed interface example; not a deployable predictor",
    }
    baseline_rule = {
        "schema": "vcc_audit.example_generation_rule.v1", "trained_model": False,
        "rule": "official generic_response_profile then build_baseline_prediction",
        "emit": "dispersed", "seed": BASELINE_SEED, "exclude_target_gene": True,
        "rounded_after_emission": False, "source_sha256": SOURCE_SHA256,
        "upstream_commit": UPSTREAM_COMMIT, "profile_source": "selected observed perturbation panel",
        "profile_observed_cells": len(pert_rows), "profile_equal_weight_per_target": True,
        "emission_control_cells": len(control_rows), "emission_template_cells": template.n_obs,
        "emission_control_membership": "baseline_template_control_ids.txt",
        "interpretation": "organizer calibration oracle; not a model candidate",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.prepare-", dir=output.parent))
    try:
        for name, data in products.items():
            data.write_h5ad(stage / f"{name}.h5ad", compression="gzip")
        id_files = {"control_ids.txt": scoring_ids, "input_control_ids.txt": primary_ids,
                    "input_alt_control_ids.txt": alternative_ids,
                    "all_control_ids.txt": full.obs_names[control_rows].astype(str).tolist(),
                    "baseline_template_control_ids.txt": full.obs_names[control_rows].astype(str).tolist(),
                    "observed_perturbation_ids.txt": full.obs_names[pert_rows].astype(str).tolist()}
        for filename, ids in id_files.items():
            (stage / filename).write_text("".join(value + "\n" for value in ids), encoding="utf-8")
        selected_pert_ids = set(full.obs_names[pert_rows])
        baseline_control_ids = set(full.obs_names[control_rows])
        with (stage / "cell_identity_map.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["source_cell_id", "cell_id", "source_target", "example_role", "baseline_control_template"])
            for i, old_id in enumerate(original_ids):
                cell_id = str(full.obs_names[i])
                role = ("public_input_control" if cell_id in memberships[0] else
                        "alternative_public_input_control" if cell_id in memberships[1] else
                        "scoring_control" if cell_id in memberships[2] else
                        "observed_perturbation" if cell_id in selected_pert_ids else "not_in_scoring_example")
                writer.writerow([old_id, cell_id, source_target[i], role, cell_id in baseline_control_ids])
        with (stage / "target_selection.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(selection_rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(selection_rows)
        for name, rule in (("input_mean", input_rule), ("truth_informed", truth_rule),
                           ("calibration_baseline", baseline_rule)):
            write_json(stage / f"{name}_generation_rule.json", rule)
        predictions = {}
        for name in ("input_mean", "input_mean_alternative", "truth_informed", "calibration_baseline"):
            rule_name = "input_mean" if name == "input_mean_alternative" else name
            path = stage / f"{rule_name}_generation_rule.json"
            predictions[name] = {"artifact": artifact(stage / f"{name}.h5ad"),
                                 "generation_rule": artifact(path), "checkpoint_sha256": sha256(path),
                                 "checkpoint_semantics": "generation-rule file hash; no trained checkpoint",
                                 "trained_model": False, "count_checks": generated_checks[name]}
        for name, data, source_ids in (("input_mean", input_prediction, primary_ids),
                                      ("input_mean_alternative", alternative_prediction, alternative_ids)):
            with (stage / f"{name}_resampling.tsv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(["prediction_id", "input_cell_id", "input_row_index"])
                writer.writerows((pred_id, source_ids[int(i)], int(i))
                                 for pred_id, i in zip(data.obs_names, sampled_rows))
        if sha256(source) != SOURCE_SHA256 or sha256(Path(__file__).resolve()) != generator_sha:
            raise RuntimeError("Source or generator changed during preparation")
        if sha256(baseline_source) != BASELINE_SOURCE_SHA256:
            raise RuntimeError("Official baseline changed during preparation")
        receipt = {
            "schema": "reference_design.vcc_public_example_source.v1",
            "status": "COMPLETE_PUBLIC_EXAMPLE_FIXTURE", "dataset": dataset,
            "scientific_scope": profile_spec["scientific_scope"],
            "vcc2026_empirical_evidence": False, "performance_scoring_performed": False,
            "pert_col": PERT_COL, "control_label": CONTROL_LABEL, "alloc_depths": [100, 200],
            "source": {"url": SOURCE_URL, "sha256": SOURCE_SHA256, "md5": SOURCE_MD5,
                       "dataset_url": "https://zenodo.org/records/13350497",
                       "dataset_citation": profile_spec["citation"],
                       "publication_doi": profile_spec["publication_doi"], "license": "CC-BY-4.0",
                       "upstream_commit": UPSTREAM_COMMIT, "shape": list(original.shape),
                       "count_matrix": "X", "count_checks": source_checks,
                       "matrix_transformation": "none; original full unique gene axis retained",
                       "unused_layers": "Source layers and raw are not copied; counts are taken from X"},
            "target_selection": {
                "rule": selection_rule,
                "selection_uses_scores_or_effects": False,
                "source_missing_perturbation_cells": int(source_perturbation.isna().sum()),
                "eligible_targets": sorted(target_rows), "selected_targets": perturbations,
                "cells_per_target": CELLS_PER_PERTURBATION,
                "target_gene_mapping": {label: label for label in perturbations},
                "source_construct_to_gene": declared_mapping,
                "source_construct_counts": {value: int(np.sum(raw_construct == value)) for value in sorted(set(raw_construct))},
                "control_constructs": profile_spec.get("control_constructs", ["control"]),
                "gene_alias_sources": profile_spec.get("mapping_sources", []),
                "nperts_interpretation": profile_spec.get("nperts_interpretation", "required to equal1 in the pinned Dixit profile"),
                "detailed_counts": "target_selection.tsv"},
            "identity": {"dataset_id": DATASET_ID, "prefix": ID_PREFIX,
                         "mapping": "cell_identity_map.tsv", "one_namespace_before_split": True},
            "split": {"rule": "sort metadata-valid source control IDs; first200 primary input, next200 alternative input, next400 scoring; remaining controls only in calibration template",
                      "input_controls": INPUT_COUNT, "input_alt_controls": INPUT_COUNT,
                      "scoring_controls": SCORING_COUNT, "source_controls": len(control_rows),
                      "observed_perturbed_cells": len(pert_rows), "observed_genes": full.n_vars,
                      "overlap": 0, "input_control_ids": primary_ids,
                      "input_alt_control_ids": alternative_ids, "scoring_control_ids": scoring_ids,
                      "meaning": "example partitions within a public source; not independent biological units"},
            "seeds": {"input_mean": INPUT_SEED, "input_mean_alternative": INPUT_SEED,
                      "truth_informed": TRUTH_SEED, "calibration_baseline": BASELINE_SEED},
            "predictions": predictions,
            "calibration": {"profile_observed_cells": len(pert_rows),
                            "profile_perturbations": len(perturbations),
                            "profile_target_gene_exclusions": int(profile.n_excluded),
                            "emission_template_cells": template.n_obs,
                            "emission_control_cells": len(control_rows),
                            "retained_prediction_cells": prediction_size, "retained_control_rows": 0,
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
    parser.add_argument("--dataset", choices=sorted(DATASET_PROFILES), default="dixit")
    args = parser.parse_args()
    result = prepare(args.source, args.output, dataset=args.dataset)
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()),
                      "dataset": args.dataset, "source_sha256": result["source"]["sha256"],
                      "selected_targets": result["target_selection"]["selected_targets"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
