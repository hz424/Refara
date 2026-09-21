#!/usr/bin/env python3
"""Independently replay original and separated public-example scoring with upstream APIs.

Run with the official Python >=3.11 environment. No reference_design module is
imported. Expected scores use newly built views and a newly computed official
bundle; adapter bundles and responses are read only as values under test.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback


UPSTREAM_COMMIT = "5e64833518a6603a0301cbe28185d49c30f4a986"
METRICS = (
    "pds_cosine", "expr_mse_unbiased_capped_norm",
    "de_wilcoxon_direction_fidelity_yield_raw", "de_wilcoxon_direction_reach_raw",
    "de_wilcoxon_sig_jaccard", "de_wilcoxon_lfc_nmae",
)
ABS_TOLERANCE = 1e-10
REL_TOLERANCE = 1e-10


def baseline_rejection_message(message: str) -> bool:
    prefix = "the baseline leg is degenerate on metric(s) that decide a ranking: "
    suffix = ". A bundle built on it could not be scored."
    if not isinstance(message, str) or not message.startswith(prefix) or not message.endswith(suffix):
        return False
    detail = message[len(prefix):-len(suffix)]
    return bool(detail) and all(item.split("=", 1)[0] in METRICS for item in detail.split("; "))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=unique)


def write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def assemble_scoring_prediction(pred, p_controls, reference, *, pert_col: str,
                                control_label: str, context_id: str):
    """Build the independent scoring view with only its explicit observation fields.

    Peripheral source metadata can contain mixed strings and missing values after
    concatenation. It is not part of this single-context scoring contract. Keep
    the original matrices, ordered genes, perturbation labels and qualified IDs.
    """
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp

    for data in (pred, p_controls):
        if (not data.obs_names.is_unique or not data.var_names.is_unique
                or not data.var_names.equals(reference.var_names)):
            raise ValueError("Independent scoring inputs must have unique cells and the full ordered gene axis")
        if pert_col not in data.obs or data.obs[pert_col].isna().any():
            raise ValueError("Independent scoring input has a missing perturbation label")
        if "context" in data.obs and (data.obs["context"].isna().any()
                                      or not (data.obs["context"].astype(str) == context_id).all()):
            raise ValueError("Independent scoring input belongs to a different context")
    predicted_labels = pred.obs[pert_col].astype(str).tolist()
    control_labels = p_controls.obs[pert_col].astype(str).tolist()
    if control_label in predicted_labels or any(label != control_label for label in control_labels):
        raise ValueError("Independent scoring inputs mix candidate and control roles")
    prediction_ids = ["prediction::" + str(cell) for cell in pred.obs_names]
    control_ids = list(p_controls.obs_names.astype(str))
    if set(prediction_ids) & set(control_ids):
        raise ValueError("Prediction/control identities collide after qualification")
    matrix = sp.vstack([sp.csr_matrix(pred.X), sp.csr_matrix(p_controls.X)], format="csr")
    obs = pd.DataFrame({pert_col: predicted_labels + control_labels,
                        "context": [context_id] * (pred.n_obs + p_controls.n_obs)},
                       index=pd.Index(prediction_ids + control_ids))
    joined = ad.AnnData(X=matrix, obs=obs, var=reference.var.copy())
    if ((joined.X[:pred.n_obs] != sp.csr_matrix(pred.X)).nnz
            or (joined.X[pred.n_obs:] != sp.csr_matrix(p_controls.X)).nnz):
        raise AssertionError("Independent assembly changed a candidate or control expression value")
    return joined


class Verifier:
    def __init__(self):
        self.bindings: dict[str, str] = {}
        self.comparisons: list[dict] = []

    def bind(self, path: Path, expected: str | None = None) -> Path:
        path = path.resolve(strict=True)
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError(f"SHA256 mismatch: {path}")
        if str(path) in self.bindings and self.bindings[str(path)] != actual:
            raise ValueError(f"File changed during verification: {path}")
        self.bindings[str(path)] = actual
        return path

    def artifact(self, spec: dict, base: Path) -> Path:
        return self.bind(base / spec["path"], spec["sha256"])

    def inventory(self, root: Path, recorded: dict) -> None:
        for relative, expected in recorded.items():
            path = (root / relative).resolve(strict=True)
            if not path.is_relative_to(root.resolve()):
                raise ValueError(f"Artifact escapes its receipt directory: {relative}")
            self.bind(path, expected)

    def compare(self, label: str, expected, actual) -> None:
        if expected is None or actual is None:
            raise ValueError(f"Unavailable value in required parity comparison: {label}")
        x, y = float(expected), float(actual)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"Nonfinite value in required parity comparison: {label}")
        error = abs(x - y)
        allowed = ABS_TOLERANCE + REL_TOLERANCE * abs(x)
        self.comparisons.append({"comparison": label, "expected": x, "observed": y,
                                 "absolute_error": error, "allowed_error": allowed,
                                 "passed": error <= allowed,
                                 "comparison_kind": "numeric_tolerance",
                                 "semantic_status": "FINITE_REQUIRED"})

    def compare_anchor_field(self, label: str, *, metric: str, field: str,
                             expected_row: dict, observed_row: dict,
                             expected_splits: list[dict], observed_splits: list[dict],
                             seeds: list[int]) -> None:
        """Keep the pinned derived metric's absent cohort counts as exact null slots.

        Official anchor.py:188–195, 272–273, 289–290: the normalized expression
        error is an aggregate ratio with no per-perturbation tidy row. Its split
        cohort counts and their extrema are null, while its split values and
        replicate statistics remain finite. No other scored field qualifies.
        """
        fields = {"replicate", "replicate_sd", "replicate_min", "replicate_max",
                  "n_perturbations_min", "n_perturbations_max"}
        if metric not in METRICS or field not in fields:
            raise ValueError("Unexpected scored anchor metric or field")
        estimator = "full_gate_raw" if metric == "de_wilcoxon_lfc_nmae" else "split_half_raw"
        for row in (expected_row, observed_row):
            if row.get("metric") != metric or row.get("estimator") != estimator:
                raise ValueError("Anchor metric or estimator differs from the pinned producer")
        expected, observed = expected_row[field], observed_row[field]
        count_field = field in {"n_perturbations_min", "n_perturbations_max"}
        if metric == "expr_mse_unbiased_capped_norm" and count_field:
            if expected is not None or observed is not None:
                raise ValueError("Derived panel anchor cohort-count extrema must both be literal null")
            if len(seeds) != 5 or len(set(seeds)) != 5:
                raise ValueError("Expected the five pinned anchor split seeds")
            for splits in (expected_splits, observed_splits):
                selected = [row for row in splits if row.get("metric") == metric]
                if (len(selected) != len(seeds)
                        or {row["split_index"] for row in selected} != set(range(len(seeds)))):
                    raise ValueError("Derived panel anchor does not cover every split exactly once")
                for row in selected:
                    if (row["seed"] != seeds[row["split_index"]]
                            or row["n_perturbations"] is not None
                            or row["value"] is None or not math.isfinite(float(row["value"]))):
                        raise ValueError("Derived panel split must have its pinned seed, finite value and null cohort count")
            self.comparisons.append({"comparison": label, "expected": None, "observed": None,
                                     "absolute_error": None, "allowed_error": None, "passed": True,
                                     "comparison_kind": "structural_null_exact",
                                     "semantic_status": "NOT_APPLICABLE_DERIVED_PANEL_AGGREGATE"})
        else:
            if count_field and (type(expected) is not int or type(observed) is not int
                                or expected <= 0 or observed <= 0 or expected != observed):
                raise ValueError("Applicable anchor cohort counts must be positive, equal integers")
            self.compare(label, expected, observed)

    def recheck(self) -> None:
        for path, expected in self.bindings.items():
            if digest(Path(path)) != expected:
                raise ValueError(f"Bound input or observed result changed: {path}")


def verify_full_roster(example: Path, manifest: dict, checks: Verifier) -> dict:
    """Authenticate every terminal response and every declared metric/pair slot."""
    summary_root = example / "audit" / "summary"
    summary = read_json(checks.bind(summary_root / "SUMMARY.json"))
    checks.inventory(summary_root, summary["artifacts"])
    checks.bind(example / "audit" / "PLAN.json", summary["plan_sha256"])
    checks.bind(example / "audit/requests/score/SCORE_PLAN.json", summary["score_plan_sha256"])
    plan = read_json(example / "audit/PLAN.json")
    checks.bind(example / "manifest.json", plan["manifest_sha256"])
    checks.bind(example / "audit/MANIFEST_SNAPSHOT.json", plan["manifest_sha256"])
    execution = read_json(checks.bind(example / "audit/EXECUTION.json"))
    with (summary_root / "METRIC_SCORES.tsv").open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    with (summary_root / "COMPLETE_MODEL_PAIRS.tsv").open(newline="") as stream:
        pairs = list(csv.DictReader(stream, delimiter="\t"))
    metrics = (*METRICS, "avg_score")
    expected_rows, expected_pairs, expected_responses = set(), set(), set()
    row_map = {(r["context_id"], r["allocation_id"], r["model_id"], r["metric"]): r for r in rows}
    pair_map = {(r["context_id"], r["allocation_id"], r["model_a"], r["model_b"], r["metric"]): r for r in pairs}
    if len(row_map) != len(rows) or len(pair_map) != len(pairs):
        raise ValueError("Duplicated model/metric or pair slots in the full summary")
    unavailable_preparations, unavailable_workers, scored_workers = [], 0, 0
    for context in manifest["contexts"]:
        context_id = context["context_id"]
        for allocation in context["allocations"]:
            allocation_id = allocation["allocation_id"]
            prepared_dir = example / "audit/prepared" / context_id / allocation_id
            prepared_path = checks.bind(prepared_dir / "prepared.json")
            prepared = read_json(prepared_path)
            checks.inventory(prepared_dir, prepared["artifacts"])
            if (prepared["context_id"] != context_id or prepared["panel_id"] != context["panel_id"]
                    or prepared["allocation_id"] != allocation_id
                    or prepared["reference"]["sha256"] != context["reference"]["sha256"]
                    or prepared["controls"]["recipe"] != allocation):
                raise ValueError("Full-grid preparation differs from its declared panel/allocation")
            checks.bind(example / "audit/requests/prepare" / context_id / f"{allocation_id}.json",
                        prepared["request_sha256"])
            unavailable = prepared.get("status") == "UNAVAILABLE_CALIBRATION"
            if unavailable:
                calibration = prepared["calibration"]
                if (calibration.get("available") is not False
                        or calibration.get("source") != "build_rejected"
                        or calibration.get("reason_code") != "OFFICIAL_BASELINE_SCALE_UNAVAILABLE"
                        or calibration.get("stage") != "build_real_bundle.baseline_leg_gate"
                        or calibration.get("upstream_exception") != "ValueError"
                        or not baseline_rejection_message(calibration.get("upstream_error", ""))):
                    raise ValueError("Unavailability is not the narrow declared official baseline rejection")
                unavailable_preparations.append({"context_id": context_id, "allocation_id": allocation_id})
            elif prepared.get("status") != "COMPLETE":
                raise ValueError("A full-grid preparation has an unresolved technical failure")
            for model in context["models"]:
                model_id = model["model_id"]
                score_dir = example / "audit/scores" / context_id / allocation_id / model_id
                response_path = checks.bind(score_dir / "response.json")
                expected_responses.add(str(response_path))
                response = read_json(response_path)
                if (response["context_id"] != context_id or response["allocation_id"] != allocation_id
                        or response["model_id"] != model_id
                        or response["prediction"]["sha256"] != model["prediction"]["sha256"]
                        or response["prepared_sha256"] != digest(prepared_path)):
                    raise ValueError("Full-grid response is not bound to the expected model and preparation")
                checks.bind(example / "audit/requests/score" / context_id / allocation_id / f"{model_id}.json",
                            response["request_sha256"])
                raw_records = response["metrics"]
                records = {r["metric"]: r for r in raw_records}
                if len(raw_records) != len(METRICS) or set(records) != set(METRICS):
                    raise ValueError("A response omits or duplicates an official metric")
                records["avg_score"] = response["aggregate"]
                if unavailable:
                    unavailable_workers += 1
                    if (response.get("status") != "UNAVAILABLE_CALIBRATION"
                            or response.get("official_scoring_performed") is not False
                            or response.get("artifact_role") != "validated_scoring_inputs_only"
                            or response.get("reason_code") != "OFFICIAL_BASELINE_SCALE_UNAVAILABLE"
                            or response.get("reason") != prepared["calibration"]["upstream_error"]):
                        raise ValueError("Unavailable calibration was mislabeled as a score or technical failure")
                    checks.inventory(score_dir, response["validated_artifacts"])
                else:
                    scored_workers += 1
                    if response.get("status") != "COMPLETE":
                        raise ValueError("Available calibration has an incomplete scoring response")
                    checks.inventory(score_dir, response["official_artifacts"])
                for metric in metrics:
                    key = (context_id, allocation_id, model_id, metric)
                    expected_rows.add(key)
                    row = row_map.get(key)
                    if row is None or row["panel_id"] != context["panel_id"]:
                        raise ValueError("A declared full-grid metric slot is missing")
                    record = records[metric]
                    if unavailable:
                        if (record.get("raw") is not None or record.get("from_replicate") is not None
                                or record.get("unavailable_reason") != response["reason"]
                                or row["raw"] != "NA" or row["from_replicate"] != "NA"
                                or row["available"].lower() != "false"
                                or row.get("worker_status") != "UNAVAILABLE_CALIBRATION"
                                or row.get("unavailable_reason_code") != "OFFICIAL_BASELINE_SCALE_UNAVAILABLE"
                                or row.get("unavailable_reason") != response["reason"]):
                            raise ValueError("An unavailable metric slot lost its explicit full NA record")
                    else:
                        if row["available"].lower() != "true" or not math.isfinite(float(record["from_replicate"])):
                            raise ValueError("An available metric has a missing/nonfinite score")
                        if float(row["from_replicate"]) != float(record["from_replicate"]):
                            raise ValueError("Summary calibrated value differs from the bound response")
                        if metric != "avg_score" and (not math.isfinite(float(record["raw"]))
                                                       or float(row["raw"]) != float(record["raw"])):
                            raise ValueError("Summary raw value differs from the bound response")
            for model_a, model_b in itertools.combinations(sorted(m["model_id"] for m in context["models"]), 2):
                for metric in metrics:
                    key = (context_id, allocation_id, model_a, model_b, metric)
                    expected_pairs.add(key)
                    pair = pair_map.get(key)
                    if pair is None:
                        raise ValueError("A declared model-pair slot is missing")
                    if unavailable and (pair["classification"] != "unavailable"
                                        or pair["ranking_reversal"] != "NA"
                                        or pair["value_a"] != "NA" or pair["value_b"] != "NA"
                                        or pair["difference_A_minus_B"] != "NA"):
                        raise ValueError("A pair with unavailable calibration was assigned a numeric comparison")
    if set(row_map) != expected_rows or set(pair_map) != expected_pairs:
        raise ValueError("Full-grid summary has extra or missing roster entries")
    bound_responses = {str(Path(r["path"]).resolve()): r for r in summary["responses"]}
    if len(bound_responses) != len(summary["responses"]) or set(bound_responses) != expected_responses:
        raise ValueError("Summary response bindings do not cover the complete full grid")
    for path, binding in bound_responses.items():
        checks.bind(Path(path), binding["sha256"])
    expected_status = "UNAVAILABLE_OFFICIAL_SCORES" if unavailable_workers else "COMPLETE_DESCRIPTIVE_VCC_AUDIT"
    expected_execution = "COMPLETE_WITH_UNAVAILABLE_ANALYSES" if unavailable_workers else "COMPLETE"
    if (summary["status"] != expected_status or execution["status"] != expected_execution
            or summary["expected_workers"] != len(expected_responses)
            or summary["complete_workers"] != len(expected_responses)
            or summary["metric_rows"] != len(rows) or summary["pair_rows"] != len(pairs)
            or summary["unavailable_score_rows"] != unavailable_workers * len(metrics)
            or summary["unavailable_calibration_workers"] != unavailable_workers
            or summary["scored_workers"] != scored_workers
            or summary["failed_or_missing_workers"] != 0):
        raise ValueError("Full-grid completion counts/status do not match authenticated terminal responses")
    return {"status": "PASS_FULL_ROSTER_AND_UNAVAILABLE_ROWS", "expected_workers": len(expected_responses),
            "terminal_workers": len(expected_responses), "scored_workers": scored_workers,
            "unavailable_calibration_workers": unavailable_workers,
            "unavailable_preparations": unavailable_preparations,
            "metric_rows": len(rows), "pair_rows": len(pairs),
            "execution_status": expected_execution, "summary_status": expected_status}


def verify(example: Path, output: Path, checks: Verifier) -> dict:
    if sys.version_info < (3, 11):
        raise RuntimeError("Use the official Python >=3.11 interpreter")
    example = example.resolve(strict=True)
    manifest_path = checks.bind(example / "manifest.json")
    manifest = read_json(manifest_path)
    if manifest.get("schema") != "reference_design.vcc_audit.v1":
        raise ValueError("Unexpected example manifest schema")
    if len(manifest["contexts"]) != 1:
        raise ValueError("This example verification requires exactly one context")
    context = manifest["contexts"][0]
    models = context["models"]
    model_ids = [m["model_id"] for m in models]
    if len(models) < 2 or len(set(model_ids)) != len(models):
        raise ValueError("Expected at least two uniquely named prediction arms")
    if any(not isinstance(value, str) or not value or value in {".", ".."}
           or Path(value).name != value for value in model_ids):
        raise ValueError("Model IDs must be single safe path components")
    original = [a for a in context["allocations"] if a["kind"] == "original"]
    separated = [a for a in context["allocations"] if a["reference_policy"] == "separated"]
    if len(original) != 1 or original[0]["reference_policy"] != "shared" or not separated:
        raise ValueError("Need one original shared construction and a separated construction")
    # Chosen from manifest order before reading scores, not from observed performance.
    selected = [original[0], separated[0]]
    shared_probes = [a for a in context["allocations"]
                     if a["kind"] == "equal_depth" and a["reference_policy"] == "shared"]
    if not shared_probes:
        raise ValueError("The full example must include an equal-depth shared design")
    # A third, prespecified fixture checks the unavailable-calibration path.
    unavailable_probe = shared_probes[0]
    full_roster = verify_full_roster(example, manifest, checks)
    write_json(output / "FULL_ROSTER_VERIFICATION.json", full_roster)
    upstream = (example / manifest["upstream"]["path"]).resolve(strict=True)
    head = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if head != UPSTREAM_COMMIT or manifest["upstream"]["commit"] != UPSTREAM_COMMIT:
        raise ValueError("Official checkout does not match the pinned revision")
    subprocess.run(["git", "-C", str(upstream), "diff", "--exit-code", "HEAD", "--",
                    "src", "pyproject.toml"], check=True, capture_output=True)
    for path in sorted((upstream / "src" / "cell_eval2").rglob("*.py")):
        checks.bind(path)
    checks.bind(upstream / "pyproject.toml")
    threads = manifest["runtime"]["num_threads"]
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
        os.environ[key] = str(threads)
    sys.path.insert(0, str(upstream / "src"))
    import anndata as ad
    import numpy as np
    import pandas as pd
    import polars as pl
    import scipy.sparse as sp
    import cell_eval2 as ce
    from cell_eval2.baseline import build_run_meta
    from cell_eval2.anchor import read_anchor
    from cell_eval2.competition import competition_members
    from cell_eval2.real_bundle import build_real_bundle
    from cell_eval2.run import metric_output_names

    if not Path(ce.__file__).resolve().is_relative_to(upstream / "src"):
        raise ValueError("Imported official scorer is outside its pinned checkout")
    if set(competition_members()) != set(METRICS):
        raise ValueError("Pinned official scored metric membership differs")
    source_receipt_path = checks.bind(example / "inputs" / "SOURCE.json")
    source_receipt = read_json(source_receipt_path)
    if source_receipt.get("status") != "COMPLETE_PUBLIC_EXAMPLE_FIXTURE":
        raise ValueError("Example source preparation is not complete")
    for spec in source_receipt["files"]:
        checks.artifact(spec, example / "inputs")
    reference_path = checks.artifact(context["reference"], example)
    baseline_path = checks.artifact(context["calibration"]["baseline_prediction"], example)
    reference = ad.read_h5ad(reference_path)
    if not reference.obs_names.is_unique or not reference.var_names.is_unique:
        raise ValueError("Reference identities must be unique")
    pert_col, control = context["pert_col"], context["control_label"]
    label_series = reference.obs[pert_col]
    if label_series.isna().any():
        raise ValueError("Missing reference perturbation label")
    labels = label_series.astype(str).to_numpy()
    real_ids = set(reference.obs_names.astype(str))
    declared = context["scoring_pool"]["control_ids"]
    if len(declared) != len(set(declared)):
        raise ValueError("Duplicate declared scoring controls")
    if set(declared) != set(reference.obs_names[labels == control]):
        raise ValueError("Example scoring controls must cover the real input's control rows")
    observed_perturbations = set(labels) - {control}
    input_memberships = {}
    for pool in context["input_pools"]:
        data = ad.read_h5ad(checks.artifact(pool["artifact"], example))
        ids = pool["control_ids"]
        if len(ids) != len(set(ids)) or not set(ids) <= set(data.obs_names):
            raise ValueError("Invalid example input control membership")
        if set(ids) & set(declared):
            raise ValueError("Example input and scoring controls overlap")
        if not (data[ids].obs[pert_col].astype(str) == control).all():
            raise ValueError("Input pool includes non-control cells")
        input_memberships[pool["input_pool_id"]] = set(ids)
    for model in models:
        conditioning = model["conditioning"]
        generated = source_receipt["predictions"][model["model_id"]]
        if (generated["artifact"]["sha256"] != model["prediction"]["sha256"] or
                generated["checkpoint_sha256"] != conditioning["checkpoint_sha256"] or
                generated["generation_rule"]["sha256"] != conditioning["checkpoint_sha256"]):
            raise ValueError("Manifest prediction or generation rule differs from the source receipt")
        if conditioning["input_pool_id"] not in input_memberships:
            raise ValueError("Model input pool is not authenticated")
        if conditioning["mode"] == "input_intervention":
            parent = next((m for m in models if m["model_id"] == conditioning["parent_model_id"]), None)
            if parent is None:
                raise ValueError("Input intervention lacks its parent prediction")
            parent_conditioning = parent["conditioning"]
            if (parent_conditioning["mode"] != "fixed_submission" or
                    not parent_conditioning["uses_input_controls"] or
                    not conditioning["uses_input_controls"] or
                    parent_conditioning["checkpoint_sha256"] != conditioning["checkpoint_sha256"]):
                raise ValueError("Input intervention changes the declared generation rule")
            if input_memberships[conditioning["input_pool_id"]] == input_memberships[parent_conditioning["input_pool_id"]]:
                raise ValueError("Input intervention must use different actual input cells")

    def load_prediction(path: Path):
        data = ad.read_h5ad(path)
        if not data.obs_names.is_unique or not data.var_names.is_unique:
            raise ValueError("Prediction axes must be unique")
        if set(data.var_names) != set(reference.var_names):
            raise ValueError("Prediction gene support differs")
        data = data[:, reference.var_names].copy()
        labs = data.obs[pert_col].astype(str)
        if control in set(labs) or set(labs) != observed_perturbations:
            raise ValueError("Example predictions must contain the complete non-control panel only")
        return data

    baseline = load_prediction(baseline_path)
    predictions = {m["model_id"]: load_prediction(checks.artifact(m["prediction"], example)) for m in models}

    def assemble(pred, p_controls):
        return assemble_scoring_prediction(pred, p_controls, reference, pert_col=pert_col,
                                           control_label=control, context_id=context["context_id"])

    def mean_row(table):
        rows = table.filter(pl.col(table.columns[0]) == "mean").to_dicts()
        if len(rows) != 1:
            raise ValueError("Official aggregate must have one mean row")
        return rows[0]

    def score_rows(table):
        rows = table.to_dicts()
        mapping = {row["metric"]: row for row in rows}
        if len(mapping) != len(rows) or not (set(METRICS) | {"avg_score"}) <= set(mapping):
            raise ValueError("Score table has duplicate or missing scored metrics")
        return mapping

    cases = []
    panel_aggregation = []

    def ratio_of_sums_evidence(tidy, *, allocation_id, model_id, expected, observed):
        # Inspect the official per-perturbation components, then independently
        # aggregate with Python math.fsum. This does not replace the metric.
        components = {}
        for metric in ("expr_mse_unbiased_capped", "expr_distance_unbiased"):
            rows = [r for r in tidy.to_dicts() if r["metric"] == metric]
            mapping = {r["perturbation"]: r["value"] for r in rows}
            if len(mapping) != len(rows) or set(mapping) != observed_perturbations:
                raise ValueError("Expression-error components omit or duplicate panel members")
            components[metric] = mapping
        numerator = components["expr_mse_unbiased_capped"]
        denominator = components["expr_distance_unbiased"]
        selected_labels = [label for label in sorted(observed_perturbations)
                           if numerator[label] is not None and denominator[label] is not None
                           and math.isfinite(float(numerator[label]))
                           and math.isfinite(float(denominator[label]))]
        if not selected_labels:
            raise ValueError("Expression-error components have no paired finite values")
        ns = [float(numerator[label]) for label in selected_labels]
        ds = [float(denominator[label]) for label in selected_labels]
        n_sum, d_sum = math.fsum(ns), math.fsum(ds)
        if not math.isfinite(n_sum) or not math.isfinite(d_sum) or d_sum <= 0:
            raise ValueError("Expression-error ratio has an invalid panel denominator")
        ratio = n_sum / d_sum
        checks.compare(f"{allocation_id}/{model_id}/independent_ratio_of_sums/upstream", ratio, expected)
        checks.compare(f"{allocation_id}/{model_id}/independent_ratio_of_sums/adapter", ratio, observed)
        mean_ratio = math.fsum(n / d for n, d in zip(ns, ds)) / len(ds) if all(d != 0 for d in ds) else None
        if mean_ratio is not None and not math.isfinite(mean_ratio):
            mean_ratio = None
        contrast = (max(ds) - min(ds) > ABS_TOLERANCE + REL_TOLERANCE * max(abs(d) for d in ds)
                    and mean_ratio is not None
                    and abs(mean_ratio - ratio) > ABS_TOLERANCE + REL_TOLERANCE * abs(ratio))
        panel_aggregation.append({
            "allocation_id": allocation_id, "model_id": model_id,
            "metric": "expr_mse_unbiased_capped_norm",
            "numerator_component": "expr_mse_unbiased_capped",
            "denominator_component": "expr_distance_unbiased",
            "panel_perturbations": len(observed_perturbations), "paired_finite_perturbations": len(ds),
            "excluded_nonfinite_labels": sorted(observed_perturbations - set(selected_labels)),
            "numerator_sum": n_sum, "denominator_sum": d_sum,
            "denominator_min": min(ds), "denominator_max": max(ds),
            "ratio_of_sums": ratio, "mean_of_per_perturbation_ratios": mean_ratio,
            "absolute_ratio_vs_mean_difference": abs(ratio - mean_ratio) if mean_ratio is not None else None,
            "unequal_denominators_and_distinct_aggregates_demonstrated": contrast,
            "official_aggregate": expected, "adapter_aggregate": observed,
            "component_source": "fresh independent official compute_metrics output",
            "component_rows": [{"perturbation": label,
                                "numerator": (float(numerator[label]) if numerator[label] is not None
                                              and math.isfinite(float(numerator[label])) else None),
                                "denominator": (float(denominator[label]) if denominator[label] is not None
                                                and math.isfinite(float(denominator[label])) else None)}
                               for label in sorted(observed_perturbations)]})
    for allocation in selected:
        allocation_id = allocation["allocation_id"]
        actual_dir = example / "audit" / "prepared" / context["context_id"] / allocation_id
        prepared_path = checks.bind(actual_dir / "prepared.json")
        prepared = read_json(prepared_path)
        if prepared.get("status") != "COMPLETE":
            raise ValueError(f"Adapter preparation is incomplete: {allocation_id}")
        if (prepared.get("context_id") != context["context_id"]
                or prepared.get("panel_id") != context["panel_id"]
                or prepared.get("allocation_id") != allocation_id
                or prepared["reference"]["sha256"] != context["reference"]["sha256"]):
            raise ValueError("Adapter preparation does not identify the expected source panel")
        request_path = example / "audit" / "requests" / "prepare" / context["context_id"] / f"{allocation_id}.json"
        checks.bind(request_path, prepared["request_sha256"])
        request = read_json(request_path)
        if request["allocation"] != allocation or prepared["controls"]["recipe"] != allocation:
            raise ValueError("Recorded allocation differs from its source manifest")
        checks.inventory(actual_dir, prepared["artifacts"])
        members = prepared["controls"]
        observation, prediction = members["observation_ids"], members["prediction_ids"]
        for ids in (observation, prediction):
            if len(ids) != len(set(ids)) or not set(ids) <= set(declared):
                raise ValueError("Recorded controls are duplicated or absent from source scoring pool")
            if not set(ids) <= real_ids or not (reference[ids].obs[pert_col].astype(str) == control).all():
                raise ValueError("Recorded members are not source control cells")
        if allocation["kind"] == "original":
            if set(observation) != set(declared) or observation != prediction:
                raise ValueError("Original design does not use the whole shared scoring pool")
        else:
            if set(observation) & set(prediction):
                raise ValueError("Separated control blocks overlap")
            stratum_col = context.get("stratum_col")
            groups = (reference.obs[stratum_col].astype(str).to_dict() if stratum_col else
                      {cell: "all" for cell in declared})
            expected_groups = {groups[cell] for cell in declared}
            for ids in (observation, prediction):
                counts = {group: sum(groups[cell] == group for cell in ids) for group in expected_groups}
                if any(n != allocation["depth_per_stratum"] for n in counts.values()):
                    raise ValueError("Recorded separated membership violates per-stratum depth")
        row_mask = (labels != control) | np.asarray([str(v) in set(observation) for v in reference.obs_names])
        real = reference[row_mask].copy()
        p_controls = reference[prediction].copy()
        cfg = ce.EvalConfig.from_preset("vcc2026")
        cfg = replace(cfg, pert_col=pert_col, control=control,
                      control_source="real" if allocation["reference_policy"] == "shared" else "pred",
                      device=manifest["runtime"]["device"], num_threads=threads,
                      de=replace(cfg.de, backend=manifest["runtime"]["de_backend"]))
        if not cfg.cache_strict:
            raise ValueError("Pinned official preset must provide strict source fingerprints")
        case_dir = output / allocation_id
        case_dir.mkdir()
        expected_real = case_dir / "real.h5ad"
        real.write_h5ad(expected_real)
        cfg.to_yaml(str(case_dir / "official_config.yaml"))
        expected_baseline = case_dir / "baseline_prediction.h5ad"
        assemble(baseline, p_controls).write_h5ad(expected_baseline)
        bundle = case_dir / "independent_real_bundle"
        build_real_bundle(str(expected_real), str(expected_baseline), config=cfg,
                          outdir=str(bundle), bundle_id="independent_public_example_parity",
                          base_seed=0, n_splits=5)
        actual_bundle = actual_dir / "real_bundle"
        expected_bundle_manifest = read_json(bundle / "manifest.json")
        actual_bundle_manifest = read_json(actual_bundle / "manifest.json")
        for field in ("source_fingerprint", "config_digest", "rule_digest", "control_source_effective",
                      "estimators", "derived_seeds", "n_splits", "bulk_target_sum"):
            if expected_bundle_manifest[field] != actual_bundle_manifest[field]:
                raise ValueError(f"Official bundle semantic mismatch for {allocation_id}/{field}")
        base_expected = mean_row(pl.read_csv(bundle / "baseline_agg.csv"))
        base_observed = mean_row(pl.read_csv(actual_bundle / "baseline_agg.csv"))
        expected_anchor, expected_splits, expected_anchor_meta = read_anchor(str(bundle))
        observed_anchor, observed_splits, observed_anchor_meta = read_anchor(str(actual_bundle))
        if (expected_anchor_meta["derived_seeds"] != observed_anchor_meta["derived_seeds"]
                or expected_anchor_meta["derived_seeds"] != expected_bundle_manifest["derived_seeds"]):
            raise ValueError("Official anchor split seeds differ")
        anchor_expected = {r["metric"]: r for r in expected_anchor.to_dicts()}
        anchor_observed = {r["metric"]: r for r in observed_anchor.to_dicts()}
        if (len(anchor_expected) != expected_anchor.height or len(anchor_observed) != observed_anchor.height
                or set(anchor_expected) != set(anchor_observed) or not set(METRICS) <= set(anchor_expected)):
            raise ValueError("Official anchor metrics are duplicated, missing or unmatched")
        for metric in METRICS:
            checks.compare(f"{allocation_id}/baseline/{metric}/raw", base_expected[metric], base_observed[metric])
            for field in ("replicate", "replicate_sd", "replicate_min", "replicate_max",
                          "n_perturbations_min", "n_perturbations_max"):
                checks.compare_anchor_field(f"{allocation_id}/anchor/{metric}/{field}",
                                            metric=metric, field=field,
                                            expected_row=anchor_expected[metric], observed_row=anchor_observed[metric],
                                            expected_splits=expected_splits.to_dicts(),
                                            observed_splits=observed_splits.to_dicts(),
                                            seeds=expected_anchor_meta["derived_seeds"])

        for model_id, prediction_data in predictions.items():
            model_dir = case_dir / model_id
            model_dir.mkdir()
            pred_path = model_dir / "prediction_with_controls.h5ad"
            assemble(prediction_data, p_controls).write_h5ad(pred_path)
            # Expected core: only original files, freshly assembled views, and official APIs.
            tidy = ce.compute_metrics(str(pred_path), str(expected_real), config=cfg)
            aggregate = ce.aggregate_metrics_wide(tidy, metrics=metric_output_names(cfg))
            meta = build_run_meta(cfg, str(expected_real), str(pred_path))
            scores = ce.score_metrics(aggregate, real_bundle=str(bundle), user_meta=meta)
            tidy.write_csv(model_dir / "per_perturbation.csv")
            aggregate.write_csv(model_dir / "aggregate.csv")
            scores.write_csv(model_dir / "score.csv")
            write_json(model_dir / "run_meta.json", meta)
            actual_model = example / "audit" / "scores" / context["context_id"] / allocation_id / model_id
            response_path = checks.bind(actual_model / "response.json")
            response = read_json(response_path)
            if response.get("status") != "COMPLETE" or response.get("model_id") != model_id:
                raise ValueError("Adapter response is incomplete or belongs to another model")
            expected_model = next(model for model in models if model["model_id"] == model_id)
            if response["prediction"]["sha256"] != expected_model["prediction"]["sha256"]:
                raise ValueError("Adapter response identifies a different source prediction")
            checks.inventory(actual_model, response["official_artifacts"])
            if response["prepared_sha256"] != digest(prepared_path):
                raise ValueError("Adapter score is not bound to its prepared allocation")
            score_request = example / "audit" / "requests" / "score" / context["context_id"] / allocation_id / f"{model_id}.json"
            checks.bind(score_request, response["request_sha256"])
            requested_model = read_json(score_request)["model"]
            if (requested_model["model_id"] != model_id or
                    requested_model["prediction"]["sha256"] != expected_model["prediction"]["sha256"]):
                raise ValueError("Score request is bound to a different model or prediction")
            actual_aggregate = mean_row(pl.read_csv(actual_model / "aggregate.csv"))
            actual_scores = score_rows(pl.read_csv(actual_model / "score.csv"))
            expected_aggregate = mean_row(aggregate)
            expected_scores = score_rows(scores)
            ratio_of_sums_evidence(tidy, allocation_id=allocation_id, model_id=model_id,
                                  expected=expected_aggregate["expr_mse_unbiased_capped_norm"],
                                  observed=actual_aggregate["expr_mse_unbiased_capped_norm"])
            response_metrics = {r["metric"]: r for r in response["metrics"]}
            if set(response_metrics) != set(METRICS):
                raise ValueError("Adapter response does not contain exactly six scored metrics")
            for metric in METRICS:
                prefix = f"{allocation_id}/{model_id}/{metric}"
                checks.compare(prefix + "/raw_csv", expected_aggregate[metric], actual_aggregate[metric])
                checks.compare(prefix + "/raw_response", expected_aggregate[metric], response_metrics[metric]["raw"])
                calibrated = expected_scores[metric]["from_replicate"]
                checks.compare(prefix + "/calibrated_csv", calibrated, actual_scores[metric]["from_replicate"])
                checks.compare(prefix + "/calibrated_response", calibrated, response_metrics[metric]["from_replicate"])
            expected_avg = expected_scores["avg_score"]["from_replicate"]
            checks.compare(f"{allocation_id}/{model_id}/avg_score_csv", expected_avg,
                           actual_scores["avg_score"]["from_replicate"])
            checks.compare(f"{allocation_id}/{model_id}/avg_score_response", expected_avg,
                           response["aggregate"]["from_replicate"])
        cases.append({"allocation_id": allocation_id, "reference_policy": allocation["reference_policy"],
                      "observation_ids": observation, "prediction_ids": prediction,
                      "models": list(predictions), "independent_bundle": str(bundle),
                      "upstream_rule_digest": expected_bundle_manifest["rule_digest"]})
    unavailability_parity = {
        "status": "NOT_REQUIRED_ALL_CALIBRATIONS_AVAILABLE",
        "selection_rule": "first equal-depth shared allocation in manifest order, fixed before reading outcomes",
        "all_unavailable_rows_checked": True,
    }
    if full_roster["unavailable_preparations"]:
        allocation_id = unavailable_probe["allocation_id"]
        actual_dir = example / "audit/prepared" / context["context_id"] / allocation_id
        prepared_path = checks.bind(actual_dir / "prepared.json")
        prepared = read_json(prepared_path)
        if prepared.get("status") != "UNAVAILABLE_CALIBRATION":
            raise ValueError("Prespecified shared unavailability fixture did not have the expected official rejection")
        observation = prepared["controls"]["observation_ids"]
        prediction = prepared["controls"]["prediction_ids"]
        if observation != prediction or len(observation) != len(set(observation)):
            raise ValueError("Prespecified shared probe must use identical unique O/P control memberships")
        if not set(observation) <= set(declared) or not (reference[observation].obs[pert_col].astype(str) == control).all():
            raise ValueError("Unavailability probe controls are not actual source scoring controls")
        stratum_col = context.get("stratum_col")
        strata = (reference.obs[stratum_col].astype(str).to_dict() if stratum_col else
                  {cell: "all" for cell in declared})
        for stratum in {strata[cell] for cell in declared}:
            if sum(strata[cell] == stratum for cell in observation) != unavailable_probe["depth_per_stratum"]:
                raise ValueError("Unavailability probe violates its prespecified per-stratum depth")
        control_set = set(observation)
        mask = (labels != control) | np.asarray([str(cell) in control_set for cell in reference.obs_names])
        real = reference[mask].copy()
        p_controls = reference[prediction].copy()
        cfg = ce.EvalConfig.from_preset("vcc2026")
        cfg = replace(cfg, pert_col=pert_col, control=control, control_source="real",
                      device=manifest["runtime"]["device"], num_threads=threads,
                      de=replace(cfg.de, backend=manifest["runtime"]["de_backend"]))
        probe_dir = output / "prespecified_shared_unavailability"
        probe_dir.mkdir()
        expected_real = probe_dir / "real.h5ad"
        expected_baseline = probe_dir / "baseline_prediction.h5ad"
        real.write_h5ad(expected_real)
        assemble(baseline, p_controls).write_h5ad(expected_baseline)
        cfg.to_yaml(str(probe_dir / "official_config.yaml"))
        try:
            build_real_bundle(str(expected_real), str(expected_baseline), config=cfg,
                              outdir=str(probe_dir / "independent_real_bundle"),
                              bundle_id="independent_prespecified_unavailability",
                              base_seed=0, n_splits=5)
        except ValueError as error:
            final_trace = error.__traceback__
            while final_trace.tb_next is not None:
                final_trace = final_trace.tb_next
            frame = final_trace.tb_frame
            if (type(error) is not ValueError
                    or Path(frame.f_code.co_filename).resolve() != (upstream / "src/cell_eval2/real_bundle.py").resolve()
                    or frame.f_code.co_name != "build_real_bundle"
                    or not baseline_rejection_message(str(error))):
                raise
            if str(error) != prepared["calibration"]["upstream_error"]:
                raise ValueError("Independent official baseline rejection differs from the adapter's recorded rejection") from error
            (probe_dir / "OFFICIAL_REJECTION_TRACEBACK.txt").write_text(traceback.format_exc())
            unavailability_parity.update({
                "status": "PASS_INDEPENDENT_OFFICIAL_REJECTION", "allocation_id": allocation_id,
                "reason_code": "OFFICIAL_BASELINE_SCALE_UNAVAILABLE",
                "stage": "build_real_bundle.baseline_leg_gate", "upstream_exception": "ValueError",
                "upstream_error": str(error), "observation_ids": observation, "prediction_ids": prediction,
                "adapter_prepared_sha256": digest(prepared_path),
                "independently_recomputed_unavailable_designs": [allocation_id],
                "other_unavailable_designs": "source bindings and full NA coverage verified; official rejection not recomputed",
                "scoring_rule_modified": False, "adapter_numerical_helpers_used": False})
        else:
            raise ValueError("Independent official calibration succeeded where the adapter reported unavailability")
    write_json(output / "UNAVAILABILITY_PARITY.json", unavailability_parity)
    write_json(output / "PANEL_AGGREGATION.json", {
        "schema": "reference_design.vcc_independent_panel_aggregation.v1",
        "cases": panel_aggregation,
        "contrast_demonstrated_in_any_case": any(
            row["unequal_denominators_and_distinct_aggregates_demonstrated"] for row in panel_aggregation),
        "metric_implementation_modified": False})
    checks.recheck()
    failed = [row for row in checks.comparisons if not row["passed"]]
    numeric = [row for row in checks.comparisons if row["comparison_kind"] == "numeric_tolerance"]
    structural = [row for row in checks.comparisons if row["comparison_kind"] == "structural_null_exact"]
    return {
        "schema": "reference_design.vcc_independent_official_parity.v1",
        "status": "FAIL_INDEPENDENT_OFFICIAL_PARITY" if failed else "PASS_INDEPENDENT_OFFICIAL_PARITY",
        "example": str(example), "upstream_commit": UPSTREAM_COMMIT,
        "official_version": ce.__version__, "scored_metrics": list(METRICS), "cases": cases,
        "selection_rule": "original allocation and first separated allocation in manifest order",
        "absolute_tolerance": ABS_TOLERANCE, "relative_tolerance": REL_TOLERANCE,
        "tolerance_rule": "abs(actual-expected) <= atol + rtol*abs(expected)",
        "comparison_count": len(checks.comparisons), "failed_comparisons": len(failed),
        "numeric_comparison_count": len(numeric), "structural_null_comparison_count": len(structural),
        "structural_null_contract": {
            "metric": "expr_mse_unbiased_capped_norm", "estimator": "split_half_raw",
            "fields": ["n_perturbations_min", "n_perturbations_max"],
            "status": "NOT_APPLICABLE_DERIVED_PANEL_AGGREGATE",
            "requires": "both extrema literally null; all five split cohort counts null with finite split values and pinned seeds",
            "official_source": "src/cell_eval2/anchor.py:188-195,272-273,289-290 at pinned commit"},
        "max_absolute_error": max(r["absolute_error"] for r in numeric),
        "panel_aggregation_sha256": digest(output / "PANEL_AGGREGATION.json"),
        "full_roster_verification_sha256": digest(output / "FULL_ROSTER_VERIFICATION.json"),
        "full_roster": full_roster,
        "unavailability_parity_sha256": digest(output / "UNAVAILABILITY_PARITY.json"),
        "unavailability_parity": unavailability_parity,
        "ratio_of_sums_contrast_demonstrated": any(
            row["unequal_denominators_and_distinct_aggregates_demonstrated"] for row in panel_aggregation),
        "expected_uses_adapter_numerical_helpers": False,
        "expected_uses_adapter_calibration_bundle": False,
        "separated_membership_source": "authenticated allocation receipt, independently checked against source cells and depth",
        "scope": "numeric parity for original and first separated allocations; independent official rejection for the first equal-depth shared allocation when unavailable; full metric/pair roster and all NA rows verified; no population inference",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    example = args.example.resolve(strict=True)
    if output == example or example in output.parents:
        parser.error("Place verification output outside the example directory")
    output.mkdir(parents=True, exist_ok=False)
    checks = Verifier()
    checks.bind(Path(__file__).resolve())
    try:
        receipt = verify(example, output, checks)
    except Exception as exc:
        receipt = {"schema": "reference_design.vcc_independent_official_parity.v1",
                   "status": "FAIL_INDEPENDENT_OFFICIAL_PARITY", "example": str(example),
                   "error": f"{type(exc).__name__}: {exc}",
                   "absolute_tolerance": ABS_TOLERANCE, "relative_tolerance": REL_TOLERANCE,
                   "completed_comparisons": len(checks.comparisons)}
        (output / "TRACEBACK.txt").write_text(traceback.format_exc())
    if checks.comparisons:
        with (output / "COMPARISONS.tsv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(checks.comparisons[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(checks.comparisons)
    write_json(output / "INPUT_BINDINGS.json", checks.bindings)
    receipt["input_bindings_sha256"] = digest(output / "INPUT_BINDINGS.json")
    if (output / "COMPARISONS.tsv").is_file():
        receipt["comparisons_sha256"] = digest(output / "COMPARISONS.tsv")
    write_json(output / "PARITY.json", receipt)
    print(json.dumps(receipt, indent=2, allow_nan=False))
    return 0 if receipt["status"] == "PASS_INDEPENDENT_OFFICIAL_PARITY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
