"""Separate Python >=3.11 process for the pinned official cell-eval2 scorer.

Run this FILE with the scorer interpreter, not the Python 3.10 package CLI::

    python /path/to/reference_design/vcc_worker.py --request request.json --output out

``prepare`` freezes a whole context under one control allocation and builds or
verifies its official calibration bundle once. ``score`` reuses that preparation
for a frozen prediction. No metric, replicate estimator, or calibration formula
is implemented here. Altered control assignments are labelled audit variants.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


UPSTREAM_COMMIT = "5e64833518a6603a0301cbe28185d49c30f4a986"
REQUEST_SCHEMA = "reference_design.vcc_worker.v1"
PREPARED_SCHEMA = "reference_design.vcc_prepared.v1"
RESPONSE_SCHEMA = "reference_design.vcc_response.v1"
UNAVAILABLE_CALIBRATION = "UNAVAILABLE_CALIBRATION"
BASELINE_UNAVAILABLE_CODE = "OFFICIAL_BASELINE_SCALE_UNAVAILABLE"
_BASELINE_ERROR_PREFIX = "the baseline leg is degenerate on metric(s) that decide a ranking: "
_BASELINE_ERROR_SUFFIX = ". A bundle built on it could not be scored."
METRICS = (
    "pds_cosine",
    "expr_mse_unbiased_capped_norm",
    "de_wilcoxon_direction_fidelity_yield_raw",
    "de_wilcoxon_direction_reach_raw",
    "de_wilcoxon_sig_jaccard",
    "de_wilcoxon_lfc_nmae",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict:
    def invalid(value):
        raise ValueError(f"Non-finite JSON constant: {value}")
    result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_json_pairs,
                        parse_constant=invalid)
    if not isinstance(result, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return result


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _ids(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty list")
    result = [_text(item, label) for item in value]
    if len(set(result)) != len(result):
        raise ValueError(f"{label} contains duplicate IDs")
    return result


def _integer(value: Any, label: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _request_structure(request: dict) -> None:
    common = {"schema", "operation"}
    if request.get("operation") == "prepare":
        required = common | {"context_id", "panel_id", "reference", "scoring_control_ids", "input_pools",
                             "allocation", "calibration", "upstream_path", "upstream_commit"}
        optional = {"run_id", "pert_col", "control_label", "stratum_col", "de_backend", "device",
                    "num_threads", "data_scope"}
        objects = ("reference", "allocation", "calibration")
    elif request.get("operation") == "score":
        required, optional = common | {"prepared", "model"}, {"run_id"}
        objects = ("prepared", "model")
    else:
        raise ValueError("Worker operation must be prepare or score")
    if required - request.keys() or request.keys() - required - optional:
        raise ValueError(f"Worker request fields differ: missing {sorted(required - request.keys())}; "
                         f"unknown {sorted(request.keys() - required - optional)}")
    for key in objects:
        if not isinstance(request[key], dict):
            raise ValueError(f"Worker {key} must be an object")


def artifact(spec: dict, directory: Path) -> Path:
    if not isinstance(spec, dict) or not {"path", "sha256"} <= set(spec):
        raise ValueError("File artifacts need path and sha256")
    path = (directory / _text(spec["path"], "Artifact path")).resolve()
    if not path.is_file() or sha256(path) != spec["sha256"]:
        raise ValueError(f"Artifact is missing or its SHA-256 differs: {path}")
    return path


def inventory(directory: Path) -> dict[str, str]:
    return {str(path.relative_to(directory)): sha256(path)
            for path in sorted(directory.rglob("*")) if path.is_file()}


def verify_inventory(directory: Path, expected: dict) -> None:
    if not isinstance(expected, dict) or not expected:
        raise ValueError("An artifact inventory must be nonempty")
    for relative, digest in expected.items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            raise ValueError(f"Unsafe or missing prepared artifact: {relative}")
        if sha256(path) != digest:
            raise ValueError(f"Prepared artifact changed: {relative}")


def _upstream(spec: dict, directory: Path):
    if sys.version_info < (3, 11):
        raise RuntimeError("The official scoring worker requires Python >=3.11")
    if spec.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("The worker requires the documented pinned cell-eval2 commit")
    root = (directory / _text(spec.get("upstream_path"), "upstream_path")).resolve()
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    if head != UPSTREAM_COMMIT:
        raise ValueError("cell-eval2 checkout is not at the pinned commit")
    dirty = subprocess.run(["git", "-C", str(root), "diff", "--name-only", "HEAD", "--",
                            "src", "pyproject.toml"], check=True,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        raise ValueError("Pinned upstream scientific source is modified: " + dirty)
    threads = _integer(spec.get("num_threads", 1), "num_threads")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[key] = str(threads)
    sys.path.insert(0, str(root / "src"))
    import cell_eval2
    import tomllib
    declared = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    if cell_eval2.__version__ != declared:
        raise RuntimeError("Install the pinned cell-eval2 source in this interpreter; version differs")
    if not Path(cell_eval2.__file__).resolve().is_relative_to(root / "src"):
        raise RuntimeError("Imported cell-eval2 is outside the pinned source tree")
    return root, cell_eval2


def _context(adata, context_id: str):
    if "context" in adata.obs.columns:
        mask = adata.obs["context"].astype(str).to_numpy() == context_id
        if not mask.any():
            raise ValueError(f"AnnData has no cells for context {context_id!r}")
        adata = adata[mask].copy()
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("AnnData cell and gene IDs must be unique within a context")
    if not adata.n_obs or not adata.n_vars:
        raise ValueError("Empty context or gene axis")
    return adata


def _counts(adata, *, fractional: bool = False) -> None:
    import numpy as np
    from scipy import sparse
    values = adata.X.data if sparse.issparse(adata.X) else np.asarray(adata.X)
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise ValueError("X must contain real raw counts")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("X must contain finite nonnegative raw counts")
    if np.any(values > 1_000_000):
        raise ValueError("A count already exceeds the per-cell count ceiling")
    if not fractional and np.any(values != np.floor(values)):
        raise ValueError("Prediction and reference cells must contain integral raw counts")
    totals = np.asarray(adata.X.sum(axis=1)).ravel()
    if np.any(totals > 1_000_000):
        raise ValueError("A cell exceeds the official 1,000,000-count ceiling")


def _labels(adata, pert_col: str):
    if pert_col not in adata.obs.columns:
        raise ValueError(f"Missing perturbation column {pert_col!r}")
    values = adata.obs[pert_col]
    if values.isna().any():
        raise ValueError("Missing perturbation labels")
    return values.astype(str).to_numpy()


def _read_h5ad(spec, directory, context_id, *, fractional=False):
    import anndata as ad
    path = artifact(spec, directory)
    data = _context(ad.read_h5ad(path), context_id)
    _counts(data, fractional=fractional)
    return data, path


def _pools(reference, request, directory):
    """Check declared public inputs without inferring independent experiments."""
    pert_col = request.get("pert_col", "target_gene")
    control = request.get("control_label", "non-targeting")
    scoring_ids = set(_ids(request["scoring_control_ids"], "scoring_control_ids"))
    pools = request["input_pools"]
    if not isinstance(pools, list) or not pools:
        raise ValueError("At least one explicit model-input pool is required")
    seen, checked = set(), []
    for pool in pools:
        name = _text(pool.get("input_pool_id"), "input_pool_id")
        if name in seen:
            raise ValueError("Duplicate model-input pool ID")
        seen.add(name)
        ids = _ids(pool.get("control_ids"), "input control IDs")
        data, path = _read_h5ad(pool["artifact"], directory, request["context_id"])
        if set(data.var_names) != set(reference.var_names):
            raise ValueError(f"Input pool {name} has a different gene axis")
        labels = dict(zip(data.obs_names.astype(str), _labels(data, pert_col)))
        if any(labels.get(cell) != control for cell in ids):
            raise ValueError(f"Input pool {name} contains missing or non-control IDs")
        overlap = scoring_ids.intersection(ids)
        if overlap:
            raise ValueError(f"Public model-input and held-out scoring controls overlap: {sorted(overlap)[:3]}")
        if pool.get("split") != "public_input":
            raise ValueError("Model-input pools must explicitly declare split='public_input'")
        checked.append({**pool, "artifact": {"path": str(path), "sha256": pool["artifact"]["sha256"]},
                        "control_ids": ids, "membership_verified": True,
                        "membership_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
    return checked


def _allocation(reference, request):
    import numpy as np
    declared = _ids(request["scoring_control_ids"], "scoring_control_ids")
    pert_col, control = request.get("pert_col", "target_gene"), request.get("control_label", "non-targeting")
    labels = dict(zip(reference.obs_names.astype(str), _labels(reference, pert_col)))
    if any(labels.get(cell) != control for cell in declared):
        raise ValueError("Scoring-control IDs contain missing or non-control cells")
    recipe = request["allocation"]
    _text(recipe.get("allocation_id"), "allocation_id")
    policy, kind = recipe.get("reference_policy"), recipe.get("kind")
    if policy not in {"shared", "separated"} or kind not in {"original", "equal_depth"}:
        raise ValueError("Unknown control allocation kind or reference policy")
    stratum_col = request.get("stratum_col")
    if stratum_col is not None and stratum_col not in reference.obs.columns:
        raise ValueError(f"Missing stratum column {stratum_col!r}")
    strata = {}
    for cell in declared:
        value = str(reference.obs.loc[cell, stratum_col]) if stratum_col else "all"
        if not value or (stratum_col and reference.obs.loc[cell, stratum_col] != reference.obs.loc[cell, stratum_col]):
            raise ValueError("Scoring-control stratum is missing")
        strata[cell] = value
    if kind == "original":
        if policy != "shared":
            raise ValueError("The original official contract shares its scoring controls")
        observation = [str(cell) for cell in reference.obs_names if str(cell) in set(declared)]
        prediction = list(observation)
    else:
        depth = _integer(recipe.get("depth_per_stratum"), "depth_per_stratum", 2)
        seed = _integer(recipe.get("seed"), "allocation seed", 0)
        rng = np.random.default_rng(seed)
        observation, second = [], []
        for group in sorted(set(strata.values())):
            cells = sorted(cell for cell in declared if strata[cell] == group)
            if len(cells) < 2 * depth:
                raise ValueError(f"Stratum {group!r} cannot supply two depth-{depth} blocks")
            chosen = np.asarray(cells)[rng.permutation(len(cells))]
            observation.extend(chosen[:depth].tolist())
            second.extend(chosen[depth:2 * depth].tolist())
        prediction = list(observation) if policy == "shared" else second
    selected = set(observation)
    labels_array = _labels(reference, pert_col)
    # Preserve original row order so supplied bundles retain their strict fingerprint.
    real = reference[(labels_array != control) | np.asarray([str(x) in selected for x in reference.obs_names])].copy()
    pred_control = reference[prediction].copy()
    if policy == "separated" and set(observation).intersection(prediction):
        raise ValueError("Separated scoring controls overlap")
    return real, pred_control, {
        "observation_ids": observation, "prediction_ids": prediction,
        "per_stratum_counts": {"observation": dict(Counter(strata[x] for x in observation)),
                               "prediction": dict(Counter(strata[x] for x in prediction))},
        "reference_policy": policy, "recipe": recipe,
        "sampling": "sorted strata; sorted cell IDs; numpy.default_rng(seed); two disjoint blocks",
    }


def _prediction(spec, directory, context, real, pred_control, cfg, *, baseline=False, data_scope="organizer_audit"):
    import anndata as ad
    import numpy as np
    pred, path = _read_h5ad(spec, directory, context, fractional=baseline)
    if set(pred.var_names) != set(real.var_names):
        raise ValueError("Prediction gene set differs; dropping or filling genes is forbidden")
    pred = pred[:, real.var_names].copy()
    labels = _labels(pred, cfg.pert_col)
    ntc_count = int(np.count_nonzero(labels == cfg.control))
    if ntc_count and not baseline:
        raise ValueError("VCC prediction uploads must not contain non-targeting rows")
    if ntc_count:
        pred = pred[labels != cfg.control].copy()
    labels = _labels(pred, cfg.pert_col)
    expected = set(_labels(real, cfg.pert_col)) - {cfg.control}
    if set(labels) != expected:
        raise ValueError("Prediction perturbation labels do not match the entire context panel")
    counts = dict(Counter(labels))
    if data_scope == "vcc2026_submission" and (len(expected) != 300 or real.n_vars != 18533
                                               or any(n != 400 for n in counts.values())):
        raise ValueError("VCC submission scope requires 300 targets, 18,533 genes, and 400 predicted cells each")
    # Submitted cell IDs are model-generated, not experimental identities. Qualify
    # only those IDs to avoid confusing them with the actual control membership.
    pred.obs_names = ["prediction::" + str(x) for x in pred.obs_names]
    if set(pred.obs_names).intersection(pred_control.obs_names):
        raise ValueError("Qualified prediction IDs collide with control IDs")
    joined = ad.concat([pred, pred_control], axis=0, join="inner", merge="same")
    if list(joined.var_names) != list(real.var_names):
        raise ValueError("Concatenation changed the frozen gene order")
    return joined, {"path": str(path), "sha256": spec["sha256"], "cells_per_perturbation": counts,
                    "baseline_control_rows_replaced": ntc_count if baseline else 0}


def _configuration(request, cell_eval2):
    cfg = cell_eval2.EvalConfig.from_preset("vcc2026")
    return replace(cfg, pert_col=request.get("pert_col", "target_gene"),
                   control=request.get("control_label", "non-targeting"),
                   control_source="real" if request["allocation"]["reference_policy"] == "shared" else "pred",
                   device=request.get("device", "cpu"),
                   num_threads=_integer(request.get("num_threads", 1), "num_threads"),
                   de=replace(cfg.de, backend=request.get("de_backend", "scanpy")))


def _score_table(cell_eval2, cfg, real_path, pred_path, bundle, output):
    from cell_eval2.baseline import build_run_meta
    from cell_eval2.run import metric_output_names
    tidy = cell_eval2.compute_metrics(str(pred_path), str(real_path), config=cfg)
    aggregate = cell_eval2.aggregate_metrics_wide(tidy, metrics=metric_output_names(cfg))
    meta = build_run_meta(cfg, str(real_path), str(pred_path))
    tidy.write_csv(output / "per_perturbation.csv")
    aggregate.write_csv(output / "aggregate.csv")
    write_json(output / "run_meta.json", meta)
    scores = cell_eval2.score_metrics(aggregate, real_bundle=str(bundle), user_meta=meta,
                                     output=str(output / "score.csv"))
    valid_counts = {metric: sum(1 for row in tidy.to_dicts()
                                if row["metric"] == metric and _finite(row["value"]) is not None)
                    for metric in METRICS if metric != "expr_mse_unbiased_capped_norm"}
    return aggregate, scores, meta, valid_counts


def _finite(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _result_values(aggregate, scores, valid_counts=None):
    rows = {row["statistic"]: row for row in aggregate.to_dicts()}
    score_rows = {row["metric"]: row for row in scores.to_dicts()}
    result = []
    for metric in METRICS:
        raw = _finite(rows.get("mean", {}).get(metric))
        calibrated = _finite(score_rows.get(metric, {}).get("from_replicate"))
        count = (_finite(valid_counts.get(metric)) if valid_counts is not None
                 else _finite(rows.get("count", {}).get(metric)))
        result.append(dict(metric=metric, raw=raw, from_replicate=calibrated,
                           valid_count=None if count is None else int(count),
                           unavailable_reason=None if raw is not None and calibrated is not None
                           else "Official scorer did not return a finite value"))
    mean = _finite(score_rows.get("avg_score", {}).get("from_replicate"))
    if mean is None or any(row["unavailable_reason"] for row in result):
        raise ValueError("The complete six-metric calibrated result is unavailable; no reduced average is allowed")
    return result, {"from_replicate": mean, "unavailable_reason": None}


def _supplied_bundle(calibration: dict, directory: Path, controls: dict) -> Path:
    """Authenticate supplied files before the official semantic checks at prepare/score."""
    if controls["recipe"]["kind"] != "original" or controls["reference_policy"] != "shared":
        raise ValueError("Supplied official bundles are only accepted for the original shared-control view")
    bundle_spec = calibration["bundle"]
    supplied = (directory / _text(bundle_spec.get("path"), "Bundle path")).resolve()
    if sha256(supplied / "manifest.json") != bundle_spec.get("manifest_sha256"):
        raise ValueError("Supplied bundle manifest hash differs")
    verify_inventory(supplied, bundle_spec["files_sha256"])
    if inventory(supplied) != bundle_spec["files_sha256"]:
        raise ValueError("Supplied bundle inventory is incomplete or contains extra files")
    return supplied


def _known_baseline_message(message: str) -> bool:
    if not message.startswith(_BASELINE_ERROR_PREFIX) or not message.endswith(_BASELINE_ERROR_SUFFIX):
        return False
    detail = message[len(_BASELINE_ERROR_PREFIX):-len(_BASELINE_ERROR_SUFFIX)]
    return bool(detail) and all(part.split("=", 1)[0] in METRICS for part in detail.split("; "))


def _classify_calibration_unavailable(error: Exception, root: Path, commit: str) -> dict | None:
    """Recognize only the pinned official baseline gate, never an arbitrary ValueError.

    The caller has authenticated the checkout with _upstream and catches only
    the build_real_bundle call. Requiring the innermost official source frame
    prevents input/configuration errors or a caller's similar text being hidden.
    """
    if commit != UPSTREAM_COMMIT or type(error) is not ValueError or not _known_baseline_message(str(error)):
        return None
    trace = error.__traceback__
    if trace is None:
        return None
    while trace.tb_next is not None:
        trace = trace.tb_next
    frame = trace.tb_frame
    if (frame.f_globals.get("__name__") != "cell_eval2.real_bundle"
            or frame.f_code.co_name != "build_real_bundle"
            or Path(frame.f_code.co_filename).resolve() != (root / "src/cell_eval2/real_bundle.py").resolve()):
        return None
    return {"available": False, "source": "build_rejected", "reason_code": BASELINE_UNAVAILABLE_CODE,
            "stage": "build_real_bundle.baseline_leg_gate", "upstream_error": str(error),
            "upstream_exception": "ValueError"}


def _unavailable_reason(prepared: dict) -> str:
    calibration = prepared.get("calibration", {})
    reason = calibration.get("upstream_error")
    if (calibration.get("available") is not False or calibration.get("source") != "build_rejected"
            or calibration.get("reason_code") != BASELINE_UNAVAILABLE_CODE
            or calibration.get("stage") != "build_real_bundle.baseline_leg_gate"
            or calibration.get("upstream_exception") != "ValueError"
            or not isinstance(reason, str) or not _known_baseline_message(reason)
            or prepared.get("reason_code") != BASELINE_UNAVAILABLE_CODE or prepared.get("reason") != reason):
        raise ValueError("Unavailable preparation lacks the bound official calibration rejection")
    return reason


def prepare(request: dict, directory: Path, output: Path, request_hash: str) -> dict:
    root, ce = _upstream(request, directory)
    from cell_eval2.real_bundle import build_real_bundle, check_submission
    from cell_eval2.baseline import build_run_meta
    import polars as pl
    context = _text(request.get("context_id"), "context_id")
    _text(request.get("panel_id"), "panel_id")
    real_source, real_path = _read_h5ad(request["reference"], directory, context)
    inputs = _pools(real_source, request, directory)
    real, pred_control, controls = _allocation(real_source, request)
    cfg = _configuration(request, ce)
    calibration = request["calibration"]
    if calibration.get("mode") not in {"build", "supplied"}:
        raise ValueError("calibration.mode must be build or supplied")
    scope = request.get("data_scope", "organizer_audit")
    if scope not in {"organizer_audit", "public_synthetic_demo", "public_data_demo",
                     "public_validation_audit", "synthetic_demo", "public_demo", "vcc2026_submission"}:
        raise ValueError("Unknown data_scope")
    baseline, baseline_binding = None, None
    if calibration["mode"] == "build":
        baseline, baseline_binding = _prediction(calibration["baseline_prediction"], directory, context,
                                                real, pred_control, cfg, baseline=True, data_scope=scope)
    else:
        supplied = _supplied_bundle(calibration, directory, controls)
    output.mkdir(parents=True, exist_ok=False)
    real.write_h5ad(output / "real.h5ad")
    pred_control.write_h5ad(output / "prediction_controls.h5ad")
    cfg.to_yaml(str(output / "config.yaml"))
    bundle = output / "real_bundle"
    parity = {"status": "unavailable", "reason": "No baseline prediction supplied for the zero-endpoint check"}
    evidence = ("original_shared_control" if controls["recipe"]["kind"] == "original"
                else "shared_control_depth_audit" if controls["reference_policy"] == "shared"
                else "separated_control_audit")
    record = {
        "schema": PREPARED_SCHEMA, "status": "COMPLETE", "run_id": request.get("run_id"),
        "context_id": context, "panel_id": request["panel_id"],
        "allocation_id": controls["recipe"]["allocation_id"], "request_sha256": request_hash,
        "reference": {"path": str(real_path), "sha256": request["reference"]["sha256"]},
        "controls": controls, "input_pools": inputs, "configuration": cfg.to_dict(),
        "upstream_path": str(root), "upstream_commit": UPSTREAM_COMMIT,
        "num_threads": cfg.num_threads, "data_scope": scope, "evidence_status": evidence,
        "competition_leaderboard_claim": False,
    }
    if calibration["mode"] == "build":
        baseline_path = output / "baseline_prediction.h5ad"
        baseline.write_h5ad(baseline_path)
        bundle_id = _text(calibration.get("bundle_id"), "bundle_id")
        try:
            manifest = build_real_bundle(str(output / "real.h5ad"), str(baseline_path), config=cfg,
                                         outdir=str(bundle), bundle_id=bundle_id, base_seed=0, n_splits=5)
        except ValueError as error:
            unavailable = _classify_calibration_unavailable(error, root, request["upstream_commit"])
            if unavailable is None:
                raise
            record.update(status=UNAVAILABLE_CALIBRATION, calibration=unavailable,
                          reason_code=unavailable["reason_code"], reason=unavailable["upstream_error"],
                          artifacts=inventory(output), official_calibration_available=False)
            write_json(output / "prepared.json", record)
            return record
        # Zero-endpoint check using the official aggregate just built. Metadata
        # is rebuilt from the actual files, but this is NOT an independent metric
        # recomputation; external QA must establish numerical replay parity.
        meta = build_run_meta(cfg, str(output / "real.h5ad"), str(baseline_path))
        table = ce.score_metrics(pl.read_csv(bundle / "baseline_agg.csv"), real_bundle=str(bundle),
                                 user_meta=meta, output=str(output / "baseline_score.csv"))
        _, average = _result_values(pl.read_csv(bundle / "baseline_agg.csv"), table)
        error = max(abs(float(row["from_replicate"])) for row in table.to_dicts()
                    if row["metric"] in METRICS)
        if error > 1e-12 or abs(average["from_replicate"]) > 1e-12:
            raise ValueError("Official baseline replay did not reproduce the zero endpoint")
        parity = {"status": "PASS", "check": "official_zero_endpoint_check",
                  "independently_recomputed": False, "max_abs_calibrated_score": error,
                  "aggregate": average["from_replicate"], "prediction": baseline_binding}
    else:
        shutil.copytree(supplied, bundle)
        manifest = read_json(bundle / "manifest.json")
        # A metadata-only full-reference probe validates the actual real-view and
        # configuration against the supplied bundle before publishing preparation.
        # It is not scored, and every model still gets its own submission check.
        probe = build_run_meta(cfg, str(output / "real.h5ad"), str(output / "real.h5ad"))
        check_submission(manifest, probe)
    from cell_eval2.anchor import read_anchor
    anchors, _, _ = read_anchor(str(bundle))
    anchor_rows = {row["metric"]: row for row in anchors.to_dicts()}
    baseline_rows = pl.read_csv(bundle / "baseline_agg.csv").to_dicts()
    baseline_mean = next(row for row in baseline_rows if row["statistic"] == "mean")
    endpoints = {}
    for metric in METRICS:
        base = _finite(baseline_mean.get(metric))
        replicate = _finite(anchor_rows.get(metric, {}).get("replicate"))
        endpoints[metric] = {"baseline": base, "replicate": replicate,
                             "span": replicate - base if base is not None and replicate is not None else None,
                             "replicate_sd": _finite(anchor_rows.get(metric, {}).get("replicate_sd"))}
    record.update({"calibration": {"bundle_id": manifest["real_bundle_id"],
                        "manifest_sha256": sha256(bundle / "manifest.json"),
                        "rule_digest": manifest.get("rule_digest"),
                        "rule_mismatches": manifest.get("rule_mismatches", []),
                        "source": "built" if calibration["mode"] == "build" else "supplied",
                        "metric_endpoints": endpoints,
                        "anchor_control_source": "per_half_observation_view",
                        "baseline_control_source": "prediction_ids" if calibration["mode"] == "build"
                                                   else "supplied_bundle_verified_at_score_time",
                        "baseline_parity": parity},
        "artifacts": inventory(output),
        "official_calibration_available": True,
    })
    write_json(output / "prepared.json", record)
    return record


def score(request: dict, directory: Path, output: Path, request_hash: str) -> dict:
    prepared_path = artifact(request["prepared"], directory)
    prepared = read_json(prepared_path)
    if prepared.get("schema") != PREPARED_SCHEMA or prepared.get("status") not in {"COMPLETE", UNAVAILABLE_CALIBRATION}:
        raise ValueError("A completed or explicitly unavailable vcc preparation is required")
    unavailable_reason = _unavailable_reason(prepared) if prepared["status"] == UNAVAILABLE_CALIBRATION else None
    verify_inventory(prepared_path.parent, prepared["artifacts"])
    root, ce = _upstream(prepared, prepared_path.parent)
    cfg = ce.EvalConfig.from_dict(prepared["configuration"])
    model = request["model"]
    model_id = _text(model.get("model_id"), "model_id")
    conditioning = model.get("conditioning", {})
    if not isinstance(conditioning, dict) or not isinstance(conditioning.get("uses_input_controls"), bool):
        raise ValueError("Model conditioning must explicitly declare boolean uses_input_controls")
    if conditioning.get("mode") not in {"fixed_submission", "input_intervention"}:
        raise ValueError("A model must declare fixed_submission or input_intervention")
    if conditioning["mode"] == "input_intervention" and not conditioning["uses_input_controls"]:
        raise ValueError("An input intervention requires declared use of the model-input controls")
    pools = {pool["input_pool_id"]: pool for pool in prepared["input_pools"]}
    if conditioning.get("input_pool_id") not in pools:
        raise ValueError("Prediction names an unverified model-input pool")
    import anndata as ad
    real = ad.read_h5ad(prepared_path.parent / "real.h5ad")
    pred_control = ad.read_h5ad(prepared_path.parent / "prediction_controls.h5ad")
    joined, binding = _prediction(model["prediction"], directory, prepared["context_id"],
                                  real, pred_control, cfg, data_scope=prepared["data_scope"])
    output.mkdir(parents=True, exist_ok=False)
    pred_path = output / "prediction_with_controls.h5ad"
    joined.write_h5ad(pred_path)
    cfg.to_yaml(str(output / "config.yaml"))
    result = {
        "schema": RESPONSE_SCHEMA, "status": "COMPLETE", "run_id": request.get("run_id"),
        "context_id": prepared["context_id"], "panel_id": prepared["panel_id"],
        "allocation_id": prepared["allocation_id"], "model_id": model_id,
        "request_sha256": request_hash, "prepared_sha256": request["prepared"]["sha256"],
        "controls": prepared["controls"],
        "calibration": prepared["calibration"], "prediction": binding,
        "conditioning": conditioning, "input_pool": pools[conditioning["input_pool_id"]],
        "model_input_role": "declared_used" if conditioning["uses_input_controls"] else "not_used",
        "input_pool_status": "declared_model_input" if conditioning["uses_input_controls"] else "available_pool",
        "evidence_status": prepared["evidence_status"], "data_scope": prepared["data_scope"],
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_path": str(root), "cell_eval2_version": ce.__version__,
        "control_source": cfg.control_source,
        "competition_leaderboard_claim": False,
    }
    if unavailable_reason is not None:
        result.update(status=UNAVAILABLE_CALIBRATION, reason_code=BASELINE_UNAVAILABLE_CODE,
                      reason=unavailable_reason, official_scoring_performed=False,
                      artifact_role="validated_scoring_inputs_only", validated_artifacts=inventory(output),
                      metrics=[{"metric": metric, "raw": None, "from_replicate": None, "valid_count": None,
                                "unavailable_reason": unavailable_reason} for metric in METRICS],
                      aggregate={"from_replicate": None, "unavailable_reason": unavailable_reason})
    else:
        aggregate, scores, meta, valid_counts = _score_table(ce, cfg, prepared_path.parent / "real.h5ad", pred_path,
                                                            prepared_path.parent / "real_bundle", output)
        metrics, calibrated = _result_values(aggregate, scores, valid_counts)
        result.update(metrics=metrics, aggregate=calibrated, environment=meta.get("environment"),
                      official_scoring_performed=True, official_artifacts=inventory(output))
    write_json(output / "response.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    request_path, output = args.request.resolve(), args.output.resolve()
    if output.exists():
        parser.error("Output path already exists; use a new path to preserve prior evidence")
    request = {}
    try:
        request = read_json(request_path)
        if request.get("schema") != REQUEST_SCHEMA:
            raise ValueError("Unknown worker request schema")
        _request_structure(request)
        operation = request.get("operation")
        if operation not in {"prepare", "score"}:
            raise ValueError("Worker operation must be prepare or score")
        request_hash = sha256(request_path)
        result = (prepare if operation == "prepare" else score)(request, request_path.parent, output, request_hash)
        print(json.dumps({"status": result["status"], "output": str(output)}))
        return 0
    except Exception as error:
        reason = f"{type(error).__name__}: {error}"
        # An incomplete run is explicit and never looks like a six-metric result.
        output.mkdir(parents=True, exist_ok=True)
        allocation = request.get("allocation") if isinstance(request.get("allocation"), dict) else {}
        model = request.get("model") if isinstance(request.get("model"), dict) else {}
        prepared_artifact = request.get("prepared") if isinstance(request.get("prepared"), dict) else {}
        identity = {"context_id": request.get("context_id"), "panel_id": request.get("panel_id"),
                    "allocation_id": allocation.get("allocation_id"),
                    "model_id": model.get("model_id"),
                    "prepared_sha256": prepared_artifact.get("sha256")}
        if request.get("operation") == "score":
            try:
                bound = read_json(artifact(request["prepared"], request_path.parent))
                identity.update({key: bound.get(key) for key in ("context_id", "panel_id", "allocation_id")})
            except Exception:
                pass  # Do not claim identities read from a changed preparation.
        failure = {"schema": RESPONSE_SCHEMA, "status": "FAILED", "run_id": request.get("run_id"),
                   **identity,
                   "request_sha256": sha256(request_path) if request_path.is_file() else None,
                   "reason": reason, "at": datetime.now(timezone.utc).isoformat(),
                   "metrics": [{"metric": metric, "raw": None, "from_replicate": None,
                                "valid_count": None, "unavailable_reason": reason} for metric in METRICS],
                   "aggregate": {"from_replicate": None, "unavailable_reason": reason},
                   "competition_leaderboard_claim": False}
        write_json(output / "response.json", failure)
        print(reason, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
