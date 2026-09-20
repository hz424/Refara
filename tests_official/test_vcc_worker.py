"""Actual H5AD validation tests for the separate official-scoring worker.

Run explicitly with the cell-eval2 Python 3.12 interpreter inside the scheduled QA job::

    /path/to/official-python3.12 -m pytest -q tests_official/test_vcc_worker.py

This directory is intentionally outside the package's default Python 3.10
``testpaths = ["tests", "analysis"]``. Run it as a separate required QA command;
the default package suite alone does not establish official-worker validation.
Set ``REFERENCE_DESIGN_OFFICIAL_SOURCE`` to the pinned cell-eval2 checkout when
it is not the sibling ``upstream_cell_eval2`` directory beside this package.

These exercise the worker's real data and file checks without mocking AnnData,
cell membership, or validators. They do not build calibration anchors and do
not establish six-metric replay parity; that requires the official end-to-end
QA. Availability-gate tests supply a small aggregate to the actual official
gate, without replacing that gate or the worker classifier. There is deliberately
no importorskip: missing scorer data dependencies
must be reported as an unexecuted/failing check, not a passing skipped suite.
"""
from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse


WORKER_PATH = Path(__file__).resolve().parents[1] / "src/reference_design/vcc_worker.py"
SPEC = importlib.util.spec_from_file_location("reference_design_worker_data_tests", WORKER_PATH)
assert SPEC is not None and SPEC.loader is not None
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)

CONTEXT = "context_a"
CONTROL = "non-targeting"
CFG = SimpleNamespace(pert_col="target_gene", control=CONTROL)
GENES = ["gene_a", "gene_b", "gene_c"]


def cells(labels, prefix, *, names=None, genes=None, sparse_counts=False):
    """Small raw-count cells with explicit identities and two scoring strata."""
    n = len(labels)
    values = np.asarray([[i + 1, 2 + i % 3, 4 + i % 2] for i in range(n)], dtype=np.float64)
    names = names or [f"{prefix}:{i}" for i in range(n)]
    obs = pd.DataFrame({"target_gene": labels, "context": [CONTEXT] * n,
                        "stratum": ["s1" if i % 2 == 0 else "s2" for i in range(n)]}, index=names)
    return ad.AnnData(sparse.csr_matrix(values) if sparse_counts else values, obs=obs,
                      var=pd.DataFrame(index=genes or GENES))


def save(data, path):
    data.write_h5ad(path)
    return {"path": str(path), "sha256": worker.sha256(path)}


@pytest.fixture
def raw_case(tmp_path):
    reference = cells(["target_a"] * 3 + ["target_b"] * 3 + [CONTROL] * 8, "reference")
    inputs = cells([CONTROL] * 4, "public_input", sparse_counts=True)
    prediction = cells(["target_a"] * 2 + ["target_b"] * 2, "model_output")
    reference_spec = save(reference, tmp_path / "reference.h5ad")
    prediction_spec = save(prediction, tmp_path / "prediction.h5ad")
    input_spec = save(inputs, tmp_path / "input.h5ad")
    request = {"context_id": CONTEXT, "pert_col": CFG.pert_col, "control_label": CONTROL,
               "stratum_col": "stratum", "reference": reference_spec,
               "scoring_control_ids": reference.obs_names[6:].tolist(),
               "allocation": {"allocation_id": "original", "kind": "original",
                              "reference_policy": "shared"},
               "input_pools": [{"input_pool_id": "public_a", "dataset_id": "synthetic_raw_fixture",
                                "split": "public_input", "artifact": input_spec,
                                "control_ids": inputs.obs_names.tolist(), "provenance": "test fixture"}]}
    return SimpleNamespace(directory=tmp_path, reference=reference, inputs=inputs, prediction=prediction,
                           prediction_spec=prediction_spec, request=request)


def test_legal_raw_h5ad_panel_preserves_cells_and_control_values(raw_case):
    case = raw_case
    reference, _ = worker._read_h5ad(case.request["reference"], case.directory, CONTEXT)
    pools = worker._pools(reference, case.request, case.directory)
    real, pred_controls, membership = worker._allocation(reference, case.request)
    joined, binding = worker._prediction(case.prediction_spec, case.directory, CONTEXT,
                                       real, pred_controls, CFG)
    assert pools[0]["membership_verified"] is True
    assert membership["observation_ids"] == membership["prediction_ids"]
    assert list(real.obs_names) == list(reference.obs_names)
    assert binding["cells_per_perturbation"] == {"target_a": 2, "target_b": 2}
    assert joined.n_obs == case.prediction.n_obs + pred_controls.n_obs
    np.testing.assert_array_equal(joined[pred_controls.obs_names].X, pred_controls.X)


def test_public_input_and_heldout_scoring_control_overlap_is_rejected(raw_case):
    case = raw_case
    overlapping = case.reference[case.request["scoring_control_ids"][:4]].copy()
    request = copy.deepcopy(case.request)
    request["input_pools"][0]["artifact"] = save(overlapping, case.directory / "overlapping_input.h5ad")
    request["input_pools"][0]["control_ids"] = overlapping.obs_names.tolist()
    with pytest.raises(ValueError, match="model-input and held-out scoring controls overlap"):
        worker._pools(case.reference, request, case.directory)


@pytest.mark.parametrize("missing_id", ["missing_cell", "reference:0"])
def test_scoring_pool_requires_actual_control_cell_ids(raw_case, missing_id):
    request = copy.deepcopy(raw_case.request)
    request["scoring_control_ids"][0] = missing_id
    with pytest.raises(ValueError, match="missing or non-control cells"):
        worker._allocation(raw_case.reference, request)


def test_declared_input_control_must_exist_and_be_control(raw_case):
    request = copy.deepcopy(raw_case.request)
    request["input_pools"][0]["control_ids"][0] = "not_in_input"
    with pytest.raises(ValueError, match="missing or non-control IDs"):
        worker._pools(raw_case.reference, request, raw_case.directory)


def test_prediction_missing_a_panel_target_is_rejected(raw_case):
    case = raw_case
    subset = case.prediction[case.prediction.obs["target_gene"] == "target_a"].copy()
    spec = save(subset, case.directory / "missing_target.h5ad")
    real, pred_controls, _ = worker._allocation(case.reference, case.request)
    with pytest.raises(ValueError, match="entire context panel"):
        worker._prediction(spec, case.directory, CONTEXT, real, pred_controls, CFG)


@pytest.mark.parametrize("invalid,match", [(-1.0, "nonnegative raw counts"),
                                         (1.5, "integral raw counts"),
                                         (np.nan, "finite nonnegative"),
                                         (np.inf, "finite nonnegative")])
@pytest.mark.parametrize("sparse_counts", [False, True])
def test_illegal_raw_values_are_rejected_from_actual_h5ad(raw_case, invalid, match, sparse_counts):
    data = cells(["target_a", "target_b"], "bad_counts", sparse_counts=sparse_counts)
    data.X[0, 0] = invalid
    spec = save(data, raw_case.directory / "invalid_counts.h5ad")
    with pytest.raises(ValueError, match=match):
        worker._read_h5ad(spec, raw_case.directory, CONTEXT)


def test_per_cell_count_ceiling_is_checked(raw_case):
    data = raw_case.prediction.copy()
    data.X[0] = [400_000, 400_000, 400_000]
    spec = save(data, raw_case.directory / "over_ceiling.h5ad")
    with pytest.raises(ValueError, match="cell exceeds the official"):
        worker._read_h5ad(spec, raw_case.directory, CONTEXT)


def test_prediction_gene_set_mismatch_is_rejected(raw_case):
    case = raw_case
    prediction = case.prediction.copy()
    prediction.var_names = ["gene_a", "gene_b", "wrong_gene"]
    spec = save(prediction, case.directory / "wrong_genes.h5ad")
    real, pred_controls, _ = worker._allocation(case.reference, case.request)
    with pytest.raises(ValueError, match="gene set differs"):
        worker._prediction(spec, case.directory, CONTEXT, real, pred_controls, CFG)


def test_same_genes_different_order_are_reordered_without_changing_values(raw_case):
    case = raw_case
    prediction = case.prediction[:, list(reversed(GENES))].copy()
    spec = save(prediction, case.directory / "reordered_genes.h5ad")
    real, pred_controls, _ = worker._allocation(case.reference, case.request)
    joined, _ = worker._prediction(spec, case.directory, CONTEXT, real, pred_controls, CFG)
    assert list(joined.var_names) == GENES
    np.testing.assert_array_equal(joined.X[:case.prediction.n_obs], case.prediction.X)


def test_model_submission_cannot_inject_its_own_scoring_controls(raw_case):
    case = raw_case
    payload = cells(["target_a", "target_b", CONTROL], "injected_prediction")
    spec = save(payload, case.directory / "injected_controls.h5ad")
    real, pred_controls, _ = worker._allocation(case.reference, case.request)
    with pytest.raises(ValueError, match="must not contain non-targeting rows"):
        worker._prediction(spec, case.directory, CONTEXT, real, pred_controls, CFG)


def test_separated_baseline_uses_assigned_prediction_controls(raw_case):
    case = raw_case
    request = copy.deepcopy(case.request)
    request["allocation"] = {"allocation_id": "separated", "kind": "equal_depth",
                             "reference_policy": "separated", "depth_per_stratum": 2, "seed": 91}
    real, pred_controls, membership = worker._allocation(case.reference, request)
    baseline_payload = cells(["target_a", "target_b", CONTROL], "baseline")
    baseline_payload.X[0, 0] = 1.5  # Official generic baseline may contain fractional raw-count means.
    baseline_payload.X[-1] = [999, 999, 999]
    spec = save(baseline_payload, case.directory / "baseline.h5ad")
    joined, binding = worker._prediction(spec, case.directory, CONTEXT, real, pred_controls,
                                       CFG, baseline=True)
    assert binding["baseline_control_rows_replaced"] == 1
    assert set(membership["observation_ids"]).isdisjoint(membership["prediction_ids"])
    actual_control_ids = joined.obs_names[joined.obs[CFG.pert_col] == CONTROL].tolist()
    assert actual_control_ids == membership["prediction_ids"]
    np.testing.assert_array_equal(joined[pred_controls.obs_names].X, pred_controls.X)
    assert joined.X[0, 0] == 1.5


def test_shared_and_separated_recipes_have_identical_observation_block(raw_case):
    case = raw_case
    request = copy.deepcopy(case.request)
    request["allocation"] = {"allocation_id": "shared", "kind": "equal_depth",
                             "reference_policy": "shared", "depth_per_stratum": 2, "seed": 91}
    _, _, shared = worker._allocation(case.reference, request)
    request["allocation"].update(allocation_id="separated", reference_policy="separated")
    _, _, separated = worker._allocation(case.reference, request)
    assert shared["observation_ids"] == separated["observation_ids"]
    assert shared["prediction_ids"] == shared["observation_ids"]
    assert set(separated["prediction_ids"]).isdisjoint(separated["observation_ids"])
    assert separated["per_stratum_counts"] == {
        "observation": {"s1": 2, "s2": 2}, "prediction": {"s1": 2, "s2": 2}}


def test_equal_depth_fails_when_two_disjoint_blocks_are_unavailable(raw_case):
    request = copy.deepcopy(raw_case.request)
    request["allocation"] = {"allocation_id": "too_deep", "kind": "equal_depth",
                             "reference_policy": "shared", "depth_per_stratum": 3, "seed": 0}
    with pytest.raises(ValueError, match="cannot supply two depth-3 blocks"):
        worker._allocation(raw_case.reference, request)


def bundle_fixture(tmp_path):
    """File-identity fixture, explicitly not a valid six-metric real bundle."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text('{"fixture": "identity checks only"}\n', encoding="utf-8")
    (bundle / "baseline_agg.csv").write_text("statistic,fixture\nmean,1\n", encoding="utf-8")
    calibration = {"mode": "supplied", "bundle": {
        "path": str(bundle), "manifest_sha256": worker.sha256(bundle / "manifest.json"),
        "files_sha256": worker.inventory(bundle)}}
    controls = {"recipe": {"kind": "original"}, "reference_policy": "shared"}
    return bundle, calibration, controls


def test_supplied_bundle_identity_inventory_accepts_exact_files(tmp_path):
    bundle, calibration, controls = bundle_fixture(tmp_path)
    assert worker._supplied_bundle(calibration, tmp_path, controls) == bundle


def test_supplied_bundle_manifest_hash_mismatch_is_rejected(tmp_path):
    _, calibration, controls = bundle_fixture(tmp_path)
    calibration["bundle"]["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="manifest hash differs"):
        worker._supplied_bundle(calibration, tmp_path, controls)


def test_supplied_bundle_changed_file_is_rejected(tmp_path):
    bundle, calibration, controls = bundle_fixture(tmp_path)
    (bundle / "baseline_agg.csv").write_text("statistic,fixture\nmean,9\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Prepared artifact changed"):
        worker._supplied_bundle(calibration, tmp_path, controls)


def test_supplied_bundle_incomplete_inventory_is_rejected(tmp_path):
    _, calibration, controls = bundle_fixture(tmp_path)
    del calibration["bundle"]["files_sha256"]["baseline_agg.csv"]
    with pytest.raises(ValueError, match="inventory is incomplete"):
        worker._supplied_bundle(calibration, tmp_path, controls)


def test_supplied_bundle_disallows_changed_reference_policy(tmp_path):
    _, calibration, controls = bundle_fixture(tmp_path)
    controls.update(recipe={"kind": "equal_depth"}, reference_policy="separated")
    with pytest.raises(ValueError, match="only accepted for the original shared-control view"):
        worker._supplied_bundle(calibration, tmp_path, controls)


def test_raw_artifact_hash_mismatch_is_rejected_before_loading(raw_case):
    spec = dict(raw_case.prediction_spec, sha256="0" * 64)
    with pytest.raises(ValueError, match="SHA-256 differs"):
        worker._read_h5ad(spec, raw_case.directory, CONTEXT)


def test_no_wrong_context_fallback(raw_case):
    with pytest.raises(ValueError, match="no cells for context"):
        worker._read_h5ad(raw_case.prediction_spec, raw_case.directory, "not_this_context")


@pytest.fixture
def official_api():
    """Use the actual pinned official module for availability-gate tests.

    Override REFERENCE_DESIGN_OFFICIAL_SOURCE when testing an installed release
    outside this workspace. Absence is an error, never a skipped validation.
    """
    root = Path(os.environ.get("REFERENCE_DESIGN_OFFICIAL_SOURCE",
                               str(WORKER_PATH.parents[3] / "upstream_cell_eval2"))).resolve()
    verified, ce = worker._upstream({"upstream_path": str(root),
                                    "upstream_commit": worker.UPSTREAM_COMMIT, "num_threads": 1}, root)
    return verified, ce


def _saturated_baseline_leg(monkeypatch):
    """Supply an aggregate to the REAL official gate; do not mock the gate itself."""
    import cell_eval2.real_bundle as rb
    import polars as pl
    row = {"statistic": "mean", **{metric: 0.5 for metric in worker.METRICS}}
    row["de_wilcoxon_direction_reach_raw"] = 1.0
    monkeypatch.setattr(rb, "_baseline_leg", lambda *args, **kwargs: (pl.DataFrame([row]), {}))
    return rb


def test_only_actual_official_baseline_rejection_is_classified(official_api, monkeypatch, tmp_path):
    root, ce = official_api
    rb = _saturated_baseline_leg(monkeypatch)
    with pytest.raises(ValueError) as caught:
        rb.build_real_bundle("not-read", "not-read", config=ce.EvalConfig.from_preset("vcc2026"),
                             outdir=str(tmp_path / "bundle"), bundle_id="gate-test")
    result = worker._classify_calibration_unavailable(caught.value, root, worker.UPSTREAM_COMMIT)
    assert result["reason_code"] == worker.BASELINE_UNAVAILABLE_CODE
    assert result["available"] is False
    assert "direction_reach_raw=1.0" in result["upstream_error"]
    assert worker._classify_calibration_unavailable(caught.value, root, "0" * 40) is None
    assert worker._classify_calibration_unavailable(caught.value, tmp_path / "other_checkout",
                                                   worker.UPSTREAM_COMMIT) is None


def test_matching_text_from_nonofficial_frame_is_not_unavailable(tmp_path):
    message = (worker._BASELINE_ERROR_PREFIX + "de_wilcoxon_direction_reach_raw=1.0 (fixture)"
               + worker._BASELINE_ERROR_SUFFIX)
    try:
        raise ValueError(message)
    except ValueError as error:
        assert worker._classify_calibration_unavailable(error, tmp_path, worker.UPSTREAM_COMMIT) is None
    assert worker._classify_calibration_unavailable(ValueError(message), tmp_path, worker.UPSTREAM_COMMIT) is None
    assert worker._classify_calibration_unavailable(RuntimeError(message), tmp_path, worker.UPSTREAM_COMMIT) is None


def test_official_configuration_valueerror_stays_failure(official_api, tmp_path):
    root, ce = official_api
    import cell_eval2.real_bundle as rb
    destination = tmp_path / "not_a_directory"
    destination.write_text("input/configuration errors must remain failures")
    with pytest.raises(ValueError, match="not a directory") as caught:
        rb.build_real_bundle("not-read", "not-read", config=ce.EvalConfig.from_preset("vcc2026"),
                             outdir=str(destination), bundle_id="gate-test")
    assert worker._classify_calibration_unavailable(caught.value, root, worker.UPSTREAM_COMMIT) is None


@pytest.fixture
def unavailable_case(raw_case, official_api, monkeypatch):
    root, ce = official_api
    _saturated_baseline_leg(monkeypatch)
    request = copy.deepcopy(raw_case.request)
    request.update(schema=worker.REQUEST_SCHEMA, operation="prepare", panel_id="entire-panel", run_id="qa",
                   upstream_path=str(root), upstream_commit=worker.UPSTREAM_COMMIT,
                   num_threads=1, data_scope="public_synthetic_demo",
                   calibration={"mode": "build", "bundle_id": "qa-bundle",
                                "baseline_prediction": raw_case.prediction_spec})
    request_path = raw_case.directory / "prepare_request.json"
    worker.write_json(request_path, request)
    output = raw_case.directory / "prepared"
    assert worker.main(["--request", str(request_path), "--output", str(output)]) == 0
    prepared = worker.read_json(output / "prepared.json")
    assert prepared["status"] == worker.UNAVAILABLE_CALIBRATION
    assert prepared["request_sha256"] == worker.sha256(request_path)
    assert prepared["controls"]["observation_ids"] == raw_case.request["scoring_control_ids"]
    assert {"real.h5ad", "prediction_controls.h5ad", "baseline_prediction.h5ad", "config.yaml"} <= set(prepared["artifacts"])
    worker.verify_inventory(output, prepared["artifacts"])
    assert not (output / "real_bundle/manifest.json").exists()
    score_request = {"schema": worker.REQUEST_SCHEMA, "operation": "score", "run_id": "qa",
                     "prepared": {"path": str(output / "prepared.json"),
                                  "sha256": worker.sha256(output / "prepared.json")},
                     "model": {"model_id": "test_model", "prediction": raw_case.prediction_spec,
                               "conditioning": {"mode": "fixed_submission", "uses_input_controls": True,
                                                "input_pool_id": "public_a", "checkpoint_sha256": "a" * 64,
                                                "provenance": "Raw-cell availability test"}}}
    return raw_case, prepared, score_request


def test_unavailable_calibration_retains_validated_real_prediction_and_all_na(unavailable_case, monkeypatch):
    case, prepared, request = unavailable_case

    def no_metric_computation(*args, **kwargs):
        raise AssertionError("An unavailable calibration must not call the model metric scorer")

    monkeypatch.setattr(worker, "_score_table", no_metric_computation)
    path = case.directory / "score_request.json"
    worker.write_json(path, request)
    output = case.directory / "scored"
    assert worker.main(["--request", str(path), "--output", str(output)]) == 0
    response = worker.read_json(output / "response.json")
    assert response["status"] == worker.UNAVAILABLE_CALIBRATION
    assert response["official_scoring_performed"] is False
    assert response["artifact_role"] == "validated_scoring_inputs_only"
    assert "official_artifacts" not in response
    assert response["prediction"]["sha256"] == request["model"]["prediction"]["sha256"]
    assert response["calibration"] == prepared["calibration"] and response["controls"] == prepared["controls"]
    assert len(response["metrics"]) == 6
    for row in response["metrics"]:
        assert row["raw"] is row["from_replicate"] is row["valid_count"] is None
        assert row["unavailable_reason"] == prepared["calibration"]["upstream_error"]
    assert response["aggregate"] == {"from_replicate": None, "unavailable_reason": response["reason"]}
    assert {"prediction_with_controls.h5ad", "config.yaml"} <= set(response["validated_artifacts"])
    worker.verify_inventory(output, response["validated_artifacts"])
    joined = ad.read_h5ad(output / "prediction_with_controls.h5ad")
    pred_controls = ad.read_h5ad(Path(request["prepared"]["path"]).parent / "prediction_controls.h5ad")
    np.testing.assert_array_equal(joined[pred_controls.obs_names].X, pred_controls.X)


@pytest.mark.parametrize("problem", ["missing_target", "wrong_genes", "fractional_count", "injected_controls",
                                     "missing_conditioning_declaration"])
def test_unavailable_calibration_still_rejects_invalid_prediction_inputs(unavailable_case, problem):
    case, prepared, request = unavailable_case
    bad = case.prediction.copy()
    if problem == "missing_target":
        bad = bad[bad.obs[CFG.pert_col] == "target_a"].copy()
    elif problem == "wrong_genes":
        bad.var_names = ["gene_a", "gene_b", "unexpected_gene"]
    elif problem == "fractional_count":
        bad.X[0, 0] = 0.5
    elif problem == "injected_controls":
        bad = cells(["target_a", "target_b", CONTROL], "injected")
    else:
        del request["model"]["conditioning"]["uses_input_controls"]
    request["model"]["prediction"] = save(bad, case.directory / "invalid_prediction.h5ad")
    path = case.directory / "score_request.json"
    worker.write_json(path, request)
    output = case.directory / "invalid_score"
    assert worker.main(["--request", str(path), "--output", str(output)]) == 1
    response = worker.read_json(output / "response.json")
    assert response["status"] == "FAILED"
    assert response["model_id"] == "test_model"
    assert response["prepared_sha256"] == request["prepared"]["sha256"]
    assert "validated_artifacts" not in response


def test_unknown_official_baseline_error_is_not_converted(raw_case, official_api, monkeypatch):
    root, ce = official_api
    import cell_eval2.real_bundle as rb

    def bad_input(*args, **kwargs):
        raise ValueError("source content or configuration mismatch")

    monkeypatch.setattr(rb, "_baseline_leg", bad_input)
    request = copy.deepcopy(raw_case.request)
    request.update(schema=worker.REQUEST_SCHEMA, operation="prepare", panel_id="entire-panel", run_id="qa",
                   upstream_path=str(root), upstream_commit=worker.UPSTREAM_COMMIT,
                   num_threads=1, data_scope="public_synthetic_demo",
                   calibration={"mode": "build", "bundle_id": "qa-bundle",
                                "baseline_prediction": raw_case.prediction_spec})
    path = raw_case.directory / "unknown_failure_request.json"
    worker.write_json(path, request)
    output = raw_case.directory / "unknown_failure"
    assert worker.main(["--request", str(path), "--output", str(output)]) == 1
    assert worker.read_json(output / "response.json")["status"] == "FAILED"
    assert not (output / "prepared.json").exists()
