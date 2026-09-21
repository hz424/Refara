"""Pure contract tests; raw H5AD and official scorer integration use the worker suite."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from reference_design import vcc
from reference_design import vcc_cli


def _executable(path, body):
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o700)
    return str(path)


def test_failed_process_is_preserved_in_execution_receipt(manifest_case):
    path, _ = manifest_case
    executable = _executable(path.parent / "failed_python", "echo deliberate-failure >&2\nexit 7\n")
    output = path.parent / "failed_audit"
    with pytest.raises(vcc_cli.WorkerFailure, match="exit 7"):
        vcc_cli.run_audit(path, output, executable)
    receipt = vcc.read_json(output / "EXECUTION.json")
    assert receipt["status"] == "FAILED"
    assert len(receipt["jobs"]) == 1
    attempt = receipt["jobs"][0]
    assert attempt["returncode"] == 7
    assert attempt["started"] <= attempt["ended"]
    assert attempt["request_sha256_before"] == attempt["request_sha256_after"] == attempt["request_sha256"]
    assert "deliberate-failure" in Path(attempt["stderr"]).read_text()


@pytest.mark.parametrize("timing", ["before", "during"])
def test_worker_rejects_request_mutation(tmp_path, timing):
    request = tmp_path / "request.json"
    request.write_text("{}")
    job = {"request": str(request), "request_sha256": vcc.sha256_file(request),
           "output_dir": str(tmp_path / "worker_output")}
    if timing == "before":
        request.write_text('{"changed":true}')
        body = "exit 0\n"
    else:
        body = 'printf changed > "$3"\nexit 0\n'
    executable = _executable(tmp_path / "mutating_python", body)
    with pytest.raises(vcc_cli.WorkerFailure, match="request changed") as caught:
        vcc_cli._worker(job, executable, tmp_path, 0, "prepare", 1)
    receipt = caught.value.receipt
    assert receipt["status"] == "FAILED"
    assert receipt["returncode"] == (None if timing == "before" else 0)


@pytest.mark.parametrize("status", ["INCOMPLETE_WORKERS", "UNAVAILABLE_OFFICIAL_SCORES"])
def test_summarize_cli_does_not_return_success_for_unavailable_audit(monkeypatch, tmp_path, status):
    monkeypatch.setattr(vcc, "summarize", lambda *args: {"status": status})
    assert vcc_cli.main(["summarize", str(tmp_path)]) == 2


@pytest.mark.parametrize("status, exit_code", [("COMPLETE", 0), ("COMPLETE_WITH_UNAVAILABLE_ANALYSES", 2)])
def test_run_cli_distinguishes_finished_unavailable_audit(monkeypatch, tmp_path, status, exit_code):
    monkeypatch.setattr(vcc_cli, "run_audit", lambda *args: {"status": status})
    assert vcc_cli.main(["run", str(tmp_path / "manifest.json"), "--python", "official-python",
                         "--output", str(tmp_path / "audit")]) == exit_code


def _write_json(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def manifest_case(tmp_path):
    upstream = tmp_path / "official"
    upstream.mkdir()

    def artifact(name, content):
        path = tmp_path / name
        path.write_bytes(content)
        return {"path": name, "sha256": vcc.sha256_file(path)}

    reference = artifact("reference.h5ad", b"contract-fixture-reference-not-a-real-h5ad")
    inputs = artifact("input.h5ad", b"contract-fixture-public-input-not-a-real-h5ad")
    baseline = artifact("baseline.h5ad", b"contract-fixture-baseline-not-a-real-h5ad")
    models = []
    for name in ("A", "B", "C"):
        models.append({
            "model_id": name,
            "prediction": artifact(f"{name}.h5ad", f"contract-prediction-{name}".encode()),
            "conditioning": {
                "mode": "fixed_submission", "input_pool_id": "public", "uses_input_controls": True,
                "checkpoint_sha256": "a" * 64,
                "provenance": "Synthetic contract fixture; no fitted model is claimed.",
            },
        })
    context = {
        "context_id": "ctx", "panel_id": "all_targets", "reference": reference,
        "scoring_pool": {"dataset_id": "synthetic", "split": "synthetic_scoring",
                         "control_ids": ["score:1", "score:2", "score:3", "score:4"],
                         "provenance": "Synthetic held-out role fixture."},
        "input_pools": [{"input_pool_id": "public", "dataset_id": "synthetic",
                         "split": "public_input", "artifact": inputs,
                         "control_ids": ["input:1", "input:2"],
                         "provenance": "Synthetic model-input role fixture."}],
        "models": models,
        "unit_declaration": {"status": "NOT_ESTABLISHED", "unit_ids": [],
                             "basis": "Synthetic cells are not biological replicates.",
                             "authority": "Fixture author."},
        "allocations": [
            {"allocation_id": "original", "kind": "original", "reference_policy": "shared"},
            {"allocation_id": "split_0", "kind": "equal_depth", "reference_policy": "separated",
             "depth_per_stratum": 2, "seed": 0},
        ],
        "calibration": {"mode": "build", "baseline_prediction": baseline, "bundle_id": "baseline"},
    }
    manifest = {"schema": vcc.SCHEMA, "audit_id": "audit", "data_scope": "public_synthetic_demo",
                "upstream": {"path": "official", "commit": vcc.UPSTREAM_COMMIT},
                "contexts": [context]}
    path = tmp_path / "manifest.json"
    _write_json(path, manifest)
    return path, manifest


def test_plan_binds_complete_context_model_roster_without_claiming_cell_validation(manifest_case):
    path, original = manifest_case
    out = path.parent / "run"
    result = vcc.plan(path, out)
    assert result["status"] == "PLANNED_NOT_SCORED"
    assert result["preparation_count"] == 2
    assert result["score_count"] == 6
    assert result["actual_cell_membership_verification"] == "PENDING_WORKER"
    assert not result["experimental_unit_independence_machine_verified"]
    assert not result["model_generation_provenance_machine_verified"]
    assert json.loads((out / "MANIFEST_SNAPSHOT.json").read_text()) == original
    for preparation in result["preparations"]:
        request = vcc.read_json(preparation["request"])
        assert request["schema"] == vcc.WORKER_SCHEMA
        assert request["operation"] == "prepare"
        assert request["panel_id"] == "all_targets"
        assert request["scoring_control_ids"] == original["contexts"][0]["scoring_pool"]["control_ids"]
        assert {model["model_id"] for model in preparation["models"]} == {"A", "B", "C"}
        assert vcc.sha256_file(preparation["request"]) == preparation["request_sha256"]
        assert Path(request["reference"]["path"]).is_absolute()
    with pytest.raises(FileExistsError):
        vcc.plan(path, out)


def test_duplicate_json_fields_are_rejected_before_planning(manifest_case):
    path, _ = manifest_case
    path.write_text('{"schema":"a","schema":"b"}')
    with pytest.raises(ValueError, match="Duplicate JSON"):
        vcc.validate_manifest(path)


@pytest.mark.parametrize("mutator, message", [
    (lambda data: data.update(undocumented_rule=True), "unknown"),
    (lambda data: data["upstream"].update(commit="0" * 40), "Upstream commit"),
    (lambda data: data["contexts"][0]["reference"].update(sha256="0" * 64), "SHA256 mismatch"),
    (lambda data: data["contexts"][0]["input_pools"][0].update(control_ids=["score:1"]), "overlap"),
    (lambda data: data["contexts"][0]["input_pools"][0].update(split="held_out_scoring"), "input_pool.split"),
    (lambda data: data["contexts"][0]["allocations"][0].update(reference_policy="separated"), "original uses"),
    (lambda data: data["contexts"][0]["allocations"][1].update(depth_per_stratum=True), "integer"),
    (lambda data: data["contexts"][0]["allocations"][1].update(seed=-1), "integer"),
    (lambda data: data["contexts"][0]["allocations"].pop(0), "exactly one original"),
    (lambda data: data["contexts"][0]["models"][1].update(model_id="A"), "Duplicate model_id"),
    (lambda data: data["contexts"][0]["unit_declaration"].update(status="DECLARED_INDEPENDENT"), "nonempty"),
    (lambda data: data["contexts"][0]["models"][0]["conditioning"].update(mode="input_intervention"), "parent_model_id"),
    (lambda data: data["contexts"][0]["models"][0]["conditioning"].update(uses_input_controls="yes"), "explicit boolean"),
])
def test_invalid_designs_fail_before_worker_execution(manifest_case, mutator, message):
    path, data = manifest_case
    mutator(data)
    _write_json(path, data)
    with pytest.raises(ValueError, match=message):
        vcc.validate_manifest(path)
    assert not (path.parent / "run").exists()


def test_intervention_requires_separate_predictions_named_input_and_same_checkpoint(manifest_case):
    path, data = manifest_case
    context = data["contexts"][0]
    other = copy.deepcopy(context["input_pools"][0])
    other["input_pool_id"] = "public_other"
    other["control_ids"] = ["input:3"]
    context["input_pools"].append(other)
    model = context["models"][1]
    model["conditioning"].update(mode="input_intervention", parent_model_id="A", input_pool_id="public_other")
    _write_json(path, data)
    assert vcc.validate_manifest(path)["contexts"][0]["models"][1]["conditioning"]["parent_model_id"] == "A"
    model["conditioning"]["checkpoint_sha256"] = "b" * 64
    _write_json(path, data)
    with pytest.raises(ValueError, match="checkpoint"):
        vcc.validate_manifest(path)


def test_unused_model_input_role_is_explicit_and_cannot_be_intervention_parent(manifest_case):
    path, data = manifest_case
    context = data["contexts"][0]
    context["models"][0]["conditioning"]["uses_input_controls"] = False
    _write_json(path, data)
    assert not vcc.validate_manifest(path)["contexts"][0]["models"][0]["conditioning"]["uses_input_controls"]
    other = copy.deepcopy(context["input_pools"][0])
    other["input_pool_id"] = "other"
    other["control_ids"] = ["input:3"]
    context["input_pools"].append(other)
    context["models"][1]["conditioning"].update(mode="input_intervention", parent_model_id="A", input_pool_id="other")
    _write_json(path, data)
    with pytest.raises(ValueError, match="parent must use input controls"):
        vcc.validate_manifest(path)


def test_supplied_bundle_has_content_inventory_and_cannot_be_reused_for_changed_reference(manifest_case):
    path, data = manifest_case
    bundle = path.parent / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}")
    (bundle / "baseline_agg.csv").write_text("statistic,x\nmean,1\n")
    context = data["contexts"][0]
    context["calibration"] = {"mode": "supplied", "bundle": {
        "path": "bundle", "manifest_sha256": vcc.sha256_file(bundle / "manifest.json")}}
    _write_json(path, data)
    with pytest.raises(ValueError, match="one original reference"):
        vcc.validate_manifest(path)
    context["allocations"] = context["allocations"][:1]
    _write_json(path, data)
    validated = vcc.validate_manifest(path)
    files = validated["contexts"][0]["calibration"]["bundle"]["files_sha256"]
    assert set(files) == {"manifest.json", "baseline_agg.csv"}
    context["calibration"]["bundle"]["files_sha256"] = files
    (bundle / "baseline_agg.csv").write_text("statistic,x\nmean,2\n")
    _write_json(path, data)
    with pytest.raises(ValueError, match="inventory or hashes"):
        vcc.validate_manifest(path)


def test_score_planning_refuses_incomplete_prepare_and_modified_request(manifest_case):
    path, _ = manifest_case
    out = path.parent / "run"
    result = vcc.plan(path, out)
    with pytest.raises(ValueError, match="incomplete"):
        vcc.score_requests(out)
    Path(result["preparations"][0]["request"]).write_text("{}")
    with pytest.raises(ValueError, match="changed after planning"):
        vcc.score_requests(out)


def test_score_planning_refuses_model_substitution_inside_preparation_plan(manifest_case):
    path, _ = manifest_case
    out = path.parent / "run"
    vcc.plan(path, out)
    plan_path = out / "PLAN.json"
    plan = vcc.read_json(plan_path)
    plan["preparations"][0]["models"][0]["prediction"] = plan["preparations"][0]["models"][1]["prediction"]
    _write_json(plan_path, plan)
    with pytest.raises(ValueError, match="models or panel differ"):
        vcc.score_requests(out)


UNAVAILABLE_REASON = (
    "the baseline leg is degenerate on metric(s) that decide a ranking: "
    "de_wilcoxon_direction_reach_raw=1.0 (denominator 0.0 is not a finite positive number "
    "(anchor=1.0, direction='higher')). A bundle built on it could not be scored."
)


def _prepared_run(manifest_case, *, unavailable_allocations=()):
    """A protocol fixture; numerical correctness is reserved for official-worker tests."""
    path, _ = manifest_case
    out = path.parent / "run"
    plan = vcc.plan(path, out)
    for preparation in plan["preparations"]:
        directory = Path(preparation["output_dir"])
        directory.mkdir(parents=True)
        materialized = directory / "real.fixture"
        materialized.write_text("not an H5AD; fixture for bundle authentication")
        allocation = preparation["allocation_id"]
        prepare_request = vcc.read_json(preparation["request"])
        input_pools = copy.deepcopy(prepare_request["input_pools"])
        for pool in input_pools:
            pool.update(membership_verified=True,
                        membership_sha256=hashlib.sha256(json.dumps(pool["control_ids"]).encode()).hexdigest())
        receipt = {
            "schema": "reference_design.vcc_prepared.v1", "status": "COMPLETE", "run_id": "audit",
            "context_id": preparation["context_id"], "panel_id": preparation["panel_id"],
            "allocation_id": allocation, "request_sha256": preparation["request_sha256"],
            "controls": {"observation_ids": ["score:1", "score:2"],
                         "prediction_ids": ["score:1", "score:2"] if allocation == "original" else ["score:3", "score:4"],
                         "per_stratum_counts": {"observation": {"all": 2}, "prediction": {"all": 2}}},
            "calibration": {"bundle_id": f"fixture-{allocation}", "manifest_sha256": "b" * 64},
            "input_pools": input_pools,
            "evidence_status": "original_shared_control" if allocation == "original" else "separated_control_audit",
            "artifacts": {"real.fixture": vcc.sha256_file(materialized)},
        }
        if allocation in unavailable_allocations:
            receipt.update(status=vcc.UNAVAILABLE_CALIBRATION,
                           reason_code="OFFICIAL_BASELINE_SCALE_UNAVAILABLE", reason=UNAVAILABLE_REASON,
                           calibration={"available": False, "source": "build_rejected",
                                        "reason_code": "OFFICIAL_BASELINE_SCALE_UNAVAILABLE",
                                        "stage": "build_real_bundle.baseline_leg_gate",
                                        "upstream_error": UNAVAILABLE_REASON, "upstream_exception": "ValueError"})
        _write_json(directory / "prepared.json", receipt)
    jobs = vcc.score_requests(out)
    return out, jobs


def _official_tables(directory, response):
    """Use the official wide aggregate and long score schemas, without computation."""
    raw = directory / "aggregate.csv"
    with raw.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["statistic", *vcc.METRICS])
        writer.writeheader()
        writer.writerow({"statistic": "mean", **{row["metric"]: row["raw"] for row in response["metrics"]}})
    scaled = directory / "score.csv"
    with scaled.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "metric", "from_baseline", "from_replicate", "anchor_source", "anchor_digest",
            "real_bundle_id", "real_bundle_digest"])
        writer.writeheader()
        for row in [*response["metrics"], {"metric": "avg_score", **response["aggregate"]}]:
            writer.writerow({"metric": row["metric"], "from_baseline": 999.0,
                             "from_replicate": row["from_replicate"], "anchor_source": "real_bundle",
                             "anchor_digest": "a" * 64, "real_bundle_id": "fixture",
                             "real_bundle_digest": "b" * 64})
    response["official_artifacts"] = {"aggregate.csv": vcc.sha256_file(raw),
                                      "score.csv": vcc.sha256_file(scaled)}


def _response(job, values, *, failed=False):
    directory = Path(job["output_dir"])
    directory.mkdir(parents=True)
    request = vcc.read_json(job["request"])
    prepared = vcc.read_json(request["prepared"]["path"])
    response = {
        "schema": "reference_design.vcc_response.v1", "status": "FAILED" if failed else "COMPLETE",
        **{key: job[key] for key in ("context_id", "panel_id", "allocation_id", "model_id")},
        "request_sha256": job["request_sha256"], "prepared_sha256": request["prepared"]["sha256"],
        "controls": prepared["controls"], "calibration": prepared["calibration"],
        "official_artifacts": {},
        "metrics": [{"metric": metric, "raw": 7.0, "from_replicate": values[metric],
                     "valid_count": 3, "unavailable_reason": None if values[metric] is not None else "official metric unavailable"}
                    for metric in vcc.METRICS],
        "aggregate": {"from_replicate": values["avg_score"], "unavailable_reason": None},
    }
    if prepared["status"] == vcc.UNAVAILABLE_CALIBRATION and not failed:
        model = request["model"]
        reason = prepared["reason"]
        conditioning = model["conditioning"]
        used = conditioning["uses_input_controls"]
        response.update(status=vcc.UNAVAILABLE_CALIBRATION, reason=reason, reason_code=prepared["reason_code"],
                        official_scoring_performed=False, artifact_role="validated_scoring_inputs_only",
                        prediction=copy.deepcopy(model["prediction"]), conditioning=copy.deepcopy(conditioning),
                        input_pool=next(pool for pool in prepared["input_pools"]
                                        if pool["input_pool_id"] == conditioning["input_pool_id"]),
                        model_input_role="declared_used" if used else "not_used",
                        input_pool_status="declared_model_input" if used else "available_pool")
        response.pop("official_artifacts")
        response["metrics"] = [{"metric": metric, "raw": None, "from_replicate": None,
                                "valid_count": None, "unavailable_reason": reason} for metric in vcc.METRICS]
        response["aggregate"] = {"from_replicate": None, "unavailable_reason": reason}
        artifacts = {}
        for name in ("prediction_with_controls.h5ad", "config.yaml"):
            artifact_path = directory / name
            artifact_path.write_text("contract fixture for validated scoring input")
            artifacts[name] = vcc.sha256_file(artifact_path)
        response["validated_artifacts"] = artifacts
    else:
        _official_tables(directory, response)
    _write_json(directory / "response.json", response)
    return response


def _all_scores(out, jobs):
    for job in jobs:
        # A/B reverse, B/C tie initially, and aggregate is intentionally different
        # from the mean of metric scores: it must be preserved, never reconstructed.
        value = ({"A": 3.0, "B": 2.0, "C": 2.0} if job["allocation_id"] == "original"
                 else {"A": 1.0, "B": 2.0, "C": 0.0})[job["model_id"]]
        values = {metric: value for metric in vcc.METRICS}
        values["avg_score"] = value + 10.0
        _response(job, values)


def test_summary_preserves_official_aggregate_and_complete_pairs_with_ties(manifest_case):
    out, jobs = _prepared_run(manifest_case)
    _all_scores(out, jobs)
    result = vcc.summarize(out)
    assert result["status"] == "COMPLETE_DESCRIPTIVE_VCC_AUDIT"
    assert result["metric_rows"] == 2 * 3 * 7
    assert result["pair_rows"] == 2 * 3 * 7
    assert result["cross_context_aggregation"] == "NOT_PERFORMED"
    assert not result["inferential_tests_performed"]
    with (out / "summary" / "METRIC_SCORES.tsv").open() as handle:
        scores = list(csv.DictReader(handle, delimiter="\t"))
    assert next(float(row["from_replicate"]) for row in scores
                if row["metric"] == "avg_score" and row["model_id"] == "A" and row["allocation_id"] == "original") == 13.0
    with (out / "summary" / "COMPLETE_MODEL_PAIRS.tsv").open() as handle:
        pairs = list(csv.DictReader(handle, delimiter="\t"))
    assert all(row["classification"] == "ranking_reversal" for row in pairs
               if row["model_a"] == "A" and row["model_b"] == "B" and row["allocation_id"] == "split_0")
    assert all(row["classification"] == "tie_in_original" for row in pairs
               if row["model_a"] == "B" and row["model_b"] == "C" and row["allocation_id"] == "split_0")


def test_missing_worker_keeps_unavailable_pairs_and_never_closes_complete(manifest_case):
    out, jobs = _prepared_run(manifest_case)
    _all_scores(out, jobs[:-1])
    result = vcc.summarize(out)
    assert result["status"] == "INCOMPLETE_WORKERS"
    assert result["failed_or_missing_workers"] == 1
    assert result["pair_rows"] == 42
    with (out / "summary" / "COMPLETE_MODEL_PAIRS.tsv").open() as handle:
        pairs = list(csv.DictReader(handle, delimiter="\t"))
    unavailable = [row for row in pairs if row["classification"] == "unavailable"]
    assert len(unavailable) == 14
    assert all(row["ranking_reversal"] == "NA" for row in unavailable)


def test_failed_worker_retains_full_unavailable_roster(manifest_case):
    out, jobs = _prepared_run(manifest_case)
    _all_scores(out, jobs[:-1])
    values = {metric: None for metric in (*vcc.METRICS, "avg_score")}
    _response(jobs[-1], values, failed=True)
    result = vcc.summarize(out)
    assert result["status"] == "INCOMPLETE_WORKERS"
    assert result["pair_rows"] == 42


def test_finished_worker_with_unavailable_metric_does_not_close_a_complete_summary(manifest_case):
    out, jobs = _prepared_run(manifest_case)
    _all_scores(out, jobs)
    response_path = Path(jobs[-1]["output_dir"]) / "response.json"
    response = vcc.read_json(response_path)
    response["metrics"][0].update(from_replicate=None, unavailable_reason="Official calibration is unavailable")
    _official_tables(response_path.parent, response)
    _write_json(response_path, response)
    result = vcc.summarize(out)
    assert result["status"] == "UNAVAILABLE_OFFICIAL_SCORES"
    assert result["complete_workers"] == 6
    assert result["unavailable_score_rows"] == 1
    assert result["pair_rows"] == 42


def test_unavailable_calibration_retains_every_model_metric_and_pair(manifest_case):
    out, jobs = _prepared_run(manifest_case, unavailable_allocations={"split_0"})
    _all_scores(out, jobs)
    result = vcc.summarize(out)
    assert result["status"] == "UNAVAILABLE_OFFICIAL_SCORES"
    assert result["expected_workers"] == result["complete_workers"] == 6
    assert result["scored_workers"] == 3
    assert result["unavailable_calibration_workers"] == 3
    assert result["unavailable_calibration_preparations"] == 1
    assert result["failed_or_missing_workers"] == 0
    assert result["metric_rows"] == result["pair_rows"] == 42
    assert result["unavailable_score_rows"] == 21
    assert vcc.read_json(out / "summary" / "WORKER_FAILURES.json") == []
    with (out / "summary" / "METRIC_SCORES.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    unavailable = [row for row in rows if row["allocation_id"] == "split_0"]
    assert len(unavailable) == 21
    assert all(row["raw"] == row["from_replicate"] == "NA" for row in unavailable)
    assert all(row["worker_status"] == vcc.UNAVAILABLE_CALIBRATION
               and row["unavailable_reason_code"] == "OFFICIAL_BASELINE_SCALE_UNAVAILABLE"
               and row["unavailable_reason"] == UNAVAILABLE_REASON for row in unavailable)
    with (out / "summary" / "COMPLETE_MODEL_PAIRS.tsv").open() as handle:
        pairs = list(csv.DictReader(handle, delimiter="\t"))
    assert all(row["classification"] == "unavailable" and row["ranking_reversal"] == "NA"
               for row in pairs if row["allocation_id"] == "split_0")


@pytest.mark.parametrize("corruption", ["reason", "reason_code", "metric_raw", "metric_scaled", "aggregate",
                                          "prediction", "input_pool", "validated_artifact", "scored_status"])
def test_unavailable_calibration_cannot_hide_corruption_or_fabricate_scores(manifest_case, corruption):
    out, jobs = _prepared_run(manifest_case, unavailable_allocations={"split_0"})
    _all_scores(out, jobs)
    job = next(job for job in jobs if job["allocation_id"] == "split_0")
    response_path = Path(job["output_dir"]) / "response.json"
    response = vcc.read_json(response_path)
    if corruption == "reason":
        response["reason"] = "unrelated execution failure"
    elif corruption == "reason_code":
        response["reason_code"] = "INPUT_CORRUPTION"
    elif corruption == "metric_raw":
        response["metrics"][0]["raw"] = 1.0
    elif corruption == "metric_scaled":
        response["metrics"][0]["from_replicate"] = 1.0
    elif corruption == "aggregate":
        response["aggregate"]["from_replicate"] = 1.0
    elif corruption == "prediction":
        response["prediction"]["sha256"] = "0" * 64
    elif corruption == "input_pool":
        response["input_pool"]["control_ids"] = ["score:1"]
    elif corruption == "validated_artifact":
        (response_path.parent / "prediction_with_controls.h5ad").write_text("changed")
    else:
        response["status"] = "COMPLETE"
    _write_json(response_path, response)
    with pytest.raises(ValueError):
        vcc.summarize(out)
    assert not (out / "summary").exists()


def test_invalid_input_failure_cannot_be_declared_calibration_unavailability(manifest_case):
    out, jobs = _prepared_run(manifest_case, unavailable_allocations={"split_0"})
    plan = vcc.read_json(out / "PLAN.json")
    preparation = next(row for row in plan["preparations"] if row["allocation_id"] == "split_0")
    path = Path(preparation["output_dir"]) / "prepared.json"
    prepared = vcc.read_json(path)
    prepared["reason"] = prepared["calibration"]["upstream_error"] = "Input gene-axis mismatch"
    _write_json(path, prepared)
    with pytest.raises(ValueError, match="Unrecognized official calibration"):
        vcc.score_requests(out)


@pytest.mark.parametrize("outcome", ["missing", "failed"])
def test_unavailable_preparation_does_not_conceal_missing_or_failed_prediction_validation(manifest_case, outcome):
    out, jobs = _prepared_run(manifest_case, unavailable_allocations={"split_0"})
    _all_scores(out, jobs[:-1])
    if outcome == "failed":
        values = {metric: None for metric in (*vcc.METRICS, "avg_score")}
        _response(jobs[-1], values, failed=True)
    result = vcc.summarize(out)
    assert result["status"] == "INCOMPLETE_WORKERS"
    assert result["complete_workers"] == 5
    assert result["scored_workers"] == 3
    assert result["unavailable_calibration_workers"] == 2
    assert result["failed_or_missing_workers"] == 1
    assert result["metric_rows"] == result["pair_rows"] == 42


@pytest.mark.parametrize("field", ["raw", "from_replicate", "avg_score"])
def test_summary_rejects_json_score_changes_with_unchanged_official_csv(manifest_case, field):
    out, jobs = _prepared_run(manifest_case)
    _all_scores(out, jobs)
    response_path = Path(jobs[0]["output_dir"]) / "response.json"
    response = vcc.read_json(response_path)
    if field == "avg_score":
        response["aggregate"]["from_replicate"] += 0.5
    else:
        response["metrics"][0][field] += 0.5
    _write_json(response_path, response)
    with pytest.raises(ValueError, match="projection differs"):
        vcc.summarize(out)
    assert not (out / "summary").exists()


@pytest.mark.parametrize("corruption", [
    "raw_column", "scaled_metric", "average", "duplicate_metric", "duplicate_mean",
    "unlisted_score", "changed_value",
])
def test_summary_rejects_invalid_official_tables_even_with_updated_hashes(manifest_case, corruption):
    out, jobs = _prepared_run(manifest_case)
    _all_scores(out, jobs)
    response_path = Path(jobs[0]["output_dir"]) / "response.json"
    response = vcc.read_json(response_path)
    if corruption == "unlisted_score":
        del response["official_artifacts"]["score.csv"]
    else:
        name = "aggregate.csv" if corruption in ("raw_column", "duplicate_mean") else "score.csv"
        table_path = response_path.parent / name
        with table_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            columns, rows = list(reader.fieldnames), list(reader)
        if corruption == "raw_column":
            columns.remove(vcc.METRICS[0])
            for row in rows:
                row.pop(vcc.METRICS[0])
        elif corruption == "scaled_metric":
            rows = [row for row in rows if row["metric"] != vcc.METRICS[0]]
        elif corruption == "average":
            rows = [row for row in rows if row["metric"] != "avg_score"]
        elif corruption in ("duplicate_metric", "duplicate_mean"):
            rows.append(rows[0].copy())
        else:
            rows[0]["from_replicate"] = "123.0"
        with table_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        response["official_artifacts"][name] = vcc.sha256_file(table_path)
    _write_json(response_path, response)
    with pytest.raises(ValueError):
        vcc.summarize(out)
    assert not (out / "summary").exists()


@pytest.mark.parametrize("corruption", ["source", "identity", "request", "roster"])
def test_summary_refuses_corrupt_bindings_without_publishing(manifest_case, corruption):
    out, jobs = _prepared_run(manifest_case)
    _all_scores(out, jobs)
    response_path = Path(jobs[0]["output_dir"]) / "response.json"
    if corruption == "source":
        (response_path.parent / "aggregate.csv").write_text("changed official output")
    elif corruption == "identity":
        response = vcc.read_json(response_path)
        response["model_id"] = "OTHER"
        _write_json(response_path, response)
    elif corruption == "request":
        Path(jobs[0]["request"]).write_text("{}")
    else:
        score_plan_path = out / "requests" / "score" / "SCORE_PLAN.json"
        score_plan = vcc.read_json(score_plan_path)
        score_plan["jobs"].pop()
        _write_json(score_plan_path, score_plan)
    with pytest.raises(ValueError):
        vcc.summarize(out)
    assert not (out / "summary").exists()
