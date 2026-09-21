"""Organizer-side planning for the separately pinned official cell-eval2 worker.

This module reads declarations and authenticates files. The worker validates the
raw H5AD cells and performs all official numerical aggregation and calibration.
Repeated allocations are descriptive; no experimental units are inferred here.
"""
from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any

SCHEMA = "reference_design.vcc_audit.v1"
WORKER_SCHEMA = "reference_design.vcc_worker.v1"
UPSTREAM_COMMIT = "5e64833518a6603a0301cbe28185d49c30f4a986"
UNAVAILABLE_CALIBRATION = "UNAVAILABLE_CALIBRATION"
METRICS = (
    "pds_cosine", "expr_mse_unbiased_capped_norm",
    "de_wilcoxon_direction_fidelity_yield_raw",
    "de_wilcoxon_direction_reach_raw", "de_wilcoxon_sig_jaccard",
    "de_wilcoxon_lfc_nmae",
)
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA = re.compile(r"^[a-f0-9]{64}$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=_object, parse_constant=_constant)


def _fields(value: Any, required: set[str], optional: set[str], where: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    missing, extra = required - value.keys(), value.keys() - required - optional
    if missing or extra:
        raise ValueError(f"{where}: missing {sorted(missing)}; unknown {sorted(extra)}")


def _text(value: Any, where: str, identifier: bool = False) -> str:
    if not isinstance(value, str) or not value.strip() or any(c in value for c in "\r\n\t"):
        raise ValueError(f"{where} must be nonempty single-line text")
    if identifier and (not _SAFE.fullmatch(value) or value in (".", "..")):
        raise ValueError(f"{where} must be a safe identifier")
    return value


def _integer(value: Any, where: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{where} must be an integer >= {minimum}")
    return value


def _ids(value: Any, where: str, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{where} must be a {'possibly empty' if allow_empty else 'nonempty'} list")
    values = [_text(item, where) for item in value]
    if len(set(values)) != len(values):
        raise ValueError(f"{where} contains duplicate IDs")
    return values


def _artifact(value: Any, directory: Path, where: str, cache: dict[str, str]) -> dict[str, str]:
    _fields(value, {"path", "sha256"}, set(), where)
    path = (directory / _text(value["path"], f"{where}.path")).resolve(strict=True)
    expected = value["sha256"]
    if not isinstance(expected, str) or not _SHA.fullmatch(expected):
        raise ValueError(f"{where}.sha256 must be a lower-case SHA256")
    if not path.is_file():
        raise ValueError(f"{where} is not a regular file: {path}")
    actual = cache.get(str(path))
    if actual is None:
        actual = cache[str(path)] = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{where} SHA256 mismatch: {path}")
    return {"path": str(path), "sha256": expected}


def _bundle(value: Any, directory: Path) -> dict[str, Any]:
    _fields(value, {"path", "manifest_sha256"}, {"files_sha256"}, "calibration.bundle")
    path = (directory / _text(value["path"], "bundle.path")).resolve(strict=True)
    if not path.is_dir():
        raise ValueError("calibration bundle must be a directory")
    expected = value["manifest_sha256"]
    if not isinstance(expected, str) or not _SHA.fullmatch(expected):
        raise ValueError("bundle.manifest_sha256 must be a lower-case SHA256")
    files: dict[str, str] = {}
    for member in sorted(path.rglob("*")):
        if member.is_symlink():
            raise ValueError("calibration bundle must not contain symlinks")
        if member.is_file():
            files[str(member.relative_to(path))] = sha256_file(member)
    if files.get("manifest.json") != expected:
        raise ValueError("calibration bundle manifest SHA256 mismatch")
    if "files_sha256" in value and value["files_sha256"] != files:
        raise ValueError("calibration bundle file inventory or hashes differ")
    return {"path": str(path), "manifest_sha256": expected, "files_sha256": files}


def validate_manifest(path: str | Path) -> dict[str, Any]:
    """Validate declarations and file hashes, returning normalized absolute paths.

    Actual H5AD cell membership, raw-count validity and gene/panel equality are
    deliberately worker checks. Provenance and unit independence remain declared.
    """
    path = Path(path).resolve(strict=True)
    data = read_json(path)
    _fields(data, {"schema", "audit_id", "upstream", "contexts"},
            {"runtime", "description", "data_scope", "tie_tolerance"}, "manifest")
    if data["schema"] != SCHEMA:
        raise ValueError(f"Expected schema {SCHEMA}")
    _text(data["audit_id"], "audit_id", True)
    if "description" in data:
        _text(data["description"], "description")
    scope = data.setdefault("data_scope", "organizer_audit")
    if scope not in ("organizer_audit", "public_synthetic_demo", "public_data_demo", "public_validation_audit"):
        raise ValueError("Unknown data_scope")
    tolerance = data.setdefault("tie_tolerance", 1e-12)
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tie_tolerance must be finite and nonnegative")
    upstream = data["upstream"]
    _fields(upstream, {"path", "commit"}, set(), "upstream")
    if upstream["commit"] != UPSTREAM_COMMIT:
        raise ValueError("Upstream commit differs from the pinned official scorer")
    upstream_path = (path.parent / _text(upstream["path"], "upstream.path")).resolve(strict=True)
    if not upstream_path.is_dir():
        raise ValueError("upstream.path must be the official source checkout")
    upstream["path"] = str(upstream_path)
    runtime = data.setdefault("runtime", {})
    _fields(runtime, set(), {"de_backend", "device", "num_threads"}, "runtime")
    if runtime.setdefault("de_backend", "scanpy") not in ("scanpy", "pdex", "gpudge"):
        raise ValueError("runtime.de_backend must be explicit: scanpy, pdex or gpudge")
    if runtime.setdefault("device", "cpu") not in ("cpu", "cuda"):
        raise ValueError("runtime.device must be explicit: cpu or cuda")
    _integer(runtime.setdefault("num_threads", 1), "num_threads", 1)
    contexts = data["contexts"]
    if not isinstance(contexts, list) or not contexts:
        raise ValueError("contexts must be a nonempty list")
    context_ids: set[str] = set()
    cache: dict[str, str] = {}
    for context in contexts:
        _fields(context, {"context_id", "panel_id", "reference", "scoring_pool", "input_pools",
                          "models", "allocations", "calibration", "unit_declaration"},
                {"pert_col", "control_label", "stratum_col"}, "context")
        context_id = _text(context["context_id"], "context_id", True)
        if context_id in context_ids:
            raise ValueError("Duplicate context_id")
        context_ids.add(context_id)
        _text(context["panel_id"], "panel_id", True)
        _text(context.setdefault("pert_col", "target_gene"), "pert_col")
        _text(context.setdefault("control_label", "non-targeting"), "control_label")
        if context.setdefault("stratum_col", None) is not None:
            _text(context["stratum_col"], "stratum_col")
        context["reference"] = _artifact(context["reference"], path.parent, "reference", cache)
        scoring = context["scoring_pool"]
        _fields(scoring, {"dataset_id", "split", "control_ids", "provenance"}, set(), "scoring_pool")
        for key in ("dataset_id", "split", "provenance"):
            _text(scoring[key], f"scoring_pool.{key}")
        if scoring["split"] not in ("held_out_scoring", "public_validation", "synthetic_scoring"):
            raise ValueError("scoring_pool.split must identify held-out, public-validation or synthetic scoring")
        scoring_ids = set(_ids(scoring["control_ids"], "scoring_pool.control_ids"))
        units = context["unit_declaration"]
        _fields(units, {"status", "unit_ids", "basis", "authority"}, set(), "unit_declaration")
        if units["status"] not in ("DECLARED_INDEPENDENT", "NOT_ESTABLISHED"):
            raise ValueError("unit_declaration.status must be DECLARED_INDEPENDENT or NOT_ESTABLISHED")
        _ids(units["unit_ids"], "unit_declaration.unit_ids", allow_empty=units["status"] == "NOT_ESTABLISHED")
        _text(units["basis"], "unit_declaration.basis")
        _text(units["authority"], "unit_declaration.authority")
        pools = context["input_pools"]
        if not isinstance(pools, list) or not pools:
            raise ValueError("input_pools must be a nonempty list")
        pool_ids: set[str] = set()
        for pool in pools:
            _fields(pool, {"input_pool_id", "dataset_id", "split", "artifact", "control_ids", "provenance"}, set(), "input_pool")
            pool_id = _text(pool["input_pool_id"], "input_pool_id", True)
            if pool_id in pool_ids:
                raise ValueError("Duplicate input_pool_id")
            pool_ids.add(pool_id)
            for key in ("dataset_id", "provenance"):
                _text(pool[key], f"input_pool.{key}")
            if pool["split"] != "public_input":
                raise ValueError("input_pool.split must be public_input")
            input_ids = set(_ids(pool["control_ids"], "input_pool.control_ids"))
            if input_ids & scoring_ids:
                raise ValueError("Input and scoring control IDs overlap; use globally namespaced IDs")
            pool["artifact"] = _artifact(pool["artifact"], path.parent, "input_pool.artifact", cache)
            if pool["artifact"]["sha256"] == context["reference"]["sha256"]:
                raise ValueError("Public model-input pool must be distinct from the held-out scoring artifact")
        models = context["models"]
        if not isinstance(models, list) or len(models) < 2:
            raise ValueError("Every context needs at least two model configurations")
        model_ids: set[str] = set()
        for model in models:
            _fields(model, {"model_id", "prediction", "conditioning"}, set(), "model")
            model_id = _text(model["model_id"], "model_id", True)
            if model_id in model_ids:
                raise ValueError("Duplicate model_id in context")
            model_ids.add(model_id)
            model["prediction"] = _artifact(model["prediction"], path.parent, "model.prediction", cache)
            conditioning = model["conditioning"]
            _fields(conditioning, {"mode", "input_pool_id", "uses_input_controls", "checkpoint_sha256", "provenance"},
                    {"parent_model_id"}, "conditioning")
            if conditioning["mode"] not in ("fixed_submission", "input_intervention"):
                raise ValueError("conditioning.mode must identify fixed_submission or input_intervention")
            _text(conditioning["input_pool_id"], "conditioning.input_pool_id", True)
            if conditioning["input_pool_id"] not in pool_ids:
                raise ValueError("conditioning refers to an unknown input_pool_id")
            if not isinstance(conditioning["uses_input_controls"], bool):
                raise ValueError("conditioning.uses_input_controls must be an explicit boolean")
            if conditioning["mode"] == "input_intervention" and not conditioning["uses_input_controls"]:
                raise ValueError("input_intervention must use input controls")
            if not isinstance(conditioning["checkpoint_sha256"], str) or not _SHA.fullmatch(conditioning["checkpoint_sha256"]):
                raise ValueError("conditioning.checkpoint_sha256 must be declared explicitly")
            _text(conditioning["provenance"], "conditioning.provenance")
            if ("parent_model_id" in conditioning) != (conditioning["mode"] == "input_intervention"):
                raise ValueError("parent_model_id is required only for an input intervention")
            if "parent_model_id" in conditioning:
                _text(conditioning["parent_model_id"], "conditioning.parent_model_id", True)
        by_model = {model["model_id"]: model for model in models}
        by_pool = {pool["input_pool_id"]: pool for pool in pools}
        for model in models:
            conditioning = model["conditioning"]
            if conditioning["mode"] == "input_intervention":
                parent_id = conditioning["parent_model_id"]
                if parent_id not in by_model or parent_id == model["model_id"]:
                    raise ValueError("input_intervention needs a distinct known parent_model_id")
                parent = by_model[parent_id]
                if parent["conditioning"]["mode"] != "fixed_submission":
                    raise ValueError("input_intervention parent must be a fixed_submission")
                if not parent["conditioning"]["uses_input_controls"]:
                    raise ValueError("input_intervention parent must use input controls")
                if conditioning["checkpoint_sha256"] != parent["conditioning"]["checkpoint_sha256"]:
                    raise ValueError("input_intervention must retain its parent's declared checkpoint")
                if conditioning["input_pool_id"] == parent["conditioning"]["input_pool_id"]:
                    raise ValueError("input_intervention must name a distinct input pool")
                new_pool = by_pool[conditioning["input_pool_id"]]
                old_pool = by_pool[parent["conditioning"]["input_pool_id"]]
                if (new_pool["artifact"]["sha256"] == old_pool["artifact"]["sha256"]
                        and set(new_pool["control_ids"]) == set(old_pool["control_ids"])):
                    raise ValueError("input_intervention must actually change the named control inputs")
                if model["prediction"]["path"] == parent["prediction"]["path"]:
                    raise ValueError("input_intervention needs a separately supplied prediction file")
        allocations = context["allocations"]
        if not isinstance(allocations, list) or not allocations:
            raise ValueError("allocations must be a nonempty list")
        allocation_ids: set[str] = set()
        originals = 0
        for allocation in allocations:
            _fields(allocation, {"allocation_id", "kind", "reference_policy"},
                    {"depth_per_stratum", "seed"}, "allocation")
            allocation_id = _text(allocation["allocation_id"], "allocation_id", True)
            if allocation_id in allocation_ids:
                raise ValueError("Duplicate allocation_id in context")
            allocation_ids.add(allocation_id)
            if allocation["reference_policy"] not in ("shared", "separated"):
                raise ValueError("reference_policy must be shared or separated")
            if allocation["kind"] == "original":
                originals += 1
                if allocation["reference_policy"] != "shared" or set(allocation) != {"allocation_id", "kind", "reference_policy"}:
                    raise ValueError("original uses all declared scoring controls shared, without seed or depth")
            elif allocation["kind"] == "equal_depth":
                _integer(allocation.get("depth_per_stratum"), "depth_per_stratum", 2)
                _integer(allocation.get("seed"), "allocation.seed", 0)
            else:
                raise ValueError("allocation.kind must be original or equal_depth")
        if originals != 1:
            raise ValueError("Every context requires exactly one original allocation")
        calibration = context["calibration"]
        if not isinstance(calibration, dict):
            raise ValueError("calibration must be an object")
        if calibration.get("mode") == "build":
            _fields(calibration, {"mode", "baseline_prediction", "bundle_id"}, set(), "calibration")
            _text(calibration["bundle_id"], "bundle_id", True)
            calibration["baseline_prediction"] = _artifact(calibration["baseline_prediction"], path.parent, "baseline_prediction", cache)
        elif calibration.get("mode") == "supplied":
            _fields(calibration, {"mode", "bundle"}, set(), "calibration")
            if len(allocations) != 1:
                raise ValueError("A supplied bundle belongs to one original reference; use build mode for changed controls")
            calibration["bundle"] = _bundle(calibration["bundle"], path.parent)
        else:
            raise ValueError("calibration.mode must be build or supplied")
    return data


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _publish(output: Path, files: dict[str, bytes]) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output must not already exist: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        for name, payload in files.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Output appeared during preparation: {output}")
        staging.replace(output)


def plan(manifest_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Authenticate inputs and write prepare requests plus a closed model roster."""
    manifest_path = Path(manifest_path).resolve(strict=True)
    output = Path(output_dir).resolve()
    manifest = validate_manifest(manifest_path)
    files: dict[str, bytes] = {"MANIFEST_SNAPSHOT.json": manifest_path.read_bytes()}
    preparations: list[dict[str, Any]] = []
    for context in manifest["contexts"]:
        for allocation in context["allocations"]:
            key = f"{context['context_id']}/{allocation['allocation_id']}"
            request = dict(
                schema=WORKER_SCHEMA, operation="prepare", run_id=manifest["audit_id"],
                context_id=context["context_id"], panel_id=context["panel_id"],
                reference=context["reference"], scoring_control_ids=context["scoring_pool"]["control_ids"],
                input_pools=context["input_pools"], allocation=allocation, calibration=context["calibration"],
                upstream_path=manifest["upstream"]["path"], upstream_commit=manifest["upstream"]["commit"],
                pert_col=context["pert_col"], control_label=context["control_label"],
                stratum_col=context["stratum_col"], data_scope=manifest["data_scope"], **manifest["runtime"],
            )
            request_name = f"requests/prepare/{key}.json"
            payload = _json_bytes(request)
            files[request_name] = payload
            preparations.append(dict(
                context_id=context["context_id"], panel_id=context["panel_id"],
                allocation_id=allocation["allocation_id"],
                request=str(output / request_name), request_sha256=hashlib.sha256(payload).hexdigest(),
                output_dir=str(output / "prepared" / key), models=context["models"],
            ))
    receipt = dict(
        schema="reference_design.vcc_plan.v1", status="PLANNED_NOT_SCORED", audit_id=manifest["audit_id"],
        manifest_path=str(manifest_path), manifest_sha256=sha256_file(manifest_path), manifest=manifest,
        preparation_count=len(preparations), score_count=sum(len(row["models"]) for row in preparations),
        normalized_manifest_sha256=hashlib.sha256(_json_bytes(manifest)).hexdigest(),
        preparations=preparations, experimental_unit_independence_machine_verified=False,
        model_generation_provenance_machine_verified=False, actual_cell_membership_verification="PENDING_WORKER",
    )
    files["PLAN.json"] = _json_bytes(receipt)
    _publish(output, files)
    return receipt


def score_requests(run_dir: str | Path) -> list[dict[str, Any]]:
    """Bind finished preparation receipts and create one scoring request per model."""
    root = Path(run_dir).resolve(strict=True)
    plan_data = read_json(root / "PLAN.json")
    _check_plan(root, plan_data)
    jobs: list[dict[str, Any]] = []
    files: dict[str, bytes] = {}
    for preparation in plan_data["preparations"]:
        request_path = Path(preparation["request"])
        if sha256_file(request_path) != preparation["request_sha256"]:
            raise ValueError("Prepare request was changed after planning")
        prepared = Path(preparation["output_dir"]) / "prepared.json"
        if not prepared.is_file():
            raise ValueError(f"Preparation is incomplete: {prepared}")
        preparation_receipt = read_json(prepared)
        if (preparation_receipt.get("schema") != "reference_design.vcc_prepared.v1"
                or preparation_receipt.get("status") not in ("COMPLETE", UNAVAILABLE_CALIBRATION)):
            raise ValueError(f"Preparation has no validated terminal outcome: {prepared}")
        if preparation_receipt["status"] == UNAVAILABLE_CALIBRATION:
            _check_unavailable_calibration(preparation_receipt)
            _check_artifacts(prepared.parent, preparation_receipt.get("artifacts"), "unavailable preparation inputs")
        for key in ("context_id", "panel_id", "allocation_id"):
            if preparation_receipt.get(key) != preparation[key]:
                raise ValueError(f"Preparation {key} differs from the planned request")
        if preparation_receipt.get("request_sha256") != preparation["request_sha256"]:
            raise ValueError("Preparation is not bound to the planned request")
        for model in preparation["models"]:
            key = f"{preparation['context_id']}/{preparation['allocation_id']}/{model['model_id']}"
            request = dict(schema=WORKER_SCHEMA, operation="score", run_id=plan_data["audit_id"],
                           prepared={"path": str(prepared), "sha256": sha256_file(prepared)}, model=model)
            payload = _json_bytes(request)
            name = key + ".json"
            files[name] = payload
            jobs.append(dict(context_id=preparation["context_id"], panel_id=preparation["panel_id"],
                             allocation_id=preparation["allocation_id"], model_id=model["model_id"],
                             request=str(root / "requests" / "score" / name),
                             request_sha256=hashlib.sha256(payload).hexdigest(),
                             output_dir=str(root / "scores" / key)))
    files["SCORE_PLAN.json"] = _json_bytes(dict(schema="reference_design.vcc_score_plan.v1", jobs=jobs,
                                              plan_sha256=sha256_file(root / "PLAN.json")))
    _publish(root / "requests" / "score", files)
    return jobs


def _check_plan(root: Path, data: dict[str, Any]) -> None:
    if data.get("schema") != "reference_design.vcc_plan.v1":
        raise ValueError("Invalid plan schema")
    if sha256_file(root / "MANIFEST_SNAPSHOT.json") != data["manifest_sha256"]:
        raise ValueError("Manifest snapshot differs from its planned hash")
    if hashlib.sha256(_json_bytes(data["manifest"])).hexdigest() != data["normalized_manifest_sha256"]:
        raise ValueError("Normalized manifest changed after planning")
    contexts = {context["context_id"]: context for context in data["manifest"]["contexts"]}
    expected = {(context["context_id"], allocation["allocation_id"])
                for context in contexts.values() for allocation in context["allocations"]}
    observed = [(row["context_id"], row["allocation_id"]) for row in data["preparations"]]
    if len(set(observed)) != len(observed) or set(observed) != expected:
        raise ValueError("Preparation plan differs from the declared allocation roster")
    for preparation in data["preparations"]:
        context = contexts[preparation["context_id"]]
        if preparation["models"] != context["models"] or preparation["panel_id"] != context["panel_id"]:
            raise ValueError("Preparation models or panel differ from the bound manifest")


def _table(columns: list[str], rows: list[dict[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: "NA" if row.get(key) is None else row.get(key) for key in columns})
    return buffer.getvalue().encode()


def _number(value: Any, where: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{where} must be finite or null")
    return float(value)


def _sign(value: float | None, tolerance: float) -> str:
    if value is None:
        return "unavailable"
    return "tie" if abs(value) <= tolerance else "A_higher" if value > 0 else "B_higher"


def _check_artifacts(directory: Path, mapping: Any, where: str) -> None:
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"{where} must contain artifact hashes")
    for name, expected in mapping.items():
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"Unsafe relative artifact path in {where}")
        member = directory / name
        if not member.is_file() or member.is_symlink() or sha256_file(member) != expected:
            raise ValueError(f"{where} artifact hash mismatch: {name}")


def _check_unavailable_calibration(prepared: dict[str, Any]) -> str:
    """Authenticate the narrow, declared official baseline-calibration refusal."""
    calibration = prepared.get("calibration")
    _fields(calibration, {"available", "source", "reason_code", "stage", "upstream_error",
                          "upstream_exception"}, set(), "unavailable calibration")
    reason = _text(calibration["upstream_error"], "upstream calibration refusal")
    if (calibration["available"] is not False
            or calibration["source"] != "build_rejected"
            or calibration["reason_code"] != "OFFICIAL_BASELINE_SCALE_UNAVAILABLE"
            or calibration["stage"] != "build_real_bundle.baseline_leg_gate"
            or calibration["upstream_exception"] != "ValueError"
            or not reason.startswith("the baseline leg is degenerate on metric(s) that decide a ranking: ")
            or not reason.endswith(". A bundle built on it could not be scored.")):
        raise ValueError("Unrecognized official calibration-unavailability classification")
    if prepared.get("reason_code") != calibration["reason_code"] or prepared.get("reason") != reason:
        raise ValueError("Preparation calibration reason binding differs")
    return reason


def _check_unavailable_response(directory: Path, response: dict[str, Any], prepared: dict[str, Any],
                                request: dict[str, Any], context: dict[str, Any],
                                records: dict[str, dict[str, Any]]) -> None:
    reason = _check_unavailable_calibration(prepared)
    if (response.get("reason_code") != prepared["reason_code"] or response.get("reason") != reason
            or response.get("official_scoring_performed") is not False
            or response.get("artifact_role") != "validated_scoring_inputs_only"
            or "official_artifacts" in response):
        raise ValueError("Unavailable response reason or scoring-stage binding differs")
    artifacts = response.get("validated_artifacts")
    _check_artifacts(directory, artifacts, "validated unavailable-scoring inputs")
    if not {"prediction_with_controls.h5ad", "config.yaml"}.issubset(artifacts):
        raise ValueError("Unavailable response must retain validated prediction and configuration artifacts")
    for metric, record in records.items():
        if ("from_replicate" not in record or (metric in METRICS and "raw" not in record)
                or record.get("raw") is not None or record.get("from_replicate") is not None
                or record.get("valid_count") is not None or record.get("unavailable_reason") != reason):
            raise ValueError(f"Unavailable calibration must retain null scores and its bound reason: {metric}")
    model = request["model"]
    prediction = response.get("prediction")
    if (not isinstance(prediction, dict) or prediction.get("sha256") != model["prediction"]["sha256"]
            or not isinstance(prediction.get("path"), str)
            or Path(prediction["path"]).resolve() != Path(model["prediction"]["path"]).resolve()
            or sha256_file(model["prediction"]["path"]) != model["prediction"]["sha256"]):
        raise ValueError("Unavailable response prediction binding differs from the frozen model")
    conditioning = model["conditioning"]
    if response.get("conditioning") != conditioning:
        raise ValueError("Unavailable response conditioning differs from the frozen model")
    pools = prepared.get("input_pools")
    if not isinstance(pools, list) or any(not isinstance(pool, dict) for pool in pools):
        raise ValueError("Unavailable preparation must retain verified input pools")
    matches = [pool for pool in pools if pool.get("input_pool_id") == conditioning["input_pool_id"]]
    if len(matches) != 1 or response.get("input_pool") != matches[0]:
        raise ValueError("Unavailable response input-pool binding differs from its preparation")
    pool = matches[0]
    declared = next(pool for pool in context["input_pools"]
                    if pool["input_pool_id"] == conditioning["input_pool_id"])
    if (pool.get("membership_verified") is not True
            or any(pool.get(key) != value for key, value in declared.items())
            or pool.get("membership_sha256") != hashlib.sha256(json.dumps(pool["control_ids"]).encode()).hexdigest()
            or sha256_file(declared["artifact"]["path"]) != declared["artifact"]["sha256"]):
        raise ValueError("Unavailable input-pool membership differs from its declared source")
    used = conditioning["uses_input_controls"]
    if (response.get("model_input_role") != ("declared_used" if used else "not_used")
            or response.get("input_pool_status") != ("declared_model_input" if used else "available_pool")):
        raise ValueError("Unavailable response misstates use of the model-input controls")


def _official_rows(directory: Path, name: str, key: str, columns: set[str],
                   expected_sha256: str) -> dict[str, dict[str, str]]:
    """Parse the authenticated CSV bytes without evaluating any scoring formula."""
    payload = (directory / name).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"Official table hash mismatch: {name}")
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8"), newline=""))
    header = reader.fieldnames
    if not header or len(set(header)) != len(header) or not columns.issubset(header):
        raise ValueError(f"Official {name} has missing or duplicate columns")
    rows: dict[str, dict[str, str]] = {}
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Official {name} has a malformed row")
        identity = row[key]
        if not identity or identity in rows:
            raise ValueError(f"Official {name} has missing or duplicate {key} rows")
        rows[identity] = row
    return rows


def _official_number(value: str, where: str) -> float | None:
    # Polars writes null as an empty field; the worker maps non-finite values
    # to null before serializing its response. Preserve that exact projection.
    if not value.strip():
        return None
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(f"Official {where} is not numeric or empty") from error
    return number if math.isfinite(number) else None


def _check_official_projection(directory: Path, response: dict[str, Any],
                               records: dict[str, dict[str, Any]]) -> None:
    """Require JSON scores to be the official CSV projection, including nulls."""
    artifacts = response["official_artifacts"]
    if not {"aggregate.csv", "score.csv"}.issubset(artifacts):
        raise ValueError("Official artifact inventory must include aggregate.csv and score.csv")
    raw_rows = _official_rows(directory, "aggregate.csv", "statistic",
                              {"statistic", *METRICS}, artifacts["aggregate.csv"])
    score_rows = _official_rows(directory, "score.csv", "metric",
                                {"metric", "from_replicate"}, artifacts["score.csv"])
    if "mean" not in raw_rows:
        raise ValueError("Official aggregate.csv is missing its mean row")
    if not {*METRICS, "avg_score"}.issubset(score_rows):
        raise ValueError("Official score.csv must retain all six metrics and avg_score")
    for metric in (*METRICS, "avg_score"):
        record = records[metric]
        if "from_replicate" not in record:
            raise ValueError(f"Worker response is missing {metric}.from_replicate")
        scaled = _number(record["from_replicate"], f"{metric}.from_replicate")
        official_scaled = _official_number(score_rows[metric]["from_replicate"],
                                           f"score.csv {metric}.from_replicate")
        if scaled != official_scaled:
            raise ValueError(f"Worker score projection differs from official score.csv: {metric}")
        if metric != "avg_score":
            if "raw" not in record:
                raise ValueError(f"Worker response is missing {metric}.raw")
            raw = _number(record["raw"], f"{metric}.raw")
            official_raw = _official_number(raw_rows["mean"][metric], f"aggregate.csv mean.{metric}")
            if raw != official_raw:
                raise ValueError(f"Worker raw projection differs from official aggregate.csv: {metric}")


def summarize(run_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Preserve official context/panel scores and compare every model pair.

    No per-perturbation averaging, cross-context weighting, calibration or new
    hypothesis testing is implemented here. Missing/failed workers produce a
    visibly INCOMPLETE summary, with every expected comparison retained as NA.
    A validated official calibration refusal is a terminal unavailable result;
    it retains the full roster without turning the refusal into an execution failure.
    """
    root = Path(run_dir).resolve(strict=True)
    output = root / "summary" if output_dir is None else Path(output_dir).resolve()
    plan_data = read_json(root / "PLAN.json")
    _check_plan(root, plan_data)
    score_plan = read_json(root / "requests" / "score" / "SCORE_PLAN.json")
    if (score_plan.get("schema") != "reference_design.vcc_score_plan.v1"
            or score_plan.get("plan_sha256") != sha256_file(root / "PLAN.json")):
        raise ValueError("Score plan is not bound to this plan")
    manifest = plan_data["manifest"]
    expected_jobs = {
        (context["context_id"], allocation["allocation_id"], model["model_id"])
        for context in manifest["contexts"] for allocation in context["allocations"]
        for model in context["models"]
    }
    jobs = score_plan["jobs"]
    observed_jobs = [(job["context_id"], job["allocation_id"], job["model_id"]) for job in jobs]
    if len(set(observed_jobs)) != len(observed_jobs) or set(observed_jobs) != expected_jobs:
        raise ValueError("Score plan does not cover the complete declared model roster")
    context_map = {context["context_id"]: context for context in manifest["contexts"]}
    rows: list[dict[str, Any]] = []
    values: dict[tuple[str, str, str, str], float | None] = {}
    response_bindings: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    controls_by_allocation: dict[tuple[str, str], dict[str, Any]] = {}
    calibration_by_allocation: dict[tuple[str, str], dict[str, Any]] = {}
    checked_preparations: set[str] = set()
    unavailable_preparations: set[str] = set()
    unavailable_workers = scored_workers = 0
    metrics_with_aggregate = (*METRICS, "avg_score")
    for job in jobs:
        if sha256_file(job["request"]) != job["request_sha256"]:
            raise ValueError("Score request was changed after planning")
        request = read_json(job["request"])
        if (request.get("operation") != "score" or request["model"]["model_id"] != job["model_id"]):
            raise ValueError("Score request model or operation differs")
        declared_model = next(model for model in context_map[job["context_id"]]["models"]
                              if model["model_id"] == job["model_id"])
        if request["model"] != declared_model:
            raise ValueError("Score request prediction or conditioning differs from the bound manifest")
        prepared_path = Path(request["prepared"]["path"])
        if sha256_file(prepared_path) != request["prepared"]["sha256"]:
            raise ValueError("Prepared receipt changed after score planning")
        prepared = read_json(prepared_path)
        if (prepared.get("schema") != "reference_design.vcc_prepared.v1"
                or prepared.get("status") not in ("COMPLETE", UNAVAILABLE_CALIBRATION)):
            raise ValueError("Score request references an incomplete preparation")
        if prepared["status"] == UNAVAILABLE_CALIBRATION:
            _check_unavailable_calibration(prepared)
            unavailable_preparations.add(str(prepared_path))
        if str(prepared_path) not in checked_preparations:
            _check_artifacts(prepared_path.parent, prepared.get("artifacts"), "prepared scoring inputs")
            checked_preparations.add(str(prepared_path))
        for key in ("context_id", "panel_id", "allocation_id"):
            if prepared.get(key) != job[key]:
                raise ValueError("Prepared identity differs from score plan")
        key = (job["context_id"], job["allocation_id"])
        for mapping, source_key in ((controls_by_allocation, "controls"),
                                    (calibration_by_allocation, "calibration")):
            value = prepared[source_key]
            if key in mapping and mapping[key] != value:
                raise ValueError(f"Models in one allocation use different {source_key}")
            mapping[key] = value
        response_path = Path(job["output_dir"]) / "response.json"
        response = read_json(response_path) if response_path.is_file() else None
        failure_reason = None
        if response is None:
            failure_reason = "worker_response_missing"
        else:
            if response.get("schema") != "reference_design.vcc_response.v1":
                raise ValueError("Unexpected worker response schema")
            for field in ("context_id", "panel_id", "allocation_id", "model_id"):
                if response.get(field) != job[field]:
                    raise ValueError(f"Worker response {field} differs from score plan")
            if (response.get("request_sha256") != job["request_sha256"]
                    or response.get("prepared_sha256") != request["prepared"]["sha256"]):
                raise ValueError("Worker response is not bound to its request and prepared receipt")
            if response.get("status") not in ("COMPLETE", "FAILED", UNAVAILABLE_CALIBRATION):
                raise ValueError("Unexpected worker response status")
            if response["status"] == "FAILED":
                failure_reason = "worker_failed"
            else:
                if response["status"] != prepared["status"]:
                    raise ValueError("Worker scoring outcome differs from its preparation availability")
                if response["status"] == "COMPLETE":
                    _check_artifacts(Path(job["output_dir"]), response.get("official_artifacts"), "official scorer")
                if response.get("controls") != prepared["controls"] or response.get("calibration") != prepared["calibration"]:
                    raise ValueError("Worker response changed control membership or calibration")
            response_bindings.append(dict(path=str(response_path), sha256=sha256_file(response_path), status=response["status"]))
        if failure_reason:
            failures.append(dict(context_id=job["context_id"], allocation_id=job["allocation_id"],
                                 model_id=job["model_id"], reason=failure_reason))
            records = {name: {"raw": None, "from_replicate": None, "valid_count": None,
                              "unavailable_reason": failure_reason} for name in metrics_with_aggregate}
        else:
            metric_rows = response["metrics"]
            if (not isinstance(metric_rows, list) or len(metric_rows) != len(METRICS)
                    or any(not isinstance(record, dict) for record in metric_rows)
                    or {record.get("metric") for record in metric_rows} != set(METRICS)):
                raise ValueError("Worker response must retain all six official metrics, including nulls")
            records = {record["metric"]: record for record in metric_rows}
            aggregate = response["aggregate"]
            if not isinstance(aggregate, dict):
                raise ValueError("Worker aggregate must be an object")
            records["avg_score"] = {"raw": None, "valid_count": None, **aggregate}
            if response["status"] == UNAVAILABLE_CALIBRATION:
                _check_unavailable_response(Path(job["output_dir"]), response, prepared, request,
                                             context_map[job["context_id"]], records)
                unavailable_workers += 1
            else:
                _check_official_projection(Path(job["output_dir"]), response, records)
                scored_workers += 1
        context = context_map[job["context_id"]]
        model = next(model for model in context["models"] if model["model_id"] == job["model_id"])
        allocation = next(a for a in context["allocations"] if a["allocation_id"] == job["allocation_id"])
        for metric in metrics_with_aggregate:
            record = records[metric]
            scaled = _number(record.get("from_replicate"), f"{metric}.from_replicate")
            raw = _number(record.get("raw"), f"{metric}.raw")
            valid_count = record.get("valid_count")
            if valid_count is not None:
                _integer(valid_count, "valid_count", 0)
            reason = record.get("unavailable_reason")
            available = scaled is not None and (metric == "avg_score" or raw is not None)
            if not available and (not isinstance(reason, str) or not reason):
                raise ValueError(f"Unavailable official score needs a reason: {metric}")
            if available and reason:
                raise ValueError(f"Available official score has an unavailable reason: {metric}")
            values[(job["context_id"], job["allocation_id"], job["model_id"], metric)] = scaled if available else None
            rows.append(dict(
                context_id=job["context_id"], panel_id=job["panel_id"], allocation_id=job["allocation_id"],
                allocation_kind=allocation["kind"], reference_policy=allocation["reference_policy"],
                evidence_status=prepared["evidence_status"], model_id=job["model_id"], metric=metric,
                raw=raw, from_replicate=scaled, valid_count=valid_count, available=available,
                unavailable_reason=reason or None,
                unavailable_reason_code=(response.get("reason_code") if response and not failure_reason else None),
                worker_status=response["status"] if response else "MISSING",
                conditioning_mode=model["conditioning"]["mode"], input_pool_id=model["conditioning"]["input_pool_id"],
                uses_input_controls=model["conditioning"]["uses_input_controls"],
                model_input_role="declared_used" if model["conditioning"]["uses_input_controls"] else "not_used",
                parent_model_id=model["conditioning"].get("parent_model_id"),
                checkpoint_sha256=model["conditioning"]["checkpoint_sha256"],
            ))
    pairs: list[dict[str, Any]] = []
    tolerance = manifest["tie_tolerance"]
    for context in manifest["contexts"]:
        original_id = next(a["allocation_id"] for a in context["allocations"] if a["kind"] == "original")
        models = {model["model_id"]: model for model in context["models"]}
        for allocation in context["allocations"]:
            for a, b in itertools.combinations(sorted(models), 2):
                for metric in metrics_with_aggregate:
                    current_a = values[(context["context_id"], allocation["allocation_id"], a, metric)]
                    current_b = values[(context["context_id"], allocation["allocation_id"], b, metric)]
                    original_a = values[(context["context_id"], original_id, a, metric)]
                    original_b = values[(context["context_id"], original_id, b, metric)]
                    difference = None if current_a is None or current_b is None else current_a - current_b
                    original_difference = None if original_a is None or original_b is None else original_a - original_b
                    direction, original_direction = _sign(difference, tolerance), _sign(original_difference, tolerance)
                    reversal = None if "unavailable" in (direction, original_direction) else (
                        direction in ("A_higher", "B_higher") and original_direction in ("A_higher", "B_higher")
                        and direction != original_direction)
                    if reversal is None:
                        classification = "unavailable"
                    elif "tie" in (direction, original_direction):
                        classification = "tie_in_both" if direction == original_direction else "tie_in_original" if original_direction == "tie" else "tie_in_allocation"
                    else:
                        classification = "ranking_reversal" if reversal else "same_direction"
                    conditioning_pair = (models[a]["conditioning"].get("parent_model_id") == b
                                         or models[b]["conditioning"].get("parent_model_id") == a)
                    pairs.append(dict(
                        context_id=context["context_id"], panel_id=context["panel_id"],
                        allocation_id=allocation["allocation_id"], original_allocation_id=original_id,
                        metric=metric, model_a=a, model_b=b, value_a=current_a, value_b=current_b,
                        difference_A_minus_B=difference, original_difference_A_minus_B=original_difference,
                        direction=direction, original_direction=original_direction,
                        ranking_reversal=reversal, classification=classification,
                        comparison_kind="conditioning_intervention" if conditioning_pair else "model_configuration_comparison",
                    ))
    files = {
        "METRIC_SCORES.tsv": _table(list(rows[0]), rows),
        "COMPLETE_MODEL_PAIRS.tsv": _table(list(pairs[0]), pairs),
        "WORKER_FAILURES.json": _json_bytes(failures),
        "CONTROL_MEMBERSHIP.json": _json_bytes([
            dict(context_id=key[0], allocation_id=key[1], **value)
            for key, value in sorted(controls_by_allocation.items())]),
        "CALIBRATION_BINDINGS.json": _json_bytes([
            dict(context_id=key[0], allocation_id=key[1], **value)
            for key, value in sorted(calibration_by_allocation.items())]),
        "UNIT_DECLARATIONS.json": _json_bytes([
            dict(context_id=context["context_id"], **context["unit_declaration"], machine_verified=False)
            for context in manifest["contexts"]]),
    }
    unavailable_rows = sum(not row["available"] for row in rows)
    receipt = dict(
        schema="reference_design.vcc_summary.v1",
        status=("INCOMPLETE_WORKERS" if failures else "UNAVAILABLE_OFFICIAL_SCORES" if unavailable_rows
                else "COMPLETE_DESCRIPTIVE_VCC_AUDIT"),
        audit_id=manifest["audit_id"], data_scope=manifest["data_scope"],
        plan_sha256=sha256_file(root / "PLAN.json"), score_plan_sha256=sha256_file(root / "requests" / "score" / "SCORE_PLAN.json"),
        expected_workers=len(jobs), complete_workers=len(jobs) - len(failures), failed_or_missing_workers=len(failures),
        scored_workers=scored_workers, unavailable_calibration_workers=unavailable_workers,
        unavailable_calibration_preparations=len(unavailable_preparations),
        context_count=len(manifest["contexts"]), metric_rows=len(rows), pair_rows=len(pairs),
        unavailable_score_rows=unavailable_rows,
        comparison_scale="official_from_replicate_higher_is_better", tie_tolerance=tolerance,
        cross_context_aggregation="NOT_PERFORMED", task_metric_averaging="NOT_PERFORMED",
        experimental_unit_independence_machine_verified=False, inferential_tests_performed=False,
        raw_cell_checks_performed_by="separate_official_worker", responses=response_bindings,
        artifacts={name: hashlib.sha256(payload).hexdigest() for name, payload in files.items()},
    )
    files["SUMMARY.json"] = _json_bytes(receipt)
    _publish(output, files)
    return receipt
