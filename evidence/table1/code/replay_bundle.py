"""Reproduce bundled role summaries and example actions without original caches.

The bundle is read only. A new output directory outside it receives reproduced
tables, action arrays, logs and the replay receipt. NumPy is the only dependency
outside the Python standard library.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
from typing import Any

import numpy as np

BUNDLE = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "BUNDLE_MANIFEST.json"
METRIC_TABLES = ("pattern_means.tsv", "role_contrasts.tsv", "role_interactions.tsv", "workflow_comparison.tsv")
ACTION_TEXT = ("action_summary.tsv", "actions.json", "injected_stress_tests.json", "role_omission_checks.json", "report.md", "cases.json")
# BLAS kernels can change Pearson norm reductions by a few float64 ULPs.
# This absolute bound is below the operation engine's 1e-10 decision tolerance.
PEARSON_ATOL = 1e-14
PEARSON_FIELD = "perturbation_centred_Pearson"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_file(root: Path, name: Any) -> Path:
    require(isinstance(name, str) and bool(name), "Manifest path must be a nonempty string")
    parts = PurePosixPath(name)
    require(not parts.is_absolute() and ".." not in parts.parts and parts.as_posix() == name,
            f"Noncanonical relative path: {name!r}")
    require("\\" not in name and "\n" not in name and "\r" not in name, f"Unsupported path: {name!r}")
    path = root.joinpath(*parts.parts)
    require(path.resolve().is_relative_to(root.resolve()), f"Path escapes its root: {name!r}")
    return path


def verify_file(root: Path, name: str, entry: dict[str, Any]) -> None:
    path = relative_file(root, name)
    require(path.is_file() and not path.is_symlink(), f"Missing or linked file: {name}")
    require(type(entry.get("bytes")) is int and entry["bytes"] >= 0, f"Invalid byte count: {name}")
    digest = entry.get("sha256")
    require(isinstance(digest, str) and len(digest) == 64 and all(char in "0123456789abcdef" for char in digest), f"Invalid SHA256: {name}")
    require(path.stat().st_size == entry["bytes"], f"File size differs: {name}")
    require(sha256(path) == digest, f"File SHA256 differs: {name}")


def verify_manifest(bundle: Path) -> dict[str, Any]:
    manifest_path = bundle / MANIFEST_NAME
    require(manifest_path.is_file() and not manifest_path.is_symlink(), "Bundle manifest is missing or linked")
    manifest = json.loads(manifest_path.read_text())
    require(isinstance(manifest, dict) and isinstance(manifest.get("files"), list), "Manifest needs a files list")
    entries = {}
    for entry in manifest["files"]:
        require(isinstance(entry, dict) and {"path", "sha256", "bytes"}.issubset(entry), "Malformed manifest file entry")
        name = entry["path"]
        relative_file(bundle, name)
        require(name != MANIFEST_NAME and name not in entries, "Manifest must exclude itself and have unique paths")
        entries[name] = entry
    require(bool(entries), "Empty bundle manifest")
    actual = set()
    for path in bundle.rglob("*"):
        require(not path.is_symlink(), f"Bundle contains a symlink: {path.relative_to(bundle)}")
        if path.is_file() and path.name != "":
            name = path.relative_to(bundle).as_posix()
            if name != MANIFEST_NAME:
                actual.add(name)
    require(actual == set(entries), f"Bundle inventory differs; missing={sorted(set(entries) - actual)}, extra={sorted(actual - set(entries))}")
    for name, entry in entries.items():
        verify_file(bundle, name, entry)
    return {"manifest_sha256": sha256(manifest_path), "verified_files": len(entries), "verified_bytes": sum(entry["bytes"] for entry in entries.values())}


def verify_receipt_files(directory: Path, receipt: dict[str, Any], key: str) -> None:
    require(isinstance(receipt.get(key), dict), f"Receipt lacks {key}")
    for name, entry in receipt[key].items():
        verify_file(directory, name, entry)


def run_command(arguments: list[str], output: Path, label: str) -> None:
    environment = os.environ.copy()
    environment.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
                       OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    result = subprocess.run(arguments, cwd=output, env=environment, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    (output / f"{label}.stdout.log").write_text(result.stdout)
    (output / f"{label}.stderr.log").write_text(result.stderr)
    require(result.returncode == 0, f"{label} failed with exit code {result.returncode}; see {label}.stderr.log")


def compare_texts(expected: Path, actual: Path, names: tuple[str, ...]) -> dict[str, Any]:
    result = {}
    for name in names:
        equal = (expected / name).read_bytes() == (actual / name).read_bytes()
        differences = []
        if not equal and name == "action_summary.tsv":
            with (expected/name).open(newline="") as stream:
                left = list(csv.reader(stream, delimiter="\t"))
            with (actual/name).open(newline="") as stream:
                right = list(csv.reader(stream, delimiter="\t"))
            require(bool(left) and len(left) == len(right) and left[0] == right[0], "Action table header or row count differs")
            require(len(set(left[0])) == len(left[0]) and "change_scores" in left[0], "Invalid action table header")
            for row_number, (a, b) in enumerate(zip(left[1:], right[1:]), 1):
                require(len(a) == len(b) == len(left[0]), "Action table column count differs")
                for column, av, bv in zip(left[0], a, b):
                    if av == bv:
                        continue
                    require(column == "change_scores", f"Exact action field differs: row {row_number}, {column}")
                    differences.append(score_difference(float(av), float(bv), f"action_summary.tsv:{row_number}:{column}"))
        elif not equal and name == "actions.json":
            left = json.loads((expected/name).read_text())
            right = json.loads((actual/name).read_text())
            compare_action_json(left, right, (), differences)
        else:
            require(equal, f"Reproduced bytes differ: {name}")
        result[name] = {"equal_bytes": equal, "sha256": sha256(actual / name), "bytes": (actual / name).stat().st_size,
                        "bounded_score_differences": len(differences),
                        "maximum_absolute_score_difference": max((d[0] for d in differences), default=0.0),
                        "maximum_relative_score_difference": max((d[1] for d in differences), default=0.0),
                        "score_absolute_tolerance": PEARSON_ATOL if differences else 0.0}
    return result


def score_difference(left: float, right: float, label: str) -> tuple[float, float]:
    require(math.isfinite(left) and math.isfinite(right), f"Nonfinite replay score: {label}")
    difference = abs(left-right)
    require(difference <= PEARSON_ATOL, f"Score differs beyond {PEARSON_ATOL}: {label}; absolute difference={difference}")
    relative = difference/max(abs(left), abs(right)) if left or right else 0.0
    return difference, relative


def compare_action_json(left: Any, right: Any, path: tuple, differences: list) -> None:
    require(type(left) is type(right), f"Action JSON type differs: {path}")
    if isinstance(left, dict):
        require(list(left) == list(right), f"Action JSON keys/order differ: {path}")
        for key in left:
            compare_action_json(left[key], right[key], path+(key,), differences)
    elif isinstance(left, list):
        require(len(left) == len(right), f"Action JSON list length differs: {path}")
        for index, (a, b) in enumerate(zip(left, right)):
            compare_action_json(a, b, path+(index,), differences)
    elif left != right:
        allowed = len(path) == 3 and type(path[0]) is int and (
            path[1:] in (("before_scores", PEARSON_FIELD), ("after_scores", PEARSON_FIELD),
                         ("before_after_max_abs_changes", "scores")))
        require(allowed and type(left) is float, f"Exact action JSON field differs: {path}")
        differences.append(score_difference(left, right, str(path)))


def compare_components(expected: Path, actual: Path) -> dict[str, Any]:
    count = 0
    values = 0
    pearson_differences = []
    exact = True
    with np.load(expected, allow_pickle=False) as saved, np.load(actual, allow_pickle=False) as replayed:
        require(set(saved.files) == set(replayed.files), "Component array names differ")
        for name in sorted(saved.files):
            left, right = saved[name], replayed[name]
            require(left.shape == right.shape and left.dtype == right.dtype, f"Component shape or dtype differs: {name}")
            require(np.issubdtype(left.dtype, np.number) or np.issubdtype(left.dtype, np.bool_), f"Nonnumeric component: {name}")
            equal = bool(np.array_equal(left, right, equal_nan=True))
            exact = exact and equal
            if not equal and name.endswith("__scores"):
                require(left.ndim == 2 and left.shape[1] == 4, f"Unexpected score axes: {name}")
                require(bool(np.array_equal(left[:, [0, 1, 3]], right[:, [0, 1, 3]], equal_nan=True)),
                        f"Non-Pearson component values differ: {name}")
                require(bool(np.array_equal(np.isnan(left[:, 2]), np.isnan(right[:, 2]))), f"Undefined Pearson support differs: {name}")
                for av, bv in zip(left[:, 2], right[:, 2]):
                    if np.isnan(av) and np.isnan(bv):
                        continue
                    if av != bv:
                        pearson_differences.append(score_difference(float(av), float(bv), name))
            else:
                require(equal, f"Component values differ: {name}")
            count += 1
            values += left.size
    return {"arrays": count, "scalar_values": values, "same_names_shapes_dtypes": True, "exact_values_equal_nan": exact,
            "non_pearson_values_exact": True, "bounded_pearson_differences": len(pearson_differences),
            "maximum_absolute_pearson_difference": max((d[0] for d in pearson_differences), default=0.0),
            "maximum_relative_pearson_difference": max((d[1] for d in pearson_differences), default=0.0),
            "pearson_absolute_tolerance": PEARSON_ATOL,
            "comparison": "Exact names, shapes, dtypes, missing-value support and non-Pearson values; Pearson scores use a bounded absolute float64 roundoff check. NPZ container bytes are not compared."}


def run(out: Path | str) -> dict[str, Any]:
    started = time.monotonic()
    bundle = BUNDLE.resolve()
    out = Path(out).expanduser().resolve()
    require(not out.exists(), "Output directory must be new")
    require(not out.is_relative_to(bundle), "Output directory must be outside the bundle")
    before = verify_manifest(bundle)
    out.mkdir(parents=True)
    result: dict[str, Any] = {"schema_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
                              "bundle_before": before, "python": sys.version, "numpy": np.__version__,
                              "original_caches_required": False, "new_training": False}
    failure = None
    try:
        metric_out = out / "metric_summary"
        run_command([sys.executable, str(bundle / "code/summarize_roles.py"), "--out-dir", str(metric_out)], out, "metric_summary")
        metric_receipt = json.loads((metric_out / "SUMMARY_RECEIPT.json").read_text())
        require(metric_receipt.get("status") == "PASS_COMPLETE_ROLE_SUMMARIES", "Metric summary receipt did not pass")
        verify_receipt_files(metric_out, metric_receipt, "outputs")
        result["metric_tables"] = compare_texts(bundle / "results", metric_out, METRIC_TABLES)
        result["metric_summary_status"] = metric_receipt["status"]
        result["metric_table_rows"] = metric_receipt["table_rows"]
        action_out = out / "actions"
        run_command([sys.executable, str(bundle / "code/role_actions.py"), "demo", "--data", str(bundle / "reader_demo"), "--out", str(action_out)], out, "actions")
        action_receipt = json.loads((action_out / "RECEIPT.json").read_text())
        require(action_receipt.get("status") == "PASS", "Action receipt did not pass")
        verify_receipt_files(action_out, action_receipt, "files")
        result["action_text"] = compare_texts(bundle / "action_results", action_out, ACTION_TEXT)
        result["action_components"] = compare_components(bundle / "action_results/components.npz", action_out / "components.npz")
        result["action_status"] = action_receipt["status"]
        result["valid_operation_cases"] = action_receipt["valid_operation_cases"]
        result["action_scope"] = action_receipt["scope"]
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
    try:
        after = verify_manifest(bundle)
        require(after == before, "Bundle changed during replay")
        result["bundle_after"] = after
        result["bundle_unchanged"] = True
    except Exception as error:
        failure = (failure + "; " if failure else "") + f"Bundle verification failed: {type(error).__name__}: {error}"
        result["bundle_unchanged"] = False
    result.update(status="PASS_BUNDLE_REPLAY" if failure is None else "FAIL_BUNDLE_REPLAY",
                  finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started)
    if failure is not None:
        result["failure"] = failure
    with (out / "REPLAY_RECEIPT.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    require(failure is None, failure or "Replay failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New output directory outside the bundle")
    args = parser.parse_args()
    run(args.out)


if __name__ == "__main__":
    main()
