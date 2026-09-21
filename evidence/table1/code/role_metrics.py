"""Score all three-reference interventions with fixed Norman metric definitions.

This is a retrospective replay of cached predictions. The run binds its source,
protocol and input inventory before decoding targets; this order does not make
previously inspected data a prospective validation set.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "2"

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterator

import numpy as np
import scipy
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
PREVIOUS = ROOT / "submission/work/reference_coupling_validation_v1"
PREVIOUS_PROTOCOL_SHA256 = "e9bc0d97149657eb1b9957275e2e30c7d4e6fd2a649de0cec0abeb0b388e0d03"
PREVIOUS_BINDING_SHA256 = "36db8b187b9cdfdc7c9fc434e855cd2d78bbad90123906e39b6fef1236b166ab"
INPUT_ADAPTER_SHA256 = "653d16a79d6dc23831bc512449b16f209a0ce182dc24d4612fd02e32f7137455"
INPUT_INVENTORY_SHA256 = "2eb15dfb0a474ad52db41f96eb761ddbb48d4978525314546695f440c0672a08"
METRICS = ("standardized_MSE", "raw_RMSE", "perturbation_centred_Pearson", "centroid_accuracy")
MODELS = ("selected_CPA", "original_CPA_fixed", "additive", "compositional_ridge", "NC_native_effect", "native_effect_ridge")
KINDS = ("state", "state", "state", "state", "native_effect", "native_effect")
ROLE_TUPLES = tuple(itertools.product(range(3), repeat=3))
OFF_DIAGONAL = tuple((o, p) for o in range(3) for p in range(3) if o != p)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def score_batch(predictions: np.ndarray, targets: np.ndarray,
                scales: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return [task, profile, metric]; all candidate targets remain available."""
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    require(predictions.ndim == 3 and targets.shape == (predictions.shape[0], predictions.shape[2]), "Task/profile/gene dimensions differ")
    residual = predictions - targets[:, None, :]
    scores = np.empty(predictions.shape[:2] + (4,), dtype=np.float64)
    scores[..., 0] = np.mean(np.square(residual / scales), axis=-1)
    scores[..., 1] = np.sqrt(np.mean(np.square(residual), axis=-1))
    p = predictions - reference
    p -= p.mean(axis=-1, keepdims=True)
    y = targets - reference
    y -= y.mean(axis=-1, keepdims=True)
    denominator = np.linalg.norm(p, axis=-1) * np.linalg.norm(y, axis=-1)[:, None]
    scores[..., 2] = np.nan
    np.divide(np.sum(p * y[:, None, :], axis=-1), denominator,
              out=scores[..., 2], where=denominator > 0)
    scores[..., 2] = np.clip(scores[..., 2], -1, 1)
    distances = cdist(predictions.reshape(-1, predictions.shape[-1]), targets, metric="euclidean")
    correct_indices = np.repeat(np.arange(len(targets)), predictions.shape[1])
    correct_distances = distances[np.arange(len(distances)), correct_indices, None]
    scores[..., 3] = (np.sum(distances > correct_distances, axis=1) / (len(targets) - 1)).reshape(predictions.shape[:2])
    return scores


def rendering_plan() -> tuple[list[tuple[Any, ...]], np.ndarray, np.ndarray]:
    """Map every role tuple to an exactly equivalent unique prediction row."""
    keys: list[tuple[Any, ...]] = []
    index: dict[tuple[Any, ...], int] = {}

    def slot(key: tuple[Any, ...]) -> int:
        if key not in index:
            index[key] = len(keys)
            keys.append(key)
        return index[key]

    direct = np.empty((3, len(MODELS)), dtype=np.int64)
    role = np.empty((len(ROLE_TUPLES), len(MODELS)), dtype=np.int64)
    for mi, kind in enumerate(KINDS):
        for i in range(3):
            key = (mi, "state", i if mi == 0 else 0, -1, -1) if kind == "state" else (mi, "TRAIN_lift")
            direct[i, mi] = slot(key)
        for ri, (o, p, i) in enumerate(ROLE_TUPLES):
            if kind == "native_effect":
                key = (mi, "observation_lift", o)
            else:
                # o == p is evaluated as the original state, avoiding a
                # subtract/add rounding artifact in a representation identity.
                key = (mi, "state", i if mi == 0 else 0, o if o != p else -1, p if o != p else -1)
            role[ri, mi] = slot(key)
    require(len(keys) == 50, "Unexpected unique rendering count")
    return keys, direct, role


def render_profiles(keys: list[tuple[Any, ...]], selected: np.ndarray,
                    fixed: np.ndarray, effects: np.ndarray,
                    controls: np.ndarray, training_control: np.ndarray) -> np.ndarray:
    """Render 50 unique profiles for every task of one depth/allocation."""
    profiles = np.empty((selected.shape[0], len(keys), selected.shape[-1]), dtype=np.float64)
    for slot, key in enumerate(keys):
        mi, kind, *rest = key
        if kind == "state":
            i, o, p = rest
            base = selected[:, i] if mi == 0 else fixed[mi - 1]
            profiles[:, slot] = base if o == -1 else base - controls[p] + controls[o]
        elif kind == "TRAIN_lift":
            profiles[:, slot] = effects[mi - 4] + training_control
        else:
            profiles[:, slot] = effects[mi - 4] + controls[rest[0]]
    return profiles


def load_adapter():
    path = PREVIOUS / "code/input_adapter.py"
    require(sha256(path) == INPUT_ADAPTER_SHA256, "Prior input adapter changed")
    require(sha256(PREVIOUS / "PROTOCOL.json") == PREVIOUS_PROTOCOL_SHA256, "Prior protocol changed")
    require(sha256(PREVIOUS / "FORECAST_BINDING.json") == PREVIOUS_BINDING_SHA256, "Prior forecast binding changed")
    require(sha256(PREVIOUS / "INPUT_INVENTORY.json") == INPUT_INVENTORY_SHA256, "Prior input inventory changed")
    spec = importlib.util.spec_from_file_location("role_value_bound_input_adapter", path)
    require(spec is not None and spec.loader is not None, "Cannot import bound input adapter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def iter_source_batches(planning) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """Yield [task, input block, gene] predictions and [block, gene] controls.

    The adapter and its validated array shapes are hash bound. Access to these
    planning-only arrays permits batches over all tasks without loading targets.
    """
    arrays = planning._arrays
    for di, _depth in enumerate(planning.metadata["depths"]):
        for ai, _allocation in enumerate(planning.metadata["allocations"]):
            yield di, ai, np.asarray(arrays["selected_CPA"][:, ai, di], dtype=np.float64), np.asarray(arrays["control_means"][ai, di], dtype=np.float64)


def self_test() -> dict[str, Any]:
    rng = np.random.default_rng(912604)
    targets = rng.normal(size=(5, 7))
    predictions = rng.normal(size=(5, 4, 7))
    scales = rng.uniform(0.2, 3.0, size=7)
    reference = rng.normal(size=7)
    vector = score_batch(predictions, targets, scales, reference)
    scalar = np.empty_like(vector)
    for ti in range(len(targets)):
        for pi, prediction in enumerate(predictions[ti]):
            residual = prediction - targets[ti]
            scalar[ti, pi] = [np.mean((residual / scales) ** 2), np.sqrt(np.mean(residual ** 2)),
                              np.corrcoef(prediction - reference, targets[ti] - reference)[0, 1],
                              sum(np.linalg.norm(prediction - candidate) > np.linalg.norm(prediction - targets[ti]) for candidate in targets) / 4]
    np.testing.assert_allclose(vector, scalar, rtol=1e-13, atol=1e-13)
    duplicate = np.array([[0., 1., 3.], [0., 1., 3.], [2., 0., 1.]])
    perfect = score_batch(duplicate[:, None, :], duplicate, np.ones(3), np.zeros(3))
    np.testing.assert_allclose(perfect[:2, 0], np.tile([0., 0., 1., .5], (2, 1)), atol=1e-15)
    constant = score_batch(np.ones((3, 1, 3)), duplicate, np.ones(3), np.zeros(3))
    require(bool(np.isnan(constant[..., 2]).all()), "Constant Pearson must be undefined")
    keys, direct_map, role_map = rendering_plan()
    selected = rng.normal(size=(5, 3, 7))
    fixed = rng.normal(size=(3, 5, 7))
    effects = rng.normal(size=(2, 5, 7))
    controls = rng.normal(size=(3, 7))
    training = rng.normal(size=7)
    rendered = render_profiles(keys, selected, fixed, effects, controls, training)
    direct = rendered[:, direct_map]
    role = rendered[:, role_map]
    max_error = 0.0
    for ri, (o, p, i) in enumerate(ROLE_TUPLES):
        for mi in range(6):
            expected = ((selected[:, i] if mi == 0 else fixed[mi - 1]) - controls[p] + controls[o]) if mi < 4 else effects[mi - 4] + controls[o]
            max_error = max(max_error, float(np.max(np.abs(role[:, ri, mi] - expected))))
            np.testing.assert_allclose(role[:, ri, mi], expected, rtol=0, atol=2e-15)
    for i in range(3):
        np.testing.assert_array_equal(direct[:, i, 0], selected[:, i])
        for mi in (4, 5):
            np.testing.assert_array_equal(direct[:, i, mi], effects[mi - 4] + training)
    return {"status": "PASS", "scalar_metric_values": int(scalar.size), "unique_profiles": len(keys),
            "metric_max_error": float(np.max(np.abs(vector - scalar))), "role_rendering_max_error": max_error,
            "strict_centroid_tie": True, "undefined_constant_Pearson": True,
            "native_TRAIN_lift": True, "all_role_tuples_checked": 27}


def run(outdir: Path | str) -> dict[str, Any]:
    started = time.monotonic()
    outdir = Path(outdir)
    protocol_path = HERE / "PROTOCOL.json"
    require(protocol_path.is_file(), "Root PROTOCOL.json must exist before execution")
    protocol_sha = sha256(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    require(isinstance(protocol, dict), "Protocol must be a JSON object")
    digest_path = HERE / "PROTOCOL.sha256"
    if digest_path.exists():
        require(digest_path.read_text().split()[0] == protocol_sha, "Root protocol digest differs")
    require(not outdir.exists(), "Output directory already exists")
    synthetic = self_test()
    adapter = load_adapter()
    planning = adapter.load_planning("Norman")
    metadata = planning.metadata
    require(metadata["models"] == list(MODELS), "Model axis differs")
    require(metadata["task_labels"] == metadata["task_ids"], "Task labels and IDs differ")
    require(metadata["depths"] == [8, 16, 32, 64, 128, 135] and metadata["allocations"] == list(range(30)), "Depth/allocation axis differs")
    require(metadata["records"] == 9900, "Task support differs")
    keys, direct_map, role_map = rendering_plan()
    outdir.mkdir(parents=True)
    binding = {"schema_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
               "study_protocol_sha256": protocol_sha, "engine_sha256": sha256(Path(__file__)),
               "previous_protocol_sha256": PREVIOUS_PROTOCOL_SHA256, "previous_forecast_binding_sha256": PREVIOUS_BINDING_SHA256,
               "input_adapter_sha256": INPUT_ADAPTER_SHA256, "input_inventory_sha256": INPUT_INVENTORY_SHA256,
               "source_bindings": metadata["source_bindings"], "target_decoding_begins_after_this_binding": True,
               "design_status": "Retrospective replay; historical targets and results were previously inspected.",
               "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "blas_threads": 2,
               "unique_prediction_profiles_per_task_depth_allocation": len(keys)}
    binding_path = outdir / "EXECUTION_BINDING.json"
    write_json(binding_path, binding)
    binding_path.chmod(0o444)
    # Only this gated call decodes observed_states in this process.
    target_map = adapter.load_targets("Norman", freeze_path=binding_path, freeze_sha256=sha256(binding_path))
    tasks = metadata["task_ids"]
    require(set(tasks) == set(target_map), "Target task support differs")
    targets = np.stack([target_map[task] for task in tasks])
    arrays = planning._arrays
    fixed = np.asarray(arrays["profiles"][1:4], dtype=np.float64)
    effects = np.stack([np.zeros_like(targets), arrays["profiles"][5] - arrays["native_effect_training_baseline"]])
    training = metadata["TRAIN_control"]
    shape_prefix = (len(tasks), len(metadata["depths"]), len(metadata["allocations"]))
    direct_cube = np.full(shape_prefix + (3, 6, 4), np.nan, dtype=np.float64)
    role_cube = np.full(shape_prefix + (27, 6, 4), np.nan, dtype=np.float64)
    completed = 0
    for di, ai, selected, controls in iter_source_batches(planning):
        predictions = render_profiles(keys, selected, fixed, effects, controls, training)
        values = score_batch(predictions, targets, arrays["scales"], metadata["TRAIN_perturbation"])
        direct_cube[:, di, ai] = values[:, direct_map]
        role_cube[:, di, ai] = values[:, role_map]
        completed += 1
        if completed % 30 == 0:
            print(json.dumps({"completed_batches": completed, "required_batches": 180, "elapsed_seconds": round(time.monotonic() - started, 3)}), flush=True)
    require(completed == 180, "Incomplete batch support")
    require(sha256(protocol_path) == protocol_sha, "Protocol changed during execution")
    for label, cube in (("direct_state", direct_cube), ("role_effect", role_cube)):
        require(bool(np.isfinite(cube[..., [0, 1, 3]]).all()), f"Nonfinite non-Pearson score in {label}")
    cube_path = outdir / "task_scores.npz"
    np.savez_compressed(cube_path, tasks=np.asarray(tasks), models=np.asarray(MODELS), model_kinds=np.asarray(KINDS),
                        model_input_conditioned=np.asarray([True, False, False, False, False, False]),
                        depths=np.asarray(metadata["depths"]), allocations=np.asarray(metadata["allocations"]),
                        metrics=np.asarray(METRICS), metric_directions=np.asarray(["lower", "lower", "higher", "higher"]),
                        input_blocks=np.arange(3), role_tuples=np.asarray(ROLE_TUPLES),
                        role_tuple_columns=np.asarray(["Cobs", "Cpred", "Cmodel"]),
                        scores_direct_state=direct_cube, scores_role_effect=role_cube,
                        unique_direct_profile_indices=direct_map, unique_role_profile_indices=role_map,
                        gene_ids=np.asarray(metadata["gene_ids"]), controls_per_block=np.asarray(metadata["depths"]) * 8,
                        controls_available=np.asarray(metadata["depths"]) * 24)
    receipt = {**binding, "status": "PASS_COMPLETE_METRIC_CUBES", "finished_utc": datetime.now(timezone.utc).isoformat(),
               "elapsed_seconds": time.monotonic() - started, "tasks": len(tasks), "depths": metadata["depths"],
               "allocations": metadata["allocations"], "models": list(MODELS), "metrics": list(METRICS),
               "direct_state_shape": list(direct_cube.shape), "role_effect_shape": list(role_cube.shape),
               "direct_state_axes": ["task", "depth", "allocation", "input_block", "model", "metric"],
               "role_effect_axes": ["task", "depth", "allocation", "role_tuple", "model", "metric"],
               "role_tuple_order": ["Cobs", "Cpred", "Cmodel"], "role_tuples": ROLE_TUPLES,
               "direct_state_rendering": "State M_i against Y; native E+saved TRAIN_control against Y.",
               "role_effect_rendering": "State M_i-b_p+b_o against Y; native E+b_o against Y, ignoring p and i.",
               "native_effect_definition": "NC E=0; native ridge E=stored profile[5]-saved native_effect_training_baseline.",
               "metric_semantics": {"standardized_MSE": "Mean squared residual divided by squared saved training scales.",
                                    "raw_RMSE": "Square root of mean squared raw residual over fixed 2000 genes.",
                                    "perturbation_centred_Pearson": "Pearson across genes after subtracting fixed saved TRAIN_perturbation from both rendered prediction and target.",
                                    "centroid_accuracy": "Fraction of 54 incorrect target centroids strictly farther than the correct centroid in raw Euclidean distance; ties receive zero credit."},
               "reference_scope": "The fixed TRAIN perturbation reference and all 55 retrieval candidates are shared across every role intervention. No reference is estimated from test targets.",
               "support": "All 55 tasks, 30 allocations, 6 depths, 6 models, 27 ordered role tuples; each cached Norman model has one fit.",
               "missingness": "Pearson NaNs are retained; no task deletion or replacement. Other metrics must be finite.",
               "undefined_values": {"direct_state": int(np.count_nonzero(~np.isfinite(direct_cube))), "role_effect": int(np.count_nonzero(~np.isfinite(role_cube)))},
               "synthetic_checks": synthetic,
               "outputs": {cube_path.name: {"sha256": sha256(cube_path), "bytes": cube_path.stat().st_size},
                           binding_path.name: {"sha256": sha256(binding_path), "bytes": binding_path.stat().st_size}}}
    write_json(outdir / "RECEIPT.json", receipt)
    print(json.dumps({"status": receipt["status"], "elapsed_seconds": receipt["elapsed_seconds"], "cube_bytes": cube_path.stat().st_size}), flush=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["self-test", "run"])
    parser.add_argument("--out-dir", type=Path, default=HERE / "empirical")
    args = parser.parse_args()
    if args.command == "self-test":
        print(json.dumps(self_test(), indent=2))
    else:
        run(args.out_dir)


if __name__ == "__main__":
    main()
