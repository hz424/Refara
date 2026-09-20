#!/usr/bin/env python3
"""Replay or retrain the focal PCA-64 ridge versus scGen comparison."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Sequence

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from perturb_nuisance_focal.io import (  # noqa: E402
    CapsuleError,
    canonical_json_bytes,
    sha256_file,
)
from perturb_nuisance_focal.materialize import (  # noqa: E402
    load_evaluation_controls,
    materialize_prediction_references,
    predict_all_instances,
)
from perturb_nuisance_focal.replay import (  # noqa: E402
    RESULT_FIELDS,
    load_acceptance_values,
    load_replay_assets,
    reproduce_focal,
)
from perturb_nuisance_focal.scoring import focal_contrast  # noqa: E402
from perturb_nuisance_focal.training import (  # noqa: E402
    TrainingConfiguration,
    check_runtime,
    compute_task_shifts,
    latent_representation,
    load_training_config,
    load_prepared_training,
    make_training_anndata,
    module_state_sha256,
    train_model,
    validate_manifest_realizations,
)


CAPSULE = REPOSITORY / "capsules/gse162632_scgen"
DEFAULT_REPLAY = CAPSULE / "replay"
DEFAULT_EXPECTED = CAPSULE / "expected/focal_expected.tsv"
DEFAULT_CONFIG = CAPSULE / "configs/realizations.json"
TRAINING_POINT_ATOL = 0.01


def _write_receipt(path: Path | None, value: Any) -> None:
    if path is None:
        return
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise CapsuleError(f"Refusing to replace an existing receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def quickstart() -> dict[str, Any]:
    """Run a small focal calculation without reading external data."""

    from perturb_nuisance_focal.scoring import validate_axis

    axis = validate_axis([0, 1], [0, 0], ["R01", "R02"], ["T01"])
    treated = np.ones((2, 1), dtype="<f8")
    c_obs = np.zeros((5, 2, 1), dtype="<f8")
    pca64 = np.full((2, 1), 0.4, dtype="<f4")
    native = np.zeros((5, 2, 1), dtype="<f4")
    native[0] = 0.5
    result = focal_contrast(
        treated,
        c_obs,
        pca64,
        native,
        np.zeros_like(native),
        np.ones(1, dtype="<f8"),
        np.ones(1, dtype="<f8"),
        axis,
    )
    result.pop("root_contrasts")
    if not result["direction_pattern_passed"]:
        raise CapsuleError("The bundled focal calculation returned an unexpected result")
    return {
        "status": "PASS",
        "check": "synthetic focal calculation",
        "result": result,
    }


def _publish_training_output(
    stage: Path,
    output: Path,
    report: dict[str, Any],
    *,
    accepted: bool,
) -> None:
    (stage / "report.json").write_bytes(canonical_json_bytes(report))
    stage.rename(output)
    if not accepted:
        raise CapsuleError(
            "The retrained model did not meet the direction and numerical checks; "
            f"details are in {output / 'report.json'}"
        )


def _training_result(
    realization_id: str,
    prepared_root: Path,
    replay_root: Path,
    output: Path,
    config: TrainingConfiguration,
) -> dict[str, Any]:
    seed = config.seed_for(realization_id)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise CapsuleError(f"Refusing to replace an existing output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage.", dir=output.parent))
    published = False
    try:
        runtime = check_runtime()
        prepared = load_prepared_training(prepared_root)
        evaluation = load_evaluation_controls(prepared_root)
        validate_manifest_realizations(prepared.manifest, config)
        replay = load_replay_assets(replay_root, config.realizations)
        expected = load_acceptance_values(replay, DEFAULT_EXPECTED)[realization_id]
        if prepared.feature_ids != evaluation.feature_ids:
            raise CapsuleError("The training and evaluation feature axes differ")
        if (
            set(prepared.root_ids) & set(evaluation.root_ids)
            or {row["cell_id"] for row in prepared.cells}
            & {row["cell_id"] for row in evaluation.rows}
            or evaluation.root_ids != replay.axis.root_ids
            or evaluation.task_ids != replay.axis.task_ids
            or not np.array_equal(evaluation.scales, replay.arrays["scales"])
            or not np.array_equal(evaluation.weights, replay.arrays["weights"])
            or not np.array_equal(prepared.scales, evaluation.scales)
            or not np.array_equal(prepared.weights, evaluation.weights)
        ):
            raise CapsuleError("The prepared and replay axes or panel values differ")
        reconstructed_c_pred = materialize_prediction_references(evaluation)
        if not np.array_equal(reconstructed_c_pred, replay.arrays["c_pred"]):
            raise CapsuleError(
                "The reconstructed prediction-reference means differ from the replay"
            )

        adata = make_training_anndata(prepared)
        model = train_model(adata, seed, config)
        latent_before_save = latent_representation(model, adata, config)
        shifts_before_save = compute_task_shifts(
            latent_before_save, prepared, config
        )
        state_sha256 = module_state_sha256(model)

        model_dir = stage / "model"
        model.save(str(model_dir), overwrite=False, save_anndata=False)
        del model
        gc.collect()
        import torch

        torch.cuda.empty_cache()
        import scgen

        reloaded = scgen.SCGEN.load(str(model_dir), adata=adata, use_gpu=True)
        reloaded.module.eval()
        if module_state_sha256(reloaded) != state_sha256:
            raise CapsuleError("The saved and reloaded scGen states differ")
        latent_after_load = latent_representation(reloaded, adata, config)
        shifts = compute_task_shifts(latent_after_load, prepared, config)
        if not np.array_equal(
            latent_after_load, latent_before_save
        ) or not np.array_equal(shifts, shifts_before_save):
            raise CapsuleError("The reloaded model changed its latent values or task shifts")
        native, prediction_c_pred = predict_all_instances(
            reloaded,
            shifts,
            evaluation,
            batch_size=config.hyperparameters["batch_size"],
        )
        if not np.array_equal(prediction_c_pred, reconstructed_c_pred):
            raise CapsuleError(
                "Prediction materialization changed the prediction-reference means"
            )
        c_pred = reconstructed_c_pred
        result = focal_contrast(
            replay.arrays["treated_means"],
            replay.arrays["c_obs"],
            replay.arrays["pca64_effect"],
            native,
            c_pred,
            replay.arrays["scales"],
            replay.arrays["weights"],
            replay.axis,
        )
        root_contrasts = np.asarray(result.pop("root_contrasts"), dtype=np.float64)
        root_checks = {
            "shared_negative_in_all_roots": bool(np.all(root_contrasts[:, 0] < 0)),
            "split_positive_in_all_roots": bool(np.all(root_contrasts[:, 1] > 0)),
            "rotation_positive_in_all_roots": bool(np.all(root_contrasts[:, 2] > 0)),
        }
        differences = {
            field: float(result[field]) - float(expected[field]) for field in RESULT_FIELDS
        }
        numeric_passed = all(abs(value) <= TRAINING_POINT_ATOL for value in differences.values())
        q1_root_check_passed = realization_id != "q1" or all(root_checks.values())
        passed = bool(
            result["direction_pattern_passed"]
            and numeric_passed
            and q1_root_check_passed
        )

        shift_path = stage / "latent_shifts.npy"
        native_path = stage / "scgen_absolute_predictions.npy"
        c_pred_path = stage / "prediction_reference_means.npy"
        np.save(shift_path, np.ascontiguousarray(shifts, dtype="<f8"), allow_pickle=False)
        np.save(native_path, np.ascontiguousarray(native, dtype="<f4"), allow_pickle=False)
        np.save(c_pred_path, np.ascontiguousarray(c_pred, dtype="<f4"), allow_pickle=False)
        report = {
            "status": "PASS" if passed else "FAIL",
            "release_version": "1.6.1",
            "realization_id": realization_id,
            "seed": seed,
            "runtime": runtime,
            "gpu": torch.cuda.get_device_name(0),
            "hyperparameters": {**config.hyperparameters, "seed": seed},
            "result": result,
            "root_direction_checks": root_checks,
            "difference_from_saved_result": differences,
            "qualification": {
                "direction_pattern_required": True,
                "point_estimate_absolute_tolerance": TRAINING_POINT_ATOL,
                "point_estimates_within_tolerance": numeric_passed,
                "q1_all_root_reversal_required": realization_id == "q1",
                "q1_all_root_reversal_passed": (
                    all(root_checks.values()) if realization_id == "q1" else None
                ),
                "latent_values_and_task_shifts_equal_after_reload": True,
                "prediction_reference_means_equal_replay": True,
                "checkpoint_byte_identity_across_hardware_required": False,
            },
            "artifacts": {
                "latent_shifts.npy": sha256_file(shift_path),
                "scgen_absolute_predictions.npy": sha256_file(native_path),
                "prediction_reference_means.npy": sha256_file(c_pred_path),
                "module_state_logical_sha256": state_sha256,
            },
        }
        _publish_training_output(stage, output, report, accepted=passed)
        published = True
        return report
    finally:
        if not published:
            shutil.rmtree(stage, ignore_errors=True)


def train_all(
    prepared: Path,
    replay_assets: Path,
    output: Path,
    config: TrainingConfiguration,
) -> dict[str, Any]:
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise CapsuleError(f"Refusing to replace an existing output directory: {output}")
    output.mkdir(parents=True)
    reports: list[dict[str, Any]] = []
    try:
        for realization_id in config.realization_ids:
            reports.append(
                _training_result(
                    realization_id,
                    prepared,
                    replay_assets,
                    output / realization_id,
                    config,
                )
            )
    except Exception:
        # Keep completed runs and any failed acceptance report for inspection.
        raise
    summary = {
        "status": "PASS",
        "release_version": "1.6.1",
        "realizations": [report["realization_id"] for report in reports],
        "all_direction_and_tolerance_checks_passed": True,
    }
    (output / "summary.json").write_bytes(canonical_json_bytes(summary))
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Replay the focal empirical result on CPU or optionally retrain scGen "
            "on a GPU."
        )
    )
    commands = result.add_subparsers(dest="command", required=True)
    quick = commands.add_parser("quickstart", help="run the bundled synthetic check")
    quick.add_argument("--receipt", type=Path)

    replay = commands.add_parser(
        "reproduce-focal",
        help="recalculate the five focal contrasts from the bundled arrays on CPU",
    )
    replay.add_argument("--assets", type=Path, default=DEFAULT_REPLAY)
    replay.add_argument("--receipt", type=Path)

    one = commands.add_parser(
        "train-one", help="retrain one scGen realization on a GPU"
    )
    one.add_argument("--realization", default="q1", metavar="ID")
    one.add_argument("--data", type=Path, required=True)
    one.add_argument("--replay-assets", type=Path, default=DEFAULT_REPLAY)
    one.add_argument("--output", type=Path, required=True)

    all_runs = commands.add_parser(
        "train-all", help="retrain all five scGen realizations sequentially"
    )
    all_runs.add_argument("--data", type=Path, required=True)
    all_runs.add_argument("--replay-assets", type=Path, default=DEFAULT_REPLAY)
    all_runs.add_argument("--output", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "quickstart":
            report = quickstart()
            _write_receipt(args.receipt, report)
        elif args.command == "reproduce-focal":
            config = load_training_config(DEFAULT_CONFIG)
            report = reproduce_focal(
                args.assets,
                DEFAULT_EXPECTED,
                expected_realizations=config.realizations,
            )
            _write_receipt(args.receipt, report)
        elif args.command == "train-one":
            config = load_training_config(DEFAULT_CONFIG)
            report = _training_result(
                args.realization,
                args.data,
                args.replay_assets,
                args.output,
                config,
            )
        else:
            config = load_training_config(DEFAULT_CONFIG)
            report = train_all(
                args.data, args.replay_assets, args.output, config
            )
    except CapsuleError as error:
        print(f"focal-scgen: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
