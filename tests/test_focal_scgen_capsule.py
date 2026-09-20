from __future__ import annotations

import json
from pathlib import Path
import runpy

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
CAPSULE = REPOSITORY / "capsules/gse162632_scgen"


def test_realization_registry_matches_the_cli() -> None:
    registry = json.loads(
        (CAPSULE / "configs/realizations.json").read_text(encoding="utf-8")
    )
    observed = [(row["id"], row["seed"]) for row in registry["realizations"]]
    assert observed == [
        ("q1", 17),
        ("q2", 29),
        ("q3", 43),
        ("q4", 763929762),
        ("q5", 85508424),
    ]


def test_replay_axes_use_release_local_pseudonyms() -> None:
    manifest = json.loads(
        (CAPSULE / "replay/manifest.json").read_text(encoding="utf-8")
    )
    root_record = next(row for row in manifest["files"] if row["name"] == "root_tasks")
    text = (CAPSULE / "replay" / root_record["path"]).read_text(encoding="utf-8")
    assert "/d" + "pc/" not in text and "/ho" + "me/" not in text
    assert "HMN" not in text
    assert all(f"R{index:02d}" in text for index in range(41, 49))


def test_training_modules_are_importable_without_gpu_packages() -> None:
    import perturb_nuisance_focal.materialize  # noqa: F401
    import perturb_nuisance_focal.training  # noqa: F401


def test_containerized_slurm_examples_isolate_and_bind_inputs() -> None:
    for name in ("train_one.sbatch", "train_all.sbatch"):
        text = (CAPSULE / "slurm" / name).read_text(encoding="utf-8")
        assert "apptainer exec --cleanenv --nv" in text
        assert "--no-home --pwd /workspace/repository" in text
        assert '--bind "$repo_root:/workspace/repository:ro"' in text
        assert ":/workspace/prepared:ro" in text
        assert ":/workspace/output:rw" in text
        assert "APPTAINERENV_CUDA_VISIBLE_DEVICES" in text


def test_failed_training_acceptance_keeps_its_report(tmp_path: Path) -> None:
    namespace = runpy.run_path(
        str(REPOSITORY / "scripts/focal_scgen.py"),
        run_name="focal_scgen_failure_report_test",
    )
    stage = tmp_path / "stage"
    output = tmp_path / "q1"
    stage.mkdir()
    report = {"status": "FAIL", "result": {"shared_D": -0.5}}

    with pytest.raises(namespace["CapsuleError"], match="details are in"):
        namespace["_publish_training_output"](
            stage,
            output,
            report,
            accepted=False,
        )

    assert not stage.exists()
    assert json.loads((output / "report.json").read_text(encoding="utf-8")) == report
