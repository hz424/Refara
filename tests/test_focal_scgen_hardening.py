from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import numpy as np
import pytest

from perturb_nuisance_focal.io import CapsuleError, canonical_json_bytes, load_manifest
from perturb_nuisance_focal.materialize import (
    EXPECTED_DISPATCH,
    EvaluationControls,
    materialize_prediction_references,
)
from perturb_nuisance_focal.replay import (
    load_acceptance_values,
    load_replay_assets,
    read_expected,
)
from perturb_nuisance_focal.training import (
    EXPECTED_REALIZATIONS,
    SCGEN_IMPLEMENTATION_SHA256,
    _check_user_site_isolation,
    _verify_scgen_implementation,
    latent_representation,
    load_training_config,
)
import perturb_nuisance_focal.training as training_module


REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = REPOSITORY / "capsules/gse162632_scgen/configs/realizations.json"
EXPECTED = REPOSITORY / "capsules/gse162632_scgen/expected/focal_expected.tsv"
REPLAY = REPOSITORY / "capsules/gse162632_scgen/replay"


def _write_test_manifest(root: Path) -> None:
    payload = root / "payload.bin"
    payload.write_bytes(b"payload")
    manifest = {
        "schema_version": 1,
        "asset_kind": "test_assets",
        "release_version": "1.6.1",
        "files": [
            {
                "name": "payload",
                "path": "payload.bin",
                "bytes": payload.stat().st_size,
                "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
            }
        ],
    }
    (root / "manifest.json").write_bytes(canonical_json_bytes(manifest))


def test_training_configuration_is_the_runtime_source() -> None:
    config = load_training_config(CONFIG)
    assert config.realizations == EXPECTED_REALIZATIONS
    assert config.hyperparameters["validation_size"] is None
    assert config.hyperparameters["enable_checkpointing"] is False


def test_expected_table_has_typed_values_and_fixed_order() -> None:
    result = read_expected(EXPECTED)
    assert tuple(result) == tuple(name for name, _seed in EXPECTED_REALIZATIONS)
    assert all(isinstance(row["seed"], int) for row in result.values())
    assert all(
        isinstance(row["direction_pattern_passed"], bool) for row in result.values()
    )


def test_expected_table_rejects_a_duplicate_realization(tmp_path: Path) -> None:
    lines = [
        line for line in EXPECTED.read_text(encoding="utf-8").splitlines() if line
    ]
    fields = lines[-1].split("\t")
    fields[0] = "q1"
    lines[-1] = "\t".join(fields)
    path = tmp_path / "expected.tsv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(CapsuleError, match="IDs or seeds"):
        read_expected(path)


def test_external_expected_values_must_match_manifest_bound_values(
    tmp_path: Path,
) -> None:
    lines = EXPECTED.read_text(encoding="utf-8").splitlines()
    fields = lines[1].split("\t")
    fields[2] = "0"
    lines[1] = "\t".join(fields)
    changed = tmp_path / "expected.tsv"
    changed.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assets = load_replay_assets(REPLAY)
    with pytest.raises(CapsuleError, match="differs from the replay manifest"):
        load_acceptance_values(assets, changed)


def test_prediction_references_are_reconstructed_without_a_model() -> None:
    rows = tuple(
        {"root_task_index": "0", "base_pool": pool}
        for pool in ("B1", "B2", "B3")
    )
    dispatch = {
        name: {
            "c_obs_pools": values[0],
            "c_pred_pools": values[1],
            "c_infer_pools": values[2],
        }
        for name, values in EXPECTED_DISPATCH.items()
    }
    controls = EvaluationControls(
        matrix=np.asarray([[1.0], [2.0], [3.0]], dtype="<f4"),
        rows=rows,
        root_tasks=({"task_index": "0"},),
        dispatch=dispatch,
        feature_ids=("feature",),
        scales=np.ones(1, dtype="<f8"),
        weights=np.ones(1, dtype="<f8"),
        root_ids=("root",),
        task_ids=("task",),
    )
    observed = materialize_prediction_references(controls)
    np.testing.assert_array_equal(
        observed[:, 0, 0], np.asarray([2.0, 2.0, 2.0, 3.0, 1.0], dtype="<f4")
    )


def test_runtime_rejects_an_enabled_user_site(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    monkeypatch.setattr(training_module.site, "ENABLE_USER_SITE", True)
    with pytest.raises(CapsuleError, match="disabled before Python starts"):
        _check_user_site_isolation()


def test_scgen_source_origin_and_digest_are_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "scgen"
    package.mkdir()
    init_path = package / "__init__.py"
    model_path = package / "_scgen.py"
    vae_path = package / "_scgenvae.py"
    init_path.write_text("", encoding="utf-8")
    model_path.write_text("class SCGEN: pass\n", encoding="utf-8")
    vae_path.write_text("class SCGENVAE: pass\n", encoding="utf-8")

    scgen = ModuleType("scgen")
    scgen.__file__ = str(init_path)
    model_module = ModuleType("scgen._scgen")
    model_module.__file__ = str(model_path)
    vae_module = ModuleType("scgen._scgenvae")
    vae_module.__file__ = str(vae_path)
    scgen_class = type("SCGEN", (), {})
    scgen_class.__module__ = "scgen._scgen"
    vae_class = type("SCGENVAE", (), {})
    vae_class.__module__ = "scgen._scgenvae"
    scgen.SCGEN = scgen_class
    vae_module.SCGENVAE = vae_class

    class Distribution:
        @staticmethod
        def locate_file(_name: str) -> Path:
            return tmp_path

    modules = {
        "scgen._scgen": model_module,
        "scgen._scgenvae": vae_module,
    }
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: modules[name],
    )
    monkeypatch.setattr(
        training_module.importlib.metadata,
        "distribution",
        lambda _name: Distribution(),
    )
    digest = hashlib.sha256(vae_path.read_bytes()).hexdigest()
    assert _verify_scgen_implementation(scgen, expected_sha256=digest) == digest
    with pytest.raises(CapsuleError, match="qualified patch"):
        _verify_scgen_implementation(scgen, expected_sha256="0" * 64)
    assert SCGEN_IMPLEMENTATION_SHA256 == (
        "00348b640b5ea36355c4cdf335a89f74621e9d4d231b2d855c1190905441a527"
    )


def test_manifest_does_not_hide_a_nested_manifest(tmp_path: Path) -> None:
    _write_test_manifest(tmp_path)
    hidden = tmp_path / "nested"
    hidden.mkdir()
    (hidden / "manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(CapsuleError, match="unlisted"):
        load_manifest(tmp_path, "test_assets")


def test_manifest_rejects_symbolic_links(tmp_path: Path) -> None:
    _write_test_manifest(tmp_path)
    (tmp_path / "link").symlink_to(tmp_path / "payload.bin")
    with pytest.raises(CapsuleError, match="symbolic link"):
        load_manifest(tmp_path, "test_assets")


def test_latent_rows_must_match_anndata_rows() -> None:
    config = load_training_config(CONFIG)

    class Adata:
        n_obs = 3

    class Model:
        @staticmethod
        def get_latent_representation(**_kwargs: object) -> np.ndarray:
            return np.zeros((4, 100), dtype=np.float32)

    with pytest.raises(CapsuleError, match="latent representation"):
        latent_representation(Model(), Adata(), config)


def test_quickstart_stdout_is_one_json_document() -> None:
    completed = subprocess.run(
        [sys.executable, str(REPOSITORY / "scripts/focal_scgen.py"), "quickstart"],
        cwd=REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "PASS"
    assert report["check"] == "synthetic focal calculation"
