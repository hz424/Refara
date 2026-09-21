"""Real frozen-weight example plus preprocessing and interface boundaries."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from reference_design.real_model import FrozenPCA, normalize_full_counts

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "real_model_reference"


@pytest.fixture
def example():
    spec = importlib.util.spec_from_file_location("real_model_example", EXAMPLE / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_inference_calls_memberships_and_reencoding(tmp_path, monkeypatch, example):
    calls = []
    native_predict = FrozenPCA.predict

    def tracked(self, context, condition):
        calls.append((context, condition))
        return native_predict(self, context, condition)

    monkeypatch.setattr(FrozenPCA, "predict", tracked)
    result = example.run(tmp_path / "result")
    assert calls == [("CD4 Naive", "IFN-gamma")] * 8
    assert result["inference"]["prediction_bank_loaded"] is False
    assert result["inference"]["uses_control_cells"] is False
    assert result["inference"]["control_cell_forwards"] == 0
    assert result["representation_max_loss_difference"] < 1e-12
    members = example.table((tmp_path / "result" / "cell_membership.tsv").read_bytes())
    input_sets = []
    for allocation in range(4):
        def group(protocol, role):
            return {r["cell_id"] for r in members if r["allocation"] == str(allocation)
                    and r["protocol"] == protocol and r["role"] == role}
        assert group("S", "input") == group("S", "observation")
        assert len(group("S", "input")) == 16
        assert len(group("O", "input")) == len(group("O", "observation")) == 8
        assert not group("O", "input") & group("O", "observation")
        assert group("O", "input") | group("O", "observation") == group("S", "input")
        assert group("S", "treated") == group("O", "treated")
        input_sets.append(frozenset(group("O", "input")))
    assert len(set(input_sets)) > 1
    scores = result["scores"]
    assert len({r["PCA_native_effect_loss"] for r in scores}) > 1
    assert all(r["NC_zero_effect_loss"] >= 0 for r in scores)
    assert json.loads((tmp_path / "result" / "RESULTS.json").read_text())["scores"] == scores
    effects = np.load(tmp_path / "result" / "inferred_effects.npy", allow_pickle=False)
    assert effects.shape == (8, 2000)
    assert np.array_equal(effects, np.broadcast_to(effects[0], effects.shape))


def test_fitted_checkpoint_binding_and_unknown_labels():
    provenance = json.loads((EXAMPLE / "PROVENANCE.json").read_text())
    model = FrozenPCA.from_npz(EXAMPLE / "data" / "model.npz",
                              expected_sha256=provenance["model"]["original_checkpoint_sha256"])
    assert model.components.shape == (128, 2000)
    assert not model.coefficients.flags.writeable
    assert np.isfinite(model.predict("CD4 Naive", "IFN-gamma")).all()
    assert not np.array_equal(model.predict("CD4 Naive", "IFN-gamma"),
                              model.predict("CD4 Naive", "IFN-beta"))
    with pytest.raises(ValueError, match="SHA256"):
        FrozenPCA.from_npz(EXAMPLE / "data" / "model.npz", expected_sha256="0" * 64)
    for context, condition in (("unseen", "IFN-gamma"), ("CD4 Naive", "unseen"), (None, "IFN-gamma")):
        with pytest.raises(ValueError, match="Unknown"):
            model.predict(context, condition)


def test_full_library_before_projection_and_column_permutation():
    counts = np.array([[2, 0, 8], [1, 3, 6]])
    expected = np.log1p([[2000, 0], [1000, 3000]]).astype(np.float32)
    actual = normalize_full_counts(counts, ["a", "b", "outside"], ["a", "b", "outside"], [0, 1])
    assert np.array_equal(actual, expected)
    permuted = normalize_full_counts(sparse.csr_matrix(counts[:, [2, 0, 1]]),
                                     ["outside", "a", "b"], ["a", "b", "outside"], [0, 1])
    assert np.array_equal(permuted, actual)
    wrong_partial_library = np.log1p(10000 * counts[0, 0] / counts[0, :2].sum())
    assert actual[0, 0] != wrong_partial_library


@pytest.mark.parametrize("counts", [
    [[0, 0]], [[1, -1]], [[1, np.nan]], [[1, np.inf]], [[1, .5]],
    [[1, 2 ** 54]], [[True, False]], [[1 + 0j, 2]], [1, 2], [],
])
def test_invalid_counts_rejected(counts):
    with pytest.raises(ValueError):
        normalize_full_counts(counts, ["a", "b"], ["a", "b"], [0])


@pytest.mark.parametrize("supplied,panel", [(["a"], [0]), (["a", "a"], [0]),
    (["a", "other"], [0]), (["a", "b"], [0, 0]), (["a", "b"], [2]),
    (["a", "b"], [-1]), (["a", "b"], [0.]), (["a", "b"], []),
])
def test_invalid_axes_rejected(supplied, panel):
    with pytest.raises(ValueError):
        normalize_full_counts([[1, 2]], supplied, ["a", "b"], panel)


def test_collision_rejected_before_inference(tmp_path, monkeypatch, example):
    destination = tmp_path / "existing"
    destination.mkdir()
    marker = destination / "do_not_overwrite.txt"
    marker.write_text("preserved")
    monkeypatch.setattr(FrozenPCA, "predict", lambda *args: pytest.fail("Must reject before inference"))
    with pytest.raises(ValueError, match="new or empty"):
        example.run(destination)
    assert marker.read_text() == "preserved"


def test_custom_input_binding_and_empty_output(tmp_path, example):
    config = json.loads((EXAMPLE / "data" / "input.json").read_text())
    for key in ("counts", "features", "cells"):
        config[key] = str(EXAMPLE / "data" / config[key])
    config["unit"] = "custom_input_interface_check"
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(config))
    output = tmp_path / "output"
    output.mkdir()
    result = example.run(output, input_path)
    assert result["unit"] == "custom_input_interface_check"
    assert result["input_binding"]["input_json"]["bytes"] == input_path.stat().st_size


def test_corrupt_model_shape_and_nonfinite_parameters(tmp_path):
    with np.load(EXAMPLE / "data" / "model.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["intercept"] = arrays["intercept"][:-1]
    np.savez(tmp_path / "bad.npz", **arrays)
    with pytest.raises(ValueError, match="axes disagree"):
        FrozenPCA.from_npz(tmp_path / "bad.npz")
    arrays["intercept"] = np.full(128, np.nan)
    np.savez(tmp_path / "bad.npz", **arrays)
    with pytest.raises(ValueError, match="finite"):
        FrozenPCA.from_npz(tmp_path / "bad.npz")
