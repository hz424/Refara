"""Frozen real scGen inference, operation attribution and input boundaries."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from reference_design.conditioned_model import FrozenSCGen
from reference_design.real_model import normalize_full_counts

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/conditioned_model_reference"
DATA = EXAMPLE.parent / "real_model_reference/data"


@pytest.fixture(scope="module")
def model():
    return FrozenSCGen(EXAMPLE / "data/scgen_seed17.npz")


@pytest.fixture(scope="module")
def inputs():
    features = [r["feature_id"] for r in csv.DictReader((DATA / "source_features.tsv").open(), delimiter="\t")]
    counts = sparse.load_npz(DATA / "counts.npz")
    axis = np.load(DATA / "feature_axis.npy", allow_pickle=False)
    expression = normalize_full_counts(counts, features, features, axis)
    return expression, counts, features, axis


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location("conditioned_example", EXAMPLE / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def predict(model, x, **kwargs):
    return model.predict_points(x, context="CD4 Naive", condition="IFN-gamma",
                                feature_ids=model.feature_ids, **kwargs)


def test_real_network_point_contract_and_changed_controls(model, inputs):
    x = inputs[0][:24]
    bank = predict(model, x)
    assert not np.allclose(bank[:8].mean(0), bank[8:16].mean(0), atol=2e-5, rtol=2e-5)
    for rows in (np.array([0]), np.array([0, 0, 0]), np.array([9, 0, 21, 2]), np.arange(8)):
        np.testing.assert_allclose(predict(model, x[rows]), bank[rows], atol=2e-5, rtol=2e-5)
    assert bank.dtype == np.float32 and np.isfinite(bank).all()


def test_raw_gene_permutation_and_missing_gene_boundary(model, inputs):
    expression, counts, features, axis = inputs
    permutation = np.random.default_rng(3).permutation(len(features))
    reordered = normalize_full_counts(counts[:, permutation], [features[i] for i in permutation], features, axis)
    assert np.array_equal(reordered, expression)
    assert np.array_equal(predict(model, reordered[:8]), predict(model, expression[:8]))
    with pytest.raises(ValueError, match="complete source"):
        normalize_full_counts(counts[:, :-1], features[:-1], features, axis)
    with pytest.raises(ValueError, match="feature axis"):
        model.predict_points(expression[:8], context="CD4 Naive", condition="IFN-gamma",
                             feature_ids=model.feature_ids[::-1])


def test_operations_recompute_inputs_and_reuse_predictions_for_rescoring(tmp_path, monkeypatch, runner):
    calls = []
    original = FrozenSCGen.predict_points

    def traced(self, expression, **kwargs):
        calls.append(np.asarray(expression).copy())
        return original(self, expression, **kwargs)

    monkeypatch.setattr(FrozenSCGen, "predict_points", traced)
    result = runner.run(tmp_path / "audit")
    assert len(calls) == 13 and sum(len(x) for x in calls) == 120
    assert result["is_bundled_training_fixture"] is True
    assert result["inference"]["rescore_only_additional_forwards"] == 0
    assert result["O_encoding"]["maximum_loss_difference"] < 1e-12
    with np.load(tmp_path / "audit/inference_arrays.npz", allow_pickle=False) as z:
        for allocation in range(4):
            order = z["allocation_control_offsets"][allocation]
            assert sorted(order.tolist()) == list(range(24))
            for block in range(3):
                rows = z["control_rows"][order.reshape(3, 8)[block]]
                assert np.array_equal(calls[allocation * 3 + block], z["normalized_input_cells"][rows])
            diagnostics = {r["operation"]: r for r in result["diagnostics"] if r["allocation"] == allocation}
            assert diagnostics["anchor_O"]["prediction_mean_sha256"] == diagnostics["rescore_only"]["prediction_mean_sha256"]
            assert diagnostics["input_only"]["prediction_mean_sha256"] != diagnostics["anchor_O"]["prediction_mean_sha256"]
            assert diagnostics["input_only"]["requires_changed_input_prediction"]
            assert diagnostics["rescore_only"]["scaled_mean_prediction_RMS_from_anchor"] == 0
            for row in diagnostics.values():
                m, p, o = ["ABC".index(row[key]) for key in ("model_input", "prediction_center", "observation_reference")]
                residual = (z["native_state_means"][allocation, m] - z["input_control_means"][allocation, p]
                            - (z["treated_mean"] - z["input_control_means"][allocation, o]))
                assert np.mean((residual / z["scales"]) ** 2) == pytest.approx(row["loss"], abs=1e-12)
        rows = list(csv.DictReader((tmp_path / "audit/role_scores.tsv").open(), delimiter="\t"))
        assert len(rows) == 108
        for row in rows:
            if row["same_encoding_contract"] == "True":
                assert float(row["native_state_loss"]) == pytest.approx(float(row["O_effect_encoding_loss"]), abs=1e-12)
            else:
                assert row["O_effect_encoding_loss"] == ""


@pytest.mark.parametrize("kind", ["negative", "nan", "overflow", "underflow", "bool", "wrong_shape"])
def test_invalid_expression_rejected(model, inputs, kind):
    x = inputs[0][:1].astype(np.float64)
    if kind == "negative": x[0, 0] = -1
    elif kind == "nan": x[0, 0] = np.nan
    elif kind == "overflow": x[0, 0] = 1e300
    elif kind == "underflow": x[0, 0] = 1e-300
    elif kind == "bool": x = x.astype(bool)
    else: x = x[:, :-1]
    with pytest.raises(ValueError): predict(model, x)


def test_unknown_task_and_checkpoint_binding_rejected(model, inputs):
    for context, condition in (("unknown", "IFN-gamma"), ("CD4 Naive", "unknown")):
        with pytest.raises(ValueError, match="Unknown trained"):
            model.predict_points(inputs[0][:1], context=context, condition=condition, feature_ids=model.feature_ids)
    with pytest.raises(ValueError, match="SHA256"):
        FrozenSCGen(EXAMPLE / "data/scgen_seed17.npz", expected_sha256="0" * 64)


@pytest.mark.parametrize("tasks", [["ab"], [{"a": 1, "b": 2}], [["a", ["b"]]], None])
def test_malformed_task_metadata_rejected(tmp_path, tasks):
    with np.load(EXAMPLE / "data/scgen_seed17.npz", allow_pickle=False) as z:
        arrays = {name: z[name] for name in z.files}
    metadata = json.loads(arrays["metadata"].item())
    metadata["tasks"] = tasks
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    np.savez(tmp_path / "malformed.npz", **arrays)
    with pytest.raises(ValueError, match="Tasks must"):
        FrozenSCGen(tmp_path / "malformed.npz")


def test_output_collision_is_rejected_before_inference(tmp_path, runner, monkeypatch):
    marker = tmp_path / "keep.txt"
    marker.write_text("unchanged")
    monkeypatch.setattr(FrozenSCGen, "predict_points", lambda *a, **k: pytest.fail("Must reject output first"))
    with pytest.raises(ValueError, match="new or empty"):
        runner.run(tmp_path)
    assert marker.read_text() == "unchanged"


def test_custom_input_is_bound_without_training_provenance_claim(tmp_path, runner):
    config = json.loads((DATA / "input.json").read_text())
    for key in ("counts", "features", "cells"):
        config[key] = str(DATA / config[key])
    config["unit"] = "caller_unit"
    source = tmp_path / "input.json"
    source.write_text(json.dumps(config))
    result = runner.run(tmp_path / "custom", source)
    assert result["unit"] == "caller_unit" and result["is_bundled_training_fixture"] is False
    assert "not authenticated" in result["input_scope"]
