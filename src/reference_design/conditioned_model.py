"""Frozen scGen posterior-mean inference with NumPy and exported real weights.

The encoder and decoder are learned nonlinear networks. Each control cell is
encoded, shifted by its fixed training-derived task vector, and decoded to a
native predicted state. Dropout is disabled and batch normalization uses stored
training statistics. This is the deterministic evaluation path, not VAE sampling
or a training interface. Different matrix libraries can differ by float32
roundoff; the bundled example records parity against the original Torch path.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class FrozenSCGen:
    """Control-dependent state predictor loaded from a safe numeric NPZ.

    ``predict_points`` requires the fitted gene order explicitly. Calls do real
    inference and have no prediction cache. Each output depends on its input
    cell and the fixed task shift, not on other cells in its inference batch.
    """

    def __init__(self, path, *, expected_sha256=None):
        payload = Path(path).read_bytes()
        self.checkpoint_sha256 = hashlib.sha256(payload).hexdigest()
        if expected_sha256 is not None:
            _require(self.checkpoint_sha256 == expected_sha256, "Checkpoint SHA256 differs")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            _require(isinstance(metadata, dict) and metadata.get("schema") == "REFARA_FROZEN_SCGEN_MEAN_V1",
                     "Unknown frozen scGen format")
            self.metadata = metadata
            self.arrays = {name: archive[name].copy() for name in archive.files if name != "metadata"}
        architecture = metadata.get("architecture")
        _require(architecture == {"input": 2000, "hidden": 512, "latent": 100, "layers": 2,
                                  "batchnorm_eps": 0.001, "negative_slope": 0.01,
                                  "inference_dtype": "float32"}, "Unsupported network architecture")
        raw_tasks = metadata.get("tasks")
        _require(isinstance(raw_tasks, list) and bool(raw_tasks)
                 and all(isinstance(t, list) and len(t) == 2
                         and all(isinstance(v, str) and v.strip() for v in t) for t in raw_tasks),
                 "Tasks must be context/condition string pairs")
        self.tasks = tuple(tuple(task) for task in raw_tasks)
        _require(len(set(self.tasks)) == len(self.tasks),
                 "Invalid or duplicated context/condition tasks")
        shapes = {"feature_axis": (2000,), "scales": (2000,), "latent_shifts": (len(self.tasks), 100),
                  "encoder_mean_weight": (100, 512), "encoder_mean_bias": (100,),
                  "decoder_output_weight": (2000, 512), "decoder_output_bias": (2000,)}
        for block, first_input in (("encoder", 2000), ("decoder", 100)):
            for layer in range(2):
                prefix = f"{block}_{layer}_"
                shapes[prefix + "weight"] = (512, first_input if layer == 0 else 512)
                for name in ("bias", "bn_weight", "bn_bias", "bn_mean", "bn_var"):
                    shapes[prefix + name] = (512,)
        _require(set(self.arrays) == set(shapes), "Unexpected or missing parameter arrays")
        for name, shape in shapes.items():
            a = self.arrays[name]
            _require(a.shape == shape and a.dtype.kind in "iuf" and np.isfinite(a).all(),
                     f"Invalid parameter array: {name}")
            if name not in ("feature_axis", "scales"):
                _require(a.dtype == np.float32, "Network weights must retain original float32 values")
            if name.endswith("bn_var"):
                _require(np.all(a >= 0), "Negative batch-normalization variance")
            a.flags.writeable = False
        axis = self.arrays["feature_axis"]
        _require(axis.dtype.kind in "iu" and np.all(axis >= 0) and len(np.unique(axis)) == len(axis),
                 "Feature indices must be nonnegative unique integers")
        _require(np.all(self.arrays["scales"] > 0), "Scoring scales must be positive")
        self.feature_ids = tuple(map(str, axis))
        self.scales = self.arrays["scales"]
        self.calls = 0
        self.control_cell_forwards = 0

    def _linear(self, x, prefix):
        return x @ self.arrays[prefix + "weight"].T + self.arrays[prefix + "bias"]

    def _hidden(self, x, block):
        for layer in range(2):
            prefix = f"{block}_{layer}_"
            x = self._linear(x, prefix)
            # Fixed running statistics, never statistics of the supplied batch.
            inverse_sd = np.float32(1.) / np.sqrt(self.arrays[prefix + "bn_var"] + np.float32(.001))
            x = ((x - self.arrays[prefix + "bn_mean"]) * inverse_sd
                 * self.arrays[prefix + "bn_weight"] + self.arrays[prefix + "bn_bias"])
            x = np.where(x >= 0, x, np.float32(.01) * x)
        return x

    def predict_points(self, expression, *, context, condition, feature_ids):
        """Predict one native state per supplied CP10K/log1p control cell.

        The output has the same (cells, 2000 genes) shape. Use the full-library
        normalization contract in the example before calling this method.
        Models, shifts, scales and normalization are never fitted on these cells.
        """
        _require(not isinstance(feature_ids, (str, bytes)) and tuple(feature_ids) == self.feature_ids,
                 "Prediction feature axis differs from the fitted axis")
        _require(isinstance(context, str) and isinstance(condition, str) and (context, condition) in self.tasks,
                 "Unknown trained context/condition")
        raw = np.asarray(expression)
        _require(raw.ndim == 2 and raw.shape[0] > 0 and raw.shape[1] == 2000
                 and raw.dtype.kind in "iuf" and np.isfinite(raw).all() and np.all(raw >= 0),
                 "Expression must be finite nonnegative cells by 2000 fitted genes")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            x = np.asarray(raw, dtype=np.float32)
        _require(np.isfinite(x).all() and not np.any((raw != 0) & (x == 0)),
                 "Expression cannot be represented as finite float32 without underflow")
        pieces = []
        shift = self.arrays["latent_shifts"][self.tasks.index((context, condition))]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for start in range(0, len(x), 512):
                latent = self._linear(self._hidden(x[start:start + 512], "encoder"), "encoder_mean_")
                shifted = np.add(latent, shift, dtype=np.float32)
                pieces.append(self._linear(self._hidden(shifted, "decoder"), "decoder_output_"))
        result = np.concatenate(pieces)
        _require(result.shape == x.shape and np.isfinite(result).all(), "Network inference overflowed")
        self.calls += 1
        self.control_cell_forwards += len(x)
        return result
