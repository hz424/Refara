"""Portable inference for the frozen Parse PCA direct-effect baseline.

This predictor uses trained context and condition labels, at unit dose. It does
not condition on control cells. Controls enter reference scoring separately.
The example assets retain their source-data terms; this module contains code
only and does not download weights or data.
"""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _labels(values, name):
    _require(isinstance(values, (list, tuple)) and bool(values), f"{name} must be a nonempty sequence")
    _require(all(isinstance(x, str) and x.strip() == x and x for x in values),
             f"{name} must contain nonempty strings")
    _require(len(set(values)) == len(values), f"{name} contains duplicate labels")
    return tuple(values)


def _parameters(value, name):
    raw = np.asarray(value)
    _require(raw.dtype.kind == "f", f"{name} must contain floating-point parameters")
    result = np.array(raw, dtype=np.float64, copy=True)
    _require(np.isfinite(result).all(), f"{name} must be finite")
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class FrozenPCA:
    """A trained additive-design PCA/ridge model with native effect output.

    Load with :meth:`from_npz`. ``predict`` runs the fitted matrix operations;
    it never reads a prediction bank. Feature IDs are the training source-axis
    indices, stored as strings in the original checkpoint.
    """

    feature_ids: tuple[str, ...]
    context_ids: tuple[str, ...]
    compound_dose_ids: tuple[str, ...]
    feature_mean: np.ndarray
    components: np.ndarray
    intercept: np.ndarray
    coefficients: np.ndarray
    checkpoint_sha256: str

    @classmethod
    def from_npz(cls, path, *, expected_sha256=None):
        """Load numeric weights without pickle, optionally binding exact bytes."""
        payload = Path(path).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if expected_sha256 is not None:
            _require(digest == expected_sha256, "Checkpoint SHA256 differs")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            required = {"metadata", "feature_mean", "components", "intercept", "coefficients"}
            _require(set(archive.files) == required, "Unexpected checkpoint members")
            metadata = json.loads(str(archive["metadata"].item()))
            _require(metadata.get("model") == "PCA", "Checkpoint is not a PCA model")
            features = _labels(metadata["feature_ids"], "Feature IDs")
            contexts = _labels(metadata["context_ids"], "Context IDs")
            compounds = _labels(metadata["compound_dose_ids"], "Condition/dose IDs")
            _require(all(x.endswith("|dose_uM=1") for x in compounds), "Only unit-dose checkpoints are supported")
            mean, components, intercept, coefficients = (
                _parameters(archive[key], key)
                for key in ("feature_mean", "components", "intercept", "coefficients")
            )
        _require(components.ndim == 2 and components.shape[0] > 0, "Components must be a nonempty matrix")
        rank, genes = components.shape
        _require(genes == len(features) and mean.shape == (genes,) and intercept.shape == (rank,)
                 and coefficients.shape == (len(contexts) + len(compounds), rank),
                 "Checkpoint feature/design axes disagree")
        return cls(features, contexts, compounds, mean, components, intercept, coefficients, digest)

    def predict(self, context: str, condition: str) -> np.ndarray:
        """Return one native effect vector, using the original design arithmetic.

        Unknown labels are rejected. There is deliberately no control argument:
        changing control membership cannot change this model's prediction.
        """
        _require(isinstance(context, str) and context in self.context_ids, "Unknown trained context")
        _require(isinstance(condition, str), "Condition must be a string")
        token = condition + "|dose_uM=1"
        _require(token in self.compound_dose_ids, "Unknown trained condition")
        design = np.zeros((1, len(self.context_ids) + len(self.compound_dose_ids)), dtype=np.float64)
        design[0, self.context_ids.index(context)] = 1.
        design[0, len(self.context_ids) + self.compound_dose_ids.index(token)] = 1.
        with np.errstate(over="ignore", invalid="ignore"):
            latent = self.intercept + design @ self.coefficients
            prediction = self.feature_mean + latent @ self.components
        _require(np.isfinite(prediction).all(), "Prediction overflowed")
        return prediction[0].copy()


def normalize_full_counts(counts, input_feature_ids, source_feature_ids, panel_indices):
    """Full-library CP10K/log1p followed by the fixed training feature projection.

    Require the complete training source axis, allowing an explicit permutation.
    Library sizes include genes outside the scoring panel. Return float32, as
    in the original training bundle. No genes are imputed or fitted here.
    """
    source = _labels(source_feature_ids, "Source feature IDs")
    supplied = _labels(input_feature_ids, "Input feature IDs")
    _require(len(supplied) == len(source) and set(supplied) == set(source),
             "Input must contain the complete source feature axis")
    indices = np.asarray(panel_indices)
    _require(indices.ndim == 1 and indices.size > 0 and indices.dtype.kind in "iu",
             "Panel indices must be a nonempty integer vector")
    _require(np.all(indices >= 0) and np.all(indices < len(source))
             and len(np.unique(indices)) == len(indices), "Panel indices are repeated or outside the source axis")
    if sparse.issparse(counts):
        _require(counts.ndim == 2, "Counts must be a cells by features matrix")
        matrix = counts.tocsr(copy=True)
        raw = matrix.data
    else:
        dense = np.asarray(counts)
        _require(dense.ndim == 2, "Counts must be a cells by features matrix")
        raw = dense
        matrix = None
    _require(raw.dtype.kind in "iuf", "Counts must be real numeric values")
    _require(np.isfinite(raw).all() and np.all(raw >= 0) and np.all(raw == np.floor(raw)),
             "Counts must be finite nonnegative integers")
    _require(np.all(raw <= 2 ** 53), "Counts exceed exact float64 integer range")
    matrix = sparse.csr_matrix(counts, dtype=np.float64, copy=True) if matrix is None else matrix.astype(np.float64)
    matrix.sum_duplicates()
    _require(matrix.shape[0] > 0 and matrix.shape[1] == len(source), "Counts and feature axes disagree")
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    _require(np.isfinite(totals).all() and np.all(totals > 0) and np.all(totals <= 2 ** 53),
             "Each cell must have a positive, exactly representable library size")
    # The full library is computed before any gene selection.
    normalized = matrix.multiply((10000. / totals)[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    lookup = {name: i for i, name in enumerate(supplied)}
    columns = [lookup[source[int(i)]] for i in indices]
    return normalized[:, columns].toarray().astype(np.float32)
