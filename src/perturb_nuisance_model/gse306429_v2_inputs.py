"""Frozen GSE306429 V2 transforms and training-profile construction.

This module is deliberately file-format agnostic.  It accepts already parsed
cell-axis records and ordered CSR row chunks; it does not open H5AD objects,
materialize predictions, or calculate benchmark scores.

The implementation retains only sufficient statistics by bio-sample, library,
and feature.  Consequently, memory use is bounded by the registered feature,
bio-sample, and library axes rather than by the number of cells.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy import sparse


CP10K_TOTAL = 10_000.0
SCALE_FLOOR = 0.1
CONTROL_ROLE = "DMSO_CONTROL"
PERTURBATION_ROLE = "PERTURBATION"
CONTROL_LABEL = "DMSO"
_FIT_PATTERN = re.compile(
    r"^train_rep(?P<training>.+)__eval_rep(?P<evaluation>.+)"
    r"__task_fold(?P<fold>[0-9]+)$"
)
_MISSING_STRINGS = frozenset({"", "na", "nan", "none", "null"})


class GSE306429InputError(ValueError):
    """Raised when an input violates the frozen V2 adapter contract."""


def _text(value: object, *, name: str, allow_empty: bool = False) -> str:
    if not isinstance(value, (str, np.str_)):
        raise GSE306429InputError(f"{name} must be a string")
    result = str(value)
    if "\0" in result or result != result.strip():
        raise GSE306429InputError(f"{name} is not a canonical string")
    if not allow_empty and result.casefold() in _MISSING_STRINGS:
        raise GSE306429InputError(f"{name} is missing")
    try:
        result.encode("utf-8")
    except UnicodeEncodeError as error:
        raise GSE306429InputError(f"{name} is not valid UTF-8") from error
    return result


def _optional_text(value: object, *, name: str) -> str:
    if value is None:
        return ""
    return _text(value, name=name, allow_empty=True)


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise GSE306429InputError(f"{name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise GSE306429InputError(f"{name} must be an integer") from error
    if str(parsed) != str(value).strip() and not isinstance(value, (int, np.integer)):
        raise GSE306429InputError(f"{name} is not a canonical integer")
    if parsed < minimum:
        raise GSE306429InputError(f"{name} must be at least {minimum}")
    return parsed


def _fit_set(value: object, *, name: str) -> frozenset[str]:
    text = _optional_text(value, name=name)
    if not text:
        return frozenset()
    labels = tuple(text.split(";"))
    if any(not label for label in labels) or len(labels) != len(set(labels)):
        raise GSE306429InputError(f"{name} has an empty or duplicate fit ID")
    for label in labels:
        _text(label, name=name)
        if _FIT_PATTERN.fullmatch(label) is None:
            raise GSE306429InputError(f"{name} contains a malformed fit ID")
    return frozenset(labels)


@dataclass(frozen=True)
class FitBoundary:
    """Training/evaluation boundary encoded by a frozen model-fit ID."""

    fit_id: str
    training_replicate: str
    evaluation_replicate: str
    heldout_task_fold: int

    @classmethod
    def parse(cls, fit_id: object) -> "FitBoundary":
        label = _text(fit_id, name="fit_id")
        match = _FIT_PATTERN.fullmatch(label)
        if match is None:
            raise GSE306429InputError("fit_id has the wrong format")
        training = _text(match.group("training"), name="training_replicate")
        evaluation = _text(match.group("evaluation"), name="evaluation_replicate")
        if training == evaluation:
            raise GSE306429InputError(
                "training and evaluation replicates must differ"
            )
        return cls(
            fit_id=label,
            training_replicate=training,
            evaluation_replicate=evaluation,
            heldout_task_fold=int(match.group("fold")),
        )


@dataclass(frozen=True)
class CellAxisRow:
    """The subset of one metadata-only cell-axis row required for fitting."""

    obs_position: int
    obs_index: str
    bio_sample_id: str
    library_id: str
    replicate: str
    context: str
    compound_name: str
    dose_um: float | None
    cell_role: str
    task_id: str
    task_fold_id: int | None
    training_fit_ids: frozenset[str]
    evaluation_fit_ids: frozenset[str]

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "CellAxisRow":
        required = {
            "obs_position",
            "obs_index",
            "bio_sample_id",
            "library_id",
            "replicate",
            "context",
            "compound_name",
            "dose_uM",
            "cell_role",
            "task_id",
            "task_fold_id",
            "training_fit_ids",
            "evaluation_fit_ids",
        }
        missing = sorted(required.difference(row))
        if missing:
            raise GSE306429InputError(
                f"cell-axis row lacks required fields: {missing}"
            )
        role = _text(row["cell_role"], name="cell_role")
        if role not in {CONTROL_ROLE, PERTURBATION_ROLE}:
            raise GSE306429InputError(f"unknown cell_role: {role!r}")
        compound = _text(row["compound_name"], name="compound_name")
        if (role == CONTROL_ROLE) != (compound == CONTROL_LABEL):
            raise GSE306429InputError(
                "cell_role and compound_name disagree about DMSO control status"
            )
        task = _optional_text(row["task_id"], name="task_id")
        fold_text = _optional_text(row["task_fold_id"], name="task_fold_id")
        dose_text = _optional_text(row["dose_uM"], name="dose_uM")
        if role == CONTROL_ROLE:
            if task or fold_text or dose_text:
                raise GSE306429InputError(
                    "a DMSO control row must not carry task or dose fields"
                )
            fold = None
            dose = None
        else:
            if not task or not fold_text or not dose_text:
                raise GSE306429InputError(
                    "a perturbation row must carry task, fold, and dose"
                )
            fold = _integer(fold_text, name="task_fold_id")
            try:
                dose = float(dose_text)
            except (TypeError, ValueError, OverflowError) as error:
                raise GSE306429InputError("dose_uM is not numeric") from error
            if not math.isfinite(dose) or dose <= 0.0:
                raise GSE306429InputError(
                    "perturbation dose_uM must be finite and positive"
                )
        training = _fit_set(row["training_fit_ids"], name="training_fit_ids")
        evaluation = _fit_set(
            row["evaluation_fit_ids"], name="evaluation_fit_ids"
        )
        if training & evaluation:
            raise GSE306429InputError(
                "cell-axis row belongs to training and evaluation for one fit"
            )
        return cls(
            obs_position=_integer(
                row["obs_position"], name="obs_position", minimum=0
            ),
            obs_index=_text(row["obs_index"], name="obs_index"),
            bio_sample_id=_text(row["bio_sample_id"], name="bio_sample_id"),
            library_id=_text(row["library_id"], name="library_id"),
            replicate=_text(row["replicate"], name="replicate"),
            context=_text(row["context"], name="context"),
            compound_name=compound,
            dose_um=dose,
            cell_role=role,
            task_id=task,
            task_fold_id=fold,
            training_fit_ids=training,
            evaluation_fit_ids=evaluation,
        )


def parse_cell_axis_rows(
    rows: Iterable[Mapping[str, object] | CellAxisRow],
) -> tuple[CellAxisRow, ...]:
    """Parse and validate a complete, source-row-ordered cell axis."""

    parsed = tuple(
        row if isinstance(row, CellAxisRow) else CellAxisRow.from_mapping(row)
        for row in rows
    )
    if not parsed:
        raise GSE306429InputError("cell axis is empty")
    positions = tuple(row.obs_position for row in parsed)
    if positions != tuple(range(len(parsed))):
        raise GSE306429InputError(
            "cell axis must contain each obs_position once in source-row order"
        )
    indices = tuple(row.obs_index for row in parsed)
    if len(indices) != len(set(indices)):
        raise GSE306429InputError("obs_index is not globally unique")
    return parsed


@dataclass(frozen=True)
class CountsChunk:
    """A contiguous CSR block on the frozen H5AD row and feature axes."""

    row_start: int
    counts: sparse.csr_matrix


def transform_raw_counts_csr(
    counts: sparse.spmatrix,
) -> sparse.csr_matrix:
    """Apply per-cell CP10K and natural log1p to a raw-count CSR block.

    Explicit sparse zeros are removed.  Duplicate or unsorted feature indices,
    zero-total cells, noninteger counts, and any nonfinite or negative value
    fail closed.
    """

    if not sparse.issparse(counts) or getattr(counts, "format", None) != "csr":
        raise GSE306429InputError("counts must be a scipy CSR object")
    if counts.ndim != 2 or counts.shape[0] == 0 or counts.shape[1] == 0:
        raise GSE306429InputError("counts must be a non-empty two-dimensional CSR")
    if not counts.has_canonical_format:
        raise GSE306429InputError(
            "counts CSR must have sorted, duplicate-free feature indices"
        )
    try:
        data = np.asarray(counts.data, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise GSE306429InputError("counts data are not numeric") from error
    if not np.isfinite(data).all():
        raise GSE306429InputError("counts contain a nonfinite value")
    if np.any(data < 0.0):
        raise GSE306429InputError("counts contain a negative value")
    if not np.equal(data, np.rint(data)).all():
        raise GSE306429InputError("counts contain a noninteger value")

    transformed = sparse.csr_matrix(
        (
            data.copy(),
            counts.indices.astype(np.int64, copy=True),
            counts.indptr.astype(np.int64, copy=True),
        ),
        shape=counts.shape,
    )
    transformed.eliminate_zeros()
    if not transformed.has_canonical_format:
        raise GSE306429InputError("canonical CSR structure was not preserved")
    totals = np.asarray(transformed.sum(axis=1), dtype=np.float64).reshape(-1)
    if not np.isfinite(totals).all() or np.any(totals <= 0.0):
        raise GSE306429InputError("every count row must have a positive finite total")
    row_nnz = np.diff(transformed.indptr)
    transformed.data *= np.repeat(CP10K_TOTAL / totals, row_nnz)
    np.log1p(transformed.data, out=transformed.data)
    if not np.isfinite(transformed.data).all():
        raise GSE306429InputError("CP10K-log1p produced a nonfinite value")
    return transformed


def _labels(
    values: Sequence[object], *, name: str, require_unique: bool = False
) -> tuple[str, ...]:
    labels = tuple(_text(value, name=name) for value in values)
    if not labels:
        raise GSE306429InputError(f"{name} is empty")
    if require_unique and len(labels) != len(set(labels)):
        raise GSE306429InputError(f"{name} must be unique")
    return labels


def _utf8_order(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values), key=lambda value: value.encode("utf-8")))


def _frozen_array(
    value: np.ndarray,
    *,
    dtype: np.dtype = np.dtype(np.float64),
) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype)
    if not np.isfinite(result).all():
        raise GSE306429InputError("constructed state-ready array is nonfinite")
    result.setflags(write=False)
    return result


def _kahan_add(
    total: np.ndarray, compensation: np.ndarray, increment: np.ndarray
) -> None:
    corrected = increment - compensation
    updated = total + corrected
    compensation[...] = (updated - total) - corrected
    total[...] = updated


@dataclass(frozen=True)
class FitTrainingInputs:
    """State-ready, equal-task training effects and training-only scales."""

    fit_id: str
    training_replicate: str
    evaluation_replicate: str
    heldout_task_fold: int
    feature_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    contexts: tuple[str, ...]
    compounds: tuple[str, ...]
    doses_um: np.ndarray
    effects: np.ndarray
    task_weights: np.ndarray
    task_bio_sample_counts: np.ndarray
    task_cell_counts: np.ndarray
    feature_scales: np.ndarray
    training_dmso_cell_count: int
    training_perturbation_cell_count: int
    training_obs_sha256: str
    bio_sample_ids_by_task: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class _SampleMetadata:
    library_id: str
    task_id: str
    context: str
    compound: str
    dose_um: float


def _selected_training_groups(
    axis: tuple[CellAxisRow, ...],
    boundary: FitBoundary,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    dict[str, _SampleMetadata],
    np.ndarray,
    np.ndarray,
    str,
]:
    selected_samples: dict[str, _SampleMetadata] = {}
    selected_libraries: set[str] = set()
    sample_by_position = np.full(len(axis), -1, dtype=np.int64)
    library_by_position = np.full(len(axis), -1, dtype=np.int64)
    selected_records: list[str] = []

    for row in axis:
        in_training = boundary.fit_id in row.training_fit_ids
        in_evaluation = boundary.fit_id in row.evaluation_fit_ids
        if row.replicate not in {
            boundary.training_replicate,
            boundary.evaluation_replicate,
        }:
            raise GSE306429InputError(
                "cell axis contains a replicate outside the fit boundary"
            )
        if in_training and in_evaluation:
            raise GSE306429InputError(
                f"obs_position {row.obs_position} crosses the fit boundary"
            )
        if in_training:
            if row.replicate != boundary.training_replicate:
                raise GSE306429InputError(
                    "training_fit_ids selects a row outside the training replicate"
                )
            if (
                row.cell_role == PERTURBATION_ROLE
                and row.task_fold_id == boundary.heldout_task_fold
            ):
                raise GSE306429InputError(
                    "training_fit_ids selects a held-out perturbation task"
                )
        if in_evaluation:
            if row.replicate != boundary.evaluation_replicate:
                raise GSE306429InputError(
                    "evaluation_fit_ids selects a row outside the evaluation replicate"
                )
            if (
                row.cell_role == PERTURBATION_ROLE
                and row.task_fold_id != boundary.heldout_task_fold
            ):
                raise GSE306429InputError(
                    "evaluation_fit_ids selects a non-held-out perturbation task"
                )
        if row.cell_role == CONTROL_ROLE:
            if (
                row.replicate == boundary.training_replicate
                and not in_training
            ):
                raise GSE306429InputError(
                    "a training-replicate DMSO row is absent from training_fit_ids"
                )
            if (
                row.replicate == boundary.evaluation_replicate
                and not in_evaluation
            ):
                raise GSE306429InputError(
                    "an evaluation-replicate DMSO row is absent from "
                    "evaluation_fit_ids"
                )
        elif (
            row.replicate == boundary.training_replicate
            and row.task_fold_id != boundary.heldout_task_fold
            and not in_training
        ):
            raise GSE306429InputError(
                "an eligible training perturbation row is absent from "
                "training_fit_ids"
            )

        if not in_training:
            continue
        selected_records.append(
            f"{row.obs_position}\0{row.obs_index}\0{row.cell_role}\n"
        )
        if row.cell_role == CONTROL_ROLE:
            selected_libraries.add(row.library_id)
            continue
        if (
            row.task_fold_id is None
            or row.dose_um is None
            or not row.task_id
        ):
            raise GSE306429InputError("selected perturbation metadata is incomplete")
        metadata = _SampleMetadata(
            library_id=row.library_id,
            task_id=row.task_id,
            context=row.context,
            compound=row.compound_name,
            dose_um=row.dose_um,
        )
        prior = selected_samples.setdefault(row.bio_sample_id, metadata)
        if prior != metadata:
            raise GSE306429InputError(
                f"bio-sample {row.bio_sample_id!r} has conflicting metadata"
            )

    sample_ids = _utf8_order(selected_samples)
    library_ids = _utf8_order(selected_libraries)
    if not sample_ids:
        raise GSE306429InputError("fit has no eligible perturbation bio-samples")
    if not library_ids:
        raise GSE306429InputError("fit has no eligible DMSO controls")
    sample_index = {sample: index for index, sample in enumerate(sample_ids)}
    library_index = {library: index for index, library in enumerate(library_ids)}

    for row in axis:
        if boundary.fit_id not in row.training_fit_ids:
            continue
        if row.cell_role == CONTROL_ROLE:
            library_by_position[row.obs_position] = library_index[row.library_id]
        else:
            sample_by_position[row.obs_position] = sample_index[row.bio_sample_id]
    missing_control = sorted(
        {
            metadata.library_id
            for metadata in selected_samples.values()
            if metadata.library_id not in library_index
        }
    )
    if missing_control:
        raise GSE306429InputError(
            f"training perturbations lack same-library DMSO support: {missing_control}"
        )
    digest = hashlib.sha256("".join(selected_records).encode("utf-8")).hexdigest()
    return (
        sample_ids,
        library_ids,
        selected_samples,
        sample_by_position,
        library_by_position,
        digest,
    )


def _coerce_chunk(
    value: CountsChunk | tuple[int, sparse.spmatrix],
) -> CountsChunk:
    if isinstance(value, CountsChunk):
        return value
    if not isinstance(value, tuple) or len(value) != 2:
        raise GSE306429InputError(
            "each counts chunk must be CountsChunk or (row_start, CSR)"
        )
    row_start, counts = value
    if not sparse.issparse(counts):
        raise GSE306429InputError("counts chunk payload is not sparse")
    return CountsChunk(
        row_start=_integer(row_start, name="row_start", minimum=0),
        counts=counts,  # type: ignore[arg-type]
    )


def build_fit_training_inputs(
    counts_chunks: Iterable[CountsChunk | tuple[int, sparse.spmatrix]],
    cell_axis_rows: Sequence[Mapping[str, object] | CellAxisRow],
    *,
    fit_id: str,
    feature_ids: Sequence[object],
) -> FitTrainingInputs:
    """Build one fit's task effects and DMSO-derived feature scales.

    Chunks must cover the complete cell axis exactly once and in source-row
    order.  Floating-point reductions use float64 and compensated accumulation
    in that fixed order.  No evaluation-replicate or held-out perturbation row
    is admitted to any returned quantity.
    """

    boundary = FitBoundary.parse(fit_id)
    axis = parse_cell_axis_rows(cell_axis_rows)
    features = _labels(feature_ids, name="feature_ids", require_unique=True)
    (
        sample_ids,
        library_ids,
        sample_metadata,
        sample_by_position,
        library_by_position,
        training_obs_sha256,
    ) = _selected_training_groups(axis, boundary)
    n_features = len(features)
    sample_sums = np.zeros((len(sample_ids), n_features), dtype=np.float64)
    sample_comp = np.zeros_like(sample_sums)
    sample_counts = np.zeros(len(sample_ids), dtype=np.int64)
    library_sums = np.zeros((len(library_ids), n_features), dtype=np.float64)
    library_comp = np.zeros_like(library_sums)
    library_counts = np.zeros(len(library_ids), dtype=np.int64)
    scale_sum = np.zeros(n_features, dtype=np.float64)
    scale_sum_comp = np.zeros(n_features, dtype=np.float64)
    scale_square_sum = np.zeros(n_features, dtype=np.float64)
    scale_square_comp = np.zeros(n_features, dtype=np.float64)

    next_row = 0
    saw_chunk = False
    for raw_chunk in counts_chunks:
        chunk = _coerce_chunk(raw_chunk)
        if chunk.row_start != next_row:
            raise GSE306429InputError(
                "counts chunks are not contiguous in source-row order"
            )
        transformed = transform_raw_counts_csr(chunk.counts)
        if transformed.shape[1] != n_features:
            raise GSE306429InputError(
                "counts feature count differs from the frozen feature axis"
            )
        stop = next_row + transformed.shape[0]
        if stop > len(axis):
            raise GSE306429InputError("counts chunks exceed the cell axis")
        saw_chunk = True
        local_sample_codes = sample_by_position[next_row:stop]
        local_library_codes = library_by_position[next_row:stop]

        for code in np.unique(local_sample_codes[local_sample_codes >= 0]):
            local_rows = np.flatnonzero(local_sample_codes == code)
            increment = np.asarray(
                transformed[local_rows].sum(axis=0), dtype=np.float64
            ).reshape(-1)
            _kahan_add(sample_sums[code], sample_comp[code], increment)
            sample_counts[code] += len(local_rows)

        control_local_rows = np.flatnonzero(local_library_codes >= 0)
        if len(control_local_rows):
            control_matrix = transformed[control_local_rows]
            control_sum = np.asarray(
                control_matrix.sum(axis=0), dtype=np.float64
            ).reshape(-1)
            control_square_sum = np.asarray(
                control_matrix.multiply(control_matrix).sum(axis=0),
                dtype=np.float64,
            ).reshape(-1)
            _kahan_add(scale_sum, scale_sum_comp, control_sum)
            _kahan_add(
                scale_square_sum, scale_square_comp, control_square_sum
            )
            for code in np.unique(local_library_codes[control_local_rows]):
                local_rows = np.flatnonzero(local_library_codes == code)
                increment = np.asarray(
                    transformed[local_rows].sum(axis=0), dtype=np.float64
                ).reshape(-1)
                _kahan_add(library_sums[code], library_comp[code], increment)
                library_counts[code] += len(local_rows)
        next_row = stop

    if not saw_chunk:
        raise GSE306429InputError("counts chunk stream is empty")
    if next_row != len(axis):
        raise GSE306429InputError(
            "counts chunks do not cover the complete cell axis"
        )
    if np.any(sample_counts <= 0):
        raise GSE306429InputError("an eligible perturbation bio-sample has no cells")
    if np.any(library_counts <= 0):
        raise GSE306429InputError("an eligible training library has no DMSO cells")

    dmso_count = int(library_counts.sum())
    if dmso_count < 2:
        raise GSE306429InputError(
            "at least two training DMSO cells are required for ddof=1 scales"
        )
    centered_ss = scale_square_sum - (scale_sum * scale_sum) / dmso_count
    cancellation_bound = (
        64.0
        * np.finfo(np.float64).eps
        * np.maximum(
            np.maximum(
                np.abs(scale_square_sum),
                (scale_sum * scale_sum) / dmso_count,
            ),
            1.0,
        )
    )
    if np.any(centered_ss < -cancellation_bound):
        raise GSE306429InputError(
            "DMSO scale moments imply a negative variance beyond roundoff"
        )
    centered_ss = np.maximum(centered_ss, 0.0)
    scales = np.maximum(np.sqrt(centered_ss / (dmso_count - 1)), SCALE_FLOOR)

    library_lookup = {library: index for index, library in enumerate(library_ids)}
    sample_effects: dict[str, np.ndarray] = {}
    for index, sample in enumerate(sample_ids):
        metadata = sample_metadata[sample]
        control_index = library_lookup[metadata.library_id]
        sample_effects[sample] = (
            sample_sums[index] / sample_counts[index]
            - library_sums[control_index] / library_counts[control_index]
        )

    samples_by_task: dict[str, list[str]] = {}
    task_metadata: dict[str, tuple[str, str, float]] = {}
    for sample in sample_ids:
        metadata = sample_metadata[sample]
        prior = task_metadata.setdefault(
            metadata.task_id,
            (metadata.context, metadata.compound, metadata.dose_um),
        )
        if prior != (metadata.context, metadata.compound, metadata.dose_um):
            raise GSE306429InputError(
                f"task {metadata.task_id!r} has conflicting metadata"
            )
        samples_by_task.setdefault(metadata.task_id, []).append(sample)

    task_ids = _utf8_order(samples_by_task)
    effects = np.zeros((len(task_ids), n_features), dtype=np.float64)
    task_sample_counts = np.zeros(len(task_ids), dtype=np.int64)
    task_cell_counts = np.zeros(len(task_ids), dtype=np.int64)
    sample_count_lookup = {
        sample: int(sample_counts[index]) for index, sample in enumerate(sample_ids)
    }
    sample_ids_by_task: list[tuple[str, ...]] = []
    contexts: list[str] = []
    compounds: list[str] = []
    doses: list[float] = []
    for task_index, task in enumerate(task_ids):
        samples = _utf8_order(samples_by_task[task])
        for sample in samples:
            effects[task_index] += sample_effects[sample]
        effects[task_index] /= len(samples)
        task_sample_counts[task_index] = len(samples)
        task_cell_counts[task_index] = sum(
            sample_count_lookup[sample] for sample in samples
        )
        context, compound, dose = task_metadata[task]
        contexts.append(context)
        compounds.append(compound)
        doses.append(dose)
        sample_ids_by_task.append(samples)

    if not np.isfinite(effects).all() or not np.isfinite(scales).all():
        raise GSE306429InputError("training inputs contain a nonfinite value")
    task_weights = np.full(len(task_ids), 1.0 / len(task_ids), dtype=np.float64)
    return FitTrainingInputs(
        fit_id=boundary.fit_id,
        training_replicate=boundary.training_replicate,
        evaluation_replicate=boundary.evaluation_replicate,
        heldout_task_fold=boundary.heldout_task_fold,
        feature_ids=features,
        task_ids=task_ids,
        contexts=tuple(contexts),
        compounds=tuple(compounds),
        doses_um=_frozen_array(np.asarray(doses, dtype=np.float64)),
        effects=_frozen_array(effects),
        task_weights=_frozen_array(task_weights),
        task_bio_sample_counts=_frozen_array(
            task_sample_counts, dtype=np.dtype(np.int64)
        ),
        task_cell_counts=_frozen_array(
            task_cell_counts, dtype=np.dtype(np.int64)
        ),
        feature_scales=_frozen_array(scales),
        training_dmso_cell_count=dmso_count,
        training_perturbation_cell_count=int(sample_counts.sum()),
        training_obs_sha256=training_obs_sha256,
        bio_sample_ids_by_task=tuple(sample_ids_by_task),
    )
