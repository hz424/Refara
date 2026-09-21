"""Production I/O primitives for the GSE306429 V2 materialization stage.

The functions in this module do not evaluate predictive performance. They parse
the frozen metadata axis, normalize admitted CSR row representations without
coalescing duplicate coordinates, build transformed atom/reference profiles,
load the deterministic internal DIRECT states, and serialize fixed-axis
``float32`` artifacts.  No function computes a metric, utility, contrast,
interval, winner, best set, or rank.

The file-aware entry points live in ``scripts/``. The numerical and
serialization primitives can be tested with synthetic fixtures before they are
applied to production states or predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
from scipy import sparse

from perturb_nuisance_model.gse306429_v2_baselines import (
    CONTEXT_MEAN,
    NO_CHANGE,
    PCA64_RIDGE,
    RBF_RIDGE,
    TWO_WAY_RIDGE,
    ContextMeanEffectDirectState,
    DirectBaselineState,
    NoChangeDirectState,
    PCA64AdditiveRidgeDirectState,
    RBFKernelRidgeDirectState,
    TwoWayAdditiveRidgeDirectState,
    canonical_state_manifest,
    canonical_state_sha256,
)
from perturb_nuisance_model.gse306429_v2_inputs import (
    CountsChunk,
    FitBoundary,
    transform_raw_counts_csr,
)
from perturb_nuisance_model.gse306429_v2_materialization import (
    BLOCK_IDS,
    FIT_IDS,
    REFERENCE_INSTANCE_IDS,
    FrozenReferenceCell,
    ReferenceMembership,
    build_reference_memberships,
)


CONTROL_ROLE = "DMSO_CONTROL"
PERTURBATION_ROLE = "PERTURBATION"
CONTROL_LABEL = "DMSO"
MAX_REFERENCE_DEPTH = 48
OUTPUT_FLOAT_DTYPE = np.dtype("<f4")
ACCUMULATION_DTYPE = np.dtype("<f8")
AXIS_SCHEMA = "gse306429_v2_evaluation_axis_v1"
PROFILE_SCHEMA = "gse306429_v2_evaluation_profile_bundle_v1"
DIRECT_BUNDLE_SCHEMA = "gse306429_v2_direct_prediction_bundle_v1"

_MISSING = frozenset({"", "na", "nan", "none", "null"})
_SHA256_HEX = frozenset("0123456789abcdef")


class GSE306429ProductionIOError(RuntimeError):
    """Raised when a production I/O invariant fails closed."""


def sha256_file(path: Path, *, block_size: int = 16 * 1024 * 1024) -> str:
    """Hash one regular file without loading it into memory."""

    resolved = Path(path)
    if not resolved.is_file() or resolved.is_symlink():
        raise GSE306429ProductionIOError(f"not a regular file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise GSE306429ProductionIOError(
            f"{label} is not a lowercase SHA-256 digest"
        )
    return value


def canonical_json_bytes(value: object) -> bytes:
    """Return the one canonical JSON representation used by this stage."""

    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _text(value: object, *, label: str, allow_empty: bool = False) -> str:
    if not isinstance(value, (str, np.str_)):
        raise GSE306429ProductionIOError(f"{label} must be text")
    result = str(value)
    if result != result.strip() or "\0" in result or "\n" in result:
        raise GSE306429ProductionIOError(f"{label} is not canonical text")
    if not allow_empty and result.casefold() in _MISSING:
        raise GSE306429ProductionIOError(f"{label} is missing")
    try:
        result.encode("utf-8")
    except UnicodeEncodeError as error:
        raise GSE306429ProductionIOError(f"{label} is not UTF-8") from error
    return result


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise GSE306429ProductionIOError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise GSE306429ProductionIOError(f"{label} must be an integer") from error
    if result < minimum:
        raise GSE306429ProductionIOError(f"{label} is below {minimum}")
    if not isinstance(value, (int, np.integer)) and str(result) != str(value):
        raise GSE306429ProductionIOError(f"{label} is not a canonical integer")
    return result


def _positive_float(value: object, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise GSE306429ProductionIOError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise GSE306429ProductionIOError(f"{label} must be numeric") from error
    if not math.isfinite(result) or result <= 0.0:
        raise GSE306429ProductionIOError(f"{label} must be finite and positive")
    return result


def _fit_ids(value: object, *, label: str) -> tuple[str, ...]:
    text = _text(value, label=label, allow_empty=True)
    if not text:
        return ()
    result = tuple(text.split(";"))
    if (
        any(item not in FIT_IDS for item in result)
        or len(result) != len(set(result))
    ):
        raise GSE306429ProductionIOError(f"{label} contains invalid fit IDs")
    return result


def _utf8_key(value: str) -> bytes:
    return value.encode("utf-8")


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values), key=_utf8_key))


def ordered_axis_sha256(
    values: Sequence[str], *, domain: bytes = b"GSE306429_V2_ORDERED_AXIS_V1\0"
) -> str:
    digest = hashlib.sha256(domain)
    for value in values:
        digest.update(_text(value, label="axis value").encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class EvaluationAtom:
    """One frozen bio-sample/plate atom on exactly one evaluation fit."""

    active_atom_position: int
    atom_id: str
    fit_id: str
    task_id: str
    library_id: str
    replicate: str
    context: str
    compound: str
    dose_um: float
    candidate_root_id: str
    dependence_cluster_id: str
    cell_positions: tuple[int, ...]
    cell_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.fit_id not in FIT_IDS:
            raise GSE306429ProductionIOError("atom fit_id is not frozen")
        boundary = FitBoundary.parse(self.fit_id)
        if self.replicate != boundary.evaluation_replicate:
            raise GSE306429ProductionIOError(
                "atom replicate differs from its evaluation fit"
            )
        for name in (
            "atom_id",
            "task_id",
            "library_id",
            "replicate",
            "context",
            "compound",
            "candidate_root_id",
            "dependence_cluster_id",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), label=f"atom {name}")
            )
        object.__setattr__(
            self,
            "active_atom_position",
            _integer(
                self.active_atom_position,
                label="active_atom_position",
                minimum=0,
            ),
        )
        object.__setattr__(
            self, "dose_um", _positive_float(self.dose_um, label="atom dose")
        )
        positions = tuple(
            _integer(value, label="atom cell position", minimum=0)
            for value in self.cell_positions
        )
        identifiers = tuple(
            _text(value, label="atom cell ID") for value in self.cell_ids
        )
        if (
            not positions
            or len(positions) != len(identifiers)
            or tuple(sorted(positions)) != positions
            or len(positions) != len(set(positions))
            or len(identifiers) != len(set(identifiers))
        ):
            raise GSE306429ProductionIOError(
                "atom cells must be nonempty, unique, aligned, and row ordered"
            )
        object.__setattr__(self, "cell_positions", positions)
        object.__setattr__(self, "cell_ids", identifiers)


@dataclass(frozen=True)
class EvaluationControl:
    """One selected DMSO reference cell (rank <= 48 within one block)."""

    control_id: str
    obs_position: int
    library_id: str
    replicate: str
    block_id: str
    block_rank: int
    evaluation_fit_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("control_id", "library_id", "replicate"):
            object.__setattr__(
                self, name, _text(getattr(self, name), label=f"control {name}")
            )
        object.__setattr__(
            self,
            "obs_position",
            _integer(self.obs_position, label="control obs_position", minimum=0),
        )
        if self.block_id not in BLOCK_IDS:
            raise GSE306429ProductionIOError("control block is not B1/B2/B3")
        rank = _integer(self.block_rank, label="control block rank", minimum=1)
        if rank > MAX_REFERENCE_DEPTH:
            raise GSE306429ProductionIOError(
                "selected control rank exceeds the frozen maximum"
            )
        object.__setattr__(self, "block_rank", rank)
        fit_ids = tuple(self.evaluation_fit_ids)
        if (
            not fit_ids
            or any(value not in FIT_IDS for value in fit_ids)
            or len(fit_ids) != len(set(fit_ids))
        ):
            raise GSE306429ProductionIOError(
                "control evaluation_fit_ids are invalid"
            )
        object.__setattr__(self, "evaluation_fit_ids", fit_ids)


@dataclass(frozen=True)
class EvaluationPlan:
    """Metadata-only plan fixing all rows, atoms, controls, roots, and axes."""

    row_count: int
    atoms: tuple[EvaluationAtom, ...]
    controls: tuple[EvaluationControl, ...]
    atoms_by_fit: Mapping[str, tuple[EvaluationAtom, ...]]
    controls_by_library: Mapping[str, tuple[EvaluationControl, ...]]
    root_ids_by_replicate: Mapping[str, tuple[str, ...]]
    ordered_obs_index_sha256: str
    atom_axis_sha256: str
    control_axis_sha256: str

    def __post_init__(self) -> None:
        if self.row_count < 1:
            raise GSE306429ProductionIOError("evaluation plan has no rows")
        if tuple(self.atoms_by_fit) != FIT_IDS:
            raise GSE306429ProductionIOError(
                "atoms_by_fit does not use frozen fit order"
            )
        if any(not self.atoms_by_fit[fit_id] for fit_id in FIT_IDS):
            raise GSE306429ProductionIOError("one frozen fit has no active atoms")
        require_sha256(
            self.ordered_obs_index_sha256, label="ordered_obs_index_sha256"
        )
        require_sha256(self.atom_axis_sha256, label="atom_axis_sha256")
        require_sha256(self.control_axis_sha256, label="control_axis_sha256")

    def reference_memberships_for_atom(
        self, atom: EvaluationAtom
    ) -> tuple[ReferenceMembership, ...]:
        controls = self.controls_by_library.get(atom.library_id)
        if controls is None:
            raise GSE306429ProductionIOError(
                f"atom {atom.atom_id} lacks same-library controls"
            )
        return build_reference_memberships(
            tuple(
                FrozenReferenceCell(
                    control_id=control.control_id,
                    library_id=control.library_id,
                    block_id=control.block_id,
                    block_rank=control.block_rank,
                )
                for control in controls
            )
        )


@dataclass
class _MutableAtom:
    position: int
    atom_id: str
    fit_id: str
    task_id: str
    library_id: str
    replicate: str
    context: str
    compound: str
    dose_um: float
    root_id: str
    cluster_id: str
    cell_positions: list[int]
    cell_ids: list[str]


_AXIS_REQUIRED = frozenset(
    {
        "obs_position",
        "obs_index",
        "bio_sample_id",
        "library_id",
        "plate",
        "context",
        "replicate",
        "compound_name",
        "dose_uM",
        "cell_role",
        "task_id",
        "task_fold_id",
        "is_active_atom",
        "active_atom_id",
        "active_atom_position",
        "candidate_root_id",
        "dependence_sensitivity_cluster_id",
        "evaluation_fit_ids",
        "reference_block",
        "reference_rank_within_block",
    }
)


def plan_evaluation_axis(
    rows: Iterable[Mapping[str, object]],
    *,
    expected_row_count: int | None = None,
) -> EvaluationPlan:
    """Parse the complete frozen cell axis without reading expression values."""

    atoms: dict[str, _MutableAtom] = {}
    controls: list[EvaluationControl] = []
    seen_obs: set[str] = set()
    obs_digest = hashlib.sha256()
    roots: dict[str, set[str]] = {}
    expected_position = 0

    for raw in rows:
        missing = _AXIS_REQUIRED.difference(raw)
        if missing:
            raise GSE306429ProductionIOError(
                f"cell-axis row lacks fields: {sorted(missing)}"
            )
        position = _integer(raw["obs_position"], label="obs_position", minimum=0)
        if position != expected_position:
            raise GSE306429ProductionIOError(
                "cell axis is not complete in source row order"
            )
        expected_position += 1
        obs_index = _text(raw["obs_index"], label="obs_index")
        if obs_index in seen_obs:
            raise GSE306429ProductionIOError("obs_index is not globally unique")
        seen_obs.add(obs_index)
        obs_digest.update(obs_index.encode("utf-8"))
        obs_digest.update(b"\n")

        role = _text(raw["cell_role"], label="cell_role")
        compound = _text(raw["compound_name"], label="compound_name")
        replicate = _text(raw["replicate"], label="replicate")
        root_id = _text(raw["candidate_root_id"], label="candidate_root_id")
        roots.setdefault(replicate, set()).add(root_id)
        evaluation_fits = _fit_ids(
            raw["evaluation_fit_ids"], label="evaluation_fit_ids"
        )
        active = _integer(
            raw["is_active_atom"], label="is_active_atom", minimum=0
        )
        if active not in {0, 1}:
            raise GSE306429ProductionIOError("is_active_atom must be zero or one")

        if role == CONTROL_ROLE:
            if compound != CONTROL_LABEL or active:
                raise GSE306429ProductionIOError(
                    "DMSO role, compound, and active flag conflict"
                )
            block = _text(
                raw["reference_block"], label="reference_block", allow_empty=True
            )
            rank_text = _text(
                raw["reference_rank_within_block"],
                label="reference_rank_within_block",
                allow_empty=True,
            )
            if not block or not rank_text:
                raise GSE306429ProductionIOError(
                    "a DMSO control lacks frozen block/rank"
                )
            rank = _integer(
                rank_text, label="reference_rank_within_block", minimum=1
            )
            if rank <= MAX_REFERENCE_DEPTH:
                controls.append(
                    EvaluationControl(
                        control_id=obs_index,
                        obs_position=position,
                        library_id=_text(raw["library_id"], label="library_id"),
                        replicate=replicate,
                        block_id=block,
                        block_rank=rank,
                        evaluation_fit_ids=evaluation_fits,
                    )
                )
            continue

        if role != PERTURBATION_ROLE or compound == CONTROL_LABEL:
            raise GSE306429ProductionIOError(
                "unknown or inconsistent perturbation cell role"
            )
        if not active:
            continue
        atom_id = _text(raw["active_atom_id"], label="active_atom_id")
        if atom_id != _text(raw["bio_sample_id"], label="bio_sample_id"):
            raise GSE306429ProductionIOError(
                "active_atom_id differs from bio_sample_id"
            )
        if len(evaluation_fits) != 1:
            raise GSE306429ProductionIOError(
                "an active atom row must belong to exactly one evaluation fit"
            )
        fit_id = evaluation_fits[0]
        boundary = FitBoundary.parse(fit_id)
        task_fold = _integer(raw["task_fold_id"], label="task_fold_id", minimum=0)
        if (
            replicate != boundary.evaluation_replicate
            or task_fold != boundary.heldout_task_fold
        ):
            raise GSE306429ProductionIOError(
                "active atom metadata conflicts with its fit boundary"
            )
        metadata = dict(
            position=_integer(
                raw["active_atom_position"],
                label="active_atom_position",
                minimum=0,
            ),
            atom_id=atom_id,
            fit_id=fit_id,
            task_id=_text(raw["task_id"], label="task_id"),
            library_id=_text(raw["library_id"], label="library_id"),
            replicate=replicate,
            context=_text(raw["context"], label="context"),
            compound=compound,
            dose_um=_positive_float(raw["dose_uM"], label="dose_uM"),
            root_id=root_id,
            cluster_id=_text(
                raw["dependence_sensitivity_cluster_id"],
                label="dependence_sensitivity_cluster_id",
            ),
        )
        current = atoms.get(atom_id)
        if current is None:
            current = _MutableAtom(
                **metadata,
                cell_positions=[],
                cell_ids=[],
            )
            atoms[atom_id] = current
        else:
            observed = (
                current.position,
                current.atom_id,
                current.fit_id,
                current.task_id,
                current.library_id,
                current.replicate,
                current.context,
                current.compound,
                current.dose_um,
                current.root_id,
                current.cluster_id,
            )
            expected = (
                metadata["position"],
                metadata["atom_id"],
                metadata["fit_id"],
                metadata["task_id"],
                metadata["library_id"],
                metadata["replicate"],
                metadata["context"],
                metadata["compound"],
                metadata["dose_um"],
                metadata["root_id"],
                metadata["cluster_id"],
            )
            if observed != expected:
                raise GSE306429ProductionIOError(
                    f"atom {atom_id} has conflicting metadata"
                )
        current.cell_positions.append(position)
        current.cell_ids.append(obs_index)

    if expected_row_count is not None and expected_position != expected_row_count:
        raise GSE306429ProductionIOError(
            "cell-axis row count differs from the frozen expectation"
        )
    if expected_position == 0 or not atoms or not controls:
        raise GSE306429ProductionIOError(
            "cell axis has no rows, active atoms, or selected controls"
        )

    atom_records = tuple(
        EvaluationAtom(
            active_atom_position=value.position,
            atom_id=value.atom_id,
            fit_id=value.fit_id,
            task_id=value.task_id,
            library_id=value.library_id,
            replicate=value.replicate,
            context=value.context,
            compound=value.compound,
            dose_um=value.dose_um,
            candidate_root_id=value.root_id,
            dependence_cluster_id=value.cluster_id,
            cell_positions=tuple(value.cell_positions),
            cell_ids=tuple(value.cell_ids),
        )
        for value in sorted(atoms.values(), key=lambda value: value.position)
    )
    if tuple(atom.active_atom_position for atom in atom_records) != tuple(
        range(len(atom_records))
    ):
        raise GSE306429ProductionIOError(
            "active_atom_position is not contiguous from zero"
        )
    lexical = tuple(
        sorted(
            atom_records,
            key=lambda atom: (
                _utf8_key(atom.task_id),
                _utf8_key(atom.candidate_root_id),
                _utf8_key(atom.atom_id),
            ),
        )
    )
    if lexical != atom_records:
        raise GSE306429ProductionIOError(
            "active atom order differs from task/root/bio-sample UTF-8 order"
        )

    if len({control.control_id for control in controls}) != len(controls):
        raise GSE306429ProductionIOError("selected control IDs are duplicated")
    controls_by_library_unsorted: dict[str, list[EvaluationControl]] = {}
    for control in controls:
        controls_by_library_unsorted.setdefault(control.library_id, []).append(control)
    controls_by_library: dict[str, tuple[EvaluationControl, ...]] = {}
    required_libraries = {atom.library_id for atom in atom_records}
    missing_libraries = required_libraries.difference(controls_by_library_unsorted)
    if missing_libraries:
        raise GSE306429ProductionIOError(
            f"active atom libraries lack controls: {sorted(missing_libraries)}"
        )
    for library_id in sorted(required_libraries, key=_utf8_key):
        values = tuple(
            sorted(
                controls_by_library_unsorted[library_id],
                key=lambda item: (BLOCK_IDS.index(item.block_id), item.block_rank),
            )
        )
        replicates = {value.replicate for value in values}
        if len(replicates) != 1:
            raise GSE306429ProductionIOError(
                f"library {library_id} controls span replicates"
            )
        for block in BLOCK_IDS:
            ranks = tuple(
                value.block_rank for value in values if value.block_id == block
            )
            if ranks != tuple(range(1, MAX_REFERENCE_DEPTH + 1)):
                raise GSE306429ProductionIOError(
                    f"library {library_id}/{block} lacks exact ranks 1..48"
                )
        controls_by_library[library_id] = values
    selected_controls = tuple(
        control
        for library_id in controls_by_library
        for control in controls_by_library[library_id]
    )

    atoms_by_fit = {
        fit_id: tuple(atom for atom in atom_records if atom.fit_id == fit_id)
        for fit_id in FIT_IDS
    }
    for fit_id, fit_atoms in atoms_by_fit.items():
        for atom in fit_atoms:
            for control in controls_by_library[atom.library_id]:
                if fit_id not in control.evaluation_fit_ids:
                    raise GSE306429ProductionIOError(
                        f"{atom.atom_id} reference control is absent from {fit_id}"
                    )
    root_map = {
        replicate: _ordered_unique(values)
        for replicate, values in sorted(roots.items(), key=lambda item: _utf8_key(item[0]))
    }
    atom_axis_digest = ordered_axis_sha256(
        tuple(atom.atom_id for atom in atom_records),
        domain=b"GSE306429_V2_ACTIVE_ATOM_AXIS_V1\0",
    )
    control_axis_digest = ordered_axis_sha256(
        tuple(control.control_id for control in selected_controls),
        domain=b"GSE306429_V2_SELECTED_CONTROL_AXIS_V1\0",
    )
    return EvaluationPlan(
        row_count=expected_position,
        atoms=atom_records,
        controls=selected_controls,
        atoms_by_fit=atoms_by_fit,
        controls_by_library=controls_by_library,
        root_ids_by_replicate=root_map,
        ordered_obs_index_sha256=obs_digest.hexdigest(),
        atom_axis_sha256=atom_axis_digest,
        control_axis_sha256=control_axis_digest,
    )


def open_cell_axis_rows(path: Path) -> Iterator[dict[str, str]]:
    """Yield one frozen TSV row without retaining the complete file."""

    resolved = Path(path)
    opener = gzip.open if resolved.suffix == ".gz" else open
    with opener(resolved, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise GSE306429ProductionIOError("cell-axis TSV has no header")
        for row in reader:
            yield dict(row)


def sort_unique_csr_without_coalescing(
    counts: sparse.spmatrix,
) -> sparse.csr_matrix:
    """Return a sorted CSR after proving each source row is duplicate-free.

    ``sum_duplicates`` is never called.  Duplicate coordinates are rejected
    before sorting, so source count values and ``nnz`` are preserved exactly.
    """

    if not sparse.isspmatrix_csr(counts) or counts.ndim != 2:
        raise GSE306429ProductionIOError("counts must be a two-dimensional CSR")
    source = counts
    if source.shape[0] < 1 or source.shape[1] < 1:
        raise GSE306429ProductionIOError("counts CSR is empty")
    data = np.asarray(source.data)
    indices = np.asarray(source.indices)
    indptr = np.asarray(source.indptr, dtype=np.int64)
    if (
        indptr.shape != (source.shape[0] + 1,)
        or indptr[0] != 0
        or indptr[-1] != len(data)
        or np.any(indptr[1:] < indptr[:-1])
    ):
        raise GSE306429ProductionIOError("counts CSR pointers are malformed")
    if np.any(indices < 0) or np.any(indices >= source.shape[1]):
        raise GSE306429ProductionIOError("counts CSR index is out of range")
    if data.dtype.kind not in {"i", "u", "f"} or data.dtype.itemsize not in {
        1,
        2,
        4,
        8,
    }:
        raise GSE306429ProductionIOError(
            "counts CSR data type is not a supported real numeric type"
        )

    # ``csr_matrix.sort_indices`` is a C-level, within-row permutation.  It
    # does not call ``sum_duplicates``.  Keep source arrays immutable by
    # sorting a private copy, and bind the data multiset before/after with two
    # commutative bit-pattern fingerprints in addition to exact nnz/indptr.
    unsigned_dtype = np.dtype(f"<u{data.dtype.itemsize}")

    def data_multiset_fingerprint(values: np.ndarray) -> tuple[int, int]:
        bits = np.ascontiguousarray(values).view(unsigned_dtype).astype(
            np.uint64, copy=False
        )
        if bits.size == 0:
            return (0, 0)
        xor = int(np.bitwise_xor.reduce(bits, dtype=np.uint64))
        mixed = bits ^ (bits >> np.uint64(30))
        mixed *= np.uint64(0xBF58476D1CE4E5B9)
        mixed ^= mixed >> np.uint64(27)
        mixed *= np.uint64(0x94D049BB133111EB)
        mixed ^= mixed >> np.uint64(31)
        total = int(np.add.reduce(mixed, dtype=np.uint64))
        return xor, total

    before_fingerprint = data_multiset_fingerprint(data)
    result = sparse.csr_matrix(
        (data.copy(), indices.copy(), indptr.copy()),
        shape=source.shape,
        copy=False,
    )
    before_nnz = int(result.nnz)
    result.sort_indices()
    if (
        int(result.nnz) != before_nnz
        or not np.array_equal(result.indptr, indptr)
        or data_multiset_fingerprint(result.data) != before_fingerprint
    ):
        raise GSE306429ProductionIOError(
            "CSR index sorting changed nnz, row pointers, or data multiset"
        )

    # Duplicate detection is vectorized over the complete chunk.  Adjacent
    # entries that straddle a row boundary are explicitly excluded.
    if result.indices.size > 1:
        within_row = np.ones(result.indices.size - 1, dtype=bool)
        boundaries = result.indptr[1:-1] - 1
        boundaries = boundaries[
            (boundaries >= 0) & (boundaries < within_row.size)
        ]
        within_row[boundaries] = False
        if np.any(
            within_row
            & (result.indices[1:] <= result.indices[:-1])
        ):
            raise GSE306429ProductionIOError(
                "duplicate CSR feature coordinates are prohibited"
            )
    if result.nnz != source.nnz or not result.has_canonical_format:
        raise GSE306429ProductionIOError(
            "CSR normalization changed nnz or failed canonicality"
        )
    return result


@dataclass(frozen=True)
class EvaluationProfiles:
    """Transformed atom means and the 144 selected controls per library."""

    atom_ids: tuple[str, ...]
    control_ids: tuple[str, ...]
    atom_means: np.ndarray
    control_profiles: np.ndarray
    atom_cell_counts: np.ndarray
    feature_count: int

    def __post_init__(self) -> None:
        atoms = tuple(self.atom_ids)
        controls = tuple(self.control_ids)
        atom_means = np.ascontiguousarray(self.atom_means, dtype="<f8")
        control_profiles = np.ascontiguousarray(self.control_profiles, dtype="<f8")
        counts = np.ascontiguousarray(self.atom_cell_counts, dtype="<i8")
        if (
            atom_means.shape != (len(atoms), self.feature_count)
            or control_profiles.shape != (len(controls), self.feature_count)
            or counts.shape != (len(atoms),)
            or np.any(counts <= 0)
            or not np.isfinite(atom_means).all()
            or not np.isfinite(control_profiles).all()
        ):
            raise GSE306429ProductionIOError(
                "evaluation-profile axes, counts, or values are invalid"
            )
        for value in (atom_means, control_profiles, counts):
            value.setflags(write=False)
        object.__setattr__(self, "atom_means", atom_means)
        object.__setattr__(self, "control_profiles", control_profiles)
        object.__setattr__(self, "atom_cell_counts", counts)


def _kahan_add(
    total: np.ndarray, compensation: np.ndarray, increment: np.ndarray
) -> None:
    corrected = increment - compensation
    updated = total + corrected
    compensation[...] = (updated - total) - corrected
    total[...] = updated


def accumulate_evaluation_profiles(
    plan: EvaluationPlan,
    chunks: Iterable[CountsChunk | tuple[int, sparse.spmatrix]],
    *,
    feature_count: int,
) -> EvaluationProfiles:
    """Transform and aggregate a complete admitted CSR stream exactly once."""

    if feature_count < 1:
        raise GSE306429ProductionIOError("feature_count must be positive")
    atom_codes = np.full(plan.row_count, -1, dtype=np.int64)
    for code, atom in enumerate(plan.atoms):
        atom_codes[np.asarray(atom.cell_positions, dtype=np.int64)] = code
    control_codes = np.full(plan.row_count, -1, dtype=np.int64)
    for code, control in enumerate(plan.controls):
        if control_codes[control.obs_position] != -1:
            raise GSE306429ProductionIOError("control position is duplicated")
        control_codes[control.obs_position] = code

    atom_sums = np.zeros((len(plan.atoms), feature_count), dtype=np.float64)
    atom_comp = np.zeros_like(atom_sums)
    atom_counts = np.zeros(len(plan.atoms), dtype=np.int64)
    control_profiles = np.full(
        (len(plan.controls), feature_count), np.nan, dtype=np.float64
    )
    next_row = 0
    saw_chunk = False
    for raw_chunk in chunks:
        if isinstance(raw_chunk, CountsChunk):
            row_start = raw_chunk.row_start
            matrix = raw_chunk.counts
        elif isinstance(raw_chunk, tuple) and len(raw_chunk) == 2:
            row_start = _integer(raw_chunk[0], label="chunk row_start", minimum=0)
            matrix = raw_chunk[1]
        else:
            raise GSE306429ProductionIOError("invalid counts chunk")
        if row_start != next_row:
            raise GSE306429ProductionIOError(
                "count chunks are not contiguous in source row order"
            )
        normalized = sort_unique_csr_without_coalescing(matrix)
        transformed = transform_raw_counts_csr(normalized)
        if transformed.shape[1] != feature_count:
            raise GSE306429ProductionIOError(
                "counts chunk feature count differs from the frozen axis"
            )
        stop = row_start + transformed.shape[0]
        if stop > plan.row_count:
            raise GSE306429ProductionIOError("counts stream exceeds cell axis")
        local_atom_codes = atom_codes[row_start:stop]
        local_control_codes = control_codes[row_start:stop]
        for code in np.unique(local_atom_codes[local_atom_codes >= 0]):
            rows = np.flatnonzero(local_atom_codes == code)
            increment = np.asarray(
                transformed[rows].sum(axis=0), dtype=np.float64
            ).reshape(-1)
            _kahan_add(atom_sums[code], atom_comp[code], increment)
            atom_counts[code] += len(rows)
        control_rows = np.flatnonzero(local_control_codes >= 0)
        if len(control_rows):
            dense = np.asarray(
                transformed[control_rows].toarray(), dtype=np.float64
            )
            codes = local_control_codes[control_rows]
            if np.any(np.isfinite(control_profiles[codes])):
                raise GSE306429ProductionIOError(
                    "a selected control row was filled more than once"
                )
            control_profiles[codes] = dense
        next_row = stop
        saw_chunk = True
    if not saw_chunk or next_row != plan.row_count:
        raise GSE306429ProductionIOError(
            "counts stream does not cover the complete cell axis"
        )
    if np.any(atom_counts <= 0) or not np.isfinite(control_profiles).all():
        raise GSE306429ProductionIOError(
            "an active atom or selected reference cell was not observed"
        )
    means = atom_sums / atom_counts[:, None]
    return EvaluationProfiles(
        atom_ids=tuple(atom.atom_id for atom in plan.atoms),
        control_ids=tuple(control.control_id for control in plan.controls),
        atom_means=means,
        control_profiles=control_profiles,
        atom_cell_counts=atom_counts,
        feature_count=feature_count,
    )


@dataclass(frozen=True)
class ObservedReferenceMatrices:
    """One fit/reference observed profile bundle, before any metric."""

    fit_id: str
    reference_instance_id: str
    atom_ids: tuple[str, ...]
    atom_profiles: np.ndarray
    c_obs_means: np.ndarray
    observed_effects: np.ndarray
    c_obs_membership_sha256: str

    def __post_init__(self) -> None:
        if self.fit_id not in FIT_IDS:
            raise GSE306429ProductionIOError("observed bundle fit is invalid")
        if self.reference_instance_id not in REFERENCE_INSTANCE_IDS:
            raise GSE306429ProductionIOError(
                "observed bundle reference instance is invalid"
            )
        expected_rows = len(self.atom_ids)
        shapes = {
            np.asarray(self.atom_profiles).shape,
            np.asarray(self.c_obs_means).shape,
            np.asarray(self.observed_effects).shape,
        }
        if len(shapes) != 1:
            raise GSE306429ProductionIOError(
                "observed bundle matrix shapes differ"
            )
        shape = next(iter(shapes))
        if len(shape) != 2 or shape[0] != expected_rows or shape[1] < 1:
            raise GSE306429ProductionIOError("observed bundle shape is invalid")
        for name in ("atom_profiles", "c_obs_means", "observed_effects"):
            value = np.ascontiguousarray(getattr(self, name), dtype="<f4")
            if not np.isfinite(value).all():
                raise GSE306429ProductionIOError(
                    f"observed bundle {name} is nonfinite"
                )
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        require_sha256(
            self.c_obs_membership_sha256, label="c_obs_membership_sha256"
        )
        reconstructed = np.subtract(
            self.atom_profiles, self.c_obs_means, dtype=np.float32
        )
        if not np.array_equal(reconstructed, self.observed_effects):
            raise GSE306429ProductionIOError(
                "observed effect is not exactly float32 atom minus C_obs"
            )


def _membership_digest(
    atom_ids: Sequence[str], memberships: Sequence[Sequence[str]], *, role: str
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "atoms": [
                    [atom_id, list(cell_ids)]
                    for atom_id, cell_ids in zip(
                        atom_ids, memberships, strict=True
                    )
                ],
                "role": role,
                "schema": "gse306429_v2_reference_membership_v1",
            }
        )
    ).hexdigest()


def observed_reference_membership_sha256(
    plan: EvaluationPlan, *, fit_id: str, reference_instance_id: str
) -> str:
    """Recompute the exact C_obs membership digest from metadata only."""

    if fit_id not in FIT_IDS or reference_instance_id not in REFERENCE_INSTANCE_IDS:
        raise GSE306429ProductionIOError("fit/reference instance is not frozen")
    atoms = plan.atoms_by_fit[fit_id]
    memberships: list[tuple[str, ...]] = []
    for atom in atoms:
        membership = next(
            item
            for item in plan.reference_memberships_for_atom(atom)
            if item.reference_instance_id == reference_instance_id
        )
        memberships.append(membership.c_obs_ids)
    return _membership_digest(
        tuple(atom.atom_id for atom in atoms), memberships, role="C_obs"
    )


def observed_reference_matrices(
    plan: EvaluationPlan,
    profiles: EvaluationProfiles,
    *,
    fit_id: str,
    reference_instance_id: str,
) -> ObservedReferenceMatrices:
    """Construct atom profiles, C_obs means, and effects without scoring."""

    if fit_id not in FIT_IDS or reference_instance_id not in REFERENCE_INSTANCE_IDS:
        raise GSE306429ProductionIOError("fit/reference instance is not frozen")
    atoms = plan.atoms_by_fit[fit_id]
    global_atom_index = {
        atom_id: index for index, atom_id in enumerate(profiles.atom_ids)
    }
    control_index = {
        control_id: index for index, control_id in enumerate(profiles.control_ids)
    }
    atom_rows: list[np.ndarray] = []
    reference_rows: list[np.ndarray] = []
    memberships: list[tuple[str, ...]] = []
    for atom in atoms:
        try:
            atom_rows.append(profiles.atom_means[global_atom_index[atom.atom_id]])
        except KeyError as error:
            raise GSE306429ProductionIOError(
                f"profiles lack atom {atom.atom_id}"
            ) from error
        membership = next(
            item
            for item in plan.reference_memberships_for_atom(atom)
            if item.reference_instance_id == reference_instance_id
        )
        ids = membership.c_obs_ids
        try:
            indices = np.asarray([control_index[value] for value in ids])
        except KeyError as error:
            raise GSE306429ProductionIOError(
                f"profiles lack selected control {error.args[0]}"
            ) from error
        reference_rows.append(
            profiles.control_profiles[indices].mean(axis=0, dtype=np.float64)
        )
        memberships.append(ids)
    atom_matrix = np.ascontiguousarray(np.vstack(atom_rows), dtype="<f4")
    reference_matrix = np.ascontiguousarray(
        np.vstack(reference_rows), dtype="<f4"
    )
    effects = np.subtract(atom_matrix, reference_matrix, dtype=np.float32)
    atom_ids = tuple(atom.atom_id for atom in atoms)
    return ObservedReferenceMatrices(
        fit_id=fit_id,
        reference_instance_id=reference_instance_id,
        atom_ids=atom_ids,
        atom_profiles=atom_matrix,
        c_obs_means=reference_matrix,
        observed_effects=effects,
        c_obs_membership_sha256=_membership_digest(
            atom_ids, memberships, role="C_obs"
        ),
    )


def _load_npy_record(directory: Path, record: Mapping[str, Any]) -> np.ndarray:
    required = {"dtype", "file", "file_sha256", "shape"}
    if set(record) != required:
        raise GSE306429ProductionIOError("state array record fields differ")
    name = _text(record["file"], label="state array filename")
    if Path(name).name != name or name.startswith("."):
        raise GSE306429ProductionIOError("state array filename is unsafe")
    path = directory / name
    if sha256_file(path) != require_sha256(
        record["file_sha256"], label="state array SHA-256"
    ):
        raise GSE306429ProductionIOError(f"state array hash differs: {name}")
    array = np.load(path, allow_pickle=False)
    expected_dtype = np.dtype(_text(record["dtype"], label="state array dtype"))
    expected_shape = tuple(
        _integer(value, label="state array shape", minimum=0)
        for value in record["shape"]
    )
    if array.dtype != expected_dtype or array.shape != expected_shape:
        raise GSE306429ProductionIOError(
            f"state array axis or dtype differs: {name}"
        )
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise GSE306429ProductionIOError(f"state array is nonfinite: {name}")
    return np.ascontiguousarray(array)


def load_internal_direct_state(
    manifest_path: Path,
    *,
    expected_method_id: str | None = None,
    expected_fit_id: str | None = None,
) -> tuple[DirectBaselineState, dict[str, Any], str]:
    """Load and independently reconstruct one internal state artifact."""

    resolved = Path(manifest_path)
    manifest_bytes = resolved.read_bytes()
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GSE306429ProductionIOError(
            "internal state manifest is not valid JSON"
        ) from error
    required = {
        "array_files",
        "canonical_state_manifest",
        "canonical_state_sha256",
        "fit_id",
        "method_id",
        "prediction_materialized",
        "profile_manifest_sha256",
        "score_or_rank_materialized",
    }
    if set(manifest) != required:
        raise GSE306429ProductionIOError(
            "internal state manifest fields differ from the frozen schema"
        )
    method_id = _text(manifest["method_id"], label="state method_id")
    fit_id = manifest["fit_id"]
    if expected_method_id is not None and method_id != expected_method_id:
        raise GSE306429ProductionIOError("state method differs from expectation")
    if expected_fit_id is not None and fit_id != expected_fit_id:
        raise GSE306429ProductionIOError("state fit differs from expectation")
    if manifest["prediction_materialized"] or manifest["score_or_rank_materialized"]:
        raise GSE306429ProductionIOError(
            "state artifact crossed its state-only boundary"
        )
    canonical = manifest["canonical_state_manifest"]
    if canonical.get("schema") != "gse306429_internal_direct_state_v1":
        raise GSE306429ProductionIOError("canonical state schema differs")
    metadata = canonical.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("method_id") != method_id:
        raise GSE306429ProductionIOError("canonical state metadata differs")
    array_files = manifest["array_files"]
    if not isinstance(array_files, Mapping):
        raise GSE306429ProductionIOError("state array_files is not a mapping")
    arrays = {
        name: _load_npy_record(resolved.parent, record)
        for name, record in array_files.items()
    }
    common = {
        "feature_ids": tuple(metadata["feature_ids"]),
        "context_ids": tuple(metadata["context_ids"]),
        "compound_ids": tuple(metadata["compound_ids"]),
    }
    if method_id == NO_CHANGE:
        if arrays or fit_id is not None:
            raise GSE306429ProductionIOError("NO_CHANGE state is not NO_FIT")
        state: DirectBaselineState = NoChangeDirectState(**common)
    elif method_id == CONTEXT_MEAN:
        state = ContextMeanEffectDirectState(
            **common,
            training_task_ids=tuple(metadata["training_task_ids"]),
            context_effects=arrays["context_effects"],
        )
    elif method_id == TWO_WAY_RIDGE:
        state = TwoWayAdditiveRidgeDirectState(
            **common,
            training_task_ids=tuple(metadata["training_task_ids"]),
            compound_dose_ids=tuple(metadata["compound_dose_ids"]),
            intercept=arrays["intercept"],
            coefficients=arrays["coefficients"],
            ridge_lambda=float(metadata["ridge_lambda"]),
        )
    elif method_id == PCA64_RIDGE:
        state = PCA64AdditiveRidgeDirectState(
            **common,
            training_task_ids=tuple(metadata["training_task_ids"]),
            compound_dose_ids=tuple(metadata["compound_dose_ids"]),
            feature_mean=arrays["feature_mean"],
            components=arrays["components"],
            latent_intercept=arrays["latent_intercept"],
            latent_coefficients=arrays["latent_coefficients"],
            ridge_lambda=float(metadata["ridge_lambda"]),
            max_rank=int(metadata["max_rank"]),
        )
    elif method_id == RBF_RIDGE:
        state = RBFKernelRidgeDirectState(
            **common,
            training_task_ids=tuple(metadata["training_task_ids"]),
            log10_dose_mean=float(metadata["log10_dose_mean"]),
            log10_dose_sample_sd=float(metadata["log10_dose_sample_sd"]),
            training_inputs=arrays["training_inputs"],
            dual_coefficients=arrays["dual_coefficients"],
            ridge_lambda=float(metadata["ridge_lambda"]),
            gamma_factor=float(metadata["gamma_factor"]),
        )
    else:
        raise GSE306429ProductionIOError(
            f"unsupported internal state method: {method_id}"
        )
    if canonical_state_manifest(state) != canonical:
        raise GSE306429ProductionIOError(
            "reconstructed canonical state manifest differs"
        )
    state_sha = canonical_state_sha256(state)
    if state_sha != require_sha256(
        manifest["canonical_state_sha256"], label="canonical state SHA-256"
    ):
        raise GSE306429ProductionIOError(
            "reconstructed canonical state hash differs"
        )
    return state, manifest, hashlib.sha256(manifest_bytes).hexdigest()


def write_npy_artifact(path: Path, values: object) -> dict[str, Any]:
    """Write one little-endian float32 NPY payload without overwriting."""

    resolved = Path(path)
    if resolved.exists() or resolved.is_symlink():
        raise GSE306429ProductionIOError(f"refusing to overwrite {resolved}")
    array = np.ascontiguousarray(values, dtype=OUTPUT_FLOAT_DTYPE)
    if array.ndim not in {1, 2} or 0 in array.shape or not np.isfinite(array).all():
        raise GSE306429ProductionIOError("output array is empty or nonfinite")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("xb") as handle:
        np.lib.format.write_array(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "path": resolved.name,
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    }


def read_npy_artifact(
    path: Path, record: Mapping[str, Any], *, mmap: bool = True
) -> np.ndarray:
    """Verify and load one fixed-axis float32 NPY artifact."""

    resolved = Path(path)
    if sha256_file(resolved) != require_sha256(
        record.get("sha256"), label="NPY artifact SHA-256"
    ):
        raise GSE306429ProductionIOError("NPY artifact hash differs")
    array = np.load(
        resolved, mmap_mode="r" if mmap else None, allow_pickle=False
    )
    if (
        array.dtype != OUTPUT_FLOAT_DTYPE
        or list(array.shape) != record.get("shape")
        or record.get("dtype") != "<f4"
        or not np.isfinite(array).all()
    ):
        raise GSE306429ProductionIOError(
            "NPY artifact shape, dtype, or finiteness differs"
        )
    return array


def deterministic_gzip_tsv(
    header: Sequence[str], rows: Iterable[Sequence[object]]
) -> bytes:
    """Serialize a UTF-8 TSV with deterministic gzip metadata."""

    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as binary:
        with io.TextIOWrapper(binary, encoding="utf-8", newline="") as text:
            writer = csv.writer(text, delimiter="\t", lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
    return raw.getvalue()


def atomic_write_json(path: Path, value: object) -> str:
    """Create one canonical JSON file atomically, never replacing a prior file."""

    resolved = Path(path)
    if resolved.exists() or resolved.is_symlink():
        raise GSE306429ProductionIOError(f"refusing to overwrite {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", dir=resolved.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, resolved)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "ACCUMULATION_DTYPE",
    "AXIS_SCHEMA",
    "CONTROL_LABEL",
    "CONTROL_ROLE",
    "DIRECT_BUNDLE_SCHEMA",
    "EvaluationAtom",
    "EvaluationControl",
    "EvaluationPlan",
    "EvaluationProfiles",
    "GSE306429ProductionIOError",
    "MAX_REFERENCE_DEPTH",
    "OUTPUT_FLOAT_DTYPE",
    "ObservedReferenceMatrices",
    "PROFILE_SCHEMA",
    "PERTURBATION_ROLE",
    "accumulate_evaluation_profiles",
    "atomic_write_json",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "deterministic_gzip_tsv",
    "load_internal_direct_state",
    "open_cell_axis_rows",
    "ordered_axis_sha256",
    "plan_evaluation_axis",
    "read_npy_artifact",
    "require_sha256",
    "sha256_file",
    "sort_unique_csr_without_coalescing",
    "observed_reference_matrices",
    "write_npy_artifact",
]
