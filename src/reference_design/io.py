"""Read labelled expression tables and verify reference membership."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


BLOCKS = ("B1", "B2", "B3")
Key = tuple[str, int, str]


@dataclass
class CellTable:
    genes: tuple[str, ...]
    values: np.ndarray
    cell_ids: tuple[str, ...]
    strata: tuple[str, ...]


@dataclass
class BlockTable:
    genes: tuple[str, ...]
    values: dict[Key, np.ndarray]
    counts: dict[Key, int] | None

    @property
    def groups(self) -> list[tuple[str, int]]:
        return sorted({key[:2] for key in self.values})


def label(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be a nonempty string without surrounding whitespace")
    return value


def integer(value: str, field: str, minimum: int = 0) -> int:
    if not isinstance(value, str) or not value.isdecimal() or int(value) < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return int(value)


def table(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        headers = tuple(reader.fieldnames or ())
        if not headers or len(set(headers)) != len(headers):
            raise ValueError(f"{path}: missing or duplicate column names")
        for header in headers:
            label(header, f"Column in {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: table is empty")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{path}: row length differs from its header")
    return headers, rows


def _columns(headers: tuple[str, ...], required: set[str], optional: set[str], path: Path) -> tuple[str, ...]:
    missing = required - set(headers)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    genes = tuple(header for header in headers if header not in required | optional)
    if not genes:
        raise ValueError(f"{path}: no gene columns")
    return genes


def _numbers(rows: list[dict[str, str]], columns: tuple[str, ...], path: Path) -> np.ndarray:
    try:
        values = np.asarray([[float(row[column]) for column in columns] for row in rows], dtype=float)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{path}: expression values must be numbers") from error
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: expression values must be finite")
    return values


def align(values: np.ndarray, genes: tuple[str, ...], expected: tuple[str, ...], path: Path) -> np.ndarray:
    if set(genes) != set(expected):
        missing, extra = sorted(set(expected) - set(genes)), sorted(set(genes) - set(expected))
        raise ValueError(f"{path}: gene set differs; missing={missing[:5]}, extra={extra[:5]}")
    order = {gene: i for i, gene in enumerate(genes)}
    return values[..., [order[gene] for gene in expected]]


def read_vector(path: Path, genes: tuple[str, ...] | None = None) -> tuple[tuple[str, ...], np.ndarray]:
    headers, rows = table(path)
    if set(headers) != {"gene", "value"}:
        raise ValueError(f"{path}: a vector needs exactly gene and value columns")
    labels = tuple(label(row["gene"], f"Gene in {path}") for row in rows)
    if len(set(labels)) != len(labels):
        raise ValueError(f"{path}: duplicate gene identifiers")
    values = _numbers(rows, ("value",), path)[:, 0]
    if genes is not None:
        values = align(values, labels, genes, path)
        labels = genes
    return labels, values


def read_cells(path: Path, genes: tuple[str, ...] | None = None) -> CellTable:
    headers, rows = table(path)
    columns = _columns(headers, {"cell_id"}, {"stratum"}, path)
    ids = tuple(label(row["cell_id"], f"Cell ID in {path}") for row in rows)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate cell identifiers")
    strata = tuple(label(row.get("stratum", "all"), f"Stratum in {path}") for row in rows)
    values = _numbers(rows, columns, path)
    if genes is not None:
        values = align(values, columns, genes, path)
        columns = genes
    return CellTable(columns, values, ids, strata)


def _key(row: dict[str, str], path: Path) -> Key:
    allocation = label(row["allocation"], f"Allocation in {path}")
    depth = integer(row["depth"], f"Depth in {path}", 1)
    block = row["block"]
    if block not in BLOCKS:
        raise ValueError(f"{path}: block must be B1, B2 or B3")
    return allocation, depth, block


def read_blocks(path: Path, genes: tuple[str, ...]) -> BlockTable:
    headers, rows = table(path)
    columns = _columns(headers, {"allocation", "depth", "block"}, {"cells_per_block"}, path)
    values = align(_numbers(rows, columns, path), columns, genes, path)
    keys = [_key(row, path) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path}: duplicate allocation/depth/block rows")
    groups = {key[:2] for key in keys}
    if set(keys) != {(*group, block) for group in groups for block in BLOCKS}:
        raise ValueError(f"{path}: each allocation/depth needs all three blocks")
    counts = None
    if "cells_per_block" in headers:
        counts = {key: integer(row["cells_per_block"], f"Cell count in {path}", 1)
                  for key, row in zip(keys, rows)}
        for allocation, depth in groups:
            group_counts = {counts[(allocation, depth, block)] for block in BLOCKS}
            if len(group_counts) != 1 or next(iter(group_counts)) % depth:
                raise ValueError(f"{path}: block sizes must match and be multiples of stratum depth")
    return BlockTable(genes, dict(zip(keys, values)), counts)


def match_blocks(actual: BlockTable, expected: BlockTable, path: Path) -> None:
    if set(actual.values) != set(expected.values):
        raise ValueError(f"{path}: allocation/depth/block keys differ from the references")
    if actual.counts is not None and expected.counts is not None and actual.counts != expected.counts:
        raise ValueError(f"{path}: cell counts differ from the references")


def verify_membership(cells: CellTable, references: BlockTable, path: Path) -> None:
    headers, rows = table(path)
    required = {"allocation", "depth", "cell_id", "stratum", "block", "within_block_index"}
    if set(headers) != required:
        raise ValueError(f"{path}: membership columns must be {sorted(required)}")
    by_id = {cell_id: i for i, cell_id in enumerate(cells.cell_ids)}
    grouped: dict[Key, list[tuple[int, str, int]]] = {}
    used: dict[tuple[str, int], set[str]] = {}
    for row in rows:
        key = _key(row, path)
        if key not in references.values:
            raise ValueError(f"{path}: membership has an unknown allocation/depth/block")
        cell_id = label(row["cell_id"], f"Cell ID in {path}")
        stratum = label(row["stratum"], f"Stratum in {path}")
        if cell_id not in by_id:
            raise ValueError(f"{path}: unknown control cell {cell_id!r}")
        index = by_id[cell_id]
        if stratum != cells.strata[index]:
            raise ValueError(f"{path}: stratum differs for cell {cell_id!r}")
        within = integer(row["within_block_index"], f"Within-block index in {path}")
        group_used = used.setdefault(key[:2], set())
        if cell_id in group_used:
            raise ValueError(f"{path}: a control cell occurs twice within an allocation/depth")
        group_used.add(cell_id)
        grouped.setdefault(key, []).append((index, stratum, within))
    if set(grouped) != set(references.values):
        raise ValueError(f"{path}: missing reference-block membership")
    strata = set(cells.strata)
    for key, members in grouped.items():
        depth = key[1]
        for stratum in strata:
            positions = sorted(within for _, group, within in members if group == stratum)
            if len(positions) != depth or any(position != index for index, position in enumerate(positions)):
                raise ValueError(f"{path}: every block needs depth cells with complete ranks in each stratum")
        if references.counts is not None and references.counts[key] != len(members):
            raise ValueError(f"{path}: cell count differs from reference metadata")
        observed = cells.values[[index for index, _, _ in members]].mean(axis=0)
        if not np.allclose(observed, references.values[key], rtol=1e-10, atol=1e-12):
            raise ValueError(f"{path}: control membership does not reproduce reference means")
