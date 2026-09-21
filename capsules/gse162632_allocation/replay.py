#!/usr/bin/env python3
"""Replay GSE162632 label-level allocation aggregation for Figure 2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PARTITIONS = 1000
DEPTHS = (4, 6, 8)
ASSIGNMENTS = 27
PATTERNS = (
    "ALL_SHARED",
    "OBS_PRED_SHARED_MODEL_SEPARATE",
    "OBS_MODEL_SHARED_PRED_SEPARATE",
    "PRED_MODEL_SHARED_OBS_SEPARATE",
    "ALL_DISJOINT",
)
ROOTS = 8
METHODS = (
    "NO_CHANGE_DIRECT_V1",
    "CONTEXT_MEAN_EFFECT_DIRECT_V1",
    "TWO_WAY_ADDITIVE_RIDGE_DIRECT_V1",
    "PCA64_ADDITIVE_RIDGE_DIRECT_V1",
    "RBF_KERNEL_RIDGE_DIRECT_V1",
    "CPA_0_8_8_ABSOLUTE_V1",
    "SCGEN_2_1_1_ABSOLUTE_V1",
    "CELLOT_522D2B9_ABSOLUTE_V1",
)
DIRECT_METHODS = 5
PAIRS = tuple(itertools.combinations(range(len(METHODS)), 2))
PATTERN_CODES = {1: "M", 2: "P", 3: "O", 4: "D"}
EXPECTED_FILES = {
    "GSE_STOCHASTIC_UNIT_UTILITIES_V1.npy":
        "7bdaf7f00926d9cf5610bde5850093fbf83cea4a9bd5ab4998a7c83fbb212300",
    "GSE162632_PATTERN_UNIT_UTILITIES_V1.npy":
        "09052b825e803c200983dbf84d5ecd4b04337fb1f8cf0d056d1b49147e4d1c7f",
    "GSE162632_ALLOCATION_INTERACTIONS_V1.npy":
        "a79800fb0efcf8c1d03ddc4383f6c87cf2f3d0c1d966ade8970abfc1e524db7a",
    "GSE_STOCHASTIC_PARTITION_AXIS_V1.tsv":
        "0bc945ec54f1dde03f73719c9fcd133864003c9c55629bcee73e160b7cf7ce60",
    "GSE_STOCHASTIC_ROLE_ASSIGNMENT_AXIS_V1.tsv":
        "810e7ccc42fe89d8f275d584159a5f9b4e0fe134921ff66155f23c3b3c29cae9",
    "CONFIGURATION_AXIS_V1.tsv":
        "792c3bd4440ecbb0586eba295a046b700e8b89ab42c2dadf7ed90dfa43a461b4",
    "CONFIGURATION_PAIR_AXIS_V1.tsv":
        "f51ceb098910d6a00313d2fc1804d16c08df5a5d33cb5a10496d3c8ccea07f00",
    "ROLE_OVERLAP_PATTERN_AXIS_V1.tsv":
        "6c34b2ce2df60cbd57b0b24067a1cb81ae47085bfe49ee91b9de1efeafeae07d",
    "PUBLIC_ROOT_AXIS_V1.tsv":
        "85f26295dace2feb319f797951ef495e5a564e607e729b4abadc304cf4b51c69",
}


class ReplayError(RuntimeError):
    """Raised when a frozen replay invariant differs."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def verify_inventory() -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, expected in EXPECTED_FILES.items():
        path = DATA / name
        require(path.is_file() and not path.is_symlink(), f"missing or unsafe input: {name}")
        digest = sha256(path)
        require(digest == expected, f"SHA-256 differs: {name}")
        observed[name] = digest
    return observed


def verify_axes() -> list[list[int]]:
    partitions = read_tsv(DATA / "GSE_STOCHASTIC_PARTITION_AXIS_V1.tsv")
    require(len(partitions) == PARTITIONS * len(DEPTHS), "partition-axis row count differs")
    observed_keys = {(int(row["partition_index"]), int(row["depth"])) for row in partitions}
    expected_keys = set(itertools.product(range(PARTITIONS), DEPTHS))
    require(observed_keys == expected_keys, "partition-axis keys differ")

    assignments = read_tsv(DATA / "GSE_STOCHASTIC_ROLE_ASSIGNMENT_AXIS_V1.tsv")
    require(len(assignments) == ASSIGNMENTS, "role-assignment row count differs")
    groups = [
        [index for index, row in enumerate(assignments) if int(row["pattern_order"]) == order]
        for order in range(len(PATTERNS))
    ]
    require([len(group) for group in groups] == [3, 6, 6, 6, 6], "pattern groups differ")
    require(
        all(assignments[index]["pattern_id"] == PATTERNS[order]
            for order, group in enumerate(groups) for index in group),
        "pattern identifiers differ",
    )

    configurations = read_tsv(DATA / "CONFIGURATION_AXIS_V1.tsv")
    require(
        [row["configuration_id"] for row in configurations] == list(METHODS)
        and [int(row["configuration_order"]) for row in configurations] == list(range(8)),
        "configuration axis differs",
    )
    pairs = read_tsv(DATA / "CONFIGURATION_PAIR_AXIS_V1.tsv")
    require(len(pairs) == len(PAIRS), "pair-axis row count differs")
    observed_pairs = [
        (int(row["source_configuration_order"]), int(row["target_configuration_order"]))
        for row in pairs
    ]
    require(observed_pairs == list(PAIRS), "pair axis differs")
    pattern_axis = read_tsv(DATA / "ROLE_OVERLAP_PATTERN_AXIS_V1.tsv")
    require([row["pattern_id"] for row in pattern_axis] == list(PATTERNS), "pattern axis differs")
    public_roots = read_tsv(DATA / "PUBLIC_ROOT_AXIS_V1.tsv")
    require(
        [int(row["root_order"]) for row in public_roots] == list(range(ROOTS))
        and [row["public_root_code"] for row in public_roots]
        == [f"ROOT_{index:02d}" for index in range(1, ROOTS + 1)],
        "public root axis differs",
    )
    return groups


def derive_arrays(values: np.ndarray, groups: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    require(
        values.dtype.str == "<f8"
        and values.shape == (PARTITIONS, len(DEPTHS), ASSIGNMENTS, ROOTS, len(METHODS))
        and np.isfinite(values).all(),
        "unit-utility array geometry or values differ",
    )
    pattern = np.stack(
        [np.mean(values[:, :, group, :, :], axis=2, dtype=np.float64) for group in groups],
        axis=2,
    )
    require(
        np.allclose(
            pattern[:, :, :, :, :DIRECT_METHODS],
            pattern[:, :, :1, :, :DIRECT_METHODS],
            rtol=1.0e-12,
            atol=1.0e-12,
        ),
        "direct-effect algebraic invariance failed",
    )
    change = pattern[:, :, 1:, :, :] - pattern[:, :, :1, :, :]
    interaction = np.stack(
        [change[..., left] - change[..., right] for left, right in PAIRS],
        axis=-1,
    )
    require(np.isfinite(pattern).all() and np.isfinite(interaction).all(), "derived arrays are non-finite")
    return pattern, interaction


def summarize(pattern: np.ndarray) -> dict[str, Any]:
    pair_contrasts = np.stack(
        [pattern[..., left] - pattern[..., right] for left, right in PAIRS], axis=-1
    )
    mean_contrasts = np.mean(pair_contrasts, axis=(0, 3), dtype=np.float64)
    anchor = mean_contrasts[:, 0, :]
    comparisons = mean_contrasts[:, 1:, :]
    displacement = comparisons - anchor[:, None, :]
    rms = np.sqrt(np.mean(np.square(displacement), axis=-1, dtype=np.float64))
    reversed_count = np.sum(anchor[:, None, :] * comparisons < 0.0, axis=-1)
    tied_count = np.sum(
        (anchor[:, None, :] == 0.0) | (comparisons == 0.0), axis=-1
    )
    return {
        "mean_contrasts": mean_contrasts,
        "mean_displacement": displacement,
        "depth_collapsed_displacement": np.mean(displacement, axis=0, dtype=np.float64),
        "rms": rms,
        "reversed_count": reversed_count,
        "tied_count": tied_count,
    }


def verify_source_data(source_root: Path, summary: dict[str, Any]) -> dict[str, Any]:
    base = source_root / "data" / "derived" / "v8" / "f3"
    rows_a = [row for row in read_tsv(base / "F3A_SOURCE_ROWS_V1.tsv") if row["dataset_id"] == "GSE162632"]
    rows_b = [row for row in read_tsv(base / "F3B_SOURCE_ROWS_V1.tsv") if row["dataset_id"] == "GSE162632"]
    rows_c = [row for row in read_tsv(base / "F3C_SOURCE_ROWS_V1.tsv") if row["dataset_id"] == "GSE162632"]
    require(len(rows_a) == 4 * len(PAIRS), "GSE F3A row count differs")
    require(len(rows_b) == len(DEPTHS) * 4, "GSE F3B row count differs")
    require(len(rows_c) == len(DEPTHS) * 4, "GSE F3C row count differs")

    max_abs = 0.0
    for row in rows_a:
        pattern = int(row["comparison_pattern_order"])
        pair = int(row["pair_order"])
        expected = float(row["mean_pair_contrast_displacement"])
        observed = float(summary["depth_collapsed_displacement"][pattern - 1, pair])
        max_abs = max(max_abs, abs(observed - expected))
    for row in rows_b:
        depth = DEPTHS.index(int(row["depth"]))
        pattern = int(row["comparison_pattern_order"])
        expected = float(row["rms_complete_vector_displacement"])
        observed = float(summary["rms"][depth, pattern - 1])
        max_abs = max(max_abs, abs(observed - expected))
    for row in rows_c:
        depth = DEPTHS.index(int(row["depth"]))
        pattern = int(row["comparison_pattern_order"])
        require(
            int(summary["reversed_count"][depth, pattern - 1]) == int(row["reversed_count"])
            and int(summary["tied_count"][depth, pattern - 1]) == int(row["tied_count"]),
            f"GSE reversal status differs at depth {DEPTHS[depth]}, pattern {pattern}",
        )
    require(max_abs <= 5.0e-12, f"GSE Figure 2 source rows differ: max abs {max_abs}")
    return {"status": "PASS", "rows_checked": len(rows_a) + len(rows_b) + len(rows_c), "max_abs_difference": max_abs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data-root", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()

    hashes = verify_inventory()
    groups = verify_axes()
    values = np.load(DATA / "GSE_STOCHASTIC_UNIT_UTILITIES_V1.npy", mmap_mode="r", allow_pickle=False)
    pattern, interaction = derive_arrays(values, groups)
    expected_pattern = np.load(DATA / "GSE162632_PATTERN_UNIT_UTILITIES_V1.npy", mmap_mode="r", allow_pickle=False)
    expected_interaction = np.load(DATA / "GSE162632_ALLOCATION_INTERACTIONS_V1.npy", mmap_mode="r", allow_pickle=False)
    require(np.array_equal(pattern, expected_pattern), "pattern utilities are not byte-value identical")
    require(np.array_equal(interaction, expected_interaction), "allocation interactions are not byte-value identical")
    summary = summarize(pattern)
    source_check: dict[str, Any] = {"status": "NOT_REQUESTED"}
    if args.source_data_root is not None:
        source_check = verify_source_data(args.source_data_root.resolve(strict=True), summary)

    receipt = {
        "schema": "GSE162632_LABEL_LEVEL_ALLOCATION_REPLAY_V1",
        "status": "PASS",
        "evidence_boundary": "LABEL_LEVEL_UTILITY_TO_FIGURE_SOURCE_ROWS_NOT_RAW_EXPRESSION_OR_MODEL_FITTING",
        "input_sha256": hashes,
        "unit_utility_shape": list(values.shape),
        "pattern_utility_shape": list(pattern.shape),
        "interaction_shape": list(interaction.shape),
        "pattern_arrays_exact": True,
        "interaction_arrays_exact": True,
        "figure_source_data_check": source_check,
        "reversal_counts_by_depth_M_P_O_D": summary["reversed_count"].astype(int).tolist(),
        "ties_by_depth_M_P_O_D": summary["tied_count"].astype(int).tolist(),
        "partitions_are_biological_replicates": False,
        "partition_frequency_is_natural_probability": False,
    }
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.receipt is not None:
        target = args.receipt.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
