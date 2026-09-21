#!/usr/bin/env python3
"""Replay the Cmodel successor from four pseudonymized Source Data files.

The replay starts at the released eight-root summaries. It recomputes the two
root shifts, the complete five-endpoint shared-sign max-|T| sensitivity family,
the midpoint identity, and the utility/output-displacement summaries. It does
not refit either model or recover the private root identifiers.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from perturb_nuisance_contracts.source_data_root import (  # noqa: E402
    SourceDataRootError,
    validate_source_data_root,
)


ROOTS = tuple(f"ROOT_{index:02d}" for index in range(1, 9))
ENDPOINTS = (
    "MARGIN_C",
    "MARGIN_A",
    "MARGIN_B",
    "SHIFT_A_MINUS_C",
    "SHIFT_B_MINUS_C",
)
ROOT_TABLE = "ED_FIG2_PANEL_A_ROOT_MARGINS.tsv"
INTERVAL_TABLE = "ED_FIG2_PANEL_B_FIVE_ENDPOINT_INTERVALS.tsv"
UTILITY_TABLE = "ED_FIG2_PANEL_C_UTILITY_RESPONSE.tsv"
MANIFEST = "ED_FIG2_PUBLIC_SOURCE_DATA_MANIFEST.json"
PUBLIC_FILES = (ROOT_TABLE, INTERVAL_TABLE, UTILITY_TABLE, MANIFEST)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

ROOT_FIELDS = (
    "root_code",
    "margin_C",
    "margin_A",
    "margin_B",
    "margin_shift_A_minus_C",
    "margin_shift_B_minus_C",
    "strict_root_reversal_C_to_A",
    "strict_root_reversal_C_to_B",
)
INTERVAL_FIELDS = (
    "endpoint_group",
    "endpoint_key",
    "display_label",
    "estimate",
    "simultaneous_lower",
    "simultaneous_upper",
    "interval_relation_to_zero",
    "interpretation",
)
UTILITY_FIELDS = (
    "root_code",
    "ridge_utility_shift_B_minus_C",
    "cellflow_utility_shift_B_minus_C",
    "margin_shift_B_minus_C",
    "ridge_standardized_rms_output_displacement_B_minus_C",
    "cellflow_standardized_rms_output_displacement_B_minus_C",
)


class CmodelPublicReplayError(RuntimeError):
    """The public projection or a replayed invariant failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CmodelPublicReplayError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key {key!r}: {path.name}")
            result[key] = value
        return result

    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON number {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise CmodelPublicReplayError(f"cannot read strict JSON: {path.name}") from error
    require(isinstance(value, dict), "public manifest root is not an object")
    canonical = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    require(raw == canonical, "public manifest is not canonical JSON")
    return value


def read_tsv(path: Path, expected_fields: tuple[str, ...]) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            require(tuple(reader.fieldnames or ()) == expected_fields, f"TSV schema differs: {path.name}")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise CmodelPublicReplayError(f"cannot read TSV: {path.name}") from error
    require(all(None not in row for row in rows), f"ragged TSV row: {path.name}")
    return rows


def finite(text: str, label: str) -> float:
    try:
        value = float(text)
    except ValueError as error:
        raise CmodelPublicReplayError(f"{label} is not numeric") from error
    require(math.isfinite(value), f"{label} is not finite")
    return value


def boolean(text: str, label: str) -> bool:
    require(text in {"true", "false"}, f"{label} is not canonical Boolean text")
    return text == "true"


def strict_reversal(first: float, second: float) -> bool:
    return first != 0.0 and second != 0.0 and ((first > 0.0) != (second > 0.0))


@dataclass(frozen=True)
class SharedSignFamily:
    estimates: np.ndarray
    standard_errors: np.ndarray
    critical_value: float
    lower: np.ndarray
    upper: np.ndarray
    sign_vector_count: int
    critical_order_one_indexed: int


@dataclass(frozen=True)
class PublicProjection:
    margins: np.ndarray
    shifts: np.ndarray
    utility_shifts: np.ndarray
    output_rms: np.ndarray
    interval_rows: tuple[Mapping[str, str], ...]
    manifest: Mapping[str, Any]


def exact_shared_sign_max_abs_t(values: np.ndarray, alpha: float = 0.05) -> SharedSignFamily:
    """Recenter, re-studentize and enumerate one shared sign per root."""

    matrix = np.asarray(values, dtype=np.float64)
    require(matrix.shape == (8, 5), f"endpoint matrix shape differs: {matrix.shape}")
    require(np.isfinite(matrix).all(), "endpoint matrix is non-finite")
    require(0.0 < alpha < 1.0, "alpha is outside (0, 1)")
    estimates = matrix.mean(axis=0)
    standard_errors = matrix.std(axis=0, ddof=1) / math.sqrt(8.0)
    require(bool(np.all(standard_errors > 0.0)), "an endpoint has zero standard error")
    influence = (matrix - estimates) / 8.0
    sum_squares = np.sum(np.square(influence), axis=0)
    maxima: list[float] = []
    for signs_tuple in product((-1.0, 1.0), repeat=8):
        signs = np.asarray(signs_tuple, dtype=np.float64)[:, None]
        signed_sum = np.sum(signs * influence, axis=0)
        denominator_squared = (8.0 / 7.0) * (
            sum_squares - np.square(signed_sum) / 8.0
        )
        tolerance = np.finfo(np.float64).eps * np.maximum(1.0, sum_squares)
        denominator_squared = np.where(
            (denominator_squared < 0.0) & (denominator_squared >= -tolerance),
            0.0,
            denominator_squared,
        )
        require(bool(np.all(denominator_squared > 0.0)), "a shared-sign resample is degenerate")
        maxima.append(float(np.max(np.abs(signed_sum) / np.sqrt(denominator_squared))))
    require(len(maxima) == 256, "shared-sign schedule differs from 256")
    order = int(math.ceil((1.0 - alpha) * len(maxima)))
    critical = float(np.sort(np.asarray(maxima, dtype=np.float64))[order - 1])
    return SharedSignFamily(
        estimates=estimates,
        standard_errors=standard_errors,
        critical_value=critical,
        lower=estimates - critical * standard_errors,
        upper=estimates + critical * standard_errors,
        sign_vector_count=len(maxima),
        critical_order_one_indexed=order,
    )


def load_public_projection(directory: Path) -> PublicProjection:
    root = directory.expanduser().resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "Cmodel Source Data directory is unsafe")
    observed = {path.name for path in root.iterdir()}
    require(observed == set(PUBLIC_FILES), "Cmodel public Source Data inventory differs")
    require(
        all((root / name).is_file() and not (root / name).is_symlink() for name in PUBLIC_FILES),
        "Cmodel public Source Data contains a non-regular member",
    )

    manifest = strict_json(root / MANIFEST)
    require(manifest.get("schema") == "ED_FIG2_PUBLIC_SOURCE_DATA_MANIFEST_V1", "public manifest schema differs")
    # Frozen provenance identifier retained after the current display moved to Fig4/Table8.
    require(manifest.get("figure") == "Extended Data Figure 2", "public manifest display differs")
    require(manifest.get("root_count") == 8, "public manifest root count differs")
    require(manifest.get("public_root_codes") == list(ROOTS), "public root order differs")
    require(manifest.get("fixed_roles") == {"Cobs": "A", "Cpred": "B"}, "fixed reference roles differ")
    require(
        manifest.get("claim_boundary") == "DESCRIPTIVE_CROSSING_INTO_UNRESOLVED_NEAR_TIE",
        "claim boundary differs",
    )
    table_hashes = manifest.get("public_tables_sha256")
    require(
        isinstance(table_hashes, Mapping)
        and set(table_hashes) == {ROOT_TABLE, INTERVAL_TABLE, UTILITY_TABLE},
        "public-table digest inventory differs",
    )
    for name in (ROOT_TABLE, INTERVAL_TABLE, UTILITY_TABLE):
        require(table_hashes.get(name) == sha256(root / name), f"public-table digest differs: {name}")
    formal_hashes = manifest.get("formal_input_sha256")
    require(
        isinstance(formal_hashes, Mapping)
        and set(formal_hashes) == {"sealed_root_table", "sealed_summary"}
        and all(isinstance(value, str) and SHA256_PATTERN.fullmatch(value) for value in formal_hashes.values()),
        "formal-input provenance digests differ",
    )
    boundary = manifest.get("inference_boundary")
    require(isinstance(boundary, Mapping), "inference boundary is absent")
    require(boundary.get("post_hoc_outcome_informed") is True, "post hoc timing differs")
    require(boundary.get("descriptive_aggregate_sign_reversal_C_to_B") is True, "descriptive reversal state differs")
    require(boundary.get("supported_new_winner_under_B") is False, "unsupported B-winner claim recorded")
    require(boundary.get("donor_population_confidence_intervals") is False, "population interval claim recorded")
    require(boundary.get("p_values_claimed") is False, "P-value claim recorded")

    root_rows = read_tsv(root / ROOT_TABLE, ROOT_FIELDS)
    interval_rows = read_tsv(root / INTERVAL_TABLE, INTERVAL_FIELDS)
    utility_rows = read_tsv(root / UTILITY_TABLE, UTILITY_FIELDS)
    require(len(root_rows) == len(utility_rows) == 8, "root-table row count differs")
    require(len(interval_rows) == 5, "five-endpoint table row count differs")
    require([row["root_code"] for row in root_rows] == list(ROOTS), "root-margin order differs")
    require([row["root_code"] for row in utility_rows] == list(ROOTS), "utility-response root order differs")
    require([row["endpoint_key"] for row in interval_rows] == list(ENDPOINTS), "endpoint order differs")

    margins = np.asarray(
        [
            [finite(row[column], f"{row['root_code']} {column}") for column in ("margin_C", "margin_A", "margin_B")]
            for row in root_rows
        ],
        dtype=np.float64,
    )
    shifts = np.asarray(
        [
            [
                finite(row["margin_shift_A_minus_C"], f"{row['root_code']} A-C shift"),
                finite(row["margin_shift_B_minus_C"], f"{row['root_code']} B-C shift"),
            ]
            for row in root_rows
        ],
        dtype=np.float64,
    )
    require(np.allclose(shifts[:, 0], margins[:, 1] - margins[:, 0], rtol=0.0, atol=5e-15), "A-C root shifts do not recompute")
    require(np.allclose(shifts[:, 1], margins[:, 2] - margins[:, 0], rtol=0.0, atol=5e-15), "B-C root shifts do not recompute")
    require(np.allclose(margins[:, 0], 0.5 * (margins[:, 1] + margins[:, 2]), rtol=0.0, atol=5e-14), "rootwise midpoint identity failed")
    require(np.allclose(shifts[:, 0], -shifts[:, 1], rtol=0.0, atol=5e-14), "equal-and-opposite shift identity failed")

    reversal_a = np.asarray([strict_reversal(row[0], row[1]) for row in margins])
    reversal_b = np.asarray([strict_reversal(row[0], row[2]) for row in margins])
    recorded_a = np.asarray([boolean(row["strict_root_reversal_C_to_A"], f"{row['root_code']} A reversal") for row in root_rows])
    recorded_b = np.asarray([boolean(row["strict_root_reversal_C_to_B"], f"{row['root_code']} B reversal") for row in root_rows])
    require(np.array_equal(reversal_a, recorded_a), "C-to-A root reversal flags differ")
    require(np.array_equal(reversal_b, recorded_b), "C-to-B root reversal flags differ")

    utility_shifts = np.asarray(
        [
            [
                finite(row["ridge_utility_shift_B_minus_C"], f"{row['root_code']} ridge utility shift"),
                finite(row["cellflow_utility_shift_B_minus_C"], f"{row['root_code']} CellFlow utility shift"),
            ]
            for row in utility_rows
        ],
        dtype=np.float64,
    )
    utility_margin_shifts = np.asarray(
        [finite(row["margin_shift_B_minus_C"], f"{row['root_code']} utility margin shift") for row in utility_rows]
    )
    require(np.allclose(utility_margin_shifts, shifts[:, 1], rtol=0.0, atol=5e-15), "margin-shift tables differ")
    require(np.allclose(utility_shifts[:, 0] - utility_shifts[:, 1], shifts[:, 1], rtol=0.0, atol=5e-15), "utility decomposition failed")
    output_rms = np.asarray(
        [
            [
                finite(row["ridge_standardized_rms_output_displacement_B_minus_C"], f"{row['root_code']} ridge RMS"),
                finite(row["cellflow_standardized_rms_output_displacement_B_minus_C"], f"{row['root_code']} CellFlow RMS"),
            ]
            for row in utility_rows
        ],
        dtype=np.float64,
    )
    require(bool(np.all(output_rms >= 0.0)), "an RMS displacement is negative")
    return PublicProjection(
        margins=margins,
        shifts=shifts,
        utility_shifts=utility_shifts,
        output_rms=output_rms,
        interval_rows=tuple(interval_rows),
        manifest=manifest,
    )


def replay(directory: Path) -> dict[str, Any]:
    projection = load_public_projection(directory)
    endpoint_matrix = np.column_stack(
        (
            projection.margins[:, 0],
            projection.margins[:, 1],
            projection.margins[:, 2],
            projection.shifts[:, 0],
            projection.shifts[:, 1],
        )
    )
    inference = exact_shared_sign_max_abs_t(endpoint_matrix)
    recorded_estimates = np.asarray([finite(row["estimate"], f"{row['endpoint_key']} estimate") for row in projection.interval_rows])
    recorded_lower = np.asarray([finite(row["simultaneous_lower"], f"{row['endpoint_key']} lower") for row in projection.interval_rows])
    recorded_upper = np.asarray([finite(row["simultaneous_upper"], f"{row['endpoint_key']} upper") for row in projection.interval_rows])
    require(np.allclose(recorded_estimates, inference.estimates, rtol=0.0, atol=5e-15), "five-endpoint estimates do not replay")
    require(np.allclose(recorded_lower, inference.lower, rtol=0.0, atol=5e-15), "five-endpoint lower bounds do not replay")
    require(np.allclose(recorded_upper, inference.upper, rtol=0.0, atol=5e-15), "five-endpoint upper bounds do not replay")

    expected_relation = []
    for lower, upper in zip(inference.lower, inference.upper, strict=True):
        expected_relation.append("spans_zero" if lower <= 0.0 <= upper else "above_zero" if lower > 0.0 else "below_zero")
    require(
        [row["interval_relation_to_zero"] for row in projection.interval_rows] == expected_relation,
        "recorded interval-to-zero relations differ",
    )
    b_index = ENDPOINTS.index("MARGIN_B")
    b_unresolved = bool(inference.lower[b_index] <= 0.0 <= inference.upper[b_index])
    require(b_unresolved, "B-condition margin is not unresolved")
    require(projection.interval_rows[b_index]["interpretation"] == "winner unresolved", "B-condition interpretation differs")
    require(bool(np.all(projection.shifts[:, 1] < 0.0)), "not all B-C root shifts are negative")
    require(int(np.count_nonzero((projection.margins[:, 0] * projection.margins[:, 2]) < 0.0)) == 5, "C-to-B root crossing count differs")

    means = projection.margins.mean(axis=0)
    aggregate_reversal = strict_reversal(float(means[0]), float(means[2]))
    require(aggregate_reversal, "aggregate C-to-B descriptive reversal is absent")
    require(not strict_reversal(float(means[0]), float(means[1])), "unexpected aggregate C-to-A reversal")
    utility_means = projection.utility_shifts.mean(axis=0)
    rms_means = projection.output_rms.mean(axis=0)
    require(
        math.isclose(float(utility_means[0] - utility_means[1]), float(projection.shifts[:, 1].mean()), rel_tol=0.0, abs_tol=5e-15),
        "equal-root utility decomposition failed",
    )

    midpoint_error = np.max(np.abs(projection.margins[:, 0] - 0.5 * (projection.margins[:, 1] + projection.margins[:, 2])))
    return {
        "schema": "CMODEL_SUCCESSOR_PUBLIC_REPLAY_V1",
        "status": "PASS_CMODEL_SUCCESSOR_PUBLIC_REPLAY",
        "scope": projection.manifest["scope"],
        "model_pair": projection.manifest["model_pair"],
        "root_count": 8,
        "endpoint_order": list(ENDPOINTS),
        "margin_equal_root_mean": {name: float(value) for name, value in zip(("C", "A", "B"), means, strict=True)},
        "margin_shift_equal_root_mean": {
            "A_MINUS_C": float(projection.shifts[:, 0].mean()),
            "B_MINUS_C": float(projection.shifts[:, 1].mean()),
        },
        "root_shift_signs": {
            "A_MINUS_C_positive": int(np.count_nonzero(projection.shifts[:, 0] > 0.0)),
            "B_MINUS_C_negative": int(np.count_nonzero(projection.shifts[:, 1] < 0.0)),
        },
        "strict_root_reversal_count": {
            "C_TO_A": int(np.count_nonzero((projection.margins[:, 0] * projection.margins[:, 1]) < 0.0)),
            "C_TO_B": int(np.count_nonzero((projection.margins[:, 0] * projection.margins[:, 2]) < 0.0)),
        },
        "descriptive_aggregate_reversal_C_to_B": aggregate_reversal,
        "B_condition_winner_unresolved": b_unresolved,
        "supported_new_winner_under_B": False,
        "midpoint_identity": {
            "relation": "MARGIN_C_EQUALS_ONE_HALF_MARGIN_A_PLUS_MARGIN_B",
            "maximum_root_absolute_error": float(midpoint_error),
            "equal_and_opposite_shifts": True,
        },
        "five_endpoint_sensitivity": {
            "procedure": "RECENTERED_RESTUDENTIZED_SHARED_SIGN_MAX_ABS_T",
            "familywise_alpha": 0.05,
            "sign_vector_count": inference.sign_vector_count,
            "critical_order_one_indexed": inference.critical_order_one_indexed,
            "critical_value": inference.critical_value,
            "estimates": {name: float(value) for name, value in zip(ENDPOINTS, inference.estimates, strict=True)},
            "simultaneous_lower": {name: float(value) for name, value in zip(ENDPOINTS, inference.lower, strict=True)},
            "simultaneous_upper": {name: float(value) for name, value in zip(ENDPOINTS, inference.upper, strict=True)},
            "interpretation": "REPEATED_ROOT_WORKING_LAW_SENSITIVITY_ONLY",
            "donor_population_confidence_intervals_claimed": False,
            "p_values_claimed": False,
        },
        "utility_response_equal_root_mean_B_minus_C": {
            "ridge": float(utility_means[0]),
            "cellflow": float(utility_means[1]),
            "ridge_minus_cellflow": float(utility_means[0] - utility_means[1]),
        },
        "standardized_rms_output_displacement_equal_root_mean_B_minus_C": {
            "ridge": float(rms_means[0]),
            "cellflow": float(rms_means[1]),
        },
        "claim_boundary": "DESCRIPTIVE_CROSSING_INTO_UNRESOLVED_NEAR_TIE",
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "public_evidence_dir",
        type=Path,
        nargs="?",
        help="legacy explicit directory containing the four Cmodel Source Data files",
    )
    result.add_argument(
        "--source-data-root",
        type=Path,
        help="extracted 05_SOURCE_DATA directory matching the v1.7.0 manifest",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if (args.public_evidence_dir is None) == (args.source_data_root is None):
        raise CmodelPublicReplayError("supply exactly one of --source-data-root or public_evidence_dir")
    snapshot = None
    if args.source_data_root is not None:
        snapshot = validate_source_data_root(args.source_data_root)
        manifest_record = snapshot.by_runtime_path()[f"data/derived/cmodel_successor/{MANIFEST}"]
        directory = snapshot.root / manifest_record.external_path.parent
    else:
        directory = args.public_evidence_dir
    report = replay(directory)
    if snapshot is not None:
        snapshot.assert_unchanged()
        report["source_data_manifest_sha256"] = snapshot.manifest_sha256
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CmodelPublicReplayError, SourceDataRootError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
