#!/usr/bin/env python3
"""Recompute the factor-isolation endpoints from pseudonymized Source Data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from perturb_nuisance_contracts.source_data_root import (  # noqa: E402
    SourceDataRootError,
    validate_source_data_root,
)
import v22_core


LONG_NAME = "KANG_V22_DONOR_ROOT_ENDPOINTS_PUBLIC_V1.tsv"
SUMMARY_NAME = "KANG_V22_PRIMARY_ENDPOINTS_PUBLIC_V1.tsv"
GATE_NAME = "KANG_V22_CLAIM_GATES_PUBLIC_V1.tsv"
MANIFEST_NAME = "KANG_V22_PUBLIC_EVIDENCE_MANIFEST_V1.json"
ENDPOINTS = ("E1", "E2", "E3", "E4")
ROOTS = tuple(f"ROOT_{index:02d}" for index in range(1, 9))
DIRECTIONS = (1, 1, -1, 1)


class PublicReplayError(RuntimeError):
    """A public-evidence identity or replay invariant failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicReplayError(message)


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
            require(key not in result, f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON number {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise PublicReplayError(f"cannot read strict JSON: {path.name}") from error
    require(isinstance(value, dict), "public manifest root is not an object")
    return value


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            require(reader.fieldnames is not None, f"TSV header is absent: {path.name}")
            require(
                len(reader.fieldnames) == len(set(reader.fieldnames)),
                f"duplicate TSV column: {path.name}",
            )
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise PublicReplayError(f"cannot read TSV: {path.name}") from error
    require(all(None not in row for row in rows), f"ragged TSV row: {path.name}")
    return list(reader.fieldnames), rows


def finite(text: str, label: str) -> float:
    try:
        value = float(text)
    except ValueError as error:
        raise PublicReplayError(f"{label} is not numeric") from error
    require(math.isfinite(value), f"{label} is not finite")
    return value


def boolean(text: str, label: str) -> bool:
    require(text in {"true", "false"}, f"{label} is not canonical Boolean text")
    return text == "true"


def serialized_bound(text: str, state: str, label: str) -> float:
    require(
        state in {"FINITE", "POSITIVE_INFINITY", "NEGATIVE_INFINITY"},
        f"{label} serialization state differs",
    )
    if state == "FINITE":
        return finite(text, label)
    expected = "Inf" if state == "POSITIVE_INFINITY" else "-Inf"
    require(text == expected, f"{label} infinity serialization differs")
    return math.inf if state == "POSITIVE_INFINITY" else -math.inf


def load_public_projection(
    directory: Path,
) -> tuple[np.ndarray, dict[str, Any], dict[str, dict[str, str]], list[dict[str, str]]]:
    root = directory.resolve(strict=True)
    expected_names = {LONG_NAME, SUMMARY_NAME, GATE_NAME, MANIFEST_NAME}
    observed_names = {path.name for path in root.iterdir()}
    require(observed_names == expected_names, "public projection inventory differs")
    require(
        all((root / name).is_file() and not (root / name).is_symlink() for name in expected_names),
        "public projection contains a non-regular member",
    )

    manifest = strict_json(root / MANIFEST_NAME)
    require(
        manifest.get("schema") == "KANG_V22_PUBLIC_EVIDENCE_MANIFEST_V1"
        and manifest.get("status")
        == "PASS_PUBLICATION_PROJECTION_FROM_COMPLETE_SEALED_RESULTS",
        "public manifest identity differs",
    )
    require(manifest.get("public_root_order") == list(ROOTS), "public root order differs")
    require(
        manifest.get("internal_donor_identifiers_redistributed") is False,
        "manifest does not exclude internal donor identifiers",
    )
    require(manifest.get("endpoint_order") == list(ENDPOINTS), "endpoint order differs")
    require(manifest.get("raw_directions") == list(DIRECTIONS), "endpoint directions differ")
    table_records = manifest.get("public_source_tables")
    require(
        isinstance(table_records, Mapping) and set(table_records) == {LONG_NAME, SUMMARY_NAME, GATE_NAME},
        "manifest public-table inventory differs",
    )
    for name in (LONG_NAME, SUMMARY_NAME, GATE_NAME):
        record = table_records[name]
        path = root / name
        require(isinstance(record, Mapping), f"manifest record differs: {name}")
        require(
            record.get("bytes") == path.stat().st_size and record.get("sha256") == sha256(path),
            f"manifest digest binding differs: {name}",
        )

    _, rows = read_tsv(root / LONG_NAME)
    _, summary_rows = read_tsv(root / SUMMARY_NAME)
    _, gate_rows = read_tsv(root / GATE_NAME)
    require(len(rows) == 32, "public root table must contain 32 rows")
    require(
        len(summary_rows) == 4
        and [row.get("endpoint") for row in summary_rows] == list(ENDPOINTS),
        "public endpoint summary differs",
    )
    require(len(gate_rows) == 3, "public claim-gate table differs")
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        endpoint = row.get("endpoint", "")
        public_root = row.get("donor_root", "")
        require(endpoint in ENDPOINTS and public_root in ROOTS, "non-public endpoint/root identity")
        key = (endpoint, public_root)
        require(key not in indexed, "duplicate endpoint/root row")
        require(
            row.get("donor_root_index") == str(ROOTS.index(public_root) + 1),
            "public root index differs",
        )
        indexed[key] = row
    require(
        set(indexed) == {(endpoint, root) for endpoint in ENDPOINTS for root in ROOTS},
        "endpoint/root Cartesian family differs",
    )

    values = np.asarray(
        [
            [
                finite(indexed[(endpoint, root)]["raw_donor_root_value"], f"{endpoint}/{root}")
                for endpoint in ENDPOINTS
            ]
            for root in ROOTS
        ],
        dtype=np.float64,
    )
    for endpoint_index, endpoint in enumerate(ENDPOINTS):
        for root_index, root in enumerate(ROOTS):
            oriented = finite(
                indexed[(endpoint, root)]["oriented_donor_root_value"],
                f"{endpoint}/{root} oriented",
            )
            require(
                oriented == DIRECTIONS[endpoint_index] * values[root_index, endpoint_index],
                f"{endpoint}/{root} orientation differs",
            )
    return values, manifest, {row["endpoint"]: row for row in summary_rows}, gate_rows


def replay(directory: Path) -> dict[str, Any]:
    values, manifest, summary, gate_rows = load_public_projection(directory)
    inference = v22_core.exact_shared_sign_max_abs_t(values)
    secondary = manifest.get("secondary_strict_reversal")
    require(
        secondary
        == {
            "status": "NOT_RUN",
            "strict_reversal_secondary_evaluated": False,
            "strict_reversal_secondary_observed": None,
        },
        "secondary strict-reversal boundary differs",
    )
    claim = v22_core.map_claim(
        inference.endpoint_supported,
        inference.uniform_8_of_8,
        strict_reversal=False,
        complete=True,
    )
    observed_claim = manifest.get("claim_mapping")
    require(isinstance(observed_claim, Mapping), "manifest claim mapping is absent")
    require(observed_claim.get("claim") == claim.claim, "replayed claim differs")
    for index, endpoint in enumerate(ENDPOINTS):
        row = summary[endpoint]
        require(int(row["raw_direction"]) == DIRECTIONS[index], f"{endpoint} direction differs")
        require(
            boolean(row["primary_support"], f"{endpoint} support")
            is bool(inference.endpoint_supported[index]),
            f"{endpoint} replayed support differs",
        )
        require(
            int(row["uniform_positive_root_count"]) == int(inference.uniform_counts[index]),
            f"{endpoint} replayed positive-root count differs",
        )
        require(
            boolean(row["uniform_8_of_8_robustness"], f"{endpoint} uniform robustness")
            is bool(inference.uniform_8_of_8[index]),
            f"{endpoint} replayed uniform robustness differs",
        )
        for field, replayed in (
            ("estimate", inference.estimates[index]),
            ("oriented_estimate", DIRECTIONS[index] * inference.estimates[index]),
            ("standard_error", inference.standard_errors[index]),
        ):
            require(finite(row[field], f"{endpoint} {field}") == float(replayed), f"{endpoint} {field} differs")
        for field, state_field, replayed in (
            (
                "simultaneous_oriented_lower_bound",
                "simultaneous_oriented_lower_bound_state",
                inference.oriented_lower[index],
            ),
            (
                "simultaneous_oriented_upper_bound",
                "simultaneous_oriented_upper_bound_state",
                inference.oriented_upper[index],
            ),
        ):
            require(
                serialized_bound(row[field], row[state_field], f"{endpoint} {field}")
                == float(replayed),
                f"{endpoint} {field} differs",
            )
    expected_gates = (
        ("OUTPUT_REPRESENTATION_ATTRIBUTION", claim.representation_pair_supported),
        ("MODEL_CONDITIONING_ALLOCATION_ATTRIBUTION", claim.cmodel_pair_supported),
        ("JOINT_ATTRIBUTION", claim.all_four_supported),
    )
    for row, (gate_id, supported) in zip(gate_rows, expected_gates, strict=True):
        require(row.get("gate_id") == gate_id, "claim-gate order differs")
        require(
            boolean(row.get("paired_primary_gate_supported", ""), f"{gate_id} support")
            is bool(supported),
            f"{gate_id} replayed support differs",
        )
        require(row.get("complete_panel_claim") == claim.claim, f"{gate_id} claim differs")
    working_law = manifest.get("working_law")
    require(isinstance(working_law, Mapping), "manifest working-law record is absent")
    require(
        working_law.get("sign_vector_count") == inference.sign_vector_count
        and working_law.get("critical_order_one_indexed")
        == inference.critical_order_one_indexed,
        "replayed working-law schedule differs",
    )
    return {
        "schema": "V22_PUBLIC_REPLAY_V1",
        "status": "PASS_PUBLIC_ROOT_REPLAY",
        "endpoint_order": list(ENDPOINTS),
        "endpoint_supported": {
            name: bool(inference.endpoint_supported[index]) for index, name in enumerate(ENDPOINTS)
        },
        "uniform_positive_root_count": {
            name: int(inference.uniform_counts[index]) for index, name in enumerate(ENDPOINTS)
        },
        "claim": claim.claim,
        "sign_vector_count": inference.sign_vector_count,
        "critical_order_one_indexed": inference.critical_order_one_indexed,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "public_evidence_dir",
        type=Path,
        nargs="?",
        help=(
            "legacy explicit directory containing the four V22 projection files; "
            "prefer --source-data-root"
        ),
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
        raise PublicReplayError(
            "supply exactly one of --source-data-root or public_evidence_dir"
        )
    snapshot = None
    if args.source_data_root is not None:
        snapshot = validate_source_data_root(args.source_data_root)
        directory = snapshot.root / "Figure_4_attribution"
    else:
        directory = args.public_evidence_dir
    report = replay(directory)
    if snapshot is not None:
        snapshot.assert_unchanged()
        report["source_data_manifest_sha256"] = snapshot.manifest_sha256
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PublicReplayError, SourceDataRootError, v22_core.V22CoreError) as error:
        print(f"V22 public replay failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
