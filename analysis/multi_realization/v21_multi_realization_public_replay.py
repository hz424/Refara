#!/usr/bin/env python3
"""Replay the public five-realization sensitivity from separate Source Data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(HERE))

from perturb_nuisance_contracts.source_data_root import (  # noqa: E402
    SourceDataRootError,
    validate_source_data_root,
)
import v21_multi_realization_analysis_core as core  # noqa: E402


PROTOCOL = HERE / "GSE162632_MULTI_REALIZATION_PUBLIC_ANALYSIS_PROTOCOL_V1.json"
CORE_SHA256 = "0e57553464e2571570ce04cc554a571f79564a067bec37a55ad3c57b9d2c295f"
PROTOCOL_SHA256 = "e73b015f18243e8d1f5a1abcfe3f3aadef97a89ccfa34b11dfde92d66ee47354"
GENERIC = REPOSITORY / "scripts" / "qualify_generic_reference_framework_v1.py"
GENERIC_SHA256 = "976b3d82c24b5153ac8b3499ec4d3d313d0853541c4f43d850acca38bfde496c"
FORMAL_NAME = "GSE162632_V21_FORMAL_ROOT_TASK_UTILITIES_2400_V1.tsv"
SUMMARY_NAME = "GSE162632_V21_ANALYSIS_SUMMARY_V1.json"
BRANCH_NAME = "GSE162632_V21_OUTCOME_BRANCH_V1.json"
FORMAL_HEADER = (
    "method_id",
    "method_class",
    "realization_id",
    "seed",
    "formal_role",
    "root_index",
    "task_index",
    "construction_id",
    "utility",
    "training_realization_is_biological_replicate",
)
ANALYSIS_INDEX_LOGICAL_SHA256 = (
    "b09aafe4ef44e4f75c6bae39efc6378a0c6769fdc395edfa17524e2a5af424f3"
)
NUMERIC_ABSOLUTE_TOLERANCE = 1e-12


class MultiRealizationReplayError(RuntimeError):
    """A public multi-realization artifact or replay invariant failed."""


class ArtifactExpectation(NamedTuple):
    """Typed replay value plus its pinned-build byte representation."""

    kind: str
    value: Any
    pinned_bytes: bytes


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MultiRealizationReplayError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key!r}")
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
        raise MultiRealizationReplayError(f"invalid strict JSON: {path.name}") from error
    require(isinstance(value, dict), f"JSON root is not an object: {path.name}")
    return value


def load_generic() -> ModuleType:
    require(GENERIC.is_file() and not GENERIC.is_symlink(), "generic inference source is absent")
    require(sha256_file(GENERIC) == GENERIC_SHA256, "generic inference source hash differs")
    specification = importlib.util.spec_from_file_location(
        "v21_public_generic_inference", GENERIC
    )
    require(
        specification is not None and specification.loader is not None,
        "cannot load generic inference source",
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def read_formal_inputs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            require(tuple(reader.fieldnames or ()) == FORMAL_HEADER, "formal input header differs")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise MultiRealizationReplayError("cannot read formal root-task table") from error
    require(len(rows) == 2400, "formal root-task row count differs")
    require(all(None not in row for row in rows), "formal root-task table contains a ragged row")

    direct = np.full(
        (len(core.DIRECT_METHODS), core.ROOTS, core.TASKS, len(core.CONSTRUCTIONS)),
        np.nan,
        dtype=np.float64,
    )
    stochastic = np.full(
        (
            len(core.STOCHASTIC_METHODS),
            len(core.REALIZATIONS),
            core.ROOTS,
            core.TASKS,
            len(core.CONSTRUCTIONS),
        ),
        np.nan,
        dtype=np.float64,
    )
    seen: set[tuple[str, str, int, int, str]] = set()
    for row in rows:
        method = row["method_id"]
        realization = row["realization_id"]
        try:
            root = int(row["root_index"])
            task = int(row["task_index"])
            value = float(row["utility"])
        except ValueError as error:
            raise MultiRealizationReplayError("formal root-task scalar is invalid") from error
        construction = row["construction_id"]
        require(0 <= root < core.ROOTS and 0 <= task < core.TASKS, "formal root/task index differs")
        require(construction in core.CONSTRUCTIONS, "formal construction differs")
        require(math.isfinite(value), "formal utility is nonfinite")
        require(
            row["training_realization_is_biological_replicate"] == "false",
            "training realization is mislabeled as a biological replicate",
        )
        identity = (method, realization, root, task, construction)
        require(identity not in seen, "duplicate formal root-task identity")
        seen.add(identity)
        construction_index = core.CONSTRUCTIONS.index(construction)
        if method in core.DIRECT_METHODS:
            require(
                row["method_class"] == "FIXED_DIRECT"
                and realization == "NOT_APPLICABLE_FIXED_DIRECT"
                and row["seed"] == ""
                and row["formal_role"] == "HISTORICAL_FIXED_DIRECT_ANCHOR",
                "fixed-direct formal metadata differs",
            )
            direct[
                core.DIRECT_METHODS.index(method), root, task, construction_index
            ] = value
        else:
            require(method in core.STOCHASTIC_METHODS, "formal method differs")
            require(realization in core.REALIZATIONS, "formal realization differs")
            realization_index = core.REALIZATIONS.index(realization)
            expected_role = (
                "HISTORICAL_FORMAL_Q1_ANCHOR"
                if realization == "q1"
                else "FORMAL_Q2_TO_Q5_REALIZATION"
            )
            require(
                row["method_class"] == "STOCHASTIC_ABSOLUTE_STATE"
                and row["seed"] == str(core.SEEDS[realization_index])
                and row["formal_role"] == expected_role,
                "stochastic formal metadata differs",
            )
            stochastic[
                core.STOCHASTIC_METHODS.index(method),
                realization_index,
                root,
                task,
                construction_index,
            ] = value
    require(np.isfinite(direct).all(), "fixed-direct Cartesian input is incomplete")
    require(np.isfinite(stochastic).all(), "stochastic Cartesian input is incomplete")
    return direct, stochastic


def scalar_text(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (float, np.floating)):
        require(math.isfinite(float(value)), "replayed TSV contains a nonfinite float")
        return format(float(value), ".17g")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if value is None:
        return ""
    require(isinstance(value, str), "unsupported replayed TSV scalar")
    return value


def tsv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    require(bool(rows), "cannot serialize an empty TSV")
    header = tuple(rows[0])
    require(all(tuple(row) == header for row in rows), "replayed TSV row fields differ")
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow([scalar_text(row[key]) for key in header])
    return stream.getvalue().encode("utf-8")


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def per_realization_rows(values: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for q_index, realization in enumerate(core.REALIZATIONS):
        for root in range(core.ROOTS):
            for construction_index, construction in enumerate(core.CONSTRUCTIONS):
                for method_index, method in enumerate(core.METHODS):
                    rows.append(
                        {
                            "realization_id": realization,
                            "seed": core.SEEDS[q_index],
                            "root_index": root,
                            "construction_id": construction,
                            "method_id": method,
                            "root_task_mean_utility": float(
                                values[q_index, root, construction_index, method_index]
                            ),
                            "direct_method_repeated_across_realization_panels": method
                            in core.DIRECT_METHODS,
                            "training_realization_is_biological_replicate": False,
                        }
                    )
    require(len(rows) == 960, "per-realization row geometry differs")
    return rows


def seed_averaged_rows(values: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in range(core.ROOTS):
        for construction_index, construction in enumerate(core.CONSTRUCTIONS):
            for method_index, method in enumerate(core.METHODS):
                rows.append(
                    {
                        "root_index": root,
                        "construction_id": construction,
                        "method_id": method,
                        "seed_averaged_root_utility": float(
                            values[root, construction_index, method_index]
                        ),
                        "task_average_precedes_realization_average": True,
                        "stochastic_realization_count": 0
                        if method in core.DIRECT_METHODS
                        else 5,
                        "training_realizations_are_biological_replicates": False,
                    }
                )
    require(len(rows) == 192, "seed-averaged row geometry differs")
    return rows


def root_family_rows(
    values: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
    value_name: str,
) -> list[dict[str, Any]]:
    require(values.shape == (core.ROOTS, len(metadata)), "root-family geometry differs")
    rows: list[dict[str, Any]] = []
    for family_index, identity in enumerate(metadata):
        require(identity.get("family_index") == family_index, "family identity order differs")
        for root in range(core.ROOTS):
            rows.append(
                {
                    **identity,
                    "root_index": root,
                    value_name: float(values[root, family_index]),
                    "biological_root_count": core.ROOTS,
                }
            )
    return rows


def expected_artifacts(result: Mapping[str, Any]) -> dict[str, ArtifactExpectation]:
    aggregation = result["aggregation"]
    tabular: dict[str, Sequence[Mapping[str, Any]]] = {
        "GSE162632_V21_PER_REALIZATION_ROOT_UTILITIES_V1.tsv": per_realization_rows(
            aggregation["per_realization_root_utilities"]
        ),
        "GSE162632_V21_SEED_AVERAGED_ROOT_UTILITIES_V1.tsv": seed_averaged_rows(
            aggregation["seed_averaged_root_utilities"]
        ),
        "GSE162632_V21_WITHIN_CONSTRUCTION_ROOT_CONTRIBUTIONS_84_V1.tsv": root_family_rows(
            result["within_root_contributions"],
            core.within_family_metadata(),
            "root_contrast",
        ),
        "GSE162632_V21_WITHIN_CONSTRUCTION_WORKING_LAW_84_V1.tsv": result[
            "within_working_law_rows"
        ],
        "GSE162632_V21_PAIRED_CONSTRUCTION_ROOT_CHANGES_56_V1.tsv": root_family_rows(
            result["paired_root_contributions"],
            core.paired_family_metadata(),
            "root_paired_change",
        ),
        "GSE162632_V21_PAIRED_CONSTRUCTION_WORKING_LAW_56_V1.tsv": result[
            "paired_working_law_rows"
        ],
        "GSE162632_V21_FOCAL_PER_REALIZATION_V1.tsv": result["focal"][
            "per_realization"
        ],
        "GSE162632_V21_FOCAL_LEAVE_ONE_REALIZATION_OUT_V1.tsv": result["focal"][
            "leave_one_realization_out"
        ],
        "GSE162632_V21_STOCHASTIC_CARTESIAN_DESCRIPTIVE_V1.tsv": result[
            "stochastic_cartesian_diagnostics"
        ],
        "GSE162632_V21_REVERSAL_MEMBERSHIP_V1.tsv": result[
            "reversal_membership"
        ]["rows"],
    }
    expected = {
        name: ArtifactExpectation("tsv", rows, tsv_bytes(rows))
        for name, rows in tabular.items()
    }
    expected[BRANCH_NAME] = ArtifactExpectation(
        "json",
        result["outcome_branch"],
        json_bytes(result["outcome_branch"]),
    )
    summary = {
        "schema_version": 1,
        "record_type": "GSE162632_V21_MULTI_REALIZATION_ANALYSIS_SUMMARY_V1",
        "status": "PASS_GSE162632_V21_COMPLETE_MULTI_REALIZATION_ANALYSIS",
        "protocol_id": "GSE162632_V21_MULTI_REALIZATION_SENSITIVITY_V1",
        "analysis_timing": "POST_HOC_AFTER_PRIOR_OUTCOME_ACCESS",
        "outcome_branch": result["outcome_branch"],
        "reversal_counts": {
            key: result["reversal_membership"][key]
            for key in (
                "unique_pair_reversal_count",
                "shared_to_split_reversal_count",
                "shared_to_rotation_reversal_count",
            )
        },
        "focal_algorithmic_diagnostics": {
            "all_per_realization_passed": result["focal"][
                "all_per_realization_passed"
            ],
            "all_leave_one_out_passed": result["focal"]["all_leave_one_out_passed"],
        },
        "working_law": {
            "within_84_critical_value": result["within_working_law"]["critical_value"],
            "within_84_fallback": result["within_working_law"]["fallback"],
            "paired_56_critical_value": result["paired_working_law"]["critical_value"],
            "paired_56_fallback": result["paired_working_law"]["fallback"],
            "sign_count": core.SIGN_COUNT,
            "sign_schedule_sha256": core.SIGN_SCHEDULE_SHA256,
            "interpretation": "SIMULTANEOUS_WORKING_LAW_SENSITIVITY_ONLY",
            "joint_140_member_claim": False,
            "randomization_inference_claimed": False,
            "population_confidence_interval_claimed": False,
        },
        "aggregation_contract": {
            "biological_root_count": 8,
            "input_level": "ROOT_TASK_UTILITY",
            "method_specific_five_realization_mean_second": True,
            "method_specific_task_mean_first": True,
            "pairwise_contrasts_after_realization_mean": True,
            "prediction_averaging_across_realizations": False,
            "same_named_realizations_paired_across_methods": False,
            "training_realizations_treated_as_biological_replicates": False,
        },
        "analysis_index_logical_sha256": ANALYSIS_INDEX_LOGICAL_SHA256,
    }
    expected[SUMMARY_NAME] = ArtifactExpectation(
        "json",
        summary,
        json_bytes(summary),
    )
    require(len(expected) == 12, "replayed output inventory is not exactly 12")
    return expected


def compare_tsv(
    path: Path,
    expected_rows: Sequence[Mapping[str, Any]],
    *,
    absolute_tolerance: float = NUMERIC_ABSOLUTE_TOLERANCE,
) -> None:
    """Compare regenerated tabular values without pinning float text formatting."""

    require(bool(expected_rows), f"expected TSV rows are empty: {path.name}")
    expected_header = tuple(expected_rows[0])
    require(
        all(tuple(row) == expected_header for row in expected_rows),
        f"expected TSV schema is inconsistent: {path.name}",
    )
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            require(
                tuple(reader.fieldnames or ()) == expected_header,
                f"replayed TSV schema differs: {path.name}",
            )
            observed_rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise MultiRealizationReplayError(
            f"cannot read replay target TSV: {path.name}"
        ) from error
    require(
        len(observed_rows) == len(expected_rows),
        f"replayed TSV row count differs: {path.name}",
    )
    require(
        all(None not in row for row in observed_rows),
        f"replayed TSV contains a ragged row: {path.name}",
    )
    for row_index, (observed, expected) in enumerate(
        zip(observed_rows, expected_rows)
    ):
        for column in expected_header:
            observed_text = observed[column]
            expected_value = expected[column]
            if isinstance(expected_value, (float, np.floating)):
                try:
                    observed_value = float(observed_text)
                except ValueError as error:
                    raise MultiRealizationReplayError(
                        f"replayed TSV numeric scalar is invalid: "
                        f"{path.name} row {row_index + 2} column {column}"
                    ) from error
                require(
                    math.isfinite(observed_value)
                    and math.isfinite(float(expected_value)),
                    f"replayed TSV numeric scalar is nonfinite: "
                    f"{path.name} row {row_index + 2} column {column}",
                )
                require(
                    abs(observed_value - float(expected_value))
                    <= absolute_tolerance,
                    f"replayed TSV numeric scalar exceeds absolute tolerance "
                    f"{absolute_tolerance:g}: {path.name} row {row_index + 2} "
                    f"column {column}",
                )
            else:
                require(
                    observed_text == scalar_text(expected_value),
                    f"replayed TSV categorical/integer scalar differs: "
                    f"{path.name} row {row_index + 2} column {column}",
                )


def compare_json_value(
    observed: Any,
    expected: Any,
    *,
    location: str,
    absolute_tolerance: float = NUMERIC_ABSOLUTE_TOLERANCE,
) -> None:
    """Recursively compare JSON, allowing tolerance only for floating scalars."""

    if isinstance(expected, Mapping):
        require(isinstance(observed, Mapping), f"replayed JSON type differs: {location}")
        require(set(observed) == set(expected), f"replayed JSON fields differ: {location}")
        for key in expected:
            compare_json_value(
                observed[key],
                expected[key],
                location=f"{location}.{key}",
                absolute_tolerance=absolute_tolerance,
            )
        return
    if isinstance(expected, list):
        require(isinstance(observed, list), f"replayed JSON type differs: {location}")
        require(len(observed) == len(expected), f"replayed JSON length differs: {location}")
        for index, (observed_item, expected_item) in enumerate(zip(observed, expected)):
            compare_json_value(
                observed_item,
                expected_item,
                location=f"{location}[{index}]",
                absolute_tolerance=absolute_tolerance,
            )
        return
    if isinstance(expected, float):
        require(
            isinstance(observed, (int, float)) and not isinstance(observed, bool),
            f"replayed JSON numeric type differs: {location}",
        )
        require(
            math.isfinite(float(observed)) and math.isfinite(expected),
            f"replayed JSON numeric scalar is nonfinite: {location}",
        )
        require(
            abs(float(observed) - expected) <= absolute_tolerance,
            f"replayed JSON numeric scalar exceeds absolute tolerance "
            f"{absolute_tolerance:g}: {location}",
        )
        return
    require(
        type(observed) is type(expected) and observed == expected,
        f"replayed JSON categorical/integer scalar differs: {location}",
    )


def compare_artifact(path: Path, expectation: ArtifactExpectation) -> bool:
    """Enforce semantic equality and report optional pinned byte identity."""

    pinned_byte_identity = path.read_bytes() == expectation.pinned_bytes
    if expectation.kind == "tsv":
        compare_tsv(path, expectation.value)
    elif expectation.kind == "json":
        observed = strict_json(path)
        compare_json_value(observed, expectation.value, location=path.name)
    else:
        raise MultiRealizationReplayError(
            f"unknown artifact expectation kind: {expectation.kind!r}"
        )
    return pinned_byte_identity


def replay(source_data_root: Path) -> dict[str, Any]:
    require(sha256_file(Path(core.__file__)) == CORE_SHA256, "analysis core hash differs")
    require(sha256_file(PROTOCOL) == PROTOCOL_SHA256, "public protocol hash differs")
    protocol = strict_json(PROTOCOL)
    public = protocol.get("public_artifacts")
    require(isinstance(public, Mapping), "protocol public-artifact record is absent")
    expected_names = public.get("exact_files")
    require(
        isinstance(expected_names, list)
        and len(expected_names) == 13
        and len(set(expected_names)) == 13,
        "protocol public-artifact inventory differs",
    )
    snapshot = validate_source_data_root(source_data_root)
    directory = snapshot.root / str(public.get("directory"))
    require(directory.is_dir() and not directory.is_symlink(), "multi-realization directory is absent")
    observed_names = {path.name for path in directory.iterdir()}
    require(observed_names == set(expected_names), "multi-realization directory inventory differs")
    require(
        all((directory / name).is_file() and not (directory / name).is_symlink() for name in expected_names),
        "multi-realization directory contains a non-regular member",
    )

    direct, stochastic = read_formal_inputs(directory / FORMAL_NAME)
    result = core.analyze_complete_inputs(
        direct_root_task=direct,
        stochastic_root_task=stochastic,
        generic=load_generic(),
        structural_complete=True,
    )
    expected = expected_artifacts(result)
    pinned_byte_identity_count = sum(
        compare_artifact(directory / name, expectation)
        for name, expectation in expected.items()
    )
    snapshot.assert_unchanged()

    branch = result["outcome_branch"]
    reversal = result["reversal_membership"]
    require(branch["branch_id"] == "STRICT_PASS", "multi-realization outcome branch differs")
    return {
        "schema": "V21_MULTI_REALIZATION_PUBLIC_REPLAY_V1",
        "status": "PASS_V21_MULTI_REALIZATION_PUBLIC_REPLAY",
        "source_data_manifest_sha256": snapshot.manifest_sha256,
        "input_root_task_rows": 2400,
        "verified_derived_artifact_count": 12,
        "semantic_numeric_absolute_tolerance": NUMERIC_ABSOLUTE_TOLERANCE,
        "pinned_byte_identity_count": pinned_byte_identity_count,
        "within_family_size": 84,
        "paired_family_size": 56,
        "outcome_branch": branch["branch_id"],
        "all_per_realization_passed": bool(
            result["focal"]["all_per_realization_passed"]
        ),
        "all_leave_one_out_passed": bool(result["focal"]["all_leave_one_out_passed"]),
        "unique_pair_reversal_count": int(reversal["unique_pair_reversal_count"]),
        "sign_count": core.SIGN_COUNT,
        "sign_schedule_sha256": core.SIGN_SCHEDULE_SHA256,
        "analysis_timing": "POST_HOC_AFTER_PRIOR_OUTCOME_ACCESS",
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--source-data-root",
        type=Path,
        required=True,
        help="extracted 05_SOURCE_DATA directory matching the v1.7.0 manifest",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(replay(args.source_data_root), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        MultiRealizationReplayError,
        SourceDataRootError,
        core.MultiRealizationAnalysisError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
