from __future__ import annotations

from decimal import Decimal
import csv
from fractions import Fraction
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import tempfile
from typing import Any


REFERENCE_ROLES = (
    "observed_effect",
    "prediction_reference",
    "model_conditioning",
)
TIE_RULE = "LITERAL_IEEE754_BINARY64_ZERO"
HOLM_SCOPE = "WITHIN_DATASET_AND_ALLOCATION_COMPLETE_ORDERED_METHOD_FAMILY"
MANIFEST = "BENCHMARK_DESIGN_AUDIT_MANIFEST_V1.sha256"
GEOMETRY_INTERPRETATION = "CANDIDATE_DEPENDENCE_GEOMETRY_NOT_INDEPENDENCE_EVIDENCE"
MAPPING_SCOPE = "DECLARED_BLOCK_ABSTRACTION_NO_CELL_MEMBERSHIP_VERIFICATION"
AGGREGATION_ORDER = "SCORE_WITHIN_ROTATION_THEN_EQUAL_WEIGHT_ROTATIONS"
DECISION_RULE = "EXACT_ONE_SIDED_SIGN_HOLM_COMPLETE_ORDERED_V1"
MISSING_UNIT_RULE = "FAIL_CLOSED_COMPLETE_UNIT_BY_METHOD_SUPPORT"
UTILITY_PROVENANCE_SCOPE = (
    "PREAGGREGATED_UNIT_UTILITIES_NO_UPSTREAM_CELL_OR_MODEL_PROVENANCE_VERIFICATION"
)
INDEPENDENCE_ESTABLISHED = "ESTABLISHED_BY_EXPERT_ASSERTION"
INDEPENDENCE_NOT_ESTABLISHED = "NOT_ESTABLISHED"
TRAIN_EVAL_ESTABLISHED = "ESTABLISHED_BY_EXPERT_ASSERTION"
TRAIN_EVAL_NOT_ESTABLISHED = "NOT_ESTABLISHED"
TRAIN_EVAL_NOT_APPLICABLE = "NOT_APPLICABLE"
DECISION_REJECTION = "DIRECTION"
DECISION_NO_DIRECTION = "NO_DIRECTION"
DECISION_WITHHELD = "WITHHELD"
WITHHELD_INDEPENDENCE = "UNIT_INDEPENDENCE_NOT_ESTABLISHED"
WITHHELD_TRAIN_EVAL = "TRAINING_EVALUATION_SEPARATION_NOT_ESTABLISHED"
WITHHELD_BELOW_KMIN = "ARITHMETIC_SUPPORT_BELOW_KMIN"
UNIT_AUDIT_TREE = {
    "ACQUISITION_EDGE_CANDIDATES_V1.tsv",
    "AUDIT_RECEIPT_V1.json",
    MANIFEST,
    "CONFIG_SNAPSHOT.json",
    "INPUT_METADATA.tsv",
    "REPORTING_CHECKLIST_V1.md",
    "UNIT_AUDIT_V1.tsv",
    "UNIT_GROUP_GEOMETRY_V1.tsv",
    "UNIT_HIERARCHY_V1.tsv",
}


class AuditError(RuntimeError):
    """A fail-closed configuration, input, or bundle error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _nonempty_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and value.strip() == value and value, f"invalid {label}")
    return value


def _unique_strings(value: Any, label: str, minimum: int = 1) -> list[str]:
    require(isinstance(value, list) and len(value) >= minimum, f"invalid {label}")
    strings = [_nonempty_string(item, label) for item in value]
    require(len(strings) == len(set(strings)), f"duplicate {label}")
    return strings


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        require(key not in value, f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def load_config(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), "config must be a regular file")
    try:
        config = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AuditError("config is not valid UTF-8 JSON") from error
    require(isinstance(config, dict), "config must be one JSON object")
    require(
        set(config)
        == {
            "schema_version",
            "audit_id",
            "dataset_ids",
            "configurations",
            "reference_declaration",
            "reference_roles",
            "allocations",
            "unit_contract",
            "utility_input_contract",
            "analysis_timing",
            "comparison_family",
            "inference",
        },
        "config top-level fields differ",
    )
    require(config["schema_version"] == "1.0.0", "unsupported schema_version")
    _nonempty_string(config["audit_id"], "audit_id")
    datasets = _unique_strings(config["dataset_ids"], "dataset_ids")

    configurations = config["configurations"]
    require(isinstance(configurations, list) and len(configurations) >= 2, "configurations differ")
    configuration_ids: list[str] = []
    for configuration in configurations:
        require(isinstance(configuration, dict), "configuration must be an object")
        require(
            set(configuration)
            == {
                "configuration_id",
                "implementation_identity",
                "terminal_state_identity",
                "output_semantics",
                "active_reference_roles",
                "training_evaluation_separation",
            },
            "configuration identity fields differ",
        )
        configuration_ids.append(
            _nonempty_string(configuration["configuration_id"], "configuration_id")
        )
        _nonempty_string(configuration["implementation_identity"], "implementation_identity")
        _nonempty_string(configuration["terminal_state_identity"], "terminal_state_identity")
        require(
            configuration["output_semantics"] in {"DIRECT_EFFECT", "ABSOLUTE_STATE"},
            "output_semantics is invalid",
        )
        active_roles = _unique_strings(
            configuration["active_reference_roles"],
            "active_reference_roles",
        )
        expected_active_roles = (
            ["observed_effect"]
            if configuration["output_semantics"] == "DIRECT_EFFECT"
            else list(REFERENCE_ROLES)
        )
        require(
            active_roles == expected_active_roles,
            "active_reference_roles differ from output_semantics",
        )
        separation = configuration["training_evaluation_separation"]
        require(isinstance(separation, dict), "training_evaluation_separation must be an object")
        require(
            set(separation) == {"status", "basis", "assertion_authority", "machine_verified"},
            "training_evaluation_separation fields differ",
        )
        require(
            separation["status"]
            in {TRAIN_EVAL_ESTABLISHED, TRAIN_EVAL_NOT_ESTABLISHED, TRAIN_EVAL_NOT_APPLICABLE},
            "training_evaluation_separation status is invalid",
        )
        _nonempty_string(separation["basis"], "training_evaluation_separation basis")
        _nonempty_string(
            separation["assertion_authority"],
            "training_evaluation_separation assertion_authority",
        )
        require(
            separation["machine_verified"] is False,
            "training/evaluation separation is recorded, not machine verified",
        )
    require(
        len(configuration_ids) == len(set(configuration_ids)),
        "duplicate configuration_id",
    )

    reference_declaration = config["reference_declaration"]
    require(isinstance(reference_declaration, dict), "reference_declaration must be an object")
    require(
        set(reference_declaration) == {"mapping_scope", "block_label_semantics"},
        "reference_declaration fields differ",
    )
    require(
        reference_declaration["mapping_scope"] == MAPPING_SCOPE,
        "reference mapping scope differs",
    )
    _nonempty_string(reference_declaration["block_label_semantics"], "block_label_semantics")

    roles = _unique_strings(config["reference_roles"], "reference_roles", minimum=3)
    require(tuple(roles) == REFERENCE_ROLES, "reference_roles must use the registered order")

    allocations = config["allocations"]
    require(isinstance(allocations, list) and allocations, "allocations must be nonempty")
    allocation_ids: list[str] = []
    for allocation in allocations:
        require(isinstance(allocation, dict), "allocation must be an object")
        require(
            set(allocation)
            == {
                "allocation_id",
                "reference_depth_label",
                "reference_depth_unit",
                "aggregation_order",
                "rotations",
            },
            "allocation fields differ",
        )
        allocation_ids.append(_nonempty_string(allocation["allocation_id"], "allocation_id"))
        _nonempty_string(allocation["reference_depth_label"], "reference_depth_label")
        _nonempty_string(allocation["reference_depth_unit"], "reference_depth_unit")
        require(allocation["aggregation_order"] == AGGREGATION_ORDER, "aggregation_order differs")
        rotations = allocation["rotations"]
        require(isinstance(rotations, list) and rotations, "rotations must be nonempty")
        rotation_ids: list[str] = []
        for rotation in rotations:
            require(isinstance(rotation, dict), "rotation must be an object")
            require(
                set(rotation)
                == {
                    "rotation_id",
                    *REFERENCE_ROLES,
                    "role_depth_labels",
                    "expected_overlap_pattern",
                },
                "rotation fields differ",
            )
            rotation_ids.append(_nonempty_string(rotation["rotation_id"], "rotation_id"))
            for role in REFERENCE_ROLES:
                _nonempty_string(rotation[role], f"rotation {role}")
            depth_labels = rotation["role_depth_labels"]
            require(
                isinstance(depth_labels, dict) and set(depth_labels) == set(REFERENCE_ROLES),
                "role_depth_labels fields differ",
            )
            for role in REFERENCE_ROLES:
                _nonempty_string(depth_labels[role], f"role depth label {role}")
            require(
                rotation["expected_overlap_pattern"] == _overlap_pattern(rotation),
                "declared overlap pattern differs from role-to-block mapping",
            )
        require(len(rotation_ids) == len(set(rotation_ids)), "duplicate rotation_id")
    require(len(allocation_ids) == len(set(allocation_ids)), "duplicate allocation_id")

    unit = config["unit_contract"]
    require(isinstance(unit, dict), "unit_contract must be an object")
    require(
        set(unit)
        == {
            "dataset_column",
            "object_id_column",
            "unit_id_column",
            "declared_unit_columns",
            "candidate_acquisition_group_columns",
            "independence_justification",
        },
        "unit_contract fields differ",
    )
    scalar_columns = [
        _nonempty_string(unit[name], name)
        for name in ("dataset_column", "object_id_column", "unit_id_column")
    ]
    declared_columns = _unique_strings(unit["declared_unit_columns"], "declared_unit_columns")
    candidate_columns = _unique_strings(
        unit["candidate_acquisition_group_columns"],
        "candidate_acquisition_group_columns",
        minimum=0,
    )
    require(
        len(set(scalar_columns + declared_columns + candidate_columns))
        == len(scalar_columns + declared_columns + candidate_columns),
        "unit-contract columns must be distinct",
    )
    independence = unit["independence_justification"]
    require(isinstance(independence, dict), "independence_justification must be an object")
    require(
        set(independence) == {"status", "basis", "assertion_authority", "machine_verified"},
        "independence_justification fields differ",
    )
    require(
        independence["status"] in {INDEPENDENCE_ESTABLISHED, INDEPENDENCE_NOT_ESTABLISHED},
        "independence status is invalid",
    )
    _nonempty_string(independence["basis"], "independence basis")
    _nonempty_string(independence["assertion_authority"], "independence assertion_authority")
    require(
        independence["machine_verified"] is False,
        "unit independence is an expert assertion, not a machine-verified fact",
    )

    utility = config["utility_input_contract"]
    require(isinstance(utility, dict), "utility_input_contract must be an object")
    require(
        set(utility)
        == {
            "input_columns",
            "dataset_id_column",
            "allocation_id_column",
            "unit_id_column",
            "method_id_column",
            "utility_column",
            "required_constant_values",
            "expected_input_sha256",
            "provenance_scope",
        },
        "utility_input_contract fields differ",
    )
    input_columns = _unique_strings(utility["input_columns"], "utility input_columns", minimum=5)
    mapped_columns = [
        _nonempty_string(utility[name], name)
        for name in (
            "dataset_id_column",
            "allocation_id_column",
            "unit_id_column",
            "method_id_column",
            "utility_column",
        )
    ]
    require(len(mapped_columns) == len(set(mapped_columns)), "utility mapped columns must be distinct")
    require(set(mapped_columns).issubset(input_columns), "utility mapped column is not an input column")
    constants = utility["required_constant_values"]
    require(isinstance(constants, dict), "required_constant_values must be an object")
    require(set(constants).issubset(input_columns), "constant column is not an input column")
    require(not set(constants).intersection(mapped_columns), "mapped utility column cannot be constant")
    for column, value in constants.items():
        _nonempty_string(column, "required constant column")
        _nonempty_string(value, f"required constant value for {column}")
    expected_input_sha256 = utility["expected_input_sha256"]
    require(
        expected_input_sha256 is None
        or (
            isinstance(expected_input_sha256, str)
            and len(expected_input_sha256) == 64
            and all(character in "0123456789abcdef" for character in expected_input_sha256)
        ),
        "expected_input_sha256 is invalid",
    )
    require(utility["provenance_scope"] == UTILITY_PROVENANCE_SCOPE, "utility provenance scope differs")

    timing = config["analysis_timing"]
    require(isinstance(timing, dict), "analysis_timing must be an object")
    require(
        set(timing)
        == {
            "evidence_timing_label",
            "configuration_fixed_before_utility_access",
            "outcome_access_status_at_configuration_fix",
        },
        "analysis_timing fields differ",
    )
    _nonempty_string(timing["evidence_timing_label"], "evidence_timing_label")
    require(
        isinstance(timing["configuration_fixed_before_utility_access"], bool),
        "configuration_fixed_before_utility_access must be boolean",
    )
    require(
        timing["outcome_access_status_at_configuration_fix"]
        in {"NOT_ACCESSED", "ACCESSED", "SYNTHETIC_NO_EMPIRICAL_OUTCOME"},
        "outcome_access_status_at_configuration_fix is invalid",
    )
    if timing["outcome_access_status_at_configuration_fix"] == "NOT_ACCESSED":
        require(
            timing["configuration_fixed_before_utility_access"] is True,
            "timing fields are inconsistent",
        )
    if timing["outcome_access_status_at_configuration_fix"] == "ACCESSED":
        require(
            timing["configuration_fixed_before_utility_access"] is False,
            "timing fields are inconsistent",
        )
    if timing["outcome_access_status_at_configuration_fix"] == "SYNTHETIC_NO_EMPIRICAL_OUTCOME":
        require(
            timing["configuration_fixed_before_utility_access"] is True,
            "timing fields are inconsistent",
        )

    family = config["comparison_family"]
    require(isinstance(family, dict), "comparison_family must be an object")
    require(
        set(family)
        == {"allocation_order", "method_order", "directionality", "utility_orientation"},
        "comparison_family fields differ",
    )
    allocation_order = _unique_strings(family["allocation_order"], "allocation_order")
    require(allocation_order == allocation_ids, "allocation_order must equal allocations order")
    methods = _unique_strings(family["method_order"], "method_order", minimum=2)
    require(methods == configuration_ids, "method_order must equal configurations order")
    require(family["directionality"] == "ordered", "only ordered directionality is registered")
    require(
        family["utility_orientation"] in {"higher_is_better", "lower_is_better"},
        "utility_orientation is invalid",
    )

    inference = config["inference"]
    require(isinstance(inference, dict), "inference must be an object")
    require(
        set(inference)
        == {
            "decision_rule_id",
            "alpha",
            "fair_binomial_probability",
            "tie_rule",
            "missing_unit_rule",
            "holm_scope",
        },
        "inference fields differ",
    )
    require(inference["decision_rule_id"] == DECISION_RULE, "decision rule differs")
    require(
        isinstance(inference["alpha"], (int, float))
        and not isinstance(inference["alpha"], bool)
        and 0 < inference["alpha"] < 1,
        "alpha must lie strictly between zero and one",
    )
    require(inference["fair_binomial_probability"] == 0.5, "fair binomial probability must be 0.5")
    require(inference["tie_rule"] == TIE_RULE, "tie rule differs")
    require(inference["missing_unit_rule"] == MISSING_UNIT_RULE, "missing-unit rule differs")
    require(inference["holm_scope"] == HOLM_SCOPE, "Holm scope differs")

    # Keep these reads alive as explicit contract checks rather than accepting
    # unused but syntactically valid sections.
    require(bool(datasets and methods), "empty comparison geometry")
    return config


def complete_ordered_family_size(config: dict[str, Any]) -> int:
    method_count = len(config["comparison_family"]["method_order"])
    return method_count * (method_count - 1)


def k_min(config: dict[str, Any]) -> int:
    family_size = complete_ordered_family_size(config)
    alpha = Decimal(str(config["inference"]["alpha"]))
    threshold = alpha / Decimal(family_size)
    value = Decimal(1)
    count = 0
    while value > threshold:
        value /= Decimal(2)
        count += 1
    return count


def validation_report(config: dict[str, Any]) -> dict[str, Any]:
    independence = config["unit_contract"]["independence_justification"]
    return {
        "audit_id": config["audit_id"],
        "complete_ordered_comparisons_per_family": complete_ordered_family_size(config),
        "configuration_identity_count": len(config["configurations"]),
        "dataset_count": len(config["dataset_ids"]),
        "decision_rule_id": config["inference"]["decision_rule_id"],
        "declared_unit_independence_established": (
            independence["status"] == INDEPENDENCE_ESTABLISHED
        ),
        "evidence_timing_label": config["analysis_timing"]["evidence_timing_label"],
        "k_min_for_smallest_holm_threshold": k_min(config),
        "method_count": len(config["comparison_family"]["method_order"]),
        "reference_mapping_scope": config["reference_declaration"]["mapping_scope"],
        "reference_role_count": len(config["reference_roles"]),
        "status": "PASS_DECLARATIVE_CONTRACT",
        "tie_rule": config["inference"]["tie_rule"],
        "upstream_provenance_machine_verified": False,
    }


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def sha256_file(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_tsv(
    path: Path,
    *,
    allow_empty: bool = False,
) -> tuple[list[str], list[dict[str, str]]]:
    require(path.is_file() and not path.is_symlink(), f"input must be a regular file: {path}")
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            header = reader.fieldnames
            require(header is not None and header, "TSV header is missing")
            require(None not in header and len(header) == len(set(header)), "TSV header is invalid")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise AuditError("input is not valid UTF-8 TSV") from error
    require(allow_empty or rows, "TSV has no data rows")
    require(all(None not in row for row in rows), "TSV row width differs from header")
    return list(header), rows


def _read_json(path: Path, label: str) -> Any:
    require(path.is_file() and not path.is_symlink(), f"{label} must be a regular file")
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AuditError(f"{label} is not valid UTF-8 JSON") from error


def _verify_closed_manifest(bundle: Path, expected_tree: set[str]) -> None:
    require(bundle.is_dir() and not bundle.is_symlink(), "unit-audit bundle must be a directory")
    entries = list(bundle.iterdir())
    require(
        all(path.is_file() and not path.is_symlink() for path in entries),
        "unit-audit bundle contains a non-regular entry",
    )
    names = {path.name for path in entries}
    require(names == expected_tree, "unit-audit bundle tree differs")
    try:
        lines = (bundle / MANIFEST).read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise AuditError("unit-audit manifest is not ASCII") from error
    declared: dict[str, str] = {}
    for line in lines:
        parts = line.split("  ", 1)
        require(len(parts) == 2, "unit-audit manifest syntax differs")
        digest, name = parts
        require(
            len(digest) == 64 and all(character in "0123456789abcdef" for character in digest),
            "unit-audit manifest digest is invalid",
        )
        require(
            name not in declared and "/" not in name and name != MANIFEST,
            "unit-audit manifest name is invalid",
        )
        declared[name] = digest
    require(list(declared) == sorted(declared), "unit-audit manifest order differs")
    require(set(declared) == names - {MANIFEST}, "unit-audit manifest is not closed")
    for name, digest in declared.items():
        require(sha256_file(bundle / name) == digest, f"unit-audit manifest digest differs: {name}")


def _validated_unit_audit_support(
    config_path: Path,
    config: dict[str, Any],
    bundle: Path,
) -> tuple[dict[str, list[str]], bytes, bytes, bytes, str]:
    _verify_closed_manifest(bundle, UNIT_AUDIT_TREE)
    require(
        (bundle / "CONFIG_SNAPSHOT.json").read_bytes() == config_path.read_bytes(),
        "unit-audit configuration snapshot differs",
    )
    metadata_path = bundle / "INPUT_METADATA.tsv"
    unit = config["unit_contract"]
    dataset_column = unit["dataset_column"]
    object_column = unit["object_id_column"]
    unit_column = unit["unit_id_column"]
    declared_columns = unit["declared_unit_columns"]
    candidate_columns = unit["candidate_acquisition_group_columns"]
    required = [dataset_column, object_column, unit_column, *declared_columns, *candidate_columns]
    header, source_rows = _read_tsv(metadata_path)
    require(set(required).issubset(header), "unit-audit metadata is missing configured columns")
    require(
        all(all(row[column] != "" for column in required) for row in source_rows),
        "unit-audit metadata contains an empty configured value",
    )
    require(
        all(row[dataset_column] in config["dataset_ids"] for row in source_rows),
        "unit-audit metadata dataset is not registered",
    )
    require(
        {row[dataset_column] for row in source_rows} == set(config["dataset_ids"]),
        "unit-audit metadata does not cover every dataset",
    )
    object_keys = [(row[dataset_column], row[object_column]) for row in source_rows]
    require(len(object_keys) == len(set(object_keys)), "unit-audit metadata has duplicate objects")
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in sorted(
        source_rows,
        key=lambda item: (item[dataset_column], item[unit_column], item[object_column]),
    ):
        groups.setdefault((row[dataset_column], row[unit_column]), []).append(row)
    independence = unit["independence_justification"]
    expected_hierarchy_rows: list[dict[str, Any]] = []
    expected_unit_rows: list[dict[str, Any]] = []
    representatives: dict[tuple[str, str], dict[str, str]] = {}
    for (dataset_id, unit_id), records in sorted(groups.items()):
        for column in [*declared_columns, *candidate_columns]:
            require(
                len({row[column] for row in records}) == 1,
                f"unit-audit metadata is inconsistent within unit for {column}",
            )
        representative = records[0]
        representatives[(dataset_id, unit_id)] = representative
        for record in records:
            expected_hierarchy_rows.append(
                {
                    "dataset_id": dataset_id,
                    "object_id": record[object_column],
                    "unit_id": unit_id,
                    **{column: record[column] for column in declared_columns},
                    "mapping_status": "OBJECT_MAPPED_TO_DECLARED_UNIT",
                }
            )
        expected_unit_rows.append(
            {
                "dataset_id": dataset_id,
                "unit_id": unit_id,
                **{column: representative[column] for column in declared_columns},
                "object_count": len(records),
                "repeated_objects_beyond_first": len(records) - 1,
                "independence_status": independence["status"],
            }
        )
    hierarchy_header = [
        "dataset_id",
        "object_id",
        "unit_id",
        *declared_columns,
        "mapping_status",
    ]
    observed_header, observed_hierarchy_rows = _read_tsv(bundle / "UNIT_HIERARCHY_V1.tsv")
    require(observed_header == hierarchy_header, "unit-audit hierarchy header differs")
    require(
        observed_hierarchy_rows
        == [
            {key: str(value) for key, value in row.items()}
            for row in expected_hierarchy_rows
        ],
        "unit-audit hierarchy differs from its metadata snapshot",
    )
    observed_header, observed_unit_rows = _read_tsv(bundle / "UNIT_AUDIT_V1.tsv")
    expected_header = [
        "dataset_id",
        "unit_id",
        *declared_columns,
        "object_count",
        "repeated_objects_beyond_first",
        "independence_status",
    ]
    require(observed_header == expected_header, "unit-audit unit table header differs")
    require(
        observed_unit_rows
        == [{key: str(value) for key, value in row.items()} for row in expected_unit_rows],
        "unit-audit unit table differs from its metadata snapshot",
    )
    established = independence["status"] == INDEPENDENCE_ESTABLISHED
    receipt_path = bundle / "AUDIT_RECEIPT_V1.json"
    receipt = _read_json(receipt_path, "unit-audit receipt")
    require(isinstance(receipt, dict), "unit-audit receipt must be an object")
    expected_geometry_rows: list[dict[str, Any]] = []
    expected_edge_rows: list[dict[str, Any]] = []
    for dataset_id in config["dataset_ids"]:
        dataset_representatives = {
            unit_id: record
            for (data, unit_id), record in representatives.items()
            if data == dataset_id
        }
        for column in candidate_columns:
            candidate_groups: dict[str, list[str]] = {}
            for unit_id, row in dataset_representatives.items():
                candidate_groups.setdefault(row[column], []).append(unit_id)
            for value, member_ids in sorted(candidate_groups.items()):
                members = sorted(member_ids)
                expected_geometry_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "candidate_group_column": column,
                        "candidate_group_value": value,
                        "declared_unit_count": len(members),
                        "object_count": sum(
                            len(groups[(dataset_id, member)]) for member in members
                        ),
                        "member_unit_ids": ";".join(members),
                        "interpretation": GEOMETRY_INTERPRETATION,
                    }
                )
                for left, right in itertools.combinations(members, 2):
                    expected_edge_rows.append(
                        {
                            "dataset_id": dataset_id,
                            "candidate_group_column": column,
                            "candidate_group_value": value,
                            "unit_id_a": left,
                            "unit_id_b": right,
                            "interpretation": GEOMETRY_INTERPRETATION,
                        }
                    )
    geometry_header = [
        "dataset_id",
        "candidate_group_column",
        "candidate_group_value",
        "declared_unit_count",
        "object_count",
        "member_unit_ids",
        "interpretation",
    ]
    observed_header, observed_geometry_rows = _read_tsv(
        bundle / "UNIT_GROUP_GEOMETRY_V1.tsv",
        allow_empty=True,
    )
    require(observed_header == geometry_header, "unit-audit geometry header differs")
    require(
        observed_geometry_rows
        == [{key: str(value) for key, value in row.items()} for row in expected_geometry_rows],
        "unit-audit geometry differs from its metadata snapshot",
    )
    edge_header = [
        "dataset_id",
        "candidate_group_column",
        "candidate_group_value",
        "unit_id_a",
        "unit_id_b",
        "interpretation",
    ]
    observed_header, observed_edge_rows = _read_tsv(
        bundle / "ACQUISITION_EDGE_CANDIDATES_V1.tsv",
        allow_empty=True,
    )
    require(observed_header == edge_header, "unit-audit edge header differs")
    require(
        observed_edge_rows
        == [{key: str(value) for key, value in row.items()} for row in expected_edge_rows],
        "unit-audit edges differ from their metadata snapshot",
    )
    require(
        (bundle / "REPORTING_CHECKLIST_V1.md").read_text(encoding="utf-8")
        == unit_audit_checklist(),
        "unit-audit checklist differs",
    )
    expected_receipt = {
        "audit_id": config["audit_id"],
        "bundle_kind": "unit_audit",
        "candidate_edge_record_count": len(expected_edge_rows),
        "candidate_group_record_count": len(expected_geometry_rows),
        "config_sha256": sha256_file(config_path),
        "declared_unit_count": len(expected_unit_rows),
        "declared_unit_independence_established": established,
        "independence_assertion_authority": independence["assertion_authority"],
        "independence_assertion_basis": independence["basis"],
        "independence_assertion_machine_verified": False,
        "metadata_sha256": sha256_file(metadata_path),
        "object_count": len(expected_hierarchy_rows),
        "record_type": "BENCHMARK_DESIGN_AUDIT_RECEIPT_V1",
        "schema_version": "1.0.0",
        "status": (
            "PASS_GEOMETRY_WITH_DECLARED_INDEPENDENCE"
            if established
            else "WITHHELD_UNIT_INDEPENDENCE_NOT_ESTABLISHED"
        ),
    }
    require(receipt == expected_receipt, "unit-audit receipt contract differs")
    require(
        receipt_path.read_text(encoding="utf-8") == canonical_json(expected_receipt),
        "unit-audit receipt encoding differs",
    )
    support = {
        dataset_id: sorted(unit_id for data, unit_id in groups if data == dataset_id)
        for dataset_id in config["dataset_ids"]
    }
    return (
        support,
        metadata_path.read_bytes(),
        receipt_path.read_bytes(),
        (bundle / MANIFEST).read_bytes(),
        sha256_file(receipt_path),
    )


def _write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_closed_bundle(
    config_path: Path,
    output_dir: Path,
    files: dict[str, bytes | str],
) -> None:
    require(not output_dir.exists() and not output_dir.is_symlink(), "output directory must not exist")
    require(output_dir.parent.is_dir() and not output_dir.parent.is_symlink(), "output parent is invalid")
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging.", dir=output_dir.parent))
    try:
        (staging / "CONFIG_SNAPSHOT.json").write_bytes(config_path.read_bytes())
        for name, value in files.items():
            require("/" not in name and name not in {"", ".", "..", MANIFEST}, "invalid output name")
            if isinstance(value, bytes):
                (staging / name).write_bytes(value)
            else:
                (staging / name).write_text(value, encoding="utf-8")
        names = sorted(path.name for path in staging.iterdir())
        manifest = "".join(f"{sha256_file(staging / name)}  {name}\n" for name in names)
        (staging / MANIFEST).write_text(manifest, encoding="ascii")
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _rows_as_tsv(fieldnames: list[str], rows: list[dict[str, Any]]) -> str:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=fieldnames,
        delimiter="\t",
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def unit_audit_checklist() -> str:
    return """# Acquisition-unit audit\n\n## Verified\n\n- Experimental-unit identifiers are declared.\n- The number of observations within each declared unit is reported.\n- Candidate acquisition-group structure is reported for each supplied grouping rule.\n- The analyst's independence statement and its basis are recorded.\n\n## Scope\n\n- This metadata audit does not establish that the declared units are independent.\n- Candidate links require study-specific protocol knowledge before they can define inferential units.\n"""


def audit_units(config_path: Path, metadata_path: Path, output_dir: Path) -> None:
    config = load_config(config_path)
    unit = config["unit_contract"]
    dataset_column = unit["dataset_column"]
    object_column = unit["object_id_column"]
    unit_column = unit["unit_id_column"]
    declared_columns = unit["declared_unit_columns"]
    candidate_columns = unit["candidate_acquisition_group_columns"]
    independence = unit["independence_justification"]
    independence_established = independence["status"] == INDEPENDENCE_ESTABLISHED
    required = [dataset_column, object_column, unit_column, *declared_columns, *candidate_columns]
    header, source_rows = _read_tsv(metadata_path)
    require(set(required).issubset(header), "metadata is missing configured columns")
    for row in source_rows:
        require(all(row[column] != "" for column in required), "metadata contains an empty configured value")
        require(row[dataset_column] in config["dataset_ids"], "metadata dataset is not registered")
    require(
        {row[dataset_column] for row in source_rows} == set(config["dataset_ids"]),
        "metadata does not cover every registered dataset",
    )
    object_keys = [(row[dataset_column], row[object_column]) for row in source_rows]
    require(len(object_keys) == len(set(object_keys)), "metadata has duplicate object identifiers")

    sorted_source = sorted(
        source_rows,
        key=lambda row: (row[dataset_column], row[unit_column], row[object_column]),
    )
    units: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in sorted_source:
        units.setdefault((row[dataset_column], row[unit_column]), []).append(row)

    hierarchy_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    unit_records: dict[tuple[str, str], dict[str, str]] = {}
    for (dataset_id, unit_id), records in sorted(units.items()):
        for column in [*declared_columns, *candidate_columns]:
            require(
                len({record[column] for record in records}) == 1,
                f"metadata is inconsistent within declared unit for {column}",
            )
        representative = records[0]
        unit_records[(dataset_id, unit_id)] = representative
        for record in records:
            hierarchy_rows.append(
                {
                    "dataset_id": dataset_id,
                    "object_id": record[object_column],
                    "unit_id": unit_id,
                    **{column: record[column] for column in declared_columns},
                    "mapping_status": "OBJECT_MAPPED_TO_DECLARED_UNIT",
                }
            )
        unit_rows.append(
            {
                "dataset_id": dataset_id,
                "unit_id": unit_id,
                **{column: representative[column] for column in declared_columns},
                "object_count": len(records),
                "repeated_objects_beyond_first": len(records) - 1,
                "independence_status": independence["status"],
            }
        )

    geometry_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    for dataset_id in config["dataset_ids"]:
        dataset_records = {
            unit_id: record
            for (record_dataset, unit_id), record in unit_records.items()
            if record_dataset == dataset_id
        }
        for column in candidate_columns:
            groups: dict[str, list[str]] = {}
            for unit_id, record in dataset_records.items():
                groups.setdefault(record[column], []).append(unit_id)
            for value, member_ids in sorted(groups.items()):
                members = sorted(member_ids)
                object_count = sum(len(units[(dataset_id, member)]) for member in members)
                geometry_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "candidate_group_column": column,
                        "candidate_group_value": value,
                        "declared_unit_count": len(members),
                        "object_count": object_count,
                        "member_unit_ids": ";".join(members),
                        "interpretation": GEOMETRY_INTERPRETATION,
                    }
                )
                for left, right in itertools.combinations(members, 2):
                    edge_rows.append(
                        {
                            "dataset_id": dataset_id,
                            "candidate_group_column": column,
                            "candidate_group_value": value,
                            "unit_id_a": left,
                            "unit_id_b": right,
                            "interpretation": GEOMETRY_INTERPRETATION,
                        }
                    )

    hierarchy_fields = ["dataset_id", "object_id", "unit_id", *declared_columns, "mapping_status"]
    unit_fields = [
        "dataset_id",
        "unit_id",
        *declared_columns,
        "object_count",
        "repeated_objects_beyond_first",
        "independence_status",
    ]
    geometry_fields = [
        "dataset_id",
        "candidate_group_column",
        "candidate_group_value",
        "declared_unit_count",
        "object_count",
        "member_unit_ids",
        "interpretation",
    ]
    edge_fields = [
        "dataset_id",
        "candidate_group_column",
        "candidate_group_value",
        "unit_id_a",
        "unit_id_b",
        "interpretation",
    ]
    receipt = {
        "audit_id": config["audit_id"],
        "bundle_kind": "unit_audit",
        "candidate_edge_record_count": len(edge_rows),
        "candidate_group_record_count": len(geometry_rows),
        "config_sha256": sha256_file(config_path),
        "declared_unit_count": len(unit_rows),
        "declared_unit_independence_established": independence_established,
        "independence_assertion_authority": independence["assertion_authority"],
        "independence_assertion_basis": independence["basis"],
        "independence_assertion_machine_verified": False,
        "metadata_sha256": sha256_file(metadata_path),
        "object_count": len(hierarchy_rows),
        "record_type": "BENCHMARK_DESIGN_AUDIT_RECEIPT_V1",
        "schema_version": "1.0.0",
        "status": (
            "PASS_GEOMETRY_WITH_DECLARED_INDEPENDENCE"
            if independence_established
            else "WITHHELD_UNIT_INDEPENDENCE_NOT_ESTABLISHED"
        ),
    }
    _write_closed_bundle(
        config_path,
        output_dir,
        {
            "ACQUISITION_EDGE_CANDIDATES_V1.tsv": _rows_as_tsv(edge_fields, edge_rows),
            "AUDIT_RECEIPT_V1.json": canonical_json(receipt),
            "INPUT_METADATA.tsv": metadata_path.read_bytes(),
            "REPORTING_CHECKLIST_V1.md": unit_audit_checklist(),
            "UNIT_AUDIT_V1.tsv": _rows_as_tsv(unit_fields, unit_rows),
            "UNIT_GROUP_GEOMETRY_V1.tsv": _rows_as_tsv(geometry_fields, geometry_rows),
            "UNIT_HIERARCHY_V1.tsv": _rows_as_tsv(hierarchy_fields, hierarchy_rows),
        },
    )


def comparison_plan_checklist() -> str:
    return """# Comparison plan\n\n## Verified\n\n- The observed-effect, prediction-centering and model-conditioning roles are explicit.\n- Each declared allocation and rotation maps roles to blocks.\n- Per-role depth labels and the rotation-averaging rule are recorded.\n- Declared overlap agrees with the overlap implied by the labels, with a 3 × 3 matrix for each rotation.\n- Every ordered method comparison under the declared decision rule is included.\n- `K_min` uses the smallest complete-family Holm threshold and an all-win fair-binomial tail.\n\n## Scope\n\n- Block and depth labels are declarations; this software does not inspect cell membership.\n- `K_min` describes arithmetic attainability, not evidence for an effect or unit independence.\n"""


def _overlap_pattern(rotation: dict[str, str]) -> str:
    blocks = [rotation[role] for role in REFERENCE_ROLES]
    if len(set(blocks)) == 1:
        return "ALL_THREE_ROLES_SHARE_ONE_BLOCK"
    if len(set(blocks)) == len(blocks):
        return "ALL_THREE_ROLES_USE_DISTINCT_BLOCKS"
    return "PARTIAL_ROLE_OVERLAP"


def plan_comparisons(config_path: Path, output_dir: Path) -> None:
    config = load_config(config_path)
    allocation_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    allocation_by_id = {allocation["allocation_id"]: allocation for allocation in config["allocations"]}
    for allocation_id in config["comparison_family"]["allocation_order"]:
        allocation = allocation_by_id[allocation_id]
        for rotation in allocation["rotations"]:
            allocation_rows.append(
                {
                    "allocation_id": allocation_id,
                    "reference_depth_label": allocation["reference_depth_label"],
                    "reference_depth_unit": allocation["reference_depth_unit"],
                    "aggregation_order": allocation["aggregation_order"],
                    "rotation_id": rotation["rotation_id"],
                    **{role: rotation[role] for role in REFERENCE_ROLES},
                    **{
                        f"{role}_depth_label": rotation["role_depth_labels"][role]
                        for role in REFERENCE_ROLES
                    },
                    "overlap_pattern": _overlap_pattern(rotation),
                }
            )
            for row_role in REFERENCE_ROLES:
                for column_role in REFERENCE_ROLES:
                    overlap_rows.append(
                        {
                            "allocation_id": allocation_id,
                            "rotation_id": rotation["rotation_id"],
                            "row_role": row_role,
                            "column_role": column_role,
                            "row_block": rotation[row_role],
                            "column_block": rotation[column_role],
                            "shares_control_block": int(rotation[row_role] == rotation[column_role]),
                        }
                    )

    methods = config["comparison_family"]["method_order"]
    family_rows: list[dict[str, Any]] = []
    ordinal = 0
    for dataset_id in config["dataset_ids"]:
        for allocation_id in config["comparison_family"]["allocation_order"]:
            for model_a in methods:
                for model_b in methods:
                    if model_a == model_b:
                        continue
                    ordinal += 1
                    family_rows.append(
                        {
                            "comparison_ordinal": ordinal,
                            "comparison_id": f"{dataset_id}::{allocation_id}::{model_a}>{model_b}",
                            "dataset_id": dataset_id,
                            "allocation_id": allocation_id,
                            "model_a_id": model_a,
                            "model_b_id": model_b,
                            "directionality": "ORDERED_A_BETTER_THAN_B",
                            "utility_orientation": config["comparison_family"]["utility_orientation"],
                            "decision_rule_id": config["inference"]["decision_rule_id"],
                        }
                    )

    family_size = complete_ordered_family_size(config)
    required_k = k_min(config)
    alpha = Decimal(str(config["inference"]["alpha"]))
    threshold = alpha / Decimal(family_size)
    minimum_tail = Decimal(1) / Decimal(2**required_k)
    resolution_rows = [
        {
            "dataset_id": dataset_id,
            "allocation_id": allocation_id,
            "complete_family_size": family_size,
            "alpha": format(alpha, "f"),
            "smallest_holm_threshold": format(threshold, "f"),
            "k_min": required_k,
            "all_win_fair_binomial_tail_at_k_min": format(minimum_tail, "f"),
            "tie_rule": config["inference"]["tie_rule"],
            "interpretation": "ATTAINABILITY_ONLY_NOT_AN_EFFECT_OR_INDEPENDENCE_CLAIM",
        }
        for dataset_id in config["dataset_ids"]
        for allocation_id in config["comparison_family"]["allocation_order"]
    ]

    allocation_fields = [
        "allocation_id",
        "reference_depth_label",
        "reference_depth_unit",
        "aggregation_order",
        "rotation_id",
        *REFERENCE_ROLES,
        *[f"{role}_depth_label" for role in REFERENCE_ROLES],
        "overlap_pattern",
    ]
    overlap_fields = [
        "allocation_id",
        "rotation_id",
        "row_role",
        "column_role",
        "row_block",
        "column_block",
        "shares_control_block",
    ]
    family_fields = [
        "comparison_ordinal",
        "comparison_id",
        "dataset_id",
        "allocation_id",
        "model_a_id",
        "model_b_id",
        "directionality",
        "utility_orientation",
        "decision_rule_id",
    ]
    resolution_fields = [
        "dataset_id",
        "allocation_id",
        "complete_family_size",
        "alpha",
        "smallest_holm_threshold",
        "k_min",
        "all_win_fair_binomial_tail_at_k_min",
        "tie_rule",
        "interpretation",
    ]
    receipt = {
        "allocation_count": len(config["allocations"]),
        "allocation_rotation_count": len(allocation_rows),
        "audit_id": config["audit_id"],
        "bundle_kind": "comparison_plan",
        "complete_ordered_family_size_per_dataset_allocation": family_size,
        "config_sha256": sha256_file(config_path),
        "configuration_identity_count": len(config["configurations"]),
        "decision_rule_id": config["inference"]["decision_rule_id"],
        "evidence_timing_label": config["analysis_timing"]["evidence_timing_label"],
        "k_min": required_k,
        "planned_comparison_count": len(family_rows),
        "record_type": "BENCHMARK_DESIGN_AUDIT_RECEIPT_V1",
        "schema_version": "1.0.0",
        "status": "PASS_COMPLETE_ORDERED_PLAN",
    }
    _write_closed_bundle(
        config_path,
        output_dir,
        {
            "KMIN_RESOLUTION_V1.tsv": _rows_as_tsv(resolution_fields, resolution_rows),
            "PAIRWISE_FAMILY_PLAN_V1.tsv": _rows_as_tsv(family_fields, family_rows),
            "PLAN_RECEIPT_V1.json": canonical_json(receipt),
            "REFERENCE_ROLE_ALLOCATIONS_V1.tsv": _rows_as_tsv(allocation_fields, allocation_rows),
            "REPORTING_CHECKLIST_V1.md": comparison_plan_checklist(),
            "ROLE_OVERLAP_MATRIX_V1.tsv": _rows_as_tsv(overlap_fields, overlap_rows),
        },
    )


def evaluation_checklist() -> str:
    return """# Utility evaluation\n\n## Verified\n\n- Input consists only of declared, pre-aggregated experimental-unit utilities.\n- The evaluation uses the supplied unit-audit bundle and exactly the same unit support.\n- Every dataset-allocation family has complete unit-by-method support.\n- All ordered pairs are generated from the declared method order.\n- Ties are literal IEEE-754 binary64 zero differences; no tolerance is used.\n- One-sided fair-binomial tails are exact, and Holm correction covers the complete ordered family.\n- `DIRECTION`, `NO_DIRECTION` and `WITHHELD` are reported separately, with below-`K_min` and failed-declaration reasons retained.\n- Directional rejection is withheld unless independence is declared established.\n\n## Scope\n\n- Independence and training/evaluation separation are analyst-supplied statements, not machine-established facts.\n- Pre-aggregated utilities cannot verify upstream cells, memberships, predictions or model states.\n- This deterministic evaluator does not perform an optimization search.\n- Ranks and means are descriptive; they do not identify a uniquely supported winner.\n"""


def _fraction_text(value: Fraction) -> str:
    decimal = Decimal(value.numerator) / Decimal(value.denominator)
    rendered = format(decimal, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _exact_upper_fair_binomial_tail(wins: int, total: int) -> Fraction:
    require(0 <= wins <= total, "invalid sign counts")
    if total == 0:
        return Fraction(1, 1)
    numerator = sum(math.comb(total, value) for value in range(wins, total + 1))
    return Fraction(numerator, 2**total)


def _format_float(value: float) -> str:
    require(math.isfinite(value), "non-finite derived value")
    return format(value, ".17g")


def evaluate_utilities(
    config_path: Path,
    utilities_path: Path,
    unit_audit_bundle: Path,
    output_dir: Path,
) -> None:
    config = load_config(config_path)
    (
        audited_support,
        unit_metadata_bytes,
        unit_audit_receipt_bytes,
        unit_audit_manifest_bytes,
        unit_audit_receipt_sha256,
    ) = _validated_unit_audit_support(config_path, config, unit_audit_bundle)
    utility_contract = config["utility_input_contract"]
    expected_header = utility_contract["input_columns"]
    header, source_rows = _read_tsv(utilities_path)
    require(header == expected_header, "utility TSV header differs")
    expected_input_sha256 = utility_contract["expected_input_sha256"]
    if expected_input_sha256 is not None:
        require(
            sha256_file(utilities_path) == expected_input_sha256,
            "utility input SHA-256 differs from the declared source binding",
        )
    for row in source_rows:
        for column, expected in utility_contract["required_constant_values"].items():
            require(row[column] == expected, f"utility constant differs: {column}")
    dataset_column = utility_contract["dataset_id_column"]
    allocation_column = utility_contract["allocation_id_column"]
    unit_column = utility_contract["unit_id_column"]
    method_column = utility_contract["method_id_column"]
    value_column = utility_contract["utility_column"]
    datasets = config["dataset_ids"]
    allocations = config["comparison_family"]["allocation_order"]
    methods = config["comparison_family"]["method_order"]
    dataset_index = {value: index for index, value in enumerate(datasets)}
    allocation_index = {value: index for index, value in enumerate(allocations)}
    method_index = {value: index for index, value in enumerate(methods)}
    values: dict[tuple[str, str, str, str], float] = {}
    for row in source_rows:
        dataset_id = row[dataset_column]
        allocation_id = row[allocation_column]
        unit_id = row[unit_column]
        method_id = row[method_column]
        require(dataset_id in dataset_index, "utility dataset is not registered")
        require(allocation_id in allocation_index, "utility allocation is not registered")
        require(method_id in method_index, "utility method is not registered")
        require(unit_id != "" and unit_id.strip() == unit_id, "utility unit_id is invalid")
        try:
            utility = float(row[value_column])
        except ValueError as error:
            raise AuditError("utility is not binary64 numeric") from error
        require(math.isfinite(utility), "utility must be finite")
        key = (dataset_id, allocation_id, unit_id, method_id)
        require(key not in values, "duplicate utility support row")
        values[key] = utility

    units_by_dataset_allocation: dict[tuple[str, str], list[str]] = {}
    for dataset_id in datasets:
        shared_units: list[str] | None = None
        for allocation_id in allocations:
            observed_units = sorted(
                {
                    unit_id
                    for value_dataset, value_allocation, unit_id, _ in values
                    if value_dataset == dataset_id and value_allocation == allocation_id
                }
            )
            require(observed_units, "dataset-allocation has no utility units")
            require(
                observed_units == audited_support[dataset_id],
                "utility units differ from the verified unit-audit support",
            )
            expected_keys = {
                (dataset_id, allocation_id, unit_id, method_id)
                for unit_id in observed_units
                for method_id in methods
            }
            observed_keys = {
                key for key in values if key[0] == dataset_id and key[1] == allocation_id
            }
            require(observed_keys == expected_keys, "missing or extraneous unit-by-method support")
            if shared_units is None:
                shared_units = observed_units
            else:
                require(observed_units == shared_units, "allocation unit support differs within dataset")
            units_by_dataset_allocation[(dataset_id, allocation_id)] = observed_units
    expected_total = sum(
        len(units_by_dataset_allocation[(dataset_id, allocation_id)]) * len(methods)
        for dataset_id in datasets
        for allocation_id in allocations
    )
    require(len(values) == expected_total, "utility support is not exactly the registered design")

    normalized_rows: list[dict[str, Any]] = []
    for dataset_id in datasets:
        for allocation_id in allocations:
            for unit_id in units_by_dataset_allocation[(dataset_id, allocation_id)]:
                for method_id in methods:
                    utility = values[(dataset_id, allocation_id, unit_id, method_id)]
                    normalized_rows.append(
                        {
                            "dataset_id": dataset_id,
                            "allocation_id": allocation_id,
                            "unit_id": unit_id,
                            "method_id": method_id,
                            "utility": _format_float(utility),
                            "utility_hex": utility.hex(),
                        }
                    )

    family_size = complete_ordered_family_size(config)
    required_k = k_min(config)
    alpha = Fraction(str(config["inference"]["alpha"]))
    orientation = config["comparison_family"]["utility_orientation"]
    orientation_multiplier = 1.0 if orientation == "higher_is_better" else -1.0
    independence_established = (
        config["unit_contract"]["independence_justification"]["status"]
        == INDEPENDENCE_ESTABLISHED
    )
    training_evaluation_separation_established = all(
        configuration["training_evaluation_separation"]["status"]
        in {TRAIN_EVAL_ESTABLISHED, TRAIN_EVAL_NOT_APPLICABLE}
        for configuration in config["configurations"]
    )
    pairwise_rows: list[dict[str, Any]] = []
    for dataset_id in datasets:
        for allocation_id in allocations:
            units = units_by_dataset_allocation[(dataset_id, allocation_id)]
            family_records: list[dict[str, Any]] = []
            for model_a in methods:
                for model_b in methods:
                    if model_a == model_b:
                        continue
                    differences = [
                        orientation_multiplier
                        * (
                            values[(dataset_id, allocation_id, unit_id, model_a)]
                            - values[(dataset_id, allocation_id, unit_id, model_b)]
                        )
                        for unit_id in units
                    ]
                    wins = sum(value > 0.0 for value in differences)
                    losses = sum(value < 0.0 for value in differences)
                    ties = sum(value == 0.0 for value in differences)
                    require(wins + losses + ties == len(units), "sign classification is not exhaustive")
                    effective = wins + losses
                    raw = _exact_upper_fair_binomial_tail(wins, effective)
                    mean_difference = math.fsum(differences) / len(differences)
                    family_records.append(
                        {
                            "comparison_id": f"{dataset_id}::{allocation_id}::{model_a}>{model_b}",
                            "dataset_id": dataset_id,
                            "allocation_id": allocation_id,
                            "model_a_id": model_a,
                            "model_b_id": model_b,
                            "declared_unit_count": len(units),
                            "effective_non_tie_units": effective,
                            "wins": wins,
                            "losses": losses,
                            "literal_zero_ties": ties,
                            "equal_unit_mean_oriented_difference": _format_float(mean_difference),
                            "equal_unit_mean_oriented_difference_hex": mean_difference.hex(),
                            "exact_one_sided_p_numerator": raw.numerator,
                            "exact_one_sided_p_denominator": raw.denominator,
                            "exact_one_sided_p": _fraction_text(raw),
                            "_raw": raw,
                            "_below_kmin": effective < required_k,
                            "evidence_scope": (
                                "DECLARATIVE_DOWNSTREAM_AUDIT_NO_UPSTREAM_CELL_OR_MODEL_PROVENANCE_VERIFICATION"
                            ),
                        }
                    )
            require(len(family_records) == family_size, "complete ordered family differs")
            ordered = sorted(
                family_records,
                key=lambda row: (
                    row["_raw"],
                    method_index[row["model_a_id"]],
                    method_index[row["model_b_id"]],
                ),
            )
            running = Fraction(0, 1)
            for rank, row in enumerate(ordered, start=1):
                adjusted = min(Fraction(1, 1), row["_raw"] * (family_size - rank + 1))
                running = max(running, adjusted)
                row["_adjusted"] = running
            for row in family_records:
                adjusted = row.pop("_adjusted")
                row.pop("_raw")
                row["holm_adjusted_p_numerator"] = adjusted.numerator
                row["holm_adjusted_p_denominator"] = adjusted.denominator
                row["holm_adjusted_p"] = _fraction_text(adjusted)
                if not independence_established:
                    decision = DECISION_WITHHELD
                    withheld_reason = WITHHELD_INDEPENDENCE
                elif not training_evaluation_separation_established:
                    decision = DECISION_WITHHELD
                    withheld_reason = WITHHELD_TRAIN_EVAL
                elif row["_below_kmin"]:
                    decision = DECISION_WITHHELD
                    withheld_reason = WITHHELD_BELOW_KMIN
                elif adjusted <= alpha:
                    decision = DECISION_REJECTION
                    withheld_reason = ""
                else:
                    decision = DECISION_NO_DIRECTION
                    withheld_reason = ""
                row["reject_at_alpha"] = (
                    int(decision == DECISION_REJECTION)
                    if decision in {DECISION_REJECTION, DECISION_NO_DIRECTION}
                    else "NA"
                )
                row["decision"] = decision
                row["withheld_reason"] = withheld_reason
                row.pop("_below_kmin")
                row["alpha"] = _fraction_text(alpha)
                row["complete_family_size"] = family_size
                row["k_min"] = required_k
                row["tie_rule"] = TIE_RULE
                pairwise_rows.append(row)

    summary_rows: list[dict[str, Any]] = []
    for dataset_id in datasets:
        for allocation_id in allocations:
            units = units_by_dataset_allocation[(dataset_id, allocation_id)]
            means = {
                method_id: math.fsum(
                    values[(dataset_id, allocation_id, unit_id, method_id)] for unit_id in units
                )
                / len(units)
                for method_id in methods
            }
            for method_id in methods:
                method_values = [
                    values[(dataset_id, allocation_id, unit_id, method_id)] for unit_id in units
                ]
                mean = means[method_id]
                if orientation == "higher_is_better":
                    point_rank = 1 + sum(other > mean for other in means.values())
                else:
                    point_rank = 1 + sum(other < mean for other in means.values())
                summary_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "allocation_id": allocation_id,
                        "method_id": method_id,
                        "declared_unit_count": len(units),
                        "equal_unit_mean_utility": _format_float(mean),
                        "equal_unit_mean_utility_hex": mean.hex(),
                        "median_unit_utility": _format_float(statistics.median(method_values)),
                        "minimum_unit_utility": _format_float(min(method_values)),
                        "maximum_unit_utility": _format_float(max(method_values)),
                        "point_rank_by_equal_unit_mean": point_rank,
                        "interpretation": "DESCRIPTIVE_POINT_SUMMARY_NOT_WINNER_AUTHORITY",
                    }
                )

    pairwise_fields = [
        "comparison_id",
        "dataset_id",
        "allocation_id",
        "model_a_id",
        "model_b_id",
        "declared_unit_count",
        "effective_non_tie_units",
        "wins",
        "losses",
        "literal_zero_ties",
        "equal_unit_mean_oriented_difference",
        "equal_unit_mean_oriented_difference_hex",
        "exact_one_sided_p_numerator",
        "exact_one_sided_p_denominator",
        "exact_one_sided_p",
        "holm_adjusted_p_numerator",
        "holm_adjusted_p_denominator",
        "holm_adjusted_p",
        "reject_at_alpha",
        "decision",
        "withheld_reason",
        "alpha",
        "complete_family_size",
        "k_min",
        "tie_rule",
        "evidence_scope",
    ]
    unit_fields = ["dataset_id", "allocation_id", "unit_id", "method_id", "utility", "utility_hex"]
    summary_fields = [
        "dataset_id",
        "allocation_id",
        "method_id",
        "declared_unit_count",
        "equal_unit_mean_utility",
        "equal_unit_mean_utility_hex",
        "median_unit_utility",
        "minimum_unit_utility",
        "maximum_unit_utility",
        "point_rank_by_equal_unit_mean",
        "interpretation",
    ]
    withheld_rows = [
        row
        for row in pairwise_rows
        if row["decision"] == DECISION_WITHHELD
    ]
    directional_rejections = [
        row for row in pairwise_rows if row["decision"] == DECISION_REJECTION
    ]
    eligible_no_direction = [
        row for row in pairwise_rows if row["decision"] == DECISION_NO_DIRECTION
    ]
    below_kmin_withheld = [
        row for row in withheld_rows if row["withheld_reason"] == WITHHELD_BELOW_KMIN
    ]
    gate_failures: list[str] = []
    if not independence_established:
        gate_failures.append(WITHHELD_INDEPENDENCE)
    if not training_evaluation_separation_established:
        gate_failures.append(WITHHELD_TRAIN_EVAL)
    receipt = {
        "audit_id": config["audit_id"],
        "bundle_kind": "utility_evaluation",
        "below_kmin_withheld_count": len(below_kmin_withheld),
        "complete_family_size_per_dataset_allocation": family_size,
        "config_sha256": sha256_file(config_path),
        "configuration_fixed_before_utility_access": config["analysis_timing"][
            "configuration_fixed_before_utility_access"
        ],
        "decision_rule_id": config["inference"]["decision_rule_id"],
        "declared_unit_independence_established": independence_established,
        "descriptive_summary_count": len(summary_rows),
        "directional_rejection_count": len(directional_rejections),
        "eligible_no_direction_count": len(eligible_no_direction),
        "evidence_timing_label": config["analysis_timing"]["evidence_timing_label"],
        "gate_failures": gate_failures,
        "independence_assertion_machine_verified": False,
        "k_min": required_k,
        "optimization_certification_uncertain_count": 0,
        "pairwise_comparison_count": len(pairwise_rows),
        "record_type": "BENCHMARK_DESIGN_AUDIT_RECEIPT_V1",
        "schema_version": "1.0.0",
        "status": (
            "WITHHELD_REQUIRED_GATE_FAILURE"
            if gate_failures
            else "PASS_CONDITIONAL_EXACT_SIGN_HOLM_EVALUATION"
        ),
        "tie_rule": TIE_RULE,
        "training_evaluation_separation_established": (
            training_evaluation_separation_established
        ),
        "unit_audit_manifest_sha256": hashlib.sha256(unit_audit_manifest_bytes).hexdigest(),
        "unit_audit_receipt_sha256": unit_audit_receipt_sha256,
        "unit_metadata_sha256": hashlib.sha256(unit_metadata_bytes).hexdigest(),
        "unit_utility_count": len(normalized_rows),
        "upstream_cell_and_model_provenance_machine_verified": False,
        "utilities_sha256": sha256_file(utilities_path),
        "withheld_comparison_count": len(withheld_rows),
    }
    _write_closed_bundle(
        config_path,
        output_dir,
        {
            "DESCRIPTIVE_SUMMARIES_V1.tsv": _rows_as_tsv(summary_fields, summary_rows),
            "EVALUATION_RECEIPT_V1.json": canonical_json(receipt),
            "INPUT_UNIT_METADATA.tsv": unit_metadata_bytes,
            "INPUT_UTILITIES.tsv": utilities_path.read_bytes(),
            "PAIRWISE_COMPLETE_FAMILY_V1.tsv": _rows_as_tsv(pairwise_fields, pairwise_rows),
            "REPORTING_CHECKLIST_V1.md": evaluation_checklist(),
            "UNIT_AUDIT_RECEIPT_SNAPSHOT.json": unit_audit_receipt_bytes,
            "UNIT_AUDIT_MANIFEST_SNAPSHOT.sha256": unit_audit_manifest_bytes,
            "UNIT_BY_METHOD_UTILITY_V1.tsv": _rows_as_tsv(unit_fields, normalized_rows),
            "WITHHELD_COMPARISONS_V1.tsv": _rows_as_tsv(pairwise_fields, withheld_rows),
        },
    )
