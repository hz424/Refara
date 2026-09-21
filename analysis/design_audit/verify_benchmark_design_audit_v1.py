#!/usr/bin/env python3
"""Verify benchmark-design-audit V1 bundles with an independent implementation.

The verifier reads the saved inputs and recomputes every machine-readable
derived table and receipt without importing the code that created the bundle.
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any


MANIFEST = "BENCHMARK_DESIGN_AUDIT_MANIFEST_V1.sha256"
ROLES = ("observed_effect", "prediction_reference", "model_conditioning")
TIE_RULE = "LITERAL_IEEE754_BINARY64_ZERO"
GEOMETRY = "CANDIDATE_DEPENDENCE_GEOMETRY_NOT_INDEPENDENCE_EVIDENCE"
MAPPING_SCOPE = "DECLARED_BLOCK_ABSTRACTION_NO_CELL_MEMBERSHIP_VERIFICATION"
AGGREGATION_ORDER = "SCORE_WITHIN_ROTATION_THEN_EQUAL_WEIGHT_ROTATIONS"
DECISION_RULE = "EXACT_ONE_SIDED_SIGN_HOLM_COMPLETE_ORDERED_V1"
MISSING_UNIT_RULE = "FAIL_CLOSED_COMPLETE_UNIT_BY_METHOD_SUPPORT"
UTILITY_PROVENANCE_SCOPE = "PREAGGREGATED_UNIT_UTILITIES_NO_UPSTREAM_CELL_OR_MODEL_PROVENANCE_VERIFICATION"
INDEPENDENCE_ESTABLISHED = "ESTABLISHED_BY_EXPERT_ASSERTION"
TRAIN_EVAL_ESTABLISHED = "ESTABLISHED_BY_EXPERT_ASSERTION"
TRAIN_EVAL_NOT_APPLICABLE = "NOT_APPLICABLE"
DECISION_REJECTION = "DIRECTION"
DECISION_NO_DIRECTION = "NO_DIRECTION"
DECISION_WITHHELD = "WITHHELD"
WITHHELD_INDEPENDENCE = "UNIT_INDEPENDENCE_NOT_ESTABLISHED"
WITHHELD_TRAIN_EVAL = "TRAINING_EVALUATION_SEPARATION_NOT_ESTABLISHED"
WITHHELD_BELOW_KMIN = "ARITHMETIC_SUPPORT_BELOW_KMIN"
AUDIT_TREE = {
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
PLAN_TREE = {
    MANIFEST,
    "CONFIG_SNAPSHOT.json",
    "KMIN_RESOLUTION_V1.tsv",
    "PAIRWISE_FAMILY_PLAN_V1.tsv",
    "PLAN_RECEIPT_V1.json",
    "REFERENCE_ROLE_ALLOCATIONS_V1.tsv",
    "REPORTING_CHECKLIST_V1.md",
    "ROLE_OVERLAP_MATRIX_V1.tsv",
}
EVALUATION_TREE = {
    MANIFEST,
    "CONFIG_SNAPSHOT.json",
    "DESCRIPTIVE_SUMMARIES_V1.tsv",
    "EVALUATION_RECEIPT_V1.json",
    "INPUT_UNIT_METADATA.tsv",
    "INPUT_UTILITIES.tsv",
    "PAIRWISE_COMPLETE_FAMILY_V1.tsv",
    "REPORTING_CHECKLIST_V1.md",
    "UNIT_AUDIT_RECEIPT_SNAPSHOT.json",
    "UNIT_AUDIT_MANIFEST_SNAPSHOT.sha256",
    "UNIT_BY_METHOD_UTILITY_V1.tsv",
    "WITHHELD_COMPARISONS_V1.tsv",
}


class VerifyError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerifyError(message)


def sha(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"not a regular file: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def tsv_text(fields: list[str], rows: list[dict[str, Any]]) -> str:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=fields,
        delimiter="\t",
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        require(key not in value, f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def load_json(path: Path) -> Any:
    require(path.is_file() and not path.is_symlink(), f"JSON file invalid: {path.name}")
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=strict_json_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VerifyError(f"invalid JSON: {path.name}") from error


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    require(path.is_file() and not path.is_symlink(), f"TSV file invalid: {path.name}")
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            require(reader.fieldnames is not None, f"missing TSV header: {path.name}")
            header = list(reader.fieldnames)
            require(len(header) == len(set(header)) and None not in header, "duplicate TSV header")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise VerifyError(f"invalid TSV: {path.name}") from error
    require(all(None not in row for row in rows), f"TSV width differs: {path.name}")
    return header, rows


def string_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{key: str(value) for key, value in row.items()} for row in rows]


def assert_tsv(bundle: Path, name: str, fields: list[str], expected: list[dict[str, Any]]) -> None:
    header, observed = read_tsv(bundle / name)
    require(header == fields, f"header differs: {name}")
    require(observed == string_rows(expected), f"semantic rows differ: {name}")


def nonempty(value: Any, label: str) -> str:
    require(isinstance(value, str) and value and value.strip() == value, f"{label} differs")
    return value


def unique_strings(value: Any, label: str, minimum: int = 1) -> list[str]:
    require(isinstance(value, list) and len(value) >= minimum, f"{label} differs")
    result = [nonempty(item, label) for item in value]
    require(len(result) == len(set(result)), f"{label} duplicates")
    return result


def overlap_pattern(rotation: dict[str, Any]) -> str:
    blocks = [rotation[role] for role in ROLES]
    if len(set(blocks)) == 1:
        return "ALL_THREE_ROLES_SHARE_ONE_BLOCK"
    if len(set(blocks)) == 3:
        return "ALL_THREE_ROLES_USE_DISTINCT_BLOCKS"
    return "PARTIAL_ROLE_OVERLAP"


def config_contract(config: Any) -> dict[str, Any]:
    require(isinstance(config, dict), "config must be an object")
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
        "config fields differ",
    )
    require(config["schema_version"] == "1.0.0", "config schema differs")
    nonempty(config["audit_id"], "audit id")
    datasets = unique_strings(config["dataset_ids"], "datasets")
    configurations = config["configurations"]
    require(isinstance(configurations, list) and len(configurations) >= 2, "configurations differ")
    configuration_ids: list[str] = []
    for item in configurations:
        require(
            isinstance(item, dict)
            and set(item)
            == {
                "configuration_id",
                "implementation_identity",
                "terminal_state_identity",
                "output_semantics",
                "active_reference_roles",
                "training_evaluation_separation",
            },
            "configuration identity differs",
        )
        configuration_ids.append(nonempty(item["configuration_id"], "configuration id"))
        nonempty(item["implementation_identity"], "implementation identity")
        nonempty(item["terminal_state_identity"], "terminal state identity")
        require(item["output_semantics"] in {"DIRECT_EFFECT", "ABSOLUTE_STATE"}, "output semantics differ")
        active_roles = unique_strings(item["active_reference_roles"], "active reference roles")
        expected_active_roles = ["observed_effect"] if item["output_semantics"] == "DIRECT_EFFECT" else list(ROLES)
        require(active_roles == expected_active_roles, "active roles/output semantics differ")
        separation = item["training_evaluation_separation"]
        require(
            isinstance(separation, dict)
            and set(separation) == {"status", "basis", "assertion_authority", "machine_verified"},
            "training/evaluation assertion differs",
        )
        require(
            separation["status"]
            in {TRAIN_EVAL_ESTABLISHED, "NOT_ESTABLISHED", TRAIN_EVAL_NOT_APPLICABLE},
            "training/evaluation status differs",
        )
        nonempty(separation["basis"], "training/evaluation basis")
        nonempty(separation["assertion_authority"], "training/evaluation authority")
        require(separation["machine_verified"] is False, "training/evaluation machine status differs")
    require(len(configuration_ids) == len(set(configuration_ids)), "configuration ids duplicate")
    declaration = config["reference_declaration"]
    require(
        isinstance(declaration, dict)
        and set(declaration) == {"mapping_scope", "block_label_semantics"},
        "reference declaration differs",
    )
    require(declaration["mapping_scope"] == MAPPING_SCOPE, "mapping scope differs")
    nonempty(declaration["block_label_semantics"], "block label semantics")
    require(tuple(config["reference_roles"]) == ROLES, "reference roles differ")
    allocation_ids: list[str] = []
    require(isinstance(config["allocations"], list) and config["allocations"], "allocations differ")
    for allocation in config["allocations"]:
        require(
            isinstance(allocation, dict)
            and set(allocation)
            == {
                "allocation_id",
                "reference_depth_label",
                "reference_depth_unit",
                "aggregation_order",
                "rotations",
            },
            "allocation fields differ",
        )
        allocation_ids.append(nonempty(allocation["allocation_id"], "allocation id"))
        nonempty(allocation["reference_depth_label"], "depth label")
        nonempty(allocation["reference_depth_unit"], "depth unit")
        require(allocation["aggregation_order"] == AGGREGATION_ORDER, "aggregation order differs")
        require(isinstance(allocation["rotations"], list) and allocation["rotations"], "rotations differ")
        rotation_ids: list[str] = []
        for rotation in allocation["rotations"]:
            require(
                isinstance(rotation, dict)
                and set(rotation)
                == {"rotation_id", *ROLES, "role_depth_labels", "expected_overlap_pattern"},
                "rotation fields differ",
            )
            rotation_ids.append(nonempty(rotation["rotation_id"], "rotation id"))
            for role in ROLES:
                nonempty(rotation[role], "role block")
            depths = rotation["role_depth_labels"]
            require(isinstance(depths, dict) and set(depths) == set(ROLES), "role depths differ")
            for role in ROLES:
                nonempty(depths[role], "role depth")
            require(rotation["expected_overlap_pattern"] == overlap_pattern(rotation), "overlap declaration differs")
        require(len(rotation_ids) == len(set(rotation_ids)), "rotation ids duplicate")
    require(len(allocation_ids) == len(set(allocation_ids)), "allocation ids duplicate")
    unit = config["unit_contract"]
    require(
        isinstance(unit, dict)
        and set(unit)
        == {
            "dataset_column",
            "object_id_column",
            "unit_id_column",
            "declared_unit_columns",
            "candidate_acquisition_group_columns",
            "independence_justification",
        },
        "unit contract differs",
    )
    scalar_columns = [nonempty(unit[name], name) for name in ("dataset_column", "object_id_column", "unit_id_column")]
    declared_columns = unique_strings(unit["declared_unit_columns"], "declared unit columns")
    candidate_columns = unique_strings(unit["candidate_acquisition_group_columns"], "candidate columns", minimum=0)
    require(len(set([*scalar_columns, *declared_columns, *candidate_columns])) == len([*scalar_columns, *declared_columns, *candidate_columns]), "unit columns overlap")
    independence = unit["independence_justification"]
    require(
        isinstance(independence, dict)
        and set(independence) == {"status", "basis", "assertion_authority", "machine_verified"},
        "independence assertion differs",
    )
    require(independence["status"] in {INDEPENDENCE_ESTABLISHED, "NOT_ESTABLISHED"}, "independence status differs")
    nonempty(independence["basis"], "independence basis")
    nonempty(independence["assertion_authority"], "independence authority")
    require(independence["machine_verified"] is False, "independence machine status differs")
    utility = config["utility_input_contract"]
    require(
        isinstance(utility, dict)
        and set(utility)
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
        "utility contract differs",
    )
    input_columns = unique_strings(utility["input_columns"], "utility input columns", minimum=5)
    mapped = [nonempty(utility[name], name) for name in ("dataset_id_column", "allocation_id_column", "unit_id_column", "method_id_column", "utility_column")]
    require(len(mapped) == len(set(mapped)) and set(mapped).issubset(input_columns), "utility mapping differs")
    constants = utility["required_constant_values"]
    require(isinstance(constants, dict) and set(constants).issubset(input_columns), "utility constants differ")
    require(not set(constants).intersection(mapped), "utility constant overlaps mapping")
    for key, value in constants.items():
        nonempty(key, "utility constant column")
        nonempty(value, "utility constant value")
    expected_sha = utility["expected_input_sha256"]
    require(expected_sha is None or (isinstance(expected_sha, str) and len(expected_sha) == 64 and all(character in "0123456789abcdef" for character in expected_sha)), "utility input hash differs")
    require(utility["provenance_scope"] == UTILITY_PROVENANCE_SCOPE, "utility provenance differs")
    timing = config["analysis_timing"]
    require(
        isinstance(timing, dict)
        and set(timing)
        == {"evidence_timing_label", "configuration_fixed_before_utility_access", "outcome_access_status_at_configuration_fix"},
        "analysis timing differs",
    )
    nonempty(timing["evidence_timing_label"], "timing label")
    require(isinstance(timing["configuration_fixed_before_utility_access"], bool), "timing boolean differs")
    outcome_status = timing["outcome_access_status_at_configuration_fix"]
    require(outcome_status in {"NOT_ACCESSED", "ACCESSED", "SYNTHETIC_NO_EMPIRICAL_OUTCOME"}, "outcome status differs")
    if outcome_status == "NOT_ACCESSED":
        require(timing["configuration_fixed_before_utility_access"] is True, "timing fields conflict")
    if outcome_status == "ACCESSED":
        require(timing["configuration_fixed_before_utility_access"] is False, "timing fields conflict")
    if outcome_status == "SYNTHETIC_NO_EMPIRICAL_OUTCOME":
        require(timing["configuration_fixed_before_utility_access"] is True, "timing fields conflict")
    family = config["comparison_family"]
    require(isinstance(family, dict) and set(family) == {"allocation_order", "method_order", "directionality", "utility_orientation"}, "comparison family differs")
    methods = unique_strings(family["method_order"], "methods", minimum=2)
    allocation_order = unique_strings(family["allocation_order"], "allocation order")
    require(methods == configuration_ids, "method/configuration order differs")
    require(allocation_order == allocation_ids, "allocation order differs")
    require(family["directionality"] == "ordered", "directionality differs")
    require(family["utility_orientation"] in {"higher_is_better", "lower_is_better"}, "orientation differs")
    inference = config["inference"]
    require(
        isinstance(inference, dict)
        and set(inference) == {"decision_rule_id", "alpha", "fair_binomial_probability", "tie_rule", "missing_unit_rule", "holm_scope"},
        "inference fields differ",
    )
    require(inference["decision_rule_id"] == DECISION_RULE, "decision rule differs")
    require(isinstance(inference["alpha"], (int, float)) and not isinstance(inference["alpha"], bool) and 0 < inference["alpha"] < 1, "alpha differs")
    require(inference["tie_rule"] == TIE_RULE, "tie rule differs")
    require(inference["fair_binomial_probability"] == 0.5, "binomial null differs")
    require(inference["missing_unit_rule"] == MISSING_UNIT_RULE, "missing-unit rule differs")
    require(inference["holm_scope"] == "WITHIN_DATASET_AND_ALLOCATION_COMPLETE_ORDERED_METHOD_FAMILY", "Holm scope differs")
    require(bool(datasets), "empty design")
    return config


def family_size(config: dict[str, Any]) -> int:
    count = len(config["comparison_family"]["method_order"])
    return count * (count - 1)


def calculate_kmin(config: dict[str, Any]) -> int:
    threshold = Decimal(str(config["inference"]["alpha"])) / Decimal(family_size(config))
    value = Decimal(1)
    count = 0
    while value > threshold:
        value /= Decimal(2)
        count += 1
    return count


def fraction_text(value: Fraction) -> str:
    rendered = format(Decimal(value.numerator) / Decimal(value.denominator), "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def float_text(value: float) -> str:
    require(math.isfinite(value), "non-finite derived value")
    return format(value, ".17g")


def exact_tail(wins: int, total: int) -> Fraction:
    if total == 0:
        return Fraction(1, 1)
    return Fraction(sum(math.comb(total, value) for value in range(wins, total + 1)), 2**total)


def audit_checklist() -> str:
    return """# Acquisition-unit audit\n\n## Verified\n\n- Experimental-unit identifiers are declared.\n- The number of observations within each declared unit is reported.\n- Candidate acquisition-group structure is reported for each supplied grouping rule.\n- The analyst's independence statement and its basis are recorded.\n\n## Scope\n\n- This metadata audit does not establish that the declared units are independent.\n- Candidate links require study-specific protocol knowledge before they can define inferential units.\n"""


def plan_checklist() -> str:
    return """# Comparison plan\n\n## Verified\n\n- The observed-effect, prediction-centering and model-conditioning roles are explicit.\n- Each declared allocation and rotation maps roles to blocks.\n- Per-role depth labels and the rotation-averaging rule are recorded.\n- Declared overlap agrees with the overlap implied by the labels, with a 3 × 3 matrix for each rotation.\n- Every ordered method comparison under the declared decision rule is included.\n- `K_min` uses the smallest complete-family Holm threshold and an all-win fair-binomial tail.\n\n## Scope\n\n- Block and depth labels are declarations; this software does not inspect cell membership.\n- `K_min` describes arithmetic attainability, not evidence for an effect or unit independence.\n"""


def evaluation_checklist() -> str:
    return """# Utility evaluation\n\n## Verified\n\n- Input consists only of declared, pre-aggregated experimental-unit utilities.\n- The evaluation uses the supplied unit-audit bundle and exactly the same unit support.\n- Every dataset-allocation family has complete unit-by-method support.\n- All ordered pairs are generated from the declared method order.\n- Ties are literal IEEE-754 binary64 zero differences; no tolerance is used.\n- One-sided fair-binomial tails are exact, and Holm correction covers the complete ordered family.\n- `DIRECTION`, `NO_DIRECTION` and `WITHHELD` are reported separately, with below-`K_min` and failed-declaration reasons retained.\n- Directional rejection is withheld unless independence is declared established.\n\n## Scope\n\n- Independence and training/evaluation separation are analyst-supplied statements, not machine-established facts.\n- Pre-aggregated utilities cannot verify upstream cells, memberships, predictions or model states.\n- This deterministic evaluator does not perform an optimization search.\n- Ranks and means are descriptive; they do not identify a uniquely supported winner.\n"""


def verify_manifest(bundle: Path) -> set[str]:
    require(bundle.is_dir() and not bundle.is_symlink(), "bundle must be a regular directory")
    entries = list(bundle.iterdir())
    require(all(path.is_file() and not path.is_symlink() for path in entries), "bundle has non-regular entries")
    names = {path.name for path in entries}
    require(MANIFEST in names, "manifest missing")
    lines = (bundle / MANIFEST).read_text(encoding="ascii").splitlines()
    declared: dict[str, str] = {}
    for line in lines:
        parts = line.split("  ", 1)
        require(len(parts) == 2, "manifest syntax differs")
        digest, name = parts
        require(len(digest) == 64 and all(ch in "0123456789abcdef" for ch in digest), "manifest digest invalid")
        require(name not in declared and "/" not in name and name != MANIFEST, "manifest name invalid")
        declared[name] = digest
    require(list(declared) == sorted(declared), "manifest order differs")
    require(set(declared) == names - {MANIFEST}, "manifest is not closed")
    for name, digest in declared.items():
        require(sha(bundle / name) == digest, f"manifest digest differs: {name}")
    return names


def verify_audit(config_path: Path, config: dict[str, Any], bundle: Path) -> None:
    unit = config["unit_contract"]
    dataset_column = unit["dataset_column"]
    object_column = unit["object_id_column"]
    unit_column = unit["unit_id_column"]
    declared_columns = unit["declared_unit_columns"]
    candidate_columns = unit["candidate_acquisition_group_columns"]
    independence = unit["independence_justification"]
    independence_established = independence["status"] == INDEPENDENCE_ESTABLISHED
    header, source = read_tsv(bundle / "INPUT_METADATA.tsv")
    required = [dataset_column, object_column, unit_column, *declared_columns, *candidate_columns]
    require(set(required).issubset(header), "metadata columns differ")
    require(all(all(row[column] for column in required) for row in source), "empty metadata value")
    require(len({(row[dataset_column], row[object_column]) for row in source}) == len(source), "duplicate object")
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in sorted(source, key=lambda item: (item[dataset_column], item[unit_column], item[object_column])):
        require(row[dataset_column] in config["dataset_ids"], "unregistered metadata dataset")
        groups.setdefault((row[dataset_column], row[unit_column]), []).append(row)
    require({key[0] for key in groups} == set(config["dataset_ids"]), "metadata dataset coverage differs")
    hierarchy: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    representatives: dict[tuple[str, str], dict[str, str]] = {}
    for (dataset_id, unit_id), records in sorted(groups.items()):
        for column in [*declared_columns, *candidate_columns]:
            require(len({row[column] for row in records}) == 1, "within-unit metadata differs")
        representative = records[0]
        representatives[(dataset_id, unit_id)] = representative
        for row in records:
            hierarchy.append({
                "dataset_id": dataset_id,
                "object_id": row[object_column],
                "unit_id": unit_id,
                **{column: row[column] for column in declared_columns},
                "mapping_status": "OBJECT_MAPPED_TO_DECLARED_UNIT",
            })
        units.append({
            "dataset_id": dataset_id,
            "unit_id": unit_id,
            **{column: representative[column] for column in declared_columns},
            "object_count": len(records),
            "repeated_objects_beyond_first": len(records) - 1,
            "independence_status": independence["status"],
        })
    geometry: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for dataset_id in config["dataset_ids"]:
        records = {unit_id: row for (data, unit_id), row in representatives.items() if data == dataset_id}
        for column in candidate_columns:
            by_value: dict[str, list[str]] = {}
            for unit_id, row in records.items():
                by_value.setdefault(row[column], []).append(unit_id)
            for value, member_ids in sorted(by_value.items()):
                members = sorted(member_ids)
                geometry.append({
                    "dataset_id": dataset_id,
                    "candidate_group_column": column,
                    "candidate_group_value": value,
                    "declared_unit_count": len(members),
                    "object_count": sum(len(groups[(dataset_id, member)]) for member in members),
                    "member_unit_ids": ";".join(members),
                    "interpretation": GEOMETRY,
                })
                for left, right in itertools.combinations(members, 2):
                    edges.append({
                        "dataset_id": dataset_id,
                        "candidate_group_column": column,
                        "candidate_group_value": value,
                        "unit_id_a": left,
                        "unit_id_b": right,
                        "interpretation": GEOMETRY,
                    })
    assert_tsv(bundle, "UNIT_HIERARCHY_V1.tsv", ["dataset_id", "object_id", "unit_id", *declared_columns, "mapping_status"], hierarchy)
    assert_tsv(bundle, "UNIT_AUDIT_V1.tsv", ["dataset_id", "unit_id", *declared_columns, "object_count", "repeated_objects_beyond_first", "independence_status"], units)
    assert_tsv(bundle, "UNIT_GROUP_GEOMETRY_V1.tsv", ["dataset_id", "candidate_group_column", "candidate_group_value", "declared_unit_count", "object_count", "member_unit_ids", "interpretation"], geometry)
    assert_tsv(bundle, "ACQUISITION_EDGE_CANDIDATES_V1.tsv", ["dataset_id", "candidate_group_column", "candidate_group_value", "unit_id_a", "unit_id_b", "interpretation"], edges)
    require((bundle / "REPORTING_CHECKLIST_V1.md").read_text(encoding="utf-8") == audit_checklist(), "audit checklist differs")
    expected_receipt = {
        "audit_id": config["audit_id"],
        "bundle_kind": "unit_audit",
        "candidate_edge_record_count": len(edges),
        "candidate_group_record_count": len(geometry),
        "config_sha256": sha(config_path),
        "declared_unit_count": len(units),
        "declared_unit_independence_established": independence_established,
        "independence_assertion_authority": independence["assertion_authority"],
        "independence_assertion_basis": independence["basis"],
        "independence_assertion_machine_verified": False,
        "metadata_sha256": sha(bundle / "INPUT_METADATA.tsv"),
        "object_count": len(hierarchy),
        "record_type": "BENCHMARK_DESIGN_AUDIT_RECEIPT_V1",
        "schema_version": "1.0.0",
        "status": (
            "PASS_GEOMETRY_WITH_DECLARED_INDEPENDENCE"
            if independence_established
            else "WITHHELD_UNIT_INDEPENDENCE_NOT_ESTABLISHED"
        ),
    }
    require(load_json(bundle / "AUDIT_RECEIPT_V1.json") == expected_receipt, "audit receipt differs")
    require((bundle / "AUDIT_RECEIPT_V1.json").read_text() == canonical_json(expected_receipt), "audit receipt encoding differs")


def verify_plan(config_path: Path, config: dict[str, Any], bundle: Path) -> None:
    allocation_by_id = {item["allocation_id"]: item for item in config["allocations"]}
    allocations: list[dict[str, Any]] = []
    overlap: list[dict[str, Any]] = []
    for allocation_id in config["comparison_family"]["allocation_order"]:
        allocation = allocation_by_id[allocation_id]
        for rotation in allocation["rotations"]:
            pattern = overlap_pattern(rotation)
            allocations.append({
                "allocation_id": allocation_id,
                "reference_depth_label": allocation["reference_depth_label"],
                "reference_depth_unit": allocation["reference_depth_unit"],
                "aggregation_order": allocation["aggregation_order"],
                "rotation_id": rotation["rotation_id"],
                **{role: rotation[role] for role in ROLES},
                **{f"{role}_depth_label": rotation["role_depth_labels"][role] for role in ROLES},
                "overlap_pattern": pattern,
            })
            for row_role in ROLES:
                for column_role in ROLES:
                    overlap.append({
                        "allocation_id": allocation_id,
                        "rotation_id": rotation["rotation_id"],
                        "row_role": row_role,
                        "column_role": column_role,
                        "row_block": rotation[row_role],
                        "column_block": rotation[column_role],
                        "shares_control_block": int(rotation[row_role] == rotation[column_role]),
                    })
    methods = config["comparison_family"]["method_order"]
    family: list[dict[str, Any]] = []
    ordinal = 0
    for dataset_id in config["dataset_ids"]:
        for allocation_id in config["comparison_family"]["allocation_order"]:
            for model_a in methods:
                for model_b in methods:
                    if model_a == model_b:
                        continue
                    ordinal += 1
                    family.append({
                        "comparison_ordinal": ordinal,
                        "comparison_id": f"{dataset_id}::{allocation_id}::{model_a}>{model_b}",
                        "dataset_id": dataset_id,
                        "allocation_id": allocation_id,
                        "model_a_id": model_a,
                        "model_b_id": model_b,
                        "directionality": "ORDERED_A_BETTER_THAN_B",
                        "utility_orientation": config["comparison_family"]["utility_orientation"],
                        "decision_rule_id": config["inference"]["decision_rule_id"],
                    })
    size = family_size(config)
    minimum = calculate_kmin(config)
    alpha = Decimal(str(config["inference"]["alpha"]))
    resolution = [{
        "dataset_id": dataset_id,
        "allocation_id": allocation_id,
        "complete_family_size": size,
        "alpha": format(alpha, "f"),
        "smallest_holm_threshold": format(alpha / Decimal(size), "f"),
        "k_min": minimum,
        "all_win_fair_binomial_tail_at_k_min": format(Decimal(1) / Decimal(2**minimum), "f"),
        "tie_rule": TIE_RULE,
        "interpretation": "ATTAINABILITY_ONLY_NOT_AN_EFFECT_OR_INDEPENDENCE_CLAIM",
    } for dataset_id in config["dataset_ids"] for allocation_id in config["comparison_family"]["allocation_order"]]
    assert_tsv(bundle, "REFERENCE_ROLE_ALLOCATIONS_V1.tsv", ["allocation_id", "reference_depth_label", "reference_depth_unit", "aggregation_order", "rotation_id", *ROLES, *[f"{role}_depth_label" for role in ROLES], "overlap_pattern"], allocations)
    assert_tsv(bundle, "ROLE_OVERLAP_MATRIX_V1.tsv", ["allocation_id", "rotation_id", "row_role", "column_role", "row_block", "column_block", "shares_control_block"], overlap)
    assert_tsv(bundle, "PAIRWISE_FAMILY_PLAN_V1.tsv", ["comparison_ordinal", "comparison_id", "dataset_id", "allocation_id", "model_a_id", "model_b_id", "directionality", "utility_orientation", "decision_rule_id"], family)
    assert_tsv(bundle, "KMIN_RESOLUTION_V1.tsv", ["dataset_id", "allocation_id", "complete_family_size", "alpha", "smallest_holm_threshold", "k_min", "all_win_fair_binomial_tail_at_k_min", "tie_rule", "interpretation"], resolution)
    require((bundle / "REPORTING_CHECKLIST_V1.md").read_text(encoding="utf-8") == plan_checklist(), "plan checklist differs")
    expected_receipt = {
        "allocation_count": len(config["allocations"]),
        "allocation_rotation_count": len(allocations),
        "audit_id": config["audit_id"],
        "bundle_kind": "comparison_plan",
        "complete_ordered_family_size_per_dataset_allocation": size,
        "config_sha256": sha(config_path),
        "configuration_identity_count": len(config["configurations"]),
        "decision_rule_id": config["inference"]["decision_rule_id"],
        "evidence_timing_label": config["analysis_timing"]["evidence_timing_label"],
        "k_min": minimum,
        "planned_comparison_count": len(family),
        "record_type": "BENCHMARK_DESIGN_AUDIT_RECEIPT_V1",
        "schema_version": "1.0.0",
        "status": "PASS_COMPLETE_ORDERED_PLAN",
    }
    require(load_json(bundle / "PLAN_RECEIPT_V1.json") == expected_receipt, "plan receipt differs")
    require((bundle / "PLAN_RECEIPT_V1.json").read_text() == canonical_json(expected_receipt), "plan receipt encoding differs")


def verify_evaluation(config_path: Path, config: dict[str, Any], bundle: Path) -> None:
    unit = config["unit_contract"]
    dataset_metadata_column = unit["dataset_column"]
    object_column = unit["object_id_column"]
    unit_metadata_column = unit["unit_id_column"]
    declared_columns = unit["declared_unit_columns"]
    candidate_columns = unit["candidate_acquisition_group_columns"]
    metadata_header, metadata = read_tsv(bundle / "INPUT_UNIT_METADATA.tsv")
    required_metadata = [dataset_metadata_column, object_column, unit_metadata_column, *declared_columns, *candidate_columns]
    require(set(required_metadata).issubset(metadata_header), "unit metadata columns differ")
    require(all(all(row[column] for column in required_metadata) for row in metadata), "empty unit metadata value")
    require(len({(row[dataset_metadata_column], row[object_column]) for row in metadata}) == len(metadata), "duplicate unit metadata object")
    metadata_groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in sorted(metadata, key=lambda item: (item[dataset_metadata_column], item[unit_metadata_column], item[object_column])):
        require(row[dataset_metadata_column] in config["dataset_ids"], "unregistered unit metadata dataset")
        metadata_groups.setdefault((row[dataset_metadata_column], row[unit_metadata_column]), []).append(row)
    require({key[0] for key in metadata_groups} == set(config["dataset_ids"]), "unit metadata coverage differs")
    independence = unit["independence_justification"]
    hierarchy: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    representatives: dict[tuple[str, str], dict[str, str]] = {}
    for (dataset_id, unit_id), records in sorted(metadata_groups.items()):
        for column in [*declared_columns, *candidate_columns]:
            require(len({row[column] for row in records}) == 1, "within-unit metadata differs")
        representative = records[0]
        representatives[(dataset_id, unit_id)] = representative
        for row in records:
            hierarchy.append({
                "dataset_id": dataset_id,
                "object_id": row[object_column],
                "unit_id": unit_id,
                **{column: row[column] for column in declared_columns},
                "mapping_status": "OBJECT_MAPPED_TO_DECLARED_UNIT",
            })
        units.append({
            "dataset_id": dataset_id,
            "unit_id": unit_id,
            **{column: representative[column] for column in declared_columns},
            "object_count": len(records),
            "repeated_objects_beyond_first": len(records) - 1,
            "independence_status": independence["status"],
        })
    audited_support = {
        dataset_id: sorted(unit_id for data, unit_id in metadata_groups if data == dataset_id)
        for dataset_id in config["dataset_ids"]
    }
    geometry: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for dataset_id in config["dataset_ids"]:
        dataset_representatives = {
            unit_id: row
            for (data, unit_id), row in representatives.items()
            if data == dataset_id
        }
        for column in candidate_columns:
            by_value: dict[str, list[str]] = {}
            for unit_id, row in dataset_representatives.items():
                by_value.setdefault(row[column], []).append(unit_id)
            for value, member_ids in sorted(by_value.items()):
                members = sorted(member_ids)
                geometry.append({
                    "dataset_id": dataset_id,
                    "candidate_group_column": column,
                    "candidate_group_value": value,
                    "declared_unit_count": len(members),
                    "object_count": sum(
                        len(metadata_groups[(dataset_id, member)]) for member in members
                    ),
                    "member_unit_ids": ";".join(members),
                    "interpretation": GEOMETRY,
                })
                for left, right in itertools.combinations(members, 2):
                    edges.append({
                        "dataset_id": dataset_id,
                        "candidate_group_column": column,
                        "candidate_group_value": value,
                        "unit_id_a": left,
                        "unit_id_b": right,
                        "interpretation": GEOMETRY,
                    })
    independence_established = independence["status"] == INDEPENDENCE_ESTABLISHED
    expected_unit_receipt = {
        "audit_id": config["audit_id"],
        "bundle_kind": "unit_audit",
        "candidate_edge_record_count": len(edges),
        "candidate_group_record_count": len(geometry),
        "config_sha256": sha(config_path),
        "declared_unit_count": len(units),
        "declared_unit_independence_established": independence_established,
        "independence_assertion_authority": independence["assertion_authority"],
        "independence_assertion_basis": independence["basis"],
        "independence_assertion_machine_verified": False,
        "metadata_sha256": sha(bundle / "INPUT_UNIT_METADATA.tsv"),
        "object_count": len(hierarchy),
        "record_type": "BENCHMARK_DESIGN_AUDIT_RECEIPT_V1",
        "schema_version": "1.0.0",
        "status": "PASS_GEOMETRY_WITH_DECLARED_INDEPENDENCE" if independence_established else "WITHHELD_UNIT_INDEPENDENCE_NOT_ESTABLISHED",
    }
    unit_receipt_path = bundle / "UNIT_AUDIT_RECEIPT_SNAPSHOT.json"
    require(load_json(unit_receipt_path) == expected_unit_receipt, "unit-audit receipt snapshot differs")
    require(unit_receipt_path.read_text(encoding="utf-8") == canonical_json(expected_unit_receipt), "unit-audit receipt snapshot encoding differs")
    manifest_snapshot_path = bundle / "UNIT_AUDIT_MANIFEST_SNAPSHOT.sha256"
    snapshot_lines = manifest_snapshot_path.read_text(encoding="ascii").splitlines()
    snapshot: dict[str, str] = {}
    for line in snapshot_lines:
        parts = line.split("  ", 1)
        require(len(parts) == 2, "unit-audit manifest snapshot syntax differs")
        digest, name = parts
        require(len(digest) == 64 and all(character in "0123456789abcdef" for character in digest), "unit-audit manifest snapshot digest differs")
        require(name not in snapshot and "/" not in name and name != MANIFEST, "unit-audit manifest snapshot name differs")
        snapshot[name] = digest
    require(list(snapshot) == sorted(snapshot), "unit-audit manifest snapshot order differs")
    require(set(snapshot) == AUDIT_TREE - {MANIFEST}, "unit-audit manifest snapshot tree differs")
    hierarchy_fields = [
        "dataset_id",
        "object_id",
        "unit_id",
        *declared_columns,
        "mapping_status",
    ]
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
    expected_snapshot = {
        "ACQUISITION_EDGE_CANDIDATES_V1.tsv": sha_bytes(
            tsv_text(edge_fields, edges).encode("utf-8")
        ),
        "AUDIT_RECEIPT_V1.json": sha_bytes(
            canonical_json(expected_unit_receipt).encode("utf-8")
        ),
        "CONFIG_SNAPSHOT.json": sha(config_path),
        "INPUT_METADATA.tsv": sha(bundle / "INPUT_UNIT_METADATA.tsv"),
        "REPORTING_CHECKLIST_V1.md": sha_bytes(audit_checklist().encode("utf-8")),
        "UNIT_AUDIT_V1.tsv": sha_bytes(tsv_text(unit_fields, units).encode("utf-8")),
        "UNIT_GROUP_GEOMETRY_V1.tsv": sha_bytes(
            tsv_text(geometry_fields, geometry).encode("utf-8")
        ),
        "UNIT_HIERARCHY_V1.tsv": sha_bytes(
            tsv_text(hierarchy_fields, hierarchy).encode("utf-8")
        ),
    }
    require(
        snapshot == expected_snapshot,
        "unit-audit manifest snapshot does not bind complete verified semantics",
    )
    expected_snapshot_text = "".join(
        f"{expected_snapshot[name]}  {name}\n" for name in sorted(expected_snapshot)
    )
    require(
        manifest_snapshot_path.read_text(encoding="ascii") == expected_snapshot_text,
        "unit-audit manifest snapshot encoding differs",
    )

    utility_contract = config["utility_input_contract"]
    header, source = read_tsv(bundle / "INPUT_UTILITIES.tsv")
    require(header == utility_contract["input_columns"], "utility header differs")
    if utility_contract["expected_input_sha256"] is not None:
        require(sha(bundle / "INPUT_UTILITIES.tsv") == utility_contract["expected_input_sha256"], "utility source hash differs")
    for row in source:
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
    method_index = {value: index for index, value in enumerate(methods)}
    values: dict[tuple[str, str, str, str], float] = {}
    for row in source:
        key = (row[dataset_column], row[allocation_column], row[unit_column], row[method_column])
        require(key not in values, "duplicate utility")
        require(key[0] in datasets and key[1] in allocations and key[3] in methods and key[2], "unregistered utility support")
        try:
            value = float(row[value_column])
        except ValueError as error:
            raise VerifyError("nonnumeric utility") from error
        require(math.isfinite(value), "non-finite utility")
        values[key] = value
    support: dict[tuple[str, str], list[str]] = {}
    for dataset_id in datasets:
        baseline: list[str] | None = None
        for allocation_id in allocations:
            units = sorted({key[2] for key in values if key[:2] == (dataset_id, allocation_id)})
            require(units, "empty utility support")
            require(units == audited_support[dataset_id], "utility support differs from unit metadata")
            expected = {(dataset_id, allocation_id, unit_id, method_id) for unit_id in units for method_id in methods}
            observed = {key for key in values if key[:2] == (dataset_id, allocation_id)}
            require(expected == observed, "incomplete utility support")
            require(baseline is None or units == baseline, "allocation support differs")
            baseline = units
            support[(dataset_id, allocation_id)] = units
    require(len(values) == sum(len(support[key]) * len(methods) for key in support), "extraneous utility support")
    normalized: list[dict[str, Any]] = []
    for dataset_id in datasets:
        for allocation_id in allocations:
            for unit_id in support[(dataset_id, allocation_id)]:
                for method_id in methods:
                    value = values[(dataset_id, allocation_id, unit_id, method_id)]
                    normalized.append({"dataset_id": dataset_id, "allocation_id": allocation_id, "unit_id": unit_id, "method_id": method_id, "utility": float_text(value), "utility_hex": value.hex()})
    size = family_size(config)
    minimum = calculate_kmin(config)
    alpha = Fraction(str(config["inference"]["alpha"]))
    multiplier = 1.0 if config["comparison_family"]["utility_orientation"] == "higher_is_better" else -1.0
    training_evaluation_separation_established = all(
        item["training_evaluation_separation"]["status"]
        in {TRAIN_EVAL_ESTABLISHED, TRAIN_EVAL_NOT_APPLICABLE}
        for item in config["configurations"]
    )
    pairwise: list[dict[str, Any]] = []
    for dataset_id in datasets:
        for allocation_id in allocations:
            units = support[(dataset_id, allocation_id)]
            records: list[dict[str, Any]] = []
            for model_a in methods:
                for model_b in methods:
                    if model_a == model_b:
                        continue
                    differences = [multiplier * (values[(dataset_id, allocation_id, unit_id, model_a)] - values[(dataset_id, allocation_id, unit_id, model_b)]) for unit_id in units]
                    wins = sum(value > 0 for value in differences)
                    losses = sum(value < 0 for value in differences)
                    ties = sum(value == 0 for value in differences)
                    effective = wins + losses
                    raw = exact_tail(wins, effective)
                    mean = math.fsum(differences) / len(differences)
                    records.append({
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
                        "equal_unit_mean_oriented_difference": float_text(mean),
                        "equal_unit_mean_oriented_difference_hex": mean.hex(),
                        "exact_one_sided_p_numerator": raw.numerator,
                        "exact_one_sided_p_denominator": raw.denominator,
                        "exact_one_sided_p": fraction_text(raw),
                        "_raw": raw,
                        "_below_kmin": effective < minimum,
                        "evidence_scope": "DECLARATIVE_DOWNSTREAM_AUDIT_NO_UPSTREAM_CELL_OR_MODEL_PROVENANCE_VERIFICATION",
                    })
            ordered = sorted(records, key=lambda row: (row["_raw"], method_index[row["model_a_id"]], method_index[row["model_b_id"]]))
            running = Fraction(0)
            for rank, row in enumerate(ordered, 1):
                running = max(running, min(Fraction(1), row["_raw"] * (size - rank + 1)))
                row["_adjusted"] = running
            for row in records:
                adjusted = row.pop("_adjusted")
                row.pop("_raw")
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
                row.update({
                    "holm_adjusted_p_numerator": adjusted.numerator,
                    "holm_adjusted_p_denominator": adjusted.denominator,
                    "holm_adjusted_p": fraction_text(adjusted),
                    "reject_at_alpha": int(decision == DECISION_REJECTION) if decision in {DECISION_REJECTION, DECISION_NO_DIRECTION} else "NA",
                    "decision": decision,
                    "withheld_reason": withheld_reason,
                    "alpha": fraction_text(alpha),
                    "complete_family_size": size,
                    "k_min": minimum,
                    "tie_rule": TIE_RULE,
                })
                row.pop("_below_kmin")
                pairwise.append(row)
    summaries: list[dict[str, Any]] = []
    orientation = config["comparison_family"]["utility_orientation"]
    for dataset_id in datasets:
        for allocation_id in allocations:
            units = support[(dataset_id, allocation_id)]
            means = {method: math.fsum(values[(dataset_id, allocation_id, unit, method)] for unit in units) / len(units) for method in methods}
            for method in methods:
                method_values = [values[(dataset_id, allocation_id, unit, method)] for unit in units]
                mean = means[method]
                rank = 1 + sum(other > mean for other in means.values()) if orientation == "higher_is_better" else 1 + sum(other < mean for other in means.values())
                summaries.append({
                    "dataset_id": dataset_id,
                    "allocation_id": allocation_id,
                    "method_id": method,
                    "declared_unit_count": len(units),
                    "equal_unit_mean_utility": float_text(mean),
                    "equal_unit_mean_utility_hex": mean.hex(),
                    "median_unit_utility": float_text(statistics.median(method_values)),
                    "minimum_unit_utility": float_text(min(method_values)),
                    "maximum_unit_utility": float_text(max(method_values)),
                    "point_rank_by_equal_unit_mean": rank,
                    "interpretation": "DESCRIPTIVE_POINT_SUMMARY_NOT_WINNER_AUTHORITY",
                })
    pair_fields = ["comparison_id", "dataset_id", "allocation_id", "model_a_id", "model_b_id", "declared_unit_count", "effective_non_tie_units", "wins", "losses", "literal_zero_ties", "equal_unit_mean_oriented_difference", "equal_unit_mean_oriented_difference_hex", "exact_one_sided_p_numerator", "exact_one_sided_p_denominator", "exact_one_sided_p", "holm_adjusted_p_numerator", "holm_adjusted_p_denominator", "holm_adjusted_p", "reject_at_alpha", "decision", "withheld_reason", "alpha", "complete_family_size", "k_min", "tie_rule", "evidence_scope"]
    summary_fields = ["dataset_id", "allocation_id", "method_id", "declared_unit_count", "equal_unit_mean_utility", "equal_unit_mean_utility_hex", "median_unit_utility", "minimum_unit_utility", "maximum_unit_utility", "point_rank_by_equal_unit_mean", "interpretation"]
    withheld = [row for row in pairwise if row["decision"] == DECISION_WITHHELD]
    directional_rejections = [row for row in pairwise if row["decision"] == DECISION_REJECTION]
    eligible_no_direction = [row for row in pairwise if row["decision"] == DECISION_NO_DIRECTION]
    below_kmin_withheld = [row for row in withheld if row["withheld_reason"] == WITHHELD_BELOW_KMIN]
    assert_tsv(bundle, "UNIT_BY_METHOD_UTILITY_V1.tsv", ["dataset_id", "allocation_id", "unit_id", "method_id", "utility", "utility_hex"], normalized)
    assert_tsv(bundle, "PAIRWISE_COMPLETE_FAMILY_V1.tsv", pair_fields, pairwise)
    assert_tsv(bundle, "WITHHELD_COMPARISONS_V1.tsv", pair_fields, withheld)
    assert_tsv(bundle, "DESCRIPTIVE_SUMMARIES_V1.tsv", summary_fields, summaries)
    require((bundle / "REPORTING_CHECKLIST_V1.md").read_text(encoding="utf-8") == evaluation_checklist(), "evaluation checklist differs")
    gate_failures: list[str] = []
    if not independence_established:
        gate_failures.append(WITHHELD_INDEPENDENCE)
    if not training_evaluation_separation_established:
        gate_failures.append(WITHHELD_TRAIN_EVAL)
    expected_receipt = {
        "audit_id": config["audit_id"],
        "below_kmin_withheld_count": len(below_kmin_withheld),
        "bundle_kind": "utility_evaluation",
        "complete_family_size_per_dataset_allocation": size,
        "config_sha256": sha(config_path),
        "configuration_fixed_before_utility_access": config["analysis_timing"]["configuration_fixed_before_utility_access"],
        "decision_rule_id": config["inference"]["decision_rule_id"],
        "declared_unit_independence_established": independence_established,
        "descriptive_summary_count": len(summaries),
        "directional_rejection_count": len(directional_rejections),
        "eligible_no_direction_count": len(eligible_no_direction),
        "evidence_timing_label": config["analysis_timing"]["evidence_timing_label"],
        "gate_failures": gate_failures,
        "independence_assertion_machine_verified": False,
        "k_min": minimum,
        "optimization_certification_uncertain_count": 0,
        "pairwise_comparison_count": len(pairwise),
        "record_type": "BENCHMARK_DESIGN_AUDIT_RECEIPT_V1",
        "schema_version": "1.0.0",
        "status": "WITHHELD_REQUIRED_GATE_FAILURE" if gate_failures else "PASS_CONDITIONAL_EXACT_SIGN_HOLM_EVALUATION",
        "tie_rule": TIE_RULE,
        "training_evaluation_separation_established": training_evaluation_separation_established,
        "unit_audit_manifest_sha256": sha(manifest_snapshot_path),
        "unit_audit_receipt_sha256": sha(unit_receipt_path),
        "unit_metadata_sha256": sha(bundle / "INPUT_UNIT_METADATA.tsv"),
        "unit_utility_count": len(normalized),
        "upstream_cell_and_model_provenance_machine_verified": False,
        "utilities_sha256": sha(bundle / "INPUT_UTILITIES.tsv"),
        "withheld_comparison_count": len(withheld),
    }
    require(load_json(bundle / "EVALUATION_RECEIPT_V1.json") == expected_receipt, "evaluation receipt differs")
    require((bundle / "EVALUATION_RECEIPT_V1.json").read_text() == canonical_json(expected_receipt), "evaluation receipt encoding differs")


def verify(config_path: Path, bundle: Path) -> None:
    require(config_path.is_file() and not config_path.is_symlink(), "config must be regular")
    names = verify_manifest(bundle)
    config_contract(load_json(bundle / "CONFIG_SNAPSHOT.json"))
    require((bundle / "CONFIG_SNAPSHOT.json").read_bytes() == config_path.read_bytes(), "config snapshot differs")
    config = config_contract(load_json(config_path))
    if names == AUDIT_TREE:
        verify_audit(config_path, config, bundle)
    elif names == PLAN_TREE:
        verify_plan(config_path, config, bundle)
    elif names == EVALUATION_TREE:
        verify_evaluation(config_path, config, bundle)
    else:
        raise VerifyError("bundle file set does not match an audit, plan or evaluation bundle")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute and verify a benchmark-design audit bundle."
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="design configuration JSON"
    )
    parser.add_argument(
        "--bundle", type=Path, required=True, help="output bundle to verify"
    )
    args = parser.parse_args()
    try:
        verify(args.config, args.bundle)
    except (VerifyError, OSError, UnicodeError, KeyError, TypeError, ValueError) as error:
        print(f"BENCHMARK_DESIGN_AUDIT_VERIFY_ERROR: {error}", file=sys.stderr)
        return 2
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
