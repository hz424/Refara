#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


class DemoError(RuntimeError):
    pass


REPOSITORY = Path(__file__).resolve().parents[1]
AUDIT_ROOT = REPOSITORY / "analysis/design_audit"
CLI = AUDIT_ROOT / "benchmark_design_audit.py"
FIXTURE = AUDIT_ROOT / "quick_demo"
CONFIG = FIXTURE / "config.json"
CONFIG_WITHHELD = FIXTURE / "config_units_not_established.json"
METADATA = FIXTURE / "metadata.tsv"
UTILITIES = FIXTURE / "utilities.tsv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def run_command(label: str, output_root: Path, *arguments: str) -> None:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-I", "-s", "-B", str(CLI), *arguments],
        cwd=REPOSITORY,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    log = output_root / "logs" / f"{label}.json"
    log.write_text(
        json.dumps(
            {
                "arguments": list(arguments),
                "returncode": completed.returncode,
                "stderr": completed.stderr,
                "stdout": completed.stdout,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise DemoError(
            f"{label} failed with exit code {completed.returncode}; see {log}"
        )


def verify_bundle(
    label: str,
    output_root: Path,
    *,
    config: Path,
    bundle: Path,
) -> None:
    run_command(
        f"verify_{label}",
        output_root,
        "verify-output",
        "--config",
        str(config),
        "--bundle",
        str(bundle),
    )


def select(
    table: list[dict[str, str]],
    **criteria: str,
) -> dict[str, str]:
    matches = [
        row
        for row in table
        if all(row.get(field) == value for field, value in criteria.items())
    ]
    if len(matches) != 1:
        raise DemoError(f"expected one row for {criteria}, found {len(matches)}")
    return matches[0]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DemoError(message)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DemoError(f"expected a JSON object: {path}")
    return value


def prepare_output(requested: Path | None) -> Path:
    if requested is None:
        parent = Path(tempfile.mkdtemp(prefix="reference-cell-demo-"))
        output_root = parent / "outputs"
    else:
        requested_path = requested.expanduser()
        if requested_path.exists() or requested_path.is_symlink():
            raise DemoError(f"output path already exists: {requested_path}")
        output_root = requested_path.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise DemoError(f"output path already exists: {output_root}")
    output_root.mkdir(parents=True)
    (output_root / "logs").mkdir()
    return output_root


def execute(output_root: Path) -> dict[str, Any]:
    require(sys.version_info[:2] == (3, 10), "the quick demo requires Python 3.10")

    positive_config = load_json(CONFIG)
    negative_config = load_json(CONFIG_WITHHELD)
    positive_without_gate = json.loads(json.dumps(positive_config))
    negative_without_gate = json.loads(json.dumps(negative_config))
    positive_without_gate.pop("audit_id")
    negative_without_gate.pop("audit_id")
    positive_gate = positive_without_gate["unit_contract"]["independence_justification"]
    negative_gate = negative_without_gate["unit_contract"]["independence_justification"]
    positive_gate.pop("status")
    positive_gate.pop("basis")
    negative_gate.pop("status")
    negative_gate.pop("basis")
    require(
        positive_without_gate == negative_without_gate,
        "the negative fixture differs beyond its audit ID and independence declaration",
    )

    expected_utility_hash = positive_config["utility_input_contract"][
        "expected_input_sha256"
    ]
    require(expected_utility_hash == sha256(UTILITIES), "utility fixture hash differs")
    require(
        negative_config["utility_input_contract"]["expected_input_sha256"]
        == expected_utility_hash,
        "positive and negative fixtures do not bind the same utilities",
    )

    run_command("validate_positive", output_root, "validate-config", "--config", str(CONFIG))
    run_command(
        "validate_units_not_established",
        output_root,
        "validate-config",
        "--config",
        str(CONFIG_WITHHELD),
    )

    plan = output_root / "comparison_plan"
    positive_audit = output_root / "unit_audit"
    positive_evaluation = output_root / "evaluation"
    negative_audit = output_root / "unit_audit_units_not_established"
    negative_evaluation = output_root / "evaluation_units_not_established"

    run_command(
        "plan",
        output_root,
        "plan-comparisons",
        "--config",
        str(CONFIG),
        "--output-dir",
        str(plan),
    )
    run_command(
        "audit_positive",
        output_root,
        "audit-units",
        "--config",
        str(CONFIG),
        "--metadata",
        str(METADATA),
        "--output-dir",
        str(positive_audit),
    )
    run_command(
        "evaluate_positive",
        output_root,
        "evaluate",
        "--config",
        str(CONFIG),
        "--utilities",
        str(UTILITIES),
        "--unit-audit-bundle",
        str(positive_audit),
        "--output-dir",
        str(positive_evaluation),
    )
    run_command(
        "audit_units_not_established",
        output_root,
        "audit-units",
        "--config",
        str(CONFIG_WITHHELD),
        "--metadata",
        str(METADATA),
        "--output-dir",
        str(negative_audit),
    )
    run_command(
        "evaluate_units_not_established",
        output_root,
        "evaluate",
        "--config",
        str(CONFIG_WITHHELD),
        "--utilities",
        str(UTILITIES),
        "--unit-audit-bundle",
        str(negative_audit),
        "--output-dir",
        str(negative_evaluation),
    )

    verify_bundle("plan", output_root, config=CONFIG, bundle=plan)
    verify_bundle("audit_positive", output_root, config=CONFIG, bundle=positive_audit)
    verify_bundle(
        "evaluation_positive",
        output_root,
        config=CONFIG,
        bundle=positive_evaluation,
    )
    verify_bundle(
        "audit_units_not_established",
        output_root,
        config=CONFIG_WITHHELD,
        bundle=negative_audit,
    )
    verify_bundle(
        "evaluation_units_not_established",
        output_root,
        config=CONFIG_WITHHELD,
        bundle=negative_evaluation,
    )

    summaries = rows(positive_evaluation / "DESCRIPTIVE_SUMMARIES_V1.tsv")
    pairwise = rows(positive_evaluation / "PAIRWISE_COMPLETE_FAMILY_V1.tsv")
    withheld = rows(negative_evaluation / "PAIRWISE_COMPLETE_FAMILY_V1.tsv")
    positive_receipt = load_json(positive_evaluation / "EVALUATION_RECEIPT_V1.json")
    negative_receipt = load_json(negative_evaluation / "EVALUATION_RECEIPT_V1.json")

    expected_means = {
        ("shared_D", "METHOD_A"): ("13.5", "1"),
        ("shared_D", "METHOD_B"): ("12.5", "2"),
        ("disjoint_D", "METHOD_A"): ("12.5", "2"),
        ("disjoint_D", "METHOD_B"): ("13.5", "1"),
    }
    for (allocation, method), (mean, rank) in expected_means.items():
        row = select(summaries, allocation_id=allocation, method_id=method)
        require(row["equal_unit_mean_utility"] == mean, "unexpected demo mean")
        require(row["point_rank_by_equal_unit_mean"] == rank, "unexpected demo rank")

    directional_pairs = {
        ("shared_D", "METHOD_A", "METHOD_B"),
        ("disjoint_D", "METHOD_B", "METHOD_A"),
    }
    for allocation, model_a, model_b in directional_pairs:
        row = select(
            pairwise,
            allocation_id=allocation,
            model_a_id=model_a,
            model_b_id=model_b,
        )
        require(row["decision"] == "DIRECTION", "expected a directional result")
        require((row["wins"], row["losses"]) == ("6", "0"), "unexpected signs")
        require(row["exact_one_sided_p"] == "0.015625", "unexpected exact p value")
        require(row["holm_adjusted_p"] == "0.03125", "unexpected adjusted p value")

    require(
        {row["decision"] for row in pairwise} == {"DIRECTION", "NO_DIRECTION"},
        "unexpected positive-run decision set",
    )
    require(len(withheld) == 4, "negative run should contain four comparisons")
    require(
        {row["decision"] for row in withheld} == {"WITHHELD"},
        "negative run should withhold every comparison",
    )
    require(
        {row["withheld_reason"] for row in withheld}
        == {"UNIT_INDEPENDENCE_NOT_ESTABLISHED"},
        "negative run has an unexpected withholding reason",
    )
    require(
        {row["reject_at_alpha"] for row in withheld} == {"NA"},
        "withheld comparisons should not have rejection decisions",
    )

    expected_positive_receipt = {
        "unit_utility_count": 24,
        "pairwise_comparison_count": 4,
        "directional_rejection_count": 2,
        "eligible_no_direction_count": 2,
        "withheld_comparison_count": 0,
        "status": "PASS_CONDITIONAL_EXACT_SIGN_HOLM_EVALUATION",
    }
    for key, expected in expected_positive_receipt.items():
        require(positive_receipt.get(key) == expected, f"unexpected positive receipt field: {key}")
    require(
        negative_receipt.get("status") == "WITHHELD_REQUIRED_GATE_FAILURE",
        "negative receipt status differs",
    )
    require(
        negative_receipt.get("directional_rejection_count") == 0
        and negative_receipt.get("withheld_comparison_count") == 4,
        "negative receipt counts differ",
    )

    result = {
        "status": "PASS_QUICK_DEMO",
        "fixture_utility_sha256": expected_utility_hash,
        "positive_config_sha256": sha256(CONFIG),
        "units_not_established_config_sha256": sha256(CONFIG_WITHHELD),
        "shared_D": {
            "METHOD_A_mean": "13.5",
            "METHOD_B_mean": "12.5",
            "direction": "METHOD_A > METHOD_B",
            "holm_adjusted_p": "0.03125",
        },
        "disjoint_D": {
            "METHOD_A_mean": "12.5",
            "METHOD_B_mean": "13.5",
            "direction": "METHOD_B > METHOD_A",
            "holm_adjusted_p": "0.03125",
        },
        "units_not_established": {
            "decision": "WITHHELD",
            "reason": "UNIT_INDEPENDENCE_NOT_ESTABLISHED",
        },
        "separate_bundle_verification": "PASS",
    }
    (output_root / "QUICK_DEMO_RECEIPT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Run the bundled synthetic design-audit example.")
    command.add_argument(
        "--output-root",
        type=Path,
        help="new directory in which to write the demo output",
    )
    return command


def main() -> int:
    arguments = parser().parse_args()
    try:
        output_root = prepare_output(arguments.output_root)
        execute(output_root)
    except (DemoError, OSError, UnicodeError, json.JSONDecodeError) as error:
        print(f"Quick demo failed: {error}", file=sys.stderr)
        return 2
    print("Quick demo passed")
    print(
        "shared_D: METHOD_A 13.5, METHOD_B 12.5; "
        "METHOD_A > METHOD_B (Holm-adjusted p=0.03125)"
    )
    print(
        "disjoint_D: METHOD_A 12.5, METHOD_B 13.5; "
        "METHOD_B > METHOD_A (Holm-adjusted p=0.03125)"
    )
    print("The descriptive leader changes from METHOD_A to METHOD_B.")
    print("units_not_established: WITHHELD (UNIT_INDEPENDENCE_NOT_ESTABLISHED)")
    print("Output bundles verified.")
    print(f"Outputs: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
