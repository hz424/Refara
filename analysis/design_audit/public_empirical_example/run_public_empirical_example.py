#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


EXAMPLE = Path(__file__).resolve().parent
AUDIT_ROOT = EXAMPLE.parent
REPOSITORY = EXAMPLE.parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from perturb_nuisance_contracts.source_data_root import (  # noqa: E402
    SourceDataRootError,
    validate_source_data_root,
)

CLI = AUDIT_ROOT / "benchmark_design_audit.py"
CONFIG = EXAMPLE / "config.json"
METADATA = EXAMPLE / "metadata.tsv"
UTILITIES_RUNTIME_PATH = Path(
    "data/derived/public_held_out_evidence/"
    "GSE162632_ROOT_UTILITIES_PUBLIC_V1.tsv"
)


def run(arguments: list[str], capture: bool = False) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [sys.executable, "-I", "-s", "-B", str(CLI), *arguments],
        check=False,
        capture_output=capture,
        text=True,
    )
    if completed.returncode != 0:
        if capture:
            sys.stderr.write(completed.stderr)
        raise RuntimeError(f"audit command failed: {arguments[0]}")
    return completed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit the supplied GSE162632 root utilities."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="new directory for the three audit bundles",
    )
    parser.add_argument(
        "--source-data-root",
        type=Path,
        required=True,
        help="extracted 05_SOURCE_DATA directory matching the v1.7.0 manifest",
    )
    args = parser.parse_args()
    snapshot = validate_source_data_root(args.source_data_root)
    utilities = snapshot.root / UTILITIES_RUNTIME_PATH
    if args.output_root.expanduser().is_symlink():
        raise RuntimeError("output root cannot be a symlink")
    output_root = args.output_root.expanduser().resolve()
    if output_root == snapshot.root or snapshot.root in output_root.parents:
        raise RuntimeError("output root cannot be inside Source Data root")
    if output_root.exists() or output_root.is_symlink():
        raise RuntimeError("output root must not exist")
    if not output_root.parent.is_dir() or output_root.parent.is_symlink():
        raise RuntimeError("output parent is invalid")
    output_root.mkdir()

    validation = run(["validate-config", "--config", str(CONFIG)], capture=True)
    validation_report = json.loads(validation.stdout)
    if validation_report["status"] != "PASS_DECLARATIVE_CONTRACT":
        raise RuntimeError("configuration did not pass the declarative contract")

    unit_audit = output_root / "unit-audit"
    comparison_plan = output_root / "comparison-plan"
    evaluation = output_root / "evaluation"
    run(
        [
            "audit-units",
            "--config",
            str(CONFIG),
            "--metadata",
            str(METADATA),
            "--output-dir",
            str(unit_audit),
        ]
    )
    run(
        [
            "plan-comparisons",
            "--config",
            str(CONFIG),
            "--output-dir",
            str(comparison_plan),
        ]
    )
    run(
        [
            "evaluate",
            "--config",
            str(CONFIG),
            "--utilities",
            str(utilities),
            "--unit-audit-bundle",
            str(unit_audit),
            "--output-dir",
            str(evaluation),
        ]
    )
    for bundle in (unit_audit, comparison_plan, evaluation):
        run(
            [
                "verify-output",
                "--config",
                str(CONFIG),
                "--bundle",
                str(bundle),
            ]
        )

    receipt = json.loads((evaluation / "EVALUATION_RECEIPT_V1.json").read_text())
    if receipt["pairwise_comparison_count"] != 168:
        raise RuntimeError("generic ordered-family size differs")
    if receipt["directional_rejection_count"] != 0:
        raise RuntimeError("eight-root support audit unexpectedly emitted a direction")
    if receipt["withheld_comparison_count"] != 168:
        raise RuntimeError("eight-root support audit did not withhold every below-Kmin direction")
    print("PASS_UTILITY_LEVEL_SUPPORT_AUDIT")
    snapshot.assert_unchanged()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SourceDataRootError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
