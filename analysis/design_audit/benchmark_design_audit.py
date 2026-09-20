#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from design_audit.core import (
    AuditError,
    audit_units,
    canonical_json,
    evaluate_utilities,
    load_config,
    plan_comparisons,
    validation_report,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        prog="benchmark-design-audit",
        description=(
            "Record a declared benchmark design and evaluate complete ordered "
            "comparisons from acquisition-unit utilities."
        ),
        epilog=(
            "The audit checks supplied declarations and calculations; it does "
            "not verify upstream cell assignments or unit independence."
        ),
    )
    subcommands = command.add_subparsers(
        dest="command", required=True, metavar="COMMAND"
    )
    validate = subcommands.add_parser(
        "validate-config", help="validate a design configuration against the fixed schema"
    )
    validate.add_argument(
        "--config", type=Path, required=True, help="design configuration JSON"
    )
    audit = subcommands.add_parser(
        "audit-units", help="check unit metadata and create an audit bundle"
    )
    audit.add_argument(
        "--config", type=Path, required=True, help="design configuration JSON"
    )
    audit.add_argument(
        "--metadata", type=Path, required=True, help="unit metadata TSV"
    )
    audit.add_argument(
        "--output-dir", type=Path, required=True, help="new output directory"
    )
    plan = subcommands.add_parser(
        "plan-comparisons", help="create the complete ordered comparison plan"
    )
    plan.add_argument(
        "--config", type=Path, required=True, help="design configuration JSON"
    )
    plan.add_argument(
        "--output-dir", type=Path, required=True, help="new output directory"
    )
    evaluate = subcommands.add_parser(
        "evaluate", help="evaluate supplied utilities with the declared rule"
    )
    evaluate.add_argument(
        "--config", type=Path, required=True, help="design configuration JSON"
    )
    evaluate.add_argument(
        "--utilities", type=Path, required=True, help="unit-by-method utility TSV"
    )
    evaluate.add_argument(
        "--unit-audit-bundle",
        type=Path,
        required=True,
        help="bundle created by audit-units",
    )
    evaluate.add_argument(
        "--output-dir", type=Path, required=True, help="new output directory"
    )
    verify = subcommands.add_parser(
        "verify-output", help="recompute and verify an output bundle"
    )
    verify.add_argument(
        "--config", type=Path, required=True, help="design configuration JSON"
    )
    verify.add_argument(
        "--bundle", type=Path, required=True, help="output bundle to verify"
    )
    return command


def main() -> int:
    arguments = parser().parse_args()
    try:
        config = load_config(arguments.config)
        if arguments.command == "validate-config":
            sys.stdout.write(canonical_json(validation_report(config)))
        elif arguments.command == "audit-units":
            audit_units(arguments.config, arguments.metadata, arguments.output_dir)
            print("PASS")
        elif arguments.command == "plan-comparisons":
            plan_comparisons(arguments.config, arguments.output_dir)
            print("PASS")
        elif arguments.command == "evaluate":
            evaluate_utilities(
                arguments.config,
                arguments.utilities,
                arguments.unit_audit_bundle,
                arguments.output_dir,
            )
            print("PASS")
        elif arguments.command == "verify-output":
            verifier = Path(__file__).with_name("verify_benchmark_design_audit_v1.py")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-s",
                    "-B",
                    str(verifier),
                    "--config",
                    str(arguments.config),
                    "--bundle",
                    str(arguments.bundle),
                ],
                check=False,
            )
            if completed.returncode != 0:
                raise AuditError("separate verifier rejected bundle")
    except (AuditError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"BENCHMARK_DESIGN_AUDIT_ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
