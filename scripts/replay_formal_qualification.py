#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


class ReplayError(RuntimeError):
    pass


REPOSITORY = Path(__file__).resolve().parents[1]
PROGRAM = REPOSITORY / "scripts/qualify_generic_reference_framework_v1.py"
CONFIG = (
    REPOSITORY
    / "configs/qualification/generic_reference_aware_framework_v1_0.json"
)
REFERENCE = (
    REPOSITORY
    / "data/derived/qualification/generic_reference_aware_framework_v1_0/report.json"
)
EXPECTED_STATUS = "NO_GO_GENERIC_FRAMEWORK_V1_OUTCOME_FREE_VALIDITY"
EXPECTED_REFERENCE_SHA256 = (
    "4f621654dd40c1966e33a94a92fe71804a4113910e94852ea842cdf46f7fe4d9"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReplayError(f"expected a JSON object: {path}")
    return value


def normalized(value: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(value)
    execution = result.get("execution")
    if not isinstance(execution, dict) or "workers" not in execution:
        raise ReplayError("qualification report does not record its worker count")
    execution.pop("workers")
    return result


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description=(
            "Reproduce the recorded "
            "NO_GO_GENERIC_FRAMEWORK_V1_OUTCOME_FREE_VALIDITY result for the "
            "generic inference procedure."
        )
    )
    command.add_argument(
        "--workers", type=int, required=True, help="number of worker processes"
    )
    command.add_argument(
        "--output-root", type=Path, required=True, help="new directory for replay outputs"
    )
    return command


def execute(workers: int, output_root: Path) -> dict[str, Any]:
    if workers < 1:
        raise ReplayError("--workers must be positive")
    output_root = output_root.expanduser().resolve()
    if output_root.exists() or output_root.is_symlink():
        raise ReplayError(f"output path already exists: {output_root}")
    output_root.mkdir(parents=True)
    generated_path = output_root / "report.json"

    if sha256(REFERENCE) != EXPECTED_REFERENCE_SHA256:
        raise ReplayError("the frozen qualification report hash differs")
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-s",
            "-B",
            str(PROGRAM),
            "--config",
            str(CONFIG),
            "--output",
            str(generated_path),
            "--workers",
            str(workers),
        ],
        cwd=REPOSITORY,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    (output_root / "QUALIFICATION_COMMAND.json").write_text(
        json.dumps(
            {
                "returncode": completed.returncode,
                "stderr": completed.stderr,
                "stdout": completed.stdout,
                "workers": workers,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if completed.returncode != 2:
        raise ReplayError(
            "the formal run did not return the frozen scientific exit code 2"
        )

    generated = load_json(generated_path)
    reference = load_json(REFERENCE)
    for label, report in (("generated", generated), ("frozen", reference)):
        if report.get("status") != EXPECTED_STATUS:
            raise ReplayError(f"{label} report has an unexpected disposition")
        if report.get("scientific_exit_code") != 2:
            raise ReplayError(f"{label} report has an unexpected scientific exit code")
        execution = report.get("execution")
        if not isinstance(execution, dict):
            raise ReplayError(f"{label} report lacks execution metadata")
        if execution.get("scenario_count") != 198:
            raise ReplayError(f"{label} report does not contain 198 scenarios")
        if execution.get("repetitions_per_scenario") != 10_000:
            raise ReplayError(f"{label} report does not use 10,000 repetitions")
    if normalized(generated) != normalized(reference):
        raise ReplayError("generated and frozen reports differ beyond worker count")

    failed = [
        scenario
        for scenario in generated.get("scenarios", [])
        if scenario.get("joint_validity_pass") is not True
    ]
    receipt = {
        "status": "PASS_REPRODUCED_FROZEN_QUALIFICATION_DISPOSITION",
        "scientific_disposition": EXPECTED_STATUS,
        "scientific_exit_code": 2,
        "scenario_count": 198,
        "repetitions_per_scenario": 10_000,
        "failed_joint_validity_scenario_count": len(failed),
        "workers": workers,
        "generated_report_sha256": sha256(generated_path),
        "frozen_report_sha256": EXPECTED_REFERENCE_SHA256,
        "reports_match_except_recorded_worker_count": True,
        "byte_identical_to_frozen_report": generated_path.read_bytes()
        == REFERENCE.read_bytes(),
    }
    (output_root / "QUALIFICATION_REPLAY_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> int:
    arguments = parser().parse_args()
    try:
        receipt = execute(arguments.workers, arguments.output_root)
    except (ReplayError, OSError, UnicodeError, json.JSONDecodeError) as error:
        print(f"Qualification replay failed: {error}", file=sys.stderr)
        return 2
    print("Formal qualification replay matched the frozen result.")
    print(f"Disposition: {receipt['scientific_disposition']}")
    print(
        "Prespecified validity criteria were not met in "
        f"{receipt['failed_joint_validity_scenario_count']} of 198 scenarios."
    )
    print(f"Outputs: {arguments.output_root.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
