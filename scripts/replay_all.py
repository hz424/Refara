#!/usr/bin/env python3
"""Run all aggregate numerical analyses from the bound Source Data release.

The command validates an extracted ``05_SOURCE_DATA`` directory and performs
the replay in a temporary workspace. It neither downloads data nor writes to
the supplied Source Data directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from perturb_nuisance_contracts.source_data_root import (  # noqa: E402
    EXPECTED_SOURCE_DATA_RELEASE_VERSION,
    SourceDataRootError,
    validate_source_data_root,
)


class ReplayAllError(RuntimeError):
    """A replay stage or runtime precondition failed."""


ANALYSIS_STAGES = (
    (
        "held_out_84",
        "scripts/build_gse162632_public_evidence_v1.py",
        "PASS_EXACT_PUBLIC_ROOT_TO_FULL_84_VERIFICATION",
        ("--check",),
    ),
    (
        "paired_construction_56",
        "scripts/build_gse162632_paired_construction_changes_public_v1.py",
        "PASS_EXACT_PAIRED_CONSTRUCTION_CHANGE_56_VERIFICATION",
        ("--check",),
    ),
    (
        "paired_construction_structural_zeros_v189",
        "analysis/diagnostic_correction_v189/replay.py",
        '"status": "PASS"',
        ("--verify",),
    ),
    (
        "multi_realization",
        "analysis/multi_realization/v21_multi_realization_public_replay.py",
        "PASS_V21_MULTI_REALIZATION_PUBLIC_REPLAY",
        (),
    ),
    (
        "cmodel_successor",
        "analysis/cmodel_successor/cmodel_successor_public_replay.py",
        "PASS_CMODEL_SUCCESSOR_PUBLIC_REPLAY",
        (),
    ),
    (
        "v22_attribution",
        "analysis/v22_attribution/v22_public_replay.py",
        "PASS_PUBLIC_ROOT_REPLAY",
        (),
    ),
)
TREE_SKIP_PARTS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "outputs",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayAllError(message)


def executable(path: Path, *, label: str) -> Path:
    # The invocation path selects pyvenv.cfg. Dereferencing a venv's Python
    # symlink would run the base environment and lose its installed packages.
    supplied = path.expanduser().absolute()
    require(supplied.is_file() and os.access(supplied, os.X_OK), f"{label} interpreter is not executable")
    return supplied


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(
            part in TREE_SKIP_PARTS
            or part.startswith(".venv-")
            or part.endswith(".egg-info")
            for part in relative.parts
        ) or path.suffix == ".pyc":
            continue
        require(not path.is_symlink(), f"unsafe code-tree symlink: {relative}")
        if path.is_dir():
            continue
        require(path.is_file(), f"unsafe code-tree member: {relative}")
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def copy_repository(destination: Path) -> None:
    shutil.copytree(
        REPOSITORY,
        destination,
        symlinks=True,
        ignore=shutil.ignore_patterns(
            ".git",
            ".pytest_cache",
            ".venv",
            ".venv-*",
            "__pycache__",
            "*.pyc",
            "*.egg-info",
            "outputs",
        ),
    )


def replay_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "MPLBACKEND": "Agg",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "SOURCE_DATE_EPOCH": "946684800",
        }
    )
    return environment


def run_stage(
    *,
    name: str,
    markers: Sequence[str],
    command: Sequence[str],
    receipt_argv: Sequence[str],
    entrypoint: Path,
    sandbox: Path,
    environment: dict[str, str],
) -> dict[str, object]:
    started = time.monotonic()
    completed = subprocess.run(
        list(command),
        cwd=sandbox,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise ReplayAllError(
            f"stage {name} failed with exit code {completed.returncode}: {detail}"
        )
    elapsed = round(time.monotonic() - started, 6)
    for marker in markers:
        require(marker in completed.stdout, f"stage {name} omitted success marker {marker}")
    status = markers[0] if len(markers) == 1 else f"PASS_{name.upper()}"
    return {
        "stage": name,
        "status": status,
        "success_markers": list(markers),
        "argv": list(receipt_argv),
        "entrypoint": entrypoint.relative_to(sandbox).as_posix(),
        "entrypoint_sha256": sha256_file(entrypoint),
        "elapsed_seconds": elapsed,
        "exit_code": completed.returncode,
    }


def replay_all(
    *,
    source_data_root: Path,
    legacy_python: Path,
    editorial_python: Path | None = None,
    analysis_only: bool = True,
) -> dict[str, object]:
    replay_started = time.monotonic()
    snapshot = validate_source_data_root(source_data_root)
    legacy = executable(legacy_python, label="legacy")
    # Retained keyword arguments are compatibility aliases; every run is numerical.
    environment = replay_environment()
    records: list[dict[str, object]] = []
    code_tree_sha256 = repository_tree_sha256(REPOSITORY)

    with tempfile.TemporaryDirectory(prefix="reference-cell-all-replay-") as temporary:
        sandbox = Path(temporary) / "code"
        copy_repository(sandbox)
        sandbox_tree_sha256 = repository_tree_sha256(sandbox)
        require(
            sandbox_tree_sha256 == code_tree_sha256,
            "copied replay tree differs from the receipt-bound code tree",
        )
        source_argument = ("--source-data-root", str(snapshot.root))
        for name, relative_script, marker, fixed_arguments in ANALYSIS_STAGES:
            command = (
                str(legacy),
                "-B",
                str(sandbox / relative_script),
                *fixed_arguments,
                *source_argument,
            )
            records.append(
                run_stage(
                    name=name,
                    markers=(marker,),
                    command=command,
                    receipt_argv=(
                        "<LEGACY_PYTHON>",
                        "-B",
                        relative_script,
                        *fixed_arguments,
                        "--source-data-root",
                        "<SOURCE_DATA_ROOT>",
                    ),
                    entrypoint=sandbox / relative_script,
                    sandbox=sandbox,
                    environment=environment,
                )
            )

        for name, relative_script, fixed_arguments in (
            ("focal_five_realization_capsule", "scripts/focal_scgen.py", ("reproduce-focal",)),
            ("matched_allocation_1000_labels", "capsules/gse162632_allocation/replay.py", source_argument),
        ):
            record = run_stage(
                name=name, markers=('"status": "PASS"',),
                command=(str(legacy), "-B", str(sandbox / relative_script), *fixed_arguments),
                receipt_argv=("<LEGACY_PYTHON>", "-B", relative_script,
                              *(fixed_arguments if name.startswith("focal") else ("--source-data-root", "<SOURCE_DATA_ROOT>"))),
                entrypoint=sandbox / relative_script, sandbox=sandbox, environment=environment,
            )
            record["status"] = f"PASS_{name.upper()}"
            records.append(record)

        generic_output = Path(temporary) / "generic-self-test.json"
        generic_record = run_stage(
            name="generic_framework_self_test",
            markers=(),
            command=(
                str(legacy),
                "-B",
                str(sandbox / "scripts/qualify_generic_reference_framework_v1.py"),
                "--self-test",
                "--workers",
                "1",
                "--output",
                str(generic_output),
            ),
            receipt_argv=(
                "<LEGACY_PYTHON>",
                "-B",
                "scripts/qualify_generic_reference_framework_v1.py",
                "--self-test",
                "--workers",
                "1",
                "--output",
                "<TEMPORARY_OUTPUT>/generic-self-test.json",
            ),
            entrypoint=sandbox / "scripts/qualify_generic_reference_framework_v1.py",
            sandbox=sandbox,
            environment=environment,
        )
        generic_report = json.loads(generic_output.read_text(encoding="utf-8"))
        require(
            generic_report.get("status") == "PASS_GENERIC_FRAMEWORK_V1_IMPLEMENTATION_SELF_TEST",
            "generic self-test report status differs",
        )
        generic_record["status"] = "PASS_GENERIC_FRAMEWORK_V1_IMPLEMENTATION_SELF_TEST"
        generic_record["success_markers"] = [
            "PASS_GENERIC_FRAMEWORK_V1_IMPLEMENTATION_SELF_TEST"
        ]
        records.append(generic_record)

        audit_output = Path(temporary) / "design-audit"
        records.append(
            run_stage(
                name="design_audit_public_empirical_example",
                markers=("PASS_UTILITY_LEVEL_SUPPORT_AUDIT",),
                command=(
                    str(legacy),
                    "-B",
                    str(
                        sandbox
                        / "analysis/design_audit/public_empirical_example/run_public_empirical_example.py"
                    ),
                    "--output-root",
                    str(audit_output),
                    *source_argument,
                ),
                receipt_argv=(
                    "<LEGACY_PYTHON>",
                    "-B",
                    "analysis/design_audit/public_empirical_example/run_public_empirical_example.py",
                    "--output-root",
                    "<TEMPORARY_OUTPUT>/design-audit",
                    "--source-data-root",
                    "<SOURCE_DATA_ROOT>",
                ),
                entrypoint=(
                    sandbox
                    / "analysis/design_audit/public_empirical_example/run_public_empirical_example.py"
                ),
                sandbox=sandbox,
                environment=environment,
            )
        )

        require(
            repository_tree_sha256(sandbox) == sandbox_tree_sha256,
            "receipt-bound replay code changed during execution",
        )

    snapshot.assert_unchanged()
    elapsed = round(time.monotonic() - replay_started, 6)
    return {
        "schema": "REFERENCE_CELL_CODE_ONLY_REPLAY_ALL_V1",
        "status": "PASS_REFERENCE_CELL_CODE_ONLY_REPLAY_ALL_V1",
        "code_release_version": "1.10.0",
        "source_data_release_version": EXPECTED_SOURCE_DATA_RELEASE_VERSION,
        "input_manifest": {
            "path": "SOURCE_DATA_ROOT_MANIFEST_V1.json",
            "sha256": snapshot.manifest_sha256,
            "payload_count": len(snapshot.payloads),
            "status": "PASS_EXACT_RELEASE_BOUND_SOURCE_DATA_ROOT",
        },
        "code_tree_sha256": code_tree_sha256,
        "mode": "ANALYSIS_ONLY",
        "stages": records,
        "elapsed_seconds": elapsed,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--source-data-root",
        type=Path,
        required=True,
        help="extracted 05_SOURCE_DATA directory matching the bundled manifest",
    )
    result.add_argument(
        "--legacy-python",
        type=Path,
        default=Path(sys.executable),
        help="Python 3.10.17 interpreter matching requirements/locked.txt",
    )
    result.add_argument(
        "--editorial-python",
        type=Path,
        help="Compatibility option; ignored by numerical-only replay",
    )
    result.add_argument(
        "--analysis-only",
        action="store_true",
        help="Compatibility option; aggregate replay is always numerical-only",
    )
    result.add_argument(
        "--receipt",
        type=Path,
        help="write the replay receipt to a new file",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    editorial = args.editorial_python
    if editorial is None:
        environment_path = os.environ.get("REFERENCE_CELL_EDITORIAL_PYTHON")
        if environment_path:
            editorial = Path(environment_path)
    report = replay_all(
        source_data_root=args.source_data_root,
        legacy_python=args.legacy_python,
        editorial_python=editorial,
        analysis_only=args.analysis_only,
    )
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.receipt is not None:
        supplied_receipt = args.receipt.expanduser()
        if supplied_receipt.is_symlink():
            raise ReplayAllError("receipt path cannot be a symlink")
        receipt = supplied_receipt.resolve()
        if receipt.exists() or receipt.is_symlink():
            raise ReplayAllError(f"receipt already exists: {receipt}")
        if not receipt.parent.is_dir() or receipt.parent.is_symlink():
            raise ReplayAllError("receipt parent is absent or unsafe")
        source_root = args.source_data_root.expanduser().resolve(strict=True)
        if receipt == source_root or source_root in receipt.parents:
            raise ReplayAllError("receipt cannot be written inside Source Data root")
        temporary = receipt.with_name(f".{receipt.name}.tmp")
        if temporary.exists() or temporary.is_symlink():
            raise ReplayAllError(f"temporary receipt path already exists: {temporary}")
        temporary.write_text(serialized, encoding="utf-8", newline="")
        os.replace(temporary, receipt)
    sys.stdout.write(serialized)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReplayAllError, SourceDataRootError, OSError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
