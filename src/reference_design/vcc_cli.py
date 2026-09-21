"""Run the organizer audit through a separate official-scoring interpreter."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from . import vcc


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _python(value: str) -> str:
    executable = shutil.which(value)
    if executable is None:
        candidate = Path(value).expanduser()
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise ValueError(f"Official Python is not executable: {value}")
        executable = str(candidate.absolute())
    # Keep the virtualenv path: resolving its symlink would select the base runtime.
    return os.path.abspath(executable)


class WorkerFailure(RuntimeError):
    """A failed attempt with its execution evidence attached."""

    def __init__(self, message: str, receipt: dict[str, Any]):
        super().__init__(message)
        self.receipt = receipt


def _worker(job: dict[str, Any], python: str, root: Path, index: int,
            phase: str, threads: int) -> dict[str, Any]:
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)
    stdout = log_dir / f"{phase}_{index:04d}.out"
    stderr = log_dir / f"{phase}_{index:04d}.err"
    command = [python, str(Path(__file__).with_name("vcc_worker.py")),
               "--request", str(job["request"]), "--output", str(job["output_dir"])]
    environment = os.environ.copy()
    # The official worker must not inherit the Python 3.10 package path.
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
        environment[key] = str(threads)
    receipt = dict(phase=phase, request=str(job["request"]),
                   request_sha256=job["request_sha256"],
                   output_dir=str(job["output_dir"]), started=_now(),
                   returncode=None, stdout=str(stdout), stderr=str(stderr))
    try:
        receipt["request_sha256_before"] = vcc.sha256_file(job["request"])
        if receipt["request_sha256_before"] != job["request_sha256"]:
            raise ValueError("Worker request changed after planning")
        with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
            result = subprocess.run(command, stdout=out, stderr=err, env=environment, check=False)
        receipt["returncode"] = result.returncode
        receipt["request_sha256_after"] = vcc.sha256_file(job["request"])
        if receipt["request_sha256_after"] != job["request_sha256"]:
            raise ValueError("Worker request changed during execution")
        if result.returncode:
            tail = "\n".join(stderr.read_text(encoding="utf-8", errors="replace").splitlines()[-15:])
            raise RuntimeError(f"Official worker failed ({phase}, exit {result.returncode}). "
                               f"See {stderr}\n{tail}")
    except Exception as error:
        receipt.update(status="FAILED", ended=_now(), error=str(error))
        raise WorkerFailure(str(error), receipt) from error
    receipt.update(status="COMPLETE", ended=_now())
    return receipt


def run_audit(manifest: str | Path, output: str | Path, official_python: str) -> dict[str, Any]:
    """Plan, prepare calibration once per design, score all models and summarize."""
    python = _python(official_python)
    declaration = vcc.validate_manifest(manifest)
    threads = declaration["runtime"]["num_threads"]
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated and threads > int(allocated):
        raise ValueError("runtime.num_threads exceeds SLURM_CPUS_PER_TASK")
    root = Path(output).absolute()
    planning = vcc.plan(manifest, root)
    execution: dict[str, Any] = dict(status="RUNNING", started=_now(),
                                    official_python=python, jobs=[])
    _write(root / "EXECUTION.json", execution)
    try:
        preparations = planning["preparations"]
        for index, job in enumerate(preparations):
            print(f"Prepare reference design {index + 1}/{len(preparations)}", flush=True)
            execution["jobs"].append(_worker(job, python, root, index, "prepare", threads))
            _write(root / "EXECUTION.json", execution)
        scoring = vcc.score_requests(root)
        for index, job in enumerate(scoring):
            print(f"Score frozen prediction {index + 1}/{len(scoring)}", flush=True)
            execution["jobs"].append(_worker(job, python, root, index, "score", threads))
            _write(root / "EXECUTION.json", execution)
        summary = vcc.summarize(root)
        execution["summary"] = summary
        if summary["status"] not in {"COMPLETE_DESCRIPTIVE_VCC_AUDIT", "UNAVAILABLE_OFFICIAL_SCORES"}:
            raise RuntimeError(f"Audit is not complete: {summary['status']}")
        execution.update(status=("COMPLETE" if summary["status"] == "COMPLETE_DESCRIPTIVE_VCC_AUDIT"
                                 else "COMPLETE_WITH_UNAVAILABLE_ANALYSES"), ended=_now())
        _write(root / "EXECUTION.json", execution)
        return execution
    except BaseException as error:
        if isinstance(error, WorkerFailure):
            execution["jobs"].append(error.receipt)
        execution.update(status="FAILED", ended=_now(), error=str(error))
        _write(root / "EXECUTION.json", execution)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reference-design vcc",
                                     description="Organizer-side reference sensitivity using official cell-eval2 scoring.")
    commands = parser.add_subparsers(dest="command", required=True)
    planning = commands.add_parser("plan", help="Validate sources and write context-level scoring requests")
    planning.add_argument("manifest", type=Path)
    planning.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run", help="Prepare, score and summarize using a separate official Python runtime")
    run.add_argument("manifest", type=Path)
    run.add_argument("--python", required=True, help="Python >=3.11 with the pinned official scorer installed")
    run.add_argument("--output", type=Path, required=True)
    summarize = commands.add_parser("summarize", help="Verify completed scoring outputs and summarize sensitivity")
    summarize.add_argument("run_dir", type=Path)
    summarize.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result = vcc.plan(args.manifest, args.output)
            print(json.dumps({key: result[key] for key in ("preparation_count", "score_count")}, indent=2))
        elif args.command == "run":
            result = run_audit(args.manifest, args.output, args.python)
            print(json.dumps({"status": result["status"], "output": str(args.output)}, indent=2))
            if result["status"] == "COMPLETE_WITH_UNAVAILABLE_ANALYSES":
                return 2
        else:
            result = vcc.summarize(args.run_dir, args.output)
            print(json.dumps(result, indent=2))
            if result["status"] != "COMPLETE_DESCRIPTIVE_VCC_AUDIT":
                return 2
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
