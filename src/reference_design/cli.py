"""Command-line allocation and reference-sensitivity scoring."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .core import PATTERNS, Prediction, allocate_controls, effect_to_state, score_references
from .io import BLOCKS, BlockTable, label, match_blocks, read_blocks, read_cells, read_vector, verify_membership


def _path(value: Any, directory: Path, field: str) -> Path:
    return directory / label(value, field)


def _schema(value: Any, required: set[str], optional: set[str], name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing or unknown:
        raise ValueError(f"{name}: missing fields {sorted(missing)}; unknown fields {sorted(unknown)}")


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field {key!r}")
        result[key] = value
    return result


def _json_constant(value: str) -> None:
    raise ValueError(f"JSON does not permit {value}")


def _destination(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Output already exists and is not empty: {path}")


def _tsv(columns: list[str], rows: list[dict[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _publish(output: Path, contents: dict[str, bytes]) -> None:
    _destination(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".reference-design-", dir=output.parent) as temporary:
        staging = Path(temporary)
        for name, data in contents.items():
            (staging / name).write_bytes(data)
        _destination(output)
        staging.replace(output)


def _allocate(args: argparse.Namespace) -> None:
    _destination(args.output)
    if args.allocations < 1 or args.seed < 0 or any(depth < 1 for depth in args.depth):
        raise ValueError("Allocations and depths must be positive; seed must be nonnegative")
    if len(set(args.depth)) != len(args.depth):
        raise ValueError("Repeated depths are not allowed")
    cells = read_cells(args.controls)
    reserved = {"allocation", "depth", "block", "cells_per_block"}
    if reserved & set(cells.genes):
        raise ValueError("Gene names conflict with reference-table metadata columns")
    references: list[dict[str, Any]] = []
    membership: list[dict[str, Any]] = []
    for allocation in range(args.allocations):
        for depth in sorted(args.depth):
            sampled = allocate_controls(cells.values, cells.cell_ids, depth,
                                        args.seed + allocation, cells.strata, blocks=args.blocks)
            for block, name in enumerate(BLOCKS[:args.blocks]):
                row = dict(allocation=str(allocation), depth=depth, block=name,
                           cells_per_block=sampled.cells_per_block)
                row.update(zip(cells.genes, sampled.means[block]))
                references.append(row)
            for cell in sampled.membership:
                membership.append(dict(
                    allocation=str(allocation), depth=depth, cell_id=cell["cell_id"],
                    stratum=cell["stratum"], block=BLOCKS[cell["block"]],
                    within_block_index=cell["within_block_index"],
                ))
    digest = hashlib.sha256()
    with args.controls.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    parameters = dict(seed=args.seed, allocations=args.allocations, depths=sorted(args.depth),
                      strata_count=len(set(cells.strata)), numpy_version=np.__version__,
                      controls_sha256=digest.hexdigest())
    _publish(args.output, {
        "references.tsv": _tsv(["allocation", "depth", "block", "cells_per_block", *cells.genes], references),
        "membership.tsv": _tsv(["allocation", "depth", "cell_id", "stratum", "block", "within_block_index"], membership),
        "allocation.json": (json.dumps(parameters, indent=2, allow_nan=False) + "\n").encode(),
    })


def _model(spec: Any, directory: Path, genes: tuple[str, ...], references: BlockTable,
           record_path: Callable[[Path], Path] | None = None) -> dict[str, Any]:
    _schema(spec, {"name", "kind", "prediction"}, {"conditioning", "representation", "baseline", "generation_record"}, "Model")
    name = label(spec["name"], "Model name")
    kind = spec["kind"]
    conditioning = spec.get("conditioning", "fixed")
    representation = spec.get("representation", "native")
    if kind not in ("state", "effect") or conditioning not in ("fixed", "block"):
        raise ValueError(f"{name}: use state/effect kind and fixed/block conditioning")
    if representation not in ("native", "effect"):
        raise ValueError(f"{name}: representation must be native or effect")
    converted = kind == "state" and representation == "effect"
    if kind == "effect" and representation != "native":
        raise ValueError(f"{name}: a native effect must use native representation")
    if ("baseline" in spec) != converted:
        raise ValueError(f"{name}: baseline is required only for a state stored in effect representation")
    def source(field: str) -> Path:
        path = _path(spec[field], directory, f"{field.capitalize()} path")
        return path if record_path is None else record_path(path)

    path = source("prediction")
    if conditioning == "fixed":
        _, values = read_vector(path, genes)
        if converted:
            _, baseline = read_vector(source("baseline"), genes)
            values = effect_to_state(values, baseline)
        return dict(name=name, kind=kind, fixed=values)
    values = read_blocks(path, genes)
    match_blocks(values, references, path)
    if converted:
        baseline_path = source("baseline")
        baseline = read_blocks(baseline_path, genes)
        match_blocks(baseline, references, baseline_path)
        values.values = {key: effect_to_state(value, baseline.values[key]) for key, value in values.values.items()}
    return dict(name=name, kind=kind, blocks=values.values)


def _figure(pairs: list[dict[str, Any]]) -> bytes:
    from .plot import write_plot

    with tempfile.TemporaryDirectory(prefix="reference-design-plot-") as directory:
        path = Path(directory) / "sensitivity.pdf"
        write_plot(pairs, path)
        return path.read_bytes()


def _score(args: argparse.Namespace) -> None:
    _destination(args.output)
    inputs: dict[str, str] = {}

    def file_digest(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def record_path(path: Path) -> Path:
        resolved = path.resolve(strict=True)
        digest = file_digest(resolved)
        previous = inputs.setdefault(str(resolved), digest)
        if previous != digest:
            raise ValueError(f"Scoring input changed during the run: {resolved}")
        return resolved

    configuration = Path(args.config)
    spec = json.loads(record_path(configuration).read_text(encoding="utf-8"),
                      object_pairs_hook=_json_object, parse_constant=_json_constant)
    _schema(spec, {"tasks"}, {"scales"}, "Configuration")
    tasks = spec["tasks"]
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks must be a nonempty list")
    directory = configuration.parent

    def source(value: Any, field: str) -> Path:
        return record_path(_path(value, directory, field))

    seen_tasks: set[str] = set()
    scores: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    task_checks: list[dict[str, Any]] = []
    for task in tasks:
        _schema(task, {"id", "treated", "references", "models"}, {"unit", "controls", "membership"}, "Task")
        task_id = label(task["id"], "Task ID")
        unit = label(task["unit"], "Unit ID") if "unit" in task else ""
        if task_id in seen_tasks:
            raise ValueError(f"Duplicate task ID {task_id!r}")
        seen_tasks.add(task_id)
        if ("controls" in task) != ("membership" in task):
            raise ValueError(f"{task_id}: provide controls and membership together")
        genes, treated = read_vector(source(task["treated"], "Treated path"))
        scales = None
        if "scales" in spec:
            _, scales = read_vector(source(spec["scales"], "Scales path"), genes)
        references = read_blocks(source(task["references"], "References path"), genes)
        reference_check = "supplied_means"
        if "controls" in task:
            cells = read_cells(source(task["controls"], "Controls path"), genes)
            verify_membership(cells, references, source(task["membership"], "Membership path"))
            reference_check = "verified_membership"
        models = task["models"]
        if not isinstance(models, list) or len(models) < 2:
            raise ValueError(f"{task_id}: provide at least two models")
        loaded = [_model(model, directory, genes, references, record_path) for model in models]
        if len({model["name"] for model in loaded}) != len(loaded):
            raise ValueError(f"{task_id}: duplicate model names")
        from .input_binding import verify_generation_record

        input_bindings = []
        for model in models:
            def selected(field: str, spec: dict[str, Any]) -> Path | None:
                return source(spec[field], field) if field in spec else None

            conditioning = model.get("conditioning", "fixed")
            binding = verify_generation_record(
                selected("generation_record", model),
                prediction_path=selected("prediction", model), model_name=model["name"],
                kind=model["kind"], conditioning=conditioning,
                representation=model.get("representation", "native"),
                controls_path=selected("controls", task),
                membership_path=selected("membership", task),
                references_path=selected("references", task),
                baseline_path=selected("baseline", model),
                expected_input_keys=sorted(references.values) if conditioning == "block" else None,
            )
            for artifact in binding["artifacts"].values():
                path = record_path(Path(artifact["path"]))
                if inputs[str(path)] != artifact["sha256"]:
                    raise ValueError(f"Generation artifact changed during the run: {path}")
            entry = dict(binding)
            if "model" in entry:
                entry["generation_model"] = entry.pop("model")
            entry["model"] = model["name"]
            input_bindings.append(entry)
        failures = 0
        maximum = 0.0
        pair_count = 0
        identities_applicable = 0
        identities_unavailable = 0
        for allocation, depth in references.groups:
            keys = [(allocation, depth, block) for block in BLOCKS]
            blocks = np.stack([references.values[key] for key in keys])
            predictions = [Prediction(model["name"], model["fixed"] if "fixed" in model else
                                      np.stack([model["blocks"][key] for key in keys]), model["kind"])
                           for model in loaded]
            result = score_references(treated, blocks, predictions, scales)
            metadata = dict(task=task_id, unit=unit, allocation=allocation, depth=depth)
            for record in result.rankings:
                scores.append(dict(**metadata, pattern=record["pattern"], model=record["model"],
                                   kind=record["kind"], MSE=record["mse"], rank=record["rank"]))
            pairs.extend(dict(**metadata, **record) for record in result.pairwise)
            failures += result.checks["identity_failures"]
            maximum = max(maximum, result.checks["max_identity_residual"])
            pair_count += len(result.pairwise)
            identities_applicable += sum(row.get("identity_applicable", True) for row in result.pairwise)
            identities_unavailable += sum(not row.get("identity_applicable", True) for row in result.pairwise)
        task_checks.append(dict(task=task_id, unit=unit, genes=len(genes),
                                reference_check=reference_check, allocation_depths=len(references.groups),
                                pair_count=pair_count, identity_failures=failures,
                                max_identity_residual=maximum,
                                identities_applicable=identities_applicable,
                                identities_unavailable=identities_unavailable,
                                input_bindings=input_bindings))
    try:
        software_version = version("reference-cell-benchmark-design")
    except PackageNotFoundError:
        software_version = "uninstalled"
    checks = dict(software_version=software_version,
                  inputs=[dict(path=path, sha256=digest) for path, digest in sorted(inputs.items())],
                  parameters=dict(loss="mean_squared_error", patterns=list(PATTERNS),
                                  scale="fixed_gene_divisors" if "scales" in spec else "unscaled",
                                  tie_tolerance=1e-12, identity_atol=1e-10, identity_rtol=1e-10),
                  identity_checks=task_checks)
    contents = {
        "scores.tsv": _tsv(["task", "unit", "allocation", "depth", "pattern", "model", "kind", "MSE", "rank"], scores),
        "pairs.tsv": _tsv(list(pairs[0]), pairs),
        "checks.json": (json.dumps(checks, indent=2, allow_nan=False) + "\n").encode(),
        "sensitivity.pdf": _figure(pairs),
    }
    for path, digest in inputs.items():
        if file_digest(Path(path)) != digest:
            raise ValueError(f"Scoring input changed during the run: {path}")
    _publish(args.output, contents)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "audit":
        from .audit_cli import main as audit_main

        return audit_main(arguments[1:])
    if arguments and arguments[0] == "vcc":
        from .vcc_cli import main as vcc_main

        return vcc_main(arguments[1:])
    parser = argparse.ArgumentParser(description="Refara: Reference allocation and role analysis. Audit reference changes and compare input-matched predictions.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("audit", help="Plan reference changes and execute invariance controls")
    commands.add_parser("vcc", help="Audit reference designs with the official VCC scorer")
    allocate = commands.add_parser("allocate", help="Make disjoint control blocks")
    allocate.add_argument("controls", type=Path)
    allocate.add_argument("--depth", type=int, action="append", required=True, help="Cells per stratum and block; repeat for several depths")
    allocate.add_argument("--blocks", type=int, choices=(2, 3), default=3)
    allocate.add_argument("--allocations", type=int, default=30)
    allocate.add_argument("--seed", type=int, default=0)
    allocate.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("score", help="Score predictions and compare shared/separated references")
    score.add_argument("config", type=Path)
    score.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run", help="Run a declared supplied-prediction reference protocol")
    run.add_argument("plan", type=Path)
    run.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser("prepare", help="Prepare a protocol from explicitly mapped AnnData or TSV inputs")
    prepare.add_argument("manifest", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    design = commands.add_parser("design", help="Use pilot results to assess model orderings and control budgets")
    design.add_argument("manifest", type=Path)
    design.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("check-design", help="Check saved organizer decisions against further reference realizations")
    check.add_argument("design", type=Path, help="design.json written by design")
    check.add_argument("assessment_scores", type=Path)
    check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(arguments)
    try:
        if args.command == "design":
            from .organizer import design as assess_design
            assess_design(args.manifest, args.output)
        elif args.command == "check-design":
            from .organizer import check_design
            check_design(args.design, args.assessment_scores, args.output)
        elif args.command == "prepare":
            from .prepare import prepare_plan
            prepare_plan(args.manifest, args.output)
        elif args.command == "run":
            from .protocol import run_plan
            run_plan(args.plan, args.output)
        else:
            _allocate(args) if args.command == "allocate" else _score(args)
    except (ValueError, OSError, FloatingPointError) as error:
        parser.error(str(error))
    return 0
