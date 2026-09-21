"""Summarize complete reference-role interventions and matched workflows.

Finite score contrasts describe the saved factorial assignments. They are not
population causal effects, independent replicates or cross-target improvements.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np

HERE = Path(__file__).resolve().parents[1]
PROTOCOL_SHA256 = "e65613d2f393663cb324535fac93efbc01ebc818865c6f063626a61d02461689"
ROLES = ("Cobs", "Cpred", "Cmodel")
PATTERNS = ("S", "M", "P", "O", "D")
GAPS = (0.0, 0.005, 0.01, 0.02)
TIE_TOLERANCE = 1e-12
ORDERED_CHANGES = tuple((old, new) for old in range(3) for new in range(3) if old != new)
GAP_LABELS = ("0", "0p005", "0p01", "0p02")
GAP_COLUMNS = tuple(f"gap_{label}_{count}_count" for label in GAP_LABELS for count in ("positive", "negative", "within"))
SUMMARY_COLUMNS = ("task_count", "allocation_count", "directed_contrasts_per_allocation", "total_task_mean_contrasts",
                   "mean_abs_task_score_change", "minimum_task_mean_change", "maximum_task_mean_change", "mean_abs_task_mean_change") + GAP_COLUMNS
PATTERN_COLUMNS = ("view", "pattern", "depth", "control_cells_per_block", "controls_available", "model", "model_kind", "metric", "metric_direction",
                   "task_count", "allocation_count", "tuples_per_allocation", "mean_score", "undefined_values")
CONTRAST_COLUMNS = ("role", "depth", "model", "model_kind", "metric", "metric_direction") + SUMMARY_COLUMNS
INTERACTION_COLUMNS = ("role_a", "role_b", "depth", "model", "model_kind", "metric", "metric_direction") + SUMMARY_COLUMNS
WORKFLOW_COLUMNS = ("view", "depth", "model", "model_kind", "metric", "metric_direction", "task_count", "allocation_count", "realizations_per_allocation", "repeated_realizations",
                    "A_allocation", "A_Cobs", "A_Cpred", "A_Cmodel", "A_anchor_task_mean", "B_repeated_task_mean", "B_minimum_task_mean", "B_maximum_task_mean",
                    "C_same_primary_mean", "C_minus_B_same_primary", "C_full_role_assignments_per_allocation", "all_task_support")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_tsv(path: Path, columns: tuple[str, ...], rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            require(set(row) == set(columns), f"Columns differ for {path.name}")
            writer.writerow({key: format(float(value), ".16g") if isinstance(value, (float, np.floating)) else value for key, value in row.items()})
            count += 1
    return count


def pattern_indices(tuples: list[tuple[int, int, int]]) -> dict[str, list[int]]:
    result = {pattern: [] for pattern in PATTERNS}
    for ri, (o, p, i) in enumerate(tuples):
        pattern = "S" if o == p == i else "M" if o == p else "P" if o == i else "O" if p == i else "D"
        result[pattern].append(ri)
    require([len(result[p]) for p in PATTERNS] == [3, 6, 6, 6, 6], "Pattern balance differs")
    return result


def role_edges(role: int, lookup: dict[tuple[int, int, int], int]) -> list[tuple[int, int]]:
    other = [index for index in range(3) if index != role]
    edges = []
    for fixed in itertools.product(range(3), repeat=2):
        for old, new in ORDERED_CHANGES:
            before = [0, 0, 0]
            before[other[0]], before[other[1]], before[role] = fixed[0], fixed[1], old
            after = before.copy()
            after[role] = new
            edges.append((lookup[tuple(before)], lookup[tuple(after)]))
    require(len(edges) == 54, "Role edge count differs")
    return edges


def interaction_cells(role_a: int, role_b: int, lookup: dict[tuple[int, int, int], int]) -> list[tuple[int, int, int, int]]:
    other = next(index for index in range(3) if index not in (role_a, role_b))
    cells = []
    for fixed in range(3):
        for a0, a1 in ORDERED_CHANGES:
            for b0, b1 in ORDERED_CHANGES:
                indices = []
                # Index order is 11, 10, 01, 00; coefficients +1,-1,-1,+1.
                for a, b in ((a1, b1), (a1, b0), (a0, b1), (a0, b0)):
                    values = [0, 0, 0]
                    values[role_a], values[role_b], values[other] = a, b, fixed
                    indices.append(lookup[tuple(values)])
                cells.append(tuple(indices))
    require(len(cells) == 108, "Interaction count differs")
    return cells


def summarize_changes(changes: np.ndarray) -> dict[str, np.ndarray | int]:
    """Input [task, allocation, directed contrast, model, metric]."""
    require(changes.ndim == 5 and bool(np.isfinite(changes).all()), "Incomplete score-change support")
    task_mean = changes.mean(axis=0)
    result: dict[str, np.ndarray | int] = {
        "task_count": changes.shape[0], "allocation_count": changes.shape[1],
        "directed_contrasts_per_allocation": changes.shape[2],
        "total_task_mean_contrasts": changes.shape[1] * changes.shape[2],
        "mean_abs_task_score_change": np.abs(changes).mean(axis=(0, 1, 2)),
        "minimum_task_mean_change": task_mean.min(axis=(0, 1)),
        "maximum_task_mean_change": task_mean.max(axis=(0, 1)),
        "mean_abs_task_mean_change": np.abs(task_mean).mean(axis=(0, 1)),
    }
    for gap, label in zip(GAPS, GAP_LABELS):
        positive = (task_mean > gap + TIE_TOLERANCE).sum(axis=(0, 1))
        negative = (task_mean < -gap - TIE_TOLERANCE).sum(axis=(0, 1))
        within = changes.shape[1] * changes.shape[2] - positive - negative
        result[f"gap_{label}_positive_count"] = positive
        result[f"gap_{label}_negative_count"] = negative
        result[f"gap_{label}_within_count"] = within
        require(bool(np.array_equal(positive, negative)), "Ordered contrast sign counts lost symmetry")
    return result


def scalar_fields(summary: dict[str, np.ndarray | int], model: int, metric: int) -> dict[str, Any]:
    result = {}
    for key, value in summary.items():
        if isinstance(value, np.ndarray):
            result[key] = int(value[model, metric]) if np.issubdtype(value.dtype, np.integer) else float(value[model, metric])
        else:
            result[key] = value
    return result


def run(outdir: Path | str) -> dict[str, Any]:
    started = time.monotonic()
    outdir = Path(outdir)
    require(sha256(HERE / "PROTOCOL.json") == PROTOCOL_SHA256, "Protocol changed")
    receipt_path = HERE / "empirical/RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    require(receipt["study_protocol_sha256"] == PROTOCOL_SHA256, "Cube protocol differs")
    cube_path = HERE / "empirical/task_scores.npz"
    cube_sha = sha256(cube_path)
    require(cube_sha == receipt["outputs"]["task_scores.npz"]["sha256"], "Cube bytes changed")
    require(sha256(HERE / "code/role_metrics.py") == receipt["engine_sha256"], "Empirical engine changed")
    outdir.mkdir(parents=True, exist_ok=True)
    binding = {"schema_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": PROTOCOL_SHA256,
               "summary_code_sha256": sha256(Path(__file__)), "cube_sha256": cube_sha, "empirical_receipt_sha256": sha256(receipt_path)}
    write_json(outdir / "SUMMARY_BINDING.json", binding)
    with np.load(cube_path, allow_pickle=False) as archive:
        direct = archive["scores_direct_state"]
        role = archive["scores_role_effect"]
        tasks = archive["tasks"].tolist()
        depths = archive["depths"].tolist()
        allocations = archive["allocations"].tolist()
        models = archive["models"].tolist()
        kinds = archive["model_kinds"].tolist()
        metrics = archive["metrics"].tolist()
        directions = archive["metric_directions"].tolist()
        tuples = [tuple(map(int, values)) for values in archive["role_tuples"]]
        control_cells = archive["controls_per_block"].tolist()
        controls_available = archive["controls_available"].tolist()
    require(direct.shape == (55, 6, 30, 3, 6, 4) and role.shape == (55, 6, 30, 27, 6, 4), "Cube shapes differ")
    require(bool(np.isfinite(direct).all() and np.isfinite(role).all()), "Complete-table summary requires all scores defined")
    require(tuples == list(itertools.product(range(3), repeat=3)), "Tuple order differs")
    lookup = {values: index for index, values in enumerate(tuples)}
    patterns = pattern_indices(tuples)

    def common(di: int, mi: int, ki: int) -> dict[str, Any]:
        return dict(depth=depths[di], model=models[mi], model_kind=kinds[mi], metric=metrics[ki], metric_direction=directions[ki])

    pattern_rows = []
    for pattern in ("direct_state",) + PATTERNS:
        cube = direct if pattern == "direct_state" else role[:, :, :, patterns[pattern]]
        means = cube.mean(axis=2).mean(axis=2).mean(axis=0)
        for di, mi, ki in itertools.product(range(6), range(6), range(4)):
            pattern_rows.append(dict(view="direct_state" if pattern == "direct_state" else "reference_effect", pattern=pattern,
                                     control_cells_per_block=control_cells[di], controls_available=controls_available[di],
                                     task_count=len(tasks), allocation_count=len(allocations), tuples_per_allocation=cube.shape[3],
                                     mean_score=float(means[di, mi, ki]), undefined_values=0, **common(di, mi, ki)))
    counts = {"pattern_means.tsv": write_tsv(outdir / "pattern_means.tsv", PATTERN_COLUMNS, pattern_rows)}

    contrast_rows = []
    edge_metadata = {}
    for r, role_name in enumerate(ROLES):
        edges = np.asarray(role_edges(r, lookup))
        edge_metadata[role_name] = edges.tolist()
        for di in range(6):
            # take preserves [task, allocation, edge, model, metric].
            changes = np.take(role[:, di], edges[:, 1], axis=2) - np.take(role[:, di], edges[:, 0], axis=2)
            summary = summarize_changes(changes)
            for mi, ki in itertools.product(range(6), range(4)):
                contrast_rows.append(dict(role=role_name, **common(di, mi, ki), **scalar_fields(summary, mi, ki)))
    counts["role_contrasts.tsv"] = write_tsv(outdir / "role_contrasts.tsv", CONTRAST_COLUMNS, contrast_rows)

    interaction_rows = []
    interaction_metadata = {}
    for a, b in itertools.combinations(range(3), 2):
        cells = np.asarray(interaction_cells(a, b, lookup))
        interaction_metadata[f"{ROLES[a]}:{ROLES[b]}"] = cells.tolist()
        for di in range(6):
            cube = role[:, di]
            changes = (np.take(cube, cells[:, 0], axis=2) - np.take(cube, cells[:, 1], axis=2)
                       - np.take(cube, cells[:, 2], axis=2) + np.take(cube, cells[:, 3], axis=2))
            summary = summarize_changes(changes)
            for mi, ki in itertools.product(range(6), range(4)):
                interaction_rows.append(dict(role_a=ROLES[a], role_b=ROLES[b], **common(di, mi, ki), **scalar_fields(summary, mi, ki)))
    counts["role_interactions.tsv"] = write_tsv(outdir / "role_interactions.tsv", INTERACTION_COLUMNS, interaction_rows)

    workflow_rows = []
    o_pairs = [(i, o) for i in range(3) for o in range(3) if i != o]
    o_indices = [lookup[(o, i, i)] for i, o in o_pairs]
    for view, cube, anchor, anchors in (("direct_state", direct, 0, ("unused", "unused", 0)),
                                       ("reference_effect_O", role[:, :, :, o_indices], o_indices.index(lookup[(1, 0, 0)]), (1, 0, 0))):
        task_means = cube.mean(axis=0)
        for di, mi, ki in itertools.product(range(6), range(6), range(4)):
            values = task_means[di, :, :, mi, ki]
            repeated_mean = float(values.mean())
            workflow_rows.append(dict(view=view, **common(di, mi, ki), task_count=len(tasks), allocation_count=len(allocations),
                                      realizations_per_allocation=cube.shape[3], repeated_realizations=len(allocations) * cube.shape[3],
                                      A_allocation=0, A_Cobs=anchors[0], A_Cpred=anchors[1], A_Cmodel=anchors[2],
                                      A_anchor_task_mean=float(values[0, anchor]), B_repeated_task_mean=repeated_mean,
                                      B_minimum_task_mean=float(values.min()), B_maximum_task_mean=float(values.max()),
                                      C_same_primary_mean=repeated_mean, C_minus_B_same_primary=0.0,
                                      C_full_role_assignments_per_allocation=27, all_task_support=True))
    counts["workflow_comparison.tsv"] = write_tsv(outdir / "workflow_comparison.tsv", WORKFLOW_COLUMNS, workflow_rows)
    require(counts == {"pattern_means.tsv": 864, "role_contrasts.tsv": 432, "role_interactions.tsv": 432, "workflow_comparison.tsv": 288}, "Summary row counts differ")
    metadata = {**binding, "status": "PASS_COMPLETE_ROLE_SUMMARIES", "table_rows": counts,
                "axes": {"tasks": tasks, "depths": depths, "allocations": allocations, "models": models, "model_kinds": kinds, "metrics": metrics, "metric_directions": directions, "role_tuples": tuples},
                "columns": {"pattern_means.tsv": PATTERN_COLUMNS, "role_contrasts.tsv": CONTRAST_COLUMNS, "role_interactions.tsv": INTERACTION_COLUMNS, "workflow_comparison.tsv": WORKFLOW_COLUMNS},
                "pattern_tuple_indices": patterns, "role_edge_tuple_indices_before_after": edge_metadata,
                "interaction_tuple_indices_11_10_01_00": interaction_metadata,
                "contrast_formula": "S(new)-S(old), other two roles fixed; 9 fixed settings times 6 ordered changes=54 per allocation.",
                "interaction_formula": "S(a1,b1)-S(a1,b0)-S(a0,b1)+S(a0,b0), third role fixed; 3 fixed settings times 6 ordered changes times 6 ordered changes=108 per allocation.",
                "meaning": "Finite factorial raw-score contrasts. Positive means higher raw metric value, not uniformly better. Interactions are score-function nonadditivity, not population causal interactions.",
                "aggregation": "Equal allocation and role-assignment weights within each task, then equal weight across all 55 tasks. mean_abs_task_score_change takes absolute value before task averaging. Other change summaries first average 55 paired task changes, then summarize all allocation/contrast realizations. Depths and metrics stay separate.",
                "target_scope": "Cobs changes the realized observed effect; Cpred changes the defined effect of state output. Such changes are declared diagnostic interventions, not corrections to one unchanged target. A Cmodel-only change holds Cpred fixed to isolate input response; jointly updating a paired Cpred=Cmodel changes the derived effect and is a different query. Fixed native effects ignore Cpred and Cmodel.",
                "directed_symmetry": "Every ordered edge has its reverse. Positive and negative counts are structurally equal and symmetric ranges follow by construction; they are not model-ranking reversals or a signed systematic bias. Interpret change magnitudes and zero/nonzero interactions. Reused tasks, controls and reversed contrasts are not independent samples; no significance test or causal claim is made.",
                "gaps": GAPS, "tie_tolerance": TIE_TOLERANCE,
                "gap_count_rule": "positive: delta>gap+1e-12; negative: delta<-(gap+1e-12); within: all remaining. Metric-unit descriptions only, not shared biological thresholds.",
                "workflow_comparison": {"A": "Canonical allocation0/input0; O anchor (Cobs=1,Cpred=0,Cmodel=0). A is a single-realization workflow, not the full capability of Systema.",
                                        "B": "Complete correct repeats: 90 direct-state or 180 O realizations per depth, each with 55 task scores.",
                                        "C_same_primary": "Exactly the same canonical scores as B. Values coincide by construction and by unchanged metric inputs.",
                                        "C_full_roles": "Additional queries across all 27 assignments. Different assignments may define different realized quantities; no pooled metric-superiority comparison.",
                                        "matched_resources": "A, B and C have the same cached inputs, role metadata, available controls, fixed gene scales, training metric centre and candidate targets. A chooses one realization; B repeats its declared view; C also issues role interventions. Role enumeration reuses existing predictions and does not consume additional controls or model fits.",
                                        "manual_parity": "A repeated Systema evaluation with explicit manual role reasoning can reconstruct the same diagnostics. An unreported diagnostic is not an erroneous answer.",
                                        "unmeasured": "No independent organizer errors, human diagnostic accuracy, user time or resource savings are estimated."},
                "undefined_values": 0}
    write_json(outdir / "SUMMARY_METADATA.json", metadata)
    final_receipt = {**binding, "status": metadata["status"], "finished_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - started,
                     "table_rows": counts, "outputs": {name: {"sha256": sha256(outdir / name), "bytes": (outdir / name).stat().st_size}
                                                     for name in sorted(list(counts) + ["SUMMARY_METADATA.json", "SUMMARY_BINDING.json"])}}
    write_json(outdir / "SUMMARY_RECEIPT.json", final_receipt)
    print(json.dumps({"status": metadata["status"], "table_rows": counts, "elapsed_seconds": final_receipt["elapsed_seconds"]}), flush=True)
    return final_receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    run(args.out_dir)


if __name__ == "__main__":
    main()
