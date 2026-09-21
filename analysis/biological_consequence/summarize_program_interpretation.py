#!/usr/bin/env python3
"""Summarize program-direction and priority changes in frozen readouts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np


THRESHOLDS = (0.0, 0.01, 0.025, 0.05)
TOP_K = (1, 3, 5, 10)
MARGINS = (0.0, 0.01, 0.025, 0.05)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict]) -> None:
    require(bool(rows), f"Refusing to write empty table: {path.name}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def active_sign(values: np.ndarray, threshold: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.where(values > threshold, 1, np.where(values < -threshold, -1, 0))


def stable_top(values: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-np.abs(values), axis=-1, kind="stable")[..., :k]


def changed_top_sets(s_values: np.ndarray, d_values: np.ndarray, k: int) -> np.ndarray:
    s_top, d_top = stable_top(s_values, k), stable_top(d_values, k)
    return np.asarray([set(a.tolist()) != set(b.tolist()) for a, b in zip(s_top, d_top)])


def winner_and_margin(values: np.ndarray, absolute: bool) -> tuple[np.ndarray, np.ndarray]:
    ranked = np.abs(values) if absolute else values
    order = np.argsort(-ranked, axis=1, kind="stable")
    winner = order[:, 0]
    margin = ranked[np.arange(len(ranked)), order[:, 0]] - ranked[np.arange(len(ranked)), order[:, 1]]
    return winner, margin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program-scores", required=True, type=Path)
    parser.add_argument("--program-readouts", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "Use a new output directory")

    source_rows = read_tsv(args.program_scores)
    require(len(source_rows) == 40 * 37, "Expected 1,480 task-program rows")
    task_meta: dict[int, dict[str, str]] = {}
    for row in source_rows:
        task = int(row["task_row"])
        meta = {key: row[key] for key in ("root_id", "task_id", "cell_type")}
        require(task not in task_meta or task_meta[task] == meta, "Task metadata disagree")
        task_meta[task] = meta
    require(sorted(task_meta) == list(range(40)), "Task rows must be 0 through 39")

    with np.load(args.program_readouts, allow_pickle=False) as archive:
        scores = np.asarray(archive["scores"], dtype=np.float64)
        program_ids = archive["program_ids"].astype(str)
        score_axis = archive["score_axis"].astype(str)
        label_ids = np.asarray(archive["label_ids"], dtype=np.int64)
        block_pairs = np.asarray(archive["block_pairs"], dtype=np.int64)
        root_index = np.asarray(archive["root_index"], dtype=np.int64)
        task_index = np.asarray(archive["task_index"], dtype=np.int64)
    require(scores.shape == (10, 6, 40, 3, 37), "Unexpected score-array shape")
    require(np.isfinite(scores).all(), "Program scores must be finite")
    require(np.array_equal(label_ids, np.arange(0, 1000, 100)), "Unexpected labels")
    require(block_pairs.tolist() == [[1, 2], [1, 3], [2, 1], [2, 3], [3, 1], [3, 2]],
            "Unexpected block-pair order")
    require(np.array_equal(root_index, np.repeat(np.arange(8), 5)), "Unexpected donor order")
    require(np.array_equal(task_index, np.tile(np.arange(5), 8)), "Unexpected task order")
    require(len(program_ids) == 37 and len(set(program_ids.tolist())) == 37,
            "Expected 37 unique programs")
    require(score_axis.tolist() == ["observed", "S_selected", "D_selected"],
            "Unexpected score-axis order")
    for task in range(40):
        task_rows = source_rows[task * 37:(task + 1) * 37]
        require([int(row["task_row"]) for row in task_rows] == [task] * 37,
                "program_scores.tsv is not task-major")
        require([row["program_id"] for row in task_rows] == program_ids.tolist(),
                "Program order differs between TSV and NPZ")
        require(int(root_index[task]) == task // 5 and int(task_index[task]) == task % 5,
                "NPZ task axes disagree with task rows")
    cell_type_order = [task_meta[task]["cell_type"] for task in range(5)]
    task_id_order = [task_meta[task]["task_id"] for task in range(5)]
    require(len(set(cell_type_order)) == 5 and len(set(task_id_order)) == 5,
            "The first donor must contain five distinct tasks")
    for root in range(8):
        require([task_meta[root * 5 + task]["cell_type"] for task in range(5)] == cell_type_order,
                "Cell-type order differs between donors")
        require([task_meta[root * 5 + task]["task_id"] for task in range(5)] == task_id_order,
                "Task order differs between donors")

    primary = scores[0, 0]
    source_matrix = np.asarray([[float(row[name]) for name in ("observed", "S_selected", "D_selected")]
                                for row in source_rows], dtype=np.float64).reshape(40, 37, 3)
    require(np.allclose(primary.transpose(0, 2, 1), source_matrix, rtol=0, atol=1e-14),
            "Primary NPZ values disagree with program_scores.tsv")

    # Predictions depend on the inference block, not the observation block. Collapse
    # the exact duplicates to 30 label-by-inference assignments.
    assignments: list[tuple[int, int, np.ndarray]] = []
    for label_position, label in enumerate(label_ids):
        for inference_block in (1, 2, 3):
            positions = np.flatnonzero(block_pairs[:, 0] == inference_block)
            require(len(positions) == 2, "Each inference block needs two observation blocks")
            first, second = scores[label_position, positions[0]], scores[label_position, positions[1]]
            require(np.array_equal(first[:, 1:], second[:, 1:]),
                    "Selected-model predictions changed with observation block")
            assignments.append((int(label), inference_block, first))
    require(len(assignments) == 30, "Expected 30 unique inference assignments")

    direction_rows: list[dict] = []
    priority_rows: list[dict] = []
    priority_margin_rows: list[dict] = []
    for label, inference_block, values in assignments:
        s_values, d_values = values[:, 1], values[:, 2]
        for threshold in THRESHOLDS:
            s_sign, d_sign = active_sign(s_values, threshold), active_sign(d_values, threshold)
            changed = s_sign * d_sign == -1
            affected_tasks = changed.any(axis=1)
            affected_donors = np.asarray([affected_tasks[root_index == root].any() for root in range(8)])
            direction_rows.append({
                "label_id": label,
                "inference_block": inference_block,
                "uses_original_primary_prediction_reference": label == 0 and inference_block == 1,
                "minimum_absolute_score_in_both_predictions": threshold,
                "task_programs_with_opposite_active_directions": int(changed.sum()),
                "tasks_with_at_least_one_opposite_active_direction": int(affected_tasks.sum()),
                "donors_with_at_least_one_opposite_active_direction": int(affected_donors.sum()),
                "task_program_denominator": 1480,
                "task_denominator": 40,
                "donor_denominator": 8,
            })
        for k in TOP_K:
            changed = changed_top_sets(s_values, d_values, k)
            overlaps = [len(set(a.tolist()) & set(b.tolist()))
                        for a, b in zip(stable_top(s_values, k), stable_top(d_values, k))]
            priority_rows.append({
                "label_id": label,
                "inference_block": inference_block,
                "uses_original_primary_prediction_reference": label == 0 and inference_block == 1,
                "top_k_by_absolute_predicted_score": k,
                "tasks_with_different_S_D_top_k_sets": int(changed.sum()),
                "mean_S_D_top_k_intersection": float(np.mean(overlaps)),
                "task_denominator": 40,
            })
            s_sorted = np.sort(np.abs(s_values), axis=1)[:, ::-1]
            d_sorted = np.sort(np.abs(d_values), axis=1)[:, ::-1]
            s_cutoff_margin = s_sorted[:, k - 1] - s_sorted[:, k]
            d_cutoff_margin = d_sorted[:, k - 1] - d_sorted[:, k]
            for margin in MARGINS:
                eligible = (s_cutoff_margin > margin) & (d_cutoff_margin > margin)
                priority_margin_rows.append({
                    "label_id": label,
                    "inference_block": inference_block,
                    "uses_original_primary_prediction_reference": label == 0 and inference_block == 1,
                    "top_k_by_absolute_predicted_score": k,
                    "minimum_kth_to_next_score_margin_in_both_predictions": margin,
                    "tasks_meeting_both_cutoff_margins": int(eligible.sum()),
                    "eligible_tasks_with_different_S_D_top_k_sets": int(np.sum(eligible & changed)),
                    "task_denominator": 40,
                })

    primary_s, primary_d, primary_o = primary[:, 1], primary[:, 2], primary[:, 0]
    primary_task_rows: list[dict] = []
    for task in range(40):
        row: dict = {"task_row": task, **task_meta[task], "program_count": 37}
        for threshold in THRESHOLDS:
            changed = active_sign(primary_s[task], threshold) * active_sign(primary_d[task], threshold) == -1
            tag = str(threshold).replace("0.", "p").replace(".", "p")
            row[f"opposite_active_direction_count_at_t_{tag}"] = int(changed.sum())
        for k in TOP_K:
            s_set, d_set = set(stable_top(primary_s[task], k).tolist()), set(stable_top(primary_d[task], k).tolist())
            row[f"shared_programs_in_top_{k}"] = len(s_set & d_set)
            row[f"S_D_top_{k}_sets_differ"] = s_set != d_set
        row["S_largest_absolute_score_program"] = program_ids[stable_top(primary_s[task], 1)[0]]
        row["D_largest_absolute_score_program"] = program_ids[stable_top(primary_d[task], 1)[0]]
        row["observed_largest_absolute_score_program"] = program_ids[stable_top(primary_o[task], 1)[0]]
        primary_task_rows.append(row)

    primary_call_rows: list[dict] = []
    for task in range(40):
        for program in range(37):
            observed = float(primary_o[task, program])
            s_value = float(primary_s[task, program])
            d_value = float(primary_d[task, program])
            row = {
                "task_row": task,
                **task_meta[task],
                "program_id": program_ids[program],
                "observed": observed,
                "S_selected": s_value,
                "D_selected": d_value,
                "observed_absolute_rank": int(1 + np.sum(np.abs(primary_o[task]) > abs(observed))),
                "S_D_opposite_exact_directions": np.sign(s_value) != np.sign(d_value),
                "observed_direction_match_when_S_D_opposed":
                    "S" if np.sign(observed) == np.sign(s_value) != np.sign(d_value)
                    else "D" if np.sign(observed) == np.sign(d_value) != np.sign(s_value)
                    else "neither" if np.sign(s_value) != np.sign(d_value)
                    else "not_applicable_no_S_D_opposition",
            }
            for threshold in THRESHOLDS:
                tag = str(threshold).replace("0.", "p").replace(".", "p")
                row[f"S_D_opposite_active_directions_at_t_{tag}"] = (
                    active_sign(np.asarray(s_value), threshold) *
                    active_sign(np.asarray(d_value), threshold) == -1).item()
            primary_call_rows.append(row)

    observed_alignment_rows: list[dict] = []
    for threshold in THRESHOLDS:
        s_sign = active_sign(primary_s, threshold)
        d_sign = active_sign(primary_d, threshold)
        observed_sign = active_sign(primary_o, threshold)
        opposed = s_sign * d_sign == -1
        evaluable = opposed & (observed_sign != 0)
        s_matches = evaluable & (observed_sign == s_sign)
        d_matches = evaluable & (observed_sign == d_sign)
        task_s = s_matches.sum(axis=1)
        task_d = d_matches.sum(axis=1)
        affected = evaluable.any(axis=1)
        observed_alignment_rows.append({
            "minimum_absolute_score_in_S_D_and_observed": threshold,
            "evaluable_opposite_direction_task_programs": int(evaluable.sum()),
            "observed_direction_matches_S_calls": int(s_matches.sum()),
            "observed_direction_matches_D_calls": int(d_matches.sum()),
            "tasks_with_at_least_one_evaluable_call": int(affected.sum()),
            "affected_tasks_with_more_S_matches": int(np.sum(affected & (task_s > task_d))),
            "affected_tasks_with_more_D_matches": int(np.sum(affected & (task_d > task_s))),
            "affected_tasks_with_equal_match_counts": int(np.sum(affected & (task_s == task_d))),
            "task_program_denominator": 1480,
            "task_denominator": 40,
        })

    # Cell-type attribution is a supporting sensitivity analysis. Each row below is
    # a donor-program group containing the same five cell types.
    attribution_rows: list[dict] = []
    attribution_summary_rows: list[dict] = []
    for mode, absolute in (("absolute_response", True), ("signed_response", False)):
        s_groups = primary_s.reshape(8, 5, 37).transpose(0, 2, 1).reshape(296, 5)
        d_groups = primary_d.reshape(8, 5, 37).transpose(0, 2, 1).reshape(296, 5)
        o_groups = primary_o.reshape(8, 5, 37).transpose(0, 2, 1).reshape(296, 5)
        s_winner, s_margin = winner_and_margin(s_groups, absolute)
        d_winner, d_margin = winner_and_margin(d_groups, absolute)
        o_winner, o_margin = winner_and_margin(o_groups, absolute)
        for group in range(296):
            root, program = divmod(group, 37)
            attribution_rows.append({
                "mode": mode,
                "root_id": task_meta[root * 5]["root_id"],
                "program_id": program_ids[program],
                "S_argmax_cell_type": task_meta[root * 5 + int(s_winner[group])]["cell_type"],
                "D_argmax_cell_type": task_meta[root * 5 + int(d_winner[group])]["cell_type"],
                "observed_argmax_cell_type": task_meta[root * 5 + int(o_winner[group])]["cell_type"],
                "S_argmax_to_runner_up_margin": float(s_margin[group]),
                "D_argmax_to_runner_up_margin": float(d_margin[group]),
                "observed_argmax_to_runner_up_margin": float(o_margin[group]),
                "S_D_argmax_cell_types_differ": bool(s_winner[group] != d_winner[group]),
            })
        for margin in MARGINS:
            changed = (s_winner != d_winner) & (s_margin > margin) & (d_margin > margin)
            observed_resolved = changed & (o_margin > margin)
            attribution_summary_rows.append({
                "mode": mode,
                "minimum_argmax_to_runner_up_margin_in_both_predictions": margin,
                "donor_program_groups_with_different_S_D_argmax": int(changed.sum()),
                "donors_with_at_least_one_change": int(len(set(np.flatnonzero(changed) // 37))),
                "programs_with_at_least_one_change": int(len(set(np.flatnonzero(changed) % 37))),
                "observed_argmax_matches_S_without_observed_margin_filter": int(np.sum(changed & (o_winner == s_winner))),
                "observed_argmax_matches_D_without_observed_margin_filter": int(np.sum(changed & (o_winner == d_winner))),
                "observed_argmax_matches_neither_without_observed_margin_filter": int(np.sum(changed & (o_winner != s_winner) & (o_winner != d_winner))),
                "changed_groups_with_observed_margin_above_threshold": int(observed_resolved.sum()),
                "observed_argmax_matches_S_with_same_margin_filter": int(np.sum(observed_resolved & (o_winner == s_winner))),
                "observed_argmax_matches_D_with_same_margin_filter": int(np.sum(observed_resolved & (o_winner == d_winner))),
                "observed_argmax_matches_neither_with_same_margin_filter": int(np.sum(observed_resolved & (o_winner != s_winner) & (o_winner != d_winner))),
                "donor_program_denominator": 296,
            })

    args.output_dir.mkdir(parents=True)
    tables = {
        "direction_threshold_sensitivity.tsv": direction_rows,
        "program_priority_sensitivity.tsv": priority_rows,
        "program_priority_margin_sensitivity.tsv": priority_margin_rows,
        "primary_observed_direction_alignment.tsv": observed_alignment_rows,
        "primary_task_interpretation.tsv": primary_task_rows,
        "primary_program_calls.tsv": primary_call_rows,
        "cell_type_attribution.tsv": attribution_rows,
        "cell_type_attribution_sensitivity.tsv": attribution_summary_rows,
    }
    for name, rows in tables.items():
        write_tsv(args.output_dir / name, rows)

    primary_direction = {
        str(row["minimum_absolute_score_in_both_predictions"]): row for row in direction_rows
        if row["uses_original_primary_prediction_reference"]
    }
    primary_priority = {
        str(row["top_k_by_absolute_predicted_score"]): row for row in priority_rows
        if row["uses_original_primary_prediction_reference"]
    }
    primary_priority_margin = [
        row for row in priority_margin_rows
        if row["uses_original_primary_prediction_reference"]
    ]
    all_s = np.stack([values[:, 1] for _, _, values in assignments])
    all_d = np.stack([values[:, 2] for _, _, values in assignments])
    s_leaders = stable_top(all_s.reshape(-1, 37), 1).ravel()
    d_leaders = stable_top(all_d.reshape(-1, 37), 1).ravel()
    same_leaders = s_leaders == d_leaders
    common_program = None
    if same_leaders.all() and len(set(s_leaders.tolist())) == 1:
        common_program = program_ids[int(s_leaders[0])]
    s_frequencies = {program_ids[index]: int(np.sum(s_leaders == index))
                     for index in sorted(set(s_leaders.tolist()))}
    d_frequencies = {program_ids[index]: int(np.sum(d_leaders == index))
                     for index in sorted(set(d_leaders.tolist()))}
    observed_primary_leaders = stable_top(primary_o, 1).ravel()
    summary = {
        "status": "PASS_POST_HOC_DESCRIPTIVE_SUMMARY",
        "scope": "Frozen eight-donor program readouts; no inferential independence is assigned to tasks or overlapping programs.",
        "thresholds": list(THRESHOLDS),
        "top_k_values": list(TOP_K),
        "primary_direction": primary_direction,
        "primary_priority": primary_priority,
        "primary_observed_direction_alignment": {
            str(row["minimum_absolute_score_in_S_D_and_observed"]): row
            for row in observed_alignment_rows
        },
        "primary_priority_cutoff_margin_sensitivity": primary_priority_margin,
        "assignment_ranges": {
            "direction_tasks": {
                str(threshold): {
                    "minimum": min(row["tasks_with_at_least_one_opposite_active_direction"] for row in direction_rows
                                   if row["minimum_absolute_score_in_both_predictions"] == threshold),
                    "maximum": max(row["tasks_with_at_least_one_opposite_active_direction"] for row in direction_rows
                                   if row["minimum_absolute_score_in_both_predictions"] == threshold),
                } for threshold in THRESHOLDS
            },
            "priority_tasks": {
                str(k): {
                    "minimum": min(row["tasks_with_different_S_D_top_k_sets"] for row in priority_rows
                                   if row["top_k_by_absolute_predicted_score"] == k),
                    "maximum": max(row["tasks_with_different_S_D_top_k_sets"] for row in priority_rows
                                   if row["top_k_by_absolute_predicted_score"] == k),
                } for k in TOP_K
            },
        },
        "prediction_reference_setting_count": 30,
        "tasks_per_prediction_reference_setting": 40,
        "S_D_same_largest_absolute_score_program_task_settings": int(same_leaders.sum()),
        "largest_absolute_score_program_common_to_all_S_D_task_settings": common_program,
        "S_largest_absolute_score_program_frequencies": s_frequencies,
        "D_largest_absolute_score_program_frequencies": d_frequencies,
        "observed_primary_largest_absolute_score_program_frequencies": {
            program_ids[index]: int(np.sum(observed_primary_leaders == index))
            for index in sorted(set(observed_primary_leaders.tolist()))
        },
        "source_sha256": {
            "program_scores.tsv": digest(args.program_scores),
            "program_readouts.npz": digest(args.program_readouts),
            "summarize_program_interpretation.py": digest(Path(__file__)),
        },
    }
    write_json(args.output_dir / "interpretation_summary.json", summary)
    write_json(args.output_dir / "INTERPRETATION_RECEIPT.json", {
        "status": "PASS",
        "analysis_timing": "Post hoc descriptive decomposition of previously frozen program readouts.",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "outputs_sha256": {name: digest(args.output_dir / name)
                           for name in sorted([*tables, "interpretation_summary.json"])},
        "source_sha256": summary["source_sha256"],
    })


if __name__ == "__main__":
    main()
