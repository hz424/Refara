"""Compare every published summary row with independent role-cube arithmetic."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import time

import numpy as np

from check_role_metrics import pattern, sha256

HERE = Path(__file__).resolve().parents[1]


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def run(args):
    started = time.monotonic()
    if sha256(args.protocol) != args.protocol_sha256:
        raise ValueError("Protocol changed")
    counts, maxima = Counter(), defaultdict(float)

    def close(name, actual, expected, exact=False):
        actual, expected = float(actual), float(expected)
        error = abs(actual - expected)
        if not np.isfinite(actual) or not np.isfinite(expected) or error > (0 if exact else 1e-10 + 1e-9 * abs(expected)):
            raise AssertionError((name, actual, expected, error))
        counts[name] += 1
        maxima[name] = max(maxima[name], error)

    mappings = {"mean_abs_task_score_change": "all_task_mean_abs_change",
                "minimum_task_mean_change": "taskmean_min_change", "maximum_task_mean_change": "taskmean_max_change",
                "mean_abs_task_mean_change": "taskmean_mean_abs_change"}
    final_counts, source_hashes = {}, {}
    for category in ("CONTRAST", "INTERACTION"):
        identity_field = "role" if category == "CONTRAST" else "role_pair"
        expected_rows = rows(HERE / "qa" / f"INDEPENDENT_ROLE_{category}S.tsv")
        expected = {(row[identity_field], row["depth"], row["model"], row["metric"]): row for row in expected_rows}
        expected_gaps = rows(HERE / "qa" / f"INDEPENDENT_ROLE_{category}_GAPS.tsv")
        gaps = {(row[identity_field], row["depth"], row["model"], row["metric"], float(row["gap"])): row for row in expected_gaps}
        path = args.results / ("role_contrasts.tsv" if category == "CONTRAST" else "role_interactions.tsv")
        supplied = rows(path)
        seen = set()
        for row in supplied:
            role = row["role"] if category == "CONTRAST" else row["role_a"] + ":" + row["role_b"]
            key = (role, row["depth"], row["model"], row["metric"])
            if key in seen or key not in expected:
                raise AssertionError(("Duplicate/unexpected summary", key))
            seen.add(key)
            wanted = expected[key]
            for destination, source in mappings.items():
                close(category + "_" + destination, row[destination], wanted[source])
            close(category + "_task_count", row["task_count"], 55, True)
            close(category + "_allocation_count", row["allocation_count"], 30, True)
            close(category + "_edge_count", row["directed_contrasts_per_allocation"], wanted["edges_per_allocation"], True)
            close(category + "_total_edge_count", row["total_task_mean_contrasts"], int(wanted["edges_per_allocation"]) * 30, True)
            for gap, label in ((0., "0"), (.005, "0p005"), (.01, "0p01"), (.02, "0p02")):
                for sign in ("positive", "negative", "within"):
                    close(category + "_gap_counts", row[f"gap_{label}_{sign}_count"], gaps[(*key, gap)][sign], True)
        if seen != set(expected) or len(supplied) != 432:
            raise AssertionError(("Incomplete summaries", category, len(seen)))
        final_counts[path.name] = len(supplied)
        source_hashes[path.name] = sha256(path)

    with np.load(args.cube, allow_pickle=False) as archive:
        cube, direct = archive["scores_role_effect"], archive["scores_direct_state"]
        roles = tuple(tuple(map(int, role)) for role in archive["role_tuples"])
        role_index = {role: i for i, role in enumerate(roles)}
        depths, models, metrics = (archive[name].tolist() for name in ("depths", "models", "metrics"))
    pattern_indices = {name: [i for i, role in enumerate(roles) if pattern(role) == name] for name in ("S", "M", "P", "O", "D")}
    pattern_rows = rows(args.results / "pattern_means.tsv")
    seen = set()
    for row in pattern_rows:
        di, mi, ki = depths.index(int(row["depth"])), models.index(row["model"]), metrics.index(row["metric"])
        name = row["pattern"]
        key = (name, di, mi, ki)
        if key in seen:
            raise AssertionError(("Duplicate pattern row", key))
        seen.add(key)
        values = direct[:, di, :, :, mi, ki] if name == "direct_state" else cube[:, di, :, pattern_indices[name], mi, ki]
        close("pattern_mean_score", row["mean_score"], np.mean(values))
        close("pattern_tuple_count", row["tuples_per_allocation"], 3 if name == "direct_state" else len(pattern_indices[name]), True)
        close("pattern_controls_per_block", row["control_cells_per_block"], int(row["depth"]) * 8, True)
        close("pattern_controls_available", row["controls_available"], int(row["depth"]) * 24, True)
        close("pattern_undefined", row["undefined_values"], np.sum(~np.isfinite(values)), True)
    if len(seen) != 864:
        raise AssertionError(("Incomplete pattern means", len(seen)))
    final_counts["pattern_means.tsv"] = len(pattern_rows)
    source_hashes["pattern_means.tsv"] = sha256(args.results / "pattern_means.tsv")

    workflow_rows = rows(args.results / "workflow_comparison.tsv")
    seen = set()
    for row in workflow_rows:
        di, mi, ki = depths.index(int(row["depth"])), models.index(row["model"]), metrics.index(row["metric"])
        key = (row["view"], di, mi, ki)
        if key in seen:
            raise AssertionError(("Duplicate workflow row", key))
        seen.add(key)
        if row["view"] == "direct_state":
            values = direct[:, di, :, :, mi, ki].mean(axis=0)
            anchor = direct[:, di, 0, 0, mi, ki].mean()
            repeats = 3
            if row["A_Cobs"] != "unused" or row["A_Cpred"] != "unused":
                raise AssertionError("Direct-state anchor incorrectly uses scoring roles")
        else:
            if row["view"] != "reference_effect_O":
                raise AssertionError(row["view"])
            # Basic slicing preserves task axis before taking the paired roles.
            base = cube[:, di, :, :, mi, ki]
            values = base[:, :, pattern_indices["O"]].mean(axis=0)
            anchor = cube[:, di, 0, role_index[(1, 0, 0)], mi, ki].mean()
            repeats = 6
            close("workflow_anchor_Cobs", row["A_Cobs"], 1, True)
            close("workflow_anchor_Cpred", row["A_Cpred"], 0, True)
        close("workflow_anchor_allocation", row["A_allocation"], 0, True)
        close("workflow_anchor_Cmodel", row["A_Cmodel"], 0, True)
        close("workflow_A_anchor", row["A_anchor_task_mean"], anchor)
        close("workflow_B_mean", row["B_repeated_task_mean"], values.mean())
        close("workflow_B_min", row["B_minimum_task_mean"], values.min())
        close("workflow_B_max", row["B_maximum_task_mean"], values.max())
        close("workflow_C_same_primary", row["C_same_primary_mean"], values.mean())
        close("workflow_same_primary_difference", row["C_minus_B_same_primary"], 0., True)
        close("workflow_repeats", row["repeated_realizations"], repeats * 30, True)
        close("workflow_repeats_per_allocation", row["realizations_per_allocation"], repeats, True)
        if row["all_task_support"] != "True":
            raise AssertionError("Unexpected incomplete workflow support")
    if len(seen) != 288:
        raise AssertionError(("Incomplete workflow comparison", len(seen)))
    final_counts["workflow_comparison.tsv"] = len(workflow_rows)
    source_hashes["workflow_comparison.tsv"] = sha256(args.results / "workflow_comparison.tsv")
    receipt = {"status": "PASS", "protocol_sha256": args.protocol_sha256,
               "production_summary_imported": False, "row_counts": final_counts,
               "comparison_counts": dict(counts), "maximum_absolute_errors": dict(maxima),
               "files_sha256": source_hashes, "cube_sha256": sha256(args.cube), "checker_sha256": sha256(__file__),
               "oracle_receipt_sha256": sha256(HERE / "qa/CONTRAST_ORACLE_RECEIPT.json"),
               "elapsed_seconds": time.monotonic() - started}
    (HERE / "qa/INDEPENDENT_SUMMARY_CHECK.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--cube", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    run(parser.parse_args())
