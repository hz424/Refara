"""Enumerate complete role contrasts independently of the study summarizer."""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "2"

import argparse
import csv
import itertools
import json
from pathlib import Path
import time

import numpy as np

from check_role_metrics import sha256


HERE = Path(__file__).resolve().parents[1]
ROLES = ("Cobs", "Cpred", "Cmodel")
PAIRS = tuple(itertools.permutations(range(3), 2))
GAPS = (0., .005, .01, .02)


def contrast_edges(cube, axis, role_index):
    # Iterate edges of the 3 x 3 x 3 cube by their starting and ending tuples.
    # This differs from the production enumeration over fixed-coordinate axes.
    for start, end in itertools.product(role_index, repeat=2):
        changed = [i for i, (left, right) in enumerate(zip(start, end)) if left != right]
        if changed == [axis]:
            yield cube[:, :, role_index[end]] - cube[:, :, role_index[start]]


def interaction_edges(cube, first_axis, second_axis, role_index):
    for start, end in itertools.product(role_index, repeat=2):
        changed = [i for i, (left, right) in enumerate(zip(start, end)) if left != right]
        if changed != [first_axis, second_axis]:
            continue
        first_changed, second_changed = list(start), list(start)
        first_changed[first_axis] = end[first_axis]
        second_changed[second_axis] = end[second_axis]
        yield (cube[:, :, role_index[end]] - cube[:, :, role_index[tuple(first_changed)]]
               - cube[:, :, role_index[tuple(second_changed)]] + cube[:, :, role_index[start]])


def summarize(values, identity, models, metrics):
    # values: task, allocation, edge, model, metric. Keep task means separate
    # from the mean magnitude across individual tasks.
    mean_tasks = values.mean(axis=0)
    task_mean_abs = np.abs(values).mean(axis=(0, 1, 2))
    task_max_abs = np.abs(values).max(axis=(0, 1, 2))
    min_value = mean_tasks.min(axis=(0, 1))
    max_value = mean_tasks.max(axis=(0, 1))
    mean_abs = np.abs(mean_tasks).mean(axis=(0, 1))
    max_abs = np.abs(mean_tasks).max(axis=(0, 1))
    signed_mean = mean_tasks.mean(axis=(0, 1))
    rows = []
    gap_rows = []
    for mi, model in enumerate(models):
        for ki, metric in enumerate(metrics):
            row = {**identity, "model": model, "metric": metric,
                   "edges_per_allocation": values.shape[2], "task_edge_values": int(np.prod(values.shape[:3])),
                   "all_task_mean_abs_change": float(task_mean_abs[mi, ki]),
                   "all_task_max_abs_change": float(task_max_abs[mi, ki]),
                   "taskmean_min_change": float(min_value[mi, ki]), "taskmean_max_change": float(max_value[mi, ki]),
                   "taskmean_mean_abs_change": float(mean_abs[mi, ki]), "taskmean_max_abs_change": float(max_abs[mi, ki]),
                   "taskmean_signed_mean_change": float(signed_mean[mi, ki])}
            rows.append(row)
            observations = mean_tasks[:, :, mi, ki]
            for gap in GAPS:
                threshold = gap + 1e-12
                positive = int(np.sum(observations > threshold))
                negative = int(np.sum(observations < -threshold))
                gap_rows.append({**identity, "model": model, "metric": metric, "gap": gap,
                                 "taskmean_values": observations.size, "positive": positive, "negative": negative,
                                 "within": int(observations.size - positive - negative)})
    return rows, gap_rows


def write_tsv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    started = time.monotonic()
    if sha256(args.protocol) != args.protocol_sha256:
        raise ValueError("Protocol changed")
    with np.load(args.cube, allow_pickle=False) as archive:
        scores = archive["scores_role_effect"]
        role_index = {tuple(map(int, row)): i for i, row in enumerate(archive["role_tuples"])}
        models, metrics = archive["models"].tolist(), archive["metrics"].tolist()
        depths = archive["depths"].tolist()
    contrast_rows, contrast_gaps, interaction_rows, interaction_gaps = [], [], [], []
    for di, depth in enumerate(depths):
        cube = scores[:, di]
        for axis, role in enumerate(ROLES):
            edges = np.stack(list(contrast_edges(cube, axis, role_index)), axis=2)
            if edges.shape != (55, 30, 54, 6, 4):
                raise ValueError(edges.shape)
            rows, gaps = summarize(edges, {"depth": depth, "role": role}, models, metrics)
            contrast_rows.extend(rows)
            contrast_gaps.extend(gaps)
        for first_axis, second_axis in itertools.combinations(range(3), 2):
            edges = np.stack(list(interaction_edges(cube, first_axis, second_axis, role_index)), axis=2)
            if edges.shape != (55, 30, 108, 6, 4):
                raise ValueError(edges.shape)
            rows, gaps = summarize(edges, {"depth": depth, "role_pair": ROLES[first_axis] + ":" + ROLES[second_axis]}, models, metrics)
            interaction_rows.extend(rows)
            interaction_gaps.extend(gaps)
    destinations = {"INDEPENDENT_ROLE_CONTRASTS.tsv": contrast_rows, "INDEPENDENT_ROLE_CONTRAST_GAPS.tsv": contrast_gaps,
                    "INDEPENDENT_ROLE_INTERACTIONS.tsv": interaction_rows, "INDEPENDENT_ROLE_INTERACTION_GAPS.tsv": interaction_gaps}
    for filename, rows in destinations.items():
        write_tsv(HERE / "qa" / filename, rows)
    receipt = {"status": "INDEPENDENT_ENUMERATION_COMPLETE", "protocol_sha256": args.protocol_sha256,
               "production_summary_imported": False, "cube_sha256": sha256(args.cube), "oracle_sha256": sha256(__file__),
               "row_counts": {name: len(rows) for name, rows in destinations.items()},
               "files_sha256": {name: sha256(HERE / "qa" / name) for name in destinations},
               "directed_edge_signs": "Both directions are included; symmetric signs are structural. Magnitudes and support are the descriptive endpoints.",
               "gap_count_rule": "Positive > gap + 1e-12; negative < -gap - 1e-12; remaining within. Numerical tolerance is separate from descriptive effect-size gaps.",
               "elapsed_seconds": time.monotonic() - started}
    (HERE / "qa/CONTRAST_ORACLE_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--cube", type=Path, required=True)
    run(parser.parse_args())
