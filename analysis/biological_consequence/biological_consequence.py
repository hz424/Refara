"""Fixed-panel consequences of S/D model selection; NumPy and Python 3.10+."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PAIRS = tuple(itertools.permutations(range(3), 2))
LABELS = tuple(range(0, 1000, 100))
MODEL_IDS = ("NO_CHANGE_DIRECT_V1", "CONTEXT_MEAN_EFFECT_DIRECT_V1",
             "TWO_WAY_ADDITIVE_RIDGE_DIRECT_V1", "PCA64_ADDITIVE_RIDGE_DIRECT_V1",
             "RBF_KERNEL_RIDGE_DIRECT_V1", "CPA_0_8_8_ABSOLUTE_V1",
             "SCGEN_2_1_1_ABSOLUTE_V1", "CELLOT_522D2B9_ABSOLUTE_V1")
METRICS = ("program_prediction_difference", "program_error_S", "program_error_D",
           "program_error_change", "rms_observed", "rms_S", "rms_D")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n")


def write_tsv(path, rows):
    require(bool(rows), "Cannot write an empty table")
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def select_models(utilities, tolerance=1e-12):
    """utilities: label, pattern(S/M/P/O/D), donor, configuration."""
    u = np.asarray(utilities, dtype=np.float64)
    require(u.ndim == 4 and u.shape[1] == 5 and u.shape[2] >= 2,
            "Expected label × five patterns × donor × model utilities")
    require(np.isfinite(u).all(), "Selection utilities must be complete and finite")
    roots, models = u.shape[2:]
    selected = np.empty((roots, 2), dtype=np.int64)
    scores = np.empty((roots, 2, models), dtype=np.float64)
    records = []
    for root in range(roots):
        other = np.arange(roots) != root
        for policy, pattern in enumerate((0, 4)):
            # Average labels within donor, then the seven selection donors equally.
            score = u[:, pattern, other, :].mean(axis=0).mean(axis=0)
            scores[root, policy] = score
            maximum = float(score.max())
            ties = np.flatnonzero(maximum - score <= tolerance)
            winner = int(ties[0])
            selected[root, policy] = winner
            ordered = np.sort(score)[::-1]
            outside = np.setdiff1d(np.arange(models), ties)
            records.append({"root_index": root, "policy": ("S", "D")[policy],
                            "selected_model_index": winner,
                            "maximum_utility": maximum,
                            "selected_utility": float(score[winner]),
                            "maximizer_indices": ties.tolist(),
                            "top_two_gap": float(ordered[0] - ordered[1]),
                            "gap_to_best_outside_tie":
                                float(maximum - score[outside].max()) if len(outside) else None,
                            "selection_root_indices": np.flatnonzero(other).tolist()})
    return selected, scores, records


def selected_effects(data, selectors, label_position, inference_block):
    """Convert stored values to float64; centre emitted state outputs exactly once."""
    roots = data["root_index"]
    n, g = data["treated_means"].shape
    effects = np.empty((n, 2, g), dtype=np.float64)
    control = np.asarray(data["control_means"][label_position, inference_block], dtype=np.float64)
    for policy in range(2):
        for row in range(n):
            model = int(selectors[int(roots[row]), policy])
            if model < 5:
                effects[row, policy] = np.asarray(data["direct_effects"][row, model], dtype=np.float64)
            else:
                native = np.asarray(data["state_predictions"][label_position, inference_block,
                                                               row, model - 5], dtype=np.float64)
                effects[row, policy] = native - control[row]
    return effects


def program_scores(effects, scales, members):
    standardized = np.asarray(effects, dtype=np.float64) / np.asarray(scales, dtype=np.float64)
    return np.stack([standardized[..., indices].mean(axis=-1) for indices in members], axis=-1)


def absolute_ranks(values):
    """Competition ranks with exact ties; all zero programs have the same rank."""
    a = np.abs(np.asarray(values, dtype=np.float64))
    return 1 + (a[..., None, :] > a[..., :, None]).sum(axis=-1)


def top_genes(effect, k):
    effect = np.asarray(effect)
    nonzero = np.flatnonzero(effect != 0)
    return nonzero[np.argsort(-np.abs(effect[nonzero]), kind="stable")[:k]]


def gene_overlap(a, b, k):
    ia, ib = top_genes(a, k), top_genes(b, k)
    shared = np.intersect1d(ia, ib)
    union = np.union1d(ia, ib)
    consistent = int(np.count_nonzero(np.sign(a[shared]) == np.sign(b[shared])))
    both_empty = len(union) == 0
    return {"size_A": len(ia), "size_B": len(ib), "intersection_size": len(shared),
            "union_size": len(union), "same_sign_intersection_size": consistent,
            "jaccard": len(shared) / len(union) if len(union) else 1.0,
            "direction_consistent_overlap": consistent / len(union) if len(union) else 1.0,
            "both_empty": bool(both_empty), "one_empty": bool((len(ia) == 0) != (len(ib) == 0))}


def aggregate(values, root_index):
    """Equal task means per donor, followed by equal donor means."""
    donor = np.stack([values[root_index == r].mean(axis=0)
                      for r in range(int(root_index.max()) + 1)])
    return donor, donor.mean(axis=0)


def representative_case(difference, observed_program_scores):
    median = float(np.median(difference))
    row = int(np.argmin(np.abs(difference - median)))
    top = np.argsort(-np.abs(observed_program_scores[row]), kind="stable")[:10]
    return row, top, median


def validate_inputs(data, axes, mapping):
    expected = {"selection_utilities": (1000, 5, 8, 8),
                "state_predictions": (10, 3, 40, 3, 2000),
                "direct_effects": (40, 5, 2000), "control_means": (10, 3, 40, 2000),
                "treated_means": (40, 2000), "scales": (2000,),
                "root_index": (40,), "task_index": (40,), "label_ids": (10,),
                "selection_label_ids": (1000,)}
    for key, shape in expected.items():
        require(key in data and data[key].shape == shape, f"Bad input shape: {key}")
        require(np.isfinite(data[key]).all(), f"Nonfinite input: {key}")
    require((data["scales"] > 0).all(), "Training scales must be positive")
    require(np.array_equal(data["label_ids"], LABELS), "Unexpected sensitivity labels")
    require(np.array_equal(data["selection_label_ids"], np.arange(1000)),
            "Selection must retain every original label in its original order")
    require(data["root_index"].dtype.kind in "iu" and data["task_index"].dtype.kind in "iu",
            "Root/task indices must be integers")
    require(sorted(zip(data["root_index"].tolist(), data["task_index"].tolist())) ==
            list(itertools.product(range(8), range(5))), "Need each of 40 root/task pairs exactly once")
    require(np.array_equal(data["root_index"], np.repeat(np.arange(8), 5)) and
            np.array_equal(data["task_index"], np.tile(np.arange(5), 8)),
            "Rows must retain original donor-major/task-minor order for display tie resolution")
    for key, size in (("model_ids", 8), ("root_ids", 8), ("task_ids", 5), ("gene_ids", 2000)):
        require(len(axes[key]) == size and len(set(axes[key])) == size, f"Bad axis: {key}")
    require(axes["pattern_ids"] == ["S", "M", "P", "O", "D"], "Pattern order mismatch")
    require(tuple(axes["model_ids"]) == MODEL_IDS, "Original configuration order differs")
    require(axes["model_output_kinds"] == ["direct_effect"] * 5 + ["absolute_state"] * 3,
            "Model output semantics/order mismatch")
    require(axes["direct_model_ids"] == axes["model_ids"][:5] and
            axes["state_model_ids"] == axes["model_ids"][5:], "Prediction model axes disagree")
    require(axes["gene_ids"] == mapping["gene_ids"], "Gene panel and mapping order disagree")
    require(len(mapping["programs"]) == 50, "The complete 50-program collection is required")
    require(len({x["program_id"] for x in mapping["programs"]}) == 50, "Program IDs must be unique")
    require(np.count_nonzero(data["direct_effects"][:, 0]) == 0, "NC must emit exact zero")
    if "weights" in data:
        require(data["weights"].shape == (2000,) and
                np.allclose(data["weights"], 1 / 2000, rtol=1e-12, atol=0),
                "Expected original equal gene weights; scales remain a separate input")
    coverage, members, ids = [], [], []
    for program in mapping["programs"]:
        ix = np.asarray(program["panel_gene_indices"], dtype=np.int64)
        count = int(program["mapped_member_count"])
        require(count == program["official_unique_member_count"],
                "Coverage denominator must be the complete official unique-member count")
        require(count > 0 and len(ix) == len(set(ix.tolist())) and
                np.all((ix >= 0) & (ix < 2000)), "Bad program mapping")
        fraction = len(ix) / count
        eligible = len(ix) >= 15 and fraction >= 0.1
        require(program["eligible"] == eligible, "Frozen coverage decision disagrees")
        coverage.append({"program_id": program["program_id"], "panel_member_count": len(ix),
                         "official_member_count": count, "coverage_fraction": fraction,
                         "eligible": eligible})
        if eligible:
            members.append(ix)
            ids.append(program["program_id"])
    return coverage, members, ids


def analyze(data, axes, members, program_ids, output):
    selectors, utilities, selection = select_models(data["selection_utilities"])
    model_labels = axes.get("model_labels", axes["model_ids"])
    selection_rows = []
    for r in selection:
        root, policy = r["root_index"], r["policy"]
        selected = r["selected_model_index"]
        selection_rows.append({**r, "root_id": axes["root_ids"][root],
                               "selected_model": model_labels[selected],
                               "selected_model_id": axes["model_ids"][selected],
                               "maximizer_indices": ",".join(map(str, r["maximizer_indices"])),
                               "maximizer_models": ",".join(axes["model_ids"][i] for i in r["maximizer_indices"]),
                               "selection_root_indices": ",".join(map(str, r["selection_root_indices"]))})
    write_tsv(output / "selection.tsv", selection_rows)
    ids = []
    for n in range(40):
        root, task = int(data["root_index"][n]), int(data["task_index"][n])
        ids.append({"task_row": n, "root_id": axes["root_ids"][root],
                    "task_id": axes["task_ids"][task],
                    "cell_type": axes.get("cell_types", axes["task_ids"])[task],
                    "selected_model_S": model_labels[selectors[root, 0]],
                    "selected_model_D": model_labels[selectors[root, 1]],
                    "selected_model_id_S": axes["model_ids"][selectors[root, 0]],
                    "selected_model_id_D": axes["model_ids"][selectors[root, 1]],
                    "selection_changed": bool(selectors[root, 0] != selectors[root, 1])})
    z_all = np.empty((10, 6, 40, 3, len(members)), dtype=np.float64)
    sensitivity, donor_sensitivity, overall_sensitivity, overlaps = [], [], [], []
    primary = None
    for lp, label in enumerate(LABELS):
        for pp, (inf, obs) in enumerate(PAIRS):
            effects = selected_effects(data, selectors, lp, inf)
            observed = np.asarray(data["treated_means"], dtype=np.float64) - np.asarray(
                data["control_means"][lp, obs], dtype=np.float64)
            all_effects = np.concatenate((observed[:, None, :], effects), axis=1)
            z = program_scores(all_effects, data["scales"], members)
            z_all[lp, pp] = z
            difference = np.abs(z[:, 2] - z[:, 1]).mean(axis=-1)
            error_s = np.abs(z[:, 1] - z[:, 0]).mean(axis=-1)
            error_d = np.abs(z[:, 2] - z[:, 0]).mean(axis=-1)
            rms = np.sqrt(np.mean(all_effects ** 2, axis=-1))
            metrics = np.column_stack((difference, error_s, error_d, error_d - error_s, rms))
            require(np.isfinite(metrics).all(), "Nonfinite derived readout")
            same = selectors[data["root_index"], 0] == selectors[data["root_index"], 1]
            require(np.array_equal(z[same, 1], z[same, 2]) and np.all(metrics[same, 0] == 0)
                    and np.all(metrics[same, 3] == 0), "Identical selections must agree exactly")
            reference = {"label_id": label, "inference_block": inf + 1,
                         "observation_block": obs + 1, "is_primary": label == 0 and pp == 0}
            for n in range(40):
                sensitivity.append({**reference, **ids[n],
                                    **dict(zip(METRICS, map(float, metrics[n])))})
                for k in (25, 50, 100):
                    for name, a, b in (("S_D", 1, 2), ("S_observed", 1, 0), ("D_observed", 2, 0)):
                        overlaps.append({**reference, "task_row": n, "root_id": ids[n]["root_id"],
                                         "task_id": ids[n]["task_id"], "K": k, "comparison": name,
                                         **gene_overlap(all_effects[n, a], all_effects[n, b], k)})
            donor, overall = aggregate(metrics, data["root_index"])
            for root in range(8):
                donor_sensitivity.append({**reference, "root_id": axes["root_ids"][root],
                                          **dict(zip(METRICS, map(float, donor[root])))})
            overall_sensitivity.append({**reference, **dict(zip(METRICS, map(float, overall)))})
            if reference["is_primary"]:
                primary = metrics.copy(), z.copy(), donor.copy(), overall.copy()
    metrics, z, donor, overall = primary
    write_tsv(output / "task_summary.tsv", [{**ids[n], **dict(zip(METRICS, map(float, metrics[n])))} for n in range(40)])
    write_tsv(output / "donor_summary.tsv", [{"root_id": axes["root_ids"][r],
              **dict(zip(METRICS, map(float, donor[r])))} for r in range(8)])
    write_tsv(output / "overall_summary.tsv", [dict(zip(METRICS, map(float, overall)))])
    write_tsv(output / "reference_sensitivity.tsv", sensitivity)
    write_tsv(output / "reference_sensitivity_donors.tsv", donor_sensitivity)
    write_tsv(output / "reference_sensitivity_overall.tsv", overall_sensitivity)
    write_tsv(output / "gene_overlaps.tsv", overlaps)
    ranks = absolute_ranks(z)
    write_tsv(output / "program_scores.tsv", [
        {**ids[n], "program_id": name, "observed": float(z[n, 0, h]),
         "S_selected": float(z[n, 1, h]), "D_selected": float(z[n, 2, h]),
         "observed_absolute_rank": int(ranks[n, 0, h]),
         "S_selected_absolute_rank": int(ranks[n, 1, h]),
         "D_selected_absolute_rank": int(ranks[n, 2, h])}
        for n in range(40) for h, name in enumerate(program_ids)])
    row, top, median = representative_case(metrics[:, 0], z[:, 0])
    example = {**ids[row], "median_program_prediction_difference": median,
               "selected_distance_to_median": float(abs(metrics[row, 0] - median)),
               "program_prediction_difference": float(metrics[row, 0]),
               "top_program_ids": [program_ids[i] for i in top],
               "selection_rule": "nearest all-40-task median; original row order breaks exact ties",
               "program_rule": "largest absolute observed shift; frozen program order breaks ties"}
    write_json(output / "example_selection.json", example)
    np.savez_compressed(output / "program_readouts.npz", scores=z_all,
                        absolute_ranks=absolute_ranks(z_all).astype(np.int16),
                        program_ids=np.asarray(program_ids), score_axis=np.asarray(["observed", "S_selected", "D_selected"]),
                        label_ids=np.asarray(LABELS), block_pairs=np.asarray(PAIRS) + 1,
                        selected_model_indices=selectors, selection_utilities=utilities,
                        root_index=data["root_index"], task_index=data["task_index"])
    return {"task_count": 40, "donor_count": 8, "eligible_program_count": len(members),
            "reference_assignment_count": 60, "primary_example_task_row": row,
            "same_selected_model_donors": int(np.sum(selectors[:, 0] == selectors[:, 1])),
            "same_winner_exact_equality_verified": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--axes", type=Path, required=True)
    parser.add_argument("--programs", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--staging-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output_dir.exists(), "Use a new output directory")
    freeze = json.loads(args.freeze.read_text())
    require(freeze["status"] == "FROZEN_BEFORE_NEW_BIOLOGICAL_READOUTS", "Protocol is not frozen")
    require(digest(args.protocol) == freeze["protocol_sha256"], "Protocol digest differs")
    require(digest(args.programs) == freeze["gene_set_mapping_sha256"], "Gene-set mapping digest differs")
    staging = json.loads(args.staging_receipt.read_text())
    require(staging["status"] == "PASS", "Input staging did not pass")
    require(digest(args.inputs) == staging["staged_files"]["benchmark_inputs.npz"]["sha256"] and
            digest(args.axes) == staging["staged_files"]["axes.json"]["sha256"], "Staged input digests differ")
    require(digest(args.programs) == staging["hallmark_mapping_sha256"], "Staging uses a different program mapping")
    input_hashes = {key: digest(getattr(args, key)) for key in
                    ("inputs", "axes", "programs", "protocol", "freeze", "staging_receipt")}
    args.output_dir.mkdir(parents=True)
    start = {"status": "STARTED_AFTER_PROTOCOL_FREEZE", "started_utc": datetime.now(timezone.utc).isoformat(),
             "files_sha256": input_hashes, "implementation_sha256": digest(__file__),
             "python_version": platform.python_version(), "numpy_version": np.__version__,
             "frozen_utc": freeze["frozen_utc"]}
    # Written before loading arrays or evaluating any new biological readout.
    write_json(args.output_dir / "FIRST_RUN_RECEIPT.json", start)
    with np.load(args.inputs, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    axes = json.loads(args.axes.read_text())
    mapping = json.loads(args.programs.read_text())
    coverage, members, program_ids = validate_inputs(data, axes, mapping)
    write_tsv(args.output_dir / "program_coverage.tsv", coverage)
    require(len(members) == freeze["eligible_programs"], "Eligible count differs from frozen coverage")
    if members:
        result = analyze(data, axes, members, program_ids, args.output_dir)
        status = "PASS_DESCRIPTIVE_FINITE_PANEL_READOUTS"
    else:
        result = {"eligible_program_count": 0, "reason": "No program satisfies frozen coverage"}
        status = "INSUFFICIENT_PROGRAM_COVERAGE"
    require(all(digest(getattr(args, key)) == value for key, value in input_hashes.items()), "Inputs changed during execution")
    write_json(args.output_dir / "analysis_receipt.json", {
        **start, "status": status, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "input_dtypes": {k: str(v.dtype) for k, v in data.items()},
        "arithmetic": "Cached arrays converted to float64 before centring, standardization and aggregation; no normalization, log transform, inference or fitting.",
        "scope": "Descriptive contrasts on eight fixed donors; tasks, labels, genes and overlapping programs are not biological replicates.",
        "outputs_sha256": {p.name: digest(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()},
        "results": result})


if __name__ == "__main__":
    main()
