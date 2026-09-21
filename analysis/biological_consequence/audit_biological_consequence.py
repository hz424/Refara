"""Alternate matrix implementation and supporting table aggregation.

This audit deliberately does not import the production analysis module.
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def table(path, data):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--axes", type=Path, required=True)
    p.add_argument("--mapping", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    assert not a.output_dir.exists(), "Use a new audit directory"
    a.output_dir.mkdir(parents=True)
    receipt = json.loads((a.results / "analysis_receipt.json").read_text())
    assert receipt["status"] == "PASS_DESCRIPTIVE_FINITE_PANEL_READOUTS"
    for name, expected in receipt["outputs_sha256"].items():
        assert sha(a.results / name) == expected, name
    for key, path in (("inputs", a.inputs), ("axes", a.axes), ("programs", a.mapping)):
        assert sha(path) == receipt["files_sha256"][key], key
    with np.load(a.inputs, allow_pickle=False) as z:
        source = {k: z[k] for k in z.files}
    axes = json.loads(a.axes.read_text())
    mapping = json.loads(a.mapping.read_text())
    selected = np.zeros((8, 2), dtype=int)
    scores = np.zeros((8, 2, 8))
    ties = {}
    u = source["selection_utilities"]
    for r in range(8):
        for policy, pattern in enumerate((0, 4)):
            for model in range(8):
                scores[r, policy, model] = sum(
                    float(np.sum(u[:, pattern, other, model], dtype=np.float64))
                    for other in range(8) if other != r) / 7000
            best = max(scores[r, policy])
            maxima = [m for m in range(8) if best - scores[r, policy, m] <= 1e-12]
            ties[r, policy] = maxima
            selected[r, policy] = maxima[0]
    with np.load(a.results / "program_readouts.npz", allow_pickle=False) as z:
        archived = {k: z[k] for k in z.files}
    assert np.array_equal(selected, archived["selected_model_indices"])
    selection_error = float(np.max(np.abs(scores - archived["selection_utilities"])))
    assert selection_error < 2e-14
    for row in rows(a.results / "selection.tsv"):
        r, policy = int(row["root_index"]), ("S", "D").index(row["policy"])
        assert list(map(int, row["maximizer_indices"].split(","))) == ties[r, policy]
        assert r not in list(map(int, row["selection_root_indices"].split(",")))
    eligible = [x for x in mapping["programs"] if x["eligible"]]
    ids = [x["program_id"] for x in eligible]
    assert ids == archived["program_ids"].tolist()
    weights = np.zeros((2000, len(eligible)))
    for h, program in enumerate(eligible):
        ix = program["panel_gene_indices"]
        weights[ix, h] = 1 / (len(ix) * source["scales"][ix])
    pairs = list(itertools.permutations(range(3), 2))
    recreated = np.zeros_like(archived["scores"])
    metrics = np.zeros((10, 6, 40, 7))
    direct = source["direct_effects"].astype(np.float64)
    direct_scores = direct @ weights
    root_index = source["root_index"]
    primary_effects = None
    for lp in range(10):
        for pp, (inf, obs) in enumerate(pairs):
            controls = source["control_means"][lp].astype(np.float64)
            native = source["state_predictions"][lp, inf].astype(np.float64)
            state = native - controls[inf, :, None, :]
            all_scores = np.concatenate((direct_scores, state @ weights), axis=1)
            measured = source["treated_means"] - controls[obs]
            predicted_s = all_scores[np.arange(40), selected[root_index, 0]]
            predicted_d = all_scores[np.arange(40), selected[root_index, 1]]
            observed = measured @ weights
            recreated[lp, pp] = np.stack((observed, predicted_s, predicted_d), axis=1)
            effects = np.concatenate((direct, state), axis=1)
            pred_s = effects[np.arange(40), selected[root_index, 0]]
            pred_d = effects[np.arange(40), selected[root_index, 1]]
            difference = np.sum(np.abs(predicted_d - predicted_s), axis=1) / len(ids)
            es = np.sum(np.abs(predicted_s - observed), axis=1) / len(ids)
            ed = np.sum(np.abs(predicted_d - observed), axis=1) / len(ids)
            metrics[lp, pp] = np.column_stack((difference, es, ed, ed - es,
                np.sqrt(np.sum(measured * measured, axis=1) / 2000),
                np.sqrt(np.sum(pred_s * pred_s, axis=1) / 2000),
                np.sqrt(np.sum(pred_d * pred_d, axis=1) / 2000)))
            if lp == pp == 0:
                primary_effects = (measured, pred_s, pred_d)
    score_error = float(np.max(np.abs(recreated - archived["scores"])))
    assert score_error < 2e-14
    names = ["program_prediction_difference", "program_error_S", "program_error_D",
             "program_error_change", "rms_observed", "rms_S", "rms_D"]
    metric_error = 0.
    for row in rows(a.results / "reference_sensitivity.tsv"):
        lp = int(row["label_id"]) // 100
        pp = pairs.index((int(row["inference_block"]) - 1, int(row["observation_block"]) - 1))
        n = int(row["task_row"])
        expected = np.array([float(row[k]) for k in names])
        metric_error = max(metric_error, float(np.max(np.abs(metrics[lp, pp, n] - expected))))
    assert metric_error < 2e-14
    donor = np.zeros((10, 6, 8, 7))
    for r in range(8):
        donor[:, :, r] = np.sum(metrics[:, :, root_index == r], axis=2) / 5
    overall = np.sum(donor, axis=2) / 8
    for row in rows(a.results / "reference_sensitivity_donors.tsv"):
        lp = int(row["label_id"]) // 100
        pp = pairs.index((int(row["inference_block"]) - 1, int(row["observation_block"]) - 1))
        r = axes["root_ids"].index(row["root_id"])
        assert np.allclose(donor[lp, pp, r], [float(row[k]) for k in names], rtol=0, atol=2e-14)
    for row in rows(a.results / "reference_sensitivity_overall.tsv"):
        lp = int(row["label_id"]) // 100
        pp = pairs.index((int(row["inference_block"]) - 1, int(row["observation_block"]) - 1))
        assert np.allclose(overall[lp, pp], [float(row[k]) for k in names], rtol=0, atol=2e-14)
    # Use the archived scalar results to verify the exact display-order rule.
    task_rows = rows(a.results / "task_summary.tsv")
    ordered = sorted(float(x["program_prediction_difference"]) for x in task_rows)
    median = (ordered[19] + ordered[20]) / 2
    candidate_rows = [i for i, row in enumerate(task_rows)
                      if float(row["program_prediction_difference"]) in (ordered[19], ordered[20])]
    example = json.loads((a.results / "example_selection.json").read_text())
    assert example["task_row"] == min(candidate_rows)
    assert abs(example["median_program_prediction_difference"] - median) < 2e-14
    programs = rows(a.results / "program_scores.tsv")
    case = {row["program_id"]: row for row in programs if int(row["task_row"]) == example["task_row"]}
    top = sorted(range(len(ids)), key=lambda h: (-abs(float(case[ids[h]]["observed"])), h))[:10]
    assert example["top_program_ids"] == [ids[h] for h in top]
    # Recompute every primary top-gene overlap with Python sorted sets.
    primary_overlaps = [row for row in rows(a.results / "gene_overlaps.tsv") if row["is_primary"] == "True"]
    overlap_error = 0.
    for row in primary_overlaps:
        n, k = int(row["task_row"]), int(row["K"])
        left, right = {"S_D": (1, 2), "S_observed": (1, 0), "D_observed": (2, 0)}[row["comparison"]]
        x, y = primary_effects[left][n], primary_effects[right][n]
        sets = [set(sorted((i for i in range(2000) if vector[i] != 0),
                           key=lambda i: (-abs(vector[i]), i))[:k]) for vector in (x, y)]
        common, union = sets[0] & sets[1], sets[0] | sets[1]
        signed = sum((x[i] > 0) == (y[i] > 0) for i in common)
        ordinary = len(common) / len(union) if union else 1.
        direction = signed / len(union) if union else 1.
        assert int(row["intersection_size"]) == len(common) and int(row["union_size"]) == len(union)
        overlap_error = max(overlap_error, abs(float(row["jaccard"]) - ordinary),
                            abs(float(row["direction_consistent_overlap"]) - direction))
    assert overlap_error < 2e-14
    support = []
    for k in (25, 50, 100):
        for comparison in ("S_D", "S_observed", "D_observed"):
            block = [r for r in primary_overlaps if int(r["K"]) == k and r["comparison"] == comparison]
            assert len(block) == 40
            j, signed = [], []
            for root in axes["root_ids"]:
                group = [r for r in block if r["root_id"] == root]
                assert len(group) == 5
                j.append(sum(float(r["jaccard"]) for r in group) / 5)
                signed.append(sum(float(r["direction_consistent_overlap"]) for r in group) / 5)
            support.append({"K": k, "comparison": comparison, "mean_jaccard": sum(j) / 8,
                            "mean_direction_consistent_overlap": sum(signed) / 8,
                            "both_empty_task_count": sum(r["both_empty"] == "True" for r in block),
                            "one_empty_task_count": sum(r["one_empty"] == "True" for r in block),
                            "task_count": 40, "donor_count": 8})
    table(a.output_dir / "Table_6_gene_overlap_summary.tsv", support)
    table(a.output_dir / "Table_6_response_magnitude.tsv", [{"quantity": key, "equal_donor_mean_RMS": float(overall[0, 0, 4+i])}
          for i, key in enumerate(("Observed", "S-selected (scGen)", "D-selected (PCA-64 ridge)"))])
    findings = {"status": "PASS", "implementation": "Alternate explicit donor sums and matrix-weighted program projection; no production analysis functions imported.",
                "files_sha256": {"inputs": sha(a.inputs), "axes": sha(a.axes), "mapping": sha(a.mapping),
                                  "analysis_receipt": sha(a.results / "analysis_receipt.json"),
                                  "audit_implementation": sha(__file__)},
                "maximum_absolute_selection_utility_difference": selection_error,
                "maximum_absolute_program_score_difference": score_error,
                "maximum_absolute_task_metric_difference": metric_error,
                "maximum_absolute_primary_gene_overlap_difference": overlap_error,
                "all_60_program_assignments_verified": True, "all_40_tasks_verified": True,
                "all_maximizer_sets_and_heldout_exclusions_verified": True,
                "task_donor_overall_aggregation_verified": True,
                "representative_case_and_top_program_order_verified": True,
                "outputs_sha256": {x.name: sha(x) for x in a.output_dir.iterdir() if x.is_file()}}
    (a.output_dir / "ALTERNATE_IMPLEMENTATION_AUDIT.json").write_text(json.dumps(findings, indent=2) + "\n")
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()
