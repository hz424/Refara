#!/usr/bin/env python3
"""Recompute reference geometry and conditioning scores from review mean arrays.

No model is fitted. All 8 roots, 4 tasks, 5 initializations and 6 mappings are
retained. The input is a confidential review capsule, not the public Source Data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

ROOTS = tuple(f"ROOT_{i:02d}" for i in range(1, 9))
TASKS = ("B", "CD14 Mono", "CD16 Mono", "CD4 T")
MAPPINGS = tuple(itertools.permutations(range(3)))
SHAPES = {
    "control_means": (8, 4, 3, 5000),
    "treated_means": (8, 4, 5000),
    "gene_scales": (8, 5000),
    "ridge_effect_terminal": (8, 4, 3, 5000),
    "ridge_state_terminal": (8, 4, 3, 5000),
    "cellflow_terminal": (8, 4, 5, 3, 5000),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_table(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: format(v, ".17g") if isinstance(v, (float, np.floating)) else v for k, v in row.items()})


def utility(pred, obs, scale):
    return -float(np.mean(np.square((pred - obs) / scale)))


def analyze(arrays):
    for name, shape in SHAPES.items():
        if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise ValueError(f"Incomplete/nonfinite {name}")
    if np.any(arrays["gene_scales"] < 0.1):
        raise ValueError("Scale floor differs")
    geometry = np.empty((8, 4, 6))
    e1 = np.empty((8, 4, 5, 6))
    e2 = np.empty((8, 4, 6))
    ru = np.empty((8, 4, 6, 3))
    fu = np.empty((8, 4, 5, 6, 3))
    rd2 = np.empty((8, 4, 6))
    fd2 = np.empty((8, 4, 5, 6))
    geometry_rows, mapping_rows = [], []
    for d, root in enumerate(ROOTS):
        scale = arrays["gene_scales"][d]
        for t, task in enumerate(TASKS):
            controls = arrays["control_means"][d, t]
            treated = arrays["treated_means"][d, t]
            for p, (a, b, c) in enumerate(MAPPINGS):
                A, B, C = controls[[a, b, c]]
                observed = treated - A
                geometry[d, t, p] = np.mean(np.square((A - B) / scale))
                re = arrays["ridge_effect_terminal"][d, t, c]
                rs = arrays["ridge_state_terminal"][d, t, c]
                ridge_u = np.asarray([[utility(re - C, observed, scale), utility(re - P, observed, scale)] for P in (A, B)])
                h = [ridge_u[k, 0] - utility(rs - P, observed, scale) for k, P in enumerate((A, B))]
                e2[d, t, p] = h[1] - h[0]
                for z, block in enumerate((a, b, c)):
                    ru[d, t, p, z] = utility(arrays["ridge_state_terminal"][d, t, block] - B, observed, scale)
                rd2[d, t, p] = np.mean(np.square((arrays["ridge_state_terminal"][d, t, b] - arrays["ridge_state_terminal"][d, t, c]) / scale))
                geometry_rows.append(dict(root_code=root, task=task, mapping_index=p + 1, physical_A=a + 1, physical_B=b + 1, physical_C=c + 1, reference_distance_squared=geometry[d, t, p], E2_before_mapping_average=e2[d, t, p]))
                for seed in range(5):
                    cf = arrays["cellflow_terminal"][d, t, seed, c]
                    flow_u = np.asarray([[utility(cf - C, observed, scale), utility(cf - P, observed, scale)] for P in (A, B)])
                    mismatch = [(ridge_u[k, 0] - flow_u[k, 1]) - 0.5 * ((ridge_u[k, 0] - flow_u[k, 0]) + (ridge_u[k, 1] - flow_u[k, 1])) for k in range(2)]
                    e1[d, t, seed, p] = mismatch[1] - mismatch[0]
                    for z, block in enumerate((a, b, c)):
                        fu[d, t, seed, p, z] = utility(arrays["cellflow_terminal"][d, t, seed, block] - B, observed, scale)
                    fd2[d, t, seed, p] = np.mean(np.square((arrays["cellflow_terminal"][d, t, seed, b] - arrays["cellflow_terminal"][d, t, seed, c]) / scale))
                    row = dict(root_code=root, task=task, initialization_index=seed + 1, mapping_index=p + 1, E1_before_mapping_average=e1[d, t, seed, p])
                    for z, label in enumerate(("A", "B", "C")):
                        row[f"ridge_utility_{label}"] = ru[d, t, p, z]
                        row[f"cellflow_utility_{label}"] = fu[d, t, seed, p, z]
                        row[f"margin_{label}"] = ru[d, t, p, z] - fu[d, t, seed, p, z]
                    row["ridge_squared_output_displacement_B_minus_C"] = rd2[d, t, p]
                    row["cellflow_squared_output_displacement_B_minus_C"] = fd2[d, t, seed, p]
                    mapping_rows.append(row)
    # Preserve the declared mapping, task, initialization reduction order.
    e1_root = e1.mean(axis=3).mean(axis=1).mean(axis=1)
    e2_root = e2.mean(axis=2).mean(axis=1)
    geom_root = geometry.mean(axis=2).mean(axis=1)
    margin_cells = ru[:, :, None, :, :] - fu
    margin_root = margin_cells.mean(axis=3).mean(axis=1).mean(axis=1)
    ridge_root = ru.mean(axis=2).mean(axis=1)
    flow_root = fu.mean(axis=3).mean(axis=1).mean(axis=1)
    ridge_rms = np.sqrt(rd2.mean(axis=2).mean(axis=1))
    flow_rms = np.sqrt(fd2.mean(axis=3).mean(axis=1).mean(axis=1))
    task_rows, root_rows = [], []
    for d, root in enumerate(ROOTS):
        for t, task in enumerate(TASKS):
            task_rows.append(dict(root_code=root, task=task, reference_distance=geometry[d, t].mean(), E1=e1[d, t].mean(axis=1).mean(), E2=e2[d, t].mean()))
        row = dict(root_code=root, reference_distance=geom_root[d], E1=e1_root[d], E2=e2_root[d])
        for z, label in enumerate(("A", "B", "C")):
            row[f"margin_{label}"] = margin_root[d, z]
        row.update(margin_shift_B_minus_C=margin_root[d, 1] - margin_root[d, 2], root_crossing_C_to_B=int(margin_root[d, 1] * margin_root[d, 2] < 0), ridge_utility_shift_B_minus_C=ridge_root[d, 1] - ridge_root[d, 2], cellflow_utility_shift_B_minus_C=flow_root[d, 1] - flow_root[d, 2], ridge_rms_output_displacement=ridge_rms[d], cellflow_rms_output_displacement=flow_rms[d])
        root_rows.append(row)
    task_residual = max(max(abs(row["E1"] - row["reference_distance"]), abs(row["E2"] - row["reference_distance"])) for row in task_rows)
    root_residual = float(max(np.max(abs(e1_root - geom_root)), np.max(abs(e2_root - geom_root))))
    midpoint_residual = float(np.max(abs(margin_root[:, 2] - (margin_root[:, 0] + margin_root[:, 1]) / 2)))
    if max(task_residual, root_residual, midpoint_residual) > 1e-12:
        raise ValueError("Reference geometry or midpoint identity failed")
    summary = dict(status="PASS_MEAN_ARRAY_REPLAY", root_count=8, task_count=4, initialization_count=5, mapping_count=6, geometry_mapping_rows=len(geometry_rows), conditioning_mapping_rows=len(mapping_rows), reference_distance_mean=float(geom_root.mean()), maximum_task_identity_residual=task_residual, maximum_root_identity_residual=root_residual, maximum_midpoint_residual=midpoint_residual, margin_C=float(margin_root[:, 2].mean()), margin_B=float(margin_root[:, 1].mean()), margin_shift_B_minus_C=float((margin_root[:, 1] - margin_root[:, 2]).mean()), negative_paired_shift_roots=int(np.count_nonzero(margin_root[:, 1] < margin_root[:, 2])), root_crossings_C_to_B=int(np.count_nonzero(margin_root[:, 1] * margin_root[:, 2] < 0)), ridge_rms_output_displacement=float(ridge_rms.mean()), cellflow_rms_output_displacement=float(flow_rms.mean()), interpretation="FIXED_PANEL_DESCRIPTIVE; GEOMETRY_IS_AN_IDENTITY_NOT_TWO_VALIDATIONS", model_fitting_performed=False)
    return summary, root_rows, task_rows, geometry_rows, mapping_rows


def replay(capsule, out_dir):
    capsule, out_dir = Path(capsule), Path(out_dir)
    manifest = json.loads((capsule / "MANIFEST.json").read_text())
    for name, digest in manifest["files"].items():
        if sha(capsule / name) != digest:
            raise ValueError(f"Hash mismatch {name}")
    with np.load(capsule / "mean_arrays.npz", allow_pickle=False) as a:
        if set(a.files) != set(SHAPES):
            raise ValueError("Unexpected mean-array roster")
        result = analyze({key: a[key] for key in SHAPES})
    summary, roots, tasks, geom, cells = result
    expected = list(csv.DictReader((capsule / "expected_roots.tsv").open(), delimiter="\t"))
    maximum = 0.0
    for row, old in zip(roots, expected, strict=True):
        if row["root_code"] != old["root_code"]:
            raise ValueError("Root axis differs")
        for key in old:
            if key != "root_code":
                maximum = max(maximum, abs(row[key] - float(old[key])))
    if maximum > 2e-12:
        raise ValueError(f"Sealed/source endpoint mismatch {maximum}")
    summary["maximum_sealed_endpoint_residual"] = maximum
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("KANG_AUDIT_ROOTS_V180.tsv", roots), ("KANG_GEOMETRY_TASKS_V180.tsv", tasks), ("KANG_GEOMETRY_TASK_MAPPING_V180.tsv", geom), ("KANG_CONDITIONING_TASK_SEED_MAPPING_V180.tsv", cells)):
        write_table(out_dir / name, rows)
    (out_dir / "KANG_MEAN_ARRAY_AUDIT_V180.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    replay(args.capsule, args.out_dir)
