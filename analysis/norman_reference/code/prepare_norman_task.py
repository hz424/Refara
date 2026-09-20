#!/usr/bin/env python3
"""Freeze the Norman combination task, then prepare training-only transforms."""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

HERE = Path(__file__).resolve().parent
META = HERE / "metadata"
OUT = HERE / "prepared"
OUT.mkdir(exist_ok=True)
SOURCE = HERE / "source" / "norman_2019_raw.h5ad"
HASH_NAMESPACE = "norman_scope_v1814"


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for data in iter(lambda: stream.read(8 << 20), b""):
            h.update(data)
    return h.hexdigest()


def rank(label, identifier):
    return hashlib.sha256(f"{HASH_NAMESPACE}\0{label}\0{identifier}".encode()).hexdigest()


def canonical(condition):
    return "+".join(sorted("RHOXF2B" if g == "RHOXF2" else g for g in condition.split("+")))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


cells = pd.read_csv(META / "eligible_cell_membership.tsv.gz", sep="\t", keep_default_na=False)
cells["original_condition"] = cells.condition
cells["condition"] = cells.condition.map(canonical)
for c in ["gene_a", "gene_b"]:
    cells[c] = cells[c].replace({"RHOXF2": "RHOXF2B"})
support = pd.read_csv(META / "double_task_support.tsv", sep="\t", keep_default_na=False)
support["original_condition"] = support.condition
support["condition"] = support.condition.map(canonical)
eligible_doubles = support[(support.single_a_n_cells >= 50) & (support.single_b_n_cells >= 50) & (support.n_cells >= 100) & (support.minimum_gemgroup_cells >= 5)].copy()
assert len(eligible_doubles) == 110
eligible_doubles["split_hash"] = eligible_doubles.condition.map(lambda c: rank("combination_split", c))
eligible_doubles = eligible_doubles.sort_values(["split_hash", "condition"]).reset_index(drop=True)
eligible_doubles["partition"] = ["TRAIN"] * 55 + ["TEST"] * 55
train_conditions = set(eligible_doubles.loc[eligible_doubles.partition.eq("TRAIN"), "condition"])
test_conditions = set(eligible_doubles.loc[eligible_doubles.partition.eq("TEST"), "condition"])
cells["partition"] = "EXCLUDED_UNSUPPORTED_DOUBLE"
cells.loc[cells.kind.eq("single"), "partition"] = "TRAIN"
cells.loc[cells.condition.isin(train_conditions), "partition"] = "TRAIN"
cells.loc[cells.condition.isin(test_conditions), "partition"] = "TEST"
cells.loc[cells.kind.eq("control"), "partition"] = "EXCLUDED_CONTROL_CONSTRUCT"
for gem, group in cells[cells.is_author_control].groupby("gemgroup"):
    ids = sorted(group.index, key=lambda idx: (rank(f"control_train_eval_gemgroup_{gem}", cells.at[idx, "cell_id"]), cells.at[idx, "cell_id"]))
    cut = len(ids) // 2
    cells.loc[ids[:cut], "partition"] = "TRAIN"
    cells.loc[ids[cut:], "partition"] = "EVAL_CONTROL"
cells = cells.sort_values("source_row").reset_index(drop=True)
train = cells[cells.partition.eq("TRAIN")].copy()
eval_controls = cells[cells.partition.eq("EVAL_CONTROL")].copy()
test = cells[cells.partition.eq("TEST")].copy()
train_controls = train[train.kind.eq("control")].copy()
assert train_controls.is_author_control.all() and eval_controls.is_author_control.all()
assert set(train.cell_id).isdisjoint(set(test.cell_id) | set(eval_controls.cell_id))
assert set(train.condition).isdisjoint(test_conditions)
assert train[train.kind.eq("single")].condition.nunique() == 105
control_support = cells[cells.is_author_control].groupby(["gemgroup", "partition"]).size().unstack(fill_value=0)
max_depth = int((control_support.EVAL_CONTROL // 3).min())
depths = [8, 16, 32, 64, 128, max_depth]
assert max_depth == 135 and max_depth > 128
eligible_doubles.sort_values("condition").to_csv(OUT / "combination_split.tsv", sep="\t", index=False)
test_tasks = eligible_doubles[eligible_doubles.partition.eq("TEST")].sort_values("condition").reset_index(drop=True)
test_tasks.insert(0, "task_index", np.arange(len(test_tasks)))
test_tasks.to_csv(OUT / "test_tasks.tsv", sep="\t", index=False)
cells.to_csv(OUT / "cell_partitions.tsv.gz", sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
control_support.to_csv(OUT / "control_split_support.tsv", sep="\t")
eval_controls.insert(0, "eval_control_index", np.arange(len(eval_controls)))
eval_controls.to_csv(OUT / "eval_control_cells.tsv", sep="\t", index=False)
train.to_csv(OUT / "train_cells.tsv.gz", sep="\t", index=False, compression={"method": "gzip", "mtime": 0})

# Three maximal disjoint blocks per gemgroup/assignment; smaller depths use
# prefixes of each block, preserving assignment identities across depths.
allocations = []
for assignment in range(30):
    label = f"allocation_{assignment:02d}"
    for gem, group in eval_controls.groupby("gemgroup"):
        ordered = sorted(group.index, key=lambda idx: (rank(f"{label}_gemgroup_{gem}", eval_controls.at[idx, "cell_id"]), eval_controls.at[idx, "cell_id"]))
        for block in range(3):
            for within, idx in enumerate(ordered[block * max_depth:(block + 1) * max_depth]):
                allocations.append({"assignment": assignment, "assignment_label": label, "gemgroup": int(gem), "block": block, "within_block_index": within, "eval_control_index": int(eval_controls.at[idx, "eval_control_index"]), "cell_id": eval_controls.at[idx, "cell_id"]})
pd.DataFrame(allocations).to_csv(OUT / "control_allocations.tsv.gz", sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
sources = pd.read_csv(META / "source_files_sha256.tsv", sep="\t").to_dict("records")
frozen_files = ["combination_split.tsv", "test_tasks.tsv", "cell_partitions.tsv.gz", "control_split_support.tsv", "eval_control_cells.tsv", "train_cells.tsv.gz", "control_allocations.tsv.gz"]
protocol = {
    "task": "Predict unseen two-gene CRISPRa combinations with both component singles represented in training.",
    "prior_exposure": "Additional analysis of a public dataset previously examined in unrelated development; not a blind or external-study validation.",
    "source_cells": "GEO CellRanger-filtered cells with guide identities, using integer RNA counts from the official pertpy mirror.",
    "assignment_qc": "good_coverage == True and number_of_cells == 1, following producer single_cell definition.",
    "control_constructs": ["NegCtrl0_NegCtrl0", "NegCtrl10_NegCtrl0", "NegCtrl11_NegCtrl0"],
    "excluded_control_constructs": ["NegCtrl1_NegCtrl0"],
    "control_annotation_source": "https://github.com/thomasmaxwellnorman/Perturbseq_GI/blob/master/GI_generate_populations.ipynb",
    "condition_alias": {"RHOXF2": "RHOXF2B"},
    "double_support": {"total_cells_min": 100, "each_gemgroup_cells_min": 5, "each_component_single_cells_min": 50},
    "all_single_conditions_train": True,
    "double_split": "Sort by SHA256 rank; first55 training, remaining55 test.",
    "control_split": "Within each gemgroup sort by SHA256 rank; floor half training, remaining evaluation controls.",
    "hash_namespace": HASH_NAMESPACE,
    "hash_algorithm": "SHA256(namespace + NUL + purpose_label + NUL + identifier); lexicographic order, identifier tie-break.",
    "reference_depth_per_gemgroup": depths,
    "reference_depth_total_each_role": [8 * d for d in depths],
    "allocation_labels": [f"allocation_{i:02d}" for i in range(30)],
    "allocation_rule": "For each assignment and gemgroup sort evaluation cells by hash; split first3*max_depth into3blocks; each requested depth takes that prefix from each block.",
    "normalization": "log1p(10000 * RNA_count / whole_33694_feature_RNA_total), prior to panel subset.",
    "features": "Top2000 unique Ensembl IDs by pooled TRAIN-control population variance; tie-break Ensembl ID. Symbols retained, duplicate symbols do not cause removal.",
    "scale": "Pooled TRAIN-control population standard deviation, floored at0.1.",
    "target_aggregation": "Equal mean across8gemgroup means for each condition; no treated cells selected by effect or model outcome.",
    "independent_biological_experiments": 1,
    "statistical_scope": "Descriptive within-screen reference sensitivity; allocations and gemgroups are not independent experiments.",
    "source_files": sources,
    "frozen_membership_files": {name: file_sha(OUT / name) for name in frozen_files},
    "script_sha256": file_sha(__file__),
    "frozen_before_expression_transforms": True,
}
freeze_path = OUT / "FROZEN_TASK_PROTOCOL.json"
if freeze_path.exists():
    previous = json.loads(freeze_path.read_text())
    assert {k: v for k, v in previous.items() if k != "frozen_at_utc"} == protocol, "Existing task freeze differs. Do not silently replace it."
else:
    protocol["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(freeze_path, protocol)
print("FROZEN", freeze_path, flush=True)

# Expression-derived work begins only after the task and cell IDs are frozen.
with h5py.File(SOURCE, "r") as f:
    raw = csr_matrix((f["X"]["data"][:], f["X"]["indices"][:], f["X"]["indptr"][:]), shape=tuple(f["X"].attrs["h5sparse_shape"]))
assert np.array_equal(np.asarray(raw.sum(axis=1)).ravel()[cells.source_row], cells.rna_total_counts)
source_totals = np.asarray(raw.sum(axis=1, dtype=np.float64)).ravel()


def transformed(rows, panel=None):
    rows = np.asarray(rows, dtype=int)
    matrix = raw[rows].astype(np.float64)
    repeats = np.diff(matrix.indptr)
    matrix.data *= np.repeat(10000.0 / source_totals[rows], repeats)
    np.log1p(matrix.data, out=matrix.data)
    if panel is not None:
        matrix = matrix[:, panel]
    return matrix


reference = transformed(train_controls.source_row)
means = np.asarray(reference.mean(axis=0)).ravel()
second = np.asarray(reference.power(2).mean(axis=0)).ravel()
variance = np.maximum(second - means ** 2, 0)
features = pd.read_csv(META / "feature_manifest.tsv", sep="\t", keep_default_na=False)
panel = np.lexsort((features.ensembl_id.to_numpy(), -variance))[:2000]
genes = features.iloc[panel].copy().reset_index(drop=True)
genes.insert(0, "panel_index", np.arange(len(genes)))
genes["train_control_mean"] = means[panel]
genes["train_control_variance"] = variance[panel]
genes["train_control_sd"] = np.sqrt(variance[panel])
genes["scale"] = np.maximum(genes.train_control_sd, 0.1)
genes.to_csv(OUT / "gene_panel.tsv", sep="\t", index=False, float_format="%.17g")
np.save(OUT / "train_control_mean.npy", means[panel], allow_pickle=False)
np.save(OUT / "scales.npy", genes.scale.to_numpy(), allow_pickle=False)
del reference
all_selected = cells[cells.partition.isin(["TRAIN", "TEST", "EVAL_CONTROL"])].copy()
matrix = transformed(all_selected.source_row, panel).astype(np.float32).toarray()
assert np.isfinite(matrix).all() and (matrix >= 0).all()
positions = pd.Series(np.arange(len(all_selected)), index=all_selected.cell_id)
var = genes.set_index("ensembl_id").copy()
var.index.name = "ensembl_id"


def export_h5ad(subset, name):
    loc = positions.loc[subset.cell_id].to_numpy()
    obs = subset[["cell_id", "source_row", "condition", "original_condition", "kind", "gemgroup", "partition"]].set_index("cell_id").copy()
    obs["gemgroup"] = obs.gemgroup.astype(str).astype("category")
    for col in ["condition", "original_condition", "kind", "partition"]:
        obs[col] = obs[col].astype("category")
    obj = ad.AnnData(X=matrix[loc].copy(), obs=obs, var=var.copy())
    obj.uns["normalization"] = protocol["normalization"]
    obj.uns["task_protocol_sha256"] = file_sha(freeze_path)
    obj.write_h5ad(OUT / name, compression="gzip", compression_opts=1)
    print("EXPORTED", name, obj.shape, flush=True)
    return loc


train_loc = export_h5ad(train, "train.h5ad")
eval_loc = export_h5ad(eval_controls, "eval_controls.h5ad")
np.save(OUT / "eval_controls.npy", matrix[eval_loc], allow_pickle=False)


def condition_means(subset, ordered_conditions, name):
    result = np.empty((len(ordered_conditions), 8, len(genes)), dtype=np.float64)
    counts = np.zeros((len(ordered_conditions), 8), dtype=np.int64)
    for i, condition in enumerate(ordered_conditions):
        for gem in range(1, 9):
            group = subset[subset.condition.eq(condition) & subset.gemgroup.eq(gem)]
            assert len(group) > 0, (condition, gem)
            loc = positions.loc[group.cell_id].to_numpy()
            result[i, gem - 1] = matrix[loc].mean(axis=0, dtype=np.float64)
            counts[i, gem - 1] = len(group)
    np.savez_compressed(OUT / name, conditions=np.array(ordered_conditions), gemgroups=np.arange(1, 9), mean_by_gemgroup=result, mean_equal_gemgroup=result.mean(axis=1), n_cells_by_gemgroup=counts)
    return counts


train_labels = sorted(train.condition.unique())
train_counts = condition_means(train, train_labels, "training_means.npz")
test_counts = condition_means(test, test_tasks.condition.to_list(), "test_means.npz")
training_tasks = train.groupby(["condition", "kind"]).size().rename("n_cells").reset_index().sort_values("condition")
training_tasks.to_csv(OUT / "training_tasks.tsv", sep="\t", index=False)
expected_train = ad.read_h5ad(OUT / "train.h5ad", backed="r")
expected_eval = ad.read_h5ad(OUT / "eval_controls.h5ad", backed="r")
assert np.array_equal(expected_train.obs_names.to_numpy(), train.cell_id.to_numpy())
assert np.array_equal(expected_eval.obs_names.to_numpy(), eval_controls.cell_id.to_numpy())
assert np.array_equal(expected_train.var_names.to_numpy(), genes.ensembl_id.to_numpy())
assert np.array_equal(expected_eval.var_names.to_numpy(), genes.ensembl_id.to_numpy())
expected_train.file.close()
expected_eval.file.close()
pass_result = {
    "status": "PASS",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "freeze_sha256": file_sha(freeze_path),
    "no_test_conditions_in_training": True,
    "training_evaluation_control_ids_disjoint": True,
    "all_reference_controls_follow_author_rule": True,
    "features_and_scales_use_only_training_controls": True,
    "all_training_conditions_support_all8gemgroups": bool((train_counts > 0).all()),
    "all_test_conditions_support_all8gemgroups": bool((test_counts > 0).all()),
    "whole_library_normalization": True,
    "train_cells": len(train),
    "training_controls": len(train_controls),
    "training_single_conditions": 105,
    "training_double_conditions": 55,
    "test_cells": len(test),
    "test_double_conditions": 55,
    "evaluation_controls": len(eval_controls),
    "excluded_author_control_cells": int(cells.author_excluded_control_construct.sum()),
    "reference_depth_per_gemgroup": depths,
    "features": len(genes),
    "arrays_finite": True,
    "files": {p.name: {"bytes": p.stat().st_size, "sha256": file_sha(p)} for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in ["INPUT_AUDIT_PASS.json", "prep_run.log"]},
}
write_json(OUT / "INPUT_AUDIT_PASS.json", pass_result)
print(json.dumps({k: v for k, v in pass_result.items() if k != "files"}, indent=2), flush=True)
