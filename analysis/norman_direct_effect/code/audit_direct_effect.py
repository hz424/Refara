#!/usr/bin/env python3
"""Independent replay of training-only selection and Norman effect/state scores.

This script imports neither production fitting nor scoring code. It uses SVD
for the 30 inner fits and final refit; reconstructs reference means from cell
membership; recomputes every added atomic score; and compares against the
original frozen state atomics. The prior CPA fit/prediction audit remains the
authority for the unchanged state predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

PATTERNS = ("S", "M", "P", "O", "D")
STATES = ("additive", "compositional_ridge", "CPA_0.8.8")
DEPTHS = (8, 16, 32, 64, 128, 135)
ALPHAS = (0., .01, .1, 1., 10., 100.)
TUPLES = tuple(itertools.product(range(3), repeat=3))
TIE = 1e-12


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pattern_indices():
    # Explicit mathematical assignment sets, without production classification.
    patterns = {
        "S": {(i, i, i) for i in range(3)},
        "M": {(j, j, i) for i in range(3) for j in range(3) if i != j},
        "P": {(j, i, j) for i in range(3) for j in range(3) if i != j},
        "O": {(i, j, j) for i in range(3) for j in range(3) if i != j},
        "D": set(itertools.permutations(range(3))),
    }
    assert sum(len(x) for x in patterns.values()) == 27
    assert len(set.union(*patterns.values())) == 27
    for assignments in patterns.values():
        for role in range(3):
            assert [sum(t[role] == i for t in assignments) for i in range(3)] == [len(assignments)//3]*3
    return [[j for j, t in enumerate(TUPLES) if t in patterns[p]] for p in PATTERNS]


IDS = pattern_indices()


class Audit:
    def __init__(self):
        self.errors = {}

    def close(self, name, actual, expected, tolerance=1e-10):
        actual, expected = np.asarray(actual), np.asarray(expected)
        assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
        assert np.isfinite(actual).all() and np.isfinite(expected).all(), name
        error = float(np.max(np.abs(actual - expected))) if actual.size else 0.
        self.errors[name] = error
        assert error <= tolerance, (name, error, tolerance)


def design(conditions, singles):
    axis = {s: i for i, s in enumerate(singles)}
    x = np.zeros((len(conditions), len(singles)))
    for i, condition in enumerate(conditions):
        if condition != "ctrl":
            pieces = condition.split("+")
            assert len(pieces) in (1, 2) and len(set(pieces)) == len(pieces)
            x[i, [axis[p] for p in pieces]] = 1
    return x


def svd_fit(x, y, alpha):
    left, singular, right = np.linalg.svd(x, full_matrices=False)
    assert singular.min() > 1e-10
    return (right.T * (singular / (singular**2 + alpha))) @ (left.T @ y)


def scalar_geometry_checks():
    examples = 0
    for controls, states in [
        ([-1., 0., 1.], [0., 0., 0.]),
        ([-1., 0., 1.], [-1., 3., .4]),
        ([0., 0., 0.], [-1., 3., .4]),
    ]:
        for effect in (0., .5, 4.):
            e = np.array([-(effect + controls[o])**2 for o, p, m in TUPLES])
            state = np.array([-(states[m]+controls[o]-controls[p])**2 for o, p, m in TUPLES])
            us = np.array([math.fsum(state[ids])/len(ids) for ids in IDS])
            ue = np.array([math.fsum(e[ids])/len(ids) for ids in IDS])
            v = math.fsum((controls[i]-controls[j])**2 for i in range(3) for j in range(i))/3
            assert np.ptp(ue) < 1e-12
            assert abs(us[4]-us[0]+v) < 1e-12
            ds, dd = ue[0]-us[0], ue[4]-us[4]
            assert abs(dd-ds-v) < 1e-12
            assert ((-v < ds < 0) == (ds < 0 < dd))
            examples += 1
    return examples


def check_hashes(original, fit, results, protocol, audit):
    for name, expected in protocol["frozen_input_sha256"].items():
        assert sha(original / "inputs" / name) == expected, name
    manifest = {}
    for line in (original / "MANIFEST.sha256").read_text().splitlines():
        digest, name = line.split(None, 1)
        manifest[name.strip().lstrip("*")] = digest
    needed = ["inputs/training_means.npz", "inputs/test_means.npz", "inputs/scales.npy",
              "inputs/eval_controls.npy", "inputs/eval_control_cells.tsv", "inputs/control_allocations.tsv.gz",
              "results/control_block_means.npy", "results/utilities.npz", "results/score_report.json"]
    for name in needed:
        assert sha(original / name) == manifest[name], name
    receipts = {}
    for directory, name in ((fit, "selection_receipt.json"), (fit, "fit_receipt.json"), (results, "score_receipt.json")):
        record = json.loads((directory / name).read_text())
        for filename, expected in record["output_sha256"].items():
            assert sha(directory / filename) == expected, (name, filename)
        receipts[name] = record
    assert receipts["fit_receipt.json"]["selection_receipt_sha256"] == sha(fit/"selection_receipt.json")
    assert receipts["score_receipt.json"]["fit_receipt_sha256"] == sha(fit/"fit_receipt.json")
    return receipts


def audit_fit(original, fit, protocol, receipts, audit):
    train = np.load(original / "inputs/training_means.npz", allow_pickle=False)
    conditions = train["conditions"].tolist()
    assert len(conditions) == len(set(conditions)) == 161
    singles = sorted(c for c in conditions if c != "ctrl" and "+" not in c)
    doubles = sorted(c for c in conditions if "+" in c)
    assert len(singles) == 105 and len(doubles) == 55
    means = train["mean_by_gemgroup"].mean(axis=1)
    audit.close("training_equal_gemgroup_means", means, train["mean_equal_gemgroup"], 0.)
    baseline = means[conditions.index("ctrl")]
    scales = np.load(original / "inputs/scales.npy", allow_pickle=False)
    x = design(conditions, singles)
    y = (means-baseline)/scales
    assert (y[conditions.index("ctrl")] == 0).all()
    namespace = protocol["validation"]["hash_namespace"]
    hashes = {c: hashlib.sha256((namespace+"\0"+c).encode()).hexdigest() for c in doubles}
    order = sorted(doubles, key=lambda c: (hashes[c], c))
    assigned = {c: r % 5 for r, c in enumerate(order)}
    membership = pd.read_csv(fit/"fold_membership.tsv", sep="\t", keep_default_na=False)
    assert membership.condition.tolist() == conditions
    for row in membership.to_dict("records"):
        c = row["condition"]
        assert row["validation_fold"] == assigned.get(c, -1)
        assert row["condition_order"] == (0 if c == "ctrl" else len(c.split("+")))
        assert row["hash_rank"] == (order.index(c) if c in assigned else -1)
        assert row["assignment_sha256"] == hashes.get(c, "")
        assert row["fold_role"] == ("out_of_fold_double" if c in assigned else "training_in_every_fold")

    oof = np.empty((6, 55, 2000))
    cv_rows, training_rows, fold_rows = [], [], []
    for fold in range(5):
        validation = np.array([i for i, c in enumerate(conditions) if assigned.get(c, -1) == fold])
        fitting = np.array([i for i in range(161) if i not in set(validation)])
        assert len(validation) == 11 and len(fitting) == 150
        assert set(conditions[i] for i in fitting).issuperset(singles+['ctrl'])
        assert set(fitting).isdisjoint(validation)
        # SVD is deliberately different from the production normal-equation solve.
        u, singular, vt = np.linalg.svd(x[fitting], full_matrices=False)
        assert singular.min() > .99
        projected = u.T @ y[fitting]
        for ai, alpha in enumerate(ALPHAS):
            coefficient = (vt.T*(singular/(singular**2+alpha))) @ projected
            vpred = x[validation] @ coefficient
            verrors = np.einsum('ij,ij->i', vpred-y[validation], vpred-y[validation])/2000
            tpred = x[fitting] @ coefficient
            terrors = np.einsum('ij,ij->i', tpred-y[fitting], tpred-y[fitting])/2000
            for idx, pred, error in zip(validation, vpred, verrors):
                c = conditions[idx]
                oof[ai, doubles.index(c)] = pred
                cv_rows.append((alpha, fold, c, error))
            for idx, error in zip(fitting, terrors):
                c = conditions[idx]
                training_rows.append((alpha, fold, c, 0 if c == "ctrl" else len(c.split("+")), error))
            fold_rows.append((alpha, fold, 150, 11, terrors.mean(), verrors.mean()))

    stored_oof = np.load(fit/"cv_oof_predictions.npz", allow_pickle=False)
    assert stored_oof["conditions"].tolist() == doubles
    assert stored_oof["alphas"].tolist() == list(ALPHAS)
    audit.close("all_660000_SVD_validation_predictions", stored_oof["standardized_effects"], oof)
    expected_cv = pd.DataFrame(cv_rows, columns=["alpha", "fold", "condition", "standardized_mse"])
    expected_training = pd.DataFrame(training_rows, columns=["alpha", "fold", "condition", "condition_order", "standardized_mse"])
    expected_fold = pd.DataFrame(fold_rows, columns=["alpha", "fold", "n_train_conditions", "n_validation_conditions", "training_mse", "validation_mse"])
    for name, expected, keys in (("cv_condition_errors", expected_cv, ["alpha", "fold", "condition"]),
                                 ("cv_training_errors", expected_training, ["alpha", "fold", "condition"]),
                                 ("cv_fold_summary", expected_fold, ["alpha", "fold"])):
        got = pd.read_csv(fit/(name+".tsv"), sep="\t").sort_values(keys).reset_index(drop=True)
        expected = expected.sort_values(keys).reset_index(drop=True)
        assert got[keys].equals(expected[keys]), name
        for col in expected.columns.difference(keys):
            audit.close(name+"_"+col, got[col], expected[col])
    cv_loss = expected_cv.groupby("alpha", sort=True).standardized_mse.mean().reindex(ALPHAS).to_numpy()
    chosen = max(a for a, e in zip(ALPHAS, cv_loss) if abs(e-cv_loss.min()) <= TIE)
    alpha_summary = pd.read_csv(fit/"cv_alpha_summary.tsv", sep="\t").set_index("alpha").loc[list(ALPHAS)]
    audit.close("all_six_validation_losses", alpha_summary.validation_mse.to_numpy(), cv_loss)
    assert alpha_summary.n_validation_conditions.tolist() == [55]*6
    assert alpha_summary.selected.tolist() == [a == chosen for a in ALPHAS]
    assert alpha_summary.within_tie_tolerance.tolist() == [abs(e-cv_loss.min()) <= TIE for e in cv_loss]
    selection = receipts["selection_receipt.json"]
    receipt = receipts["fit_receipt.json"]
    assert selection["selected_alpha"] == receipt["selected_alpha"] == chosen
    assert selection["test_task_labels_read"] is False
    assert set(selection["input_sha256"]) == {"training_means.npz", "gene_panel.tsv", "scales.npy"}
    assert receipt["native_output"] == "effect" and receipt["has_intercept"] is False
    assert receipt["reference_input_arguments"] == []
    for record in (selection, receipt):
        assert record["test_expression_used"] is False and record["evaluation_controls_used"] is False
        assert record["alpha_grid_expanded"] is False
    events = receipt["input_access_events"]
    selection_position = next(i for i, e in enumerate(events) if e.get("event") == "selection_receipt_written")
    refit_position = next(i for i, e in enumerate(events) if e.get("event") == "all_training_refit_checkpoint_written")
    label_position = next(i for i, e in enumerate(events) if e.get("file") == "test_tasks.tsv")
    assert selection_position < refit_position < label_position
    assert set(receipt["input_sha256"]) == {"training_means.npz", "gene_panel.tsv", "scales.npy", "test_tasks.tsv"}

    coefficient = svd_fit(x, y, chosen)
    checkpoint = np.load(fit/"direct_effect_checkpoint.npz", allow_pickle=False)
    audit.close("refit_standardized_coefficients", checkpoint["standardized_coefficients"], coefficient)
    audit.close("refit_original_scale_coefficients", checkpoint["effect_coefficients"], coefficient*scales)
    audit.close("refit_baseline", checkpoint["training_control_baseline"], baseline, 0.)
    audit.close("refit_scales", checkpoint["scales"], scales, 0.)
    audit.close("refit_training_design", checkpoint["training_design"], x, 0.)
    assert checkpoint["feature_names"].tolist() == singles
    assert checkpoint["training_conditions"].tolist() == conditions
    training_predictions = x @ coefficient
    training_errors = np.mean((training_predictions-y)**2, axis=1)
    final_training = pd.read_csv(fit/"refit_training_errors.tsv", sep="\t").set_index("condition").loc[conditions]
    audit.close("refit_training_errors", final_training.standardized_mse.to_numpy(), training_errors)
    pred = np.load(fit/"test_effect_predictions.npz", allow_pickle=False)
    test = pd.read_csv(original/"inputs/test_tasks.tsv", sep="\t").condition.tolist()
    assert len(test) == len(set(test)) == 55 and set(test).isdisjoint(conditions)
    assert pred["conditions"].tolist() == test
    tx = design(test, singles)
    expected_standardized = tx @ coefficient
    audit.close("all_110000_fixed_standardized_test_effects", pred["standardized_effects"], expected_standardized)
    audit.close("all_110000_fixed_original_scale_test_effects", pred["effects"], expected_standardized*scales)
    audit.close("test_condition_design", pred["input_design"], tx, 0.)
    assert pred["effects"].shape == (55, 2000), "Prediction has unexpected reference-dependent axes"
    assert not np.any(design(["ctrl"], singles) @ coefficient)
    return {"selected_alpha": chosen, "all_six_validation_mse": cv_loss.tolist(),
            "training_conditions": 161, "validation_doubles": 55, "inner_fits_replayed": 30,
            "test_expression_or_evaluation_controls_in_selection": False,
            "selection_independent_SVD_replay": True, "fixed_test_effect_vectors": 55}


def derive_controls(original, audit):
    controls = np.load(original/"inputs/eval_controls.npy", mmap_mode="r")
    cells = pd.read_csv(original/"inputs/eval_control_cells.tsv", sep="\t", dtype=str)
    membership = pd.read_csv(original/"inputs/control_allocations.tsv.gz", sep="\t", dtype={"gemgroup": str, "cell_id": str})
    assert len(membership) == 30*8*3*135
    assert not membership.duplicated(["assignment", "eval_control_index"]).any()
    assert not membership.duplicated(["assignment", "gemgroup", "block", "within_block_index"]).any()
    assert set(membership.assignment) == set(range(30)) and set(membership.block) == {0, 1, 2}
    assert set(membership.gemgroup) == set(cells.gemgroup) and len(set(cells.gemgroup)) == 8
    ix = membership.eval_control_index.to_numpy()
    assert np.array_equal(cells.cell_id.to_numpy()[ix], membership.cell_id.to_numpy())
    assert np.array_equal(cells.gemgroup.to_numpy()[ix], membership.gemgroup.to_numpy())
    b = np.empty((30, 6, 3, 2000))
    for allocation in range(30):
        for block in range(3):
            member = membership[(membership.assignment == allocation)&(membership.block == block)]
            for _, group in member.groupby("gemgroup"):
                assert sorted(group.within_block_index) == list(range(135))
            for di, depth in enumerate(DEPTHS):
                select = member[member.within_block_index < depth]
                assert len(select) == 8*depth and select.groupby("gemgroup").size().tolist() == [depth]*8
                # Equal numbers from each capture partition make this directly
                # pooled selected-cell mean the specified equal-partition mean.
                b[allocation, di, block] = controls[select.eval_control_index.to_numpy()].mean(axis=0, dtype=np.float64)
    audit.close("all_1080000_reference_mean_coordinates", np.load(original/"results/control_block_means.npy"), b)
    return b


def compare_table(path, expected, keys, audit):
    got = pd.read_csv(path, sep="\t")
    assert len(got) == len(expected) and not got.duplicated(keys).any(), path.name
    got = got.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    assert got[keys].equals(expected[keys]), path.name
    for col in expected.columns.difference(keys):
        audit.close(path.stem+"_"+col, got[col], expected[col])
    ds, dd, v = got.d_S.to_numpy(), got.d_D.to_numpy(), got.V.to_numpy()
    assert (v >= 0).all()
    audit.close(path.stem+"_identity", dd-ds, v)
    audit.close(path.stem+"_reported_identity_residual", got.identity_residual.to_numpy(), dd-(ds+v), 1e-14)
    audit.close(path.stem+"_predicted_d_D", got.d_D_predicted.to_numpy(), ds+v, 1e-14)
    raw = (ds > -v)&(ds < 0)
    expect_cross = (ds < -TIE)&(ds+v > TIE)
    cross = (ds < -TIE)&(dd > TIE)
    s_tie, d_tie = np.abs(ds) <= TIE, np.abs(dd) <= TIE
    assert np.array_equal(expect_cross, cross), path.name
    for col, values in (("condition_strict_raw", raw), ("expected_crossing", expect_cross),
                        ("observed_crossing", cross), ("S_tie", s_tie), ("D_tie", d_tie)):
        assert np.array_equal(got[col].to_numpy(), values), (path.name, col)
    classifications = []
    for s, d, st, dt, flip in zip(ds, dd, s_tie, d_tie, cross):
        if st or dt:
            classifications.append("numerical_tie")
        elif flip:
            classifications.append("state_to_effect")
        elif s > 0 and d > 0:
            classifications.append("effect_preferred_both")
        elif s < 0 and d < 0:
            classifications.append("state_preferred_both")
        else:
            raise AssertionError("Unexplained positive-to-negative crossing")
    assert got.classification.tolist() == classifications, path.name
    return {"rows": len(got), "crossings": int(cross.sum()), "ties": int((s_tie|d_tie).sum()),
            "identity_max_abs": float(np.max(np.abs(dd-ds-v)))}


def audit_scores(original, fit, results, audit):
    b = derive_controls(original, audit)
    scales = np.load(original/"inputs/scales.npy", allow_pickle=False)
    pred = np.load(fit/"test_effect_predictions.npz", allow_pickle=False)
    obs = np.load(original/"inputs/test_means.npz", allow_pickle=False)
    old = np.load(original/"results/utilities.npz", allow_pickle=False)
    new = np.load(results/"effect_utilities.npz", allow_pickle=False)
    tasks = pred["conditions"].tolist()
    fixed_effects = pred["effects"]
    assert tasks == old["tasks"].tolist() == obs["conditions"].tolist() == new["conditions"].tolist()
    assert old["models"][:3].tolist() == list(STATES) == new["state_models"].tolist()
    assert old["patterns"].tolist() == list(PATTERNS) == new["patterns"].tolist()
    assert old["depths"].tolist() == list(DEPTHS) == new["depths"].tolist()
    assert np.array_equal(old["role_tuples"], TUPLES) and np.array_equal(new["role_tuples"], TUPLES)
    y = obs["mean_by_gemgroup"].mean(axis=1)
    audit.close("observed_equal_gemgroup_means", y, obs["mean_equal_gemgroup"], 0.)
    old_states = old["atomic_utility"][..., :3, :]
    state_pattern = np.stack([old_states[..., ids].mean(axis=-1) for ids in IDS], axis=-1)
    audit.close("all_original_state_pattern_averages", state_pattern, old["utility"][..., :3, :], 1e-12)
    standardized_b = b/scales
    # Three unordered distances divided by3, algebraically the original six
    # ordered distances divided by6, with a separately implemented reduction.
    distances = [np.einsum('...g,...g->...', standardized_b[..., i, :]-standardized_b[..., j, :],
                          standardized_b[..., i, :]-standardized_b[..., j, :])/2000
                 for i in range(3) for j in range(i)]
    v = np.stack(distances).mean(axis=0)
    audit.close("all_reference_penalties_old", old["V"], v, 1e-12)
    audit.close("all_reference_penalties_new", new["V"], v, 1e-12)
    audit.close("all_state_D_minus_S_penalties", state_pattern[..., 4]-state_pattern[..., 0],
                -np.broadcast_to(v[..., None], (55, 30, 6, 3)), 1e-10)
    atomic = np.empty((55, 30, 6, 27))
    for ti in range(55):
        residual = fixed_effects[ti]/scales - y[ti]/scales + standardized_b
        by_obs = -np.einsum('...g,...g->...', residual, residual)/2000
        for j, (o, _, _) in enumerate(TUPLES):
            atomic[ti, ..., j] = by_obs[..., o]
    effect_pattern = np.stack([atomic[..., ids].mean(axis=-1) for ids in IDS], axis=-1)
    audit.close("all_267300_effect_atomic_utilities", new["effect_atomic_utility"], atomic)
    audit.close("all_49500_effect_pattern_utilities", new["effect_utility"], effect_pattern)
    audit.close("effect_pattern_invariance", effect_pattern, np.repeat(effect_pattern[..., :1], 5, axis=-1), 1e-12)
    # Independent Python-scalar gene summation checks include all27tuples for
    # fixed endpoints of each axis, not selected using result size or sign.
    scalar_checks = 0
    for ti, ai, di in itertools.product((0, 27, 54), (0, 29), (0, 5)):
        for j, (o, _, _) in enumerate(TUPLES):
            value = -math.fsum(((float(fixed_effects[ti,g])-float(y[ti,g])+float(b[ai,di,o,g]))/float(scales[g]))**2
                               for g in range(2000))/2000
            assert abs(value-atomic[ti,ai,di,j]) < 1e-10
            scalar_checks += 1
    ds = effect_pattern[..., 0, None]-state_pattern[..., 0]
    dd = effect_pattern[..., 4, None]-state_pattern[..., 4]
    audit.close("all_29700_saved_S_margins", new["d_S"], ds)
    audit.close("all_29700_saved_D_margins", new["d_D"], dd)
    rows = []
    for ti, ai, di, mi in itertools.product(range(55), range(30), range(6), range(3)):
        rows.append((tasks[ti], ai, DEPTHS[di], DEPTHS[di]*8, "direct_effect_ridge", STATES[mi],
                     ds[ti,ai,di,mi], v[ai,di], dd[ti,ai,di,mi]))
    base = pd.DataFrame(rows, columns=["condition", "allocation", "depth_per_gemgroup", "cells_per_role",
        "effect_model", "state_model", "d_S", "V", "d_D"])
    common = ["depth_per_gemgroup", "cells_per_role", "effect_model", "state_model"]
    records = {"task_allocation_pairs": compare_table(results/"task_allocation_pairs.tsv.gz", base,
                                                     ["condition", "allocation"]+common, audit)}
    base["crossing_unit"] = (base.d_S < -TIE)&(base.d_D > TIE)
    for name, keys, ncol in (("task_mean_pairs", ["condition"]+common, "n_allocations"),
                             ("allocation_mean_pairs", ["allocation"]+common, "n_tasks"),
                             ("overall_mean_pairs", common, "n_task_allocation_pairs")):
        groups = base.groupby(keys)
        expected = groups[["d_S", "V", "d_D"]].mean()
        expected[ncol] = groups.size()
        expected["crossing_unit_count"] = groups.crossing_unit.sum()
        records[name] = compare_table(results/(name+".tsv"), expected.reset_index(), keys, audit)
    assert {k: v["rows"] for k,v in records.items()} == {
        "task_allocation_pairs": 29700, "task_mean_pairs": 990,
        "allocation_mean_pairs": 540, "overall_mean_pairs": 18}
    return {"scalar_atomic_checks": scalar_checks, "all_atomic_utilities": atomic.size,
            "effect_predictions_fixed_in_all_assignments": True,
            "old_state_atomicals_reused_without_new_model_fit": True,
            "table_checks": records}


def run_audit(original, fit, results, protocol_path):
    """Return a portable validation report without writing temporary files."""
    original, fit, results, protocol_path = map(Path, (original, fit, results, protocol_path))
    synthetic = scalar_geometry_checks()
    assert sha(protocol_path) == protocol_path.with_suffix(".json.sha256").read_text().split()[0]
    protocol = json.loads(protocol_path.read_text())
    audit = Audit()
    receipts = check_hashes(original, fit, results, protocol, audit)
    print("Frozen source and output hashes verified", flush=True)
    fitting = audit_fit(original, fit, protocol, receipts, audit)
    print("All30inner fits, training-only selection and fixed prediction vectors independently verified", flush=True)
    scoring = audit_scores(original, fit, results, audit)
    report = {"status": "PASS_INDEPENDENT_EFFECT_AUDIT", "production_functions_imported": False,
        "scope": "Training-only model selection and added fixed-effect comparison; unchanged state atomics retain prior CPA validation",
        "protocol_sha256": sha(protocol_path), "script_sha256": sha(__file__),
        "fit_receipt_sha256": sha(fit/"fit_receipt.json"),
        "score_receipt_sha256": sha(results/"score_receipt.json"),
        "fixed_effect_predictions_sha256": sha(fit/"test_effect_predictions.npz"),
        "original_utilities_sha256": sha(original/"results/utilities.npz"),
        "synthetic_geometry_examples": synthetic, "selection_and_fit": fitting, "scoring": scoring,
        "maximum_numeric_error": max(audit.errors.values()), "numeric_checks_max_abs": audit.errors}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--fit", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path, help="Optional JSON report destination")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps({"status": "PASS_SCALAR_GEOMETRY_AUDIT", "examples": scalar_geometry_checks()}))
        return
    assert all((args.original, args.fit, args.results, args.protocol))
    report = run_audit(args.original, args.fit, args.results, args.protocol)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({k:report[k] for k in ("status", "maximum_numeric_error")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
