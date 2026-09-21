#!/usr/bin/env python3
"""Score a frozen direct-effect predictor against the original Norman states."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

PATTERNS = ("S", "M", "P", "O", "D")
STATE_MODELS = ("additive", "compositional_ridge", "CPA_0.8.8")
TUPLES = tuple(itertools.product(range(3), repeat=3))
TOL = 1e-12


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def pattern(t):
    o, p, m = t
    if o == p == m:
        return "S"
    if o == p:
        return "M"
    if o == m:
        return "P"
    if p == m:
        return "O"
    return "D"


PATTERN_IDS = [[i for i, t in enumerate(TUPLES) if pattern(t) == p] for p in PATTERNS]


def effect_utilities(effect, treated, controls, scales):
    """Score all tuples before averaging. Effect has no reference-block axis."""
    f, y, b, s = [np.asarray(x, dtype=np.float64) for x in (effect, treated, controls, scales)]
    if f.ndim != 1 or y.shape != f.shape or b.shape[-2:] != (3, len(f)):
        raise ValueError("Effect, target or control axes differ")
    if s.shape != f.shape or np.any(s <= 0) or not all(np.isfinite(x).all() for x in (f, y, b, s)):
        raise ValueError("Nonfinite arrays or invalid scales")
    # Each tuple uses the same effect vector and the designated observation block.
    residual = (f - y + b) / s
    by_observation = -np.mean(residual ** 2, axis=-1)
    atomic = np.stack([by_observation[..., o] for o, _, _ in TUPLES], axis=-1)
    utility = np.stack([atomic[..., ids].mean(axis=-1) for ids in PATTERN_IDS], axis=-1)
    return atomic, utility


def reference_distance(controls, scales):
    b = np.asarray(controls, dtype=np.float64) / scales
    return sum(np.mean((b[..., i, :] - b[..., j, :]) ** 2, axis=-1)
               for i, j in itertools.permutations(range(3), 2)) / 6


def annotate(frame):
    frame = frame.copy()
    frame["d_D_predicted"] = frame.d_S + frame.V
    frame["identity_residual"] = frame.d_D - frame.d_D_predicted
    frame["condition_strict_raw"] = (frame.d_S > -frame.V) & (frame.d_S < 0)
    frame["expected_crossing"] = (frame.d_S < -TOL) & (frame.d_D_predicted > TOL)
    frame["observed_crossing"] = (frame.d_S < -TOL) & (frame.d_D > TOL)
    frame["S_tie"] = frame.d_S.abs() <= TOL
    frame["D_tie"] = frame.d_D.abs() <= TOL
    frame["classification"] = np.select(
        [frame.S_tie | frame.D_tie, frame.observed_crossing,
         (frame.d_S > TOL) & (frame.d_D > TOL), (frame.d_S < -TOL) & (frame.d_D < -TOL)],
        ["numerical_tie", "state_to_effect", "effect_preferred_both", "state_preferred_both"],
        default="unexpected_reverse_crossing")
    if frame.identity_residual.abs().max() > 1e-9:
        raise AssertionError("d_D = d_S + V failed")
    if not frame.expected_crossing.equals(frame.observed_crossing):
        raise AssertionError("Condition and observed crossing differ at the declared tolerance")
    if (frame.classification == "unexpected_reverse_crossing").any():
        raise AssertionError("Positive reference penalty gave a reverse crossing")
    return frame


def make_tables(effect_utility, state_utility, v, tasks, depths):
    ds = effect_utility[..., 0, None] - state_utility[..., :3, 0]
    dd = effect_utility[..., 4, None] - state_utility[..., :3, 4]
    records = []
    for ti, task in enumerate(tasks):
        for ai in range(ds.shape[1]):
            for di, depth in enumerate(depths):
                for mi, model in enumerate(STATE_MODELS):
                    records.append((str(task), ai, int(depth), int(depth)*8,
                                    "direct_effect_ridge", model,
                                    ds[ti, ai, di, mi], v[ai, di], dd[ti, ai, di, mi]))
    base = pd.DataFrame(records, columns=["condition", "allocation", "depth_per_gemgroup",
        "cells_per_role", "effect_model", "state_model", "d_S", "V", "d_D"])
    base = annotate(base)
    common = ["depth_per_gemgroup", "cells_per_role", "effect_model", "state_model"]
    tables = {"task_allocation_pairs": base}
    for name, group, size_name in (
        ("task_mean_pairs", ["condition"] + common, "n_allocations"),
        ("allocation_mean_pairs", ["allocation"] + common, "n_tasks"),
        ("overall_mean_pairs", common, "n_task_allocation_pairs"),
    ):
        grouped = base.groupby(group, sort=False)
        avg = grouped[["d_S", "V", "d_D"]].mean().reset_index()
        counts = grouped.agg(**{size_name:("d_S", "size")},
                             crossing_unit_count=("observed_crossing", "sum")).reset_index()
        tables[name] = annotate(avg.merge(counts, on=group, validate="one_to_one"))
    return tables, ds, dd


def score(args):
    protocol = json.loads(args.protocol.read_text())
    assert sha(args.protocol) == args.protocol.with_suffix(".json.sha256").read_text().split()[0]
    assert protocol["model"]["id"] == "direct_effect_ridge"
    assert protocol["evaluation"]["state_models"] == list(STATE_MODELS)
    assert protocol["evaluation"]["tie_tolerance"] == TOL
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Refusing to overwrite an evaluated model")
    args.output.mkdir(parents=True, exist_ok=True)

    fit_receipt = json.loads((args.fit / "fit_receipt.json").read_text())
    # Fitting supplies these standardized names and binds the frozen prediction bytes.
    pred_path = args.fit / "test_effect_predictions.npz"
    assert fit_receipt["protocol_sha256"] == sha(args.protocol)
    assert fit_receipt["test_expression_used"] is False
    assert fit_receipt["output_sha256"][pred_path.name] == sha(pred_path)
    pred = np.load(pred_path, allow_pickle=False)
    tasks, effects = pred["conditions"], pred["effects"]
    if effects.shape != (55, 2000):
        raise ValueError("Expected one fixed 2000-gene effect vector per test task")

    norman = args.original
    for filename, digest in protocol["frozen_input_sha256"].items():
        assert sha(norman / "inputs" / filename) == digest, filename
    source_receipt = json.loads((norman / "results/score_report.json").read_text())
    for name in ("utilities.npz", "control_block_means.npy"):
        assert sha(norman / "results" / name) == source_receipt["output_sha256"][name], name
    assert source_receipt["status"] == "PASS"
    assert source_receipt["protocol_sha256"] == protocol["existing_scoring_protocol_sha256"]
    old = np.load(norman / "results/utilities.npz", allow_pickle=False)
    obs = np.load(norman / "inputs/test_means.npz", allow_pickle=False)
    scales = np.load(norman / "inputs/scales.npy", allow_pickle=False)
    genes = pd.read_csv(norman / "inputs/gene_panel.tsv", sep="\t")
    assert np.array_equal(pred["ensembl_ids"], genes.ensembl_id.to_numpy(dtype=str))
    assert np.array_equal(pred["gene_symbols"], genes.gene_symbol.to_numpy(dtype=str))
    np.testing.assert_allclose(effects / scales, pred["standardized_effects"], rtol=0, atol=1e-12)
    controls = np.load(norman / "results/control_block_means.npy", allow_pickle=False)
    assert np.array_equal(tasks, old["tasks"]) and np.array_equal(tasks, obs["conditions"])
    assert tuple(old["models"][:3]) == STATE_MODELS
    assert tuple(old["patterns"]) == PATTERNS
    assert np.array_equal(old["role_tuples"], np.asarray(TUPLES))
    assert old["depths"].tolist() == protocol["evaluation"]["depth_per_gemgroup"]
    assert controls.shape == (30, 6, 3, 2000)
    for pi, ids in enumerate(PATTERN_IDS):
        np.testing.assert_allclose(old["atomic_utility"][..., ids].mean(axis=-1),
                                   old["utility"][..., pi], rtol=0, atol=1e-12)

    v = reference_distance(controls, scales)
    np.testing.assert_allclose(v, old["V"], rtol=0, atol=1e-12)
    state_identity = old["utility"][..., :3, 4] - old["utility"][..., :3, 0] + v[..., None]
    assert np.max(np.abs(state_identity)) < 1e-9
    atomic = np.empty((55, 30, 6, 27))
    utilities = np.empty((55, 30, 6, 5))
    for ti in range(len(tasks)):
        atomic[ti], utilities[ti] = effect_utilities(
            effects[ti], obs["mean_equal_gemgroup"][ti], controls, scales)
    spread = float(np.max(np.ptp(utilities, axis=-1)))
    assert spread < 1e-12
    tables, ds, dd = make_tables(utilities, old["utility"], v, tasks, old["depths"])
    expected_rows = {"task_allocation_pairs":29700, "task_mean_pairs":990,
                     "allocation_mean_pairs":540, "overall_mean_pairs":18}
    for name, table in tables.items():
        assert len(table) == expected_rows[name]
        suffix = ".tsv.gz" if name == "task_allocation_pairs" else ".tsv"
        compression = {"method":"gzip", "mtime":0} if suffix.endswith("gz") else None
        table.to_csv(args.output / (name+suffix), sep="\t", index=False, compression=compression)
    np.savez_compressed(args.output / "effect_utilities.npz", conditions=tasks,
        patterns=np.asarray(PATTERNS), state_models=np.asarray(STATE_MODELS),
        depths=old["depths"], role_tuples=old["role_tuples"],
        effect_utility=utilities, effect_atomic_utility=atomic, V=v, d_S=ds, d_D=dd)
    summary = tables["overall_mean_pairs"].to_dict("records")
    report = dict(status="PASS_DIRECT_EFFECT_COMPARISONS", protocol_sha256=sha(args.protocol),
        fit_receipt_sha256=sha(args.fit/"fit_receipt.json"),
        fixed_effect_predictions_sha256=sha(pred_path),
        original_utilities_sha256=sha(norman/"results/utilities.npz"),
        original_controls_sha256=sha(norman/"results/control_block_means.npy"),
        treated_targets_sha256=sha(norman/"inputs/test_means.npz"),
        effect_reference_dependence="none: one fixed vector per task, no reference input",
        pair_orientation="effect-minus-state", all_pairs_retained=True,
        all_conditions_match_observed_crossings=True,
        independent_pooled_experiments=1, biological_significance_test_performed=False,
        effect_pattern_spread_max_abs=spread,
        state_identity_max_abs=float(np.max(np.abs(state_identity))),
        table_rows={name:len(table) for name,table in tables.items()},
        identity_max_abs={name:float(table.identity_residual.abs().max()) for name,table in tables.items()},
        crossing_counts={name:int(table.observed_crossing.sum()) for name,table in tables.items()},
        overall_pair_results=summary, script_sha256=sha(__file__),
        output_sha256={p.name:sha(p) for p in sorted(args.output.iterdir()) if p.is_file()})
    write_json(args.output/"score_receipt.json", report)
    print(json.dumps({key:report[key] for key in ("status","table_rows","identity_max_abs","crossing_counts")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True,
                        help="Original Norman source component with inputs/ and results/")
    parser.add_argument("--output", type=Path, required=True)
    score(parser.parse_args())


if __name__ == "__main__":
    main()
