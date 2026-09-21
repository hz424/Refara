#!/usr/bin/env python3
"""Rebuild all effect/state margin tables from the bundled scored utilities."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from score_direct_effect import PATTERNS, PATTERN_IDS, STATE_MODELS, TUPLES, make_tables


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROWS = {
    "task_allocation_pairs": 29700,
    "task_mean_pairs": 990,
    "allocation_mean_pairs": 540,
    "overall_mean_pairs": 18,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare_tables(actual, expected):
    left = pd.read_csv(actual, sep="\t", float_precision="round_trip")
    right = pd.read_csv(expected, sep="\t", float_precision="round_trip")
    require(left.shape == right.shape and list(left.columns) == list(right.columns),
            "Table axes differ: " + actual.name)
    maximum = 0.0
    for column in left:
        if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(right[column]):
            a, b = left[column].to_numpy(), right[column].to_numpy()
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-12,
                                       err_msg=actual.name + ":" + column)
            if len(a):
                maximum = max(maximum, float(np.max(np.abs(a.astype(float) - b.astype(float)))))
        else:
            require(left[column].fillna("").astype(str).equals(right[column].fillna("").astype(str)),
                    "Table labels differ: " + actual.name + ":" + column)
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effect-utilities", type=Path, default=ROOT / "data/effect_utilities.npz")
    parser.add_argument("--state-utilities", type=Path,
                        default=ROOT.parent / "norman_reference/data/utilities.npz")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() or not any(args.output.iterdir()),
            "Use a new or empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = json.loads((args.reference_dir / "score_receipt.json").read_text())
    require(receipt["status"] == "PASS_DIRECT_EFFECT_COMPARISONS", "Scoring receipt did not pass")
    require(sha(args.effect_utilities) == receipt["output_sha256"]["effect_utilities.npz"],
            "Effect utility bytes differ")
    require(sha(args.state_utilities) == receipt["original_utilities_sha256"],
            "Retained state utility bytes differ")
    with np.load(args.effect_utilities, allow_pickle=False) as effect, \
            np.load(args.state_utilities, allow_pickle=False) as state:
        require(effect["effect_utility"].shape == (55, 30, 6, 5)
                and effect["effect_atomic_utility"].shape == (55, 30, 6, 27)
                and state["utility"].shape == (55, 30, 6, 4, 5), "Utility axes differ")
        require(np.array_equal(effect["conditions"], state["tasks"]), "Task order differs")
        require(np.array_equal(effect["depths"], state["depths"]), "Reference depths differ")
        require(tuple(effect["patterns"]) == tuple(state["patterns"]) == PATTERNS,
                "Pattern order differs")
        require(tuple(effect["state_models"]) == tuple(state["models"][:3]) == STATE_MODELS,
                "State model order differs")
        require(np.array_equal(effect["role_tuples"], np.asarray(TUPLES)), "Atomic tuple order differs")
        for key in ("effect_utility", "effect_atomic_utility", "V", "d_S", "d_D"):
            require(np.isfinite(effect[key]).all(), "Nonfinite scored array: " + key)
        np.testing.assert_allclose(effect["V"], state["V"], rtol=0, atol=1e-12)
        for pattern_index, tuple_ids in enumerate(PATTERN_IDS):
            np.testing.assert_allclose(effect["effect_atomic_utility"][..., tuple_ids].mean(axis=-1),
                                       effect["effect_utility"][..., pattern_index], rtol=0, atol=1e-12)
        spread = float(np.max(np.ptp(effect["effect_utility"], axis=-1)))
        require(spread <= 1e-12, "Fixed effect scores vary across balanced patterns")
        tables, ds, dd = make_tables(effect["effect_utility"], state["utility"], effect["V"],
                                     effect["conditions"], effect["depths"])
        np.testing.assert_allclose(ds, effect["d_S"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(dd, effect["d_D"], rtol=0, atol=1e-12)
    records = {}
    for name, table in tables.items():
        require(len(table) == EXPECTED_ROWS[name], "Incomplete comparison table: " + name)
        suffix = ".tsv.gz" if name == "task_allocation_pairs" else ".tsv"
        filename = name + suffix
        expected = args.reference_dir / filename
        require(sha(expected) == receipt["output_sha256"][filename], "Reference table bytes differ")
        output = args.output / filename
        compression = {"method": "gzip", "mtime": 0} if suffix.endswith("gz") else None
        table.to_csv(output, sep="\t", index=False, compression=compression)
        maximum = compare_tables(output, expected)
        records[filename] = {
            "rows": len(table), "max_abs_numeric_difference": maximum,
            "byte_identical": sha(output) == sha(expected),
            "sha256": sha(output), "reference_sha256": sha(expected),
            "identity_max_abs": float(table.identity_residual.abs().max()),
            "condition_reversal_agreement": bool(table.expected_crossing.equals(table.observed_crossing)),
        }
    report = {
        "status": "PASS_NORMAN_EFFECT_SUMMARY_REPLAY",
        "effect_utilities_sha256": sha(args.effect_utilities),
        "state_utilities_sha256": sha(args.state_utilities),
        "score_receipt_sha256": sha(args.reference_dir / "score_receipt.json"),
        "tables": records, "effect_pattern_spread_max_abs": spread,
        "training_replayed": False,
        "scope": "Reconstruction of four comparison tables from frozen scored utilities",
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
    }
    (args.output / "replay_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
