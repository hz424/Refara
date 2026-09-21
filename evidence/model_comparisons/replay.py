#!/usr/bin/env python3
"""Recompute reference-design model comparisons from scored seed/unit inputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
FAMILIES = ("NC", "CM", "TW", "PCA", "RBF", "scGen", "CPA", "CellOT")
SEEDS = ("17", "29", "43")
UNITS = tuple(f"root_{i:02d}" for i in range(8))
DEPTHS = {("gse162632", "original8"): (4, 6, 8),
          ("gse162632", "cap48"): (4, 6, 8),
          ("parse", "common48"): (16, 24, 32, 40, 48),
          ("parse", "common128"): (16, 24, 32, 40, 48)}
POLICIES = {"S": "ALL_SHARED", "M": "OBS_PRED_SHARED_MODEL_SEPARATE",
            "P": "OBS_MODEL_SHARED_PRED_SEPARATE",
            "O": "PRED_MODEL_SHARED_OBS_SEPARATE", "D": "ALL_DISJOINT"}
INPUT_FIELDS = ("dataset", "regime", "seed", "depth", "unit", "policy_code", "family", "utility")
PAIRS = tuple(itertools.combinations(FAMILIES, 2))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_rows(path, rows):
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def authenticate(root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text())
    require(manifest["schema"] == "scored-model-comparisons-v1", "Unknown input schema")
    expected = {"data/scored_unit_utilities.tsv", "expected/current_mean_utilities.tsv",
                "expected/current_pair_margins.tsv"}
    require(set(manifest["files"]) == expected, "Incomplete input/validation file manifest")
    for name, pin in manifest["files"].items():
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe input path")
        path = root / relative
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root),
                "Input is missing or outside the capsule: " + name)
        require(path.stat().st_size == pin["bytes"] and sha256(path) == pin["sha256"],
                "Input hash differs: " + name)
    return manifest


def load_values(records):
    """Validate the complete supplied level: allocation-mean scores per seed/unit."""
    values = {}
    for row in records:
        require(set(row) == set(INPUT_FIELDS), "Unexpected input columns")
        key = (row["dataset"], row["regime"], row["seed"], int(row["depth"]),
               row["unit"], row["policy_code"], row["family"])
        require(key not in values, "Duplicate seed/unit score")
        utility = float(row["utility"])
        require(math.isfinite(utility) and utility <= 0, "Utility must be finite negative MSE (or zero)")
        values[key] = utility
    expected = {(d, r, s, depth, unit, policy, family)
                for (d, r), depths in DEPTHS.items()
                for s, depth, unit, policy, family in itertools.product(SEEDS, depths, UNITS, POLICIES, FAMILIES)}
    require(set(values) == expected, "Incomplete or unexpected group/seed/depth/unit/policy/family axes")
    return values


def ranks(values):
    """Higher utility is better; use midranks only for exact ties."""
    require(all(math.isfinite(v) for v in values.values()), "Nonfinite rank input")
    return {family: 1 + sum(v > value for v in values.values())
            + (sum(v == value for v in values.values()) - 1) / 2
            for family, value in values.items()}


def classify(shared, current):
    require(math.isfinite(shared) and math.isfinite(current), "Nonfinite pair margin")
    if shared == current == 0:
        return "tie_both"
    if shared == 0 or current == 0:
        return "tie_transition"
    return "reversal" if (shared > 0) != (current > 0) else "stable_order"


def aggregate(values):
    """Average scores over seeds within units, then give each unit equal weight."""
    models, seed_models, unit_models = [], [], []
    means = {}
    for (dataset, regime), depths in DEPTHS.items():
        for depth, policy, family in itertools.product(depths, POLICIES, FAMILIES):
            unit_scores = []
            for unit in UNITS:
                score = math.fsum(values[dataset, regime, seed, depth, unit, policy, family]
                                  for seed in SEEDS) / len(SEEDS)
                unit_scores.append(score)
                unit_models.append(dict(dataset=dataset, regime=regime, depth=depth, unit=unit,
                                        policy_code=policy, family=family, utility=score))
            score = math.fsum(unit_scores) / len(UNITS)
            means[dataset, regime, depth, policy, family] = score
            models.append(dict(dataset=dataset, regime=regime, depth=depth, policy_code=policy,
                               family=family, utility=score, mse=-score))
            for seed in SEEDS:
                seed_score = math.fsum(values[dataset, regime, seed, depth, unit, policy, family]
                                       for unit in UNITS) / len(UNITS)
                seed_models.append(dict(dataset=dataset, regime=regime, seed=seed, depth=depth,
                                        policy_code=policy, family=family, utility=seed_score))
    rank_lookup = {}
    for (dataset, regime), depths in DEPTHS.items():
        for depth, policy in itertools.product(depths, POLICIES):
            current = {family: means[dataset, regime, depth, policy, family] for family in FAMILIES}
            rank_lookup[dataset, regime, depth, policy] = ranks(current)
    for row in models:
        row["rank"] = rank_lookup[row["dataset"], row["regime"], row["depth"], row["policy_code"]][row["family"]]
    comparisons, summaries = [], []
    for (dataset, regime), depths in DEPTHS.items():
        for depth, policy in itertools.product(depths, POLICIES):
            group_pairs = []
            for a, b in PAIRS:
                margin = means[dataset, regime, depth, policy, a] - means[dataset, regime, depth, policy, b]
                shared = means[dataset, regime, depth, "S", a] - means[dataset, regime, depth, "S", b]
                row = dict(dataset=dataset, regime=regime, depth=depth, policy_code=policy,
                           family_a=a, family_b=b, utility_margin=margin, shared_utility_margin=shared,
                           reference_change=margin-shared, status_exact=classify(shared, margin))
                comparisons.append(row)
                group_pairs.append(row)
            summaries.append(dict(dataset=dataset, regime=regime, depth=depth, policy_code=policy,
                rms_reference_change=math.sqrt(math.fsum(r["reference_change"] ** 2 for r in group_pairs) / len(PAIRS)),
                reversals=sum(r["status_exact"] == "reversal" for r in group_pairs),
                stable_order=sum(r["status_exact"] == "stable_order" for r in group_pairs),
                tie_both=sum(r["status_exact"] == "tie_both" for r in group_pairs),
                tie_transition=sum(r["status_exact"] == "tie_transition" for r in group_pairs),
                pair_denominator=len(PAIRS)))
    return {"MODEL_SCORES.tsv": models, "SEED_MODEL_SCORES.tsv": seed_models,
            "UNIT_MODEL_SCORES.tsv": unit_models, "PAIR_MARGINS.tsv": comparisons,
            "POLICY_SUMMARY.tsv": summaries}


def compare_reference(tables, root=ROOT):
    """Cross-check computed values with separately delivered paper source tables."""
    errors = []
    key_fields = ("dataset", "regime", "depth", "policy_code")
    for output, expected, extra, numeric in (
        ("MODEL_SCORES.tsv", "current_mean_utilities.tsv", ("family",), ("utility",)),
        ("PAIR_MARGINS.tsv", "current_pair_margins.tsv", ("family_a", "family_b"),
         ("utility_margin", "shared_utility_margin")),
    ):
        keys = key_fields + extra
        actual = {tuple(str(r[k]) for k in keys): r for r in tables[output]}
        reference = {}
        for row in read_rows(Path(root) / "expected" / expected):
            key = tuple(row[k] for k in keys)
            require(key not in reference, "Duplicate reference row")
            reference[key] = row
        require(set(reference) == set(actual), "Published comparison axes differ")
        for key, saved in reference.items():
            got = actual[key]
            for field in numeric:
                error = abs(float(saved[field]) - got[field])
                require(math.isfinite(error) and error <= 1e-12, "Published value differs: " + repr((key, field)))
                errors.append(error)
            if output == "PAIR_MARGINS.tsv":
                require(got["status_exact"] == saved["status_exact"], "Published reversal/tie identity differs")
                expected_sign = (float(saved["utility_margin"]) > 0) - (float(saved["utility_margin"]) < 0)
                actual_sign = (got["utility_margin"] > 0) - (got["utility_margin"] < 0)
                require(expected_sign == actual_sign, "Published exact pair order differs")
    return {"model_values": len(tables["MODEL_SCORES.tsv"]),
            "pair_orders_and_reversal_identities": len(tables["PAIR_MARGINS.tsv"]),
            "maximum_absolute_value_difference": max(errors)}


def recompute(root=ROOT):
    root = Path(root)
    manifest = authenticate(root)
    values = load_values(read_rows(root / "data/scored_unit_utilities.tsv"))
    # The reference tables are used only after all results have been computed.
    tables = aggregate(values)
    checks = compare_reference(tables, root)
    primary = [r for r in tables["POLICY_SUMMARY.tsv"] if r["depth"] == max(DEPTHS[r["dataset"], r["regime"]])]
    leaders = [r for r in tables["MODEL_SCORES.tsv"]
               if r["depth"] == max(DEPTHS[r["dataset"], r["regime"]]) and r["rank"] == 1]
    report = dict(status="PASS_MODEL_COMPARISON_NUMERICAL_REPLAY", source_manifest_sha256=sha256(root / "SOURCE_MANIFEST.json"),
        origin_release=manifest["origin_release"], input_rows=len(values), groups=4, families=list(FAMILIES),
        seeds=list(SEEDS), biological_units_per_group=8, allocations_already_averaged=1000,
        input_level="negative-MSE scores per seed and unit, already averaged over allocations",
        aggregation="mean scored seeds within unit, then equal mean over eight units; compare models afterwards",
        exact_tie_rule="exact zero; average ranks for exact equal utilities", checks=checks,
        maximum_depth_comparisons=primary, maximum_depth_leaders=leaders,
        fresh_model_fitting=False, fresh_prediction_scoring=False, fresh_allocation_scoring=False,
        interpretation="Within-resource training regimes reuse evaluation units; allocations are descriptive, not independent biological replicates.")
    return tables, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.output.expanduser().resolve()
    require(not out.exists() and not out.is_relative_to(ROOT), "Use a new output directory outside the capsule")
    tables, report = recompute()
    out.mkdir(parents=True)
    for name, rows in tables.items():
        write_rows(out / name, rows)
    report["outputs"] = {name: {"sha256": sha256(out / name), "bytes": (out / name).stat().st_size}
                         for name in tables}
    (out / "REPLAY.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
