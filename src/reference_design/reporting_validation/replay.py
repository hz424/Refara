"""Portable replay of the completed GSE181897 retrospective score comparison."""
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
import scipy

from . import _retrospective as k
from .assessment_intervals import finite_panel_risk_bounds, paired_rule_difference_bounds


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def table(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def save(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def write_table(path, rows):
    with Path(path).open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def equivalent(actual, expected, path="result"):
    """Tolerance verifies exports; classification itself keeps strict zero tests."""
    if isinstance(expected, dict):
        k.require(isinstance(actual, dict) and set(actual) == set(expected), "Different keys: " + path)
        for key in expected:
            equivalent(actual[key], expected[key], path + "/" + str(key))
    elif isinstance(expected, list):
        k.require(isinstance(actual, list) and len(actual) == len(expected), "Different lengths: " + path)
        for j, (a, b) in enumerate(zip(actual, expected)):
            equivalent(a, b, path + "/" + str(j))
    elif isinstance(expected, float):
        k.require(isinstance(actual, (float, int)) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), "Numeric difference: " + path)
    else:
        k.require(actual == expected, "Value difference: " + path)


def verify_table(actual, expected, ignored=()):
    k.require(len(actual) == len(expected), "Table length differs")
    for index, (a, b) in enumerate(zip(actual, expected)):
        for key, text in b.items():
            if key in ignored:
                continue
            value = a[key]
            if value is None:
                k.require(text == "", f"Missing value differs {index}/{key}")
            elif type(value) is bool:
                k.require(str(value) == text, f"Boolean differs {index}/{key}")
            elif isinstance(value, (int, float)):
                k.require(math.isclose(float(value), float(text), rel_tol=1e-12, abs_tol=1e-12), f"Number differs {index}/{key}")
            else:
                k.require(str(value) == text, f"Text differs {index}/{key}")


def read_partition(capsule, partition, protocol):
    rows = table(capsule / "source_data" / f"{partition}_rows.tsv.gz")
    contract = protocol["array_contracts"][partition]
    with np.load(capsule / "source_data" / f"{partition}_scores.npz", allow_pickle=False) as archive:
        k.require(set(archive.files) == {"losses", "fit_losses"}, "Unexpected scored arrays")
        losses, fits = archive["losses"], archive["fit_losses"]
    k.require(list(losses.shape) == contract["loss_shape"] and list(fits.shape) == contract["fit_loss_shape"], "Scored axes differ")
    k.require(losses.dtype == fits.dtype == np.float64 and np.isfinite(losses).all()
              and np.isfinite(fits).all() and (losses >= 0).all() and (fits >= 0).all(), "Invalid scored losses")
    k.require(contract["family_order"] == list(k.FAMILIES) and contract["seed_reduction"] == "mean of squared losses"
              and contract["anchor_index"] == 0 and contract["fold_pairs"] == [[2*i, 2*i+1] for i in range(32)], "Loss reduction contract changed")
    k.require([int(r["task_index"]) for r in rows] == list(range(len(rows))) and len(rows) == len(losses), "Task row order changed")
    keys = [(r["native_donor"], r["context"], r["condition"]) for r in rows]
    k.require(len(keys) == len(set(keys)) and all(r["partition"] == partition for r in rows), "Task identity collision")
    for j, family in enumerate(k.FAMILIES):
        indices = [i for i, fit in enumerate(contract["fits"]) if fit["family"] == family]
        k.require(np.array_equal(losses[..., j], fits[..., indices].mean(axis=-1)), "Seed loss reduction changed")
    return rows, losses


def compact_evaluate(plan, comparisons, records, companions):
    """Same finite-panel algebra; omit repeated per-candidate JSON expansions."""
    ids = [r["pair_id"] for r in comparisons]
    prefixes = k.companion_prefixes(companions)
    summary, paired, selected_hashes = [], [], []
    points = [("matched", p["published_M"], p["decisions"]) for p in plan["curve"]]
    points += [("natural", p["published_M"], p["directions"]) for p in plan["natural_sets"]]
    natural_companion = [dict(rule_id="cross_fitting_first_partition", pair_id=r["pair_id"],
                              direction=r["candidate_direction"] if r["eligible"] else "hold") for r in companions]
    points.append(("natural", sum(r["eligible"] for r in companions), natural_companion))
    for scope, m, decisions in points:
        decisions = list(decisions)
        if scope == "matched" and m in prefixes:
            decisions.extend(dict(rule_id="cross_fitting_first_partition", pair_id=r["pair_id"],
                                  direction=r["candidate_direction"] if r["pair_id"] in prefixes[m] else "hold") for r in companions)
        by_rule = {}
        for rule in dict.fromkeys(d["rule_id"] for d in decisions):
            directions = {d["pair_id"]: d["direction"] for d in decisions if d["rule_id"] == rule}
            by_rule[rule] = directions
            risk = finite_panel_risk_bounds(ids, directions, records)
            c = risk["counts"]
            k.require(c["published"] == m, "Publication count changed at assessment")
            point_reversals = sum(d != "hold" and records[key]["mean"] is not None
                and (1 if d == "a" else -1) * records[key]["mean"] < -k.TAU for key, d in directions.items())
            summary.append(dict(scope=scope, rule_id=rule, N=len(ids), M=m, coverage=m/len(ids),
                S=c["supported"], R=c["reversed"], U=c["unresolved"], held=c["held"],
                unassessable_published=c["unassessable_publications"],
                reversal_fraction=risk["reversal_fraction"], unresolved_fraction=risk["unresolved_fraction"],
                true_reversal_fraction_bounds=risk["true_reversal_fraction_bounds"],
                point_direction_reversals=point_reversals,
                utility="no_publications" if m == 0 else "below_practical_coverage" if m/len(ids) < .25 else "meets_coverage_floor"))
            selected_hashes.append(dict(scope=scope, M=m, rule_id=rule,
                decision_sha256=hashlib.sha256(json.dumps(directions, sort_keys=True, separators=(",", ":")).encode()).hexdigest()))
        if scope == "matched":
            for left, right in itertools.combinations(k.RULES, 2):
                result = paired_rule_difference_bounds(ids, by_rule[left], by_rule[right], records)
                paired.append(dict(M=m, rule_a=left, rule_b=right,
                    **{key: value for key, value in result.items() if key not in ("rule_j", "rule_k", "comparisons")}))
    return dict(primary_M=plan["primary_M"], primary_status=plan["primary_status"],
        common_achievable_M=plan["common_achievable_M"], companion_attainable_M=sorted(prefixes),
        summary=summary, paired_conditional_bounds=paired, decision_hashes=selected_hashes)


def replay(capsule, output):
    """Recompute every reported coverage point from losslessly saved scores."""
    started = time.monotonic()
    capsule, output = Path(capsule).resolve(), Path(output).resolve()
    k.require(not output.exists() and capsule not in output.parents,
              "Use a new output directory outside the capsule")
    manifest = load(capsule / "MANIFEST.json")
    for name, digest in manifest["files"].items():
        relative = Path(name)
        k.require(not relative.is_absolute() and ".." not in relative.parts, "Nonportable manifest path")
        k.require(file_hash(capsule / relative) == digest, "Capsule file changed: " + name)
    code = load(capsule / "code_contract.json")
    for name, digest in code.items():
        k.require(Path(name).name == name and file_hash(Path(__file__).parent / name) == digest,
                  "Replay code differs from reviewed export: " + name)
    protocol = load(capsule / "protocol.json")
    k.require(protocol["mode"] == "retrospective_used_data_only" and protocol["scientific_confirmation"] is False,
              "This capsule cannot claim prospective confirmation")
    output.mkdir(parents=True)
    contexts = protocol["rosters"]["secondary_contexts"]
    primary = protocol["rosters"]["primary_context"]
    crows, closs = read_partition(capsule, "C", protocol)
    prows, ploss = read_partition(capsule, "P", protocol)
    penalty = k.calibration_penalties(closs)
    candidates, companions, schedules = [], [], []
    for context in contexts:
        c, d, receipt = k.candidates_for_context(prows, ploss, context, penalty,
            bootstrap_count=protocol["publication_bootstrap_resamples"], seed=protocol["publication_bootstrap_seed"])
        candidates.extend(c); companions.extend(d); schedules.append(receipt)
    frozen = dict(mode=protocol["mode"], scientific_confirmation=False,
        prior_exposure=protocol["prior_exposure"], native_unit_caveat=protocol["native_unit_caveat"],
        protocol_sha256=file_hash(capsule / "protocol.json"), penalty_by_pair=penalty.tolist(),
        candidates=candidates, companions=companions, publication_bootstraps=schedules,
        coverage_protocol=k.coverage_protocol(),
        interpretation="This replay freeze precedes A reads in this process; the data are historical and previously inspected.")
    save(output / "PUBLICATION_FROZEN.json", frozen)
    save(output / "ASSESSMENT_OPENED.json", {"publication_freeze_sha256": file_hash(output / "PUBLICATION_FROZEN.json")})
    arows, aloss = read_partition(capsule, "A", protocol)
    k.require(not {r["native_donor"] for r in prows}.intersection(r["native_donor"] for r in arows), "P/A units overlap")
    panels, all_intervals = {}, {}
    expected_coverage = load(capsule / "expected/coverage.json")
    for panel, selected_contexts in (("primary", [primary]), ("secondary", contexts)):
        comparisons = k.roster(selected_contexts)
        ids = {r["pair_id"] for r in comparisons}
        plan = k.coverage_plans([r for r in candidates if r["pair_id"] in ids], k.coverage_protocol(), len(ids))
        expected = expected_coverage[panel]
        for name in ("primary_M", "primary_status", "common_achievable_M"):
            equivalent(plan[name], expected[name], panel + "/" + name)
        records, diagnostics = {}, []
        for context in selected_contexts:
            donors, matrix = k.context_matrix(arows, aloss, context)
            interval = k.working_intervals(matrix.mean(axis=1), [r["pair_id"] for r in k.roster([context])],
                multiplicity=len(comparisons), joint=panel == "primary",
                resamples=protocol["assessment_bootstrap_resamples"], seed=protocol["assessment_bootstrap_seed"])
            records.update(interval.pop("records"))
            diagnostics.append(dict(context=context, donors=donors, **interval))
        all_intervals[panel] = dict(records=records, diagnostics=diagnostics)
        panels[panel] = compact_evaluate(plan, comparisons, records, [r for r in companions if r["pair_id"] in ids])
        for row in panels[panel]["summary"]:
            if row["scope"] == "matched" and row["rule_id"] in k.RULES:
                equivalent({key: row[key] for key in ("S", "R", "U")},
                           expected["all_common_M"][str(row["M"])]["counts"][row["rule_id"]])
        del plan
    equivalent(all_intervals, load(capsule / "expected/assessment_intervals.json"), "intervals")
    summary = [dict(panel=panel, **row) for panel, value in panels.items() for row in value["summary"]]
    verify_table(summary, table(capsule / "expected/summary.tsv.gz"))
    pools = k.pool_sensitivity(arows, aloss, table(capsule / "source_data/donor_pools.tsv.gz"), contexts)
    verify_table(pools, table(capsule / "expected/pool_sensitivity.tsv.gz"))
    costs = k.work_accounting(prows, crows, primary)
    verify_table(costs, table(capsule / "expected/work_accounting.tsv.gz"))
    save(output / "assessment_intervals.json", all_intervals)
    write_table(output / "summary.tsv", summary)
    write_table(output / "pool_sensitivity.tsv", pools)
    write_table(output / "work_accounting.tsv", costs)
    result = dict(status="PASS_EXACT_RETROSPECTIVE_REPLAY", mode=protocol["mode"], scientific_confirmation=False,
        prior_exposure=protocol["prior_exposure"], native_unit_caveat=protocol["native_unit_caveat"],
        interval_calibration_performed=False, panels=panels,
        manifest_sha256=file_hash(capsule / "MANIFEST.json"),
        runtime=dict(seconds=time.monotonic()-started, numpy=np.__version__, scipy=scipy.__version__),
        scope="Saved-score replay; no new data, training or inference. Finite-panel risk bounds inherit unvalidated conditional-donor working intervals. No matched-coverage advantage was established by this retrospective experiment.")
    save(output / "RESULTS.json", result)
    return result
