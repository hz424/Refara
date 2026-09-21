"""Candidate reporting workflow. Only explicitly synthetic engineering inputs run.

Losses must already have been scored per seed. This module does not fit models,
run inference, construct assessment confidence intervals, or certify independence.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import fmean

RULES = ("score_margin", "repeated_splitting", "cross_fitting",
         "independent_unit_resampling", "existing_refara_reporting")
MODE = "synthetic_engineering_only"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value, label):
    require(type(value) in (int, float) and math.isfinite(value), f"Nonfinite or nonnumeric {label}")
    return float(value)


def mean(values):
    value = fmean(values)
    require(math.isfinite(value), "Nonfinite arithmetic result")
    return value


def quantile(values, q):
    values = sorted(values)
    require(bool(values) and 0 <= q <= 1, "Invalid quantile")
    return values[math.floor((len(values) - 1) * q)]


def sign(value, tolerance):
    return 1 if value > tolerance else -1 if value < -tolerance else 0


def validate_protocol(protocol, admission):
    # Deliberate hard stop until independent review and numerical/data admission.
    require(protocol.get("mode") == admission.get("mode") == MODE,
            "Candidate runner rejects real data: admission, numerical freeze and review are incomplete")
    require(protocol.get("status") == "synthetic_fixture_frozen", "Protocol is not a frozen synthetic fixture")
    require(admission.get("status") == "synthetic_fixture_admitted", "Synthetic admission missing")
    require(protocol["rules"] == list(RULES), "All five baselines are mandatory")
    require(protocol["margin_convention"] == "loss_b_minus_loss_a", "Margin orientation changed")
    require(protocol["unit_weighting"] == "equal" and protocol["seed_aggregation"] == "mean_after_scoring", "Unsupported weighting")
    require(protocol["assessment_interval_method"] and protocol["multiplicity_policy"], "Interval contract missing")
    require(protocol["independent_unit_definition"] and protocol["overlap_component_definition"], "Independence declaration missing")
    for key in ("controls_per_unit", "repeated_splits", "crossfit_folds", "crossfit_repeats", "bootstrap_repetitions", "minimum_publication_units", "minimum_assessment_units"):
        require(type(protocol[key]) is int and protocol[key] > 0, f"Invalid {key}")
    require(protocol["crossfit_folds"] >= 2, "Crossfit needs at least two folds")
    require(protocol["crossfit_folds"] == 2 and protocol["controls_per_unit"] % 2 == 0,
            "Candidate adapter supports balanced O-half two-fold designs only")
    require(protocol["repeated_splits"] == 2 * protocol["crossfit_repeats"], "Shared allocation bank size differs")
    require(protocol["anchor_allocation_id"] == "0", "Anchor must be first frozen orientation")
    require(protocol["independence_block_basis"] and protocol["minimum_primary_count"] > 0, "Missing block or coverage contract")
    require(len(protocol["fit_seeds"]) == len(set(protocol["fit_seeds"])) > 0, "Seed roster invalid")
    for key in ("tie_tolerance", "score_floor"):
        require(number(protocol[key], key) >= 0, f"Negative {key}")
    for key in ("repeated_quantile", "bootstrap_quantile", "primary_coverage_target", "minimum_primary_coverage", "confidence_level"):
        require(0 < number(protocol[key], key) < 1, f"Invalid {key}")
    require(protocol["minimum_primary_coverage"] <= protocol["primary_coverage_target"], "Coverage floor exceeds target")
    require(protocol["coverage_grid"] == sorted(set(protocol["coverage_grid"])), "Coverage grid invalid")
    require(all(0 < number(v, "coverage") <= 1 for v in protocol["coverage_grid"]), "Coverage out of range")
    require(type(protocol["bootstrap_seed"]) is int, "Bootstrap seed missing")
    require(set(protocol["work_caps"]) == set(RULES), "Each rule requires a work cap")
    for caps in protocol["work_caps"].values():
        require(set(caps) == {"control_cell_forwards", "scored_seed_pairs"}, "Work cap fields invalid")
        require(all(type(v) is int and v >= 0 for v in caps.values()), "Invalid work cap")


def validate_development(bundle, protocol, admission):
    validate_protocol(protocol, admission)
    require(bundle.get("mode") == MODE, "Input is not declared synthetic")
    require(bundle["source_kind"] == "hand_constructed_fixture", "Real loss banks are not admitted")
    roster = bundle["roster"]
    pair_ids = [row["pair_id"] for row in roster]
    require(len(pair_ids) == len(set(pair_ids)) > 0, "Duplicate/empty fixed roster")
    require(len({(r["task_id"], *sorted((r["model_a"], r["model_b"]))) for r in roster}) == len(roster), "Duplicate unordered pair")
    require(all(r["model_a"] != r["model_b"] and r["task_weight"] == 1 for r in roster), "Invalid pair or unsupported task weighting")
    units = {row["unit_id"]: row for row in bundle["units"]}
    require(len(units) == len(bundle["units"]), "Duplicate units")
    components = {}
    for unit in units.values():
        require(unit["partition"] in ("calibration", "publication", "assessment"), "Unknown unit partition")
        require(unit["native_experiment_id"] and unit["independence_block_id"], "Missing native unit identity")
        component = unit["independence_block_id"]
        require(components.setdefault(component, unit["partition"]) == unit["partition"], "Required independence block crosses partitions")
    require(len({u["native_experiment_id"] for u in units.values()}) == len(units), "Native experiment counted twice")
    partitions = {p: sorted(u for u, row in units.items() if row["partition"] == p) for p in ("calibration", "publication", "assessment")}
    require(partitions["calibration"] and len(partitions["publication"]) >= protocol["minimum_publication_units"] and len(partitions["assessment"]) >= protocol["minimum_assessment_units"], "Insufficient declared independent units")
    # These are sampling-model-required blocks, not arbitrary technical-pool graphs.
    require(len(components) == len(units), "Multiple rows share a required independence block; aggregate before analysis")
    controls = bundle["controls"]
    require(set(controls) == set(units), "Incomplete physical control roster")
    seen = set()
    for unit, cells in controls.items():
        require(len(cells) == len(set(cells)) == protocol["controls_per_unit"], "Physical control budget mismatch")
        require(not seen.intersection(cells), "Physical control reused across declared units")
        seen.update(cells)
    require(bundle["inferences"] and bundle["records"], "Empty input evidence")
    lookup = {r["pair_id"]: r for r in roster}
    grouped = defaultdict(list)
    inference_usage = defaultdict(set)
    partition_usage = defaultdict(set)
    partition_controls = defaultdict(set)
    partition_scored = defaultdict(int)
    uncached_conceptual = defaultdict(int)
    unique_controls = defaultdict(set)
    scored = defaultdict(int)
    seen_record_ids = set()
    fold_groups = defaultdict(list)
    for row in bundle["records"]:
        rule, unit, pair = row["rule_id"], row["unit_id"], row["pair_id"]
        require(rule in RULES and unit in units and pair in lookup, "Unknown record identity")
        require(units[unit]["partition"] != "assessment", "Assessment outcomes in publication input")
        require(row["record_id"] not in seen_record_ids, "Duplicate scoring record")
        seen_record_ids.add(row["record_id"])
        require(row["kind"] in ("anchor", "allocation", "fold"), "Unknown scoring kind")
        require(row["target_id"] == f"{unit}:{lookup[pair]['task_id']}:fixed_target", "Paired target changed across comparisons")
        roles = row["roles"]
        require(set(roles) == {"model_input", "prediction_center", "observation_center"}, "All reference roles must be explicit")
        for cells in roles.values():
            require(bool(cells) and len(cells) == len(set(cells)) and set(cells) <= set(controls[unit]), "Invalid physical role membership")
            unique_controls[rule].update(cells)
            partition_controls[(rule, units[unit]["partition"])].update(cells)
        require(set(roles["prediction_center"]) == set(roles["model_input"])
                and not set(roles["model_input"]).intersection(roles["observation_center"])
                and len(roles["model_input"]) == len(roles["observation_center"]) == protocol["controls_per_unit"] // 2,
                "All primary scores must use the same balanced O-half reference design")
        require(len(row["seed_losses"]) == len(protocol["fit_seeds"]), "Seed support incomplete")
        require({r["seed"] for r in row["seed_losses"]} == set(protocol["fit_seeds"]), "Seed roster changed")
        values = []
        for seed in row["seed_losses"]:
            loss_a, loss_b = number(seed["loss_a"], "loss_a"), number(seed["loss_b"], "loss_b")
            require(loss_a >= 0 and loss_b >= 0, "Squared losses cannot be negative")
            values.append(loss_b - loss_a)
            for suffix in ("a", "b"):
                receipt_id = seed[f"inference_{suffix}"]
                require(receipt_id in bundle["inferences"], "Missing inference receipt")
                receipt = bundle["inferences"][receipt_id]
                require(receipt["model_id"] == lookup[pair][f"model_{suffix}"] and receipt["seed"] == seed["seed"], "Inference model/seed mismatch")
                require(receipt["input_cells"] == sorted(roles["model_input"]), "Changed model input requires matching inference")
                require(receipt["input_sha256"] == digest(sorted(roles["model_input"])), "Inference input hash mismatch")
                require(receipt["target_id"] == row["target_id"], "Inference target mismatch")
                require(receipt["reuse_contract"] == "exact_input_set_only", "Unproved prediction cache reuse")
                require(receipt["control_cell_forwards"] == len(receipt["input_cells"]), "Invalid forward accounting")
                require(type(receipt["cached"]) is bool and number(receipt["wall_seconds"], "wall_seconds") >= 0, "Invalid inference cost metadata")
                inference_usage[rule].add(receipt_id)
                partition_usage[(rule, units[unit]["partition"])].add(receipt_id)
                uncached_conceptual[rule] += len(receipt["input_cells"])
        grouped[(rule, unit, pair)].append((row, mean(values)))
        scored[rule] += len(row["seed_losses"])
        partition_scored[(rule, units[unit]["partition"])] += len(row["seed_losses"])
        if rule == "cross_fitting":
            require(row["kind"] == "fold", "Crossfit cannot relabel an allocation")
            require(set(roles["model_input"]) == set(controls[unit]) - set(roles["observation_center"]), "Crossfit input must be held-out-fold complement")
            require(set(roles["prediction_center"]) == set(roles["model_input"]), "Crossfit prediction center differs from training fold")
            fold_groups[(unit, pair, row["repeat_id"])].append(row)
    for rows in fold_groups.values():
        require(len(rows) == protocol["crossfit_folds"], "Crossfit fold count mismatch")
        require(len({r["fold_id"] for r in rows}) == len(rows), "Duplicated fold")
        heldout = [c for r in rows for c in r["roles"]["observation_center"]]
        require(len(heldout) == len(set(heldout)) and set(heldout) == set(controls[rows[0]["unit_id"]]), "Crossfit held-out folds must partition physical controls once")
    for rule in RULES:
        for unit in partitions["publication"]:
            for pair in pair_ids:
                rows = grouped[(rule, unit, pair)]
                expected = protocol["repeated_splits"] if rule in ("repeated_splitting", "cross_fitting", "independent_unit_resampling") else 1
                require(len(rows) == expected, "Incomplete rule/unit/pair support")
                if rule in ("cross_fitting", "independent_unit_resampling"):
                    require(len({r["repeat_id"] for r, _ in rows}) == protocol["crossfit_repeats"], "Crossfit repeat count mismatch")
                elif rule == "repeated_splitting":
                    require(all(r["kind"] == "allocation" for r, _ in rows), "Repeated splitting requires allocation evidence")
                else:
                    require(rows[0][0]["kind"] == "anchor", "Rule requires its declared anchor")
                require(set(c for row, _ in rows for cells in row["roles"].values() for c in cells) == set(controls[unit]), "Rules do not use the same unique control budget")
        require(scored[rule] <= protocol["work_caps"][rule]["scored_seed_pairs"], "Scoring work exceeds cap")
        forwards = sum(bundle["inferences"][key]["control_cell_forwards"] for key in inference_usage[rule])
        require(forwards <= protocol["work_caps"][rule]["control_cell_forwards"], "Inference work exceeds cap")
    for unit in partitions["calibration"]:
        for pair in pair_ids:
            rows = grouped[("existing_refara_reporting", unit, pair)]
            require(sum(r["kind"] == "anchor" for r, _ in rows) == 1 and sum(r["kind"] == "allocation" for r, _ in rows) == protocol["repeated_splits"], "Refara needs calibration-only anchor and complete allocation cases")
            allocations = {r["allocation_id"]: r for r, _ in rows if r["kind"] == "allocation"}
            require(set(allocations) == {str(i) for i in range(protocol["repeated_splits"])}, "Incomplete calibration allocation bank")
            anchor = next(r for r, _ in rows if r["kind"] == "anchor")
            require(all(anchor[key] == allocations["0"][key] for key in ("roles", "seed_losses", "target_id")), "Calibration anchor differs from first allocation")
    for unit in partitions["publication"]:
        for pair in pair_ids:
            bank = {r["allocation_id"]: r for r, _ in grouped[("cross_fitting", unit, pair)]}
            require(set(bank) == {str(i) for i in range(protocol["repeated_splits"])}, "Crossfit allocation bank incomplete")
            for rule in ("repeated_splitting", "independent_unit_resampling"):
                rows = {r["allocation_id"]: r for r, _ in grouped[(rule, unit, pair)]}
                require(set(rows) == set(bank), "Shared allocation IDs differ")
                require(all(rows[k][field] == bank[k][field] for k in bank for field in ("roles", "seed_losses", "target_id", "fold_id", "repeat_id")), "Rules must reuse the identical scored allocation bank")
            for rule in ("score_margin", "existing_refara_reporting"):
                anchor = grouped[(rule, unit, pair)][0][0]
                require(all(anchor[field] == bank["0"][field] for field in ("roles", "seed_losses", "target_id")), "Publication anchor differs from first shared allocation")
    require(all(units[u]["partition"] == "publication" or r == "existing_refara_reporting" for r, u, _ in grouped), "Undeclared calibration records")
    accounting = {}
    for rule in RULES:
        receipts = [bundle["inferences"][key] for key in inference_usage[rule]]
        accounting[rule] = {"unique_physical_controls": len(unique_controls[rule]), "control_cell_forwards": sum(r["control_cell_forwards"] for r in receipts),
            "executed_control_cell_forwards": sum(r["control_cell_forwards"] for r in receipts if not r["cached"]),
            "conceptual_control_cell_forwards_without_reuse": uncached_conceptual[rule],
            "unique_inference_receipts": len(receipts), "cached_inference_receipts": sum(r["cached"] for r in receipts),
            "recorded_inference_wall_seconds": sum(r["wall_seconds"] for r in receipts), "scored_seed_pairs": scored[rule],
            "calibration_work_included": rule == "existing_refara_reporting",
            "unit_bootstrap_draws": protocol["bootstrap_repetitions"] if rule == "independent_unit_resampling" else 0,
            "unit_bootstrap_arithmetic_elements": protocol["bootstrap_repetitions"] * len(partitions["publication"]) * len(roster) if rule == "independent_unit_resampling" else 0}
        accounting[rule]["by_partition"] = {part: {"unique_physical_controls": len(partition_controls[(rule, part)]),
            "unique_inference_receipts": len(partition_usage[(rule, part)]),
            "control_cell_forwards": sum(bundle["inferences"][key]["control_cell_forwards"] for key in partition_usage[(rule, part)]),
            "scored_seed_pairs": partition_scored[(rule, part)]} for part in ("calibration", "publication")}
    all_receipts = set().union(*inference_usage.values())
    accounting["shared_execution"] = {"unique_inference_receipts": len(all_receipts),
        "executed_control_cell_forwards": sum(bundle["inferences"][key]["control_cell_forwards"] for key in all_receipts if not bundle["inferences"][key]["cached"]),
        "interpretation": "Shared receipts are counted once globally; baseline attribution may overlap. Synthetic work only."}
    return roster, units, partitions, grouped, accounting


def build_candidates(bundle, protocol, admission):
    roster, units, partitions, grouped, accounting = validate_development(bundle, protocol, admission)
    tau = protocol["tie_tolerance"]
    rng = random.Random(protocol["bootstrap_seed"])
    n = len(partitions["publication"])
    # One whole-unit schedule shared across every task/model pair preserves clustering.
    schedule = [[rng.randrange(n) for _ in range(n)] for _ in range(protocol["bootstrap_repetitions"])]
    candidates = []
    penalty = {}
    for pair in roster:
        model_pair = tuple(sorted((pair["model_a"], pair["model_b"])))
        shifts = []
        for unit in partitions["calibration"]:
            rows = grouped[("existing_refara_reporting", unit, pair["pair_id"])]
            anchor = next(v for r, v in rows if r["kind"] == "anchor")
            shifts.extend(abs(v - anchor) for r, v in rows if r["kind"] == "allocation")
        penalty[model_pair] = max(penalty.get(model_pair, 0), *shifts)
    for rule in RULES:
        for pair in roster:
            unit_cases = [grouped[(rule, unit, pair["pair_id"])] for unit in partitions["publication"]]
            unit_means = [mean([value for _, value in cases]) for cases in unit_cases]
            estimate = mean(unit_means)
            direction = sign(estimate, tau)
            companion = None
            if rule == "repeated_splitting":
                case_ids = [{row["allocation_id"] for row, _ in cases} for cases in unit_cases]
                require(all(v == case_ids[0] for v in case_ids), "Allocation IDs differ across units")
                require(len(case_ids[0]) == protocol["repeated_splits"], "Duplicate allocation IDs")
                pooled = [mean([next(v for row, v in cases if row["allocation_id"] == key) for cases in unit_cases]) for key in sorted(case_ids[0])]
                score = quantile([direction * v for v in pooled], protocol["repeated_quantile"])
                companion = abs(estimate)
            elif rule == "independent_unit_resampling":
                score = quantile([direction * mean([unit_means[i] for i in draw]) for draw in schedule], protocol["bootstrap_quantile"])
            elif rule == "existing_refara_reporting":
                score = abs(estimate) - penalty[tuple(sorted((pair["model_a"], pair["model_b"])))]
            else:
                score = abs(estimate)
            eligible = direction != 0 and score > protocol["score_floor"] + tau
            candidates.append({"rule_id": rule, "pair_id": pair["pair_id"], "estimate": estimate,
                "candidate_direction": "a" if direction > 0 else "b" if direction < 0 else "hold",
                "reliability_score": score, "eligible": eligible,
                "reason": "eligible" if eligible else "direction_tie" if not direction else "score_below_floor",
                "mean_margin_companion_score": companion})
    return candidates, accounting, {"unit_ids": partitions["publication"], "schedule_sha256": digest(schedule), "draw_count": len(schedule), "resampling_level": "whole_independent_unit"}


def coverage_plans(candidates, protocol, n):
    available = {}
    for rule in RULES:
        rows = sorted((r for r in candidates if r["rule_id"] == rule and r["eligible"]), key=lambda r: (-r["reliability_score"], r["pair_id"]))
        plans = {0: []}
        count = 0
        while count < len(rows):
            end = count + 1
            # Connected score-tie blocks: no pair closer than tau is split.
            while end < len(rows) and rows[end - 1]["reliability_score"] - rows[end]["reliability_score"] <= protocol["tie_tolerance"]:
                end += 1
            plans[end] = [r["pair_id"] for r in rows[:end]]
            count = end
        available[rule] = plans
    common = sorted(set.intersection(*(set(v) for v in available.values())))
    primary = max(m for m in common if m / n <= protocol["primary_coverage_target"])
    curves = []
    for m in common:
        decisions = []
        thresholds = {}
        for rule in RULES:
            eligible = [r for r in candidates if r["rule_id"] == rule and r["eligible"]]
            selected_ids = set(available[rule][m])
            selected_scores = [r["reliability_score"] for r in eligible if r["pair_id"] in selected_ids]
            rejected_scores = [r["reliability_score"] for r in eligible if r["pair_id"] not in selected_ids]
            thresholds[rule] = {"lower_inclusive": max(protocol["score_floor"], max(rejected_scores) - protocol["tie_tolerance"]) if rejected_scores else protocol["score_floor"],
                "upper_exclusive": min(selected_scores) - protocol["tie_tolerance"] if selected_scores else None,
                "comparison": "score > threshold + tie_tolerance", "selected_ids_sha256": digest(sorted(selected_ids))}
        for row in candidates:
            selected = row["pair_id"] in available[row["rule_id"]][m]
            decisions.append({"rule_id": row["rule_id"], "pair_id": row["pair_id"],
                "direction": row["candidate_direction"] if selected else "hold",
                "reason": "reported" if selected else row["reason"] if not row["eligible"] else "below_coverage_block"})
        curves.append({"published_M": m, "coverage": m / n, "decisions": decisions, "threshold_intervals": thresholds})
    return {"fixed_roster_N": n, "common_achievable_M": common, "primary_M": primary,
        "primary_status": "evaluable" if primary >= protocol["minimum_primary_count"] and primary / n >= protocol["minimum_primary_coverage"] else "not_evaluable",
        "secondary_grid_to_M": [{"target": c, "M": max(m for m in common if m / n <= c)} for c in protocol["coverage_grid"]],
        "natural_sets": [{"rule_id": rule, "published_M": sum(r["eligible"] for r in candidates if r["rule_id"] == rule),
             "directions": [{"rule_id": rule, "pair_id": r["pair_id"], "direction": r["candidate_direction"] if r["eligible"] else "hold", "reason": r["reason"]} for r in candidates if r["rule_id"] == rule]} for rule in RULES],
        "attainable_counts_by_rule": {rule: sorted(plans) for rule, plans in available.items()},
        "curve": curves}


def freeze_publication(bundle, protocol, admission, destination):
    candidates, accounting, bootstrap = build_candidates(bundle, protocol, admission)
    plan = coverage_plans(candidates, protocol, len(bundle["roster"]))
    frozen = {"schema": "reporting_candidate.freeze.v1", "mode": MODE,
        "scientific_validation": False, "development_input_sha256": digest(bundle),
        "protocol": protocol, "admission": admission, "protocol_sha256": digest(protocol),
        "admission_sha256": digest(admission), "roster": bundle["roster"], "units": bundle["units"],
        "membership_sha256": digest(bundle["units"]), "candidates": candidates,
        "plans": plan, "work_accounting": accounting, "bootstrap": bootstrap,
        "runner_sha256": file_hash(__file__)}
    path = Path(destination)
    path.mkdir(parents=True, exist_ok=False)
    with (path / "PUBLICATION_FROZEN.json").open("xb") as stream:
        stream.write(canonical(frozen))
    return file_hash(path / "PUBLICATION_FROZEN.json")


def assess(frozen_directory, expected_freeze_sha256, assessment_path, output_path):
    directory = Path(frozen_directory)
    frozen_path = directory / "PUBLICATION_FROZEN.json"
    frozen_bytes = frozen_path.read_bytes()
    require(hashlib.sha256(frozen_bytes).hexdigest() == expected_freeze_sha256, "Publication freeze was modified")
    frozen = json.loads(frozen_bytes)
    protocol = frozen["protocol"]
    validate_protocol(protocol, frozen["admission"])
    require(frozen["runner_sha256"] == file_hash(__file__), "Runner changed after publication freeze")
    require(digest(protocol) == frozen["protocol_sha256"], "Protocol changed after freeze")
    require(not Path(output_path).exists(), "Assessment output already exists")
    # Claim the single opening before any assessment file read, even for invalid input.
    with (directory / "ASSESSMENT_OPENED.json").open("xb") as stream:
        stream.write(canonical({"publication_freeze_sha256": expected_freeze_sha256, "attempt": 1}))
    assessment_bytes = Path(assessment_path).read_bytes()
    assessment = json.loads(assessment_bytes)
    require(assessment["mode"] == MODE and assessment["source_kind"] == "hand_constructed_fixture", "Real assessment is not admitted")
    require(assessment["publication_freeze_sha256"] == expected_freeze_sha256, "Assessment binds another publication freeze")
    require(assessment["membership_sha256"] == frozen["membership_sha256"], "Assessment unit roster changed")
    required_units = sorted(u["unit_id"] for u in frozen["units"] if u["partition"] == "assessment")
    require(assessment["unit_ids"] == required_units, "Assessment independent-unit support changed")
    require(assessment["interval_method"] == protocol["assessment_interval_method"] and assessment["confidence_level"] == protocol["confidence_level"] and assessment["multiplicity_policy"] == protocol["multiplicity_policy"], "Assessment interval procedure changed")
    by_pair = {r["pair_id"]: r for r in assessment["intervals"]}
    require(len(by_pair) == len(assessment["intervals"]) and set(by_pair) == {r["pair_id"] for r in frozen["roster"]}, "Assessment must retain complete fixed roster")
    for row in by_pair.values():
        if row["status"] == "unassessable":
            require(row.get("reason") and "lower" not in row and "upper" not in row, "Unassessable requires reason, not a fabricated interval")
        else:
            require(row["status"] == "interval", "Invalid interval status")
            number(row["lower"], "lower")
            number(row["upper"], "upper")
            # Compare original JSON numbers before any float64 rounding collapses
            # distinct exact integers above 2**53 into an apparently valid bound.
            require(row["lower"] <= row["upper"], "Invalid interval")
    curves = []
    classified = []
    natural_rows = []
    points = [dict(point, scope="matched", rules=list(RULES)) for point in frozen["plans"]["curve"]]
    points += [{"scope": "natural", "rules": [point["rule_id"]], "published_M": point["published_M"],
                "coverage": point["published_M"] / frozen["plans"]["fixed_roster_N"], "decisions": point["directions"]} for point in frozen["plans"]["natural_sets"]]
    for point in points:
        for rule in point["rules"]:
            counts = {"supported": 0, "reversed": 0, "unresolved": 0, "hold": 0, "unassessable_published": 0}
            for decision in (d for d in point["decisions"] if d["rule_id"] == rule):
                row = by_pair[decision["pair_id"]]
                reason = None
                if decision["direction"] == "hold":
                    status = "hold"
                elif row["status"] == "unassessable":
                    status, reason = "unresolved", row["reason"]
                    counts["unassessable_published"] += 1
                else:
                    lo, hi = (row["lower"], row["upper"]) if decision["direction"] == "a" else (-row["upper"], -row["lower"])
                    status = "supported" if lo > 0 else "reversed" if hi < 0 else "unresolved"
                counts[status] += 1
                classified.append({"scope": point["scope"], "rule_id": rule, "M": point["published_M"], "pair_id": decision["pair_id"], "direction": decision["direction"], "status": status, "reason": reason})
            m = point["published_M"]
            require(counts["supported"] + counts["reversed"] + counts["unresolved"] == m, "Published denominator changed")
            output = curves if point["scope"] == "matched" else natural_rows
            output.append({"rule_id": rule, "fixed_roster_N": frozen["plans"]["fixed_roster_N"], "published_M": m,
                "coverage": point["coverage"], **counts, "reversal_fraction": counts["reversed"] / m if m else None,
                "unresolved_fraction": counts["unresolved"] / m if m else None})
    comparisons = []
    for m in frozen["plans"]["common_achievable_M"]:
        rows = {r["rule_id"]: r for r in curves if r["published_M"] == m}
        for i, left in enumerate(RULES):
            for right in RULES[i + 1:]:
                a, b = rows[left], rows[right]
                comparisons.append({"M": m, "rule_a": left, "rule_b": right,
                    "delta_reversed_a_minus_b": a["reversed"] - b["reversed"],
                    "delta_unresolved_a_minus_b": a["unresolved"] - b["unresolved"],
                    "a_has_fewer_reversals_without_more_unresolved": bool(m and a["reversed"] < b["reversed"] and a["unresolved"] <= b["unresolved"]),
                    "interpretation": "descriptive synthetic counts only; no uncertainty test or advantage claim"})
    report = {"schema": "reporting_candidate.assessment.v1", "mode": MODE, "scientific_validation": False,
        "publication_freeze_sha256": expected_freeze_sha256, "assessment_sha256": hashlib.sha256(assessment_bytes).hexdigest(),
        "primary_M": frozen["plans"]["primary_M"], "primary_status": frozen["plans"]["primary_status"],
        "curves": curves, "natural_release": natural_rows, "classifications": classified, "comparisons": comparisons,
        "scope": "Mechanical classification of supplied intervals for mean directions. No CI estimation, calibration, multiplicity validation, per-unit direction guarantee, or biological-independence proof."}
    with Path(output_path).open("xb") as stream:
        stream.write(canonical(report))
    return report
