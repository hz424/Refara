#!/usr/bin/env python3
"""Freeze reporting rules, then compare their recorded risk and coverage."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import itertools
import json
import math
from pathlib import Path

from reference_design import organizer as org
from reference_design.cli import _destination

RULES = ("single_split", "repeated_primary", "development_margin")
RANKINGS = ("anchor_margin", "calibrated_margin")
ARTIFACT = "reference_design_reporting_comparison"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def protocol(value):
    required = {"schema_version", "minimum_gaps", "primary_minimum_gap",
                "coverage_grid", "numerical_tolerance"}
    require(isinstance(value, dict) and required <= set(value), "Incomplete comparison protocol")
    require(type(value["schema_version"]) is int and value["schema_version"] == 1,
            "Unsupported comparison protocol")
    for field, lower, upper in (("minimum_gaps", 0, None), ("coverage_grid", 0, 1)):
        values = value[field]
        require(isinstance(values, list) and bool(values), field + " must be a nonempty list")
        for item in values:
            org._number(item, field, minimum=lower, positive=field == "coverage_grid")
            require(upper is None or item <= upper, "Coverage cannot exceed one")
        require(values == sorted(set(values)), field + " must be sorted and distinct")
    org._number(value["primary_minimum_gap"], "primary_minimum_gap", minimum=0)
    require(value["primary_minimum_gap"] in value["minimum_gaps"], "Primary gap is absent")
    org._number(value["numerical_tolerance"], "numerical_tolerance", positive=True)
    if "fixed_threshold_rules" in value:
        require(value["fixed_threshold_rules"] == list(RULES), "Unsupported reporting rules")
    if "rankings" in value:
        require(value["rankings"].get("tie_order") == ["case", "model_a", "model_b"],
                "Unsupported ranking tie order")
    return value


def direction(margin, tolerance):
    return 1 if margin > tolerance else -1 if margin < -tolerance else 0


def support(cases, anchor, name):
    require(bool(cases), name + " scores must not be empty")
    supports = {tuple(sorted(values)) for case in cases.values()
                for values in case["scores"].values()}
    require(len(supports) == 1, name + " realization support differs between cases")
    result = list(next(iter(supports)))
    require(anchor in result, name + " anchor is absent")
    return result


def _configuration(manifest, analysis):
    org._schema(manifest, org.CONFIG_FIELDS | org.PATH_FIELDS,
                {"geometry", "geometry_tolerance"}, "Organizer manifest")
    config = {key: manifest[key] for key in org.CONFIG_FIELDS}
    config["policies"] = list(org.BASE_POLICIES)
    org._configuration(config)
    config["models"] = sorted(config["models"])
    config["budgets"] = sorted(config["budgets"],
                               key=lambda row: (row["control_cells_available"], row["id"]))
    require(config["minimum_gap"] == analysis["primary_minimum_gap"],
            "Manifest minimum_gap differs from the protocol primary gap")
    require(config["numerical_tolerance"] == analysis["numerical_tolerance"],
            "Manifest numerical_tolerance differs from the protocol")
    return config


def _development_statistics(cases, config):
    result = []
    anchor, tau = config["anchor_realization"], config["numerical_tolerance"]
    for case_id, case in sorted(cases.items()):
        for budget in config["budgets"]:
            bid = budget["id"]
            values = case["scores"][bid]
            for a, b in itertools.combinations(config["models"], 2):
                m0 = org._margin(values[anchor], a, b, config)
                sign = direction(m0, tau)
                margins = [org._margin(scores, a, b, config) for scores in values.values()]
                result.append(dict(case=case_id, unit=case["unit"], budget=bid,
                                   model_a=a, model_b=b, anchor_margin=m0,
                                   max_displacement=max(abs(value - m0) for value in margins),
                                   minimum_oriented_margin=min(sign * value for value in margins)
                                   if sign else None))
    return result


def _build_plan(config, analysis, inputs, development, anchors, realizations):
    """Derive every selection from development summaries and assessment anchors."""
    tau = config["numerical_tolerance"]
    pairs = list(itertools.combinations(config["models"], 2))
    calibration, thresholds = [], []
    for budget in config["budgets"]:
        bid = budget["id"]
        dev = [row for row in development if row["budget"] == bid]
        for a, b in pairs:
            shifts = [row["max_displacement"] for row in dev
                      if (row["model_a"], row["model_b"]) == (a, b)]
            calibration.append(dict(budget=bid, model_a=a, model_b=b, B=max(shifts)))
        for gap in analysis["minimum_gaps"]:
            unsupported = [row for row in dev if row["minimum_oriented_margin"] is None
                           or row["minimum_oriented_margin"] <= gap + tau]
            threshold = max([gap] + [abs(row["anchor_margin"]) for row in unsupported])
            thresholds.append(dict(budget=bid, minimum_gap=gap, threshold=threshold,
                                   development_comparisons=len(dev),
                                   development_unsupported=len(unsupported)))
    bounds = {(row["budget"], row["model_a"], row["model_b"]): row["B"] for row in calibration}
    comparisons = []
    for row in anchors:
        for a, b in pairs:
            margin = org._margin(row["scores"], a, b, config)
            sign = direction(margin, tau)
            bound = bounds[(row["budget"], a, b)]
            comparisons.append(dict(comparison_id=len(comparisons), case=row["case"],
                                    unit=row["unit"], budget=row["budget"], model_a=a, model_b=b,
                                    anchor_margin=margin, direction=sign,
                                    favored_model=a if sign == 1 else b if sign == -1 else None,
                                    B=bound, anchor_margin_priority=abs(margin),
                                    calibrated_margin_priority=abs(margin) - bound))
    rankings, selections = [], []
    for budget in config["budgets"]:
        bid = budget["id"]
        rows = [row for row in comparisons if row["budget"] == bid]
        for method in RANKINGS:
            ordered = sorted(rows, key=lambda row: (-row[method + "_priority"],
                                                   row["case"], row["model_a"], row["model_b"]))
            rankings.append(dict(budget=bid, ranking=method,
                                 comparison_ids=[row["comparison_id"] for row in ordered]))
        for gap in analysis["minimum_gaps"]:
            threshold = next(row["threshold"] for row in thresholds
                             if row["budget"] == bid and row["minimum_gap"] == gap)
            for rule in RULES:
                def released(row):
                    cutoff = threshold if rule == "development_margin" else (
                        row["B"] + gap if rule == "repeated_primary" else gap)
                    return row["direction"] != 0 and abs(row["anchor_margin"]) > cutoff + tau
                selections.append(dict(budget=bid, minimum_gap=gap, rule=rule,
                                       comparison_ids=[row["comparison_id"] for row in rows if released(row)]))
    return dict(artifact_type=ARTIFACT, schema_version=1, configuration=config,
                protocol=analysis, inputs=inputs, development_realizations=realizations,
                development_statistics=development, assessment_anchors=anchors,
                calibration=calibration, development_thresholds=thresholds,
                comparisons=comparisons, rankings=rankings, selections=selections)


def freeze(manifest_path, protocol_path, output_path):
    """Publish a self-contained plan without opening assessment realizations."""
    manifest_path, protocol_path, output = Path(manifest_path), Path(protocol_path), Path(output_path)
    _destination(output)
    initial_hashes = {"manifest": org._digest(manifest_path), "protocol": org._digest(protocol_path)}
    analysis = protocol(org._json(protocol_path))
    manifest = org._json(manifest_path)
    config = _configuration(manifest, analysis)
    paths = {field: manifest_path.parent / org.label(manifest[field], field) for field in org.PATH_FIELDS}
    paths.update(manifest=manifest_path, protocol=protocol_path,
                 comparison_code=Path(__file__), organizer_code=Path(org.__file__))
    inputs = {field: dict(name=path.name, sha256=org._digest(path)) for field, path in paths.items()}
    require(all(inputs[field]["sha256"] == digest for field, digest in initial_hashes.items()),
            "Manifest or protocol changed while reading")
    development = org._scores(paths["development_scores"], config)
    assessment = org._scores(paths["assessment_anchors"], config, anchor_only=True)
    require(not set(development) & set(assessment), "Development and assessment cases must be disjoint")
    realizations = support(development, config["anchor_realization"], "Development")
    support(assessment, config["anchor_realization"], "Assessment anchor")
    anchors = [dict(case=case_id, unit=case["unit"], budget=budget["id"],
                    scores=case["scores"][budget["id"]][config["anchor_realization"]])
               for case_id, case in sorted(assessment.items()) for budget in config["budgets"]]
    plan = _build_plan(config, analysis, inputs, _development_statistics(development, config),
                       anchors, realizations)
    require(all(org._digest(path) == inputs[field]["sha256"] for field, path in paths.items()),
            "A planning input changed while freezing")
    org._emit(output, {"plan.json": org._encoded(plan),
                       "calibration.tsv": org._tabular(plan["calibration"]),
                       "development_thresholds.tsv": org._tabular(plan["development_thresholds"])}, inputs)
    return plan


def _validated_plan(plan_path):
    path = Path(plan_path)
    receipt = org._json(path.parent / "receipt.json")
    require(receipt.get("outputs", {}).get(path.name) == org._digest(path),
            "Frozen plan hash differs from its receipt")
    plan = org._json(path)
    fields = {"artifact_type", "schema_version", "configuration", "protocol", "inputs",
              "development_realizations", "development_statistics", "assessment_anchors",
              "calibration", "development_thresholds", "comparisons", "rankings", "selections"}
    org._schema(plan, fields, set(), "Comparison plan")
    require(plan["artifact_type"] == ARTIFACT and type(plan["schema_version"]) is int
            and plan["schema_version"] == 1, "Unsupported comparison plan")
    config = org._configuration(plan["configuration"])
    analysis = protocol(plan["protocol"])
    require(config["minimum_gap"] == analysis["primary_minimum_gap"]
            and config["numerical_tolerance"] == analysis["numerical_tolerance"],
            "Frozen configuration differs from its protocol")
    require(plan["inputs"]["comparison_code"]["sha256"] == org._digest(Path(__file__)),
            "Comparison code differs from the frozen implementation")
    require(plan["inputs"]["organizer_code"]["sha256"] == org._digest(Path(org.__file__)),
            "Organizer code differs from the frozen implementation")
    require(plan["inputs"] == receipt.get("inputs"), "Frozen input receipt differs")
    expected = _build_plan(config, analysis, plan["inputs"], plan["development_statistics"],
                           plan["assessment_anchors"], plan["development_realizations"])
    require(plan == expected, "Frozen selections differ from their calibration and anchors")
    return plan


def _assessment(plan, assessment_path):
    config = plan["configuration"]
    cases = org._scores(assessment_path, config)
    expected_cases = {row["case"] for row in plan["assessment_anchors"]}
    require(set(cases) == expected_cases, "Assessment cases differ from the frozen plan")
    realizations = support(cases, config["anchor_realization"], "Assessment")
    for row in plan["assessment_anchors"]:
        case = cases[row["case"]]
        require(case["unit"] == row["unit"], "Assessment unit differs from the frozen anchor")
        require(case["scores"][row["budget"]][config["anchor_realization"]] == row["scores"],
                "Assessment anchor values differ from the frozen plan")
    records = []
    for row in plan["comparisons"]:
        scores = cases[row["case"]]["scores"][row["budget"]]
        minimum = min(row["direction"] * org._margin(values, row["model_a"], row["model_b"], config)
                      for values in scores.values()) if row["direction"] else None
        records.append(dict(**row, minimum_oriented_margin=minimum,
                            strict_reversal=minimum is not None and minimum < -config["numerical_tolerance"],
                            no_direction=row["direction"] == 0))
    return records, realizations


def _error(row, gap, tau):
    return row["no_direction"] or row["minimum_oriented_margin"] <= gap + tau


def _summary(selected, population, gap, tau):
    count = len(selected)
    unsupported = sum(_error(row, gap, tau) for row in selected)
    totals = Counter(row["unit"] for row in population)
    by_unit = defaultdict(list)
    for row in selected:
        by_unit[row["unit"]].append(row)
    risks = [sum(_error(row, gap, tau) for row in rows) / len(rows) for rows in by_unit.values()]
    return dict(comparisons=len(population), selected=count, coverage=count / len(population),
                unsupported=unsupported, risk=unsupported / count if count else None,
                strict_reversals=sum(row["strict_reversal"] for row in selected),
                strict_reversal_risk=sum(row["strict_reversal"] for row in selected) / count if count else None,
                no_direction=sum(row["no_direction"] for row in selected), units=len(totals),
                units_with_selections=len(by_unit),
                unit_macro_coverage=sum(len(by_unit[unit]) / total for unit, total in totals.items()) / len(totals),
                unit_macro_risk=sum(risks) / len(risks) if risks else None)


def _group_summaries(selected, population, gap, tau, fields):
    groups = sorted({tuple(row[field] for field in fields) for row in population})
    return [dict(zip(fields, key), **_summary(
        [row for row in selected if tuple(row[field] for field in fields) == key],
        [row for row in population if tuple(row[field] for field in fields) == key], gap, tau))
            for key in groups]


def _prefix_rows(ordered, gap, tau, method):
    """Accumulate full risk curves without quadratic rescanning."""
    unit_totals = Counter(row["unit"] for row in ordered)
    unit_selected, unit_errors = Counter(), Counter()
    priorities = [row[method + "_priority"] for row in ordered]
    tie_counts = Counter(priorities)
    selected_ties = Counter()
    unsupported = reversals = no_direction = 0
    for k, row in enumerate(ordered, 1):
        error = _error(row, gap, tau)
        unsupported += error
        reversals += row["strict_reversal"]
        no_direction += row["no_direction"]
        unit_selected[row["unit"]] += 1
        unit_errors[row["unit"]] += error
        priority = priorities[k - 1]
        selected_ties[priority] += 1
        yield dict(comparisons=len(ordered), selected=k, coverage=k / len(ordered),
                   unsupported=unsupported, risk=unsupported / k,
                   strict_reversals=reversals, strict_reversal_risk=reversals / k,
                   no_direction=no_direction, units=len(unit_totals),
                   units_with_selections=len(unit_selected),
                   unit_macro_coverage=sum(unit_selected[u] / total for u, total in unit_totals.items()) / len(unit_totals),
                   unit_macro_risk=sum(unit_errors[u] / n for u, n in unit_selected.items()) / len(unit_selected),
                   cutoff_priority=priority, cutoff_tie_count=tie_counts[priority],
                   cutoff_tie_selected=selected_ties[priority],
                   cutoff_splits_tie=selected_ties[priority] < tie_counts[priority])


def check(plan_path, assessment_path, output_path):
    """Check the already-published plan against separately supplied outcomes."""
    plan_path, assessment_path, output = Path(plan_path), Path(assessment_path), Path(output_path)
    _destination(output)
    plan = _validated_plan(plan_path)
    plan_hash, assessment_hash = org._digest(plan_path), org._digest(assessment_path)
    records, realizations = _assessment(plan, assessment_path)
    by_id = {row["comparison_id"]: row for row in records}
    tau = plan["configuration"]["numerical_tolerance"]
    rules, rule_units, rule_pairs = [], [], []
    curves, grid, grid_units, grid_pairs, areas = [], [], [], [], []
    for selection in plan["selections"]:
        bid, gap, rule = (selection[field] for field in ("budget", "minimum_gap", "rule"))
        context = dict(budget=bid, minimum_gap=gap, rule=rule)
        population = [row for row in records if row["budget"] == bid]
        selected = [by_id[index] for index in selection["comparison_ids"]]
        rules.append(dict(**context, **_summary(selected, population, gap, tau)))
        rule_units.extend(dict(**context, **row) for row in _group_summaries(selected, population, gap, tau, ["unit"]))
        rule_pairs.extend(dict(**context, **row) for row in _group_summaries(selected, population, gap, tau, ["model_a", "model_b"]))
    for ranking in plan["rankings"]:
        ordered = [by_id[index] for index in ranking["comparison_ids"]]
        bid, method = ranking["budget"], ranking["ranking"]
        for gap in plan["protocol"]["minimum_gaps"]:
            context = dict(budget=bid, minimum_gap=gap, ranking=method)
            prefixes = list(_prefix_rows(ordered, gap, tau, method))
            curves.extend(dict(**context, **row) for row in prefixes)
            areas.append(dict(**context, comparisons=len(ordered),
                              aurc=sum(row["risk"] for row in prefixes) / len(prefixes),
                              strict_reversal_aurc=sum(row["strict_reversal_risk"] for row in prefixes) / len(prefixes)))
            for coverage in plan["protocol"]["coverage_grid"]:
                k = math.ceil(coverage * len(ordered))
                grid_context = dict(**context, requested_coverage=coverage)
                grid.append(dict(**grid_context, **prefixes[k - 1]))
                grid_units.extend(dict(**grid_context, **row) for row in
                                  _group_summaries(ordered[:k], ordered, gap, tau, ["unit"]))
                grid_pairs.extend(dict(**grid_context, **row) for row in
                                  _group_summaries(ordered[:k], ordered, gap, tau, ["model_a", "model_b"]))
    config = plan["configuration"]
    development_cases = {row["case"] for row in plan["development_statistics"]}
    development_units = {row["unit"] for row in plan["development_statistics"]}
    summary = dict(schema_version=1, artifact_type=ARTIFACT, protocol=plan["protocol"],
                   plan_sha256=plan_hash, assessment_sha256=assessment_hash,
                   development_cases=len(development_cases), development_units=len(development_units),
                   assessment_cases=len({row["case"] for row in records}),
                   assessment_units=len({row["unit"] for row in records}),
                   development_realizations=plan["development_realizations"],
                   assessment_realizations=realizations,
                   costs={"budgets_per_realization": config["budgets"],
                          "development_realizations_per_case_budget": len(plan["development_realizations"]),
                          "assessment_realizations_per_case_budget": len(realizations),
                          "distinct_controls_across_realizations": None,
                          "total_prediction_calls": None},
                   unit_macro_definition="Mean over declared units; risk excludes units with zero selections, coverage includes all units.",
                   scope=config["evaluation_scope"], rule_summary=rules, aurc=areas)
    contents = {"check.json": org._encoded(summary)}
    for filename, rows in (("rule_summary.tsv", rules), ("rule_units.tsv", rule_units),
                           ("rule_pairs.tsv", rule_pairs), ("risk_coverage.tsv", curves),
                           ("coverage_grid.tsv", grid), ("coverage_units.tsv", grid_units),
                           ("coverage_pairs.tsv", grid_pairs), ("aurc.tsv", areas),
                           ("comparison_assessments.tsv", records)):
        contents[filename] = org._tabular(rows)
    require(org._digest(plan_path) == plan_hash and org._digest(assessment_path) == assessment_hash,
            "A frozen plan or assessment input changed while checking")
    inputs = dict(plan={"name": plan_path.name, "sha256": plan_hash},
                  assessment={"name": assessment_path.name, "sha256": assessment_hash})
    org._emit(output, contents, inputs)
    return summary


def run(manifest_path, assessment_path, protocol_path, output_path):
    """Finish publishing the plan and receipt before opening assessment scores."""
    output = Path(output_path)
    _destination(output)
    freeze(manifest_path, protocol_path, output / "freeze")
    return check(output / "freeze" / "plan.json", assessment_path, output / "check")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "check", "run"):
        command = commands.add_parser(name)
        command.add_argument("--output", required=True, type=Path)
        if name != "check":
            command.add_argument("--manifest", required=True, type=Path)
            command.add_argument("--protocol", required=True, type=Path)
        else:
            command.add_argument("--plan", required=True, type=Path)
        if name != "freeze":
            command.add_argument("--assessment", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.manifest, args.protocol, args.output)
    elif args.command == "check":
        check(args.plan, args.assessment, args.output)
    else:
        run(args.manifest, args.assessment, args.protocol, args.output)
    print(json.dumps({"status": "complete", "stage": args.command}))


if __name__ == "__main__":
    main()
