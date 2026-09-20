"""Hand-constructed workflow fixture; no scientific data or empirical result."""
from __future__ import annotations

from .reporting_candidate import MODE, RULES, digest


def fixture():
    protocol = {"mode": MODE, "status": "synthetic_fixture_frozen", "rules": list(RULES),
        "margin_convention": "loss_b_minus_loss_a", "unit_weighting": "equal",
        "seed_aggregation": "mean_after_scoring", "fit_seeds": [0, 1],
        "assessment_interval_method": "hand_specified_golden_bounds_not_inference",
        "multiplicity_policy": "not_applicable_synthetic_classifier_fixture",
        "independent_unit_definition": "fictional distinct experiment",
        "overlap_component_definition": "fictional distinct physical component",
        "independence_block_basis": "Fictional required sampling blocks, not technical library overlap",
        "anchor_allocation_id": "0", "minimum_primary_count": 1,
        "controls_per_unit": 4, "repeated_splits": 4, "crossfit_folds": 2, "crossfit_repeats": 2,
        "bootstrap_repetitions": 19, "bootstrap_seed": 100, "minimum_publication_units": 2,
        "minimum_assessment_units": 2, "tie_tolerance": 1e-12, "score_floor": 0,
        "repeated_quantile": .05, "bootstrap_quantile": .05, "primary_coverage_target": .5,
        "minimum_primary_coverage": .25, "confidence_level": .95, "coverage_grid": [.25, .5, .75, 1.],
        "work_caps": {rule: {"control_cell_forwards": 10000, "scored_seed_pairs": 10000} for rule in RULES}}
    admission = {"mode": MODE, "status": "synthetic_fixture_admitted", "authority": "invented engineering fixture"}
    units = [{"unit_id": unit, "partition": partition, "native_experiment_id": "experiment_" + unit,
              "independence_block_id": "block_" + unit} for unit, partition in
             [("C0", "calibration"), ("P0", "publication"), ("P1", "publication"), ("A0", "assessment"), ("A1", "assessment")]]
    roster = [{"pair_id": f"pair_{i}", "task_id": f"task_{i}", "cell_type": "fictional",
               "perturbation": "fictional", "dose": "fictional", "time": "fictional",
               "model_a": "model_A", "model_b": "model_B", "task_weight": 1} for i in range(4)]
    bundle = {"mode": MODE, "source_kind": "hand_constructed_fixture", "units": units,
              "roster": roster, "controls": {u["unit_id"]: [u["unit_id"] + f"_cell_{i}" for i in range(4)] for u in units},
              "inferences": {}, "records": []}
    margins = [4., 3., 2., 0.]

    def add(rule, unit, pair, kind, case, margin):
        controls = bundle["controls"][unit]
        fold = case % 2
        observation = controls[fold * 2:fold * 2 + 2]
        inputs = [c for c in controls if c not in observation]
        roles = {"model_input": inputs, "prediction_center": inputs, "observation_center": observation}
        target = f"{unit}:{pair['task_id']}:fixed_target"
        losses = []
        for seed in (0, 1):
            row = {"seed": seed, "loss_a": 10., "loss_b": 10. + margin}
            for suffix in ("a", "b"):
                receipt = {"model_id": pair[f"model_{suffix}"], "seed": seed, "input_cells": sorted(inputs),
                    "input_sha256": digest(sorted(inputs)), "target_id": target, "reuse_contract": "exact_input_set_only",
                    "control_cell_forwards": len(inputs), "cached": False, "wall_seconds": .001}
                key = digest(receipt)
                bundle["inferences"][key] = receipt
                row[f"inference_{suffix}"] = key
            losses.append(row)
        bundle["records"].append({"record_id": f"{rule}:{unit}:{pair['pair_id']}:{kind}:{case}",
            "rule_id": rule, "unit_id": unit, "pair_id": pair["pair_id"], "kind": kind,
            "allocation_id": str(case), "fold_id": str(fold), "repeat_id": str(case // 2),
            "target_id": target, "roles": roles, "seed_losses": losses})

    for rule in RULES:
        for unit in ("P0", "P1"):
            for index, pair in enumerate(roster):
                value = margins[index]
                if rule in ("repeated_splitting", "cross_fitting", "independent_unit_resampling"):
                    for case in range(4):
                        offset = (-.25 if case % 2 == 0 else .25) if value else 0
                        add(rule, unit, pair, "allocation" if rule == "repeated_splitting" else "fold", case, value + offset)
                else:
                    add(rule, unit, pair, "anchor", 0, value - .25 if value else value)
    for pair in roster:
        add("existing_refara_reporting", "C0", pair, "anchor", 0, 1.)
        for case in range(4):
            add("existing_refara_reporting", "C0", pair, "allocation", case, 1. if case % 2 == 0 else 1.5)
    return bundle, protocol, admission


def assessment(freeze_sha, bundle, protocol):
    return {"mode": MODE, "source_kind": "hand_constructed_fixture", "publication_freeze_sha256": freeze_sha,
        "membership_sha256": digest(bundle["units"]), "unit_ids": ["A0", "A1"],
        "interval_method": protocol["assessment_interval_method"], "confidence_level": protocol["confidence_level"],
        "multiplicity_policy": protocol["multiplicity_policy"], "intervals": [
            {"pair_id": "pair_0", "status": "interval", "lower": 1., "upper": 2.},
            {"pair_id": "pair_1", "status": "interval", "lower": -2., "upper": -1.},
            {"pair_id": "pair_2", "status": "interval", "lower": 0., "upper": 1.},
            {"pair_id": "pair_3", "status": "unassessable", "reason": "invented missing support"}]}
