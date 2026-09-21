"""Compare allocations of 16 controls using reusable per-cell predictions.

Run after installing the package:
    python examples/fixed_budget.py --output /tmp/refara-fixed-budget
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from reference_design.fixed_budget import (
    RULES, fixed_argmin, reduce_seed_losses, rule_cost, score_independent_effect,
    score_reference_design,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Directory for allocation_scores.tsv and assessment.json")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        if not args.output.is_dir() or any(args.output.iterdir()):
            parser.error("--output must be a new or empty directory")
    rng = np.random.default_rng(7)
    controls = rng.normal(size=(16, 3))
    treated_mean = np.array([0.7, -0.2, 0.4])
    scales = np.array([1.0, 2.0, 0.5])  # Fixed training scales.
    # Two fixed point-model fits: every output depends only on its input cell.
    state_fits = [0.8 * controls + np.array([0.5, -0.1, shift]) for shift in (0.2, 0.3)]
    direct_effect = np.array([0.5, -0.1, 0.3])
    rows = []
    print("rule                  state loss  direct loss  cached forwards  selected")
    for rule in RULES:
        seed_losses = [score_reference_design(controls, state, treated_mean, scales, rule)
                       for state in state_fits]
        state_loss = float(reduce_seed_losses(seed_losses))
        direct_loss = score_reference_design(controls, direct_effect, treated_mean, scales, rule)
        chosen = ("state", "direct")[fixed_argmin([state_loss, direct_loss])]
        cost = rule_cost(16, rule)
        forwards = cost["unique_input_control_forwards_with_cache"]
        rows.append({"rule": rule, "state_loss": state_loss, "direct_loss": direct_loss,
                     "cached_forwards_per_state_fit": forwards, **cost, "selected_candidate": chosen})
        print(f"{rule:21} {state_loss:10.4f}  {direct_loss:11.4f}  {forwards:15d}  {chosen}")
    # A separate assessment pool; its observed controls are disjoint from input.
    assessment = rng.normal(size=(24, 3))
    assessment_fits = [0.8 * assessment + np.array([0.5, -0.1, shift]) for shift in (0.2, 0.3)]

    def independent_loss(prediction):
        return score_independent_effect(assessment, prediction, treated_mean, scales,
                                        input_rows=np.arange(8), observation_rows=np.arange(8, 24))

    assessment_losses = {
        "state": float(reduce_seed_losses([independent_loss(fit) for fit in assessment_fits])),
        "direct": independent_loss(direct_effect),
    }
    for row in rows:
        row["independent_assessment_loss"] = assessment_losses[row["selected_candidate"]]
        print(f"{row['rule']}: selected {row['selected_candidate']}, independent loss {row['independent_assessment_loss']:.4f}")
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / "allocation_scores.tsv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        assessment_result = {
            "example": "Synthetic demonstration; no biological replication or recommended design",
            "input_controls": 8, "observed_reference_controls": 16,
            "state_fit_aggregation": "Mean of per-fit squared losses for the same two fits used during selection",
            "candidate_independent_losses": assessment_losses,
            "selected_by_rule": {
                row["rule"]: {"candidate": row["selected_candidate"],
                               "independent_loss": row["independent_assessment_loss"]} for row in rows
            },
        }
        (args.output / "assessment.json").write_text(json.dumps(assessment_result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
