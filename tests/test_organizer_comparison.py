"""Reporting comparisons use frozen inputs and explicit finite-sample losses."""
import csv
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "evidence/organizer/compare.py"
API = runpy.run_path(str(SCRIPT))
COLUMNS = ["case", "unit", "budget", "realization", "model", "score"]


def write_table(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_table(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def example(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    analysis = dict(schema_version=1, minimum_gaps=[0, .005, .01, .02],
                    primary_minimum_gap=.01, coverage_grid=[.1, .25, .5, .75, .9, 1],
                    numerical_tolerance=1e-12)
    config = dict(schema_version=1, target="heldout_effect",
                  metric={"name": "synthetic_loss", "direction": "lower"}, models=["A", "B", "C"],
                  budgets=[dict(id="small", control_cells_available=3, input_cells=1,
                                observation_cells=2, description="Three synthetic controls per realization")],
                  roles={field: "Synthetic fixture" for field in API["org"].ROLE_FIELDS},
                  anchor_realization="r0", minimum_gap=.01, numerical_tolerance=1e-12,
                  evaluation_scope="Synthetic finite allocations", development_scores="development.tsv",
                  assessment_anchors="anchors.tsv")

    def rows(case, unit, anchor, repeat):
        return [dict(case=case, unit=unit, budget="small", realization=realization, model=model, score=score)
                for realization, values in (("r0", anchor), ("r1", repeat))
                for model, score in zip(config["models"], values)]

    development = rows("dev", "development_pool", [5, 6, 5], [5, 6, 7])
    assessment = rows("p", "donor_1", [5, 8, 9], [5, 8, 4]) + rows("q", "donor_2", [5, 7, 6], [5, 7, 8])
    write_table(tmp_path / "development.tsv", development)
    write_table(tmp_path / "anchors.tsv", [row for row in assessment if row["realization"] == "r0"])
    write_table(tmp_path / "assessment.tsv", assessment)
    write_json(tmp_path / "manifest.json", config)
    write_json(tmp_path / "protocol.json", analysis)
    return tmp_path / "manifest.json", tmp_path / "protocol.json", tmp_path / "assessment.tsv"


def row_at(rows, **match):
    return next(row for row in rows if all(str(row[key]) == str(value) for key, value in match.items()))


def test_manual_calibration_rules_and_discrete_aurc(tmp_path):
    manifest, protocol, assessment = example(tmp_path / "inputs")
    plan = API["freeze"](manifest, protocol, tmp_path / "freeze")
    assert {(row["model_a"], row["model_b"]): row["B"] for row in plan["calibration"]} == {
        ("A", "B"): 0, ("A", "C"): 2, ("B", "C"): 2}
    assert {row["threshold"] for row in plan["development_thresholds"]} == {1}
    result = API["check"](tmp_path / "freeze/plan.json", assessment, tmp_path / "check")
    primary = {row["rule"]: row for row in result["rule_summary"] if row["minimum_gap"] == .01}
    assert (primary["single_split"]["selected"], primary["single_split"]["unsupported"]) == (6, 3)
    for rule in ("repeated_primary", "development_margin"):
        assert (primary[rule]["selected"], primary[rule]["unsupported"]) == (3, 1)
    # Anchor order has errors 1,0,0,1,0,1; calibrated order has 0,1,0,1,0,1.
    areas = {row["ranking"]: row["aurc"] for row in result["aurc"] if row["minimum_gap"] == .01}
    assert areas["anchor_margin"] == pytest.approx(97 / 180)
    assert areas["calibrated_margin"] == pytest.approx(67 / 180)
    curve = read_table(tmp_path / "check/risk_coverage.tsv")
    third = row_at(curve, ranking="anchor_margin", minimum_gap=.01, selected=3)
    assert float(third["risk"]) == pytest.approx(1 / 3)
    assert float(third["unit_macro_risk"]) == .25
    assert float(third["unit_macro_coverage"]) == .5
    for method in API["RANKINGS"]:
        full = row_at(curve, ranking=method, minimum_gap=.01, selected=6)
        assert float(full["risk"]) == .5
        assert int(full["strict_reversals"]) == 3
    assert result["costs"]["development_realizations_per_case_budget"] == 2
    assert result["costs"]["assessment_realizations_per_case_budget"] == 2
    assert result["costs"]["distinct_controls_across_realizations"] is None
    assert not list(tmp_path.rglob("*.md"))


def test_grid_ceil_ties_and_pair_composition(tmp_path):
    manifest, protocol, assessment = example(tmp_path / "inputs")
    API["run"](manifest, assessment, protocol, tmp_path / "result")
    grid = read_table(tmp_path / "result/check/coverage_grid.tsv")
    anchor = [row for row in grid if row["ranking"] == "anchor_margin" and row["minimum_gap"] == "0.01"]
    assert [int(row["selected"]) for row in anchor] == [1, 2, 3, 5, 6, 6]
    cutoff = row_at(anchor, requested_coverage=.75)
    assert int(cutoff["cutoff_tie_count"]) == 3
    assert int(cutoff["cutoff_tie_selected"]) == 2
    assert cutoff["cutoff_splits_tie"] == "True"
    pairs = read_table(tmp_path / "result/check/coverage_pairs.tsv")
    chosen = [row for row in pairs if row["ranking"] == "calibrated_margin"
              and row["minimum_gap"] == "0.01" and row["requested_coverage"] == "0.5"]
    assert {(row["model_a"], row["model_b"]): int(row["selected"]) for row in chosen} == {
        ("A", "B"): 2, ("A", "C"): 1, ("B", "C"): 0}
    empty = row_at(chosen, model_a="B", model_b="C")
    assert empty["risk"] == "" and empty["unit_macro_risk"] == ""


def test_assessment_outcomes_cannot_change_frozen_selection(tmp_path):
    manifest, protocol, assessment = example(tmp_path / "inputs")
    plan = API["freeze"](manifest, protocol, tmp_path / "freeze")
    original_bytes = (tmp_path / "freeze/plan.json").read_bytes()
    first = API["check"](tmp_path / "freeze/plan.json", assessment, tmp_path / "first")
    rows = read_table(assessment)
    anchors = {(row["case"], row["model"]): row["score"] for row in rows if row["realization"] == "r0"}
    for row in rows:
        if row["realization"] != "r0":
            row["score"] = anchors[(row["case"], row["model"])]
    write_table(assessment, rows)
    second = API["check"](tmp_path / "freeze/plan.json", assessment, tmp_path / "second")
    assert (tmp_path / "freeze/plan.json").read_bytes() == original_bytes
    assert [row["selected"] for row in first["rule_summary"]] == [row["selected"] for row in second["rule_summary"]]
    assert sum(row["unsupported"] for row in first["rule_summary"]) > 0
    assert sum(row["unsupported"] for row in second["rule_summary"]) == 0
    assert API["freeze"](manifest, protocol, tmp_path / "second_freeze") == plan


def test_run_publishes_plan_receipt_before_reading_assessment(tmp_path, monkeypatch):
    manifest, protocol, assessment = example(tmp_path / "inputs")
    original = API["org"]._scores
    reads = []

    def read(path, *args, **kwargs):
        if Path(path) == assessment:
            plan = tmp_path / "result/freeze/plan.json"
            receipt = json.loads((plan.parent / "receipt.json").read_text())
            assert receipt["outputs"]["plan.json"] == hashlib.sha256(plan.read_bytes()).hexdigest()
            reads.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(API["org"], "_scores", read)
    API["run"](manifest, assessment, protocol, tmp_path / "result")
    assert reads == [str(assessment)]


def test_freeze_does_not_require_assessment_and_can_be_relocated(tmp_path):
    manifest, protocol, assessment = example(tmp_path / "inputs")
    hidden = tmp_path / "hidden.tsv"
    assessment.rename(hidden)
    API["freeze"](manifest, protocol, tmp_path / "freeze")
    moved = tmp_path / "moved"
    shutil.move(str(tmp_path / "freeze"), moved)
    shutil.rmtree(manifest.parent)
    result = API["check"](moved / "plan.json", hidden, tmp_path / "check")
    assert result["assessment_cases"] == 2


def test_row_model_and_realization_order_do_not_affect_analysis(tmp_path):
    manifest, protocol, assessment = example(tmp_path / "inputs")
    original = API["freeze"](manifest, protocol, tmp_path / "first")
    config = json.loads(manifest.read_text())
    config["models"].reverse()
    write_json(manifest, config)
    for filename in ("development.tsv", "anchors.tsv", "assessment.tsv"):
        path = manifest.parent / filename
        write_table(path, list(reversed(read_table(path))))
    reordered = API["freeze"](manifest, protocol, tmp_path / "second")
    for field in ("calibration", "development_thresholds", "comparisons", "rankings", "selections",
                  "development_statistics", "assessment_anchors", "development_realizations"):
        assert reordered[field] == original[field]
    first = API["check"](tmp_path / "first/plan.json", assessment, tmp_path / "first_check")
    second = API["check"](tmp_path / "second/plan.json", assessment, tmp_path / "second_check")
    assert first["rule_summary"] == second["rule_summary"]
    assert first["aurc"] == second["aurc"]


def test_higher_is_better_metric_preserves_the_equivalent_decisions(tmp_path):
    manifest, protocol, assessment = example(tmp_path / "inputs")
    lower = API["run"](manifest, assessment, protocol, tmp_path / "lower")
    config = json.loads(manifest.read_text())
    config["metric"]["direction"] = "higher"
    write_json(manifest, config)
    for filename in ("development.tsv", "anchors.tsv", "assessment.tsv"):
        path = manifest.parent / filename
        rows = read_table(path)
        for row in rows:
            row["score"] = -float(row["score"])
        write_table(path, rows)
    higher = API["run"](manifest, assessment, protocol, tmp_path / "higher")
    assert higher["rule_summary"] == lower["rule_summary"]
    assert higher["aurc"] == lower["aurc"]


def test_development_and_assessment_counts_are_recorded_separately(tmp_path):
    manifest, protocol, assessment = example(tmp_path / "inputs")
    rows = read_table(assessment)
    rows += [dict(row, realization="r2") for row in rows if row["realization"] == "r1"]
    write_table(assessment, rows)
    result = API["run"](manifest, assessment, protocol, tmp_path / "result")
    assert result["development_realizations"] == ["r0", "r1"]
    assert result["assessment_realizations"] == ["r0", "r1", "r2"]
    assert result["costs"]["assessment_realizations_per_case_budget"] == 3


def test_staged_command_line(tmp_path):
    manifest, protocol, assessment = example(tmp_path / "inputs")
    frozen = tmp_path / "frozen"
    commands = [
        ["freeze", "--manifest", str(manifest), "--protocol", str(protocol), "--output", str(frozen)],
        ["check", "--plan", str(frozen / "plan.json"), "--assessment", str(assessment),
         "--output", str(tmp_path / "checked")],
    ]
    for arguments in commands:
        result = subprocess.run([sys.executable, str(SCRIPT), *arguments],
                                cwd=tmp_path, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["stage"] == arguments[0]
    assert (tmp_path / "checked/aurc.tsv").is_file()


def test_same_pair_rankings_are_identical(tmp_path):
    manifest, protocol, _ = example(tmp_path / "inputs")
    plan = API["freeze"](manifest, protocol, tmp_path / "freeze")
    lookup = {row["comparison_id"]: row for row in plan["comparisons"]}
    for a, b in (("A", "B"), ("A", "C"), ("B", "C")):
        orders = [[lookup[index]["case"] for index in ranking["comparison_ids"]
                   if (lookup[index]["model_a"], lookup[index]["model_b"]) == (a, b)]
                  for ranking in plan["rankings"]]
        assert orders[0] == orders[1]


def test_no_direction_is_unsupported_without_an_invented_winner(tmp_path):
    manifest, protocol, assessment = example(tmp_path / "inputs")
    for path in (manifest.parent / "anchors.tsv", assessment):
        rows = read_table(path)
        for row in rows:
            if row["case"] == "q" and row["model"] == "C" and row["realization"] == "r0":
                row["score"] = "5"
        write_table(path, rows)
    plan = API["freeze"](manifest, protocol, tmp_path / "freeze")
    comparison = row_at(plan["comparisons"], case="q", model_a="A", model_b="C")
    assert comparison["direction"] == 0 and comparison["favored_model"] is None
    assert all(comparison["comparison_id"] not in row["comparison_ids"] for row in plan["selections"])
    API["check"](tmp_path / "freeze/plan.json", assessment, tmp_path / "check")
    result = row_at(read_table(tmp_path / "check/risk_coverage.tsv"),
                    ranking="anchor_margin", minimum_gap=.01, selected=6)
    assert int(result["no_direction"]) == 1 and int(result["unsupported"]) == 4
    assert int(result["strict_reversals"]) == 3


@pytest.mark.parametrize("margin,unsupported,reversal", [(1e-12, True, False),
                                                          (-1e-12, True, False),
                                                          (-2e-12, True, True),
                                                          (2e-12, False, False)])
def test_risk_and_reversal_tolerance_boundaries(tmp_path, margin, unsupported, reversal):
    manifest, protocol, assessment = example(tmp_path / "inputs")
    config = json.loads(manifest.read_text())
    config["models"] = ["A", "B"]
    write_json(manifest, config)
    development = [dict(case="dev", unit="dev_pool", budget="small", realization=r,
                        model=m, score=score) for r in ("r0", "r1") for m, score in (("A", 0), ("B", 1))]
    rows = [dict(case="assessment", unit="donor", budget="small", realization=r,
                 model=m, score=score) for r in ("r0", "r1")
            for m, score in (("A", 0), ("B", 2 if r == "r0" else margin))]
    write_table(manifest.parent / "development.tsv", development)
    write_table(manifest.parent / "anchors.tsv", [row for row in rows if row["realization"] == "r0"])
    write_table(assessment, rows)
    result = API["run"](manifest, assessment, protocol, tmp_path / "result")
    summary = row_at(result["rule_summary"], minimum_gap=0, rule="single_split")
    assert summary["selected"] == 1
    assert summary["unsupported"] == int(unsupported)
    assert summary["strict_reversals"] == int(reversal)


@pytest.mark.parametrize("failure", ["overlap", "empty_development", "missing_model", "duplicate_score",
                                      "nonfinite", "different_realizations", "primary_gap", "tolerance",
                                      "model_type", "duplicate_model"])
def test_invalid_planning_inputs_publish_nothing(tmp_path, failure):
    manifest, protocol, _ = example(tmp_path / "inputs")
    path = manifest.parent / "development.tsv"
    rows = read_table(path)
    if failure == "overlap":
        for row in rows:
            row["case"] = "p"
    elif failure == "empty_development":
        rows = []
    elif failure == "missing_model":
        rows.pop()
    elif failure == "duplicate_score":
        rows.append(rows[0])
    elif failure == "nonfinite":
        rows[0]["score"] = "nan"
    elif failure == "different_realizations":
        rows += [dict(row, case="dev2", realization="r2" if row["realization"] == "r1" else "r0")
                 for row in rows]
    else:
        config = json.loads(manifest.read_text())
        if failure == "model_type":
            config["models"] = "ABC"
        elif failure == "duplicate_model":
            config["models"].append("A")
        else:
            config["minimum_gap" if failure == "primary_gap" else "numerical_tolerance"] *= 2
        write_json(manifest, config)
    write_table(path, rows)
    with pytest.raises(ValueError):
        API["freeze"](manifest, protocol, tmp_path / "result")
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize("failure", ["anchor", "unit", "case", "realization", "missing_model", "nonfinite"])
def test_invalid_assessment_cannot_publish_results(tmp_path, failure):
    manifest, protocol, assessment = example(tmp_path / "inputs")
    API["freeze"](manifest, protocol, tmp_path / "freeze")
    rows = read_table(assessment)
    if failure == "anchor":
        rows[0]["score"] = "99"
    elif failure == "unit":
        for row in rows:
            row["unit"] = "unrelated"
    elif failure == "case":
        for row in rows:
            if row["case"] == "p":
                row["case"] = "unplanned"
    elif failure == "realization":
        for row in rows:
            if row["case"] == "p" and row["realization"] == "r1":
                row["realization"] = "another"
    elif failure == "missing_model":
        rows.pop()
    else:
        rows[-1]["score"] = "inf"
    write_table(assessment, rows)
    with pytest.raises(ValueError):
        API["check"](tmp_path / "freeze/plan.json", assessment, tmp_path / "result")
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize("update_receipt", [False, True])
def test_changed_frozen_selection_is_rejected(tmp_path, update_receipt):
    manifest, protocol, assessment = example(tmp_path / "inputs")
    API["freeze"](manifest, protocol, tmp_path / "freeze")
    path = tmp_path / "freeze/plan.json"
    plan = json.loads(path.read_text())
    plan["selections"][0]["comparison_ids"] = []
    write_json(path, plan)
    if update_receipt:
        receipt_path = path.parent / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["outputs"]["plan.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
        write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="Frozen"):
        API["check"](path, assessment, tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_changed_development_changes_calibration_without_using_assessment(tmp_path):
    manifest, protocol, _ = example(tmp_path / "inputs")
    original = API["freeze"](manifest, protocol, tmp_path / "first")
    path = manifest.parent / "development.tsv"
    rows = read_table(path)
    for row in rows:
        if row["model"] == "C" and row["realization"] == "r1":
            row["score"] = "9"
    write_table(path, rows)
    changed = API["freeze"](manifest, protocol, tmp_path / "second")
    assert row_at(original["calibration"], model_a="A", model_b="C")["B"] == 2
    assert row_at(changed["calibration"], model_a="A", model_b="C")["B"] == 4
    assert original["assessment_anchors"] == changed["assessment_anchors"]
    assert original["selections"] != changed["selections"]


def test_all_hold_and_all_release_have_correct_denominators(tmp_path):
    manifest, protocol, assessment = example(tmp_path / "inputs")
    analysis = json.loads(protocol.read_text())
    analysis["minimum_gaps"].append(100)
    write_json(protocol, analysis)
    API["run"](manifest, assessment, protocol, tmp_path / "result")
    rows = read_table(tmp_path / "result/check/rule_summary.tsv")
    for row in rows:
        if row["minimum_gap"] == "100":
            assert int(row["selected"]) == 0 and row["risk"] == "" and row["unit_macro_risk"] == ""
            assert int(row["units_with_selections"]) == 0 and float(row["unit_macro_coverage"]) == 0
    full = row_at(rows, rule="single_split", minimum_gap=0)
    assert int(full["selected"]) == int(full["comparisons"]) == 6
