from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest


from reference_design.reporting_validation.reporting_candidate import RULES, assess, build_candidates, canonical, coverage_plans, freeze_publication, quantile
from reference_design.reporting_validation.synthetic_fixture import assessment, fixture


def run(tmp_path, changed=None):
    bundle, protocol, admission = fixture()
    seal = freeze_publication(bundle, protocol, admission, tmp_path / "frozen")
    supplied = assessment(seal, bundle, protocol)
    if changed:
        changed(supplied)
    path = tmp_path / "assessment.json"
    path.write_bytes(canonical(supplied))
    return assess(tmp_path / "frozen", seal, path, tmp_path / "result.json")


def test_hand_calculated_scores_and_crossfit_repeated_mean_identity():
    bundle, protocol, admission = fixture()
    rows, costs, bootstrap = build_candidates(bundle, protocol, admission)
    lookup = {(r["rule_id"], r["pair_id"]): r for r in rows}
    assert lookup[("score_margin", "pair_0")]["reliability_score"] == 3.75
    assert lookup[("repeated_splitting", "pair_0")]["reliability_score"] == 3.75
    assert lookup[("existing_refara_reporting", "pair_0")]["reliability_score"] == 3.25
    assert lookup[("independent_unit_resampling", "pair_0")]["reliability_score"] == 4
    for pair in bundle["roster"]:
        repeated = lookup[("repeated_splitting", pair["pair_id"])]
        crossfit = lookup[("cross_fitting", pair["pair_id"])]
        assert repeated["estimate"] == crossfit["estimate"]
        assert repeated["mean_margin_companion_score"] == crossfit["reliability_score"]
    assert all(not lookup[(rule, "pair_3")]["eligible"] for rule in RULES)
    assert costs["repeated_splitting"] == {**costs["cross_fitting"]}
    assert bootstrap["resampling_level"] == "whole_independent_unit"
    assert len(bootstrap["unit_ids"]) == 2


def test_complete_curve_and_fixed_denominator_golden(tmp_path):
    result = run(tmp_path)
    assert result["scientific_validation"] is False
    assert result["primary_M"] == 2 and result["primary_status"] == "evaluable"
    for row in result["curves"]:
        m = row["published_M"]
        assert row["fixed_roster_N"] == 4
        assert row["coverage"] == m / 4
        assert row["supported"] == int(m >= 1)
        assert row["reversed"] == int(m >= 2)
        assert row["unresolved"] == int(m >= 3)
        if m == 0:
            assert row["reversal_fraction"] is row["unresolved_fraction"] is None


def test_missing_assessment_is_unresolved_not_removed(tmp_path):
    def changed(value):
        value["intervals"][1] = {"pair_id": "pair_1", "status": "unassessable", "reason": "missing unit support"}
    result = run(tmp_path, changed)
    for row in result["curves"]:
        if row["published_M"] == 2:
            assert row["unresolved"] == row["unassessable_published"] == 1
            assert row["unresolved_fraction"] == .5 and row["coverage"] == .5


def test_negative_direction_orients_interval(tmp_path):
    bundle, protocol, admission = fixture()
    for row in bundle["records"]:
        if row["pair_id"] == "pair_0" and row["unit_id"].startswith("P"):
            for seed in row["seed_losses"]:
                seed["loss_b"] = 20 - seed["loss_b"]
    seal = freeze_publication(bundle, protocol, admission, tmp_path / "frozen")
    path = tmp_path / "a.json"
    path.write_bytes(canonical(assessment(seal, bundle, protocol)))
    result = assess(tmp_path / "frozen", seal, path, tmp_path / "result.json")
    rows = [r for r in result["classifications"] if r["pair_id"] == "pair_0" and r["M"] == 1]
    assert all(r["direction"] == "b" and r["status"] == "reversed" for r in rows)


def test_score_tie_blocks_cannot_be_split():
    _, protocol, _ = fixture()
    rows = [{"rule_id": rule, "pair_id": f"p{i}", "reliability_score": score,
             "candidate_direction": "a", "eligible": True, "reason": "eligible"}
            for rule in RULES for i, score in enumerate((3, 3, 1))]
    plans = coverage_plans(rows, protocol, 3)
    assert plans["common_achievable_M"] == [0, 2, 3]
    assert plans["primary_M"] == 0 and plans["primary_status"] == "not_evaluable"


def test_empirical_quantile_never_interpolates_to_release():
    assert quantile([0.] * 4 + [100.] * 60, .05) == 0.


def test_one_rule_allhold_retains_natural_results_and_secondaries(tmp_path):
    bundle, protocol, admission = fixture()
    for row in bundle["records"]:
        if row["unit_id"] == "C0" and row["kind"] == "allocation" and row["allocation_id"] == "1":
            for loss in row["seed_losses"]:
                loss["loss_b"] = 1000.
    seal = freeze_publication(bundle, protocol, admission, tmp_path / "frozen")
    path = tmp_path / "assessment.json"
    path.write_bytes(canonical(assessment(seal, bundle, protocol)))
    result = assess(tmp_path / "frozen", seal, path, tmp_path / "result.json")
    assert result["primary_status"] == "not_evaluable"
    natural = {r["rule_id"]: r for r in result["natural_release"]}
    assert natural["existing_refara_reporting"]["published_M"] == 0
    assert natural["score_margin"]["published_M"] == 3
    assert all(r["fixed_roster_N"] == 4 for r in natural.values())


def test_bootstrap_requires_same_crossfit_bank():
    b, p, a = fixture()
    row = next(r for r in b["records"] if r["rule_id"] == "independent_unit_resampling")
    row["seed_losses"][0]["loss_b"] += 1
    with pytest.raises(ValueError, match="identical scored allocation bank"):
        build_candidates(b, p, a)


def test_anchor_parity_is_not_optional():
    b, p, a = fixture()
    row = next(r for r in b["records"] if r["rule_id"] == "score_margin")
    row["seed_losses"][0]["loss_b"] += 1
    with pytest.raises(ValueError, match="anchor differs"):
        build_candidates(b, p, a)


@pytest.mark.parametrize("mutation,match", [
    (lambda b,p: b["units"][3].update(independence_block_id=b["units"][1]["independence_block_id"]), "crosses partitions"),
    (lambda b,p: b["records"].pop(), "calibration-only"),
    (lambda b,p: p["work_caps"]["cross_fitting"].update(scored_seed_pairs=0), "work exceeds"),
    (lambda b,p: b["records"][0]["seed_losses"].pop(), "Seed support"),
    (lambda b,p: b["records"][0]["roles"].update(model_input=b["records"][0]["roles"]["observation_center"]), "O-half"),
    (lambda b,p: b["records"][0]["seed_losses"][0].update(loss_b=float("nan")), "Nonfinite"),
    (lambda b,p: b["units"][1].update(native_experiment_id=b["units"][2]["native_experiment_id"]), "counted twice"),
])
def test_invalid_inputs_stop(mutation, match):
    b, p, a = fixture()
    mutation(b, p)
    with pytest.raises(ValueError, match=match):
        build_candidates(b, p, a)


def test_crossfit_cannot_be_old_allocation_relabel():
    b, p, a = fixture()
    next(r for r in b["records"] if r["rule_id"] == "cross_fitting")["kind"] = "allocation"
    with pytest.raises(ValueError, match="cannot relabel"):
        build_candidates(b, p, a)


def test_crossfit_duplicate_physical_fold_stops():
    b, p, a = fixture()
    rows = [r for r in b["records"] if r["rule_id"] == "cross_fitting" and r["unit_id"] == "P0" and r["pair_id"] == "pair_0"]
    rows[1]["roles"] = deepcopy(rows[0]["roles"])
    rows[1]["seed_losses"] = deepcopy(rows[0]["seed_losses"])
    with pytest.raises(ValueError, match="partition physical controls"):
        build_candidates(b, p, a)


def test_real_data_hard_gate():
    b, p, a = fixture()
    p["mode"] = "real"
    with pytest.raises(ValueError, match="rejects real data"):
        build_candidates(b, p, a)


def test_freeze_tamper_stops_before_assessment_open(tmp_path):
    b, p, a = fixture()
    seal = freeze_publication(b, p, a, tmp_path / "frozen")
    path = tmp_path / "frozen/PUBLICATION_FROZEN.json"
    value = json.loads(path.read_text())
    value["protocol"]["score_floor"] = 100
    path.write_bytes(canonical(value))
    with pytest.raises(ValueError, match="freeze was modified"):
        assess(tmp_path / "frozen", seal, tmp_path / "never_opened.json", tmp_path / "result.json")
    assert not (tmp_path / "frozen/ASSESSMENT_OPENED.json").exists()


def test_cannot_reopen_assessment_or_tune_after_open(tmp_path):
    run(tmp_path)
    frozen = tmp_path / "frozen/PUBLICATION_FROZEN.json"
    import hashlib
    with pytest.raises(FileExistsError):
        assess(frozen.parent, hashlib.sha256(frozen.read_bytes()).hexdigest(), tmp_path / "assessment.json", tmp_path / "other.json")


def test_hash_and_parse_use_same_single_read_for_both_inputs(tmp_path, monkeypatch):
    import hashlib
    bundle, protocol, admission = fixture()
    frozen = tmp_path / "frozen"
    seal = freeze_publication(bundle, protocol, admission, frozen)
    supplied_path = tmp_path / "assessment.json"
    supplied_bytes = canonical(assessment(seal, bundle, protocol))
    supplied_path.write_bytes(supplied_bytes)
    paths = {frozen / "PUBLICATION_FROZEN.json", supplied_path}
    reads = dict.fromkeys(paths, 0)
    original = Path.read_bytes
    def read_once(path):
        if path in reads:
            reads[path] += 1
            assert reads[path] == 1, "Input must not be reread after parsing"
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", read_once)
    report = assess(frozen, seal, supplied_path, tmp_path / "result.json")
    assert set(reads.values()) == {1}
    assert report["assessment_sha256"] == hashlib.sha256(supplied_bytes).hexdigest()


@pytest.mark.parametrize("change,match", [
    (lambda v: v["intervals"].pop(), "complete fixed roster"),
    (lambda v: v.update(unit_ids=["A0"]), "unit support"),
    (lambda v: v.update(multiplicity_policy="posthoc"), "procedure changed"),
    (lambda v: v["intervals"][0].update(lower=3), "Invalid interval"),
    (lambda v: v["intervals"][0].update(lower=2**53+1, upper=2**53), "Invalid interval"),
])
def test_assessment_support_and_interval_contract_locked(tmp_path, change, match):
    with pytest.raises(ValueError, match=match):
        run(tmp_path, change)
