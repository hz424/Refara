"""Numerical aggregation and failure controls for the four-group claim capsule."""
import importlib.util
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CAPSULE = ROOT / "evidence/model_comparisons"
SPEC = importlib.util.spec_from_file_location("model_comparison_replay", CAPSULE / "replay.py")
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def test_supplied_seed_unit_scores_reconstruct_current_four_group_comparisons():
    tables, report = replay.recompute()
    assert report["input_rows"] == 15360
    assert len(tables["MODEL_SCORES.tsv"]) == 640
    assert len(tables["SEED_MODEL_SCORES.tsv"]) == 1920
    assert len(tables["UNIT_MODEL_SCORES.tsv"]) == 5120
    assert len(tables["PAIR_MARGINS.tsv"]) == 2240
    assert len(tables["POLICY_SUMMARY.tsv"]) == 80
    assert report["checks"]["maximum_absolute_value_difference"] <= 1e-12
    assert not report["fresh_model_fitting"] and not report["fresh_prediction_scoring"]
    # Current evidence is tested as a regression, not assumed by aggregate().
    assert all(r["reversals"] == (10 if r["policy_code"] in "POD" else 0)
               for r in report["maximum_depth_comparisons"])


def test_score_before_seed_averaging_and_equal_unit_aggregation():
    values = replay.load_values(replay.read_rows(CAPSULE / "data/scored_unit_utilities.tsv"))
    group = ("gse162632", "original8", 8, "S", "CPA")
    # These are errors squared separately before averaging. Squaring their mean
    # instead would give 49/9, whereas the required mean score is -7.
    for seed, utility in zip(replay.SEEDS, (-1.0, -4.0, -16.0)):
        for unit in replay.UNITS:
            values[group[0], group[1], seed, group[2], unit, group[3], group[4]] = utility
    tables = replay.aggregate(values)
    row = next(r for r in tables["MODEL_SCORES.tsv"]
               if (r["dataset"], r["regime"], r["depth"], r["policy_code"], r["family"]) == group)
    assert row["utility"] == -7.0
    for seed in replay.SEEDS:
        values[group[0], group[1], seed, group[2], "root_00", group[3], group[4]] -= 8
    changed = replay.aggregate(values)
    row = next(r for r in changed["MODEL_SCORES.tsv"]
               if (r["dataset"], r["regime"], r["depth"], r["policy_code"], r["family"]) == group)
    assert row["utility"] == -8.0


def test_aggregator_reports_a_changed_order_without_requiring_published_direction():
    values = replay.load_values(replay.read_rows(CAPSULE / "data/scored_unit_utilities.tsv"))
    for seed in replay.SEEDS:
        for unit in replay.UNITS:
            values["gse162632", "original8", seed, 8, unit, "S", "NC"] = 0.0
            values["gse162632", "original8", seed, 8, unit, "D", "NC"] = -100.0
    tables = replay.aggregate(values)
    nc_pairs = [r for r in tables["PAIR_MARGINS.tsv"] if
                (r["dataset"], r["regime"], r["depth"], r["policy_code"], r["family_a"])
                == ("gse162632", "original8", 8, "D", "NC")]
    assert len(nc_pairs) == 7 and all(r["status_exact"] == "reversal" for r in nc_pairs)
    with pytest.raises(ValueError, match="Published value differs"):
        replay.compare_reference(tables)


def test_exact_ties_and_reversal_direction_are_not_tolerance_zeroed():
    assert replay.ranks({"a": 0.0, "b": 0.0, "c": -1.0}) == {"a": 1.5, "b": 1.5, "c": 3.0}
    assert replay.classify(1e-15, -1e-15) == "reversal"
    assert replay.classify(0.0, 0.0) == "tie_both"
    assert replay.classify(0.0, -1.0) == "tie_transition"
    assert replay.classify(-1.0, -2.0) == "stable_order"


@pytest.mark.parametrize("change", ["missing", "duplicate", "nan", "positive", "unknown_policy"])
def test_bad_score_axes_and_nonfinite_values_fail(change):
    rows = replay.read_rows(CAPSULE / "data/scored_unit_utilities.tsv")
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(dict(rows[0]))
    elif change in ("nan", "positive"):
        rows[0]["utility"] = "nan" if change == "nan" else "1"
    else:
        rows[0]["policy_code"] = "unknown"
    with pytest.raises(ValueError):
        replay.load_values(rows)


def test_source_tampering_is_rejected_before_computation(tmp_path):
    copy = tmp_path / "capsule"
    shutil.copytree(CAPSULE, copy)
    path = copy / "data/scored_unit_utilities.tsv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Input hash differs"):
        replay.recompute(copy)


def test_cli_emits_numerical_tables_and_refuses_existing_output(tmp_path):
    out = tmp_path / "numeric"
    replay.main(["--output", str(out)])
    assert set(p.name for p in out.iterdir()) == {
        "MODEL_SCORES.tsv", "SEED_MODEL_SCORES.tsv", "UNIT_MODEL_SCORES.tsv",
        "PAIR_MARGINS.tsv", "POLICY_SUMMARY.tsv", "REPLAY.json"}
    with pytest.raises(ValueError, match="new output"):
        replay.main(["--output", str(out)])
