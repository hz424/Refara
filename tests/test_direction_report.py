from __future__ import annotations

from dataclasses import FrozenInstanceError
from fractions import Fraction
import csv
import json
from pathlib import Path
import runpy
import sys

import pytest

from reference_design.direction_report import Comparison, ReportedDirection, summarize_directions


def test_hand_calculated_counts_keep_unresolved_in_the_published_denominator():
    roster = [Comparison(f"task_{i}", "A", "B") for i in range(7)]
    decisions = [("a", (1, 3)), ("b", (-4, -2)), ("a", (-3, -1)),
                 ("b", (-1, 1)), ("a", (0, 2)), ("hold", None), ("hold", (2, 3))]
    reports = [ReportedDirection(pair, direction, interval)
               for pair, (direction, interval) in zip(roster, decisions)]
    result = summarize_directions(roster, reversed(reports))
    assert result["counts"] == dict(candidates=7, published=5, supported=2, reversed=1, unresolved=2, held=2)
    assert result["coverage"] == 5 / 7
    assert result["reversal_fraction"] == 1 / 5
    assert result["unresolved_fraction"] == 2 / 5
    assert [row["task"] for row in result["comparisons"]] == [pair.task for pair in roster]
    assert result["comparisons"][1]["oriented_interval"] == [2., 4.]
    assert result["comparisons"][6]["status"] == "hold"
    assert result["comparisons"][6]["oriented_interval"] is None


@pytest.mark.parametrize("direction,interval,status", [
    ("a", (1, 4), "supported"), ("b", (1, 4), "reversed"),
    ("a", (-4, -1), "reversed"), ("b", (-4, -1), "supported"),
    ("a", (0, 4), "unresolved"), ("b", (0, 4), "unresolved"),
    ("a", (-4, 0), "unresolved"), ("b", (-4, 0), "unresolved"),
    ("a", (-1, 1), "unresolved"), ("b", (-1, 1), "unresolved"),
    ("a", (0, 0), "unresolved"), ("b", (0, 0), "unresolved"),
    ("a", (1, 1), "supported"), ("b", (-1, -1), "supported"),
    ("a", (1e-300, 2e-300), "supported"),
])
def test_strict_directional_interval_classification(direction, interval, status):
    pair = Comparison("task", "A", "B")
    report = summarize_directions([pair], [ReportedDirection(pair, direction, interval)])
    assert report["comparisons"][0]["status"] == status


def test_all_holds_have_zero_coverage_and_undefined_conditional_fractions():
    roster = [Comparison("task_1", "A", "B"), Comparison("task_2", "A", "B")]
    reports = [ReportedDirection(roster[0], "hold"), ReportedDirection(roster[1], "hold", (-2, -1))]
    result = summarize_directions(roster, reports)
    assert result["counts"]["published"] == 0
    assert result["coverage"] == 0
    assert result["reversal_fraction"] is None and result["unresolved_fraction"] is None
    assert "not verified" in result["interpretation"]


def test_identifiers_and_interval_are_immutable():
    pair = Comparison("task", "A", "B")
    bounds = [1., 2.]
    record = ReportedDirection(pair, "a", bounds)
    bounds[0] = -1.
    assert record.interval == (1., 2.)
    with pytest.raises(FrozenInstanceError):
        pair.task = "changed"
    with pytest.raises(FrozenInstanceError):
        record.direction = "b"


@pytest.mark.parametrize("interval", [(), (1,), (1, 2, 3), (2, 1),
    (float("nan"), 1), (0, float("inf")), (-float("inf"), 0),
    (True, 1), (0, False), ("0", "1"), (0j, 1), 3., "01", b"01",
    {0: "lower", 1: "upper"}, (0, 10**1000), {0, 1}, frozenset((0, 1))])
def test_invalid_intervals_are_rejected_even_for_holds(interval):
    pair = Comparison("task", "A", "B")
    for direction in ("a", "b", "hold"):
        with pytest.raises(ValueError):
            ReportedDirection(pair, direction, interval)


@pytest.mark.parametrize("direction", ["a", "b"])
def test_published_directions_require_an_interval(direction):
    with pytest.raises(ValueError, match="requires an interval"):
        ReportedDirection(Comparison("task", "A", "B"), direction)


def test_float_rounding_cannot_hide_an_inverted_exact_integer_interval():
    # Both values convert to the same float, but the supplied interval is inverted.
    with pytest.raises(ValueError, match="lower bound exceeds"):
        ReportedDirection(Comparison("task", "A", "B"), "a", (2**53 + 1, 2**53))


@pytest.mark.parametrize("interval", [
    (Fraction(1, 10**400), Fraction(2, 10**400)),
    (Fraction(-2, 10**400), Fraction(-1, 10**400)),
    (0, Fraction(1, 10**400)),
])
def test_nonzero_bounds_cannot_underflow_and_change_the_zero_boundary(interval):
    for direction in ("a", "b", "hold"):
        with pytest.raises(ValueError, match="underflows to zero"):
            ReportedDirection(Comparison("task", "A", "B"), direction, interval)


def test_ordinary_fraction_bounds_and_unicode_labels_serialize_to_plain_json():
    pair = Comparison("CD4 T/刺激/6h", "模型 A", "模型 B")
    report = ReportedDirection(pair, "b", (Fraction(-1, 3), Fraction(-1, 7)))
    result = summarize_directions([pair], [report])
    decoded = json.loads(json.dumps(result, ensure_ascii=False, allow_nan=False))
    assert decoded == result
    assert decoded["comparisons"][0]["status"] == "supported"
    assert decoded["comparisons"][0]["oriented_interval"] == [1 / 7, 1 / 3]
    assert decoded["comparisons"][0]["task"] == pair.task


def test_all_hold_serialization_preserves_null_fractions():
    pair = Comparison("task", "A", "B")
    decoded = json.loads(json.dumps(summarize_directions([pair], [ReportedDirection(pair, "hold")]), allow_nan=False))
    assert decoded["reversal_fraction"] is None
    assert decoded["unresolved_fraction"] is None
    assert decoded["comparisons"][0]["interval"] is None


@pytest.mark.parametrize("direction", [None, "A", "model_a", "tie", "", 0, 1, True])
def test_unknown_or_implicit_directions_are_rejected(direction):
    with pytest.raises(ValueError, match="Direction must"):
        ReportedDirection(Comparison("task", "A", "B"), direction, (0, 1))


@pytest.mark.parametrize("fields", [("", "A", "B"), ("task", "", "B"),
    ("task", "A", "A"), ("task ", "A", "B"), (None, "A", "B"),
    ("task\nother", "A", "B"), ("task", "A", 2)])
def test_invalid_comparison_identifiers_are_rejected(fields):
    with pytest.raises(ValueError):
        Comparison(*fields)


def test_roster_requires_all_candidates_and_rejects_duplicate_or_unknown_reports():
    first, second = Comparison("t1", "A", "B"), Comparison("t2", "A", "B")
    one, two = ReportedDirection(first, "a", (1, 2)), ReportedDirection(second, "hold")
    with pytest.raises(ValueError, match="explicit"):
        summarize_directions([first, second], [one])
    with pytest.raises(ValueError, match="Duplicate report"):
        summarize_directions([first, second], [one, two, one])
    with pytest.raises(ValueError, match="not in the roster"):
        summarize_directions([first], [one, two])
    reversed_pair = Comparison("t1", "B", "A")
    with pytest.raises(ValueError, match="duplicate task/model pair"):
        summarize_directions([first, reversed_pair], [])
    with pytest.raises(ValueError, match="duplicate task/model pair"):
        summarize_directions([first, first], [])
    with pytest.raises(ValueError, match="orientation"):
        summarize_directions([first], [ReportedDirection(reversed_pair, "hold")])


@pytest.mark.parametrize("roster,reports", [([], []), (None, []), (["task"], []),
    ([Comparison("task", "A", "B")], ["hold"]), ([Comparison("task", "A", "B")], None)])
def test_empty_or_malformed_collections_are_rejected(roster, reports):
    with pytest.raises(ValueError):
        summarize_directions(roster, reports)


def test_example_writes_json_tsv_and_refuses_to_overwrite(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "examples/direction_report.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--output", str(tmp_path)])
    runpy.run_path(str(script), run_name="__main__")
    path = tmp_path / "direction_summary.json"
    contents = path.read_bytes()
    result = json.loads(contents)
    assert result["counts"] == dict(candidates=4, published=3, supported=1, reversed=1, unresolved=1, held=1)
    assert result["coverage"] == .75
    assert result["reversal_fraction"] == result["unresolved_fraction"] == 1 / 3
    with (tmp_path / "direction_comparisons.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 4 and rows[-1]["interval_lower"] == ""
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(script), run_name="__main__")
    assert error.value.code == 2 and path.read_bytes() == contents


@pytest.mark.parametrize("collision", ["file", "nonempty_directory", "symlink", "dangling_symlink"])
def test_example_rejects_output_collisions_without_changing_existing_content(tmp_path, monkeypatch, collision):
    script = Path(__file__).resolve().parents[1] / "examples/direction_report.py"
    output = tmp_path / "result"
    sentinel = None
    if collision == "file":
        sentinel = output
        sentinel.write_text("existing")
    elif collision == "nonempty_directory":
        output.mkdir()
        sentinel = output / "unrelated.txt"
        sentinel.write_text("existing")
    else:
        target = tmp_path / "target"
        if collision == "symlink":
            target.mkdir()
        output.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(sys, "argv", [str(script), "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(script), run_name="__main__")
    assert error.value.code == 2
    if sentinel is not None:
        assert sentinel.read_text() == "existing"
    assert not (output / "direction_summary.json").exists()


def test_example_does_not_publish_partial_results_when_staging_fails(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "examples/direction_report.py"
    output = tmp_path / "result"
    original_write = Path.write_bytes

    def fail_on_tsv(path, data):
        if path.name == "direction_comparisons.tsv":
            raise OSError("simulated disk failure")
        return original_write(path, data)

    monkeypatch.setattr(Path, "write_bytes", fail_on_tsv)
    monkeypatch.setattr(sys, "argv", [str(script), "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(script), run_name="__main__")
    assert error.value.code == 2
    assert not output.exists()
    assert not list(tmp_path.glob(".reference-design-*"))


def test_example_preserves_files_created_after_initial_destination_check(tmp_path, monkeypatch):
    from reference_design import cli

    script = Path(__file__).resolve().parents[1] / "examples/direction_report.py"
    output = tmp_path / "result"
    original_destination = cli._destination
    calls = 0

    def competing_writer(path):
        nonlocal calls
        calls += 1
        if calls == 3:  # Last check, after both files have been staged.
            path.mkdir()
            (path / "other_writer.txt").write_text("keep")
        original_destination(path)

    monkeypatch.setattr(cli, "_destination", competing_writer)
    monkeypatch.setattr(sys, "argv", [str(script), "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(script), run_name="__main__")
    assert error.value.code == 2
    assert (output / "other_writer.txt").read_text() == "keep"
    assert not (output / "direction_summary.json").exists()
