"""The public organizer example retains the recorded split and paper outcomes."""
import csv
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
EXAMPLE = REPOSITORY / "evidence" / "organizer"


def test_staged_export_preserves_partition_and_anchor_values(tmp_path):
    api = runpy.run_path(str(EXAMPLE / "export.py"))
    output = tmp_path / "inputs"
    study = api["export_inputs"](output, stage="design")
    assert not (output / "assessment_scores.tsv").exists()
    assert len(study["development_cases"]) == 27
    assert len(study["assessment_cases"]) == 28
    assert not set(study["development_cases"]) & set(study["assessment_cases"])
    api["export_inputs"](output, stage="assessment")
    scores = api["read_scores"](output / "assessment_scores.tsv", study, study["assessment_cases"])
    anchors = api["read_scores"](output / "assessment_anchors.tsv", study,
                                 study["assessment_cases"], anchor_only=True)
    assert len(scores) == 28 * 6 * 90 * 6
    assert len(anchors) == 28 * 6 * 6
    assert all(scores[key] == value for key, value in anchors.items())
    # Reusing the frozen input directory must not replace its tables.
    before = (output / "assessment_scores.tsv").read_bytes()
    with pytest.raises(ValueError, match="Output already exists"):
        api["export_inputs"](output, stage="all")
    assert (output / "assessment_scores.tsv").read_bytes() == before


def test_relocated_replay_reproduces_all_gaps_and_budget_holds(tmp_path):
    relocated = tmp_path / "public example"
    shutil.copytree(EXAMPLE, relocated)
    output = tmp_path / "organizer results"
    environment = dict(os.environ, PYTHONPATH=str(REPOSITORY / "src"))
    process = subprocess.run([sys.executable, str(relocated / "replay.py"),
                              "--output", str(output)], cwd=tmp_path,
                             env=environment, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "pass"
    assert receipt["configurations"] == 4
    assert receipt["all_designs_frozen_before_assessment_export"]
    assert not receipt["expected_results_used_for_planning"]
    primary = {row["policy"]: row for row in receipt["primary_pair_results"]}
    assert (primary["single_split"]["released"], primary["single_split"]["unsupported_releases"]) == (323, 65)
    assert primary["single_split"]["strict_reversals"] == 24
    for policy in ("repeated_primary", "geometry_scaled"):
        assert (primary[policy]["released"], primary[policy]["unsupported_releases"]) == (165, 0)
    budgets = {row["policy"]: row for row in receipt["primary_budget_results"]}
    assert (budgets["repeated_primary"]["released"], budgets["repeated_primary"]["held"],
            budgets["repeated_primary"]["supported"], budgets["repeated_primary"]["unsupported"]) == (2, 26, 2, 0)
    assert (budgets["single_split"]["released"], budgets["single_split"]["unsupported"]) == (24, 9)
    assert receipt["strict_geometry_gains"] == 0
    frozen = json.loads((output / "designs_frozen.json").read_text())
    assert {item["minimum_gap"] for item in frozen["designs"]} == {0.0, 0.005, 0.01, 0.02}
    for item in frozen["designs"]:
        path = output / item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        artifact = json.loads(path.read_text())
        assert set(artifact["inputs"]) == {"manifest", "development_scores", "assessment_anchors", "geometry"}
    with (output / "results" / "geometry_comparison.tsv").open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 4 * 6
    assert {row["outcome"] for row in rows} == {"parity"}
