"""The comparison replay remains usable after moving its evidence directory."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parents[1]


def test_relocated_comparison_checks_all_resources_and_preserves_plans(tmp_path):
    example = tmp_path / "relocated evidence"
    shutil.copytree(REPOSITORY / "evidence/organizer", example,
                    ignore=shutil.ignore_patterns("__pycache__"))
    output = tmp_path / "results"
    result = subprocess.run(
        [sys.executable, str(example / "evaluate.py"), "--output", str(output)],
        cwd=tmp_path, env=dict(os.environ, PYTHONPATH=str(REPOSITORY / "src")),
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "pass"
    assert receipt["expected_results_checked"]
    assert receipt["all_plans_frozen_before_assessment_export"]
    assert receipt["results"]["comparison_rules.tsv"]["rows"] == 96
    assert receipt["results"]["comparison_aurc.tsv"]["rows"] == 64
    for group in ("original8", "cap48"):
        assert receipt["datasets"][group]["development_units"] == 8
        assert receipt["datasets"][group]["assessment_units"] == 39
        assert receipt["datasets"][group]["assessment_cases"] == 78
    frozen = json.loads((output / "plans_frozen.json").read_text())
    assert not frozen["assessment_tables_exported"]
    for item in frozen["plans"]:
        plan = output / item["plan"]
        assert hashlib.sha256(plan.read_bytes()).hexdigest() == item["sha256"]
        assessment = output / "datasets" / item["resource"] / "check/check.json"
        assert json.loads(assessment.read_text())["plan_sha256"] == item["sha256"]
