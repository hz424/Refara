"""The fixed-loss capsule is portable and rejects stale numerical outputs."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


SOURCE = Path(__file__).resolve().parents[1] / "evidence/jerber_reference_audit"


def run_copy(tmp_path):
    capsule = tmp_path / "standalone"
    shutil.copytree(SOURCE, capsule)
    return capsule


def execute(capsule, tmp_path):
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-B", str(capsule / "replay.py"), "--output", str(tmp_path / "output")],
        cwd=tmp_path, env=env, text=True, capture_output=True, check=False,
    )


def test_portable_complete_replay(tmp_path):
    result = execute(run_copy(tmp_path), tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "output/RESULTS.json").read_text())
    assert report["root_task_pair_allocation"]["events"] == 7680
    assert report["task_pair_allocation"]["events"] == 384
    assert report["task_pairs_reversed_in_all_16"] == 9
    assert report["raw_model_inference_replayed"] is False
    assert report["independent_experiment_validation"] is False


def test_stale_direction_table_is_rejected(tmp_path):
    capsule = run_copy(tmp_path)
    target = capsule / "task_pair_directions.tsv"
    with target.open() as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields, rows = reader.fieldnames, list(reader)
    rows[0]["O_margin"] = str(float(rows[0]["O_margin"]) + 0.01)
    with target.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    result = execute(capsule, tmp_path)
    assert result.returncode != 0
    assert "stale numeric output" in result.stderr
    assert not (tmp_path / "output").exists()


def test_missing_row_cannot_reduce_the_denominator(tmp_path):
    capsule = run_copy(tmp_path)
    target = capsule / "root_task_pair_directions.tsv"
    lines = target.read_text().splitlines(keepends=True)
    target.write_text("".join(lines[:-1]))
    result = execute(capsule, tmp_path)
    assert result.returncode != 0
    assert "output row count changed" in result.stderr
    assert not (tmp_path / "output").exists()


def test_misdeclared_root_count_is_rejected(tmp_path):
    capsule = run_copy(tmp_path)
    target = capsule / "manifest.json"
    value = json.loads(target.read_text())
    value["root_count"] = 19
    target.write_text(json.dumps(value))
    result = execute(capsule, tmp_path)
    assert result.returncode != 0
    assert "declared root geometry differs" in result.stderr
    assert not (tmp_path / "output").exists()
