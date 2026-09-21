from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPOSITORY
    / "analysis/multi_realization/v21_multi_realization_public_replay.py"
)


def load_replay():
    specification = importlib.util.spec_from_file_location(
        "_test_v21_multi_realization_public_replay",
        SCRIPT,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_typed_tsv_comparison_allows_only_small_float_differences(
    tmp_path: Path,
) -> None:
    replay = load_replay()
    expected = [{"identity": "root-a", "index": 1, "value": 0.25, "gate": True}]
    path = tmp_path / "values.tsv"
    path.write_text(
        "identity\tindex\tvalue\tgate\nroot-a\t1\t0.2500000000005\ttrue\n",
        encoding="utf-8",
    )
    replay.compare_tsv(path, expected)
    path.write_text(
        "identity\tindex\tvalue\tgate\nroot-b\t1\t0.2500000000005\ttrue\n",
        encoding="utf-8",
    )
    with pytest.raises(replay.MultiRealizationReplayError):
        replay.compare_tsv(path, expected)
    path.write_text(
        "identity\tindex\tvalue\tgate\nroot-a\t1\t0.250000000002\ttrue\n",
        encoding="utf-8",
    )
    with pytest.raises(replay.MultiRealizationReplayError):
        replay.compare_tsv(path, expected)


def test_typed_json_comparison_preserves_schema_and_integer_identity() -> None:
    replay = load_replay()
    expected = {"family": [1, {"estimate": 0.25, "status": "PASS"}]}
    replay.compare_json_value(
        {"family": [1, {"estimate": 0.2500000000005, "status": "PASS"}]},
        expected,
        location="root",
    )
    with pytest.raises(replay.MultiRealizationReplayError):
        replay.compare_json_value(
            {"family": [1.0, {"estimate": 0.25, "status": "PASS"}]},
            expected,
            location="root",
        )


def test_complete_multi_realization_replay(source_data_root: Path) -> None:
    report = load_replay().replay(source_data_root)
    assert report["status"] == "PASS_V21_MULTI_REALIZATION_PUBLIC_REPLAY"
    assert report["input_root_task_rows"] == 2400
    assert report["verified_derived_artifact_count"] == 12
    assert report["semantic_numeric_absolute_tolerance"] == 1e-12
    assert report["pinned_byte_identity_count"] == 12
    assert report["within_family_size"] == 84
    assert report["paired_family_size"] == 56
    assert report["outcome_branch"] == "STRICT_PASS"
