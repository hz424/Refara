from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import json
import subprocess

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "analysis/cmodel_successor/cmodel_successor_public_replay.py"


def load_replay_module():
    specification = importlib.util.spec_from_file_location(
        "_test_cmodel_successor_public_replay",
        SCRIPT,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_shared_sign_family_is_exact_and_deterministic() -> None:
    replay = load_replay_module()
    roots = np.arange(1.0, 9.0)[:, None]
    endpoints = np.arange(1.0, 6.0)[None, :]
    values = roots * endpoints + np.square(roots) / (endpoints + 1.0)
    first = replay.exact_shared_sign_max_abs_t(values)
    second = replay.exact_shared_sign_max_abs_t(values.copy())
    assert first.sign_vector_count == 256
    assert first.critical_order_one_indexed == 244
    assert np.array_equal(first.estimates, second.estimates)
    assert first.critical_value == second.critical_value
    assert np.all(first.lower < first.estimates)
    assert np.all(first.estimates < first.upper)


def test_documented_cli_accepts_external_source_root(source_data_root: Path) -> None:
    """Exercise external-to-runtime mapping, which a hydrated-table test misses."""
    result = subprocess.run(
        [sys.executable, "-B", str(SCRIPT), "--source-data-root", str(source_data_root)],
        check=True, capture_output=True, text=True,
    )
    report = json.loads(result.stdout)
    assert report["status"] == "PASS_CMODEL_SUCCESSOR_PUBLIC_REPLAY"
    assert report["root_count"] == 8
    assert report["B_condition_winner_unresolved"] is True


def test_cmodel_successor_public_projection_replays(hydrated_data_repository: Path) -> None:
    replay = load_replay_module()
    report = replay.replay(
        hydrated_data_repository / "data/derived/cmodel_successor"
    )
    assert report["status"] == "PASS_CMODEL_SUCCESSOR_PUBLIC_REPLAY"
    assert report["root_count"] == 8
    assert report["root_shift_signs"] == {
        "A_MINUS_C_positive": 8,
        "B_MINUS_C_negative": 8,
    }
    assert report["strict_root_reversal_count"] == {"C_TO_A": 0, "C_TO_B": 5}
    assert report["descriptive_aggregate_reversal_C_to_B"] is True
    assert report["B_condition_winner_unresolved"] is True
    assert report["supported_new_winner_under_B"] is False
    sensitivity = report["five_endpoint_sensitivity"]
    assert sensitivity["sign_vector_count"] == 256
    assert sensitivity["critical_order_one_indexed"] == 244
    assert math.isclose(
        sensitivity["critical_value"],
        3.1685802460421075,
        rel_tol=0.0,
        abs_tol=5e-15,
    )
    assert math.isclose(
        sensitivity["simultaneous_lower"]["MARGIN_B"],
        -0.004500116151113126,
        rel_tol=0.0,
        abs_tol=5e-15,
    )
    assert math.isclose(
        sensitivity["simultaneous_upper"]["MARGIN_B"],
        0.0024782405570089477,
        rel_tol=0.0,
        abs_tol=5e-15,
    )
    assert math.isclose(
        report["margin_equal_root_mean"]["B"],
        -0.0010109377970520888,
        rel_tol=0.0,
        abs_tol=5e-15,
    )
    assert report["midpoint_identity"]["maximum_root_absolute_error"] <= 5e-14
    assert report["utility_response_equal_root_mean_B_minus_C"] == {
        "ridge": 0.001856017005995636,
        "cellflow": 0.021140277633706586,
        "ridge_minus_cellflow": -0.01928426062771095,
    }
    rms = report["standardized_rms_output_displacement_equal_root_mean_B_minus_C"]
    assert math.isclose(rms["ridge"], 0.06392876715310016, rel_tol=0.0, abs_tol=5e-15)
    assert math.isclose(rms["cellflow"], 0.2079940172085688, rel_tol=0.0, abs_tol=5e-15)
