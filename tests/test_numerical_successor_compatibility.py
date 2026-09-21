from __future__ import annotations

import ast
import fnmatch
import hashlib
import importlib.util
from pathlib import Path
import re
import sys

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
OLD_BASELINE_SHA256 = (
    "86069b860e3626a64cecf689e86580a57a74d7afd7793e56794fcbdb6332ea89"
)
OLD_V22_SHA256 = (
    "987575dac8e7dc7e0d528fb3ee361b6d87d95df9ed294865ac63bd45e5ac7700"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(name: str, relative_path: str):
    path = REPOSITORY / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


old_direct = _load(
    "gse306429_v2_baselines_for_test",
    "src/perturb_nuisance_model/gse306429_v2_baselines.py",
)
new_direct = _load(
    "gse306429_v3_baselines_for_test",
    "src/perturb_nuisance_model/gse306429_v3_baselines.py",
)
old_v22 = _load("v22_core_for_test", "analysis/v22_attribution/v22_core.py")
new_v22 = _load(
    "v22_numerics_v2_compatibility_test",
    "analysis/v22_attribution/v22_numerics_v2.py",
)


def _training_data(fixed_dose: bool) -> dict[str, object]:
    rows, genes = 15, 8
    return {
        "task_ids": [f"row_{index:03d}" for index in range(rows)],
        "contexts": [f"context_{index % 3}" for index in range(rows)],
        "compounds": [f"compound_{index % 4}" for index in range(rows)],
        "doses_um": [1.0] * rows
        if fixed_dose
        else [1.0 if index % 2 else 2.0 for index in range(rows)],
        "effects": np.random.default_rng(166).normal(size=(rows, genes)),
        "feature_ids": [f"gene_{index}" for index in range(genes)],
    }


def _queries(data: dict[str, object]) -> dict[str, object]:
    return {
        key: data[key]
        for key in ("contexts", "compounds", "doses_um", "feature_ids")
    }


def test_direct_baselines_match_on_normal_inputs() -> None:
    fitters = (
        "fit_context_mean_effect_direct",
        "fit_two_way_additive_ridge_direct",
        "fit_pca64_additive_ridge_direct",
        "fit_rbf_kernel_ridge_direct",
    )
    for fixed_dose in (False, True):
        data = _training_data(fixed_dose)
        for name in fitters:
            old_state = getattr(old_direct, name)(**data)
            new_state = getattr(new_direct, name)(**data)
            old_prediction = old_state.predict(**_queries(data))
            new_prediction = new_state.predict(**_queries(data))
            assert old_prediction.dtype == new_prediction.dtype
            assert old_prediction.shape == new_prediction.shape
            assert old_prediction.tobytes() == new_prediction.tobytes()
            assert old_direct.canonical_state_sha256(old_state) == (
                new_direct.canonical_state_sha256(new_state)
            )


def test_no_change_state_matches() -> None:
    data = _training_data(False)
    arguments = {
        "feature_ids": data["feature_ids"],
        "context_ids": sorted(set(data["contexts"])),
        "compound_ids": sorted(set(data["compounds"])),
    }
    old_state = old_direct.make_no_change_direct_state(**arguments)
    new_state = new_direct.make_no_change_direct_state(**arguments)
    assert old_direct.canonical_state_sha256(old_state) == (
        new_direct.canonical_state_sha256(new_state)
    )
    np.testing.assert_array_equal(
        old_state.predict(**_queries(data)), new_state.predict(**_queries(data))
    )


def test_v22_numerics_match_on_normal_inputs() -> None:
    for index in range(16):
        rng = np.random.default_rng(834 + index)
        predicted, observed = rng.normal(size=(2, 13))
        scales = rng.uniform(0.0, 2.0, size=13)
        assert old_v22.standardized_negative_mse(
            predicted, observed, scales
        ) == new_v22.standardized_negative_mse(predicted, observed, scales)

    rng = np.random.default_rng(20260912)
    controls, effects = rng.normal(size=(2, 11, 7))
    states = controls + effects
    old_fit = old_v22.fit_matched_rank4_ridge(controls, effects, states)
    new_fit = new_v22.fit_matched_rank4_ridge(controls, effects, states)
    for target in ("effect", "state"):
        old_state = getattr(old_fit, target)
        new_state = getattr(new_fit, target)
        for field in ("control_center", "components", "coefficients"):
            assert getattr(old_state, field).tobytes() == getattr(
                new_state, field
            ).tobytes()
        assert old_state.predict(controls).tobytes() == new_state.predict(
            controls
        ).tobytes()

    for emitted in ("STATE", "EFFECT"):
        for requested in ("STATE", "EFFECT"):
            old = old_v22.adapt_output(
                effects, emitted, requested, controls
            )
            new = new_v22.adapt_output(
                effects, emitted, requested, controls
            )
            assert old.tobytes() == new.tobytes()


def test_historical_sources_and_manifest_remain_explicit() -> None:
    old_baseline = REPOSITORY / (
        "src/perturb_nuisance_model/gse306429_v2_baselines.py"
    )
    old_numerics = REPOSITORY / "analysis/v22_attribution/v22_core.py"
    assert _sha256(old_baseline) == OLD_BASELINE_SHA256
    assert _sha256(old_numerics) == OLD_V22_SHA256

    records = {}
    for line in (REPOSITORY / "MANIFEST.sha256").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        records[relative] = digest
    for relative in (
        "src/perturb_nuisance_model/gse306429_v2_baselines.py",
        "src/perturb_nuisance_model/gse306429_v3_baselines.py",
        "analysis/v22_attribution/v22_core.py",
        "analysis/v22_attribution/v22_numerics_v2.py",
        "tests/test_numerical_boundaries.py",
        "tests/test_numerical_successor_compatibility.py",
    ):
        assert records[relative] == _sha256(REPOSITORY / relative)


def test_wheel_scope_keeps_archived_models_outside_supported_api() -> None:
    pyproject = (REPOSITORY / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(
        r"(?ms)^\[tool\.setuptools\.packages\.find\]\n(?P<body>.*?)(?=^\[|\Z)",
        pyproject,
    )
    assert match is not None
    body = match.group("body")

    def array_value(name: str) -> list[str]:
        value = re.search(rf"(?m)^{name}\s*=\s*(\[[^\n]+\])$", body)
        assert value is not None
        parsed = ast.literal_eval(value.group(1))
        assert isinstance(parsed, list) and all(
            isinstance(item, str) for item in parsed
        )
        return parsed

    discovery = {
        "where": array_value("where"),
        "include": array_value("include"),
        "exclude": array_value("exclude"),
    }
    assert discovery["include"] == [
        "perturb_nuisance_contracts*",
        "perturb_nuisance_focal*",
        "reference_design*",
    ]
    assert discovery["exclude"] == ["perturb_nuisance_model*"]

    source_root = REPOSITORY / discovery["where"][0]
    packages = {
        ".".join(path.parent.relative_to(source_root).parts)
        for path in source_root.rglob("__init__.py")
    }
    selected = {
        package
        for package in packages
        if any(
            fnmatch.fnmatchcase(package, pattern)
            for pattern in discovery["include"]
        )
        and not any(
            fnmatch.fnmatchcase(package, pattern)
            for pattern in discovery["exclude"]
        )
    }
    assert selected == {
        "perturb_nuisance_contracts",
        "perturb_nuisance_focal",
        "reference_design",
        "reference_design.reporting_validation",
    }
