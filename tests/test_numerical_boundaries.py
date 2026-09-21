from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


REPOSITORY = Path(__file__).resolve().parents[1]


def _load(name: str, relative_path: str):
    path = REPOSITORY / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


direct = _load(
    "gse306429_v3_baselines_for_test",
    "src/perturb_nuisance_model/gse306429_v3_baselines.py",
)
numerics = _load(
    "v22_numerics_v2_for_test",
    "analysis/v22_attribution/v22_numerics_v2.py",
)


def _training_data(rows: int, genes: int) -> dict[str, object]:
    return {
        "task_ids": [f"row_{index:03d}" for index in range(rows)],
        "contexts": [f"context_{index % 3}" for index in range(rows)],
        "compounds": [f"compound_{index % 4}" for index in range(rows)],
        "doses_um": [1.0] * rows,
        "effects": np.random.default_rng(143 + rows + genes).normal(
            size=(rows, genes)
        ),
        "feature_ids": [f"gene_{index}" for index in range(genes)],
    }


def _queries(data: dict[str, object]) -> dict[str, object]:
    return {
        key: data[key]
        for key in ("contexts", "compounds", "doses_um", "feature_ids")
    }


@pytest.mark.parametrize("genes", [1, 3])
def test_single_row_pca_returns_the_training_effect(genes: int) -> None:
    data = _training_data(1, genes)
    model = direct.fit_pca64_additive_ridge_direct(**data)
    assert model.rank == 0
    np.testing.assert_array_equal(model.predict(**_queries(data)), data["effects"])


def test_constant_nonunit_dose_uses_the_zero_variance_rule() -> None:
    data = _training_data(6, 3)
    data.update(
        contexts=["one"] * 6,
        compounds=["compound"] * 6,
        doses_um=[0.3] * 6,
    )
    model = direct.fit_rbf_kernel_ridge_direct(**data)
    assert model.log10_dose_sample_sd == 0.0

    query = {
        "contexts": ["one"],
        "compounds": ["compound"],
        "doses_um": [0.6],
        "feature_ids": data["feature_ids"],
    }
    expected = np.asarray(data["effects"]).sum(axis=0) / 7.0
    np.testing.assert_allclose(model.predict(**query)[0], expected, atol=1e-13)


def test_unequal_near_constant_doses_remain_distinct() -> None:
    data = _training_data(6, 3)
    data["doses_um"] = [0.3] * 5 + [0.3 + 1e-12]
    model = direct.fit_rbf_kernel_ridge_direct(**data)
    expected = np.log10(np.asarray(data["doses_um"])).std(ddof=1)
    assert expected > 0.0
    assert model.log10_dose_sample_sd == expected


def test_empty_score_is_rejected() -> None:
    with pytest.raises(numerics.V22CoreError, match="one-dimensional"):
        numerics.standardized_negative_mse([], [], [])


def test_score_overflow_is_rejected() -> None:
    with pytest.raises(numerics.V22CoreError, match="floating-point range"):
        numerics.standardized_negative_mse([1e308], [-1e308], [1.0])


def test_rank_four_ridge_rejects_an_empty_gene_axis() -> None:
    empty = np.empty((5, 0))
    with pytest.raises(numerics.V22CoreError, match="nonempty gene axis"):
        numerics.fit_matched_rank4_ridge(empty, empty, empty)
