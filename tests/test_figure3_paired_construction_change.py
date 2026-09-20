from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from types import ModuleType



REPO = Path(__file__).resolve().parents[1]
RENDERER = REPO / "analysis/paired_construction/source_tables.py"


def load_renderer() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "figure3_v20_paired_construction_renderer", RENDERER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complete_paired_change_family_and_focal_display_contract(
    hydrated_data_repository: Path,
) -> None:
    renderer = load_renderer()
    paired = renderer.read_paired_changes(hydrated_data_repository)

    assert len(paired["summary_rows"]) == 56
    assert len(paired["root_rows"]) == 448
    assert [row["paired_change_index"] for row in paired["summary_rows"]] == list(
        range(56)
    )
    assert {
        row["max_t_family_size"] for row in paired["summary_rows"]
    } == {56}
    assert all(
        "post" in row["evidence_role"].lower()
        for row in paired["summary_rows"]
    )

    assert set(paired["selected_summaries"]) == set(renderer.PAIRED_TARGET_ORDER)
    for target in renderer.PAIRED_TARGET_ORDER:
        summary = paired["selected_summaries"][target]
        roots = paired["selected_roots"][target]
        assert summary["method_a"] == renderer.PCA64_METHOD_ID
        assert summary["method_b"] == renderer.SCGEN_METHOD_ID
        assert summary["point_paired_change"] > 0.0
        assert summary["max_t_lower"] > 0.0
        assert len(roots) == 8
        assert all(row["root_paired_change"] > 0.0 for row in roots)


