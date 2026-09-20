from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
MANIFEST = REPOSITORY / "provenance/SOURCE_DATA_ROOT_MANIFEST_V1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_script(name: str, relative: str):
    specification = importlib.util.spec_from_file_location(name, REPOSITORY / relative)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_code_archive_has_manifest_but_no_bound_source_data_payloads() -> None:
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert value["release_version"] == "1.10.0"
    assert value["payload_count"] == len(value["payloads"]) >= 74
    assert len({item["external_path"] for item in value["payloads"]}) == value["payload_count"]
    assert len({item["runtime_path"] for item in value["payloads"]}) == value["payload_count"]
    embedded = [
        item["runtime_path"]
        for item in value["payloads"]
        if (REPOSITORY / item["runtime_path"]).exists()
        or (REPOSITORY / item["runtime_path"]).is_symlink()
    ]
    assert embedded == []


def test_current_source_binding_changes_only_the_two_corrected_jerber_payloads() -> None:
    current = json.loads(MANIFEST.read_text(encoding="utf-8"))
    historical = json.loads(
        (REPOSITORY / "provenance/V183_IMMUTABLE_INPUTS.json").read_text(encoding="utf-8")
    )["scientific_payloads"]
    current_hashes = {row["external_path"]: row["sha256"] for row in current["payloads"]}
    assert set(current_hashes) == set(historical)
    assert {name for name in historical if current_hashes[name] != historical[name]} == {
        "data/derived/extended_data_figure_4_jerber/JERBER_HISTORICAL_POINT_RANKINGS_V1.tsv",
        "data/derived/extended_data_figure_4_jerber/JERBER_POINT_RANKING_AUTHORITY_V1.json",
    }


def test_archived_multi_realization_core_and_protocol_are_exact() -> None:
    assert sha256(
        REPOSITORY / "analysis/multi_realization/v21_multi_realization_analysis_core.py"
    ) == "0e57553464e2571570ce04cc554a571f79564a067bec37a55ad3c57b9d2c295f"
    assert sha256(
        REPOSITORY
        / "analysis/multi_realization/GSE162632_MULTI_REALIZATION_PUBLIC_ANALYSIS_PROTOCOL_V1.json"
    ) == "e73b015f18243e8d1f5a1abcfe3f3aadef97a89ccfa34b11dfde92d66ee47354"


def test_all_replay_stage_roster_and_network_separation() -> None:
    replay = load_script("_test_replay_all", "scripts/replay_all.py")
    assert [stage[0] for stage in replay.ANALYSIS_STAGES] == [
        "held_out_84",
        "paired_construction_56",
        "paired_construction_structural_zeros_v189",
        "multi_realization",
        "cmodel_successor",
        "v22_attribution",
    ]
    source = (REPOSITORY / "scripts/replay_all.py").read_text(encoding="utf-8")
    assert "fetch_source_data" not in source


def test_fetch_fails_closed_without_explicit_archive_identity() -> None:
    fetch = load_script("_test_fetch_source_data", "scripts/fetch_source_data.py")
    with pytest.raises(SystemExit):
        fetch.parser().parse_args(["--doi", "10.0000/not-a-frozen-record"])
    with pytest.raises(fetch.FetchSourceDataError, match="HTTPS"):
        fetch.download_archive(
            "http://example.invalid/source-data.zip",
            Path("unused.archive"),
            byte_limit=1,
        )
    with pytest.raises(fetch.FetchSourceDataError, match="credential-free HTTPS"):
        fetch.HTTPSOnlyRedirectHandler().redirect_request(
            None,
            None,
            302,
            "Found",
            {},
            "http://example.invalid/intermediate",
        )


@pytest.mark.parametrize("compatibility", [{}, {"analysis_only": False, "editorial_python": Path("missing-editorial-python")}])
def test_aggregate_replay_completes_numerical_stages_without_artwork_runtime(
    tmp_path, monkeypatch, compatibility,
) -> None:
    """Exercise orchestration with no renderer tree or graphical interpreter."""
    from types import SimpleNamespace

    replay = load_script("_test_numeric_replay", "scripts/replay_all.py")
    checked = []
    snapshot = SimpleNamespace(root=tmp_path, manifest_sha256="frozen-source", payloads=(1,),
                               assert_unchanged=lambda: checked.append("source"))
    monkeypatch.setattr(replay, "validate_source_data_root", lambda path: snapshot)
    monkeypatch.setattr(replay, "copy_repository", lambda path: path.mkdir())
    monkeypatch.setattr(replay, "repository_tree_sha256", lambda path: "unchanged-code")
    commands = []

    def stage(**kw):
        commands.append(kw["command"])
        if kw["name"] == "generic_framework_self_test":
            output = Path(kw["command"][kw["command"].index("--output") + 1])
            output.write_text(json.dumps({"status": "PASS_GENERIC_FRAMEWORK_V1_IMPLEMENTATION_SELF_TEST"}))
        return {"stage": kw["name"], "status": "PASS"}

    monkeypatch.setattr(replay, "run_stage", stage)
    result = replay.replay_all(source_data_root=tmp_path, legacy_python=Path(sys.executable), **compatibility)
    assert result["mode"] == "ANALYSIS_ONLY"
    assert result["code_release_version"] == "1.10.0"
    assert result["source_data_release_version"] == "1.10.0"
    assert [row["stage"] for row in result["stages"]] == [
        *[row[0] for row in replay.ANALYSIS_STAGES],
        "focal_five_realization_capsule", "matched_allocation_1000_labels",
        "generic_framework_self_test", "design_audit_public_empirical_example",
    ]
    assert checked == ["source"]
    assert all("figure" not in str(command[2]) for command in commands)
