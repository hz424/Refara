from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from perturb_nuisance_contracts.source_data_root import (
    SourceDataRootError,
    hydrate_source_data,
    validate_source_data_root,
)
import perturb_nuisance_contracts.source_data_root as source_data_module


REPOSITORY = Path(__file__).resolve().parents[1]
BUNDLED_MANIFEST = REPOSITORY / "provenance/SOURCE_DATA_ROOT_MANIFEST_V1.json"


def canonical(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def synthetic_root(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    value = json.loads(BUNDLED_MANIFEST.read_text(encoding="utf-8"))
    root = tmp_path / "05_SOURCE_DATA"
    root.mkdir()
    for index, record in enumerate(value["payloads"]):
        payload = f"synthetic payload {index:02d}\n".encode("ascii")
        record["bytes"] = len(payload)
        record["sha256"] = hashlib.sha256(payload).hexdigest()
        destination = root / record["external_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    expected = tmp_path / "expected.json"
    raw = canonical(value)
    expected.write_bytes(raw)
    (root / "SOURCE_DATA_ROOT_MANIFEST_V1.json").write_bytes(raw)
    return root, expected, value


def test_validate_and_hydrate_exact_synthetic_inventory(tmp_path: Path) -> None:
    root, expected, value = synthetic_root(tmp_path)
    snapshot = validate_source_data_root(root, expected_manifest=expected)
    assert len(snapshot.payloads) == len(value["payloads"])
    destination = tmp_path / "runtime-repository"
    destination.mkdir()
    copied = hydrate_source_data(snapshot, destination)
    assert len(copied) == len(value["payloads"])
    for record in value["payloads"]:
        source = root / record["external_path"]
        hydrated = destination / record["runtime_path"]
        assert hydrated.read_bytes() == source.read_bytes()
    snapshot.assert_unchanged()


@pytest.mark.parametrize("failure", ["missing", "tamper", "extra", "symlink"])
def test_validator_fails_closed_on_payload_tree_changes(
    tmp_path: Path,
    failure: str,
) -> None:
    root, expected, value = synthetic_root(tmp_path)
    first = root / value["payloads"][0]["external_path"]
    if failure == "missing":
        first.unlink()
    elif failure == "tamper":
        first.write_bytes(b"changed\n")
    elif failure == "extra":
        extra = root / "data/derived/undeclared.txt"
        extra.write_text("undeclared\n", encoding="utf-8")
    else:
        first.unlink()
        first.symlink_to(expected)
    with pytest.raises(SourceDataRootError):
        validate_source_data_root(root, expected_manifest=expected)


def test_external_manifest_must_equal_bundled_bytes(tmp_path: Path) -> None:
    root, expected, value = synthetic_root(tmp_path)
    value["status"] = "CHANGED"
    (root / "SOURCE_DATA_ROOT_MANIFEST_V1.json").write_bytes(canonical(value))
    with pytest.raises(SourceDataRootError, match="manifests differ"):
        validate_source_data_root(root, expected_manifest=expected)


def test_default_validator_rejects_the_exact_historical_v183_manifest(tmp_path: Path) -> None:
    value = json.loads(BUNDLED_MANIFEST.read_text(encoding="utf-8"))
    historical = json.loads(
        (REPOSITORY / "provenance/V183_IMMUTABLE_INPUTS.json").read_text(encoding="utf-8")
    )["scientific_payloads"]
    old_sizes = {
        "JERBER_HISTORICAL_POINT_RANKINGS_V1.tsv": 2383,
        "JERBER_POINT_RANKING_AUTHORITY_V1.json": 3689,
    }
    value["release_version"] = "1.8.3"
    for record in value["payloads"]:
        record["sha256"] = historical[record["external_path"]]
        if Path(record["external_path"]).name in old_sizes:
            record["bytes"] = old_sizes[Path(record["external_path"]).name]
    raw = canonical(value)
    assert hashlib.sha256(raw).hexdigest() == (
        "fbe01eca4b975a9131a6d79bd72e50813c49bdb5adb827455660bb290aeb6eb2"
    )
    root = tmp_path / "historical-source-data"
    root.mkdir()
    (root / "SOURCE_DATA_ROOT_MANIFEST_V1.json").write_bytes(raw)
    with pytest.raises(SourceDataRootError, match="manifests differ"):
        validate_source_data_root(root)


def test_hydration_rejects_linked_parent_before_external_write(tmp_path: Path) -> None:
    root, expected, _value = synthetic_root(tmp_path)
    snapshot = validate_source_data_root(root, expected_manifest=expected)
    destination = tmp_path / "runtime"
    outside = tmp_path / "outside"
    destination.mkdir()
    outside.mkdir()
    (destination / "data").symlink_to(outside, target_is_directory=True)
    with pytest.raises(SourceDataRootError, match="parent is a symlink"):
        hydrate_source_data(snapshot, destination)
    assert list(outside.iterdir()) == []


def test_namespace_traversal_errors_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, expected, _value = synthetic_root(tmp_path)

    def failed_walk(*args, **kwargs):
        kwargs["onerror"](PermissionError("synthetic scandir denial"))
        return iter(())

    monkeypatch.setattr(source_data_module.os, "walk", failed_walk)
    with pytest.raises(SourceDataRootError, match="cannot traverse protected namespace"):
        validate_source_data_root(root, expected_manifest=expected)


def test_snapshot_detects_matching_manifest_semantic_change(tmp_path: Path) -> None:
    root, expected, value = synthetic_root(tmp_path)
    snapshot = validate_source_data_root(root, expected_manifest=expected)
    record = next(
        item
        for item in value["payloads"]
        if item["role"] == "V22_ATTRIBUTION_EVIDENCE"
    )
    record["role"] = "FIGURE_SOURCE_DATA"
    changed = canonical(value)
    expected.write_bytes(changed)
    (root / "SOURCE_DATA_ROOT_MANIFEST_V1.json").write_bytes(changed)
    with pytest.raises(SourceDataRootError, match="manifest changed"):
        snapshot.assert_unchanged()


def test_supplied_official_source_data_root_is_exact(source_data_snapshot) -> None:
    assert len(source_data_snapshot.payloads) == json.loads(BUNDLED_MANIFEST.read_text())["payload_count"]
    assert source_data_snapshot.manifest_sha256 == (
        hashlib.sha256(BUNDLED_MANIFEST.read_bytes()).hexdigest()
    )
    assert sum(
        record.role == "MULTI_REALIZATION_SENSITIVITY"
        for record in source_data_snapshot.payloads
    ) == 13
    assert sum(
        record.role == "CMODEL_SUCCESSOR_EVIDENCE"
        for record in source_data_snapshot.payloads
    ) == 4
    source_data_snapshot.assert_unchanged()
