from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType


REPO = Path(__file__).resolve().parents[1]
BUILDER = REPO / "analysis/low_replication/source_workbook_legacy.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("figure2_source_workbook", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_data_build_is_read_only_and_byte_deterministic(
    hydrated_data_repository: Path,
) -> None:
    builder = load_builder()
    package = hydrated_data_repository
    input_dir = package / "data/derived/figure2"
    inputs_before = tree_hashes(input_dir)

    first = builder.build(package)
    first_hashes = {
        "crosswalk": sha256(package / first["crosswalk"]["path"]),
        "workbook": sha256(package / first["workbook"]["path"]),
    }
    assert tree_hashes(input_dir) == inputs_before

    second = builder.build(package)
    second_hashes = {
        "crosswalk": sha256(package / second["crosswalk"]["path"]),
        "workbook": sha256(package / second["workbook"]["path"]),
    }
    assert tree_hashes(input_dir) == inputs_before
    assert second_hashes == first_hashes
