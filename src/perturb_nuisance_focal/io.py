"""Input checks shared by the focal replay and retraining commands."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import string
from typing import Any, Mapping, Sequence

import numpy as np


class CapsuleError(RuntimeError):
    """The capsule is incomplete or differs from its manifest."""


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise CapsuleError(f"Refusing to replace an existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def read_json(path: Path, label: str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise CapsuleError(f"{label} is missing or is not a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CapsuleError(f"Could not read {label}") from error
    if not isinstance(value, dict):
        raise CapsuleError(f"{label} must contain a JSON object")
    return value


def _relative_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CapsuleError(f"{label} has no path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise CapsuleError(f"{label} must use a normalized relative path")
    return path


def load_manifest(root: Path, expected_kind: str) -> dict[str, Any]:
    """Validate every payload before returning its manifest."""

    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise CapsuleError("The asset directory is missing or is not a regular directory")
    manifest = read_json(root / "manifest.json", "asset manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("asset_kind") != expected_kind
        or manifest.get("release_version") != "1.6.1"
    ):
        raise CapsuleError("The asset manifest is for a different capsule release")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise CapsuleError("The asset manifest has no file inventory")

    listed: set[str] = set()
    logical_names: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise CapsuleError("The asset manifest contains a malformed file record")
        name = record.get("name")
        expected_bytes = record.get("bytes")
        expected_sha256 = record.get("sha256")
        if not isinstance(name, str) or not name or name in logical_names:
            raise CapsuleError("Each asset record must have a unique non-empty name")
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 0
        ):
            raise CapsuleError(f"The byte count for {name!r} is invalid")
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in string.hexdigits for character in expected_sha256)
            or expected_sha256 != expected_sha256.lower()
        ):
            raise CapsuleError(f"The SHA-256 value for {name!r} is invalid")
        logical_names.add(name)
        relative = _relative_path(record.get("path"), "asset record")
        text = relative.as_posix()
        if text == "manifest.json":
            raise CapsuleError("manifest.json cannot list itself as a payload")
        if text in listed:
            raise CapsuleError(f"The asset manifest lists {text} more than once")
        listed.add(text)
        path = root / relative
        current = root
        for part in relative.parts[:-1]:
            current /= part
            if current.is_symlink():
                raise CapsuleError(f"A payload path crosses a symbolic link: {text}")
        if not path.is_file() or path.is_symlink():
            raise CapsuleError(f"Missing capsule file: {text}")
        if expected_bytes != path.stat().st_size or expected_sha256 != sha256_file(path):
            raise CapsuleError(f"Capsule file does not match its manifest: {text}")

    observed: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise CapsuleError(f"The capsule contains a symbolic link: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise CapsuleError(f"The capsule contains a non-regular payload: {relative}")
        if relative != "manifest.json":
            observed.add(relative)
    if observed != listed:
        missing = sorted(listed - observed)
        added = sorted(observed - listed)
        detail = "; ".join(
            part
            for part in (
                f"missing: {', '.join(missing)}" if missing else "",
                f"unlisted: {', '.join(added)}" if added else "",
            )
            if part
        )
        raise CapsuleError(f"The capsule inventory differs ({detail})")
    return manifest


def file_record(manifest: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    records = manifest.get("files", ())
    matches = [record for record in records if record.get("name") == name]
    if len(matches) != 1:
        raise CapsuleError(f"The manifest does not define one {name!r} payload")
    return matches[0]


def load_array(root: Path, manifest: Mapping[str, Any], name: str) -> np.ndarray:
    record = file_record(manifest, name)
    path = Path(root) / _relative_path(record.get("path"), name)
    try:
        values = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise CapsuleError(f"Could not read {name}") from error
    expected_shape = record.get("shape")
    expected_dtype = record.get("dtype")
    if (
        not isinstance(values, np.ndarray)
        or list(values.shape) != expected_shape
        or values.dtype.str != expected_dtype
        or not np.issubdtype(values.dtype, np.number)
        or not np.isfinite(values).all()
    ):
        raise CapsuleError(f"{name} has the wrong shape, type or numeric content")
    return values


def read_tsv(
    root: Path,
    manifest: Mapping[str, Any],
    name: str,
    columns: Sequence[str],
) -> list[dict[str, str]]:
    record = file_record(manifest, name)
    path = Path(root) / _relative_path(record.get("path"), name)
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if tuple(reader.fieldnames or ()) != tuple(columns):
                raise CapsuleError(f"{name} has unexpected columns")
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise CapsuleError(f"Could not read {name}") from error
    if not rows:
        raise CapsuleError(f"{name} is empty")
    expected = set(columns)
    if any(
        set(row) != expected or any(value is None for value in row.values())
        for row in rows
    ):
        raise CapsuleError(f"{name} contains a malformed row")
    return rows
