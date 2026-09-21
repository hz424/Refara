"""Validate the external Source Data tree and copy verified files into a replay workspace."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
from typing import Any, Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPECTED_MANIFEST = (
    REPOSITORY_ROOT / "provenance" / "SOURCE_DATA_ROOT_MANIFEST_V1.json"
)
EXTERNAL_MANIFEST_NAME = "SOURCE_DATA_ROOT_MANIFEST_V1.json"
EXPECTED_SOURCE_DATA_RELEASE_VERSION = "1.10.0"
EXPECTED_TOP_LEVEL_KEYS = {
    "payload_count",
    "payloads",
    "protected_external_namespaces",
    "record_type",
    "release_version",
    "schema_version",
    "status",
}
EXPECTED_PAYLOAD_KEYS = {
    "bytes",
    "external_path",
    "role",
    "runtime_path",
    "sha256",
}
EXPECTED_ROLES = {
    "CMODEL_SUCCESSOR_EVIDENCE",
    "FIGURE_SOURCE_DATA",
    "HELD_OUT_EVIDENCE",
    "MULTI_REALIZATION_SENSITIVITY",
    "V22_ATTRIBUTION_EVIDENCE",
}
EXPECTED_NAMESPACES = (
    "Extended_Data_Figure_2_attribution",
    "Figure_4_attribution",
    "data/derived",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class SourceDataRootError(RuntimeError):
    """The external Source Data identity or filesystem contract failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceDataRootError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json_bytes(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise SourceDataRootError(f"cannot read {label}: {path}") from error

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key in {label}: {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON number {token}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise SourceDataRootError(f"invalid strict JSON in {label}: {path}") from error
    require(isinstance(value, dict), f"{label} root must be an object")
    canonical = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    require(raw == canonical, f"{label} is not canonical JSON")
    return value, raw


def _safe_relative(value: Any, *, label: str) -> PurePosixPath:
    require(isinstance(value, str) and value != "", f"{label} must be a string")
    path = PurePosixPath(value)
    require(not path.is_absolute(), f"{label} must be relative")
    require(
        path.as_posix() == value
        and all(part not in {"", ".", ".."} for part in path.parts),
        f"{label} is not a normalized safe POSIX path: {value!r}",
    )
    return path


def _under_namespace(path: PurePosixPath, namespace: str) -> bool:
    prefix = PurePosixPath(namespace).parts
    return path.parts[: len(prefix)] == prefix


@dataclass(frozen=True)
class PayloadRecord:
    external_path: PurePosixPath
    runtime_path: PurePosixPath
    bytes: int
    sha256: str
    role: str


@dataclass(frozen=True)
class SourceDataSnapshot:
    root: Path
    expected_manifest: Path
    manifest_sha256: str
    payloads: tuple[PayloadRecord, ...]
    fingerprint: str

    def by_runtime_path(self) -> dict[str, PayloadRecord]:
        return {record.runtime_path.as_posix(): record for record in self.payloads}

    def assert_unchanged(self) -> None:
        observed = validate_source_data_root(
            self.root,
            expected_manifest=self.expected_manifest,
        )
        require(
            observed.manifest_sha256 == self.manifest_sha256,
            "Source Data manifest changed during replay",
        )
        require(
            observed.fingerprint == self.fingerprint,
            "Source Data payload fingerprint changed during replay",
        )


def _parse_expected_manifest(
    path: Path,
) -> tuple[tuple[PayloadRecord, ...], bytes]:
    require(path.is_file() and not path.is_symlink(), "expected manifest is absent or unsafe")
    value, raw = _strict_json_bytes(path, label="bundled Source Data manifest")
    require(set(value) == EXPECTED_TOP_LEVEL_KEYS, "Source Data manifest fields differ")
    require(value.get("schema_version") == 1, "Source Data manifest schema differs")
    require(
        value.get("record_type") == "REFERENCE_CELL_SOURCE_DATA_ROOT_MANIFEST_V1",
        "Source Data manifest record type differs",
    )
    require(
        value.get("release_version") == EXPECTED_SOURCE_DATA_RELEASE_VERSION,
        "Source Data release version differs",
    )
    require(
        value.get("status") == "FROZEN_CODE_ONLY_SOURCE_DATA_BINDING",
        "Source Data manifest status differs",
    )
    require(
        value.get("protected_external_namespaces") == list(EXPECTED_NAMESPACES),
        "protected Source Data namespaces differ",
    )
    payload_values = value.get("payloads")
    require(isinstance(payload_values, list), "Source Data payload inventory is not a list")
    require(isinstance(value.get("payload_count"), int) and value["payload_count"] == len(payload_values) and len(payload_values) >= 74, "payload count differs from the release-bound inventory")

    records: list[PayloadRecord] = []
    for index, item in enumerate(payload_values):
        require(isinstance(item, Mapping), f"payload record {index} is not an object")
        require(set(item) == EXPECTED_PAYLOAD_KEYS, f"payload record {index} fields differ")
        external = _safe_relative(item.get("external_path"), label=f"payload {index} external_path")
        runtime = _safe_relative(item.get("runtime_path"), label=f"payload {index} runtime_path")
        size = item.get("bytes")
        digest = item.get("sha256")
        role = item.get("role")
        require(isinstance(size, int) and not isinstance(size, bool) and size >= 0, f"payload {index} bytes differs")
        require(isinstance(digest, str) and SHA256_PATTERN.fullmatch(digest) is not None, f"payload {index} SHA-256 differs")
        require(role in EXPECTED_ROLES, f"payload {index} role differs")
        require(
            any(_under_namespace(external, namespace) for namespace in EXPECTED_NAMESPACES),
            f"payload {index} is outside protected external namespaces",
        )
        require(
            runtime.parts[:2] == ("data", "derived"),
            f"payload {index} runtime path is outside data/derived",
        )
        records.append(PayloadRecord(external, runtime, size, digest, str(role)))

    external_names = [record.external_path.as_posix() for record in records]
    runtime_names = [record.runtime_path.as_posix() for record in records]
    require(external_names == sorted(external_names), "payload records are not external-path sorted")
    require(len(set(external_names)) == len(records), "duplicate external payload path")
    require(len(set(runtime_names)) == len(records), "duplicate runtime payload path")
    multi = [record for record in records if record.role == "MULTI_REALIZATION_SENSITIVITY"]
    require(len(multi) == 13, "multi-realization payload inventory is not exactly 13")
    require(
        all(
            record.external_path.parts[:3]
            == ("data", "derived", "v21_multi_realization_sensitivity")
            for record in multi
        ),
        "multi-realization payload directory differs",
    )
    cmodel = [record for record in records if record.role == "CMODEL_SUCCESSOR_EVIDENCE"]
    require(len(cmodel) == 4, "Cmodel-successor payload inventory is not exactly 4")
    require(
        all(
            record.external_path.parts[:1] == ("Extended_Data_Figure_2_attribution",)
            and record.runtime_path.parts[:3]
            == ("data", "derived", "cmodel_successor")
            and record.external_path.name == record.runtime_path.name
            for record in cmodel
        ),
        "Cmodel-successor Source Data paths differ",
    )
    aliases = [record for record in records if record.external_path != record.runtime_path]
    require(len(aliases) == 12, "Source Data runtime alias count differs")
    attribution_aliases = [
        record
        for record in aliases
        if record.external_path.parts[0] == "Figure_4_attribution"
    ]
    require(
        len(attribution_aliases) == 4
        and
        all(
            record.external_path.parts[0] == "Figure_4_attribution"
            and record.runtime_path.parts[:3]
            == ("data", "derived", "figure6_attribution")
            and record.external_path.name == record.runtime_path.name
            for record in attribution_aliases
        ),
        "Figure 4 attribution runtime aliases differ",
    )
    jerber_aliases = [
        record
        for record in aliases
        if record.external_path.parts[:3]
        == ("data", "derived", "extended_data_figure_4_jerber")
    ]
    require(
        len(jerber_aliases) == 4
        and all(
            record.runtime_path.parts[:3]
            == ("data", "derived", "extended_data_figure_3_jerber")
            and record.external_path.name == record.runtime_path.name
            for record in jerber_aliases
        ),
        "Extended Data Figure 4 Jerber runtime aliases differ",
    )
    require(
        set(cmodel) | set(attribution_aliases) | set(jerber_aliases) == set(aliases),
        "unexpected Source Data runtime alias",
    )
    return tuple(records), raw


def _regular_payload(root: Path, relative: PurePosixPath) -> Path:
    current = root
    for part in relative.parts:
        current = current / part
        require(not current.is_symlink(), f"Source Data path contains a symlink: {relative}")
    try:
        resolved = current.resolve(strict=True)
    except OSError as error:
        raise SourceDataRootError(f"Source Data payload is absent: {relative}") from error
    require(resolved == current, f"Source Data path resolution differs: {relative}")
    try:
        mode = resolved.stat().st_mode
    except OSError as error:
        raise SourceDataRootError(f"cannot stat Source Data payload: {relative}") from error
    require(stat.S_ISREG(mode), f"Source Data payload is not a regular file: {relative}")
    return resolved


def _namespace_files(root: Path, namespace: str) -> set[str]:
    base = root / namespace
    require(base.is_dir() and not base.is_symlink(), f"protected namespace is absent or unsafe: {namespace}")
    observed: set[str] = set()

    def traversal_error(error: OSError) -> None:
        raise SourceDataRootError(
            f"cannot traverse protected namespace {namespace}: {error}"
        ) from error

    for current, directories, files in os.walk(
        base,
        followlinks=False,
        onerror=traversal_error,
    ):
        current_path = Path(current)
        for name in directories:
            child = current_path / name
            require(not child.is_symlink(), f"protected namespace contains a symlink: {child.relative_to(root)}")
        for name in files:
            child = current_path / name
            relative = child.relative_to(root).as_posix()
            require(not child.is_symlink(), f"protected namespace contains a symlink: {relative}")
            require(stat.S_ISREG(child.stat().st_mode), f"protected namespace contains a special file: {relative}")
            observed.add(relative)
    return observed


def _fingerprint(records: Iterable[PayloadRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.external_path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(record.runtime_path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record.bytes).encode("ascii"))
        digest.update(b"\0")
        digest.update(record.sha256.encode("ascii"))
        digest.update(b"\0")
        digest.update(record.role.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _safe_runtime_parent(destination: Path, runtime: PurePosixPath) -> Path:
    """Create a runtime parent without ever traversing a link or special node."""

    current = destination
    for part in runtime.parts[:-1]:
        child = current / part
        if child.exists() or child.is_symlink():
            require(not child.is_symlink(), f"copy destination parent is a symlink: {runtime}")
            try:
                mode = child.stat().st_mode
            except OSError as error:
                raise SourceDataRootError(
                    f"cannot inspect copy destination parent: {runtime}"
                ) from error
            require(stat.S_ISDIR(mode), f"copy destination parent is not a directory: {runtime}")
        else:
            try:
                child.mkdir()
            except OSError as error:
                raise SourceDataRootError(
                    f"cannot create copy destination parent: {runtime}"
                ) from error
            require(
                child.is_dir() and not child.is_symlink(),
                f"created copy destination parent is unsafe: {runtime}",
            )
        resolved = child.resolve(strict=True)
        require(
            resolved == child and destination in resolved.parents,
            f"copy destination parent escapes the destination directory: {runtime}",
        )
        current = child
    return current


def validate_source_data_root(
    source_data_root: Path | str,
    *,
    expected_manifest: Path | str = DEFAULT_EXPECTED_MANIFEST,
) -> SourceDataSnapshot:
    """Validate an extracted ``05_SOURCE_DATA`` directory, fail closed."""

    expected_supplied = Path(expected_manifest).expanduser()
    require(not expected_supplied.is_symlink(), "expected manifest path is a symlink")
    try:
        expected_path = expected_supplied.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise SourceDataRootError("expected manifest path cannot be resolved") from error
    records, expected_raw = _parse_expected_manifest(expected_path)
    supplied = Path(source_data_root).expanduser()
    require(not supplied.is_symlink(), f"Source Data root is a symlink: {supplied}")
    try:
        root = supplied.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise SourceDataRootError(f"Source Data root is absent: {supplied}") from error
    require(root.is_dir(), f"Source Data root is not a directory: {root}")

    external_manifest = root / EXTERNAL_MANIFEST_NAME
    require(
        external_manifest.is_file() and not external_manifest.is_symlink(),
        f"Source Data root lacks a regular {EXTERNAL_MANIFEST_NAME}",
    )
    _value, external_raw = _strict_json_bytes(
        external_manifest,
        label="external Source Data manifest",
    )
    require(external_raw == expected_raw, "external and bundled Source Data manifests differ")

    expected_by_namespace = {
        namespace: {
            record.external_path.as_posix()
            for record in records
            if _under_namespace(record.external_path, namespace)
        }
        for namespace in EXPECTED_NAMESPACES
    }
    for namespace, expected_names in expected_by_namespace.items():
        observed_names = _namespace_files(root, namespace)
        require(
            observed_names == expected_names,
            f"protected Source Data inventory differs under {namespace}",
        )

    for record in records:
        path = _regular_payload(root, record.external_path)
        require(path.stat().st_size == record.bytes, f"Source Data byte count differs: {record.external_path}")
        require(sha256_file(path) == record.sha256, f"Source Data digest differs: {record.external_path}")

    return SourceDataSnapshot(
        root=root,
        expected_manifest=expected_path,
        manifest_sha256=sha256_bytes(expected_raw),
        payloads=records,
        fingerprint=_fingerprint(records),
    )


def hydrate_source_data(
    snapshot: SourceDataSnapshot,
    destination_repository: Path | str,
    *,
    runtime_paths: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Copy validated payloads into an otherwise data-free temporary tree."""

    supplied_destination = Path(destination_repository)
    require(not supplied_destination.is_symlink(), "copy destination is a symlink")
    destination = supplied_destination.resolve(strict=True)
    require(destination.is_dir(), "copy destination is not a directory")
    require(
        destination != snapshot.root
        and snapshot.root not in destination.parents
        and destination not in snapshot.root.parents,
        "copy destination and Source Data root must be disjoint",
    )
    by_runtime = snapshot.by_runtime_path()
    selected = set(by_runtime) if runtime_paths is None else set(runtime_paths)
    require(selected <= set(by_runtime), "copy request includes an unknown runtime path")
    copied: list[str] = []
    for relative in sorted(selected):
        record = by_runtime[relative]
        source = _regular_payload(snapshot.root, record.external_path)
        parent = _safe_runtime_parent(destination, record.runtime_path)
        target = parent / record.runtime_path.name
        require(not target.exists() and not target.is_symlink(), f"copy target already exists: {relative}")
        try:
            with source.open("rb") as input_handle, target.open("xb") as output_handle:
                shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
        except OSError as error:
            raise SourceDataRootError(f"cannot copy runtime payload: {relative}") from error
        require(
            target.is_file()
            and not target.is_symlink()
            and target.resolve(strict=True) == target,
            f"copied target is unsafe: {relative}",
        )
        require(target.stat().st_size == record.bytes, f"copied file size differs: {relative}")
        require(sha256_file(target) == record.sha256, f"copied file digest differs: {relative}")
        copied.append(relative)
    snapshot.assert_unchanged()
    return tuple(copied)
