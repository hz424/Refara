#!/usr/bin/env python3
"""Opt-in download and verification of a separately hosted Source Data archive.

No Source Data DOI or archive URL is configured in this review version. Callers
must supply an explicit HTTPS archive URL, record identifier and SHA-256 digest.
``scripts/replay_all.py`` does not invoke this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile
import tempfile
from typing import BinaryIO, Sequence
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
import zipfile


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from perturb_nuisance_contracts.source_data_root import (  # noqa: E402
    EXTERNAL_MANIFEST_NAME,
    SourceDataRootError,
    validate_source_data_root,
)


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
DEFAULT_MAX_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_MAX_EXTRACTED_BYTES = 4 * 1024 * 1024 * 1024
DEFAULT_MAX_ARCHIVE_MEMBERS = 10_000


class FetchSourceDataError(RuntimeError):
    """Download, archive-safety, or publication precondition failed."""


class HTTPSOnlyRedirectHandler(HTTPRedirectHandler):
    """Reject every redirect hop that is not credential-free HTTPS."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlparse(newurl)
        require(
            parsed.scheme == "https"
            and bool(parsed.netloc)
            and parsed.username is None
            and parsed.password is None,
            "archive redirect is not credential-free HTTPS",
        )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FetchSourceDataError(message)


def safe_member_name(value: str) -> PurePosixPath:
    require("\\" not in value and "\x00" not in value, "archive member name is unsafe")
    path = PurePosixPath(value)
    require(
        value not in {"", "."}
        and not path.is_absolute()
        and all(part not in {"", ".", ".."} for part in path.parts),
        f"archive member path is unsafe: {value!r}",
    )
    return path


def copy_limited(source: BinaryIO, destination: BinaryIO, *, limit: int) -> int:
    total = 0
    while True:
        block = source.read(1024 * 1024)
        if not block:
            return total
        total += len(block)
        require(total <= limit, f"download exceeds byte limit {limit}")
        destination.write(block)


def download_archive(url: str, destination: Path, *, byte_limit: int) -> str:
    parsed = urlparse(url)
    require(
        parsed.scheme == "https"
        and bool(parsed.netloc)
        and parsed.username is None
        and parsed.password is None,
        "archive URL must be explicit credential-free HTTPS",
    )
    request = Request(url, headers={"User-Agent": "reference-cell-code/1.8.15"})
    digest = hashlib.sha256()
    total = 0
    try:
        opener = build_opener(HTTPSOnlyRedirectHandler())
        with opener.open(request, timeout=60) as response, destination.open("xb") as output:
            final = urlparse(response.geturl())
            require(
                final.scheme == "https"
                and bool(final.netloc)
                and final.username is None
                and final.password is None,
                "archive redirect left HTTPS",
            )
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                require(total <= byte_limit, f"download exceeds byte limit {byte_limit}")
                digest.update(block)
                output.write(block)
    except (OSError, ValueError) as error:
        raise FetchSourceDataError("Source Data archive download failed") from error
    require(total > 0, "downloaded archive is empty")
    return digest.hexdigest()


def destination_for_member(root: Path, name: str) -> Path:
    relative = safe_member_name(name)
    destination = root.joinpath(*relative.parts)
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = parent.resolve(strict=True)
    require(
        resolved_parent == root or root in resolved_parent.parents,
        f"archive member escapes extraction root: {name!r}",
    )
    require(
        not destination.exists() and not destination.is_symlink(),
        f"archive contains a duplicate member: {name!r}",
    )
    return destination


def extract_zip(
    archive: Path,
    root: Path,
    *,
    byte_limit: int,
    member_limit: int,
) -> None:
    try:
        with zipfile.ZipFile(archive) as source:
            members = source.infolist()
            require(len(members) <= member_limit, f"ZIP exceeds member limit {member_limit}")
            total = sum(member.file_size for member in members)
            require(total <= byte_limit, f"extracted ZIP exceeds byte limit {byte_limit}")
            for member in members:
                path = safe_member_name(member.filename.rstrip("/"))
                mode = member.external_attr >> 16
                require(
                    not stat.S_ISLNK(mode),
                    f"zip archive contains a symbolic link: {member.filename!r}",
                )
                destination = root.joinpath(*path.parts)
                if member.is_dir():
                    require(
                        not destination.exists() or destination.is_dir(),
                        f"zip directory conflicts with a file: {member.filename!r}",
                    )
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination = destination_for_member(root, member.filename)
                with source.open(member) as input_handle, destination.open("xb") as output:
                    observed = copy_limited(
                        input_handle,
                        output,
                        limit=member.file_size,
                    )
                require(observed == member.file_size, f"ZIP member size differs: {member.filename!r}")
    except (OSError, zipfile.BadZipFile) as error:
        raise FetchSourceDataError("invalid or unreadable ZIP archive") from error


def extract_tar(
    archive: Path,
    root: Path,
    *,
    byte_limit: int,
    member_limit: int,
) -> None:
    try:
        with tarfile.open(archive, mode="r|*") as source:
            member_count = 0
            total = 0
            for member in source:
                member_count += 1
                require(member_count <= member_limit, f"TAR exceeds member limit {member_limit}")
                if member.isfile():
                    total += member.size
                    require(total <= byte_limit, f"extracted TAR exceeds byte limit {byte_limit}")
                path = safe_member_name(member.name.rstrip("/"))
                destination = root.joinpath(*path.parts)
                require(
                    not (member.issym() or member.islnk() or member.isdev()),
                    f"tar archive contains a link or special member: {member.name!r}",
                )
                if member.isdir():
                    require(
                        not destination.exists() or destination.is_dir(),
                        f"tar directory conflicts with a file: {member.name!r}",
                    )
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                require(member.isfile(), f"unsupported tar member: {member.name!r}")
                destination = destination_for_member(root, member.name)
                input_handle = source.extractfile(member)
                require(input_handle is not None, f"cannot extract tar member: {member.name!r}")
                with input_handle, destination.open("xb") as output:
                    observed = copy_limited(
                        input_handle,
                        output,
                        limit=member.size,
                    )
                require(observed == member.size, f"TAR member size differs: {member.name!r}")
    except (OSError, tarfile.TarError) as error:
        raise FetchSourceDataError("invalid or unreadable TAR archive") from error


def extract_archive(
    archive: Path,
    root: Path,
    *,
    byte_limit: int,
    member_limit: int,
) -> None:
    if zipfile.is_zipfile(archive):
        extract_zip(
            archive,
            root,
            byte_limit=byte_limit,
            member_limit=member_limit,
        )
        return
    if tarfile.is_tarfile(archive):
        extract_tar(
            archive,
            root,
            byte_limit=byte_limit,
            member_limit=member_limit,
        )
        return
    raise FetchSourceDataError("archive must be ZIP or a TAR format recognized by Python")


def locate_source_data_root(extraction_root: Path) -> Path:
    candidates = sorted(
        path.parent
        for path in extraction_root.rglob(EXTERNAL_MANIFEST_NAME)
        if path.is_file() and not path.is_symlink()
    )
    require(len(candidates) == 1, "archive must contain exactly one Source Data root manifest")
    candidate = candidates[0]
    require(
        candidate == extraction_root or candidate.parent == extraction_root,
        "Source Data root must be the archive root or its single top-level directory",
    )
    return candidate


def fetch(
    *,
    url: str,
    record_id: str,
    expected_archive_sha256: str,
    output_dir: Path,
    doi: str | None,
    max_archive_bytes: int,
    max_extracted_bytes: int,
    max_archive_members: int,
) -> dict[str, object]:
    require(bool(record_id.strip()), "record identifier must be non-empty")
    require(
        SHA256_PATTERN.fullmatch(expected_archive_sha256) is not None,
        "expected archive SHA-256 must be 64 lowercase hexadecimal characters",
    )
    require(max_archive_bytes > 0, "maximum archive byte count must be positive")
    require(max_extracted_bytes > 0, "maximum extracted byte count must be positive")
    require(max_archive_members > 0, "maximum archive member count must be positive")
    supplied_output = output_dir.expanduser()
    require(not supplied_output.is_symlink(), "output directory cannot be a symlink")
    output = supplied_output.resolve()
    require(not output.exists() and not output.is_symlink(), "output directory must not exist")
    require(output.parent.is_dir() and not output.parent.is_symlink(), "output parent is absent or unsafe")

    with tempfile.TemporaryDirectory(
        prefix=".source-data-fetch-",
        dir=output.parent,
    ) as temporary:
        temporary_root = Path(temporary)
        archive = temporary_root / "download.archive"
        observed_archive_sha256 = download_archive(
            url,
            archive,
            byte_limit=max_archive_bytes,
        )
        require(
            observed_archive_sha256 == expected_archive_sha256,
            "downloaded archive SHA-256 differs from the caller-supplied digest",
        )
        extraction = temporary_root / "extracted"
        extraction.mkdir()
        extract_archive(
            archive,
            extraction,
            byte_limit=max_extracted_bytes,
            member_limit=max_archive_members,
        )
        candidate = locate_source_data_root(extraction)
        snapshot = validate_source_data_root(candidate)
        snapshot.assert_unchanged()
        os.replace(candidate, output)

    return {
        "schema": "REFERENCE_CELL_SOURCE_DATA_FETCH_V1",
        "status": "PASS_FETCHED_AND_VERIFIED_SOURCE_DATA_V1",
        "record_id": record_id,
        "doi": doi,
        "archive_sha256": expected_archive_sha256,
        "source_data_manifest_sha256": snapshot.manifest_sha256,
        "payload_count": len(snapshot.payloads),
        "output_dir": str(output),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--url",
        required=True,
        help="explicit HTTPS URL for a ZIP or TAR Source Data archive",
    )
    result.add_argument(
        "--max-extracted-bytes",
        type=int,
        default=DEFAULT_MAX_EXTRACTED_BYTES,
        help=f"extracted payload limit in bytes (default: {DEFAULT_MAX_EXTRACTED_BYTES})",
    )
    result.add_argument(
        "--max-archive-members",
        type=int,
        default=DEFAULT_MAX_ARCHIVE_MEMBERS,
        help=f"archive member limit (default: {DEFAULT_MAX_ARCHIVE_MEMBERS})",
    )
    result.add_argument(
        "--record-id",
        required=True,
        help="human-readable repository record/version identifier",
    )
    result.add_argument(
        "--expected-archive-sha256",
        required=True,
        help="independently obtained lowercase SHA-256 of the complete archive",
    )
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument(
        "--doi",
        help=(
            "optional DOI metadata only; DOI resolution is not supported, "
            "so --url and --expected-archive-sha256 remain required"
        ),
    )
    result.add_argument(
        "--max-archive-bytes",
        type=int,
        default=DEFAULT_MAX_ARCHIVE_BYTES,
        help=f"download limit in bytes (default: {DEFAULT_MAX_ARCHIVE_BYTES})",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = fetch(
        url=args.url,
        record_id=args.record_id,
        expected_archive_sha256=args.expected_archive_sha256,
        output_dir=args.output_dir,
        doi=args.doi,
        max_archive_bytes=args.max_archive_bytes,
        max_extracted_bytes=args.max_extracted_bytes,
        max_archive_members=args.max_archive_members,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FetchSourceDataError, SourceDataRootError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
