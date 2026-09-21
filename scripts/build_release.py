#!/usr/bin/env python3
"""Build the wheel and complete source archive from a clean tagged checkout."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    def git(*arguments: str) -> str:
        return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()

    if git("status", "--porcelain", "--untracked-files=normal"):
        raise SystemExit("Commit or remove working-tree changes before building a release.")
    version = re.search(r'^version = "([^"]+)"$', (root / "pyproject.toml").read_text(), re.M).group(1)
    head = git("rev-parse", "HEAD")
    manifested = set()
    for line in (root / "MANIFEST.sha256").read_text().splitlines():
        digest, filename = line.split("  ", 1)
        manifested.add(filename)
        if hashlib.sha256((root / filename).read_bytes()).hexdigest() != digest:
            raise SystemExit(f"Manifest mismatch: {filename}")
    tracked = set(git("ls-files").splitlines()) - {"MANIFEST.sha256"}
    if manifested != tracked:
        raise SystemExit("Manifest must cover every tracked file except itself.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise SystemExit("Use an empty output directory.")
    source_archive = output / ('refara-' + version + '-source.zip')
    subprocess.run(["git", "archive", "--format=zip", f"--prefix=refara-{version}/",
                    f"--output={source_archive}", "HEAD"],
                   cwd=root, check=True)
    # Build only the committed payload; ignored build/lib leftovers must not
    # leak an obsolete module into an otherwise clean checkout's wheel.
    with tempfile.TemporaryDirectory(prefix="refara-build-") as temporary:
        with zipfile.ZipFile(source_archive) as archive:
            archive.extractall(temporary)
        frozen_source = Path(temporary) / f"refara-{version}"
        subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                        "--wheel-dir", str(output), str(frozen_source)], check=True,
                       env=dict(os.environ, SOURCE_DATE_EPOCH=git("show", "-s", "--format=%ct", "HEAD")))
    assets = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(output.iterdir()) if path.is_file()}
    receipt = {"version": version, "commit": head, "tree": git("rev-parse", "HEAD^{tree}"),
               "assets_sha256": assets, "source_manifest_sha256":
               hashlib.sha256((root / "MANIFEST.sha256").read_bytes()).hexdigest()}
    (output / "RELEASE.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (output / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in sorted(output.iterdir()) if path.is_file()))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
