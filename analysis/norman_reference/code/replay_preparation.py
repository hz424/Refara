#!/usr/bin/env python3
"""Replay frozen Norman preparation from a verified public count matrix.

The supplied preparation program is copied byte-for-byte to a temporary layout
so its original analysis hash and relative paths remain valid.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5ad", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--frozen-protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, help="Optional supplied sufficient statistics for numerical replay comparison.")
    parser.add_argument("--preparation-program", type=Path, default=Path(__file__).with_name("prepare_norman_task.py"))
    args = parser.parse_args()
    protocol = json.loads(args.frozen_protocol.read_text())
    expected_source = next(x for x in protocol["source_files"] if x["file"] == "source/norman_2019_raw.h5ad")
    assert sha256(args.source_h5ad) == expected_source["sha256"], "Source matrix checksum differs."
    assert sha256(args.preparation_program) == protocol["script_sha256"], "Frozen preparation program checksum differs."
    if args.output_dir.exists():
        raise FileExistsError("Choose a new output directory; existing analyses are not overwritten.")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="norman_replay_", dir=args.output_dir.parent) as temp:
        root = Path(temp)
        (root / "source").mkdir()
        (root / "prepared").mkdir()
        (root / "metadata").symlink_to(args.metadata_dir.resolve(), target_is_directory=True)
        (root / "source" / "norman_2019_raw.h5ad").symlink_to(args.source_h5ad.resolve())
        shutil.copyfile(args.preparation_program, root / "prepare_norman_task.py")
        shutil.copyfile(args.frozen_protocol, root / "prepared" / "FROZEN_TASK_PROTOCOL.json")
        env = dict(os.environ)
        env.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        subprocess.run([sys.executable, "-u", str(root / "prepare_norman_task.py")], check=True, env=env)
        result = {"status": "PASS", "source_sha256": expected_source["sha256"], "frozen_protocol_sha256": sha256(args.frozen_protocol), "comparisons": {}}
        if args.reference_dir:
            for name in ["eval_controls.npy", "train_control_mean.npy", "scales.npy"]:
                expected = args.reference_dir / name
                if expected.exists():
                    a, b = np.load(expected, mmap_mode="r"), np.load(root / "prepared" / name, mmap_mode="r")
                    assert np.array_equal(a, b), name
                    result["comparisons"][name] = "exact_array_equal"
            for name in ["test_means.npz", "training_means.npz"]:
                expected = args.reference_dir / name
                if expected.exists():
                    with np.load(expected) as a, np.load(root / "prepared" / name) as b:
                        assert set(a.files) == set(b.files), name
                        assert all(np.array_equal(a[k], b[k]) for k in a.files), name
                    result["comparisons"][name] = "exact_arrays_equal"
        (root / "prepared" / "REPLAY_VALIDATION.json").write_text(json.dumps(result, indent=2) + "\n")
        shutil.move(str(root / "prepared"), str(args.output_dir))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
