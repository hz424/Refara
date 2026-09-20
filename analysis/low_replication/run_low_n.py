#!/usr/bin/env python3
"""Replay the prespecified Figure 5b extension with the archived generator."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def load():
    freeze = json.loads((HERE / 'LOW_N_FROZEN_SPEC.json').read_text())
    for name, sha in freeze['bound_sha256'].items():
        if digest(HERE / name) != sha:
            raise RuntimeError('Frozen extension source changed: ' + name)
    if np.__version__ != '1.26.4':
        raise RuntimeError('Replay requires NumPy 1.26.4')
    runner = module('low_n_runner', HERE / 'frozen_original/run_comparator_v1.py')
    numerics = module('low_n_numerics', HERE / 'comparator_numerics_low_n.py')
    core = module('low_n_core', HERE / 'frozen_original/general_directional_family_v3.py')
    spec = json.loads((HERE / 'frozen_original/COMPARATOR_EXECUTABLE_SPEC_V1.json').read_text())
    with (HERE / 'LOW_N_CELL_LEDGER.tsv').open(newline='') as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    return runner, numerics, core, spec, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-count', type=int, choices=(2, 3, 4, 6), required=True)
    parser.add_argument('--output-dir', type=Path, default=HERE / 'results')
    args = parser.parse_args()
    runner, numerics, core, spec, rows = load()
    row = next(r for r in rows if int(r['root_count']) == args.root_count)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f'root_{args.root_count:02d}_counts.json'
    if path.exists():
        raise FileExistsError(path)
    started = time.monotonic()
    cell = runner.run_cell(row, spec, numerics, core)
    payload = dict(
        cell=cell,
        runtime=dict(python=platform.python_version(), numpy=np.__version__),
        elapsed_seconds=time.monotonic() - started,
        low_n_frozen_spec_sha256=digest(HERE / 'LOW_N_FROZEN_SPEC.json'),
        seed_namespace=row['seed_namespace'],
        seed_formula='SHA256(namespace + NUL + cell_id + NUL + OUTER + NUL + six_digit_index + NUL + purpose); unsigned big-endian integer',
        outer_indices=[0, 2499], stream_purposes=list(runner.STREAM_LABELS),
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(root_count=args.root_count, outer_counts=cell['outer_counts'], elapsed_seconds=payload['elapsed_seconds'])), flush=True)
    if cell['outer_counts']['failed_replicates']:
        raise RuntimeError('At least one replicate failed, without replacement')


if __name__ == '__main__':
    main()
