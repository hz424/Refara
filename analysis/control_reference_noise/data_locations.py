"""Locations of inputs that are not distributed with this repository.

Every script in this directory reads released study outputs (per-seed utility arrays, per-cell prediction banks,
score caches, fitted models) that are too large or too restricted to bundle. Set the environment variables below;
a script stops with a message naming the variable it needs.

REFARA_EVALUATION_ROOT   directory containing <dataset>/<regime>/ with EVALUATION_BINDING.json,
                         PANEL_SEED_UTILITIES.npy, banks/ and score_cache_v1/ (datasets: parse, gse162632)
REFARA_OUTPUT_ROOT       directory for outputs (created if absent)
REFARA_TRAINING_CODE     directory of the study's training and evaluation code (control_bank.py only)
REFARA_PARSE_TASKS       evaluation task table of the Parse analysis, TSV (disjoint_cells.py only)
REFARA_PARSE_CONTROL_POOL  directory with pool_expr_log1p.npy and groups.json, the 768 hash-ranked control
                         cells per donor and cell type (disjoint_cells.py only)
"""
from __future__ import annotations

import os
from pathlib import Path


def _directory(name: str, what: str, create: bool = False) -> Path:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Set {name} to {what}. See analysis/control_reference_noise/README.md.")
    path = Path(value)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    elif not path.exists():
        raise SystemExit(f"{name}={value} does not exist.")
    return path


def evaluation_root() -> Path:
    return _directory("REFARA_EVALUATION_ROOT", "the directory holding <dataset>/<regime>/ evaluation outputs")


def output_root() -> Path:
    return _directory("REFARA_OUTPUT_ROOT", "a directory for outputs", create=True)


def training_code() -> Path:
    return _directory("REFARA_TRAINING_CODE", "the directory of the study's training and evaluation code")


def parse_tasks() -> Path:
    return _directory("REFARA_PARSE_TASKS", "the Parse evaluation task table (TSV)")


def parse_control_pool() -> Path:
    return _directory("REFARA_PARSE_CONTROL_POOL", "the directory with pool_expr_log1p.npy and groups.json")
