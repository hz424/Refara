#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPOSITORY=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3.10}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    printf 'Python 3.10 was not found. Set PYTHON_BIN to a Python 3.10 executable.\n' >&2
    exit 2
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(
        f"Python 3.10 is required; found {sys.version_info.major}.{sys.version_info.minor}"
    )
PY

DEMO_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/reference-cell-demo.XXXXXX")
"$PYTHON_BIN" -m venv "$DEMO_ROOT/venv"
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
cd "$REPOSITORY"
"$DEMO_ROOT/venv/bin/python" -I -s -B scripts/run_quick_demo.py \
    --output-root "$DEMO_ROOT/results"
