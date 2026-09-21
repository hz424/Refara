"""Exercise symlink-style virtual environments without network installation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import venv

import pytest


REPO = Path(__file__).resolve().parents[1]


def load(relative):
    name = "_venv_regression_" + relative.replace("/", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def isolated_venv(tmp_path_factory):
    root = tmp_path_factory.mktemp("symlink-venv")
    venv.EnvBuilder(with_pip=False, symlinks=True).create(root)
    python = root / "bin/python"
    if not python.is_symlink():
        pytest.skip("requires a platform supporting symlink-style venvs")
    purelib = Path(subprocess.check_output(
        [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        text=True,
    ).strip())
    (purelib / "reference_cell_venv_probe.py").write_text("TOKEN = 'venv-only'\n")
    return root, python


@pytest.mark.parametrize("relative,function", [
    ("scripts/replay_all.py", "executable"),
])
def test_interpreter_helpers_keep_venv_imports(isolated_venv, relative, function):
    root, python = isolated_venv
    module = load(relative)
    selected = getattr(module, function)(python, label="legacy")
    result = json.loads(subprocess.check_output([
        str(selected), "-I", "-c",
        "import json,sys,reference_cell_venv_probe as p; "
        "print(json.dumps({'prefix':sys.prefix,'token':p.TOKEN}))",
    ], text=True))
    assert result == {"prefix": str(root), "token": "venv-only"}


