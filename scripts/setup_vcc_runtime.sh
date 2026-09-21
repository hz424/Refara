#!/usr/bin/env bash
# Create separate adapter and official-scorer environments in a NEW directory.
set -Eeuo pipefail

vcc_repo=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
vcc_python310=''
vcc_python312=''
vcc_output=''
vcc_commit='5e64833518a6603a0301cbe28185d49c30f4a986'

vcc_usage() {
    cat <<'USAGE'
Usage: bash scripts/setup_vcc_runtime.sh \
  --python310 /absolute/path/to/python3.10 \
  --python312 /absolute/path/to/python3.12 \
  --output /absolute/path/to/new-runtime-directory

Creates py310/, official/, and official_source/ without changing existing environments.
Run installation in an appropriate CPU allocation. See docs/VCC_INSTALLATION.md.
USAGE
}

while (($#)); do
    case "$1" in
        --python310|--python312|--output)
            if (($# < 2)); then vcc_usage >&2; exit 2; fi
            case "$1" in
                --python310) vcc_python310=$2 ;;
                --python312) vcc_python312=$2 ;;
                --output) vcc_output=$2 ;;
            esac
            shift 2 ;;
        --help|-h) vcc_usage; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; vcc_usage >&2; exit 2 ;;
    esac
done

for vcc_python in "$vcc_python310" "$vcc_python312"; do
    if [[ "$vcc_python" != /* || ! -x "$vcc_python" ]]; then
        printf 'Supply an absolute executable path for each Python interpreter.\n' >&2
        exit 2
    fi
done
if [[ "$vcc_output" != /* || -e "$vcc_output" || -L "$vcc_output" ]]; then
    printf 'The output must be an absolute path that does not already exist.\n' >&2
    exit 2
fi
command -v git >/dev/null
"$vcc_python310" -I -c 'import sys; assert sys.version_info[:2] == (3, 10), "--python310 must select CPython 3.10"; assert sys.implementation.name == "cpython"'
"$vcc_python312" -I -c 'import sys; assert sys.version_info[:2] == (3, 12), "--python312 must select CPython 3.12"; assert sys.implementation.name == "cpython"'

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
vcc_threads=${SLURM_CPUS_PER_TASK:-2}
if [[ ! "$vcc_threads" =~ ^[1-9][0-9]*$ ]]; then
    printf 'SLURM_CPUS_PER_TASK must be a positive integer when set.\n' >&2
    exit 2
fi
export OMP_NUM_THREADS="$vcc_threads"
export OPENBLAS_NUM_THREADS="$vcc_threads"
export MKL_NUM_THREADS="$vcc_threads"
export NUMBA_NUM_THREADS="$vcc_threads"
export POLARS_MAX_THREADS="$vcc_threads"

mkdir -p -- "$(dirname -- "$vcc_output")"
mkdir -- "$vcc_output"
mkdir -- "$vcc_output/logs"
vcc_stage='create_environments'
vcc_fail() {
    local vcc_exit=$?
    trap - ERR
    printf 'status=FAILED\nstage=%s\nexit_code=%s\n' "$vcc_stage" "$vcc_exit" > "$vcc_output/SETUP_FAILED.txt"
    printf 'Setup failed at %s. Logs: %s/logs\n' "$vcc_stage" "$vcc_output" >&2
    exit "$vcc_exit"
}
trap vcc_fail ERR

printf 'Creating CPython 3.10 and 3.12 environments.\n'
"$vcc_python310" -I -m venv "$vcc_output/py310"
"$vcc_python312" -I -m venv "$vcc_output/official"

vcc_stage='install_adapter_dependencies'
printf 'Installing pinned adapter dependencies.\n'
"$vcc_output/py310/bin/python" -I -m pip install --no-deps \
    -r "$vcc_repo/requirements/vcc-build-py310.txt" \
    -r "$vcc_repo/requirements/vcc-py310-linux.txt" \
    -r "$vcc_repo/requirements/unified-prepare-py310.txt" > "$vcc_output/logs/adapter_dependencies.log" 2>&1
vcc_stage='install_editable_adapter'
"$vcc_output/py310/bin/python" -I -m pip install --no-deps --no-build-isolation -e "$vcc_repo" \
    > "$vcc_output/logs/adapter_install.log" 2>&1
"$vcc_output/py310/bin/python" -I -m pip check > "$vcc_output/logs/adapter_pip_check.log" 2>&1

vcc_stage='checkout_official_source'
printf 'Checking out official cell-eval2 at %s.\n' "$vcc_commit"
git clone --filter=blob:none --no-checkout https://github.com/ArcInstitute/cell-eval2.git \
    "$vcc_output/official_source" > "$vcc_output/logs/official_checkout.log" 2>&1
git -C "$vcc_output/official_source" checkout --detach "$vcc_commit" \
    >> "$vcc_output/logs/official_checkout.log" 2>&1
[[ $(git -C "$vcc_output/official_source" rev-parse HEAD) == "$vcc_commit" ]]

vcc_stage='install_official_dependencies'
printf 'Installing pinned CPU scorer dependencies.\n'
"$vcc_output/official/bin/python" -I -m pip install --no-deps \
    -r "$vcc_repo/requirements/vcc-build-py312.txt" \
    -r "$vcc_repo/requirements/vcc-py312-linux.txt" > "$vcc_output/logs/official_dependencies.log" 2>&1
vcc_stage='install_official_scorer'
"$vcc_output/official/bin/python" -I -m pip install --no-deps --no-build-isolation \
    "$vcc_output/official_source" > "$vcc_output/logs/official_install.log" 2>&1
"$vcc_output/official/bin/python" -I -m pip check > "$vcc_output/logs/official_pip_check.log" 2>&1

vcc_stage='record_and_check_runtime_metadata'
"$vcc_output/py310/bin/python" -I -m pip freeze --all > "$vcc_output/environment_py310.txt"
"$vcc_output/official/bin/python" -I -m pip freeze --all > "$vcc_output/environment_official.txt"
"$vcc_output/py310/bin/python" -I -m reference_design --help > "$vcc_output/logs/adapter_help.txt"
"$vcc_output/official/bin/python" -I -c 'import cell_eval2; assert cell_eval2.__version__ == "0.16.0"; print(cell_eval2.__version__)' \
    > "$vcc_output/logs/official_import.txt" 2>&1

"$vcc_output/py310/bin/python" -I - "$vcc_output" "$vcc_repo" "$vcc_commit" <<'PY'
import hashlib
from importlib.metadata import version
from importlib.util import find_spec
import json
from pathlib import Path
import platform
import subprocess
import sys

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

output, repository, commit = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
with (repository / "pyproject.toml").open("rb") as stream:
    project = tomllib.load(stream)["project"]
adapter_version = version(project["name"])
expected_version = project["version"]
if adapter_version != expected_version:
    raise RuntimeError(f"Expected adapter {expected_version}; found {adapter_version}")
adapter_spec = find_spec("reference_design")
adapter_source = Path(adapter_spec.origin).resolve() if adapter_spec and adapter_spec.origin else None
expected_source = (repository / "src/reference_design/__init__.py").resolve()
if adapter_source != expected_source:
    raise RuntimeError(f"Expected adapter source {expected_source}; found {adapter_source}")
official = json.loads(subprocess.check_output([
    str(output / "official/bin/python"), "-I", "-c",
    'import json,platform; from importlib.metadata import version; '
    'print(json.dumps({"python":platform.python_version(),"cell_eval2":version("cell-eval2")}))',
], text=True))
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
record = {
    "status": "COMPLETE_RUNTIME_SETUP",
    "adapter": {"version": adapter_version, "python": platform.python_version(),
                "executable": str(output / "py310/bin/python"), "editable_source": str(repository)},
    "official": {**official, "executable": str(output / "official/bin/python"),
                 "source_checkout": str(output / "official_source"), "commit": commit},
    "platform": {"system": platform.system(), "machine": platform.machine(),
                 "libc": platform.libc_ver()},
    "dependency_checks": "BOTH_PIP_CHECKS_PASS",
    "scientific_tests_run_by_setup": False,
    "requirements_sha256": {
        name: digest(repository / "requirements" / name) for name in (
            "vcc-build-py310.txt", "vcc-build-py312.txt", "vcc-py310-linux.txt", "vcc-py312-linux.txt",
            "unified-prepare-py310.txt")},
    "adapter_pyproject_sha256": digest(repository / "pyproject.toml"),
    "setup_script_sha256": digest(repository / "scripts/setup_vcc_runtime.sh"),
    "environment_sha256": {name: digest(output / name) for name in (
        "environment_py310.txt", "environment_official.txt")},
}
temporary = output / "SETUP_COMPLETE.json.tmp"
temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(output / "SETUP_COMPLETE.json")
PY

trap - ERR
printf 'Runtime setup complete. Receipt: %s/SETUP_COMPLETE.json\n' "$vcc_output"
printf 'Adapter: %s/py310/bin/reference-design\nOfficial Python: %s/official/bin/python\n' "$vcc_output" "$vcc_output"
