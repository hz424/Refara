# Install the VCC audit runtimes

Refara and the official scorer use separate Python environments. Refara's
Python 3.10 environment includes AnnData preparation, MSE scoring, comparison
reporting and the VCC adapter:

| Component | Version | Recorded Python |
|---|---|---|
| Current toolkit | `reference-cell-benchmark-design 1.10.0` | CPython 3.10.x |
| Official CPU scorer | `cell-eval2 0.16.0` ([source revision](https://github.com/ArcInstitute/cell-eval2/tree/5e64833518a6603a0301cbe28185d49c30f4a986)) | CPython 3.12.14 |

The adapter package requires Python 3.10. The pinned cell-eval2 version requires
Python 3.11 or later; this installation route uses Python 3.12 to match the
recorded environment.

You need Bash, Git, both Python executables with `venv` support, and access to
GitHub and the Python package index. The CPU route uses the `scanpy` backend
for differential expression and installs no GPU extras.

## Create new environments

Run from this repository. Supply absolute executable paths and a new output directory:

```bash
bash scripts/setup_vcc_runtime.sh \
  --python310 /path/to/python3.10 \
  --python312 /path/to/python3.12 \
  --output /path/to/new-vcc-runtime
```

The script creates:

- `py310/`: Refara's editable installation, pinned dependencies and AnnData preparation support;
- `official/`: the official scorer and its pinned CPU dependencies;
- `official_source/`: the official Git checkout at the recorded commit;
- `environment_py310.txt` and `environment_official.txt`: actual installed versions, including build tools;
- `logs/`: installation, dependency-check and import logs;
- `SETUP_COMPLETE.json`: interpreter, platform, source revision and dependency files.

If setup fails, inspect the logs and `SETUP_FAILED.txt`, then retry in a new directory. Keep this repository in place because the adapter installation is editable.

The script verifies both dependency sets with `pip check`, checks the adapter CLI import and official version, and saves the results. Run scientific scoring separately.

## Install through Slurm

For a site using Slurm, put the resource requests in a job script. Replace the account, partition and paths with local values:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=reference-vcc-setup
#SBATCH --account=YOUR_PROJECT
#SBATCH --partition=YOUR_CPU_PARTITION
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=reference-vcc-setup-%j.out
#SBATCH --error=reference-vcc-setup-%j.err
set -euo pipefail
cd /path/to/Refara
bash scripts/setup_vcc_runtime.sh \
  --python310 /path/to/python3.10 \
  --python312 /path/to/python3.12 \
  --output /path/to/new-vcc-runtime
```

Submit with `sbatch setup_vcc.sbatch`. Installation runs as a CPU job and sets
numerical-library thread limits from `SLURM_CPUS_PER_TASK`. The recorded
installation used two CPUs and 12 GB. Choose resources for subsequent scoring
jobs according to the size of your inputs.

## Connect the installed runtimes

From this repository, run the public Adamson example in a CPU job with at least two CPUs, using the environments and checkout created above:

```bash
/path/to/new-vcc-runtime/py310/bin/python examples/vcc_audit/run.py \
  --official-python /path/to/new-vcc-runtime/official/bin/python \
  --upstream /path/to/new-vcc-runtime/official_source \
  --output /path/to/new-vcc-example
```

The runner defaults to `adamson10x005`, downloads its pinned public H5AD and checks its SHA256 before preparing inputs. To reuse a downloaded copy, add `--source /path/to/AdamsonWeissman2016_GSM2406677_10X005.h5ad`. Add `--prepare-only` to create the frozen inputs and `manifest.json` without scoring. The [audit guide](VCC_AUDIT.md#public-example-and-verification) gives the source, selection rules and independent parity command. All example outputs use a new directory.

For an audit manifest describing your own inputs, use the same installed runtime paths.
Set the manifest's `upstream.path` to `/path/to/new-vcc-runtime/official_source` and keep its pinned `upstream.commit`. Then:

```bash
/path/to/new-vcc-runtime/py310/bin/reference-design vcc plan \
  /path/to/manifest.json --output /path/to/new-plan

/path/to/new-vcc-runtime/py310/bin/reference-design vcc run \
  /path/to/manifest.json \
  --python /path/to/new-vcc-runtime/official/bin/python \
  --output /path/to/new-audit
```

`run` makes its own plan, so its output directory must differ from a prior planning directory. Execute numerical scoring in an appropriate CPU allocation and set manifest `runtime.num_threads` no higher than the requested CPUs. Input format, calibration, unavailable scores, public examples and interpretation are described in [VCC_AUDIT.md](VCC_AUDIT.md).

## Check the installed code

Run these two suites from this repository in the CPU allocation. The first checks the complete Python 3.10 suite, including preparation, protocol, organizer, MSE and VCC contracts; the second checks the H5AD worker in the official runtime:

```bash
export REFERENCE_DESIGN_OFFICIAL_SOURCE=/path/to/new-vcc-runtime/official_source

/path/to/new-vcc-runtime/py310/bin/python -m pytest -q

/path/to/new-vcc-runtime/official/bin/python -m pytest -q \
  tests_official/test_vcc_worker.py
```

Keep the test results and scoring outputs together. When every score is
available, the summary reports `COMPLETE_DESCRIPTIVE_VCC_AUDIT`.

If the official scorer reports a recognized baseline-scale calibration rejection,
the audit retains all model and metric rows and scores the valid allocations. A completed run
with these recognized rejections reports `UNAVAILABLE_OFFICIAL_SCORES`, execution
status `COMPLETE_WITH_UNAVAILABLE_ANALYSES`, and exit code 2. Missing or failed
workers remain execution failures. The [audit guide](VCC_AUDIT.md) explains
how to distinguish these outcomes.

## Pinned dependencies

The runtime lists [vcc-py310-linux.txt](../requirements/vcc-py310-linux.txt) and [vcc-py312-linux.txt](../requirements/vcc-py312-linux.txt) preserve package versions from the successful Linux x86-64 installations recorded on 13 September 2026. Local editable paths and the official Git dependency are excluded from those lists: the setup script installs this repository explicitly and installs the separate official checkout at the pinned commit. The combined installer adds [unified-prepare-py310.txt](../requirements/unified-prepare-py310.txt), containing AnnData 0.11.4, h5py 3.16.0, array-api-compat 1.15.0 and natsort 8.4.0, to the Python 3.10 VCC pins. This retains their NumPy 2.2.6 dependency set. Existing [locked.txt](../requirements/locked.txt) and [prepare-locked.txt](../requirements/prepare-locked.txt) retain the historical NumPy 1.26.4 replay environment; install those in a separate environment.

[vcc-build-py310.txt](../requirements/vcc-build-py310.txt) pins the adapter's declared build tools. [vcc-build-py312.txt](../requirements/vcc-build-py312.txt) pins Hatchling 1.32.0, the backend recorded in the successfully built official wheel, and its build dependencies. The dependency names follow the [Hatchling release metadata](https://pypi.org/pypi/hatchling/1.32.0/json). Source installation uses `--no-build-isolation --no-deps` after these explicit dependencies have been installed.

These files pin package versions. The recorded VCC environment is Linux x86-64 with CPython 3.10.21 and 3.12.14; Python patch versions, platform wheels, libc, BLAS and CPU architecture can also affect installation and numerical results. The setup output records the actual platform and packages, and each scoring run records its scorer version and configuration. Use those records when checking another platform. The historical figure-replay guide specifies its earlier Python 3.10.17 environment.

<a id="inherited-vcc-validation"></a>

## Earlier validation

The earlier combined-installer validation used toolkit `1.8.20.dev2` with
CPython 3.10.21. The [historical VCC checks](UNIFIED_INTEGRATION.md#historical-vcc-checks)
record installation tests and public-data scoring for `1.8.20.dev1`. They include
calibration refusals and independent comparisons with the pinned official scorer.
