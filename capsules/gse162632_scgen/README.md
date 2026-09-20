# Focal scGen reproducibility capsule

Reproduce the PCA-64 ridge versus scGen comparison in GSE162632 on CPU.
The replay builds perturbation effects from saved absolute predictions and
reference means, scores every root–task row, and applies the paper's task,
rotation and root aggregation.

After installing the CPU environment, run this command from the repository root:

```bash
python scripts/focal_scgen.py reproduce-focal
```

The command checks inputs against `replay/manifest.json` and recalculates the
five realization-specific contrasts, including the primary analysis's
eight-root reversal. It uses bundled data and takes less than a minute on CPU.

## Optional retraining

Retraining requires a CUDA GPU and the separate prepared-data archive.
Build the pinned Apptainer image from the environment directory;
the definition resolves its two local inputs relative to that directory:

```bash
cd capsules/gse162632_scgen/environment
apptainer build scgen-cu117.sif container.def
cd ../../..
```

After extracting the prepared-data archive, submit one realization from the
repository root:

```bash
export CAPSULE_IMAGE="$PWD/capsules/gse162632_scgen/environment/scgen-cu117.sif"
export PREPARED_ASSETS=/absolute/path/to/focal-scgen-prepared-training-v1.6.1
export OUTPUT_DIR="$PWD/outputs/q1"
sbatch capsules/gse162632_scgen/slurm/train_one.sbatch
```

The report records runtime, GPU model, hyperparameters and recalculated
contrasts. The acceptance check
requires the expected direction pattern and a 0.01 absolute tolerance for
each focal point estimate. Checkpoint bytes are not compared across GPU
models. A failed acceptance check returns a nonzero status and keeps its
`report.json` for diagnosis.

To submit all five realizations as a Slurm array, keep `CAPSULE_IMAGE` and
`PREPARED_ASSETS` from the preceding example and run:

```bash
export OUTPUT_ROOT="$PWD/outputs/all"
sbatch capsules/gse162632_scgen/slurm/train_all.sbatch
```

The Slurm examples request one GPU, eight CPUs, 64 GB of memory and 15 minutes
per realization; cluster-specific account and partition settings may need to
be changed. If unprivileged builds are not enabled on the cluster, build the
image on another Apptainer host and copy the `.sif` file to the cluster before
submission.

## Contents

```text
replay/                  Pseudonymized arrays and axes used by the CPU replay
expected/                Reference values used to verify the five focal contrasts
configs/                 Training seeds and hyperparameters
environment/             GPU dependency snapshot and Apptainer definition
slurm/                   One-realization and five-realization job examples
```

The replay and training assets use release-local root, cell and feature
pseudonyms. They do not contain the production label map, selection salts, raw
count matrices or model checkpoints. The replay and training bundles share the
same fixed 2,000-feature order. The underlying study is GSE162632; the
associated analysis deposit is Zenodo record 4273999.
