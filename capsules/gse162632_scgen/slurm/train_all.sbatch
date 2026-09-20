#!/bin/bash
#SBATCH --job-name=focal-scgen-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --array=0-4

set -euo pipefail

: "${PREPARED_ASSETS:?Set PREPARED_ASSETS to the extracted training archive}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new parent directory}"

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
realizations=(q1 q2 q3 q4 q5)
realization=${realizations[$SLURM_ARRAY_TASK_ID]}
output="$OUTPUT_ROOT/$realization"
export PYTHONNOUSERSITE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

if [[ -n "${CAPSULE_IMAGE:-}" ]]; then
    [[ -f "$CAPSULE_IMAGE" ]] || { printf 'Container image not found: %s\n' "$CAPSULE_IMAGE" >&2; exit 2; }
    [[ -d "$PREPARED_ASSETS" ]] || { printf 'Prepared-data directory not found: %s\n' "$PREPARED_ASSETS" >&2; exit 2; }
    capsule_image=$(realpath -- "$CAPSULE_IMAGE")
    prepared_assets=$(realpath -- "$PREPARED_ASSETS")
    output_root=$(realpath -m -- "$OUTPUT_ROOT")
    mkdir -p -- "$output_root"
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
    fi
    apptainer exec --cleanenv --nv --no-home --pwd /workspace/repository \
        --bind "$repo_root:/workspace/repository:ro" \
        --bind "$prepared_assets:/workspace/prepared:ro" \
        --bind "$output_root:/workspace/output:rw" \
        "$capsule_image" python \
        /workspace/repository/scripts/focal_scgen.py train-one \
        --realization "$realization" \
        --data /workspace/prepared \
        --output "/workspace/output/$realization"
else
    "${PYTHON:-python}" "$repo_root/scripts/focal_scgen.py" train-one \
        --realization "$realization" \
        --data "$PREPARED_ASSETS" \
        --output "$output"
fi
