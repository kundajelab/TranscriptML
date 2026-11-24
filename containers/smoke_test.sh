#!/bin/bash
#SBATCH --job-name=tml_sif_smoke
#SBATCH --partition=akundaje,gpu
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=tml_sif_smoke_%j.out
#SBATCH --error=tml_sif_smoke_%j.err

# GPU smoke test for the TranscriptML Apptainer image on Sherlock:
# - stages the .sif and repo (src, tests, scripts, pyproject.toml) to $L_SCRATCH_JOB
# - asserts CUDA availability + GPU matmul inside `apptainer exec --nv --cleanenv`
# - verifies Sherlock site bind mounts ($SCRATCH, $OAK, $L_SCRATCH_JOB)
# - verifies CLI startup time and runs the full pytest suite with --nv
#
# Usage:
#   cd /oak/stanford/groups/akundaje/isvock/software/TranscriptML
#   sbatch containers/smoke_test.sh
#   APPTAINER_IMAGE=/path/to/image.sif sbatch containers/smoke_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${REPO_ROOT:-}" && -d "${REPO_ROOT}/src/transcriptml" ]]; then
  :
elif [[ -d "${PWD}/src/transcriptml" ]]; then
  REPO_ROOT="${PWD}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/src/transcriptml" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
elif [[ -d "${SCRIPT_DIR}/../src/transcriptml" ]]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
elif [[ -n "${OAK:-}" && -d "${OAK}/isvock/software/TranscriptML/src/transcriptml" ]]; then
  REPO_ROOT="${OAK}/isvock/software/TranscriptML"
else
  echo "Could not locate TranscriptML repository root (set REPO_ROOT)." >&2
  exit 1
fi

APPTAINER_IMAGE="${APPTAINER_IMAGE:-${OAK}/isvock/containers/transcriptml-deps_v1.sif}"
RUN_SCRATCH="${L_SCRATCH_JOB:-${TMPDIR:-/tmp}/tml_smoke_${SLURM_JOB_ID:-$$}}"
mkdir -p \
  "${RUN_SCRATCH}/cache/matplotlib" \
  "${RUN_SCRATCH}/cache/numba" \
  "${RUN_SCRATCH}/cache/torch" \
  "${RUN_SCRATCH}/cache/xdg" \
  "${RUN_SCRATCH}/tmp"

echo "[smoke] Host: $(hostname)"
echo "[smoke] Repo: ${REPO_ROOT}"
echo "[smoke] Image: ${APPTAINER_IMAGE}"
echo "[smoke] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"

t0="$(date +%s)"
STAGED_SIF="${RUN_SCRATCH}/image.sif"
cp -f "${APPTAINER_IMAGE}" "${STAGED_SIF}.tmp.$$"
mv -f "${STAGED_SIF}.tmp.$$" "${STAGED_SIF}"
t1="$(date +%s)"
echo "[smoke] Staged ${APPTAINER_IMAGE} -> ${STAGED_SIF} in $((t1 - t0))s"

# Stage TranscriptML src, tests, scripts, and pyproject.toml to local scratch
mkdir -p "${RUN_SCRATCH}/repo"
tar -C "${REPO_ROOT}" \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='*.egg-info' \
  -cf - src tests scripts pyproject.toml | tar -C "${RUN_SCRATCH}/repo" -xf -
t2="$(date +%s)"
echo "[smoke] Staged repo (src+tests+scripts) in $((t2 - t1))s"

APPTAINER_CMD=(
  apptainer exec --cleanenv --nv
  --bind "${RUN_SCRATCH}:${RUN_SCRATCH}"
  --env "PYTHONPATH=${RUN_SCRATCH}/repo/src"
  --env "TMPDIR=${RUN_SCRATCH}/tmp"
  --env "MPLCONFIGDIR=${RUN_SCRATCH}/cache/matplotlib"
  --env "NUMBA_CACHE_DIR=${RUN_SCRATCH}/cache/numba"
  --env "TORCH_HOME=${RUN_SCRATCH}/cache/torch"
  --env "XDG_CACHE_HOME=${RUN_SCRATCH}/cache/xdg"
  --env "PYTHONNOUSERSITE=1"
  --env "PYTHONDONTWRITEBYTECODE=1"
  --env "PYTHONFAULTHANDLER=1"
  --env "OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}"
)
if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  APPTAINER_CMD+=(--env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}")
fi
if [[ -n "${NVIDIA_VISIBLE_DEVICES+x}" ]]; then
  APPTAINER_CMD+=(--env "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES}")
fi
if [[ -n "${SCRATCH:-}" ]]; then
  APPTAINER_CMD+=(--env "HOST_SCRATCH=${SCRATCH}")
fi
if [[ -n "${OAK:-}" ]]; then
  APPTAINER_CMD+=(--env "HOST_OAK=${OAK}")
fi
if [[ -n "${L_SCRATCH_JOB:-}" ]]; then
  APPTAINER_CMD+=(--env "L_SCRATCH_JOB=${L_SCRATCH_JOB}")
fi

"${APPTAINER_CMD[@]}" "${STAGED_SIF}" python - <<'PY'
import os, sys, time
from pathlib import Path

t0 = time.perf_counter()
import torch
t1 = time.perf_counter()

print(f"Python: {sys.version.split()[0]}")
print(f"torch: {torch.__version__} (cuda={torch.version.cuda}, available={torch.cuda.is_available()})")
assert torch.version.cuda is not None, "Container PyTorch build is missing CUDA!"
assert torch.cuda.is_available(), (
    f"CUDA is NOT available inside Apptainer --nv! "
    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
)
print(f"GPU[0]: {torch.cuda.get_device_name(0)} (count={torch.cuda.device_count()})")
x = torch.randn(64, 64, device="cuda")
y = (x @ x.T).sum().item()
print(f"CUDA matmul ok: {y:.3f}")
print(f"import torch elapsed: {t1 - t0:.2f}s")

import tangermeme, numba, pysam, pyarrow, polars, h5py, scipy, matplotlib, seaborn
print("All core + optional libraries imported successfully.")
print("MPLCONFIGDIR:", os.environ.get("MPLCONFIGDIR"))
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))

# Verify Sherlock bind mounts are accessible inside the container.
for env_key in ("HOST_SCRATCH", "HOST_OAK", "L_SCRATCH_JOB"):
    val = os.environ.get(env_key)
    if val:
        p = Path(val)
        assert p.is_dir(), f"Expected bind-mounted directory {env_key}={val} to be accessible!"
        print(f"Mount ok: {env_key}={val}")
PY

echo "[smoke] Checking CLI help startup time..."
time "${APPTAINER_CMD[@]}" "${STAGED_SIF}" python -m transcriptml.cli.main --help >/dev/null

echo "[smoke] Running full pytest suite inside GPU container (--nv)..."
"${APPTAINER_CMD[@]}" "${STAGED_SIF}" pytest -q "${RUN_SCRATCH}/repo/tests"
