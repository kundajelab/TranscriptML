#!/bin/bash
#SBATCH --job-name=tml_sif_smoke
#SBATCH --partition=akundaje,gpu
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=tml_sif_smoke_%j.out
#SBATCH --error=tml_sif_smoke_%j.err

# Smoke test the TranscriptML Apptainer image on a Sherlock GPU node and run pytest.
# Usage:
#   sbatch containers/smoke_test.sh
#   APPTAINER_IMAGE=/path/to/image.sif sbatch containers/smoke_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/src/transcriptml" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi

APPTAINER_IMAGE="${APPTAINER_IMAGE:-${OAK}/isvock/containers/transcriptml-deps_v1.sif}"
RUN_SCRATCH="${L_SCRATCH_JOB:-${TMPDIR:-/tmp}/tml_smoke_${SLURM_JOB_ID:-$$}}"
mkdir -p "${RUN_SCRATCH}/cache/matplotlib" "${RUN_SCRATCH}/cache/numba" "${RUN_SCRATCH}/cache/torch" "${RUN_SCRATCH}/cache/xdg" "${RUN_SCRATCH}/tmp"

t0="$(date +%s)"
STAGED_SIF="${RUN_SCRATCH}/image.sif"
cp -f "${APPTAINER_IMAGE}" "${STAGED_SIF}"
t1="$(date +%s)"
echo "[smoke] Staged ${APPTAINER_IMAGE} -> ${STAGED_SIF} in $((t1 - t0))s"

# Stage TranscriptML src + tests to local scratch
mkdir -p "${RUN_SCRATCH}/repo"
tar -C "${REPO_ROOT}" --exclude='__pycache__' --exclude='*.pyc' -cf - src tests pyproject.toml | tar -C "${RUN_SCRATCH}/repo" -xf -
t2="$(date +%s)"
echo "[smoke] Staged repo src+tests in $((t2 - t1))s"

export APPTAINERENV_PYTHONPATH="${RUN_SCRATCH}/repo/src"
export APPTAINERENV_TMPDIR="${RUN_SCRATCH}/tmp"
export APPTAINERENV_MPLCONFIGDIR="${RUN_SCRATCH}/cache/matplotlib"
export APPTAINERENV_NUMBA_CACHE_DIR="${RUN_SCRATCH}/cache/numba"
export APPTAINERENV_TORCH_HOME="${RUN_SCRATCH}/cache/torch"
export APPTAINERENV_XDG_CACHE_HOME="${RUN_SCRATCH}/cache/xdg"
export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
export APPTAINERENV_PYTHONFAULTHANDLER=1
export APPTAINERENV_OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

apptainer exec --nv --cleanenv --bind "${RUN_SCRATCH}:${RUN_SCRATCH}" "${STAGED_SIF}" python - <<'PY'
import os, sys, time
t0 = time.perf_counter()
import torch
t1 = time.perf_counter()
print(f"Python: {sys.version.split()[0]}")
print(f"torch: {torch.__version__} (cuda={torch.version.cuda}, available={torch.cuda.is_available()})")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    x = torch.randn(64, 64, device="cuda")
    y = (x @ x.T).sum().item()
    print(f"CUDA matmul ok: {y:.3f}")
print(f"import torch elapsed: {t1 - t0:.2f}s")
import tangermeme, numba, pysam, pyarrow, polars, h5py, scipy, matplotlib, seaborn
print("All core + optional libraries imported successfully.")
print("MPLCONFIGDIR:", os.environ.get("MPLCONFIGDIR"))
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
PY

echo "[smoke] Checking CLI help startup time..."
time apptainer exec --cleanenv --bind "${RUN_SCRATCH}:${RUN_SCRATCH}" "${STAGED_SIF}" python -m transcriptml.cli.main --help >/dev/null

echo "[smoke] Running pytest suite..."
apptainer exec --cleanenv --bind "${RUN_SCRATCH}:${RUN_SCRATCH}" "${STAGED_SIF}" \
  pytest -q "${RUN_SCRATCH}/repo/tests"
