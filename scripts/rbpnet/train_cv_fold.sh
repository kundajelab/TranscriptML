#!/bin/bash
#SBATCH --partition=akundaje
#SBATCH --job-name=tml_rbp_cv
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH -C GPU_MEM:48GB
#SBATCH --output=slurm_output/%x_%A_%a.out
#SBATCH --error=slurm_output/%x_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  if [[ -f "${SLURM_SUBMIT_DIR}/scripts/rbpnet/rbpnet_config.sh" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/scripts/rbpnet"
  elif [[ -f "${SLURM_SUBMIT_DIR}/rbpnet_config.sh" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
  fi
fi
source "${SCRIPT_DIR}/rbpnet_config.sh"
setup_transcriptml_env

FOLD="${SLURM_ARRAY_TASK_ID:-${1:-}}"
if [[ -z "${FOLD}" ]]; then
  echo "Provide a fold argument or run as a Slurm array job." >&2
  exit 1
fi
if [[ ! "${FOLD}" =~ ^[0-9]+$ || "${FOLD}" -ge "${N_FOLDS}" ]]; then
  echo "Fold must be an integer in [0, $((N_FOLDS - 1))]; got ${FOLD}." >&2
  exit 1
fi
if [[ ! -f "${CV_PLAN}" ]]; then
  echo "Missing chromosome CV plan ${CV_PLAN}; run create_chromosome_cv_plan.sh first." >&2
  exit 1
fi
if [[ ! -f "${BUNDLE_DIR}/metadata.json" ]]; then
  echo "Missing RBPNet bundle metadata at ${BUNDLE_DIR}." >&2
  exit 1
fi

FOLD_DIR="${CV_ROOT}/fold${FOLD}"
transcriptml train "${BASE_TRAIN_CONFIG}" \
  --cv-plan "${CV_PLAN}" \
  --fold "${FOLD}" \
  --dataset "${BUNDLE_DIR}" \
  --output-dir "${FOLD_DIR}/model"
