#!/bin/bash
#SBATCH --partition=akundaje
#SBATCH --job-name=tml_mpra_ism
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -C GPU_MEM:48GB
#SBATCH --output=slurm_output/%x_%A_%a.out
#SBATCH --error=slurm_output/%x_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  if [[ -f "${SLURM_SUBMIT_DIR}/scripts/mpra/mpra_config.sh" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/scripts/mpra"
  elif [[ -f "${SLURM_SUBMIT_DIR}/mpra_config.sh" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
  fi
fi
source "${SCRIPT_DIR}/mpra_config.sh"
setup_transcriptml_env

FOLD="${SLURM_ARRAY_TASK_ID}"
CHECKPOINT="${CV_ROOT}/fold${FOLD}/model/best.pt"
OUT_DIR="${INTERPRET_ROOT}/ism/fold${FOLD}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Expected trained fold checkpoint at ${CHECKPOINT}." >&2
  echo "Run submit_train_eval_cv.sh first, or set CV_ROOT in ${SCRIPT_DIR}/mpra_config.sh." >&2
  exit 1
fi
if [[ ! -f "${INTERPRET_DATASET_DIR}/X.npy" ]]; then
  echo "Expected MPRA dataset bundle at ${INTERPRET_DATASET_DIR}." >&2
  echo "Set INTERPRET_DATASET_DIR in ${SCRIPT_DIR}/mpra_config.sh if you want to interpret a different bundle." >&2
  exit 1
fi

transcriptml ism \
  --checkpoint "${CHECKPOINT}" \
  --dataset "${INTERPRET_DATASET_DIR}" \
  --out-dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --batch-size "${PRED_BATCH_SIZE}" \
  --mutation-batch-size "${MUTATION_BATCH_SIZE}"
