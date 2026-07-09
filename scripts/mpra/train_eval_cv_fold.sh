#!/bin/bash
#SBATCH --partition=akundaje
#SBATCH --job-name=tml_mpra_cv
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

if [[ ! -f "${DATASET_DIR}/X.npy" ]]; then
  echo "Expected MPRA dataset bundle at ${DATASET_DIR}." >&2
  echo "Run build_legnet_input.sh first, or set DATASET_DIR in ${SCRIPT_DIR}/mpra_config.sh." >&2
  exit 1
fi

CONFIG_PATH="$(
  python "${SCRIPT_DIR}/write_cv_fold_artifacts.py" \
    --dataset "${DATASET_DIR}" \
    --base-config "${BASE_TRAIN_CONFIG}" \
    --cv-root "${CV_ROOT}" \
    --fold "${FOLD}" \
    --n-folds "${N_FOLDS}" \
    --seed "${CV_SEED}" \
    --val-offset "${CV_VAL_OFFSET}"
)"

transcriptml train "${CONFIG_PATH}"

FOLD_DIR="${CV_ROOT}/fold${FOLD}"
mkdir -p "${FOLD_DIR}/eval"

transcriptml evaluate \
  "${FOLD_DIR}/model/best.pt" \
  "${FOLD_DIR}/dataset" \
  "${FOLD_DIR}/eval/${EVAL_SPLIT}_predictions.csv" \
  --split "${EVAL_SPLIT}" \
  --batch-size "${PRED_BATCH_SIZE}" \
  --device "${DEVICE}"
