#!/bin/bash
#SBATCH --partition=akundaje
#SBATCH --job-name=tml_ism
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -C GPU_MEM:48GB
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/sherlock_config.sh"
setup_transcriptml_env

FOLD="${SLURM_ARRAY_TASK_ID}"
CHECKPOINT="${CV_ROOT}/fold${FOLD}/model/best.pt"
OUT_DIR="${INTERPRET_ROOT}/ism/fold${FOLD}"

transcriptml ism \
  "${CHECKPOINT}" \
  "${INTERPRET_DATASET_DIR}" \
  "${OUT_DIR}" \
  --device "${DEVICE}" \
  --batch-size "${PRED_BATCH_SIZE}" \
  --mutation-batch-size "${MUTATION_BATCH_SIZE}"
