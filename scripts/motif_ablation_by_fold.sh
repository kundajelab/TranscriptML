#!/bin/bash
#SBATCH --partition=akundaje
#SBATCH --job-name=tml_motif_ablate
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH -C GPU_MEM:48GB
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/sherlock_config.sh"
setup_transcriptml_env

FOLD="${SLURM_ARRAY_TASK_ID}"
CHECKPOINT="${CV_ROOT}/fold${FOLD}/model/best.pt"

for spec in "${MOTIF_ABLATION_SPECS[@]}"; do
  IFS="|" read -r LABEL MOTIF <<< "${spec}"
  transcriptml motif-ablation \
    "${CHECKPOINT}" \
    "${INTERPRET_DATASET_DIR}" \
    "${INTERPRET_ROOT}/motif_ablation/${LABEL}/fold${FOLD}" \
    --motif "${MOTIF}" \
    --n-scrambles "${N_SCRAMBLES}" \
    --strategy "${MOTIF_STRATEGY}" \
    --seed "${MOTIF_SEED}" \
    --device "${DEVICE}" \
    --batch-size "${PRED_BATCH_SIZE}"
done
