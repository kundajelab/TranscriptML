#!/bin/bash
#SBATCH --partition=akundaje
#SBATCH --job-name=tml_motif_epi
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH -C GPU_MEM:48GB
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/sherlock_config.sh"
setup_transcriptml_env

FOLD="${SLURM_ARRAY_TASK_ID}"
CHECKPOINT="${CV_ROOT}/fold${FOLD}/model/best.pt"

for spec in "${MOTIF_EPISTASIS_SPECS[@]}"; do
  IFS="|" read -r LABEL MOTIF MOTIF2 <<< "${spec}"
  args=(
    epistasis
    "${CHECKPOINT}"
    "${INTERPRET_DATASET_DIR}"
    "${INTERPRET_ROOT}/motif_epistasis/${LABEL}/fold${FOLD}"
    --motif "${MOTIF}"
    --n-scrambles "${N_SCRAMBLES}"
    --strategy "${MOTIF_STRATEGY}"
    --seed "${MOTIF_SEED}"
    --max-pairs "${MAX_EPISTASIS_PAIRS}"
    --device "${DEVICE}"
    --batch-size "${PRED_BATCH_SIZE}"
  )
  if [[ -n "${MOTIF2}" ]]; then
    args+=(--motif2 "${MOTIF2}")
  fi
  transcriptml "${args[@]}"
done
