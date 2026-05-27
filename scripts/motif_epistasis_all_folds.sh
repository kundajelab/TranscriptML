#!/bin/bash
#SBATCH --partition=akundaje
#SBATCH --job-name=tml_motif_epi_all
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH --time=16:00:00
#SBATCH -C GPU_MEM:48GB
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/sherlock_config.sh"
setup_transcriptml_env

for FOLD in $(seq 0 $((N_FOLDS - 1))); do
  CHECKPOINT="${CV_ROOT}/fold${FOLD}/model/best.pt"
  for spec in "${MOTIF_EPISTASIS_SPECS[@]}"; do
    parse_motif_epistasis_spec "${spec}"
    args=(
      epistasis
      "${CHECKPOINT}"
      "${INTERPRET_DATASET_DIR}"
      "${INTERPRET_ROOT}/motif_epistasis/${MOTIF_SPEC_LABEL}/fold${FOLD}"
      --motif "${MOTIF_SPEC_1}"
      --n-scrambles "${N_SCRAMBLES}"
      --strategy "${MOTIF_STRATEGY}"
      --seed "${MOTIF_SEED}"
      --max-pairs "${MAX_EPISTASIS_PAIRS}"
      --device "${DEVICE}"
      --batch-size "${PRED_BATCH_SIZE}"
    )
    if [[ -n "${MOTIF_SPEC_2}" ]]; then
      args+=(--motif2 "${MOTIF_SPEC_2}")
    fi
    transcriptml "${args[@]}"
  done
done
