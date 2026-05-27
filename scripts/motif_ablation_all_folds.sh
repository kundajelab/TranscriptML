#!/bin/bash
#SBATCH --partition=akundaje
#SBATCH --job-name=tml_motif_ablate_all
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH -C GPU_MEM:48GB
#SBATCH --output=slurm_output/%x_%j.out
#SBATCH --error=slurm_output/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  if [[ -f "${SLURM_SUBMIT_DIR}/scripts/sherlock_config.sh" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/scripts"
  elif [[ -f "${SLURM_SUBMIT_DIR}/sherlock_config.sh" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
  fi
fi
source "${SCRIPT_DIR}/sherlock_config.sh"
setup_transcriptml_env

for FOLD in $(seq 0 $((N_FOLDS - 1))); do
  CHECKPOINT="${CV_ROOT}/fold${FOLD}/model/best.pt"
  for spec in "${MOTIF_ABLATION_SPECS[@]}"; do
    parse_motif_ablation_spec "${spec}"
    transcriptml motif-ablation \
      "${CHECKPOINT}" \
      "${INTERPRET_DATASET_DIR}" \
      "${INTERPRET_ROOT}/motif_ablation/${MOTIF_SPEC_LABEL}/fold${FOLD}" \
      --motif "${MOTIF_SPEC_1}" \
      --n-scrambles "${N_SCRAMBLES}" \
      --strategy "${MOTIF_STRATEGY}" \
      --seed "${MOTIF_SEED}" \
      --region "${MOTIF_REGION}" \
      --device "${DEVICE}" \
      --batch-size "${PRED_BATCH_SIZE}"
  done
done
