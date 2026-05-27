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
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  if [[ -f "${SLURM_SUBMIT_DIR}/scripts/sherlock_config.sh" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/scripts"
  elif [[ -f "${SLURM_SUBMIT_DIR}/sherlock_config.sh" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
  fi
fi
source "${SCRIPT_DIR}/sherlock_config.sh"
setup_transcriptml_env

FOLD="${SLURM_ARRAY_TASK_ID}"
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
    --region "${MOTIF_REGION}"
    --max-pairs "${MAX_EPISTASIS_PAIRS}"
    --device "${DEVICE}"
    --batch-size "${PRED_BATCH_SIZE}"
  )
  if [[ -n "${MOTIF_SPEC_2}" ]]; then
    args+=(--motif2 "${MOTIF_SPEC_2}")
  fi
  transcriptml "${args[@]}"
done
