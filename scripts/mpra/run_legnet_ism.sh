#!/bin/bash
#SBATCH --partition=akundaje
#SBATCH --job-name=tml_legnet_ism
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -C GPU_MEM:48GB
#SBATCH --output=slurm_output/%x_%j.out
#SBATCH --error=slurm_output/%x_%j.err

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

if [[ ! -f "${LEGNET_CHECKPOINT}" ]]; then
  echo "Expected trained LegNet checkpoint at ${LEGNET_CHECKPOINT}." >&2
  echo "Run train_legnet.sh first, or set LEGNET_CHECKPOINT in ${SCRIPT_DIR}/mpra_config.sh." >&2
  exit 1
fi
if [[ ! -f "${INTERPRET_DATASET_DIR}/X.npy" ]]; then
  echo "Expected MPRA dataset bundle at ${INTERPRET_DATASET_DIR}." >&2
  echo "Set INTERPRET_DATASET_DIR in ${SCRIPT_DIR}/mpra_config.sh if you want to interpret a different bundle." >&2
  exit 1
fi

transcriptml ism \
  --checkpoint "${LEGNET_CHECKPOINT}" \
  --dataset "${INTERPRET_DATASET_DIR}" \
  --out-dir "${ISM_OUT_DIR}" \
  --device "${DEVICE}" \
  --batch-size "${PRED_BATCH_SIZE}" \
  --mutation-batch-size "${MUTATION_BATCH_SIZE}"
