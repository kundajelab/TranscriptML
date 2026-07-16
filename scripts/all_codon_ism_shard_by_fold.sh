#!/bin/bash
#SBATCH --partition=akundaje,gpu
#SBATCH --job-name=tml_all_codon_ism_shard
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH -C GPU_MEM:48GB
#SBATCH --output=slurm_output/%x_%A_%a.out
#SBATCH --error=slurm_output/%x_%A_%a.err

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

FOLD="${FOLD:-${1:-}}"
if [[ -z "${FOLD}" ]]; then
  echo "FOLD must be exported by the submit script or passed as the first argument" >&2
  exit 2
fi

CODON_ISM_SHARDS_PER_FOLD="${CODON_ISM_SHARDS_PER_FOLD:-10}"
SHARD="${SLURM_ARRAY_TASK_ID:-${2:-0}}"
CHECKPOINT="${CV_ROOT}/fold${FOLD}/model/best.pt"
OUT_DIR="${INTERPRET_ROOT}/all_codon_ism/fold${FOLD}/shard${SHARD}"

transcriptml codon-ism \
  --checkpoint "${CHECKPOINT}" \
  --dataset "${INTERPRET_DATASET_DIR}" \
  --out-dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --batch-size "${PRED_BATCH_SIZE}" \
  --mutation-batch-size "${MUTATION_BATCH_SIZE}" \
  --mutation-policy all \
  --table-format npz \
  --sequence-shard-index "${SHARD}" \
  --sequence-shards "${CODON_ISM_SHARDS_PER_FOLD}"
