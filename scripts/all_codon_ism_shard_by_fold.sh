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
parse_workflow_args "$@"
set -- "${WORKFLOW_POSITIONAL_ARGS[@]+"${WORKFLOW_POSITIONAL_ARGS[@]}"}"
setup_transcriptml_env

CODON_ISM_SHARDS_PER_FOLD="${CODON_ISM_SHARDS_PER_FOLD:-10}"
if ! [[ "${CODON_ISM_SHARDS_PER_FOLD}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CODON_ISM_SHARDS_PER_FOLD must be a positive integer" >&2
  exit 2
fi

if [[ -n "${FOLD:-}" ]]; then
  SHARD="${SLURM_ARRAY_TASK_ID:-${SHARD:-${2:-0}}}"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  TASK_ID="${SLURM_ARRAY_TASK_ID}"
  FOLD=$((TASK_ID / CODON_ISM_SHARDS_PER_FOLD))
  SHARD=$((TASK_ID % CODON_ISM_SHARDS_PER_FOLD))
else
  FOLD="${1:-}"
  SHARD="${2:-0}"
fi
if [[ -z "${FOLD}" ]]; then
  echo "FOLD must be set, derived from SLURM_ARRAY_TASK_ID, or passed as the first argument" >&2
  exit 2
fi

CHECKPOINT="${CV_ROOT}/fold${FOLD}/model/best.pt"
OUT_DIR="${INTERPRET_ROOT}/all_codon_ism/fold${FOLD}/shard${SHARD}"
STAGED_INTERPRET_DATASET="$(stage_dataset_bundle "${INTERPRET_DATASET_DIR}")"

transcriptml codon-ism \
  --checkpoint "${CHECKPOINT}" \
  --dataset "${STAGED_INTERPRET_DATASET}" \
  --out-dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --batch-size "${PRED_BATCH_SIZE}" \
  --mutation-batch-size "${MUTATION_BATCH_SIZE}" \
  --mutation-policy all \
  --table-format npz \
  --sequence-shard-index "${SHARD}" \
  --sequence-shards "${CODON_ISM_SHARDS_PER_FOLD}"
