#!/bin/bash
#SBATCH --partition=akundaje
#SBATCH --job-name=tml_train_split
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
  if [[ -f "${SLURM_SUBMIT_DIR}/scripts/sherlock_config.sh" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/scripts"
  elif [[ -f "${SLURM_SUBMIT_DIR}/sherlock_config.sh" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
  fi
fi
source "${SCRIPT_DIR}/sherlock_config.sh"
setup_transcriptml_env

mkdir -p "${TRAIN_OUTPUT_ROOT}" "${MODEL_DIR}" "${EVAL_DIR}"

if [[ "${REQUIRE_SPLIT_FILE}" == "1" && ! -f "${DATASET_DIR}/splits.json" ]]; then
  echo "Expected predefined splits at ${DATASET_DIR}/splits.json." >&2
  echo "Set SPLIT_COL to your target-table split column before running build_saluki_gtf.sh." >&2
  echo "Set REQUIRE_SPLIT_FILE=0 only if you intentionally want TranscriptML's config split behavior." >&2
  exit 1
fi

writer_args=(
  --dataset "${DATASET_DIR}"
  --base-config "${BASE_TRAIN_CONFIG}"
  --output-dir "${MODEL_DIR}"
  --config-path "${GENERATED_TRAIN_CONFIG}"
)
if [[ -n "${TRAIN_SEED}" ]]; then
  writer_args+=(--seed "${TRAIN_SEED}")
fi

CONFIG_PATH="$(python "${SCRIPT_DIR}/write_train_config.py" "${writer_args[@]}")"

transcriptml train "${CONFIG_PATH}"

transcriptml evaluate \
  --checkpoint "${MODEL_DIR}/best.pt" \
  --dataset "${DATASET_DIR}" \
  --out-csv "${EVAL_DIR}/${EVAL_SPLIT}_predictions.csv" \
  --split "${EVAL_SPLIT}" \
  --batch-size "${PRED_BATCH_SIZE}" \
  --device "${DEVICE}"
