#!/bin/bash
#SBATCH --partition=akundaje
#SBATCH --job-name=tml_legnet_train
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

if [[ ! -f "${DATASET_DIR}/X.npy" ]]; then
  echo "Expected MPRA dataset bundle at ${DATASET_DIR}." >&2
  echo "Run build_legnet_input.sh first, or set DATASET_DIR in ${SCRIPT_DIR}/mpra_config.sh." >&2
  exit 1
fi

mkdir -p "${TRAIN_OUTPUT_ROOT}" "${MODEL_DIR}" "${EVAL_DIR}"

writer_args=(
  --dataset "${DATASET_DIR}"
  --base-config "${BASE_TRAIN_CONFIG}"
  --output-dir "${MODEL_DIR}"
  --config-path "${GENERATED_TRAIN_CONFIG}"
)
if [[ -n "${TRAIN_SEED}" ]]; then
  writer_args+=(--seed "${TRAIN_SEED}")
fi

CONFIG_PATH="$(python "${SCRIPT_DIR}/write_legnet_train_config.py" "${writer_args[@]}")"

transcriptml train "${CONFIG_PATH}"

run_eval=0
if [[ "${RUN_SPLIT_EVALUATION}" == "1" ]]; then
  run_eval=1
elif [[ "${RUN_SPLIT_EVALUATION}" == "auto" && -f "${DATASET_DIR}/splits.json" ]]; then
  run_eval=1
fi

if [[ "${run_eval}" == "1" ]]; then
  transcriptml evaluate \
    --checkpoint "${MODEL_DIR}/best.pt" \
    --dataset "${DATASET_DIR}" \
    --out-csv "${EVAL_DIR}/${EVAL_SPLIT}_predictions.csv" \
    --split "${EVAL_SPLIT}" \
    --batch-size "${PRED_BATCH_SIZE}" \
    --device "${DEVICE}"
else
  echo "Skipping standalone evaluate step because ${DATASET_DIR}/splits.json is absent."
  echo "Training still writes random-split test predictions to ${MODEL_DIR}/test_predictions.csv when test examples exist."
fi
