#!/bin/bash
#SBATCH --partition=akundaje
#SBATCH --job-name=tml_mpra_hps
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -C GPU_MEM:48GB
#SBATCH --output=slurm_output/%x_%A_%a.out
#SBATCH --error=slurm_output/%x_%A_%a.err

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

COMBO_INDEX="${COMBO_INDEX:-${SLURM_ARRAY_TASK_ID:?Set COMBO_INDEX or run through submit_hparam_sweep_cv.sh}}"
COMBO_DIR="$(printf "%s/combo_%04d" "${SWEEP_ROOT}" "${COMBO_INDEX}")"
mkdir -p "${COMBO_DIR}"

for FOLD in $(seq 0 $((N_FOLDS - 1))); do
  FOLD_DIR="${COMBO_DIR}/fold${FOLD}"
  if [[ "${SWEEP_SKIP_COMPLETED}" == "1" \
      && -f "${FOLD_DIR}/model/summary.json" \
      && -f "${FOLD_DIR}/eval/${EVAL_SPLIT}_predictions.summary.json" ]]; then
    echo "Skipping completed combo ${COMBO_INDEX}, fold ${FOLD}"
    continue
  fi

  CONFIG_PATH="$(
    python "${SCRIPT_DIR}/hparam_sweep_utils.py" write-fold-artifacts \
      --dataset "${DATASET_DIR}" \
      --base-config "${BASE_TRAIN_CONFIG}" \
      --cv-root "${SWEEP_ROOT}" \
      --fold "${FOLD}" \
      --n-folds "${N_FOLDS}" \
      --seed "${CV_SEED}" \
      --val-offset "${CV_VAL_OFFSET}" \
      --sweep-table "${SWEEP_TABLE}" \
      --combo-index "${COMBO_INDEX}" \
      --default-model-name legnet
  )"

  transcriptml train "${CONFIG_PATH}"

  mkdir -p "${FOLD_DIR}/eval"
  transcriptml evaluate \
    --checkpoint "${FOLD_DIR}/model/best.pt" \
    --dataset "${FOLD_DIR}/dataset" \
    --out-csv "${FOLD_DIR}/eval/${EVAL_SPLIT}_predictions.csv" \
    --split "${EVAL_SPLIT}" \
    --batch-size "${PRED_BATCH_SIZE}" \
    --device "${DEVICE}"
done

python "${SCRIPT_DIR}/summarize_hparam_sweep.py" \
  --sweep-root "${SWEEP_ROOT}" \
  --sweep-table "${SWEEP_TABLE}" \
  --out "${SWEEP_ROOT}/sweep_summary.tsv"
