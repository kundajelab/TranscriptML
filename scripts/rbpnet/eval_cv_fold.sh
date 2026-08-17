#!/bin/bash
#SBATCH --partition=akundaje
#SBATCH --job-name=tml_rbp_eval
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH -C GPU_MEM:48GB
#SBATCH --output=slurm_output/%x_%A_%a.out
#SBATCH --error=slurm_output/%x_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  if [[ -f "${SLURM_SUBMIT_DIR}/scripts/rbpnet/rbpnet_config.sh" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/scripts/rbpnet"
  elif [[ -f "${SLURM_SUBMIT_DIR}/rbpnet_config.sh" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
  fi
fi
source "${SCRIPT_DIR}/rbpnet_config.sh"
setup_transcriptml_env

case "${EVAL_SPLIT}" in
  train|val|test|all) ;;
  *)
    echo "EVAL_SPLIT must be train, val, test, or all; got ${EVAL_SPLIT}." >&2
    exit 1
    ;;
esac
if [[ "${EVAL_SAVE_PROFILES}" != "0" && "${EVAL_SAVE_PROFILES}" != "1" ]]; then
  echo "EVAL_SAVE_PROFILES must be 0 or 1; got ${EVAL_SAVE_PROFILES}." >&2
  exit 1
fi

FOLD="${SLURM_ARRAY_TASK_ID:-${1:-}}"
if [[ -z "${FOLD}" ]]; then
  echo "Provide a fold argument or run as a Slurm array job." >&2
  exit 1
fi
if [[ ! "${FOLD}" =~ ^[0-9]+$ || "${FOLD}" -ge "${N_FOLDS}" ]]; then
  echo "Fold must be an integer in [0, $((N_FOLDS - 1))]; got ${FOLD}." >&2
  exit 1
fi

CHECKPOINT="${CV_ROOT}/fold${FOLD}/model/${EVAL_CHECKPOINT_NAME}"
OUT_DIR="${EVAL_ROOT}/fold${FOLD}/${EVAL_SPLIT}"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing fold checkpoint ${CHECKPOINT}; run submit_train_cv.sh first." >&2
  exit 1
fi
if [[ ! -f "${BUNDLE_DIR}/config.json" ]]; then
  echo "Missing RBPNet bundle at ${BUNDLE_DIR}; run scan_select_bundle.sh first." >&2
  exit 1
fi

command=(
  transcriptml evaluate
  --checkpoint "${CHECKPOINT}"
  --dataset "${BUNDLE_DIR}"
  --out-dir "${OUT_DIR}"
  --split "${EVAL_SPLIT}"
  --batch-size "${EVAL_BATCH_SIZE}"
  --device "${EVAL_DEVICE}"
  --calibration-bins "${EVAL_CALIBRATION_BINS}"
  --enrichment-pseudocount "${EVAL_ENRICHMENT_PSEUDOCOUNT}"
  --representative-seed "${EVAL_REPRESENTATIVE_SEED}"
  --representative-per-tier "${EVAL_REPRESENTATIVE_PER_TIER}"
  --representative-min-profile-count "${EVAL_REPRESENTATIVE_MIN_PROFILE_COUNT}"
)
if [[ "${EVAL_SAVE_PROFILES}" == "1" ]]; then
  command+=(--save-profiles)
fi

mkdir -p "${OUT_DIR}"
"${command[@]}"
