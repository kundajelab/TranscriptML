#!/bin/bash
#SBATCH --partition=akundaje
#SBATCH --job-name=tml_mpra_deep_shap
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
  if [[ -f "${SLURM_SUBMIT_DIR}/scripts/mpra/mpra_config.sh" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/scripts/mpra"
  elif [[ -f "${SLURM_SUBMIT_DIR}/mpra_config.sh" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
  fi
fi
source "${SCRIPT_DIR}/mpra_config.sh"
setup_transcriptml_env

case "${DEEP_SHAP_REFERENCE_STRATEGY}" in
  full-mono|full-dinuc) ;;
  *)
    echo "MPRA/LegNet DeepSHAP supports only full-mono and full-dinuc references" >&2
    exit 2
    ;;
esac
if ! [[ "${DEEP_SHAP_SEQUENCE_SHARDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "DEEP_SHAP_SEQUENCE_SHARDS must be a positive integer" >&2
  exit 2
fi
if ! [[ "${N_FOLDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "N_FOLDS must be a positive integer" >&2
  exit 2
fi

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  TASK_ID="${SLURM_ARRAY_TASK_ID}"
  if ! [[ "${TASK_ID}" =~ ^[0-9]+$ ]]; then
    echo "SLURM_ARRAY_TASK_ID must be a non-negative integer" >&2
    exit 2
  fi
  FOLD=$((TASK_ID / DEEP_SHAP_SEQUENCE_SHARDS))
  SHARD=$((TASK_ID % DEEP_SHAP_SEQUENCE_SHARDS))
else
  FOLD="${FOLD:-${1:-}}"
  SHARD="${SHARD:-${2:-0}}"
fi
if ! [[ "${FOLD}" =~ ^[0-9]+$ ]] || ((FOLD >= N_FOLDS)); then
  echo "FOLD must be an integer from 0 through $((N_FOLDS - 1))" >&2
  exit 2
fi
if ! [[ "${SHARD}" =~ ^[0-9]+$ ]] || ((SHARD >= DEEP_SHAP_SEQUENCE_SHARDS)); then
  echo "SHARD must be an integer from 0 through $((DEEP_SHAP_SEQUENCE_SHARDS - 1))" >&2
  exit 2
fi

CHECKPOINT="${CV_ROOT}/fold${FOLD}/model/${DEEP_SHAP_CHECKPOINT_NAME}"
OUT_DIR="${DEEP_SHAP_OUT_ROOT}/fold${FOLD}"
if ((DEEP_SHAP_SEQUENCE_SHARDS > 1)); then
  OUT_DIR="${OUT_DIR}/shard${SHARD}"
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Expected trained fold checkpoint at ${CHECKPOINT}." >&2
  echo "Run submit_train_eval_cv.sh first, or change CV_ROOT/DEEP_SHAP_CHECKPOINT_NAME." >&2
  exit 1
fi
if [[ ! -f "${INTERPRET_DATASET_DIR}/X.npy" ]]; then
  echo "Expected MPRA dataset bundle at ${INTERPRET_DATASET_DIR}." >&2
  echo "Set INTERPRET_DATASET_DIR in ${SCRIPT_DIR}/mpra_config.sh." >&2
  exit 1
fi

DEEP_SHAP_ARGS=(
  deep-shap
  --checkpoint "${CHECKPOINT}"
  --dataset "${INTERPRET_DATASET_DIR}"
  --out-dir "${OUT_DIR}"
  --device "${DEVICE}"
  --batch-size "${PRED_BATCH_SIZE}"
  --reference-strategy "${DEEP_SHAP_REFERENCE_STRATEGY}"
  --n-references "${DEEP_SHAP_N_REFERENCES}"
  --seed "${DEEP_SHAP_SEED}"
  --target "${DEEP_SHAP_TARGET}"
  --attribution-batch-size "${DEEP_SHAP_ATTRIBUTION_BATCH_SIZE}"
  --confidence-threshold "${DEEP_SHAP_CONFIDENCE_THRESHOLD}"
  --zero-atol "${DEEP_SHAP_ZERO_ATOL}"
  --ig-points "${DEEP_SHAP_IG_POINTS}"
  --max-ig-points "${DEEP_SHAP_MAX_IG_POINTS}"
  --convergence-atol "${DEEP_SHAP_CONVERGENCE_ATOL}"
  --convergence-rtol "${DEEP_SHAP_CONVERGENCE_RTOL}"
  --reference-predictions-format "${DEEP_SHAP_REFERENCE_PREDICTIONS_FORMAT}"
)

if ((DEEP_SHAP_SEQUENCE_SHARDS > 1)); then
  DEEP_SHAP_ARGS+=(
    --sequence-shard-index "${SHARD}"
    --sequence-shards "${DEEP_SHAP_SEQUENCE_SHARDS}"
  )
fi
if [[ -n "${DEEP_SHAP_MINIMUM_EFFECT_SIZE}" ]]; then
  DEEP_SHAP_ARGS+=(--minimum-effect-size "${DEEP_SHAP_MINIMUM_EFFECT_SIZE}")
fi

transcriptml "${DEEP_SHAP_ARGS[@]}"
