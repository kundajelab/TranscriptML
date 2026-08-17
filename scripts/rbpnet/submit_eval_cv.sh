#!/bin/bash

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

if [[ ! -f "${BUNDLE_DIR}/config.json" ]]; then
  echo "Missing RBPNet bundle at ${BUNDLE_DIR}; run scan_select_bundle.sh first." >&2
  exit 1
fi

missing=()
for ((fold = 0; fold < N_FOLDS; fold++)); do
  checkpoint="${CV_ROOT}/fold${fold}/model/${EVAL_CHECKPOINT_NAME}"
  if [[ ! -f "${checkpoint}" ]]; then
    missing+=("${checkpoint}")
  fi
done
if (( ${#missing[@]} > 0 )); then
  echo "Cannot submit evaluation: missing ${#missing[@]} fold checkpoint(s):" >&2
  printf '  %s\n' "${missing[@]}" >&2
  echo "Run submit_train_cv.sh first or change EVAL_CHECKPOINT_NAME." >&2
  exit 1
fi

mkdir -p "${EVAL_ROOT}" slurm_output
sbatch --array="0-$((N_FOLDS - 1))" "${SCRIPT_DIR}/eval_cv_fold.sh"
