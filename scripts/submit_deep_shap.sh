#!/bin/bash
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

if ! [[ "${N_FOLDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "N_FOLDS must be a positive integer" >&2
  exit 2
fi
if ! [[ "${DEEP_SHAP_SEQUENCE_SHARDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "DEEP_SHAP_SEQUENCE_SHARDS must be a positive integer" >&2
  exit 2
fi

N_TASKS=$((N_FOLDS * DEEP_SHAP_SEQUENCE_SHARDS))
ARRAY_SPEC="0-$((N_TASKS - 1))"
if [[ -n "${DEEP_SHAP_MAX_CONCURRENT}" ]]; then
  if ! [[ "${DEEP_SHAP_MAX_CONCURRENT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "DEEP_SHAP_MAX_CONCURRENT must be empty or a positive integer" >&2
    exit 2
  fi
  ARRAY_SPEC="${ARRAY_SPEC}%${DEEP_SHAP_MAX_CONCURRENT}"
fi

sbatch --array="${ARRAY_SPEC}" "${SCRIPT_DIR}/run_deep_shap.sh"
