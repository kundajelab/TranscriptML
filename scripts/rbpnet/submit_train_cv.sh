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

if [[ ! -f "${CV_PLAN}" ]]; then
  echo "Missing chromosome CV plan ${CV_PLAN}; run create_chromosome_cv_plan.sh first." >&2
  exit 1
fi

mkdir -p "${CV_ROOT}" slurm_output
sbatch --array="0-$((N_FOLDS - 1))" "${SCRIPT_DIR}/train_cv_fold.sh"
