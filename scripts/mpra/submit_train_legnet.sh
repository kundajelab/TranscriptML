#!/bin/bash
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
parse_workflow_args "$@"
require_no_positional_args
prepare_train_submission "${TRAIN_OUTPUT_ROOT}"

sbatch "${SCRIPT_DIR}/train_legnet.sh"
