#!/bin/bash
#SBATCH --partition=akundaje
#SBATCH --job-name=tml_rbp_cvplan
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_output/%x_%j.out
#SBATCH --error=slurm_output/%x_%j.err

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

if [[ ! -f "${BUNDLE_DIR}/metadata.json" ]]; then
  echo "Missing RBPNet bundle metadata at ${BUNDLE_DIR}; run scan_select_bundle.sh first." >&2
  exit 1
fi

transcriptml cv create-chromosome-plan \
  --dataset "${BUNDLE_DIR}" \
  --output "${CV_PLAN}" \
  --n-folds "${N_FOLDS}" \
  --group-col "${CHROMOSOME_GROUP_COL}"
