#!/bin/bash
#SBATCH --partition=akundaje
#SBATCH --job-name=tml_mpra_build
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
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

if [[ -z "${MPRA_TABLE}" ]]; then
  echo "Set MPRA_TABLE in ${SCRIPT_DIR}/mpra_config.sh before running build_legnet_input.sh." >&2
  exit 1
fi

mkdir -p "${DATASET_DIR}"

args=(
  build-mpra
  "${MPRA_TABLE}"
  "${DATASET_DIR}"
  --sequence-col "${SEQUENCE_COL}"
)

if [[ -n "${TARGET_COL}" ]]; then
  args+=(--target-col "${TARGET_COL}")
fi
if [[ -n "${ID_COL}" ]]; then
  args+=(--id-col "${ID_COL}")
fi
if [[ -n "${MPRA_LENGTH}" ]]; then
  args+=(--length "${MPRA_LENGTH}")
fi
if [[ -n "${METADATA_COLS}" ]]; then
  args+=(--metadata-cols "${METADATA_COLS}")
fi
if [[ -n "${SPLIT_COL}" ]]; then
  args+=(--split-col "${SPLIT_COL}")
fi
if [[ -n "${DELIMITER}" ]]; then
  args+=(--delimiter "${DELIMITER}")
fi

transcriptml "${args[@]}"
