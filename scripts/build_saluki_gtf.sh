#!/bin/bash
#SBATCH --partition=akundaje
#SBATCH --job-name=tml_build_saluki
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_output/%x_%j.out
#SBATCH --error=slurm_output/%x_%j.err

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
parse_workflow_args "$@"
require_no_positional_args
setup_transcriptml_env

mkdir -p "${DATASET_DIR}"

args=(
  build-saluki-gtf
  --gtf "${GTF}"
  --fasta "${FASTA}"
  --out-dir "${DATASET_DIR}"
  --target-id-col "${TARGET_ID_COL}"
  --length "${SALUKI_LENGTH}"
  --truncate-from "${SALUKI_TRUNCATE_FROM}"
)

if [[ -n "${TARGETS}" ]]; then
  args+=(--targets "${TARGETS}")
fi
if [[ -n "${TARGET_COL}" ]]; then
  args+=(--target-col "${TARGET_COL}")
fi
if [[ -n "${SPLIT_COL}" ]]; then
  args+=(--split-col "${SPLIT_COL}")
fi
if [[ -n "${METADATA_COLS}" ]]; then
  args+=(--metadata-cols "${METADATA_COLS}")
fi
case "${SALUKI_MRNA_ONLY,,}" in
  1|true|yes) args+=(--mrna-only) ;;
  0|false|no|"") ;;
  *)
    echo "--mrna-only config value must be 0/1, false/true, or no/yes; got ${SALUKI_MRNA_ONLY}" >&2
    exit 2
    ;;
esac

transcriptml "${args[@]}"
