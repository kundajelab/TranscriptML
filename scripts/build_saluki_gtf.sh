#!/bin/bash
#SBATCH --partition=akundaje
#SBATCH --job-name=tml_build_saluki
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/sherlock_config.sh"
setup_transcriptml_env

mkdir -p "${DATASET_DIR}"

args=(
  build-saluki-gtf
  --gtf "${GTF}"
  --fasta "${FASTA}"
  --out-dir "${DATASET_DIR}"
  --target-id-col "${TARGET_ID_COL}"
  --length "${SALUKI_LENGTH}"
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

transcriptml "${args[@]}"
