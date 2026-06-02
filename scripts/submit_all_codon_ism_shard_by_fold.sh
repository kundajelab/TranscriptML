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

CODON_ISM_SHARDS_PER_FOLD="${CODON_ISM_SHARDS_PER_FOLD:-10}"
if (( CODON_ISM_SHARDS_PER_FOLD <= 0 )); then
  echo "CODON_ISM_SHARDS_PER_FOLD must be positive" >&2
  exit 2
fi

for ((FOLD = 0; FOLD < N_FOLDS; FOLD++)); do
  sbatch \
    --array=0-$((CODON_ISM_SHARDS_PER_FOLD - 1)) \
    --export=ALL,FOLD="${FOLD}",CODON_ISM_SHARDS_PER_FOLD="${CODON_ISM_SHARDS_PER_FOLD}" \
    "${SCRIPT_DIR}/all_codon_ism_shard_by_fold.sh"
done
