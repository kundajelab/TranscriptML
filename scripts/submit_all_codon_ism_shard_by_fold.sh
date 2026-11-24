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
parse_workflow_args "$@"
require_no_positional_args
export_workflow_env

CODON_ISM_SHARDS_PER_FOLD="${CODON_ISM_SHARDS_PER_FOLD:-10}"
if ! [[ "${N_FOLDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "N_FOLDS must be a positive integer" >&2
  exit 2
fi
if ! [[ "${CODON_ISM_SHARDS_PER_FOLD}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CODON_ISM_SHARDS_PER_FOLD must be a positive integer" >&2
  exit 2
fi

N_TASKS=$((N_FOLDS * CODON_ISM_SHARDS_PER_FOLD))
ARRAY_SPEC="0-$((N_TASKS - 1))"
if [[ -n "${CODON_ISM_MAX_CONCURRENT:-}" ]]; then
  if ! [[ "${CODON_ISM_MAX_CONCURRENT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "CODON_ISM_MAX_CONCURRENT must be empty or a positive integer" >&2
    exit 2
  fi
  ARRAY_SPEC="${ARRAY_SPEC}%${CODON_ISM_MAX_CONCURRENT}"
fi

sbatch \
  --array="${ARRAY_SPEC}" \
  --export=ALL,CODON_ISM_SHARDS_PER_FOLD="${CODON_ISM_SHARDS_PER_FOLD}" \
  "${SCRIPT_DIR}/all_codon_ism_shard_by_fold.sh"
