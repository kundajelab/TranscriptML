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

if [[ ! -f "${SWEEP_TABLE}" ]]; then
  echo "SWEEP_TABLE does not exist: ${SWEEP_TABLE}" >&2
  echo "Create one with: python ${SCRIPT_DIR}/make_saluki_hparam_grid.py --preset smoke --out ${SWEEP_TABLE}" >&2
  exit 1
fi

mkdir -p "${SWEEP_ROOT}" slurm_output
NORMALIZED_SWEEP_TABLE="$(
  python "${SCRIPT_DIR}/hparam_sweep_utils.py" normalize-table \
    --table "${SWEEP_TABLE}" \
    --out "${SWEEP_ROOT}/sweep_table.tsv"
)"
N_COMBOS="$(
  python "${SCRIPT_DIR}/hparam_sweep_utils.py" count-rows \
    --table "${NORMALIZED_SWEEP_TABLE}"
)"

if (( N_COMBOS <= 0 )); then
  echo "Sweep table has no hyperparameter rows: ${SWEEP_TABLE}" >&2
  exit 1
fi

ARRAY_SPEC="0-$((N_COMBOS - 1))"
if [[ -n "${SWEEP_MAX_CONCURRENT:-}" ]]; then
  ARRAY_SPEC="${ARRAY_SPEC}%${SWEEP_MAX_CONCURRENT}"
fi

sbatch \
  --array="${ARRAY_SPEC}" \
  --export=ALL,SWEEP_TABLE="${NORMALIZED_SWEEP_TABLE}" \
  "${SCRIPT_DIR}/hparam_sweep_cv_combo.sh"
