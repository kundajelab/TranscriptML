#!/bin/bash

# Shared Sherlock defaults for MPRA/LegNet runs.
#
# Normal workflow:
#   1. Copy scripts/mpra/ to a writable run directory, or copy the full
#      scripts/ directory if you want the transcriptome scripts too.
#   2. Edit the "User settings" blocks in this copied file.
#   3. Leave the "Internal setup" block alone unless you are changing how the
#      scripts discover their own location.

# ---------------------------------------------------------------------------
# Internal setup: usually leave this alone.
# ---------------------------------------------------------------------------
_MPRA_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_CONFIG_DIR="${SCRIPT_CONFIG_DIR:-${_MPRA_SCRIPT_DIR}}"

# Advanced option: source another shell config before applying defaults below.
if [[ -n "${TRANSCRIPTML_RUN_CONFIG:-}" ]]; then
  if [[ ! -f "${TRANSCRIPTML_RUN_CONFIG}" ]]; then
    echo "TRANSCRIPTML_RUN_CONFIG does not exist: ${TRANSCRIPTML_RUN_CONFIG}" >&2
    return 1 2>/dev/null || exit 1
  fi
  TRANSCRIPTML_RUN_CONFIG_DIR="$(cd "$(dirname "${TRANSCRIPTML_RUN_CONFIG}")" && pwd)"
  TRANSCRIPTML_RUN_CONFIG="${TRANSCRIPTML_RUN_CONFIG_DIR}/$(basename "${TRANSCRIPTML_RUN_CONFIG}")"
  source "${TRANSCRIPTML_RUN_CONFIG}"
else
  TRANSCRIPTML_RUN_CONFIG_DIR=""
fi

# ---------------------------------------------------------------------------
# User settings: environment and TranscriptML source.
# ---------------------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-transcript-ml}"
SHERLOCK_CONDA_ROOT="${SHERLOCK_CONDA_ROOT:-${GROUP_HOME:-${HOME}}/miniconda}"

# If this copied scripts/mpra directory is outside the TranscriptML repo and the
# package is not installed in CONDA_ENV, set TRANSCRIPTML_REPO to the clean repo
# checkout. If scripts/mpra still lives inside the repo, this is auto-detected.
TRANSCRIPTML_REPO="${TRANSCRIPTML_REPO:-}"
_TRANSCRIPTML_REPO_CANDIDATE="$(cd "${_MPRA_SCRIPT_DIR}/../.." && pwd)"
if [[ -z "${TRANSCRIPTML_REPO}" && -d "${_TRANSCRIPTML_REPO_CANDIDATE}/src/transcriptml" ]]; then
  TRANSCRIPTML_REPO="${_TRANSCRIPTML_REPO_CANDIDATE}"
fi

# ---------------------------------------------------------------------------
# User settings: MPRA table inputs for build_legnet_input.sh.
# ---------------------------------------------------------------------------
MPRA_TABLE="${MPRA_TABLE:-}"
SEQUENCE_COL="${SEQUENCE_COL:-sequence}"
TARGET_COL="${TARGET_COL:-stability}"
ID_COL="${ID_COL:-}"
SPLIT_COL="${SPLIT_COL:-}"
METADATA_COLS="${METADATA_COLS:-}"
MPRA_LENGTH="${MPRA_LENGTH:-}"
DELIMITER="${DELIMITER:-}"

# ---------------------------------------------------------------------------
# User settings: output locations.
# ---------------------------------------------------------------------------
RUN_NAME="${RUN_NAME:-MPRA_LegNet}"
RUN_ROOT="${RUN_ROOT:-/scratch/users/${USER:-user}/RNAStability/TranscriptML/${RUN_NAME}}"
DATASET_DIR="${DATASET_DIR:-${RUN_ROOT}/data/mpra}"
TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-${RUN_ROOT}/train_eval}"
MODEL_DIR="${MODEL_DIR:-${TRAIN_OUTPUT_ROOT}/model}"
EVAL_DIR="${EVAL_DIR:-${TRAIN_OUTPUT_ROOT}/eval}"
GENERATED_TRAIN_CONFIG="${GENERATED_TRAIN_CONFIG:-${TRAIN_OUTPUT_ROOT}/train_config.json}"
CV_ROOT="${CV_ROOT:-${RUN_ROOT}/cv10}"
INTERPRET_ROOT="${INTERPRET_ROOT:-${RUN_ROOT}/interpret}"
INTERPRET_DATASET_DIR="${INTERPRET_DATASET_DIR:-${DATASET_DIR}}"
ISM_OUT_DIR="${ISM_OUT_DIR:-${INTERPRET_ROOT}/ism}"
LEGNET_CHECKPOINT="${LEGNET_CHECKPOINT:-${MODEL_DIR}/best.pt}"

# ---------------------------------------------------------------------------
# User settings: training and runtime knobs.
# ---------------------------------------------------------------------------
BASE_TRAIN_CONFIG="${BASE_TRAIN_CONFIG:-${SCRIPT_CONFIG_DIR}/example_legnet_train_config.json}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
TRAIN_SEED="${TRAIN_SEED:-}"
N_FOLDS="${N_FOLDS:-10}"
CV_SEED="${CV_SEED:-42}"
CV_VAL_OFFSET="${CV_VAL_OFFSET:-1}"

# auto: run the extra transcriptml evaluate step only when DATASET_DIR has
# splits.json. Training itself always writes test predictions from the chosen
# split under MODEL_DIR when the split has test examples.
RUN_SPLIT_EVALUATION="${RUN_SPLIT_EVALUATION:-auto}"

PRED_BATCH_SIZE="${PRED_BATCH_SIZE:-128}"
MUTATION_BATCH_SIZE="${MUTATION_BATCH_SIZE:-512}"
DEVICE="${DEVICE:-cuda}"

# ---------------------------------------------------------------------------
# Internal helpers: leave these alone unless you are editing the scripts.
# ---------------------------------------------------------------------------
setup_transcriptml_env() {
  module load gcc/10.1.0
  module load openblas/0.3.10

  source "${SHERLOCK_CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"

  if [[ -n "${TRANSCRIPTML_REPO}" ]]; then
    if [[ ! -d "${TRANSCRIPTML_REPO}" ]]; then
      echo "TRANSCRIPTML_REPO does not exist: ${TRANSCRIPTML_REPO}" >&2
      return 1
    fi
    cd "${TRANSCRIPTML_REPO}"
    if [[ -d "${TRANSCRIPTML_REPO}/src" ]]; then
      export PYTHONPATH="${TRANSCRIPTML_REPO}/src:${PYTHONPATH:-}"
    fi
  elif ! command -v transcriptml >/dev/null 2>&1; then
    echo "TRANSCRIPTML_REPO is unset and transcriptml is not on PATH after conda activation." >&2
    echo "Set TRANSCRIPTML_REPO in the copied mpra_config.sh, or install TranscriptML in ${CONDA_ENV}." >&2
    return 1
  fi
}
