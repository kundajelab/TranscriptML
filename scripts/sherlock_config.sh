#!/bin/bash

# Shared Sherlock defaults. For run-specific paths or knobs, set
# TRANSCRIPTML_RUN_CONFIG to a separate config file instead of editing this file.

_TRANSCRIPTML_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSCRIPTML_REPO="${TRANSCRIPTML_REPO:-$(cd "${_TRANSCRIPTML_SCRIPT_DIR}/.." && pwd)}"

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

CONDA_ENV="${CONDA_ENV:-transcript-ml}"
SHERLOCK_CONDA_ROOT="${SHERLOCK_CONDA_ROOT:-${GROUP_HOME:-${HOME}}/miniconda}"

# Data-processing inputs for build_saluki_gtf.sh.
GTF="${GTF:-/path/to/annotations.gtf}"
FASTA="${FASTA:-/path/to/genome.fa}"
TARGETS="${TARGETS:-/path/to/targets.csv}"
TARGET_ID_COL="${TARGET_ID_COL:-transcript_id}"
TARGET_COL="${TARGET_COL:-log_kdeg}"
SPLIT_COL="${SPLIT_COL:-}"
METADATA_COLS="${METADATA_COLS:-}"
SALUKI_LENGTH="${SALUKI_LENGTH:-12288}"

# Main run locations.
RUN_NAME="${RUN_NAME:-saluki_human_example}"
RUN_ROOT="${RUN_ROOT:-/scratch/users/${USER:-user}/TranscriptML/${RUN_NAME}}"
DATASET_DIR="${DATASET_DIR:-${RUN_ROOT}/data/saluki}"
CV_ROOT="${CV_ROOT:-${RUN_ROOT}/cv10}"
INTERPRET_ROOT="${INTERPRET_ROOT:-${RUN_ROOT}/interpret}"
INTERPRET_DATASET_DIR="${INTERPRET_DATASET_DIR:-${DATASET_DIR}}"

# 10-fold CV settings.
N_FOLDS="${N_FOLDS:-10}"
CV_SEED="${CV_SEED:-42}"
BASE_TRAIN_CONFIG="${BASE_TRAIN_CONFIG:-${TRANSCRIPTML_REPO}/scripts/example_train_config.json}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"

# Runtime settings.
PRED_BATCH_SIZE="${PRED_BATCH_SIZE:-128}"
MUTATION_BATCH_SIZE="${MUTATION_BATCH_SIZE:-512}"
DEVICE="${DEVICE:-cuda}"

# Specs use top-level pipes as separators; pipes inside bracket alternatives are preserved.
parse_motif_ablation_spec() {
  local spec="$1"
  MOTIF_SPEC_LABEL="${spec%%|*}"
  MOTIF_SPEC_1="${spec#*|}"
  MOTIF_SPEC_2=""
  if [[ "${MOTIF_SPEC_LABEL}" == "${spec}" || -z "${MOTIF_SPEC_LABEL}" || -z "${MOTIF_SPEC_1}" ]]; then
    echo "Invalid motif ablation spec: ${spec}" >&2
    return 1
  fi
}

parse_motif_epistasis_spec() {
  local spec="$1"
  local token=""
  local field=0
  local in_bracket=0
  local char
  local i
  MOTIF_SPEC_LABEL=""
  MOTIF_SPEC_1=""
  MOTIF_SPEC_2=""

  for ((i = 0; i < ${#spec}; i++)); do
    char="${spec:i:1}"
    if [[ "${char}" == "[" ]]; then
      in_bracket=1
    elif [[ "${char}" == "]" ]]; then
      in_bracket=0
    elif [[ "${char}" == "|" && "${in_bracket}" -eq 0 && "${field}" -lt 2 ]]; then
      if [[ "${field}" -eq 0 ]]; then
        MOTIF_SPEC_LABEL="${token}"
      else
        MOTIF_SPEC_1="${token}"
      fi
      token=""
      field=$((field + 1))
      continue
    fi
    token+="${char}"
  done

  if [[ "${field}" -eq 1 ]]; then
    MOTIF_SPEC_1="${token}"
  else
    MOTIF_SPEC_2="${token}"
  fi
  if [[ -z "${MOTIF_SPEC_LABEL}" || -z "${MOTIF_SPEC_1}" ]]; then
    echo "Invalid motif epistasis spec: ${spec}" >&2
    return 1
  fi
}

# Motifs used by the ablation scripts: label|motif.
if ! declare -p MOTIF_ABLATION_SPECS >/dev/null 2>&1; then
  MOTIF_ABLATION_SPECS=(
    "PRE|UGUA[A|U|C]AUA"
    "ARE_nonamer|UUAUUUAUU"
    "DRACH|GGACU"
    "let7_7mer_m8|CUACCUC"
    "miR16_7mer_m8|UGCUGCU"
  )
fi

# Motif pairs used by the epistasis scripts: label|motif1|motif2.
# Leave motif2 empty to test pairs of the same motif.
if ! declare -p MOTIF_EPISTASIS_SPECS >/dev/null 2>&1; then
  MOTIF_EPISTASIS_SPECS=(
    "PRE_PRE|UGUA[A|U|C]AUA|"
    "ARE_ARE|UUAUUUAUU|"
    "GGACU_GGACU|GGACU|"
    "PRE_ARE|UGUA[A|U|C]AUA|UUAUUUAUU"
    "PRE_GGACU|UGUA[A|U|C]AUA|GGACU"
    "ARE_GGACU|UUAUUUAUU|GGACU"
    "let7_miR16|CUACCUC|UGCUGCU"
  )
fi

N_SCRAMBLES="${N_SCRAMBLES:-10}"
MOTIF_STRATEGY="${MOTIF_STRATEGY:-random_different}"
MOTIF_SEED="${MOTIF_SEED:-123}"
MAX_EPISTASIS_PAIRS="${MAX_EPISTASIS_PAIRS:-5000}"

setup_transcriptml_env() {
  module load gcc/10.1.0
  module load openblas/0.3.10

  source "${SHERLOCK_CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"

  cd "${TRANSCRIPTML_REPO}"
  export PYTHONPATH="${TRANSCRIPTML_REPO}/src:${PYTHONPATH:-}"
}
