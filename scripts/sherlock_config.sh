#!/bin/bash

# Shared Sherlock defaults.
#
# Normal workflow:
#   1. Copy the full scripts/ directory to a writable run directory.
#   2. Edit the "User settings" blocks in this copied file.
#   3. Leave the "Internal setup" block alone unless you are changing how the
#      scripts discover their own location.

# ---------------------------------------------------------------------------
# Internal setup: usually leave this alone.
# ---------------------------------------------------------------------------
_TRANSCRIPTML_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_CONFIG_DIR="${SCRIPT_CONFIG_DIR:-${_TRANSCRIPTML_SCRIPT_DIR}}"

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

# If this copied scripts directory is outside the TranscriptML repo and the
# package is not installed in CONDA_ENV, set TRANSCRIPTML_REPO to the clean repo
# checkout. If scripts/ still lives inside the repo, this is auto-detected.
TRANSCRIPTML_REPO="${TRANSCRIPTML_REPO:-}"
_TRANSCRIPTML_REPO_CANDIDATE="$(cd "${_TRANSCRIPTML_SCRIPT_DIR}/.." && pwd)"
if [[ -z "${TRANSCRIPTML_REPO}" && -d "${_TRANSCRIPTML_REPO_CANDIDATE}/src/transcriptml" ]]; then
  TRANSCRIPTML_REPO="${_TRANSCRIPTML_REPO_CANDIDATE}"
fi

# ---------------------------------------------------------------------------
# User settings: data-processing inputs for build_saluki_gtf.sh.
# ---------------------------------------------------------------------------
GTF="${GTF:-/oak/stanford/groups/akundaje/isvock/genomes/Hsap/MANE_SELECT.GRCh38.v1.4.refseq.gtf}"
FASTA="${FASTA:-/oak/stanford/groups/akundaje/isvock/genomes/Hsap/hg38.fa}"
TARGETS="${TARGETS:-oak/stanford/groups/akundaje/isvock/Data_RNAdegNet/RDC_TTDB/all_cells_avg_MANE.csv}}"
TARGET_ID_COL="${TARGET_ID_COL:-transcript_id}"
TARGET_COL="${TARGET_COL:-log_kdeg}"
SPLIT_COL="${SPLIT_COL:-}"
METADATA_COLS="${METADATA_COLS:-}"
SALUKI_LENGTH="${SALUKI_LENGTH:-12288}"

# ---------------------------------------------------------------------------
# User settings: output locations.
# ---------------------------------------------------------------------------
RUN_NAME="${RUN_NAME:-RDC_TTDB_All_SalukiExact}"
RUN_ROOT="${RUN_ROOT:-/scratch/users/${USER:-user}/RNAStability/TranscriptML/${RUN_NAME}}"
DATASET_DIR="${DATASET_DIR:-${RUN_ROOT}/data/saluki}"
CV_ROOT="${CV_ROOT:-${RUN_ROOT}/cv10}"
INTERPRET_ROOT="${INTERPRET_ROOT:-${RUN_ROOT}/interpret}"
INTERPRET_DATASET_DIR="${INTERPRET_DATASET_DIR:-${DATASET_DIR}}"

# ---------------------------------------------------------------------------
# User settings: CV and runtime knobs.
# ---------------------------------------------------------------------------
N_FOLDS="${N_FOLDS:-10}"
CV_SEED="${CV_SEED:-42}"
BASE_TRAIN_CONFIG="${BASE_TRAIN_CONFIG:-${SCRIPT_CONFIG_DIR}/example_train_config.json}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"

# Runtime settings.
PRED_BATCH_SIZE="${PRED_BATCH_SIZE:-128}"
MUTATION_BATCH_SIZE="${MUTATION_BATCH_SIZE:-512}"
DEVICE="${DEVICE:-cuda}"

# Set to all or transcript if you want to do full-transcript analysis
MOTIF_REGION="${MOTIF_REGION:-3utr}"

# ---------------------------------------------------------------------------
# Internal helpers: leave these alone unless you are editing the scripts.
# ---------------------------------------------------------------------------
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
    "GRE|UGUUUGUUUGU"
    "mir19_7mer_m8|UUUGCAC"
    "mir17_7mer_m8|GCACUUU"
    "mir29_7mer_m8|UGGUGCU"
    "mir15_7mer_m8|UGCUGCU"
    "mir130_7mer_m8|UGCACUA"
    "random_ctl1|GCGUCC"
    "random_ctl2|CGCGA"
  )
fi

# Motif pairs used by the epistasis scripts: label|motif1|motif2.
# Leave motif2 empty to test pairs of the same motif.
if ! declare -p MOTIF_EPISTASIS_SPECS >/dev/null 2>&1; then
  MOTIF_EPISTASIS_SPECS=(
    "PRE_PRE|UGUA[A|U|C]AUA|"
    "ARE_ARE|UUAUUUAUU|"
    "PRE_ARE|UGUA[A|U|C]AUA|UUAUUUAUU"
  )
fi

N_SCRAMBLES="${N_SCRAMBLES:-10}"
MOTIF_STRATEGY="${MOTIF_STRATEGY:-random_different}"
MOTIF_SEED="${MOTIF_SEED:-123}"
MAX_EPISTASIS_PAIRS="${MAX_EPISTASIS_PAIRS:-10000}"

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
    echo "Set TRANSCRIPTML_REPO in the copied sherlock_config.sh, or install TranscriptML in ${CONDA_ENV}." >&2
    return 1
  fi
}
