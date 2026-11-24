#!/bin/bash

# Shared Sherlock defaults for the eCLIP -> RBPNet chromosome-CV workflow.
# Copy scripts/rbpnet to a writable run directory and edit this file there.
# EXPERIMENTAL: Every preprocessing/modeling stage has been minimally tested and
# only confirmed on PUM2 eCLIP data. Substantially more validation is required
# than for other TranscriptML workflows.

_RBPNET_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_CONFIG_DIR="${SCRIPT_CONFIG_DIR:-${_RBPNET_SCRIPT_DIR}}"

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

# Runtime backend: "apptainer" (recommended on Sherlock) or "conda" (legacy fallback).
RUNTIME_BACKEND="${RUNTIME_BACKEND:-apptainer}"
APPTAINER_IMAGE="${APPTAINER_IMAGE:-${OAK:+${OAK}/isvock/containers/transcriptml-deps_v1.sif}}"
TRANSCRIPTML_SRC_TAR="${TRANSCRIPTML_SRC_TAR:-}"
CODE_SNAPSHOT_ENABLED="${CODE_SNAPSHOT_ENABLED:-1}"
REQUIRE_CODE_SNAPSHOT="${REQUIRE_CODE_SNAPSHOT:-0}"
STAGE_SIF_TO_LSCRATCH="${STAGE_SIF_TO_LSCRATCH:-1}"
STAGE_DATASET_TO_LSCRATCH="${STAGE_DATASET_TO_LSCRATCH:-1}"
JOB_START_JITTER_MAX="${JOB_START_JITTER_MAX:-15}"

CONDA_ENV="${CONDA_ENV:-transcript-ml2}"
SHERLOCK_CONDA_ROOT="${SHERLOCK_CONDA_ROOT:-${GROUP_HOME:-${HOME}}/miniconda}"
TRANSCRIPTML_REPO="${TRANSCRIPTML_REPO:-}"
_TRANSCRIPTML_REPO_CANDIDATE="$(cd "${_RBPNET_SCRIPT_DIR}/../.." && pwd)"
if [[ -z "${TRANSCRIPTML_REPO}" && -d "${_TRANSCRIPTML_REPO_CANDIDATE}/src/transcriptml" ]]; then
  TRANSCRIPTML_REPO="${_TRANSCRIPTML_REPO_CANDIDATE}"
elif [[ -z "${TRANSCRIPTML_REPO}" && -n "${OAK:-}" && -d "${OAK}/isvock/software/TranscriptML/src/transcriptml" ]]; then
  TRANSCRIPTML_REPO="${OAK}/isvock/software/TranscriptML"
fi

# Standard assay inputs. SMINPUT_BAM and each IP_BAMS entry may use LABEL=PATH.
GENOME_FASTA="${GENOME_FASTA:-}"
GTF="${GTF:-}"
SMINPUT_BAM="${SMINPUT_BAM:-}"
if ! declare -p IP_BAMS >/dev/null 2>&1; then
  IP_BAMS=()
fi

_PRE_RUN_ROOT="${RUN_ROOT:-}"
_PRE_PROCESSED_DIR="${PROCESSED_DIR:-}"
_PRE_WINDOW_PREFIX="${WINDOW_PREFIX:-}"
_PRE_SELECTION_PREFIX="${SELECTION_PREFIX:-}"
_PRE_BUNDLE_DIR="${BUNDLE_DIR:-${DATASET_DIR:-}}"
_PRE_CV_PLAN="${CV_PLAN:-}"
_PRE_EXP_NAME="${EXP_NAME:-}"
_PRE_EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-}"
_PRE_EXP_ROOT="${EXP_ROOT:-}"
_PRE_CV_ROOT="${CV_ROOT:-}"
_PRE_EVAL_ROOT="${EVAL_ROOT:-}"
_PRE_BASE_TRAIN_CONFIG="${BASE_TRAIN_CONFIG:-}"

RUN_NAME="${RUN_NAME:-RBPNet_eCLIP}"
RUN_ROOT="${RUN_ROOT:-/scratch/users/${USER:-user}/TranscriptML/${RUN_NAME}}"
PROCESSED_DIR="${PROCESSED_DIR:-${RUN_ROOT}/processed/eclip}"
WINDOW_PREFIX="${WINDOW_PREFIX:-${RUN_ROOT}/windows/windows_100nt}"
SELECTION_PREFIX="${SELECTION_PREFIX:-${RUN_ROOT}/selection/selected_regions}"
BUNDLE_DIR="${BUNDLE_DIR:-${DATASET_DIR:-${RUN_ROOT}/data/rbpnet}}"
DATASET_DIR="${DATASET_DIR:-${BUNDLE_DIR}}"
CV_PLAN="${CV_PLAN:-${RUN_ROOT}/cv/cv5_chromosomes.json}"
EXP_NAME="${EXP_NAME:-}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-${RUN_ROOT}/experiments}"
if [[ -n "${EXP_NAME}" ]]; then
  EXP_ROOT="${EXP_ROOT:-${EXPERIMENTS_DIR}/${EXP_NAME}}"
else
  EXP_ROOT="${EXP_ROOT:-${RUN_ROOT}}"
fi
CV_ROOT="${CV_ROOT:-${EXP_ROOT}/cv}"

COORDINATE_SPACE="${COORDINATE_SPACE:-mature_transcript}"
READ1_RNA_STRAND="${READ1_RNA_STRAND:-opposite}"
MIN_MAPQ="${MIN_MAPQ:-1}"
SIGNAL_COMPRESSION="${SIGNAL_COMPRESSION:-gzip}"
SIGNAL_COMPRESSION_LEVEL="${SIGNAL_COMPRESSION_LEVEL:-1}"
OVERWRITE="${OVERWRITE:-0}"

# Leave WINDOW_STRIDE empty to use 1 for original_rbpnet and 50 otherwise.
WINDOW_SIZE="${WINDOW_SIZE:-100}"
WINDOW_STRIDE="${WINDOW_STRIDE:-}"
MIN_SMINPUT_TPM="${MIN_SMINPUT_TPM:-0}"
# Optional comma-separated biological annotations, e.g. "3putr" or "cds,3putr".
REGION_TYPES="${REGION_TYPES:-}"
DISCARD_MIXED="${DISCARD_MIXED:-0}"
ONLY_MIXED="${ONLY_MIXED:-0}"
SELECTION_STRATEGY="${SELECTION_STRATEGY:-peak_gray_negative}"
MIN_TOTAL_COUNT="${MIN_TOTAL_COUNT:-8}"
MIN_IP_COUNT="${MIN_IP_COUNT:-0}"
MIN_SMINPUT_COUNT="${MIN_SMINPUT_COUNT:-0}"
PEAK_FDR="${PEAK_FDR:-0.05}"
PEAK_MIN_LOG2_RATIO="${PEAK_MIN_LOG2_RATIO:-1.0}"
NEGATIVE_FDR="${NEGATIVE_FDR:-0.05}"
NEGATIVE_MAX_LOG2_RATIO="${NEGATIVE_MAX_LOG2_RATIO:--0.5}"
STITCH_GAP="${STITCH_GAP:-0}"
POISSON_NULL="${POISSON_NULL:-ip_locus_density}"
REPLICATE_MODE="${REPLICATE_MODE:-per_ip}"

INPUT_LENGTH="${INPUT_LENGTH:-300}"
PROFILE_LENGTH="${PROFILE_LENGTH:-300}"
MAX_JITTER="${MAX_JITTER:-32}"
TRANSCRIPT_END_POLICY="${TRANSCRIPT_END_POLICY:-shift_to_fit}"

N_FOLDS="${N_FOLDS:-5}"
CHROMOSOME_GROUP_COL="${CHROMOSOME_GROUP_COL:-group_chromosome}"
BASE_TRAIN_CONFIG="${BASE_TRAIN_CONFIG:-${SCRIPT_CONFIG_DIR}/example_train_config.json}"
DEVICE="${DEVICE:-cuda}"

# Scientific checkpoint evaluation. Each fold report is written under
# EVAL_ROOT/fold<N>/<EVAL_SPLIT>/ and always uses checkpoint-recorded indices.
EVAL_ROOT="${EVAL_ROOT:-${CV_ROOT}/evaluation}"
EVAL_CHECKPOINT_NAME="${EVAL_CHECKPOINT_NAME:-best.pt}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
EVAL_DEVICE="${EVAL_DEVICE:-${DEVICE}}"
EVAL_SAVE_PROFILES="${EVAL_SAVE_PROFILES:-0}"
EVAL_CALIBRATION_BINS="${EVAL_CALIBRATION_BINS:-10}"
EVAL_ENRICHMENT_PSEUDOCOUNT="${EVAL_ENRICHMENT_PSEUDOCOUNT:-0.5}"
EVAL_REPRESENTATIVE_SEED="${EVAL_REPRESENTATIVE_SEED:-123}"
EVAL_REPRESENTATIVE_PER_TIER="${EVAL_REPRESENTATIVE_PER_TIER:-3}"
EVAL_REPRESENTATIVE_MIN_PROFILE_COUNT="${EVAL_REPRESENTATIVE_MIN_PROFILE_COUNT:-10}"

_WORKFLOW_INITIAL_PWD="${PWD}"
_DEFAULT_RUN_ROOT="${RUN_ROOT}"
_DEFAULT_PROCESSED_DIR="${PROCESSED_DIR}"
_DEFAULT_WINDOW_PREFIX="${WINDOW_PREFIX}"
_DEFAULT_SELECTION_PREFIX="${SELECTION_PREFIX}"
_DEFAULT_BUNDLE_DIR="${BUNDLE_DIR}"
_DEFAULT_CV_PLAN="${CV_PLAN}"
_DEFAULT_EXPERIMENTS_DIR="${EXPERIMENTS_DIR}"
_DEFAULT_EXP_ROOT="${EXP_ROOT}"
_DEFAULT_CV_ROOT="${CV_ROOT}"
_DEFAULT_EVAL_ROOT="${EVAL_ROOT}"
_DEFAULT_BASE_TRAIN_CONFIG="${BASE_TRAIN_CONFIG}"

_CLI_CHANGED=0
_CLI_ROOT_CHANGED=0
_CLI_EXP_OR_ROOT_CHANGED=0
_CLI_SET_RUN_NAME=0
_CLI_SET_RUN_ROOT=0
_CLI_SET_PROCESSED_DIR=0
_CLI_SET_WINDOW_PREFIX=0
_CLI_SET_SELECTION_PREFIX=0
_CLI_SET_BUNDLE_DIR=0
_CLI_SET_CV_PLAN=0
_CLI_SET_EXP_NAME=0
_CLI_SET_EXPERIMENTS_DIR=0
_CLI_SET_EXP_ROOT=0
_CLI_SET_CV_ROOT=0
_CLI_SET_EVAL_ROOT=0
_CLI_SET_BASE_TRAIN_CONFIG=0
WORKFLOW_POSITIONAL_ARGS=()

_workflow_abspath() {
  local p="$1"
  if [[ -z "${p}" ]]; then
    printf '%s' ""
    return 0
  fi
  if [[ "${p}" == /* ]]; then
    printf '%s' "${p}"
    return 0
  fi
  printf '%s/%s' "${_WORKFLOW_INITIAL_PWD%/}" "${p#./}"
}

_resolve_workflow_paths() {
  if [[ "${_CLI_SET_RUN_ROOT}" == "0" ]] && { [[ "${_CLI_SET_RUN_NAME}" == "1" ]] || [[ -z "${_PRE_RUN_ROOT}" && "${RUN_ROOT}" == "${_DEFAULT_RUN_ROOT}" ]]; }; then
    RUN_ROOT="/scratch/users/${USER:-user}/TranscriptML/${RUN_NAME}"
  fi
  if [[ "${_CLI_SET_PROCESSED_DIR}" == "0" ]] && { [[ "${_CLI_ROOT_CHANGED}" == "1" ]] || [[ -z "${_PRE_PROCESSED_DIR}" && "${PROCESSED_DIR}" == "${_DEFAULT_PROCESSED_DIR}" ]]; }; then
    PROCESSED_DIR="${RUN_ROOT}/processed/eclip"
  fi
  if [[ "${_CLI_SET_WINDOW_PREFIX}" == "0" ]] && { [[ "${_CLI_ROOT_CHANGED}" == "1" ]] || [[ -z "${_PRE_WINDOW_PREFIX}" && "${WINDOW_PREFIX}" == "${_DEFAULT_WINDOW_PREFIX}" ]]; }; then
    WINDOW_PREFIX="${RUN_ROOT}/windows/windows_100nt"
  fi
  if [[ "${_CLI_SET_SELECTION_PREFIX}" == "0" ]] && { [[ "${_CLI_ROOT_CHANGED}" == "1" ]] || [[ -z "${_PRE_SELECTION_PREFIX}" && "${SELECTION_PREFIX}" == "${_DEFAULT_SELECTION_PREFIX}" ]]; }; then
    SELECTION_PREFIX="${RUN_ROOT}/selection/selected_regions"
  fi
  if [[ "${_CLI_SET_BUNDLE_DIR}" == "0" ]] && { [[ "${_CLI_ROOT_CHANGED}" == "1" ]] || [[ -z "${_PRE_BUNDLE_DIR}" && "${BUNDLE_DIR}" == "${_DEFAULT_BUNDLE_DIR}" ]]; }; then
    BUNDLE_DIR="${RUN_ROOT}/data/rbpnet"
  fi
  DATASET_DIR="${BUNDLE_DIR}"
  if [[ "${_CLI_SET_CV_PLAN}" == "0" ]] && { [[ "${_CLI_ROOT_CHANGED}" == "1" ]] || [[ -z "${_PRE_CV_PLAN}" && "${CV_PLAN}" == "${_DEFAULT_CV_PLAN}" ]]; }; then
    CV_PLAN="${RUN_ROOT}/cv/cv5_chromosomes.json"
  fi
  if [[ "${_CLI_SET_EXPERIMENTS_DIR}" == "0" ]] && { [[ "${_CLI_ROOT_CHANGED}" == "1" ]] || [[ -z "${_PRE_EXPERIMENTS_DIR}" && "${EXPERIMENTS_DIR}" == "${_DEFAULT_EXPERIMENTS_DIR}" ]]; }; then
    EXPERIMENTS_DIR="${RUN_ROOT}/experiments"
  fi
  if [[ "${_CLI_SET_EXP_ROOT}" == "0" ]] && { [[ "${_CLI_EXP_OR_ROOT_CHANGED}" == "1" ]] || [[ -z "${_PRE_EXP_ROOT}" && "${EXP_ROOT}" == "${_DEFAULT_EXP_ROOT}" ]]; }; then
    if [[ -n "${EXP_NAME}" ]]; then
      EXP_ROOT="${EXPERIMENTS_DIR}/${EXP_NAME}"
    else
      EXP_ROOT="${RUN_ROOT}"
    fi
  fi
  if [[ "${_CLI_SET_CV_ROOT}" == "0" ]] && { [[ "${_CLI_EXP_OR_ROOT_CHANGED}" == "1" ]] || [[ -z "${_PRE_CV_ROOT}" && "${CV_ROOT}" == "${_DEFAULT_CV_ROOT}" ]]; }; then
    CV_ROOT="${EXP_ROOT}/cv"
  fi
  if [[ "${_CLI_SET_EVAL_ROOT}" == "0" ]] && { [[ "${_CLI_SET_CV_ROOT}" == "1" || "${_CLI_EXP_OR_ROOT_CHANGED}" == "1" ]] || [[ -z "${_PRE_EVAL_ROOT}" && "${EVAL_ROOT}" == "${_DEFAULT_EVAL_ROOT}" ]]; }; then
    EVAL_ROOT="${CV_ROOT}/evaluation"
  fi
}

print_workflow_help() {
  cat <<EOF
Usage: $(basename "$0") [options]

Experiment & directory options:
  -e, --exp, --experiment NAME     Scope CV/evaluation outputs under \${RUN_ROOT}/experiments/NAME
      --exp-root DIR               Explicit experiment root directory
      --experiments-dir DIR        Parent directory for named experiments (default: \${RUN_ROOT}/experiments)
  -r, --run-root DIR               Dataset/project root directory (default: ${RUN_ROOT})
      --run-name NAME              Run name used in default RUN_ROOT (default: ${RUN_NAME})
  -d, --dataset-dir, --bundle-dir DIR Shared RBPNet dataset bundle directory (default: ${BUNDLE_DIR})
      --processed-dir DIR          Preprocessed eCLIP signals directory (default: ${PROCESSED_DIR})
      --window-prefix PREFIX       Window scan output prefix (default: ${WINDOW_PREFIX})
      --selection-prefix PREFIX    Region selection output prefix (default: ${SELECTION_PREFIX})
      --cv-plan FILE               Chromosome CV plan JSON (default: ${CV_PLAN})
      --cv-root DIR                CV output root (default: \${EXP_ROOT}/cv)
      --eval-root DIR              Evaluation output root (default: \${CV_ROOT}/evaluation)

Model & runtime options:
  -c, --train-config FILE          Base JSON training config (default: ${BASE_TRAIN_CONFIG})
      --n-folds INT                Number of CV folds (default: ${N_FOLDS})
      --eval-split SPLIT           Evaluation split: train, val, test, or all (default: ${EVAL_SPLIT})
      --checkpoint-name NAME       Fold checkpoint filename (default: ${EVAL_CHECKPOINT_NAME})
      --gtf FILE                   Annotation GTF file
      --fasta, --genome-fasta FILE Genome FASTA file
      --sminput-bam SPEC           SMInput BAM path or LABEL=PATH
      --ip-bam SPEC                IP BAM path or LABEL=PATH (repeatable)
      --config, --run-config FILE  Source an additional shell configuration file
  -s, --set KEY=VALUE              Override any config variable in rbpnet_config.sh (repeatable)
  -h, --help                       Show this help message and exit
EOF
}

parse_workflow_args() {
  WORKFLOW_POSITIONAL_ARGS=()
  if (( $# == 0 )); then
    return 0
  fi

  local arg key val
  local ip_bams_cleared=0
  while (( $# > 0 )); do
    arg="$1"
    shift
    case "${arg}" in
      --)
        while (( $# > 0 )); do
          WORKFLOW_POSITIONAL_ARGS+=("$1")
          shift
        done
        break
        ;;
      -h|--help)
        print_workflow_help
        exit 0
        ;;
      --*=*)
        set -- "${arg%%=*}" "${arg#*=}" "$@"
        continue
        ;;
      -e|--exp|--experiment)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        EXP_NAME="$1"; shift
        _CLI_SET_EXP_NAME=1; _CLI_EXP_OR_ROOT_CHANGED=1; _CLI_CHANGED=1
        ;;
      --exp-root)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        EXP_ROOT="$(_workflow_abspath "$1")"; shift
        _CLI_SET_EXP_ROOT=1; _CLI_EXP_OR_ROOT_CHANGED=1; _CLI_CHANGED=1
        ;;
      --experiments-dir)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        EXPERIMENTS_DIR="$(_workflow_abspath "$1")"; shift
        _CLI_SET_EXPERIMENTS_DIR=1; _CLI_EXP_OR_ROOT_CHANGED=1; _CLI_CHANGED=1
        ;;
      -r|--run-root)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        RUN_ROOT="$(_workflow_abspath "$1")"; shift
        _CLI_SET_RUN_ROOT=1; _CLI_ROOT_CHANGED=1; _CLI_EXP_OR_ROOT_CHANGED=1; _CLI_CHANGED=1
        ;;
      --run-name)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        RUN_NAME="$1"; shift
        _CLI_SET_RUN_NAME=1; _CLI_ROOT_CHANGED=1; _CLI_EXP_OR_ROOT_CHANGED=1; _CLI_CHANGED=1
        ;;
      -c|--train-config|--base-config)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        BASE_TRAIN_CONFIG="$(_workflow_abspath "$1")"; shift
        _CLI_SET_BASE_TRAIN_CONFIG=1; _CLI_CHANGED=1
        ;;
      -d|--dataset|--dataset-dir|--bundle-dir)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        BUNDLE_DIR="$(_workflow_abspath "$1")"
        DATASET_DIR="${BUNDLE_DIR}"
        shift
        _CLI_SET_BUNDLE_DIR=1; _CLI_CHANGED=1
        ;;
      --processed-dir)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        PROCESSED_DIR="$(_workflow_abspath "$1")"; shift
        _CLI_SET_PROCESSED_DIR=1; _CLI_CHANGED=1
        ;;
      --window-prefix)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        WINDOW_PREFIX="$(_workflow_abspath "$1")"; shift
        _CLI_SET_WINDOW_PREFIX=1; _CLI_CHANGED=1
        ;;
      --selection-prefix)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        SELECTION_PREFIX="$(_workflow_abspath "$1")"; shift
        _CLI_SET_SELECTION_PREFIX=1; _CLI_CHANGED=1
        ;;
      --cv-plan)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        CV_PLAN="$(_workflow_abspath "$1")"; shift
        _CLI_SET_CV_PLAN=1; _CLI_CHANGED=1
        ;;
      --cv-root)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        CV_ROOT="$(_workflow_abspath "$1")"; shift
        _CLI_SET_CV_ROOT=1; _CLI_CHANGED=1
        ;;
      --eval-root)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        EVAL_ROOT="$(_workflow_abspath "$1")"; shift
        _CLI_SET_EVAL_ROOT=1; _CLI_CHANGED=1
        ;;
      --eval-split)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        EVAL_SPLIT="$1"; shift
        _CLI_CHANGED=1
        ;;
      --eval-checkpoint-name|--checkpoint-name)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        EVAL_CHECKPOINT_NAME="$1"; shift
        _CLI_CHANGED=1
        ;;
      --n-folds)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        N_FOLDS="$1"; shift
        _CLI_CHANGED=1
        ;;
      --gtf)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        GTF="$(_workflow_abspath "$1")"; shift
        _CLI_CHANGED=1
        ;;
      --fasta|--genome-fasta)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        GENOME_FASTA="$(_workflow_abspath "$1")"; shift
        _CLI_CHANGED=1
        ;;
      --sminput-bam)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        SMINPUT_BAM="$1"; shift
        _CLI_CHANGED=1
        ;;
      --ip-bam)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        if (( ip_bams_cleared == 0 )); then
          IP_BAMS=()
          ip_bams_cleared=1
        fi
        IP_BAMS+=("$1"); shift
        _CLI_CHANGED=1
        ;;
      --config|--run-config)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        TRANSCRIPTML_RUN_CONFIG="$(_workflow_abspath "$1")"; shift
        if [[ ! -f "${TRANSCRIPTML_RUN_CONFIG}" ]]; then
          echo "TRANSCRIPTML_RUN_CONFIG does not exist: ${TRANSCRIPTML_RUN_CONFIG}" >&2
          return 1
        fi
        source "${TRANSCRIPTML_RUN_CONFIG}"
        _CLI_CHANGED=1
        ;;
      -s|--set)
        if (( $# == 0 )); then echo "Missing KEY=VALUE for ${arg}" >&2; return 2; fi
        if [[ "$1" != *=* ]]; then
          echo "Expected KEY=VALUE for ${arg}, got: $1" >&2
          return 2
        fi
        key="${1%%=*}"
        val="${1#*=}"
        shift
        if ! [[ "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
          echo "Invalid variable name in --set: ${key}" >&2
          return 2
        fi
        printf -v "${key}" '%s' "${val}"
        printf -v "_CLI_SET_${key}" '%s' "1"
        case "${key}" in
          RUN_NAME|RUN_ROOT)
            _CLI_ROOT_CHANGED=1
            _CLI_EXP_OR_ROOT_CHANGED=1
            ;;
          EXP_NAME|EXP_ROOT|EXPERIMENTS_DIR)
            _CLI_EXP_OR_ROOT_CHANGED=1
            ;;
          DATASET_DIR)
            BUNDLE_DIR="${val}"
            _CLI_SET_BUNDLE_DIR=1
            ;;
          BUNDLE_DIR)
            DATASET_DIR="${val}"
            ;;
        esac
        _CLI_CHANGED=1
        ;;
      -*)
        echo "Unknown option: ${arg}" >&2
        return 2
        ;;
      *)
        WORKFLOW_POSITIONAL_ARGS+=("${arg}")
        ;;
    esac
  done

  if [[ "${_CLI_CHANGED}" == "1" ]]; then
    _resolve_workflow_paths
  fi
}

require_no_positional_args() {
  if (( ${#WORKFLOW_POSITIONAL_ARGS[@]} > 0 )); then
    echo "Unexpected positional argument(s): ${WORKFLOW_POSITIONAL_ARGS[*]}" >&2
    return 2
  fi
}

prepare_code_snapshot() {
  if [[ "${RUNTIME_BACKEND}" != "apptainer" || "${CODE_SNAPSHOT_ENABLED}" != "1" ]]; then
    return 0
  fi
  if [[ -n "${TRANSCRIPTML_SRC_TAR}" ]]; then
    return 0
  fi
  if [[ -z "${TRANSCRIPTML_REPO}" || ! -d "${TRANSCRIPTML_REPO}/src/transcriptml" ]]; then
    return 0
  fi
  local snapshot_dir="${EXP_ROOT}/code_snapshot"
  if ! mkdir -p "${snapshot_dir}" 2>/dev/null; then
    return 0
  fi
  local tmp_tar="${snapshot_dir}/.transcriptml_snapshot_${$}.tar.gz"
  if ! tar -C "${TRANSCRIPTML_REPO}" \
      --exclude='__pycache__' \
      --exclude='*.pyc' \
      --exclude='*.pyo' \
      --exclude='*.egg-info' \
      -cf - src pyproject.toml 2>/dev/null | gzip -n > "${tmp_tar}"; then
    rm -f "${tmp_tar}"
    return 0
  fi
  local content_hash=""
  if command -v sha256sum >/dev/null 2>&1; then
    content_hash="$(sha256sum "${tmp_tar}" | awk '{print substr($1, 1, 12)}')"
  elif command -v shasum >/dev/null 2>&1; then
    content_hash="$(shasum -a 256 "${tmp_tar}" | awk '{print substr($1, 1, 12)}')"
  else
    content_hash="$(date -u +%Y%m%dT%H%M%SZ)_$$"
  fi
  local git_prefix=""
  if command -v git >/dev/null 2>&1 && git -C "${TRANSCRIPTML_REPO}" rev-parse --short HEAD >/dev/null 2>&1; then
    git_prefix="$(git -C "${TRANSCRIPTML_REPO}" rev-parse --short HEAD)_"
  fi
  local final_tar="${snapshot_dir}/transcriptml_${git_prefix}${content_hash}.tar.gz"
  if [[ -f "${final_tar}" ]]; then
    rm -f "${tmp_tar}"
  else
    mv -f "${tmp_tar}" "${final_tar}"
  fi
  TRANSCRIPTML_SRC_TAR="${final_tar}"
}

export_workflow_env() {
  prepare_code_snapshot
  export RUNTIME_BACKEND APPTAINER_IMAGE TRANSCRIPTML_SRC_TAR CODE_SNAPSHOT_ENABLED REQUIRE_CODE_SNAPSHOT
  export STAGE_SIF_TO_LSCRATCH STAGE_DATASET_TO_LSCRATCH JOB_START_JITTER_MAX
  export CONDA_ENV SHERLOCK_CONDA_ROOT TRANSCRIPTML_REPO TRANSCRIPTML_RUN_CONFIG
  export GENOME_FASTA GTF SMINPUT_BAM
  export RUN_NAME RUN_ROOT PROCESSED_DIR WINDOW_PREFIX SELECTION_PREFIX BUNDLE_DIR DATASET_DIR CV_PLAN
  export EXP_NAME EXPERIMENTS_DIR EXP_ROOT CV_ROOT EVAL_ROOT
  export COORDINATE_SPACE READ1_RNA_STRAND MIN_MAPQ SIGNAL_COMPRESSION SIGNAL_COMPRESSION_LEVEL OVERWRITE
  export WINDOW_SIZE WINDOW_STRIDE MIN_SMINPUT_TPM REGION_TYPES DISCARD_MIXED ONLY_MIXED SELECTION_STRATEGY
  export MIN_TOTAL_COUNT MIN_IP_COUNT MIN_SMINPUT_COUNT PEAK_FDR PEAK_MIN_LOG2_RATIO NEGATIVE_FDR NEGATIVE_MAX_LOG2_RATIO
  export STITCH_GAP POISSON_NULL REPLICATE_MODE
  export INPUT_LENGTH PROFILE_LENGTH MAX_JITTER TRANSCRIPT_END_POLICY
  export N_FOLDS CHROMOSOME_GROUP_COL BASE_TRAIN_CONFIG DEVICE
  export EVAL_CHECKPOINT_NAME EVAL_SPLIT EVAL_BATCH_SIZE EVAL_DEVICE EVAL_SAVE_PROFILES
  export EVAL_CALIBRATION_BINS EVAL_ENRICHMENT_PSEUDOCOUNT EVAL_REPRESENTATIVE_SEED
  export EVAL_REPRESENTATIVE_PER_TIER EVAL_REPRESENTATIVE_MIN_PROFILE_COUNT
}

prepare_train_submission() {
  local target_root="${1:-${EXP_ROOT}}"
  if [[ ! -f "${BASE_TRAIN_CONFIG}" ]]; then
    echo "BASE_TRAIN_CONFIG does not exist: ${BASE_TRAIN_CONFIG}" >&2
    return 1
  fi
  local snapshot_dir
  if [[ -n "${EXP_NAME}" ]]; then
    snapshot_dir="${EXP_ROOT}/config_snapshot"
  else
    snapshot_dir="${target_root}/config_snapshot"
  fi
  mkdir -p "${snapshot_dir}"
  local snapshot_cfg="${snapshot_dir}/base_train_config.json"
  local src_real dst_real
  src_real="$(cd "$(dirname "${BASE_TRAIN_CONFIG}")" && pwd)/$(basename "${BASE_TRAIN_CONFIG}")"
  dst_real="$(cd "${snapshot_dir}" && pwd)/base_train_config.json"
  if [[ "${src_real}" != "${dst_real}" ]]; then
    cp "${BASE_TRAIN_CONFIG}" "${snapshot_cfg}"
  fi
  BASE_TRAIN_CONFIG="${snapshot_cfg}"
  export_workflow_env
}

stage_dataset_bundle() {
  local src_dir="$1"
  local overlay_dir="${2:-}"
  if [[ "${STAGE_DATASET_TO_LSCRATCH:-1}" != "1" || -z "${L_SCRATCH_JOB:-}" || ! -d "${L_SCRATCH_JOB}" ]]; then
    printf '%s\n' "${overlay_dir:-${src_dir}}"
    return 0
  fi
  if [[ ! -d "${src_dir}" ]]; then
    echo "[runtime] ERROR: dataset directory does not exist: ${src_dir}" >&2
    return 1
  fi
  if [[ "${src_dir}" == "${L_SCRATCH_JOB}"/* && -z "${overlay_dir}" ]]; then
    printf '%s\n' "${src_dir}"
    return 0
  fi

  local canonical_src
  canonical_src="$(cd "${src_dir}" && pwd -P)" || {
    echo "[runtime] ERROR: unable to resolve dataset directory: ${src_dir}" >&2
    return 1
  }
  local base_key
  if command -v sha256sum >/dev/null 2>&1; then
    base_key="$(printf '%s' "${canonical_src}" | sha256sum | awk '{print substr($1, 1, 16)}')"
  else
    base_key="$(printf '%s' "${canonical_src}" | tr '/ ' '__')"
  fi
  local datasets_root="${L_SCRATCH_JOB}/datasets"
  local base_dst="${datasets_root}/base_${base_key}"

  if [[ ! -f "${base_dst}/.staged_ok" ]]; then
    local req_kb avail_kb margin_kb
    req_kb="$(du -skL "${canonical_src}" 2>/dev/null | awk '{print $1}')"
    avail_kb="$(df -Pk "${L_SCRATCH_JOB}" 2>/dev/null | awk 'NR==2 {print $4}')"
    margin_kb=$((5 * 1024 * 1024))
    if [[ -n "${req_kb}" && -n "${avail_kb}" ]] && (( avail_kb < req_kb + margin_kb )); then
      echo "[runtime] Skipping dataset staging (${req_kb} KB needed, ${avail_kb} KB free on ${L_SCRATCH_JOB}); reading directly." >&2
      printf '%s\n' "${overlay_dir:-${src_dir}}"
      return 0
    fi
    local t0 t1 tmp_base="${datasets_root}/.base_${base_key}.tmp.$$"
    t0="$(date +%s)"
    rm -rf "${tmp_base}"
    if ! mkdir -p "${tmp_base}"; then
      echo "[runtime] ERROR: failed to create staging directory ${tmp_base}" >&2
      return 1
    fi
    if ! cp -aL "${canonical_src}/." "${tmp_base}/"; then
      rm -rf "${tmp_base}"
      echo "[runtime] ERROR: failed to copy dataset ${canonical_src} -> ${tmp_base}" >&2
      return 1
    fi
    if [[ -f "${canonical_src}/X.npy" && ! -s "${tmp_base}/X.npy" ]]; then
      rm -rf "${tmp_base}"
      echo "[runtime] ERROR: staged X.npy is missing or empty under ${tmp_base}" >&2
      return 1
    fi
    if [[ -f "${canonical_src}/config.json" && ! -s "${tmp_base}/config.json" ]]; then
      rm -rf "${tmp_base}"
      echo "[runtime] ERROR: staged config.json is missing or empty under ${tmp_base}" >&2
      return 1
    fi
    if ! touch "${tmp_base}/.staged_ok"; then
      rm -rf "${tmp_base}"
      echo "[runtime] ERROR: failed to write staging marker under ${tmp_base}" >&2
      return 1
    fi
    rm -rf "${base_dst}"
    if ! mv -f "${tmp_base}" "${base_dst}"; then
      rm -rf "${tmp_base}"
      echo "[runtime] ERROR: failed to finalize staged dataset ${base_dst}" >&2
      return 1
    fi
    t1="$(date +%s)"
    echo "[runtime] Staged dataset ${canonical_src} -> ${base_dst} ($((req_kb / 1024)) MB) in $((t1 - t0))s" >&2
  fi

  printf '%s\n' "${base_dst}"
}

_run_in_transcriptml_container() {
  local -a cmd=(apptainer exec --cleanenv)
  if [[ "${_TML_APPTAINER_USE_NV:-0}" == "1" ]]; then
    cmd+=(--nv)
  fi
  if [[ -n "${L_SCRATCH_JOB:-}" && -d "${L_SCRATCH_JOB}" ]]; then
    cmd+=(--bind "${L_SCRATCH_JOB}:${L_SCRATCH_JOB}")
  elif [[ -n "${_TML_RUNTIME_ROOT:-}" && -d "${_TML_RUNTIME_ROOT}" ]]; then
    cmd+=(--bind "${_TML_RUNTIME_ROOT}:${_TML_RUNTIME_ROOT}")
  fi
  cmd+=(
    --env "PYTHONPATH=${_TML_STAGED_SRC_DIR}"
    --env "TMPDIR=${_TML_RUNTIME_ROOT}/tmp"
    --env "MPLCONFIGDIR=${_TML_RUNTIME_ROOT}/cache/matplotlib"
    --env "NUMBA_CACHE_DIR=${_TML_RUNTIME_ROOT}/cache/numba"
    --env "TORCH_HOME=${_TML_RUNTIME_ROOT}/cache/torch"
    --env "XDG_CACHE_HOME=${_TML_RUNTIME_ROOT}/cache/xdg"
    --env "PYTHONNOUSERSITE=1"
    --env "PYTHONDONTWRITEBYTECODE=1"
    --env "PYTHONFAULTHANDLER=1"
    --env "OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}"
    --env "MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}"
    --env "OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}"
  )
  if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then
    cmd+=(--env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}")
  fi
  if [[ -n "${NVIDIA_VISIBLE_DEVICES+x}" ]]; then
    cmd+=(--env "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES}")
  fi
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    cmd+=(--env "SLURM_JOB_ID=${SLURM_JOB_ID}")
  fi
  if [[ -n "${L_SCRATCH_JOB:-}" ]]; then
    cmd+=(--env "L_SCRATCH_JOB=${L_SCRATCH_JOB}")
  fi
  "${cmd[@]}" "${_TML_STAGED_SIF}" "$@"
}

_setup_apptainer_env() {
  if ! command -v apptainer >/dev/null 2>&1; then
    echo "RUNTIME_BACKEND=apptainer requested, but 'apptainer' is not on PATH." >&2
    return 1
  fi
  if [[ -z "${APPTAINER_IMAGE}" || ! -f "${APPTAINER_IMAGE}" ]]; then
    echo "APPTAINER_IMAGE does not exist: ${APPTAINER_IMAGE:-<unset>}" >&2
    echo "Build it with containers/build_sherlock_sif.sh or set APPTAINER_IMAGE." >&2
    return 1
  fi

  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${JOB_START_JITTER_MAX:-0}" =~ ^[1-9][0-9]*$ ]]; then
    local jitter=$((RANDOM % JOB_START_JITTER_MAX))
    if (( jitter > 0 )); then
      sleep "${jitter}"
    fi
  fi

  if [[ -n "${L_SCRATCH_JOB:-}" && -d "${L_SCRATCH_JOB}" ]]; then
    _TML_RUNTIME_ROOT="${L_SCRATCH_JOB}/tml_runtime"
  else
    _TML_RUNTIME_ROOT="${TMPDIR:-/tmp}/tml_runtime_${USER:-user}_${SLURM_JOB_ID:-$$}"
  fi
  mkdir -p \
    "${_TML_RUNTIME_ROOT}/tmp" \
    "${_TML_RUNTIME_ROOT}/cache/matplotlib" \
    "${_TML_RUNTIME_ROOT}/cache/numba" \
    "${_TML_RUNTIME_ROOT}/cache/torch" \
    "${_TML_RUNTIME_ROOT}/cache/xdg"

  local t0 t1 t2
  t0="$(date +%s)"
  if [[ "${STAGE_SIF_TO_LSCRATCH:-1}" == "1" && -n "${L_SCRATCH_JOB:-}" && -d "${L_SCRATCH_JOB}" ]]; then
    _TML_STAGED_SIF="${_TML_RUNTIME_ROOT}/$(basename "${APPTAINER_IMAGE}")"
    if [[ ! -f "${_TML_STAGED_SIF}" ]]; then
      local tmp_sif="${_TML_STAGED_SIF}.tmp.$$"
      if ! cp -f "${APPTAINER_IMAGE}" "${tmp_sif}" || ! mv -f "${tmp_sif}" "${_TML_STAGED_SIF}"; then
        rm -f "${tmp_sif}"
        echo "ERROR: failed to stage APPTAINER_IMAGE ${APPTAINER_IMAGE} -> ${_TML_STAGED_SIF}" >&2
        return 1
      fi
    fi
  else
    _TML_STAGED_SIF="${APPTAINER_IMAGE}"
  fi
  t1="$(date +%s)"

  local code_root="${_TML_RUNTIME_ROOT}/transcriptml_src"
  if [[ ! -d "${code_root}/src/transcriptml" ]]; then
    local tmp_code="${code_root}.tmp.$$"
    rm -rf "${tmp_code}" "${code_root}"
    mkdir -p "${tmp_code}" || return 1
    if [[ -n "${TRANSCRIPTML_SRC_TAR}" ]]; then
      if [[ ! -f "${TRANSCRIPTML_SRC_TAR}" ]]; then
        rm -rf "${tmp_code}"
        echo "ERROR: TRANSCRIPTML_SRC_TAR was explicitly set but does not exist: ${TRANSCRIPTML_SRC_TAR}" >&2
        return 1
      fi
      if ! tar -C "${tmp_code}" -xzf "${TRANSCRIPTML_SRC_TAR}"; then
        rm -rf "${tmp_code}"
        echo "ERROR: failed to extract TRANSCRIPTML_SRC_TAR: ${TRANSCRIPTML_SRC_TAR}" >&2
        return 1
      fi
    elif [[ "${REQUIRE_CODE_SNAPSHOT:-0}" == "1" ]]; then
      rm -rf "${tmp_code}"
      echo "ERROR: REQUIRE_CODE_SNAPSHOT=1 but TRANSCRIPTML_SRC_TAR is unset. Submit via a submit_*.sh wrapper or export TRANSCRIPTML_SRC_TAR." >&2
      return 1
    elif [[ -n "${TRANSCRIPTML_REPO}" && -d "${TRANSCRIPTML_REPO}/src/transcriptml" ]]; then
      if ! tar -C "${TRANSCRIPTML_REPO}" \
          --exclude='__pycache__' \
          --exclude='*.pyc' \
          --exclude='*.pyo' \
          --exclude='*.egg-info' \
          -cf - src pyproject.toml | tar -C "${tmp_code}" -xf -; then
        rm -rf "${tmp_code}"
        echo "ERROR: failed to stage TranscriptML source from ${TRANSCRIPTML_REPO}" >&2
        return 1
      fi
    else
      rm -rf "${tmp_code}"
      echo "Could not locate TranscriptML source (TRANSCRIPTML_SRC_TAR and TRANSCRIPTML_REPO are unset or invalid)." >&2
      return 1
    fi
    if [[ ! -d "${tmp_code}/src/transcriptml" ]] || ! mv -f "${tmp_code}" "${code_root}"; then
      rm -rf "${tmp_code}"
      echo "ERROR: staged TranscriptML source is invalid under ${code_root}" >&2
      return 1
    fi
  fi
  _TML_STAGED_SRC_DIR="${code_root}/src"
  t2="$(date +%s)"

  _TML_APPTAINER_USE_NV=0
  if [[ "${DEVICE:-}" != "cpu" ]] && { \
      [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]] \
      || [[ -n "${SLURM_JOB_GPUS:-}" ]] \
      || [[ -n "${SLURM_STEP_GPUS:-}" ]] \
      || [[ -e /dev/nvidiactl ]]; }; then
    _TML_APPTAINER_USE_NV=1
  fi

  unset PYTHONPATH || true
  export PYTHONNOUSERSITE=1

  transcriptml() {
    _run_in_transcriptml_container python -m transcriptml.cli.main "$@"
  }
  python() {
    _run_in_transcriptml_container python "$@"
  }
  python3() {
    _run_in_transcriptml_container python "$@"
  }

  echo "[runtime] backend=apptainer host=$(hostname) nv=${_TML_APPTAINER_USE_NV} sif_stage=$((t1 - t0))s code_stage=$((t2 - t1))s sif=${_TML_STAGED_SIF}" >&2
}

setup_transcriptml_env() {
  if [[ "${RUNTIME_BACKEND:-apptainer}" == "apptainer" ]]; then
    _setup_apptainer_env
    return $?
  fi

  module load gcc/10.1.0
  module load openblas/0.3.10
  source "${SHERLOCK_CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"

  local cache_root="${L_SCRATCH_JOB:-${TMPDIR:-/tmp}/tml_runtime_${USER:-user}_${SLURM_JOB_ID:-$$}}/tml_cache"
  mkdir -p "${cache_root}/tmp" "${cache_root}/matplotlib" "${cache_root}/numba" "${cache_root}/torch" "${cache_root}/xdg"
  export TMPDIR="${cache_root}/tmp"
  export MPLCONFIGDIR="${cache_root}/matplotlib"
  export NUMBA_CACHE_DIR="${cache_root}/numba"
  export TORCH_HOME="${cache_root}/torch"
  export XDG_CACHE_HOME="${cache_root}/xdg"
  export PYTHONNOUSERSITE=1
  export PYTHONFAULTHANDLER=1
  unset PYTHONPATH || true

  if [[ -n "${TRANSCRIPTML_REPO}" ]]; then
    if [[ ! -d "${TRANSCRIPTML_REPO}/src/transcriptml" ]]; then
      echo "TRANSCRIPTML_REPO is not a TranscriptML checkout: ${TRANSCRIPTML_REPO}" >&2
      return 1
    fi
    cd "${TRANSCRIPTML_REPO}"
    export PYTHONPATH="${TRANSCRIPTML_REPO}/src"
  elif ! command -v transcriptml >/dev/null 2>&1; then
    echo "Set TRANSCRIPTML_REPO or install TranscriptML in ${CONDA_ENV}." >&2
    return 1
  fi
}
