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

CONDA_ENV="${CONDA_ENV:-transcript-ml}"
SHERLOCK_CONDA_ROOT="${SHERLOCK_CONDA_ROOT:-${GROUP_HOME:-${HOME}}/miniconda}"
TRANSCRIPTML_REPO="${TRANSCRIPTML_REPO:-}"
_TRANSCRIPTML_REPO_CANDIDATE="$(cd "${_RBPNET_SCRIPT_DIR}/../.." && pwd)"
if [[ -z "${TRANSCRIPTML_REPO}" && -d "${_TRANSCRIPTML_REPO_CANDIDATE}/src/transcriptml" ]]; then
  TRANSCRIPTML_REPO="${_TRANSCRIPTML_REPO_CANDIDATE}"
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

export_workflow_env() {
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

setup_transcriptml_env() {
  module load gcc/10.1.0
  module load openblas/0.3.10
  source "${SHERLOCK_CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
  if [[ -n "${TRANSCRIPTML_REPO}" ]]; then
    if [[ ! -d "${TRANSCRIPTML_REPO}/src/transcriptml" ]]; then
      echo "TRANSCRIPTML_REPO is not a TranscriptML checkout: ${TRANSCRIPTML_REPO}" >&2
      return 1
    fi
    cd "${TRANSCRIPTML_REPO}"
    export PYTHONPATH="${TRANSCRIPTML_REPO}/src:${PYTHONPATH:-}"
  elif ! command -v transcriptml >/dev/null 2>&1; then
    echo "Set TRANSCRIPTML_REPO or install TranscriptML in ${CONDA_ENV}." >&2
    return 1
  fi
}
