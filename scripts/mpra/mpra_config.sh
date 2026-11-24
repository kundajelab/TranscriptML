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
# Runtime backend: "apptainer" (recommended on Sherlock) or "conda" (legacy fallback).
RUNTIME_BACKEND="${RUNTIME_BACKEND:-apptainer}"
APPTAINER_IMAGE="${APPTAINER_IMAGE:-${OAK:+${OAK}/isvock/containers/transcriptml-deps_v1.sif}}"
TRANSCRIPTML_SRC_TAR="${TRANSCRIPTML_SRC_TAR:-}"
CODE_SNAPSHOT_ENABLED="${CODE_SNAPSHOT_ENABLED:-1}"
STAGE_SIF_TO_LSCRATCH="${STAGE_SIF_TO_LSCRATCH:-1}"
STAGE_DATASET_TO_LSCRATCH="${STAGE_DATASET_TO_LSCRATCH:-1}"
JOB_START_JITTER_MAX="${JOB_START_JITTER_MAX:-15}"

# Legacy Conda settings (used only when RUNTIME_BACKEND=conda).
CONDA_ENV="${CONDA_ENV:-transcript-ml2}"
SHERLOCK_CONDA_ROOT="${SHERLOCK_CONDA_ROOT:-${GROUP_HOME:-${HOME}}/miniconda}"

# Clean TranscriptML checkout used to snapshot/stage src/ into jobs.
TRANSCRIPTML_REPO="${TRANSCRIPTML_REPO:-}"
_TRANSCRIPTML_REPO_CANDIDATE="$(cd "${_MPRA_SCRIPT_DIR}/../.." && pwd)"
if [[ -z "${TRANSCRIPTML_REPO}" && -d "${_TRANSCRIPTML_REPO_CANDIDATE}/src/transcriptml" ]]; then
  TRANSCRIPTML_REPO="${_TRANSCRIPTML_REPO_CANDIDATE}"
elif [[ -z "${TRANSCRIPTML_REPO}" && -n "${OAK:-}" && -d "${OAK}/isvock/software/TranscriptML/src/transcriptml" ]]; then
  TRANSCRIPTML_REPO="${OAK}/isvock/software/TranscriptML"
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
_PRE_RUN_ROOT="${RUN_ROOT:-}"
_PRE_DATASET_DIR="${DATASET_DIR:-}"
_PRE_EXP_NAME="${EXP_NAME:-}"
_PRE_EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-}"
_PRE_EXP_ROOT="${EXP_ROOT:-}"
_PRE_TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-}"
_PRE_MODEL_DIR="${MODEL_DIR:-}"
_PRE_EVAL_DIR="${EVAL_DIR:-}"
_PRE_GENERATED_TRAIN_CONFIG="${GENERATED_TRAIN_CONFIG:-}"
_PRE_CV_ROOT="${CV_ROOT:-}"
_PRE_SWEEP_ROOT="${SWEEP_ROOT:-}"
_PRE_SWEEP_TABLE="${SWEEP_TABLE:-}"
_PRE_INTERPRET_ROOT="${INTERPRET_ROOT:-}"
_PRE_INTERPRET_DATASET_DIR="${INTERPRET_DATASET_DIR:-}"
_PRE_ISM_OUT_DIR="${ISM_OUT_DIR:-}"
_PRE_LEGNET_CHECKPOINT="${LEGNET_CHECKPOINT:-}"
_PRE_DEEP_SHAP_RUN_NAME="${DEEP_SHAP_RUN_NAME:-}"
_PRE_DEEP_SHAP_OUT_ROOT="${DEEP_SHAP_OUT_ROOT:-}"
_PRE_CV_MODEL="${CV_MODEL:-}"
_PRE_BASE_TRAIN_CONFIG="${BASE_TRAIN_CONFIG:-}"

RUN_NAME="${RUN_NAME:-MPRA_LegNet}"
RUN_ROOT="${RUN_ROOT:-/scratch/users/${USER:-user}/RNAStability/TranscriptML/${RUN_NAME}}"
DATASET_DIR="${DATASET_DIR:-${RUN_ROOT}/data/mpra}"
EXP_NAME="${EXP_NAME:-}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-${RUN_ROOT}/experiments}"
if [[ -n "${EXP_NAME}" ]]; then
  EXP_ROOT="${EXP_ROOT:-${EXPERIMENTS_DIR}/${EXP_NAME}}"
else
  EXP_ROOT="${EXP_ROOT:-${RUN_ROOT}}"
fi
TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-${EXP_ROOT}/train_eval}"
MODEL_DIR="${MODEL_DIR:-${TRAIN_OUTPUT_ROOT}/model}"
EVAL_DIR="${EVAL_DIR:-${TRAIN_OUTPUT_ROOT}/eval}"
GENERATED_TRAIN_CONFIG="${GENERATED_TRAIN_CONFIG:-${TRAIN_OUTPUT_ROOT}/train_config.json}"
CV_ROOT="${CV_ROOT:-${EXP_ROOT}/cv10}"
SWEEP_ROOT="${SWEEP_ROOT:-${EXP_ROOT}/hparam_sweep}"
SWEEP_TABLE="${SWEEP_TABLE:-${SCRIPT_CONFIG_DIR}/legnet_hparams.tsv}"
INTERPRET_ROOT="${INTERPRET_ROOT:-${EXP_ROOT}/interpret}"
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
CV_MODEL="${CV_MODEL:-legnet}"
SWEEP_SKIP_COMPLETED="${SWEEP_SKIP_COMPLETED:-1}"
SWEEP_MAX_CONCURRENT="${SWEEP_MAX_CONCURRENT:-20}"

# auto: run the extra transcriptml evaluate step only when DATASET_DIR has
# splits.json. Training itself always writes test predictions from the chosen
# split under MODEL_DIR when the split has test examples.
RUN_SPLIT_EVALUATION="${RUN_SPLIT_EVALUATION:-auto}"

PRED_BATCH_SIZE="${PRED_BATCH_SIZE:-128}"
MUTATION_BATCH_SIZE="${MUTATION_BATCH_SIZE:-512}"
DEVICE="${DEVICE:-cuda}"

# DeepLIFT/SHAP settings. MPRA/LegNet supports only full-mono and full-dinuc
# references. Change DEEP_SHAP_RUN_NAME when comparing configurations that use
# the same strategy. Sequence sharding creates one independent output directory
# per fold and shard.
DEEP_SHAP_REFERENCE_STRATEGY="${DEEP_SHAP_REFERENCE_STRATEGY:-full-mono}"
DEEP_SHAP_RUN_NAME="${DEEP_SHAP_RUN_NAME:-${DEEP_SHAP_REFERENCE_STRATEGY}}"
DEEP_SHAP_OUT_ROOT="${DEEP_SHAP_OUT_ROOT:-${INTERPRET_ROOT}/deep_shap/${DEEP_SHAP_RUN_NAME}}"
DEEP_SHAP_CHECKPOINT_NAME="${DEEP_SHAP_CHECKPOINT_NAME:-best.pt}"
DEEP_SHAP_N_REFERENCES="${DEEP_SHAP_N_REFERENCES:-20}"
DEEP_SHAP_SEED="${DEEP_SHAP_SEED:-123}"
DEEP_SHAP_TARGET="${DEEP_SHAP_TARGET:-0}"
DEEP_SHAP_ATTRIBUTION_BATCH_SIZE="${DEEP_SHAP_ATTRIBUTION_BATCH_SIZE:-1}"
DEEP_SHAP_CONFIDENCE_THRESHOLD="${DEEP_SHAP_CONFIDENCE_THRESHOLD:-0.95}"
DEEP_SHAP_MINIMUM_EFFECT_SIZE="${DEEP_SHAP_MINIMUM_EFFECT_SIZE:-}"
DEEP_SHAP_ZERO_ATOL="${DEEP_SHAP_ZERO_ATOL:-1e-6}"
DEEP_SHAP_IG_POINTS="${DEEP_SHAP_IG_POINTS:-8}"
DEEP_SHAP_MAX_IG_POINTS="${DEEP_SHAP_MAX_IG_POINTS:-32}"
DEEP_SHAP_CONVERGENCE_ATOL="${DEEP_SHAP_CONVERGENCE_ATOL:-1e-3}"
DEEP_SHAP_CONVERGENCE_RTOL="${DEEP_SHAP_CONVERGENCE_RTOL:-0.01}"
DEEP_SHAP_REFERENCE_PREDICTIONS_FORMAT="${DEEP_SHAP_REFERENCE_PREDICTIONS_FORMAT:-none}"
DEEP_SHAP_SEQUENCE_SHARDS="${DEEP_SHAP_SEQUENCE_SHARDS:-1}"
DEEP_SHAP_MAX_CONCURRENT="${DEEP_SHAP_MAX_CONCURRENT:-20}"

# ---------------------------------------------------------------------------
# Internal helpers: leave these alone unless you are editing the scripts.
# ---------------------------------------------------------------------------
_WORKFLOW_INITIAL_PWD="${PWD}"
_DEFAULT_RUN_ROOT="${RUN_ROOT}"
_DEFAULT_DATASET_DIR="${DATASET_DIR}"
_DEFAULT_EXPERIMENTS_DIR="${EXPERIMENTS_DIR}"
_DEFAULT_EXP_ROOT="${EXP_ROOT}"
_DEFAULT_TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT}"
_DEFAULT_MODEL_DIR="${MODEL_DIR}"
_DEFAULT_EVAL_DIR="${EVAL_DIR}"
_DEFAULT_GENERATED_TRAIN_CONFIG="${GENERATED_TRAIN_CONFIG}"
_DEFAULT_CV_ROOT="${CV_ROOT}"
_DEFAULT_SWEEP_ROOT="${SWEEP_ROOT}"
_DEFAULT_SWEEP_TABLE="${SWEEP_TABLE}"
_DEFAULT_INTERPRET_ROOT="${INTERPRET_ROOT}"
_DEFAULT_INTERPRET_DATASET_DIR="${INTERPRET_DATASET_DIR}"
_DEFAULT_ISM_OUT_DIR="${ISM_OUT_DIR}"
_DEFAULT_LEGNET_CHECKPOINT="${LEGNET_CHECKPOINT}"
_DEFAULT_DEEP_SHAP_RUN_NAME="${DEEP_SHAP_RUN_NAME}"
_DEFAULT_DEEP_SHAP_OUT_ROOT="${DEEP_SHAP_OUT_ROOT}"
_DEFAULT_CV_MODEL="${CV_MODEL}"
_DEFAULT_BASE_TRAIN_CONFIG="${BASE_TRAIN_CONFIG}"
TRAIN_MODEL_OVERRIDE="${TRAIN_MODEL_OVERRIDE:-}"

_CLI_CHANGED=0
_CLI_ROOT_CHANGED=0
_CLI_EXP_OR_ROOT_CHANGED=0
_CLI_SET_RUN_NAME=0
_CLI_SET_RUN_ROOT=0
_CLI_SET_DATASET_DIR=0
_CLI_SET_EXP_NAME=0
_CLI_SET_EXPERIMENTS_DIR=0
_CLI_SET_EXP_ROOT=0
_CLI_SET_TRAIN_OUTPUT_ROOT=0
_CLI_SET_MODEL_DIR=0
_CLI_SET_EVAL_DIR=0
_CLI_SET_GENERATED_TRAIN_CONFIG=0
_CLI_SET_CV_ROOT=0
_CLI_SET_SWEEP_ROOT=0
_CLI_SET_SWEEP_TABLE=0
_CLI_SET_INTERPRET_ROOT=0
_CLI_SET_INTERPRET_DATASET_DIR=0
_CLI_SET_ISM_OUT_DIR=0
_CLI_SET_LEGNET_CHECKPOINT=0
_CLI_SET_DEEP_SHAP_REFERENCE_STRATEGY=0
_CLI_SET_DEEP_SHAP_RUN_NAME=0
_CLI_SET_DEEP_SHAP_OUT_ROOT=0
_CLI_SET_CV_MODEL=0
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

_extract_model_name_from_config() {
  local cfg="$1"
  if [[ ! -f "${cfg}" ]]; then
    return 0
  fi
  local py_cmd=""
  if command -v python3 >/dev/null 2>&1; then
    py_cmd="python3"
  elif command -v python >/dev/null 2>&1; then
    py_cmd="python"
  fi
  if [[ -n "${py_cmd}" ]]; then
    command "${py_cmd}" -c '
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as fh:
        data = json.load(fh)
    model = data.get("model")
    if isinstance(model, dict) and model.get("name"):
        print(str(model["name"]).strip())
    elif isinstance(model, str) and model.strip():
        print(model.strip())
except Exception:
    pass
' "${cfg}" 2>/dev/null || true
  fi
}

_resolve_cv_model_from_config() {
  if [[ "${_CLI_SET_CV_MODEL}" == "1" || -n "${_PRE_CV_MODEL}" || "${CV_MODEL}" != "${_DEFAULT_CV_MODEL}" ]]; then
    return 0
  fi
  if [[ -f "${BASE_TRAIN_CONFIG}" ]]; then
    local extracted
    extracted="$(_extract_model_name_from_config "${BASE_TRAIN_CONFIG}")"
    if [[ -n "${extracted}" ]]; then
      CV_MODEL="${extracted}"
      _DEFAULT_CV_MODEL="${CV_MODEL}"
    fi
  fi
}

_resolve_workflow_paths() {
  if [[ "${_CLI_SET_RUN_ROOT}" == "0" ]] && { [[ "${_CLI_SET_RUN_NAME}" == "1" ]] || [[ -z "${_PRE_RUN_ROOT}" && "${RUN_ROOT}" == "${_DEFAULT_RUN_ROOT}" ]]; }; then
    RUN_ROOT="/scratch/users/${USER:-user}/RNAStability/TranscriptML/${RUN_NAME}"
  fi
  if [[ "${_CLI_SET_DATASET_DIR}" == "0" ]] && { [[ "${_CLI_ROOT_CHANGED}" == "1" ]] || [[ -z "${_PRE_DATASET_DIR}" && "${DATASET_DIR}" == "${_DEFAULT_DATASET_DIR}" ]]; }; then
    DATASET_DIR="${RUN_ROOT}/data/mpra"
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
  if [[ "${_CLI_SET_TRAIN_OUTPUT_ROOT}" == "0" ]] && { [[ "${_CLI_EXP_OR_ROOT_CHANGED}" == "1" ]] || [[ -z "${_PRE_TRAIN_OUTPUT_ROOT}" && "${TRAIN_OUTPUT_ROOT}" == "${_DEFAULT_TRAIN_OUTPUT_ROOT}" ]]; }; then
    TRAIN_OUTPUT_ROOT="${EXP_ROOT}/train_eval"
  fi
  if [[ "${_CLI_SET_MODEL_DIR}" == "0" ]] && { [[ "${_CLI_SET_TRAIN_OUTPUT_ROOT}" == "1" || "${_CLI_EXP_OR_ROOT_CHANGED}" == "1" ]] || [[ -z "${_PRE_MODEL_DIR}" && "${MODEL_DIR}" == "${_DEFAULT_MODEL_DIR}" ]]; }; then
    MODEL_DIR="${TRAIN_OUTPUT_ROOT}/model"
  fi
  if [[ "${_CLI_SET_EVAL_DIR}" == "0" ]] && { [[ "${_CLI_SET_TRAIN_OUTPUT_ROOT}" == "1" || "${_CLI_EXP_OR_ROOT_CHANGED}" == "1" ]] || [[ -z "${_PRE_EVAL_DIR}" && "${EVAL_DIR}" == "${_DEFAULT_EVAL_DIR}" ]]; }; then
    EVAL_DIR="${TRAIN_OUTPUT_ROOT}/eval"
  fi
  if [[ "${_CLI_SET_GENERATED_TRAIN_CONFIG}" == "0" ]] && { [[ "${_CLI_SET_TRAIN_OUTPUT_ROOT}" == "1" || "${_CLI_EXP_OR_ROOT_CHANGED}" == "1" ]] || [[ -z "${_PRE_GENERATED_TRAIN_CONFIG}" && "${GENERATED_TRAIN_CONFIG}" == "${_DEFAULT_GENERATED_TRAIN_CONFIG}" ]]; }; then
    GENERATED_TRAIN_CONFIG="${TRAIN_OUTPUT_ROOT}/train_config.json"
  fi
  if [[ "${_CLI_SET_CV_ROOT}" == "0" ]] && { [[ "${_CLI_EXP_OR_ROOT_CHANGED}" == "1" ]] || [[ -z "${_PRE_CV_ROOT}" && "${CV_ROOT}" == "${_DEFAULT_CV_ROOT}" ]]; }; then
    CV_ROOT="${EXP_ROOT}/cv10"
  fi
  if [[ "${_CLI_SET_SWEEP_ROOT}" == "0" ]] && { [[ "${_CLI_EXP_OR_ROOT_CHANGED}" == "1" ]] || [[ -z "${_PRE_SWEEP_ROOT}" && "${SWEEP_ROOT}" == "${_DEFAULT_SWEEP_ROOT}" ]]; }; then
    SWEEP_ROOT="${EXP_ROOT}/hparam_sweep"
  fi
  if [[ "${_CLI_SET_INTERPRET_ROOT}" == "0" ]] && { [[ "${_CLI_EXP_OR_ROOT_CHANGED}" == "1" ]] || [[ -z "${_PRE_INTERPRET_ROOT}" && "${INTERPRET_ROOT}" == "${_DEFAULT_INTERPRET_ROOT}" ]]; }; then
    INTERPRET_ROOT="${EXP_ROOT}/interpret"
  fi
  if [[ "${_CLI_SET_INTERPRET_DATASET_DIR}" == "0" ]] && { [[ "${_CLI_SET_DATASET_DIR}" == "1" || "${_CLI_ROOT_CHANGED}" == "1" ]] || [[ -z "${_PRE_INTERPRET_DATASET_DIR}" && "${INTERPRET_DATASET_DIR}" == "${_DEFAULT_INTERPRET_DATASET_DIR}" ]]; }; then
    INTERPRET_DATASET_DIR="${DATASET_DIR}"
  fi
  if [[ "${_CLI_SET_ISM_OUT_DIR}" == "0" ]] && { [[ "${_CLI_SET_INTERPRET_ROOT}" == "1" || "${_CLI_EXP_OR_ROOT_CHANGED}" == "1" ]] || [[ -z "${_PRE_ISM_OUT_DIR}" && "${ISM_OUT_DIR}" == "${_DEFAULT_ISM_OUT_DIR}" ]]; }; then
    ISM_OUT_DIR="${INTERPRET_ROOT}/ism"
  fi
  if [[ "${_CLI_SET_LEGNET_CHECKPOINT}" == "0" ]] && { [[ "${_CLI_SET_MODEL_DIR}" == "1" || "${_CLI_SET_TRAIN_OUTPUT_ROOT}" == "1" || "${_CLI_EXP_OR_ROOT_CHANGED}" == "1" ]] || [[ -z "${_PRE_LEGNET_CHECKPOINT}" && "${LEGNET_CHECKPOINT}" == "${_DEFAULT_LEGNET_CHECKPOINT}" ]]; }; then
    LEGNET_CHECKPOINT="${MODEL_DIR}/best.pt"
  fi
  if [[ "${_CLI_SET_DEEP_SHAP_RUN_NAME}" == "0" ]] && { [[ "${_CLI_SET_DEEP_SHAP_REFERENCE_STRATEGY}" == "1" ]] || [[ -z "${_PRE_DEEP_SHAP_RUN_NAME}" && "${DEEP_SHAP_RUN_NAME}" == "${_DEFAULT_DEEP_SHAP_RUN_NAME}" ]]; }; then
    DEEP_SHAP_RUN_NAME="${DEEP_SHAP_REFERENCE_STRATEGY}"
  fi
  if [[ "${_CLI_SET_DEEP_SHAP_OUT_ROOT}" == "0" ]] && { [[ "${_CLI_EXP_OR_ROOT_CHANGED}" == "1" || "${_CLI_SET_INTERPRET_ROOT}" == "1" || "${_CLI_SET_DEEP_SHAP_RUN_NAME}" == "1" || "${_CLI_SET_DEEP_SHAP_REFERENCE_STRATEGY}" == "1" ]] || [[ -z "${_PRE_DEEP_SHAP_OUT_ROOT}" && "${DEEP_SHAP_OUT_ROOT}" == "${_DEFAULT_DEEP_SHAP_OUT_ROOT}" ]]; }; then
    DEEP_SHAP_OUT_ROOT="${INTERPRET_ROOT}/deep_shap/${DEEP_SHAP_RUN_NAME}"
  fi
  _resolve_cv_model_from_config
}

print_workflow_help() {
  cat <<EOF
Usage: $(basename "$0") [options]

Experiment & directory options:
  -e, --exp, --experiment NAME     Scope model/CV/interpret outputs under \${RUN_ROOT}/experiments/NAME
      --exp-root DIR               Explicit experiment root directory
      --experiments-dir DIR        Parent directory for named experiments (default: \${RUN_ROOT}/experiments)
  -r, --run-root DIR               Dataset/project root directory (default: ${RUN_ROOT})
      --run-name NAME              Run name used in default RUN_ROOT (default: ${RUN_NAME})
  -d, --dataset, --dataset-dir DIR Shared MPRA dataset bundle directory (default: ${DATASET_DIR})
      --interpret-dataset-dir DIR  Dataset bundle for interpretation (default: \${DATASET_DIR})
      --cv-root DIR                CV output root (default: \${EXP_ROOT}/cv10)
      --interpret-root DIR         Interpretation output root (default: \${EXP_ROOT}/interpret)
      --train-output-root DIR      Single-split training output root (default: \${EXP_ROOT}/train_eval)
      --sweep-root DIR             Hyperparameter sweep root (default: \${EXP_ROOT}/hparam_sweep)
      --ism-out-dir DIR            Single-checkpoint ISM output directory (default: \${INTERPRET_ROOT}/ism)
      --checkpoint FILE            Single-checkpoint path for run_legnet_ism.sh

Model & runtime options:
  -c, --train-config FILE          Base JSON training config (default: ${BASE_TRAIN_CONFIG})
  -m, --model NAME                 Override model name (defaults to model.name in train config)
      --n-folds INT                Number of CV folds (default: ${N_FOLDS})
      --seed, --cv-seed INT        CV split seed (default: ${CV_SEED})
      --train-seed INT             Single-split training seed override
      --eval-split SPLIT           Evaluation split name (default: ${EVAL_SPLIT})
      --sweep-table FILE           Hyperparameter sweep TSV/CSV table
      --mpra-table FILE            Input MPRA CSV/TSV table for build_legnet_input.sh
      --sequence-col COL           MPRA sequence column name
      --target-col COL             MPRA target column name
      --deep-shap-strategy NAME    DeepSHAP reference strategy (full-mono or full-dinuc)
      --deep-shap-run-name NAME    DeepSHAP run subdirectory name
      --deep-shap-out-root DIR     DeepSHAP output root directory
      --config, --run-config FILE  Source an additional shell configuration file
  -s, --set KEY=VALUE              Override any config variable in mpra_config.sh (repeatable)
  -h, --help                       Show this help message and exit
EOF
}

parse_workflow_args() {
  WORKFLOW_POSITIONAL_ARGS=()
  if (( $# == 0 )); then
    _resolve_cv_model_from_config
    return 0
  fi

  local arg key val
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
      -m|--model)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        CV_MODEL="$1"; TRAIN_MODEL_OVERRIDE="$1"; shift
        _CLI_SET_CV_MODEL=1; _CLI_CHANGED=1
        ;;
      -d|--dataset|--dataset-dir)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        DATASET_DIR="$(_workflow_abspath "$1")"; shift
        _CLI_SET_DATASET_DIR=1; _CLI_CHANGED=1
        ;;
      --interpret-dataset|--interpret-dataset-dir)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        INTERPRET_DATASET_DIR="$(_workflow_abspath "$1")"; shift
        _CLI_SET_INTERPRET_DATASET_DIR=1; _CLI_CHANGED=1
        ;;
      --cv-root)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        CV_ROOT="$(_workflow_abspath "$1")"; shift
        _CLI_SET_CV_ROOT=1; _CLI_CHANGED=1
        ;;
      --interpret-root)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        INTERPRET_ROOT="$(_workflow_abspath "$1")"; shift
        _CLI_SET_INTERPRET_ROOT=1; _CLI_CHANGED=1
        ;;
      --train-output-root)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        TRAIN_OUTPUT_ROOT="$(_workflow_abspath "$1")"; shift
        _CLI_SET_TRAIN_OUTPUT_ROOT=1; _CLI_CHANGED=1
        ;;
      --model-dir)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        MODEL_DIR="$(_workflow_abspath "$1")"; shift
        _CLI_SET_MODEL_DIR=1; _CLI_CHANGED=1
        ;;
      --eval-dir)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        EVAL_DIR="$(_workflow_abspath "$1")"; shift
        _CLI_SET_EVAL_DIR=1; _CLI_CHANGED=1
        ;;
      --generated-train-config)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        GENERATED_TRAIN_CONFIG="$(_workflow_abspath "$1")"; shift
        _CLI_SET_GENERATED_TRAIN_CONFIG=1; _CLI_CHANGED=1
        ;;
      --sweep-root)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        SWEEP_ROOT="$(_workflow_abspath "$1")"; shift
        _CLI_SET_SWEEP_ROOT=1; _CLI_CHANGED=1
        ;;
      --sweep-table)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        SWEEP_TABLE="$(_workflow_abspath "$1")"; shift
        _CLI_SET_SWEEP_TABLE=1; _CLI_CHANGED=1
        ;;
      --ism-out-dir)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        ISM_OUT_DIR="$(_workflow_abspath "$1")"; shift
        _CLI_SET_ISM_OUT_DIR=1; _CLI_CHANGED=1
        ;;
      --checkpoint)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        LEGNET_CHECKPOINT="$(_workflow_abspath "$1")"; shift
        _CLI_SET_LEGNET_CHECKPOINT=1; _CLI_CHANGED=1
        ;;
      --deep-shap-strategy|--reference-strategy)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        DEEP_SHAP_REFERENCE_STRATEGY="$1"; shift
        _CLI_SET_DEEP_SHAP_REFERENCE_STRATEGY=1; _CLI_CHANGED=1
        ;;
      --deep-shap-run-name)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        DEEP_SHAP_RUN_NAME="$1"; shift
        _CLI_SET_DEEP_SHAP_RUN_NAME=1; _CLI_CHANGED=1
        ;;
      --deep-shap-out-root)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        DEEP_SHAP_OUT_ROOT="$(_workflow_abspath "$1")"; shift
        _CLI_SET_DEEP_SHAP_OUT_ROOT=1; _CLI_CHANGED=1
        ;;
      --n-folds)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        N_FOLDS="$1"; shift
        _CLI_CHANGED=1
        ;;
      --seed|--cv-seed)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        CV_SEED="$1"; shift
        _CLI_CHANGED=1
        ;;
      --train-seed)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        TRAIN_SEED="$1"; shift
        _CLI_CHANGED=1
        ;;
      --eval-split)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        EVAL_SPLIT="$1"; shift
        _CLI_CHANGED=1
        ;;
      --mpra-table)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        MPRA_TABLE="$(_workflow_abspath "$1")"; shift
        _CLI_CHANGED=1
        ;;
      --sequence-col)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        SEQUENCE_COL="$1"; shift
        _CLI_CHANGED=1
        ;;
      --target-col)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        TARGET_COL="$1"; shift
        _CLI_CHANGED=1
        ;;
      --id-col)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        ID_COL="$1"; shift
        _CLI_CHANGED=1
        ;;
      --split-col)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        SPLIT_COL="$1"; shift
        _CLI_CHANGED=1
        ;;
      --metadata-cols)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        METADATA_COLS="$1"; shift
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
          CV_MODEL)
            TRAIN_MODEL_OVERRIDE="${val}"
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
  else
    _resolve_cv_model_from_config
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

run_host_python() {
  if [[ "${RUNTIME_BACKEND}" == "apptainer" && -n "${APPTAINER_IMAGE}" && -f "${APPTAINER_IMAGE}" ]] \
      && command -v apptainer >/dev/null 2>&1; then
    apptainer exec --cleanenv "${APPTAINER_IMAGE}" python "$@"
  elif command -v python >/dev/null 2>&1; then
    command python "$@"
  else
    command python3 "$@"
  fi
}

export_workflow_env() {
  prepare_code_snapshot
  export RUNTIME_BACKEND APPTAINER_IMAGE TRANSCRIPTML_SRC_TAR CODE_SNAPSHOT_ENABLED
  export STAGE_SIF_TO_LSCRATCH STAGE_DATASET_TO_LSCRATCH JOB_START_JITTER_MAX
  export CONDA_ENV SHERLOCK_CONDA_ROOT TRANSCRIPTML_REPO TRANSCRIPTML_RUN_CONFIG
  export MPRA_TABLE SEQUENCE_COL TARGET_COL ID_COL SPLIT_COL METADATA_COLS MPRA_LENGTH DELIMITER
  export RUN_NAME RUN_ROOT DATASET_DIR EXP_NAME EXPERIMENTS_DIR EXP_ROOT
  export TRAIN_OUTPUT_ROOT MODEL_DIR EVAL_DIR GENERATED_TRAIN_CONFIG
  export CV_ROOT SWEEP_ROOT SWEEP_TABLE INTERPRET_ROOT INTERPRET_DATASET_DIR ISM_OUT_DIR LEGNET_CHECKPOINT
  export BASE_TRAIN_CONFIG EVAL_SPLIT TRAIN_SEED N_FOLDS CV_SEED CV_VAL_OFFSET CV_MODEL TRAIN_MODEL_OVERRIDE
  export SWEEP_SKIP_COMPLETED SWEEP_MAX_CONCURRENT RUN_SPLIT_EVALUATION
  export PRED_BATCH_SIZE MUTATION_BATCH_SIZE DEVICE
  export DEEP_SHAP_REFERENCE_STRATEGY DEEP_SHAP_RUN_NAME DEEP_SHAP_OUT_ROOT DEEP_SHAP_CHECKPOINT_NAME
  export DEEP_SHAP_N_REFERENCES DEEP_SHAP_SEED DEEP_SHAP_TARGET DEEP_SHAP_ATTRIBUTION_BATCH_SIZE
  export DEEP_SHAP_CONFIDENCE_THRESHOLD DEEP_SHAP_MINIMUM_EFFECT_SIZE DEEP_SHAP_ZERO_ATOL
  export DEEP_SHAP_IG_POINTS DEEP_SHAP_MAX_IG_POINTS DEEP_SHAP_CONVERGENCE_ATOL DEEP_SHAP_CONVERGENCE_RTOL
  export DEEP_SHAP_REFERENCE_PREDICTIONS_FORMAT DEEP_SHAP_SEQUENCE_SHARDS DEEP_SHAP_MAX_CONCURRENT
}

prepare_train_submission() {
  local target_root="${1:-${EXP_ROOT}}"
  if [[ ! -f "${BASE_TRAIN_CONFIG}" ]]; then
    echo "BASE_TRAIN_CONFIG does not exist: ${BASE_TRAIN_CONFIG}" >&2
    return 1
  fi
  _resolve_cv_model_from_config
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

_resolve_cv_model_from_config

stage_dataset_bundle() {
  local src_dir="$1"
  local overlay_dir="${2:-}"
  if [[ "${STAGE_DATASET_TO_LSCRATCH:-1}" != "1" || -z "${L_SCRATCH_JOB:-}" || ! -d "${L_SCRATCH_JOB}" ]]; then
    printf '%s\n' "${overlay_dir:-${src_dir}}"
    return 0
  fi
  if [[ ! -d "${src_dir}" ]]; then
    printf '%s\n' "${overlay_dir:-${src_dir}}"
    return 0
  fi

  if [[ -z "${overlay_dir}" && -L "${src_dir}/X.npy" ]]; then
    local real_x=""
    real_x="$(readlink -f "${src_dir}/X.npy" 2>/dev/null || true)"
    if [[ -n "${real_x}" && -f "${real_x}" ]]; then
      local real_base
      real_base="$(dirname "${real_x}")"
      if [[ "${real_base}" != "${src_dir}" ]]; then
        overlay_dir="${src_dir}"
        src_dir="${real_base}"
      fi
    fi
  fi

  if [[ "${src_dir}" == "${L_SCRATCH_JOB}"/* && -z "${overlay_dir}" ]]; then
    printf '%s\n' "${src_dir}"
    return 0
  fi

  local canonical_src
  canonical_src="$(cd "${src_dir}" && pwd -P)"
  local base_key
  if command -v sha256sum >/dev/null 2>&1; then
    base_key="$(printf '%s' "${canonical_src}" | sha256sum | awk '{print substr($1, 1, 16)}')"
  else
    base_key="$(printf '%s' "${canonical_src}" | tr '/ ' '__')"
  fi
  local base_dst="${L_SCRATCH_JOB}/datasets/base_${base_key}"

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
    local t0 t1
    t0="$(date +%s)"
    mkdir -p "${base_dst}"
    cp -aL "${canonical_src}/." "${base_dst}/"
    touch "${base_dst}/.staged_ok"
    t1="$(date +%s)"
    echo "[runtime] Staged dataset ${canonical_src} -> ${base_dst} ($((req_kb / 1024)) MB) in $((t1 - t0))s" >&2
  fi

  if [[ -z "${overlay_dir}" ]]; then
    printf '%s\n' "${base_dst}"
    return 0
  fi

  local canonical_overlay
  canonical_overlay="$(cd "${overlay_dir}" && pwd -P)"
  local ov_key
  if command -v sha256sum >/dev/null 2>&1; then
    ov_key="$(printf '%s' "${canonical_overlay}" | sha256sum | awk '{print substr($1, 1, 16)}')"
  else
    ov_key="$(printf '%s' "${canonical_overlay}" | tr '/ ' '__')"
  fi
  local ov_dst="${L_SCRATCH_JOB}/datasets/overlay_${ov_key}"
  rm -rf "${ov_dst}"
  mkdir -p "${ov_dst}"
  local item base_name
  for item in "${base_dst}"/*; do
    [[ -e "${item}" ]] || continue
    base_name="$(basename "${item}")"
    [[ "${base_name}" == ".staged_ok" || "${base_name}" == "splits.json" ]] && continue
    ln -s "${item}" "${ov_dst}/${base_name}"
  done
  for item in "${canonical_overlay}"/*; do
    [[ -e "${item}" || -L "${item}" ]] || continue
    base_name="$(basename "${item}")"
    if [[ "${base_name}" == "splits.json" || ! -e "${ov_dst}/${base_name}" ]]; then
      cp -aL "${item}" "${ov_dst}/${base_name}"
    fi
  done
  printf '%s\n' "${ov_dst}"
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
      cp -f "${APPTAINER_IMAGE}" "${_TML_STAGED_SIF}"
    fi
  else
    _TML_STAGED_SIF="${APPTAINER_IMAGE}"
  fi
  t1="$(date +%s)"

  local code_root="${_TML_RUNTIME_ROOT}/transcriptml_src"
  if [[ ! -d "${code_root}/src/transcriptml" ]]; then
    rm -rf "${code_root}"
    mkdir -p "${code_root}"
    if [[ -n "${TRANSCRIPTML_SRC_TAR}" && -f "${TRANSCRIPTML_SRC_TAR}" ]]; then
      tar -C "${code_root}" -xzf "${TRANSCRIPTML_SRC_TAR}"
    elif [[ -n "${TRANSCRIPTML_REPO}" && -d "${TRANSCRIPTML_REPO}/src/transcriptml" ]]; then
      tar -C "${TRANSCRIPTML_REPO}" \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='*.pyo' \
        --exclude='*.egg-info' \
        -cf - src pyproject.toml | tar -C "${code_root}" -xf -
    else
      echo "Could not locate TranscriptML source (TRANSCRIPTML_SRC_TAR and TRANSCRIPTML_REPO are unset or invalid)." >&2
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
  _resolve_cv_model_from_config
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
    if [[ ! -d "${TRANSCRIPTML_REPO}" ]]; then
      echo "TRANSCRIPTML_REPO does not exist: ${TRANSCRIPTML_REPO}" >&2
      return 1
    fi
    cd "${TRANSCRIPTML_REPO}"
    if [[ -d "${TRANSCRIPTML_REPO}/src" ]]; then
      export PYTHONPATH="${TRANSCRIPTML_REPO}/src"
    fi
  elif ! command -v transcriptml >/dev/null 2>&1; then
    echo "TRANSCRIPTML_REPO is unset and transcriptml is not on PATH after conda activation." >&2
    echo "Set TRANSCRIPTML_REPO in the copied mpra_config.sh, or install TranscriptML in ${CONDA_ENV}." >&2
    return 1
  fi
  _resolve_cv_model_from_config
}
