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
SALUKI_TRUNCATE_FROM="${SALUKI_TRUNCATE_FROM:-5prime}"

# ---------------------------------------------------------------------------
# User settings: output locations.
# ---------------------------------------------------------------------------
_PRE_RUN_ROOT="${RUN_ROOT:-}"
_PRE_DATASET_DIR="${DATASET_DIR:-}"
_PRE_EXP_NAME="${EXP_NAME:-}"
_PRE_EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-}"
_PRE_EXP_ROOT="${EXP_ROOT:-}"
_PRE_CV_ROOT="${CV_ROOT:-}"
_PRE_SWEEP_ROOT="${SWEEP_ROOT:-}"
_PRE_SWEEP_TABLE="${SWEEP_TABLE:-}"
_PRE_INTERPRET_ROOT="${INTERPRET_ROOT:-}"
_PRE_INTERPRET_DATASET_DIR="${INTERPRET_DATASET_DIR:-}"
_PRE_TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-}"
_PRE_MODEL_DIR="${MODEL_DIR:-}"
_PRE_EVAL_DIR="${EVAL_DIR:-}"
_PRE_GENERATED_TRAIN_CONFIG="${GENERATED_TRAIN_CONFIG:-}"
_PRE_DEEP_SHAP_RUN_NAME="${DEEP_SHAP_RUN_NAME:-}"
_PRE_DEEP_SHAP_OUT_ROOT="${DEEP_SHAP_OUT_ROOT:-}"
_PRE_CV_MODEL="${CV_MODEL:-}"
_PRE_BASE_TRAIN_CONFIG="${BASE_TRAIN_CONFIG:-}"

RUN_NAME="${RUN_NAME:-RDC_TTDB_All_SalukiExact}"
RUN_ROOT="${RUN_ROOT:-/scratch/users/${USER:-user}/RNAStability/TranscriptML/${RUN_NAME}}"
DATASET_DIR="${DATASET_DIR:-${RUN_ROOT}/data/saluki}"
EXP_NAME="${EXP_NAME:-}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-${RUN_ROOT}/experiments}"
if [[ -n "${EXP_NAME}" ]]; then
  EXP_ROOT="${EXP_ROOT:-${EXPERIMENTS_DIR}/${EXP_NAME}}"
else
  EXP_ROOT="${EXP_ROOT:-${RUN_ROOT}}"
fi
CV_ROOT="${CV_ROOT:-${EXP_ROOT}/cv10}"
SWEEP_ROOT="${SWEEP_ROOT:-${EXP_ROOT}/hparam_sweep}"
SWEEP_TABLE="${SWEEP_TABLE:-${SCRIPT_CONFIG_DIR}/saluki_hparams.tsv}"
INTERPRET_ROOT="${INTERPRET_ROOT:-${EXP_ROOT}/interpret}"
INTERPRET_DATASET_DIR="${INTERPRET_DATASET_DIR:-${DATASET_DIR}}"
TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-${EXP_ROOT}/train_eval}"
MODEL_DIR="${MODEL_DIR:-${TRAIN_OUTPUT_ROOT}/model}"
EVAL_DIR="${EVAL_DIR:-${TRAIN_OUTPUT_ROOT}/eval}"
GENERATED_TRAIN_CONFIG="${GENERATED_TRAIN_CONFIG:-${TRAIN_OUTPUT_ROOT}/train_config.json}"

# ---------------------------------------------------------------------------
# User settings: CV and runtime knobs.
# ---------------------------------------------------------------------------
N_FOLDS="${N_FOLDS:-10}"
CV_SEED="${CV_SEED:-42}"
CV_VAL_OFFSET="${CV_VAL_OFFSET:-1}"
CV_MODEL="${CV_MODEL:-saluki_exact}"
BASE_TRAIN_CONFIG="${BASE_TRAIN_CONFIG:-${SCRIPT_CONFIG_DIR}/example_train_config.json}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
TRAIN_SEED="${TRAIN_SEED:-}"
REQUIRE_SPLIT_FILE="${REQUIRE_SPLIT_FILE:-1}"
SWEEP_SKIP_COMPLETED="${SWEEP_SKIP_COMPLETED:-1}"
SWEEP_MAX_CONCURRENT="${SWEEP_MAX_CONCURRENT:-}"

# Runtime settings.
PRED_BATCH_SIZE="${PRED_BATCH_SIZE:-128}"
MUTATION_BATCH_SIZE="${MUTATION_BATCH_SIZE:-512}"
DEVICE="${DEVICE:-cuda}"

# DeepLIFT/SHAP settings. DEEP_SHAP_RUN_NAME controls the directory below
# INTERPRET_ROOT, so change it when comparing multiple configurations that use
# the same reference strategy. Set DEEP_SHAP_SEQUENCE_SHARDS above one to split
# each fold across independent SLURM array tasks.
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
DEEP_SHAP_MAX_CONCURRENT="${DEEP_SHAP_MAX_CONCURRENT:-}"
DEEP_SHAP_CDS_CHANNEL="${DEEP_SHAP_CDS_CHANNEL:-}"
DEEP_SHAP_INCLUDE_STOP_CODONS="${DEEP_SHAP_INCLUDE_STOP_CODONS:-0}"
DEEP_SHAP_RECODE_INITIATION_CODON="${DEEP_SHAP_RECODE_INITIATION_CODON:-0}"
DEEP_SHAP_RECODE_TERMINAL_STOP="${DEEP_SHAP_RECODE_TERMINAL_STOP:-0}"

# Region-ablation settings. Each FAMILY=COUNT entry overrides the universal N;
# Use the custom fold workflow for subsets or disabled families.
# Empty target lists default to the 10th percentile in TranscriptML.
REGION_JUNCTION_DENSITIES="${REGION_JUNCTION_DENSITIES:-}"
REGION_JUNCTION_DENSITY_QUANTILES="${REGION_JUNCTION_DENSITY_QUANTILES:-}"
REGION_ABLATION_FAMILIES="${REGION_ABLATION_FAMILIES:-3utr_shuffle,junction_density,cds_synonymous,3utr_random,junction_scatter}"
  # Possible values:
    # 5utr_shuffle
    # 5utr_random
    # cds_nt_shuffle
    # cds_codon_shuffle
    # cds_synonymous
    # cds_random
    # 3utr_shuffle
    # 3utr_random
    # junction_scatter
    # junction_density
REGION_ABLATION_CUSTOM_RUN_NAME="${REGION_ABLATION_CUSTOM_RUN_NAME:-custom}"
REGION_ABLATION_N="${REGION_ABLATION_N:-100}"
REGION_JUNCTION_COUNTS="${REGION_JUNCTION_COUNTS:-1,5,10,15,20,25,30,35,40,45,50}"
REGION_JUNCTION_MIN_SPACING="${REGION_JUNCTION_MIN_SPACING:-25}"
REGION_ABLATION_SEED="${REGION_ABLATION_SEED:-123}"
if ! declare -p REGION_ABLATION_N_FOR >/dev/null 2>&1; then
  REGION_ABLATION_N_FOR=()
fi
if ! declare -p REGION_ABLATION_DISABLED >/dev/null 2>&1; then
  REGION_ABLATION_DISABLED=()
fi

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

_WORKFLOW_INITIAL_PWD="${SLURM_SUBMIT_DIR:-${PWD}}"
_DEFAULT_RUN_ROOT="${RUN_ROOT}"
_DEFAULT_DATASET_DIR="${DATASET_DIR}"
_DEFAULT_EXPERIMENTS_DIR="${EXPERIMENTS_DIR}"
_DEFAULT_EXP_ROOT="${EXP_ROOT}"
_DEFAULT_CV_ROOT="${CV_ROOT}"
_DEFAULT_SWEEP_ROOT="${SWEEP_ROOT}"
_DEFAULT_SWEEP_TABLE="${SWEEP_TABLE}"
_DEFAULT_INTERPRET_ROOT="${INTERPRET_ROOT}"
_DEFAULT_INTERPRET_DATASET_DIR="${INTERPRET_DATASET_DIR}"
_DEFAULT_TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT}"
_DEFAULT_MODEL_DIR="${MODEL_DIR}"
_DEFAULT_EVAL_DIR="${EVAL_DIR}"
_DEFAULT_GENERATED_TRAIN_CONFIG="${GENERATED_TRAIN_CONFIG}"
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
_CLI_SET_CV_ROOT=0
_CLI_SET_SWEEP_ROOT=0
_CLI_SET_SWEEP_TABLE=0
_CLI_SET_INTERPRET_ROOT=0
_CLI_SET_INTERPRET_DATASET_DIR=0
_CLI_SET_TRAIN_OUTPUT_ROOT=0
_CLI_SET_MODEL_DIR=0
_CLI_SET_EVAL_DIR=0
_CLI_SET_GENERATED_TRAIN_CONFIG=0
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
    "${py_cmd}" -c '
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
    DATASET_DIR="${RUN_ROOT}/data/saluki"
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
  -d, --dataset, --dataset-dir DIR Shared dataset bundle directory (default: ${DATASET_DIR})
      --interpret-dataset-dir DIR  Dataset bundle for interpretation (default: \${DATASET_DIR})
      --cv-root DIR                CV output root (default: \${EXP_ROOT}/cv10)
      --interpret-root DIR         Interpretation output root (default: \${EXP_ROOT}/interpret)
      --train-output-root DIR      Single-split training output root (default: \${EXP_ROOT}/train_eval)
      --sweep-root DIR             Hyperparameter sweep root (default: \${EXP_ROOT}/hparam_sweep)

Model & runtime options:
  -c, --train-config FILE          Base JSON training config (default: ${BASE_TRAIN_CONFIG})
  -m, --model NAME                 Override model name (defaults to model.name in train config)
      --n-folds INT                Number of CV folds (default: ${N_FOLDS})
      --seed, --cv-seed INT        CV split seed (default: ${CV_SEED})
      --train-seed INT             Single-split training seed override
      --eval-split SPLIT           Evaluation split name (default: ${EVAL_SPLIT})
      --sweep-table FILE           Hyperparameter sweep TSV/CSV table
      --deep-shap-strategy NAME    DeepSHAP reference strategy (default: ${DEEP_SHAP_REFERENCE_STRATEGY})
      --deep-shap-run-name NAME    DeepSHAP run subdirectory name
      --deep-shap-out-root DIR     DeepSHAP output root directory
      --region-custom-run-name STR Custom region-ablation run name
      --region-families LIST       Comma-separated region-ablation families
      --motif-region REGION        Motif analysis region (5utr, cds, 3utr, or empty)
      --config, --run-config FILE  Source an additional shell configuration file
  -s, --set KEY=VALUE              Override any config variable in sherlock_config.sh (repeatable)
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
      --region-custom-run-name)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        REGION_ABLATION_CUSTOM_RUN_NAME="$1"; shift
        _CLI_CHANGED=1
        ;;
      --region-families)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        REGION_ABLATION_FAMILIES="$1"; shift
        _CLI_CHANGED=1
        ;;
      --motif-region)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        MOTIF_REGION="$1"; shift
        _CLI_CHANGED=1
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
      --gtf)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        GTF="$(_workflow_abspath "$1")"; shift
        _CLI_CHANGED=1
        ;;
      --fasta|--genome-fasta)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        FASTA="$(_workflow_abspath "$1")"; shift
        _CLI_CHANGED=1
        ;;
      --targets)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        TARGETS="$(_workflow_abspath "$1")"; shift
        _CLI_CHANGED=1
        ;;
      --target-col)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        TARGET_COL="$1"; shift
        _CLI_CHANGED=1
        ;;
      --target-id-col)
        if (( $# == 0 )); then echo "Missing value for ${arg}" >&2; return 2; fi
        TARGET_ID_COL="$1"; shift
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

export_workflow_env() {
  export CONDA_ENV SHERLOCK_CONDA_ROOT TRANSCRIPTML_REPO TRANSCRIPTML_RUN_CONFIG
  export GTF FASTA TARGETS TARGET_ID_COL TARGET_COL SPLIT_COL METADATA_COLS SALUKI_LENGTH SALUKI_TRUNCATE_FROM
  export RUN_NAME RUN_ROOT DATASET_DIR EXP_NAME EXPERIMENTS_DIR EXP_ROOT
  export CV_ROOT SWEEP_ROOT SWEEP_TABLE INTERPRET_ROOT INTERPRET_DATASET_DIR
  export TRAIN_OUTPUT_ROOT MODEL_DIR EVAL_DIR GENERATED_TRAIN_CONFIG
  export N_FOLDS CV_SEED CV_VAL_OFFSET CV_MODEL BASE_TRAIN_CONFIG TRAIN_MODEL_OVERRIDE
  export EVAL_SPLIT TRAIN_SEED REQUIRE_SPLIT_FILE SWEEP_SKIP_COMPLETED SWEEP_MAX_CONCURRENT
  export PRED_BATCH_SIZE MUTATION_BATCH_SIZE DEVICE
  export DEEP_SHAP_REFERENCE_STRATEGY DEEP_SHAP_RUN_NAME DEEP_SHAP_OUT_ROOT DEEP_SHAP_CHECKPOINT_NAME
  export DEEP_SHAP_N_REFERENCES DEEP_SHAP_SEED DEEP_SHAP_TARGET DEEP_SHAP_ATTRIBUTION_BATCH_SIZE
  export DEEP_SHAP_CONFIDENCE_THRESHOLD DEEP_SHAP_MINIMUM_EFFECT_SIZE DEEP_SHAP_ZERO_ATOL
  export DEEP_SHAP_IG_POINTS DEEP_SHAP_MAX_IG_POINTS DEEP_SHAP_CONVERGENCE_ATOL DEEP_SHAP_CONVERGENCE_RTOL
  export DEEP_SHAP_REFERENCE_PREDICTIONS_FORMAT DEEP_SHAP_SEQUENCE_SHARDS DEEP_SHAP_MAX_CONCURRENT
  export DEEP_SHAP_CDS_CHANNEL DEEP_SHAP_INCLUDE_STOP_CODONS DEEP_SHAP_RECODE_INITIATION_CODON DEEP_SHAP_RECODE_TERMINAL_STOP
  export REGION_JUNCTION_DENSITIES REGION_JUNCTION_DENSITY_QUANTILES REGION_ABLATION_FAMILIES
  export REGION_ABLATION_CUSTOM_RUN_NAME REGION_ABLATION_N REGION_JUNCTION_COUNTS REGION_JUNCTION_MIN_SPACING REGION_ABLATION_SEED
  export MOTIF_REGION N_SCRAMBLES MOTIF_STRATEGY MOTIF_SEED MAX_EPISTASIS_PAIRS
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
  _resolve_cv_model_from_config
}
