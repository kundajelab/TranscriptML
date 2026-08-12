#!/bin/bash

# Shared Sherlock defaults for the eCLIP -> RBPNet chromosome-CV workflow.
# Copy scripts/rbpnet to a writable run directory and edit this file there.

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

RUN_NAME="${RUN_NAME:-RBPNet_eCLIP}"
RUN_ROOT="${RUN_ROOT:-/scratch/users/${USER:-user}/TranscriptML/${RUN_NAME}}"
PROCESSED_DIR="${PROCESSED_DIR:-${RUN_ROOT}/processed/eclip}"
WINDOW_PREFIX="${WINDOW_PREFIX:-${RUN_ROOT}/windows/windows_100nt}"
SELECTION_PREFIX="${SELECTION_PREFIX:-${RUN_ROOT}/selection/selected_regions}"
BUNDLE_DIR="${BUNDLE_DIR:-${RUN_ROOT}/data/rbpnet}"
CV_PLAN="${CV_PLAN:-${RUN_ROOT}/cv/cv5_chromosomes.json}"
CV_ROOT="${CV_ROOT:-${RUN_ROOT}/cv}"

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
# Optional comma-separated exact annotations, e.g. "3putr" or "cds,3putr".
REGION_TYPES="${REGION_TYPES:-}"
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
