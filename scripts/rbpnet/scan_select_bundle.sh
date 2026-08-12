#!/bin/bash
#SBATCH --partition=akundaje
#SBATCH --job-name=tml_rbp_bundle
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_output/%x_%j.out
#SBATCH --error=slurm_output/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  if [[ -f "${SLURM_SUBMIT_DIR}/scripts/rbpnet/rbpnet_config.sh" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/scripts/rbpnet"
  elif [[ -f "${SLURM_SUBMIT_DIR}/rbpnet_config.sh" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
  fi
fi
source "${SCRIPT_DIR}/rbpnet_config.sh"
setup_transcriptml_env

if [[ ! -f "${PROCESSED_DIR}/signals.h5" ]]; then
  echo "Missing ${PROCESSED_DIR}/signals.h5; run preprocess.sh first." >&2
  exit 1
fi

stride="${WINDOW_STRIDE}"
if [[ -z "${stride}" ]]; then
  if [[ "${SELECTION_STRATEGY}" == "original_rbpnet" ]]; then
    stride=1
  else
    stride=50
  fi
fi

overwrite_args=()
if [[ "${OVERWRITE}" == "1" ]]; then
  overwrite_args+=(--overwrite)
fi

transcriptml rbpnet scan-windows \
  --processed-dir "${PROCESSED_DIR}" \
  --window-size "${WINDOW_SIZE}" \
  --stride "${stride}" \
  --min-sminput-tpm "${MIN_SMINPUT_TPM}" \
  --output-prefix "${WINDOW_PREFIX}" \
  "${overwrite_args[@]}"

selection_args=(
  --processed-dir "${PROCESSED_DIR}"
  --windows "${WINDOW_PREFIX}.parquet"
  --strategy "${SELECTION_STRATEGY}"
  --min-sminput-tpm "${MIN_SMINPUT_TPM}"
  --output-prefix "${SELECTION_PREFIX}"
)
if [[ -n "${REGION_TYPES}" ]]; then
  selection_args+=(--region-types "${REGION_TYPES}")
fi
case "${SELECTION_STRATEGY}" in
  original_rbpnet)
    selection_args+=(--poisson-null "${POISSON_NULL}")
    ;;
  broad_coverage)
    selection_args+=(
      --min-total-count "${MIN_TOTAL_COUNT}"
      --min-ip-count "${MIN_IP_COUNT}"
      --min-sminput-count "${MIN_SMINPUT_COUNT}"
      --replicate-mode "${REPLICATE_MODE}"
    )
    ;;
  peak_gray_negative)
    selection_args+=(
      --min-total-count "${MIN_TOTAL_COUNT}"
      --min-ip-count "${MIN_IP_COUNT}"
      --min-sminput-count "${MIN_SMINPUT_COUNT}"
      --peak-fdr "${PEAK_FDR}"
      --peak-min-log2-ratio "${PEAK_MIN_LOG2_RATIO}"
      --negative-fdr "${NEGATIVE_FDR}"
      --negative-max-log2-ratio "${NEGATIVE_MAX_LOG2_RATIO}"
      --stitch-gap "${STITCH_GAP}"
    )
    ;;
  *)
    echo "Unknown SELECTION_STRATEGY: ${SELECTION_STRATEGY}" >&2
    exit 1
    ;;
esac
selection_args+=("${overwrite_args[@]}")
transcriptml rbpnet select-regions "${selection_args[@]}"

transcriptml rbpnet make-bundle \
  --processed-dir "${PROCESSED_DIR}" \
  --selection-manifest "${SELECTION_PREFIX}.parquet" \
  --output-dir "${BUNDLE_DIR}" \
  --input-length "${INPUT_LENGTH}" \
  --profile-length "${PROFILE_LENGTH}" \
  --max-jitter "${MAX_JITTER}" \
  --transcript-end-policy "${TRANSCRIPT_END_POLICY}" \
  "${overwrite_args[@]}"
