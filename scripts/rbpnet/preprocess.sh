#!/bin/bash
#SBATCH --partition=akundaje
#SBATCH --job-name=tml_rbp_preprocess
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
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

if [[ -z "${GENOME_FASTA}" || -z "${GTF}" || -z "${SMINPUT_BAM}" || ${#IP_BAMS[@]} -eq 0 ]]; then
  echo "Set GENOME_FASTA, GTF, SMINPUT_BAM, and at least one IP_BAMS entry in rbpnet_config.sh." >&2
  exit 1
fi

command=(
  transcriptml rbpnet preprocess
  --genome-fasta "${GENOME_FASTA}"
  --gtf "${GTF}"
  --sminput-bam "${SMINPUT_BAM}"
  --coordinate-space "${COORDINATE_SPACE}"
  --read1-rna-strand "${READ1_RNA_STRAND}"
  --min-mapq "${MIN_MAPQ}"
  --signal-compression "${SIGNAL_COMPRESSION}"
  --output-dir "${PROCESSED_DIR}"
)
for bam in "${IP_BAMS[@]}"; do
  command+=(--ip-bam "${bam}")
done
if [[ "${SIGNAL_COMPRESSION}" == "gzip" ]]; then
  command+=(--signal-compression-level "${SIGNAL_COMPRESSION_LEVEL}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  command+=(--overwrite)
fi
"${command[@]}"
