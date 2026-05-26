#!/bin/bash

# Edit this file for each new TranscriptML dataset/model run.

_TRANSCRIPTML_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSCRIPTML_REPO="${TRANSCRIPTML_REPO:-$(cd "${_TRANSCRIPTML_SCRIPT_DIR}/.." && pwd)}"
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

# Motifs used by the ablation scripts: label|motif.
MOTIF_ABLATION_SPECS=(
  "PRE|UGUANAUA"
  "ARE_nonamer|UUAUUUAUU"
  "GGACU|GGACU"
  "let7_7mer_m8|CUACCUC"
  "miR16_7mer_m8|UGCUGCU"
)

# Motif pairs used by the epistasis scripts: label|motif1|motif2.
# Leave motif2 empty to test pairs of the same motif.
MOTIF_EPISTASIS_SPECS=(
  "PRE_PRE|UGUANAUA|"
  "ARE_ARE|UUAUUUAUU|"
  "GGACU_GGACU|GGACU|"
  "PRE_ARE|UGUANAUA|UUAUUUAUU"
  "PRE_GGACU|UGUANAUA|GGACU"
  "ARE_GGACU|UUAUUUAUU|GGACU"
  "let7_miR16|CUACCUC|UGCUGCU"
)

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
