#!/bin/bash
#SBATCH --partition=akundaje
#SBATCH --job-name=tml_region_ablate
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -C GPU_MEM:48GB
#SBATCH --output=slurm_output/%x_%A_%a.out
#SBATCH --error=slurm_output/%x_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  if [[ -f "${SLURM_SUBMIT_DIR}/scripts/sherlock_config.sh" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/scripts"
  elif [[ -f "${SLURM_SUBMIT_DIR}/sherlock_config.sh" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
  fi
fi
source "${SCRIPT_DIR}/sherlock_config.sh"
setup_transcriptml_env

FOLD="${SLURM_ARRAY_TASK_ID}"
CHECKPOINT="${CV_ROOT}/fold${FOLD}/model/best.pt"
OUT_DIR="${INTERPRET_ROOT}/region_ablation/fold${FOLD}"

REGION_ARGS=(
  --checkpoint "${CHECKPOINT}"
  --dataset "${INTERPRET_DATASET_DIR}"
  --out-dir "${OUT_DIR}"
  --n-ablations "${REGION_ABLATION_N}"
  --junction-counts "${REGION_JUNCTION_COUNTS}"
  --junction-min-spacing "${REGION_JUNCTION_MIN_SPACING}"
  --seed "${REGION_ABLATION_SEED}"
  --device "${DEVICE}"
  --batch-size "${PRED_BATCH_SIZE}"
  --mutation-batch-size "${MUTATION_BATCH_SIZE}"
)

for spec in "${REGION_ABLATION_N_FOR[@]}"; do
  REGION_ARGS+=(--n-ablations-for "${spec}")
done
for family in "${REGION_ABLATION_DISABLED[@]}"; do
  REGION_ARGS+=("--disable-${family//_/-}")
done

transcriptml region-ablation "${REGION_ARGS[@]}"
