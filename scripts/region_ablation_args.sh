#!/bin/bash
# Sourced by the all-family and custom fold jobs after sherlock_config.sh.
build_region_ablation_args() {
  local mode="$1" spec family count
  if [[ "${mode}" == all ]]; then
    if (( ${#REGION_ABLATION_DISABLED[@]} )); then
      echo 'Use region_ablation_custom_by_fold.sh to disable families.' >&2
      return 1
    fi
    for spec in "${REGION_ABLATION_N_FOR[@]}"; do
      count="${spec#*=}"
      count="${count//[[:space:]]/}"
      if [[ "${count}" =~ ^[+-]?0+$ ]]; then
        echo 'Use region_ablation_custom_by_fold.sh for zero replicate overrides.' >&2
        return 1
      fi
    done
    OUT_DIR="${INTERPRET_ROOT}/region_ablation/fold${FOLD}"
    family=all
  else
    if [[ ! "${REGION_ABLATION_CUSTOM_RUN_NAME}" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
      echo 'REGION_ABLATION_CUSTOM_RUN_NAME must be a simple directory name.' >&2
      return 1
    fi
    OUT_DIR="${INTERPRET_ROOT}/region_ablation_custom/${REGION_ABLATION_CUSTOM_RUN_NAME}/fold${FOLD}"
    family="${REGION_ABLATION_FAMILIES}"
  fi
  REGION_ARGS=(
    --checkpoint "${CV_ROOT}/fold${FOLD}/model/best.pt"
    --dataset "${INTERPRET_DATASET_DIR}"
    --out-dir "${OUT_DIR}"
    --families "${family}"
    --n-ablations "${REGION_ABLATION_N}"
    --junction-counts "${REGION_JUNCTION_COUNTS}"
    --junction-min-spacing "${REGION_JUNCTION_MIN_SPACING}"
    --seed "${REGION_ABLATION_SEED}"
    --device "${DEVICE}"
    --batch-size "${PRED_BATCH_SIZE}"
    --mutation-batch-size "${MUTATION_BATCH_SIZE}"
  )
  if [[ -n "${REGION_JUNCTION_DENSITIES}" ]]; then
    REGION_ARGS+=(--junction-densities "${REGION_JUNCTION_DENSITIES}")
  fi
  if [[ -n "${REGION_JUNCTION_DENSITY_QUANTILES}" ]]; then
    REGION_ARGS+=(--junction-density-quantiles "${REGION_JUNCTION_DENSITY_QUANTILES}")
  fi
  for spec in "${REGION_ABLATION_N_FOR[@]}"; do
    REGION_ARGS+=(--n-ablations-for "${spec}")
  done
  for family in "${REGION_ABLATION_DISABLED[@]}"; do
    REGION_ARGS+=("--disable-${family//_/-}")
  done
}
