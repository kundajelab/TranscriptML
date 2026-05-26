#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/sherlock_config.sh"

sbatch --array=0-$((N_FOLDS - 1)) "${SCRIPT_DIR}/ism_by_fold.sh"
