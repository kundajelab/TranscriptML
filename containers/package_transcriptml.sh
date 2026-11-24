#!/bin/bash
# Create a compact, versioned TranscriptML source tarball (src/ + pyproject.toml)
# for staging into Slurm jobs on $L_SCRATCH_JOB.
#
# Usage:
#   bash containers/package_transcriptml.sh [OUTPUT_TAR_GZ]
#   bash containers/package_transcriptml.sh /oak/stanford/groups/akundaje/isvock/containers/releases/transcriptml_v0.1.0.tar.gz

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ ! -d "${REPO_ROOT}/src/transcriptml" ]]; then
  echo "Expected TranscriptML src/ under ${REPO_ROOT}" >&2
  exit 1
fi

DEFAULT_TAG="snapshot"
if command -v git >/dev/null 2>&1 && git -C "${REPO_ROOT}" rev-parse --short HEAD >/dev/null 2>&1; then
  GIT_SHA="$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
  DIRTY=""
  if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain -- src pyproject.toml 2>/dev/null || true)" ]]; then
    DIRTY="_dirty"
  fi
  DEFAULT_TAG="${GIT_SHA}${DIRTY}_$(date -u +%Y%m%dT%H%M%SZ)"
fi

OUT_TAR="${1:-${OAK:-${REPO_ROOT}}/isvock/containers/releases/transcriptml_${DEFAULT_TAG}.tar.gz}"
mkdir -p "$(dirname "${OUT_TAR}")"

TMP_TAR="${OUT_TAR}.tmp.$$"
tar -C "${REPO_ROOT}" \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='*.egg-info' \
  -czf "${TMP_TAR}" \
  src pyproject.toml

mv -f "${TMP_TAR}" "${OUT_TAR}"
printf '%s\n' "${OUT_TAR}"
