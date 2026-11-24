#!/bin/bash
#SBATCH --job-name=build_tml_sif
#SBATCH --partition=akundaje,normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=build_tml_sif_%j.out
#SBATCH --error=build_tml_sif_%j.err

# Build the versioned TranscriptML dependencies Apptainer (.sif) image on Sherlock.
# Usage:
#   sbatch containers/build_sherlock_sif.sh
#   IMAGE_VERSION=v2 sbatch containers/build_sherlock_sif.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${CONTAINERS_DIR:-}" && -f "${CONTAINERS_DIR}/Apptainer.def" ]]; then
  SCRIPT_DIR="${CONTAINERS_DIR}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/containers/Apptainer.def" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/containers"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/Apptainer.def" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
elif [[ -n "${OAK:-}" && -f "${OAK}/isvock/software/TranscriptML/containers/Apptainer.def" ]]; then
  SCRIPT_DIR="${OAK}/isvock/software/TranscriptML/containers"
fi

IMAGE_VERSION="${IMAGE_VERSION:-v1}"
OUT_DIR="${OUT_DIR:-${OAK}/isvock/containers}"
OUT_SIF="${OUT_SIF:-${OUT_DIR}/transcriptml-deps_${IMAGE_VERSION}.sif}"
DEF_FILE="${SCRIPT_DIR}/Apptainer.def"

mkdir -p "${OUT_DIR}"

# Keep Apptainer cache & temp build files off $HOME (15 GB quota).
BUILD_SCRATCH="${L_SCRATCH_JOB:-${SCRATCH}/tmp/apptainer_build_${SLURM_JOB_ID:-$$}}"
mkdir -p "${BUILD_SCRATCH}/cache" "${BUILD_SCRATCH}/tmp"
export APPTAINER_CACHEDIR="${BUILD_SCRATCH}/cache"
export APPTAINER_TMPDIR="${BUILD_SCRATCH}/tmp"
export TMPDIR="${BUILD_SCRATCH}/tmp"

LOCAL_SIF="${BUILD_SCRATCH}/transcriptml-deps_${IMAGE_VERSION}.sif"

echo "[build_sif] Host: $(hostname)"
echo "[build_sif] Apptainer: $(apptainer --version)"
echo "[build_sif] Def: ${DEF_FILE}"
echo "[build_sif] Build scratch: ${BUILD_SCRATCH}"
echo "[build_sif] Target: ${OUT_SIF}"

cd "${SCRIPT_DIR}"

# Build inside node-local scratch first, then atomically move to $OAK.
# Two-stage (sandbox -> SIF) keeps peak RAM low and lets us cap mksquashfs
# threads/RAM to the Slurm allocation. Use zstd for fast builds & runtime reads.
MKSQUASHFS_PROCS="${SLURM_CPUS_PER_TASK:-4}"
MKSQUASHFS_MEM="${MKSQUASHFS_MEM:-1G}"
MKSQUASHFS_COMP="${MKSQUASHFS_COMP:--comp zstd -Xcompression-level 3}"
SANDBOX_DIR="${BUILD_SCRATCH}/sandbox_${IMAGE_VERSION}"
rm -rf "${SANDBOX_DIR}"

apptainer build --force --sandbox "${SANDBOX_DIR}" "${DEF_FILE}"
apptainer build --force \
  --mksquashfs-args "-processors ${MKSQUASHFS_PROCS} -mem ${MKSQUASHFS_MEM} ${MKSQUASHFS_COMP}" \
  "${LOCAL_SIF}" "${SANDBOX_DIR}"
rm -rf "${SANDBOX_DIR}"

echo "[build_sif] Copying built image to ${OUT_SIF}..."
TMP_DEST="${OUT_SIF}.tmp.${SLURM_JOB_ID:-$$}"
cp -f "${LOCAL_SIF}" "${TMP_DEST}"
mv -f "${TMP_DEST}" "${OUT_SIF}"

echo "[build_sif] Built successfully: $(ls -lh "${OUT_SIF}")"
echo "[build_sif] Frozen package list inside image:"
apptainer exec --cleanenv "${OUT_SIF}" cat /opt/transcriptml/pip-freeze.txt | head -n 40
