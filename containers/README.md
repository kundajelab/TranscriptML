# TranscriptML Sherlock Apptainer Runtime

This directory contains the Apptainer container definition and helper scripts used to run TranscriptML on Stanford Sherlock without runtime Python/Conda imports from `$OAK`.

## Files

- `Apptainer.def`: Container definition built on `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` plus all `TranscriptML` runtime and optional dependencies (`analysis`, `rbpnet`, `dev`). `TranscriptML` itself is intentionally **not** baked into the image so code edits do not require rebuilding the `.sif`.
- `requirements.txt`: Python dependencies installed inside the image.
- `build_sherlock_sif.sh`: Builds a versioned `.sif` in node-local scratch (`$L_SCRATCH_JOB`) and copies it atomically to `$OAK/isvock/containers/transcriptml-deps_<version>.sif`.
- `package_transcriptml.sh`: Packages `src/` and `pyproject.toml` into a compact (`<1 MB`) versioned tarball for pinned release runs (`TRANSCRIPTML_SRC_TAR`).
- `smoke_test.sh`: GPU validation job (`sbatch containers/smoke_test.sh`) that stages the `.sif` and repo to `$L_SCRATCH_JOB`, checks CUDA + core imports, times CLI startup, and runs `pytest`.

## Rebuilding the Container (Only Needed When Dependencies Change)

```bash
cd /oak/stanford/groups/akundaje/isvock/software/TranscriptML
IMAGE_VERSION=v2 sbatch containers/build_sherlock_sif.sh
```

Then update `APPTAINER_IMAGE` in your `sherlock_config.sh` (or pass `-s APPTAINER_IMAGE=...`).

## Updating TranscriptML Code (No Container Rebuild Needed)

By default (`CODE_SNAPSHOT_ENABLED=1`), every `submit_*.sh` script automatically creates a content-addressed `<1 MB` snapshot tarball of `TranscriptML/src` under `${EXP_ROOT}/code_snapshot/` and stages it to `$L_SCRATCH_JOB` at job startup.

For strictly pinned release runs across multiple experiments, cut a named tarball once:

```bash
bash containers/package_transcriptml.sh \
  /oak/stanford/groups/akundaje/isvock/containers/releases/transcriptml_v0.1.0.tar.gz
```

and set `TRANSCRIPTML_SRC_TAR` in `sherlock_config.sh` (or pass `-s TRANSCRIPTML_SRC_TAR=...`).

## Falling Back to Legacy Conda

Every workflow script retains the Conda backend as an opt-in fallback:

```bash
bash scripts/submit_train_eval_cv.sh -s RUNTIME_BACKEND=conda
```
