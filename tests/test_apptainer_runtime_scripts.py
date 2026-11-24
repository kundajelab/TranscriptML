"""Tests for Apptainer runtime wrapping, code snapshots, dataset staging, and array caps."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def _base_env() -> dict[str, str]:
    drop_prefixes = (
        "DEEP_SHAP_",
        "REGION_",
        "MOTIF_",
        "SLURM_",
        "TRANSCRIPTML_",
        "EVAL_",
        "L_SCRATCH",
        "APPTAINER",
        "RUNTIME_",
        "STAGE_",
        "CODE_SNAPSHOT",
        "JOB_START_JITTER",
        "CODON_ISM_",
        "SWEEP_",
        "CUDA_",
        "NVIDIA_",
    )
    drop_keys = {
        "RUN_NAME",
        "RUN_ROOT",
        "DATASET_DIR",
        "EXP_NAME",
        "EXPERIMENTS_DIR",
        "EXP_ROOT",
        "CV_ROOT",
        "INTERPRET_ROOT",
        "INTERPRET_DATASET_DIR",
        "N_FOLDS",
        "FOLD",
        "SHARD",
        "DEVICE",
        "PYTHONPATH",
    }
    return {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(drop_prefixes) and k not in drop_keys
    }


def test_prepare_code_snapshot_creates_and_reuses_content_addressed_tarball(tmp_path: Path):
    run_root = tmp_path / "run_root"
    env = _base_env()
    env.update(
        RUN_ROOT=str(run_root),
        EXP_NAME="snap_exp",
        TRANSCRIPTML_REPO=str(REPO_ROOT),
        RUNTIME_BACKEND="apptainer",
        CODE_SNAPSHOT_ENABLED="1",
    )
    cmd = f"""
set -euo pipefail
source {shlex.quote(str(SCRIPTS / 'sherlock_config.sh'))}
export_workflow_env
printf 'FIRST=%s\\n' "${{TRANSCRIPTML_SRC_TAR}}"
TRANSCRIPTML_SRC_TAR=""
export_workflow_env
printf 'SECOND=%s\\n' "${{TRANSCRIPTML_SRC_TAR}}"
"""
    res = subprocess.run(["bash", "-c", cmd], env=env, text=True, capture_output=True)
    assert res.returncode == 0, res.stderr
    vals = dict(line.split("=", 1) for line in res.stdout.splitlines() if "=" in line)
    first = Path(vals["FIRST"])
    second = Path(vals["SECOND"])
    assert first == second
    assert first.is_file()
    assert first.parent == run_root / "experiments" / "snap_exp" / "code_snapshot"

    with tarfile.open(first, "r:gz") as tf:
        names = tf.getnames()
    assert "pyproject.toml" in names
    assert "src/transcriptml/cli/main.py" in names
    assert not any("__pycache__" in n for n in names)


def test_stage_dataset_bundle_deduplicates_and_overlays_fold_splits(tmp_path: Path):
    lscratch = tmp_path / "lscratch"
    lscratch.mkdir()
    src_ds = tmp_path / "shared_dataset"
    src_ds.mkdir()
    (src_ds / "X.npy").write_bytes(b"FAKE_X_DATA")
    (src_ds / "y.npy").write_bytes(b"FAKE_Y_DATA")
    (src_ds / "schema.json").write_text("{}", encoding="utf-8")

    fold0_ds = tmp_path / "fold0" / "dataset"
    fold0_ds.mkdir(parents=True)
    (fold0_ds / "X.npy").symlink_to(src_ds / "X.npy")
    (fold0_ds / "y.npy").symlink_to(src_ds / "y.npy")
    (fold0_ds / "schema.json").symlink_to(src_ds / "schema.json")
    (fold0_ds / "splits.json").write_text('{"train": [0], "val": [1], "test": [2]}', encoding="utf-8")

    env = _base_env()
    env.update(
        RUN_ROOT=str(tmp_path / "run"),
        L_SCRATCH_JOB=str(lscratch),
        STAGE_DATASET_TO_LSCRATCH="1",
    )
    cmd = f"""
set -euo pipefail
source {shlex.quote(str(SCRIPTS / 'sherlock_config.sh'))}
S0="$(stage_dataset_bundle {shlex.quote(str(src_ds))} {shlex.quote(str(fold0_ds))})"
S1="$(stage_dataset_bundle {shlex.quote(str(fold0_ds))})"
printf 'S0=%s\\nS1=%s\\n' "$S0" "$S1"
"""
    res = subprocess.run(["bash", "-c", cmd], env=env, text=True, capture_output=True)
    assert res.returncode == 0, res.stderr
    vals = dict(line.split("=", 1) for line in res.stdout.splitlines() if "=" in line)
    staged0 = Path(vals["S0"])
    staged1 = Path(vals["S1"])
    assert staged0 == staged1
    assert str(staged0).startswith(str(lscratch))
    # X.npy inside overlay symlinks to the local NVMe base copy (not original src_ds)
    assert staged0.joinpath("X.npy").resolve().parent.parent == lscratch / "datasets"
    assert staged0.joinpath("X.npy").read_bytes() == b"FAKE_X_DATA"
    assert '"test": [2]' in staged0.joinpath("splits.json").read_text(encoding="utf-8")
    # Persistent fold0 symlink still points to original shared_dataset
    assert (fold0_ds / "X.npy").is_symlink()
    assert (fold0_ds / "X.npy").resolve() == (src_ds / "X.npy").resolve()
    # Only one copy event should have occurred
    assert res.stderr.count("Staged dataset") == 1


def test_setup_apptainer_env_scrubs_host_pythonpath_and_forwards_gpu_and_caches(tmp_path: Path):
    lscratch = tmp_path / "lscratch"
    lscratch.mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_apptainer = fake_bin / "apptainer"
    fake_apptainer.write_text("#!/bin/bash\nprintf 'ARG:%s\\n' \"$@\"\n", encoding="utf-8")
    fake_apptainer.chmod(0o755)

    fake_sif = tmp_path / "transcriptml-deps_v1.sif"
    fake_sif.write_bytes(b"FAKE_SIF")

    env = _base_env()
    env.update(
        PATH=f"{fake_bin}:{env.get('PATH', '')}",
        PYTHONPATH="/oak/stanford/groups/akundaje/isvock/basenji/bin",
        RUN_ROOT=str(tmp_path / "run"),
        L_SCRATCH_JOB=str(lscratch),
        SLURM_JOB_ID="424242",
        SLURM_CPUS_PER_TASK="4",
        CUDA_VISIBLE_DEVICES="3",
        DEVICE="cuda",
        RUNTIME_BACKEND="apptainer",
        APPTAINER_IMAGE=str(fake_sif),
        TRANSCRIPTML_REPO=str(REPO_ROOT),
        JOB_START_JITTER_MAX="0",
    )
    cmd = f"""
set -euo pipefail
source {shlex.quote(str(SCRIPTS / 'sherlock_config.sh'))}
setup_transcriptml_env
transcriptml --help
"""
    res = subprocess.run(["bash", "-c", cmd], env=env, text=True, capture_output=True)
    assert res.returncode == 0, res.stderr
    args = [line[4:] for line in res.stdout.splitlines() if line.startswith("ARG:")]
    assert args[:3] == ["exec", "--cleanenv", "--nv"]
    assert f"PYTHONPATH={lscratch}/tml_runtime/transcriptml_src/src" in args
    assert not any("basenji" in a for a in args)
    assert "CUDA_VISIBLE_DEVICES=3" in args
    assert f"MPLCONFIGDIR={lscratch}/tml_runtime/cache/matplotlib" in args
    assert f"NUMBA_CACHE_DIR={lscratch}/tml_runtime/cache/numba" in args
    assert "PYTHONNOUSERSITE=1" in args
    assert "PYTHONFAULTHANDLER=1" in args
    assert str(lscratch / "tml_runtime" / "transcriptml-deps_v1.sif") in args
    assert args[-4:] == ["python", "-m", "transcriptml.cli.main", "--help"]


def test_unified_codon_ism_shard_submit_and_worker(tmp_path: Path):
    work = tmp_path / "scripts"
    work.mkdir()
    for name in ("submit_all_codon_ism_shard_by_fold.sh", "all_codon_ism_shard_by_fold.sh"):
        shutil.copyfile(SCRIPTS / name, work / name)
        subprocess.run(["bash", "-n", str(work / name)], check=True)

    (work / "sherlock_config.sh").write_text(
        f"RUN_ROOT={shlex.quote(str(tmp_path / 'run'))}\n"
        f"source {shlex.quote(str(SCRIPTS / 'sherlock_config.sh'))}\n"
        "setup_transcriptml_env() { :; }\n"
        'transcriptml() { printf "%s\\n" "$@"; }\n'
        'sbatch() { printf "SBATCH:%s\\n" "$@"; }\n',
        encoding="utf-8",
    )

    env = _base_env()
    env.update(N_FOLDS="4", CODON_ISM_SHARDS_PER_FOLD="5", CODON_ISM_MAX_CONCURRENT="12")
    sub = subprocess.run(
        ["bash", str(work / "submit_all_codon_ism_shard_by_fold.sh")],
        env=env,
        text=True,
        capture_output=True,
    )
    assert sub.returncode == 0, sub.stderr
    lines = sub.stdout.splitlines()
    assert lines[0] == "SBATCH:--array=0-19%12"
    assert lines[-1].endswith("/all_codon_ism_shard_by_fold.sh")

    # Worker maps SLURM_ARRAY_TASK_ID=13 -> fold=2, shard=3
    worker_env = {
        **env,
        "SLURM_ARRAY_TASK_ID": "13",
        "CV_ROOT": "/cv",
        "INTERPRET_ROOT": "/out",
        "INTERPRET_DATASET_DIR": "/ds",
    }
    w = subprocess.run(
        ["bash", str(work / "all_codon_ism_shard_by_fold.sh")],
        env=worker_env,
        text=True,
        capture_output=True,
    )
    assert w.returncode == 0, w.stderr
    wargs = w.stdout.splitlines()
    assert wargs[wargs.index("--checkpoint") + 1] == "/cv/fold2/model/best.pt"
    assert wargs[wargs.index("--out-dir") + 1] == "/out/all_codon_ism/fold2/shard3"
    assert wargs[wargs.index("--sequence-shard-index") + 1] == "3"
    assert wargs[wargs.index("--sequence-shards") + 1] == "5"


def test_extract_model_name_uses_system_python_not_path_conda(tmp_path: Path):
    if not Path("/usr/bin/python3").is_file():
        pytest.skip("Requires /usr/bin/python3")
    fake_bin = tmp_path / "fake_conda_bin"
    fake_bin.mkdir()
    for name in ("python", "python3"):
        p = fake_bin / name
        p.write_text("#!/bin/bash\necho SHOULD_NOT_RUN >&2\nexit 99\n", encoding="utf-8")
        p.chmod(0o755)
    cfg = tmp_path / "train.json"
    cfg.write_text('{"model": {"name": "saluki_cnn"}}', encoding="utf-8")

    env = _base_env()
    env["PATH"] = f"{fake_bin}:{env.get('PATH', '')}"
    cmd = f"""
set -euo pipefail
source {shlex.quote(str(SCRIPTS / 'sherlock_config.sh'))}
_extract_model_name_from_config {shlex.quote(str(cfg))}
"""
    res = subprocess.run(["bash", "-c", cmd], env=env, text=True, capture_output=True)
    assert res.returncode == 0, res.stderr
    assert "SHOULD_NOT_RUN" not in res.stderr
    assert res.stdout.strip() == "saluki_cnn"


def test_stage_dataset_bundle_fails_hard_on_copy_error_without_creating_marker(tmp_path: Path):
    lscratch = tmp_path / "lscratch"
    lscratch.mkdir()
    src_ds = tmp_path / "bad_dataset"
    src_ds.mkdir()
    (src_ds / "X.npy").write_bytes(b"DATA")

    # Inject a failing `cp` wrapper to simulate a mid-copy I/O failure.
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_cp = fake_bin / "cp"
    fake_cp.write_text("#!/bin/bash\nexit 1\n", encoding="utf-8")
    fake_cp.chmod(0o755)

    env = _base_env()
    env.update(
        PATH=f"{fake_bin}:{env.get('PATH', '')}",
        RUN_ROOT=str(tmp_path / "run"),
        L_SCRATCH_JOB=str(lscratch),
        STAGE_DATASET_TO_LSCRATCH="1",
    )
    cmd = f"""
set -euo pipefail
source {shlex.quote(str(SCRIPTS / 'sherlock_config.sh'))}
STAGED="$(stage_dataset_bundle {shlex.quote(str(src_ds))})"
echo "UNREACHABLE:$STAGED"
"""
    res = subprocess.run(["bash", "-c", cmd], env=env, text=True, capture_output=True)
    assert res.returncode != 0
    assert "UNREACHABLE:" not in res.stdout
    assert not list((lscratch / "datasets").glob("*/.staged_ok"))


def test_setup_apptainer_env_rejects_missing_explicit_tar_and_enforces_require_snapshot(tmp_path: Path):
    lscratch = tmp_path / "lscratch"
    lscratch.mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_apptainer = fake_bin / "apptainer"
    fake_apptainer.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    fake_apptainer.chmod(0o755)
    fake_sif = tmp_path / "deps.sif"
    fake_sif.write_bytes(b"SIF")

    base = _base_env()
    base.update(
        PATH=f"{fake_bin}:{base.get('PATH', '')}",
        RUN_ROOT=str(tmp_path / "run"),
        L_SCRATCH_JOB=str(lscratch),
        RUNTIME_BACKEND="apptainer",
        APPTAINER_IMAGE=str(fake_sif),
        TRANSCRIPTML_REPO=str(REPO_ROOT),
        JOB_START_JITTER_MAX="0",
    )

    # 1. Explicit TRANSCRIPTML_SRC_TAR that does not exist must fail (never fall back to repo)
    env_missing = {**base, "TRANSCRIPTML_SRC_TAR": str(tmp_path / "nonexistent.tar.gz")}
    res1 = subprocess.run(
        ["bash", "-c", f"set -euo pipefail; source {shlex.quote(str(SCRIPTS / 'sherlock_config.sh'))}; setup_transcriptml_env"],
        env=env_missing,
        text=True,
        capture_output=True,
    )
    assert res1.returncode != 0
    assert "does not exist" in res1.stderr

    # 2. REQUIRE_CODE_SNAPSHOT=1 with empty TRANSCRIPTML_SRC_TAR must fail
    env_req = {**base, "TRANSCRIPTML_SRC_TAR": "", "REQUIRE_CODE_SNAPSHOT": "1"}
    res2 = subprocess.run(
        ["bash", "-c", f"set -euo pipefail; source {shlex.quote(str(SCRIPTS / 'sherlock_config.sh'))}; setup_transcriptml_env"],
        env=env_req,
        text=True,
        capture_output=True,
    )
    assert res2.returncode != 0
    assert "REQUIRE_CODE_SNAPSHOT=1" in res2.stderr

