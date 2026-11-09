"""Test CLI argument parsing, experiment scoping, and backwards compatibility across Sherlock workflow scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _clean_env() -> dict[str, str]:
    drop_prefixes = (
        "DEEP_SHAP_",
        "REGION_",
        "MOTIF_",
        "SLURM_",
        "TRANSCRIPTML_",
        "EVAL_",
    )
    drop_keys = {
        "RUN_NAME",
        "RUN_ROOT",
        "DATASET_DIR",
        "BUNDLE_DIR",
        "PROCESSED_DIR",
        "WINDOW_PREFIX",
        "SELECTION_PREFIX",
        "CV_PLAN",
        "EXP_NAME",
        "EXPERIMENTS_DIR",
        "EXP_ROOT",
        "CV_ROOT",
        "SWEEP_ROOT",
        "SWEEP_TABLE",
        "INTERPRET_ROOT",
        "INTERPRET_DATASET_DIR",
        "TRAIN_OUTPUT_ROOT",
        "MODEL_DIR",
        "EVAL_DIR",
        "GENERATED_TRAIN_CONFIG",
        "ISM_OUT_DIR",
        "LEGNET_CHECKPOINT",
        "CV_MODEL",
        "BASE_TRAIN_CONFIG",
        "TRAIN_MODEL_OVERRIDE",
        "N_FOLDS",
        "FOLD",
        "SHARD",
    }
    return {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(drop_prefixes) and k not in drop_keys
    }


def test_saluki_legacy_and_multi_experiment_cli(tmp_path: Path):
    work = tmp_path / "scripts"
    work.mkdir()
    for name in (
        "submit_train_eval_cv.sh",
        "train_eval_cv_fold.sh",
        "submit_ism_by_fold.sh",
        "ism_by_fold.sh",
        "submit_deep_shap.sh",
        "run_deep_shap.sh",
    ):
        shutil.copyfile(SCRIPTS / name, work / name)
        subprocess.run(["bash", "-n", str(work / name)], check=True)

    run_root = tmp_path / "run_root"
    dataset_dir = run_root / "data" / "saluki"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "X.npy").touch()

    default_cfg = work / "example_train_config.json"
    shutil.copyfile(SCRIPTS / "example_train_config.json", default_cfg)

    cnn_cfg = tmp_path / "saluki_cnn.json"
    cnn_cfg.write_text(
        json.dumps(
            {
                "dataset": "placeholder",
                "output_dir": "placeholder",
                "model": {"name": "saluki_cnn", "params": {"filters": 32}},
            }
        ),
        encoding="utf-8",
    )

    (work / "sherlock_config.sh").write_text(
        f"RUN_ROOT={shlex.quote(str(run_root))}\n"
        f"source {shlex.quote(str(SCRIPTS / 'sherlock_config.sh'))}\n"
        "setup_transcriptml_env() { _resolve_cv_model_from_config; }\n"
        'transcriptml() { printf "%s\\n" "$@"; }\n'
        'sbatch() { printf "SBATCH:%s\\n" "$@"; printf "ENV_EXP=%s\\nENV_CV=%s\\nENV_CFG=%s\\nENV_MODEL=%s\\n" "${EXP_NAME}" "${CV_ROOT}" "${BASE_TRAIN_CONFIG}" "${CV_MODEL}"; }\n',
        encoding="utf-8",
    )

    env = _clean_env()

    # 1. Legacy single-model workflow (no --exp)
    legacy = subprocess.run(
        ["bash", str(work / "submit_train_eval_cv.sh")],
        env=env,
        text=True,
        capture_output=True,
    )
    assert legacy.returncode == 0, legacy.stderr
    lines = legacy.stdout.splitlines()
    assert lines[0] == "SBATCH:--array=0-9"
    assert f"ENV_CV={run_root}/cv10" in lines
    assert f"ENV_CFG={run_root}/cv10/config_snapshot/base_train_config.json" in lines
    assert "ENV_MODEL=saluki_exact" in lines

    # 2. Multi-experiment workflow with --exp and relative --train-config
    # (even when SLURM_SUBMIT_DIR is inherited from an Open OnDemand session)
    ood_env = {**env, "SLURM_SUBMIT_DIR": "/var/www/ood/apps/sys/dashboard"}
    exp_sub = subprocess.run(
        [
            "bash",
            "scripts/submit_train_eval_cv.sh",
            "--exp",
            "cnn_v1",
            "--train-config",
            "saluki_cnn.json",
            "--n-folds",
            "5",
        ],
        cwd=tmp_path,
        env=ood_env,
        text=True,
        capture_output=True,
    )
    assert exp_sub.returncode == 0, exp_sub.stderr
    exp_lines = exp_sub.stdout.splitlines()
    assert exp_lines[0] == "SBATCH:--array=0-4"
    expected_exp_root = run_root / "experiments" / "cnn_v1"
    expected_snapshot = expected_exp_root / "config_snapshot" / "base_train_config.json"
    assert expected_snapshot.is_file()
    assert f"ENV_EXP=cnn_v1" in exp_lines
    assert f"ENV_CV={expected_exp_root}/cv10" in exp_lines
    assert f"ENV_CFG={expected_snapshot}" in exp_lines
    assert "ENV_MODEL=saluki_cnn" in exp_lines

    # 3. Interpretation worker with --exp scopes checkpoint and output to experiment
    ckpt = expected_exp_root / "cv10" / "fold2" / "model" / "best.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.touch()
    ism_res = subprocess.run(
        ["bash", str(work / "ism_by_fold.sh"), "--exp", "cnn_v1", "2"],
        env=env,
        text=True,
        capture_output=True,
    )
    assert ism_res.returncode == 0, ism_res.stderr
    ism_args = ism_res.stdout.splitlines()
    assert ism_args[0] == "ism"
    assert ism_args[ism_args.index("--checkpoint") + 1] == str(ckpt)
    assert ism_args[ism_args.index("--dataset") + 1] == str(dataset_dir)
    assert ism_args[ism_args.index("--out-dir") + 1] == f"{expected_exp_root}/interpret/ism/fold2"


def test_mpra_and_rbpnet_experiment_scoping(tmp_path: Path):
    # Test MPRA config scoping
    mpra_work = tmp_path / "mpra"
    mpra_work.mkdir()
    shutil.copyfile(SCRIPTS / "mpra" / "ism_by_fold.sh", mpra_work / "ism_by_fold.sh")
    mpra_root = tmp_path / "mpra_run"
    (mpra_root / "data" / "mpra").mkdir(parents=True)
    (mpra_root / "data" / "mpra" / "X.npy").touch()
    mpra_ckpt = mpra_root / "experiments" / "legnet_alt" / "cv10" / "fold1" / "model" / "best.pt"
    mpra_ckpt.parent.mkdir(parents=True)
    mpra_ckpt.touch()

    (mpra_work / "mpra_config.sh").write_text(
        f"RUN_ROOT={shlex.quote(str(mpra_root))}\n"
        f"source {shlex.quote(str(SCRIPTS / 'mpra' / 'mpra_config.sh'))}\n"
        "setup_transcriptml_env() { :; }\n"
        'transcriptml() { printf "%s\\n" "$@"; }\n',
        encoding="utf-8",
    )

    mpra_res = subprocess.run(
        ["bash", str(mpra_work / "ism_by_fold.sh"), "--exp", "legnet_alt", "1"],
        env=_clean_env(),
        text=True,
        capture_output=True,
    )
    assert mpra_res.returncode == 0, mpra_res.stderr
    mpra_args = mpra_res.stdout.splitlines()
    assert mpra_args[mpra_args.index("--checkpoint") + 1] == str(mpra_ckpt)
    assert mpra_args[mpra_args.index("--dataset") + 1] == f"{mpra_root}/data/mpra"
    assert mpra_args[mpra_args.index("--out-dir") + 1] == f"{mpra_root}/experiments/legnet_alt/interpret/ism/fold1"

    # Test RBPNet config scoping
    rbp_work = tmp_path / "rbpnet"
    rbp_work.mkdir()
    shutil.copyfile(SCRIPTS / "rbpnet" / "eval_cv_fold.sh", rbp_work / "eval_cv_fold.sh")
    rbp_root = tmp_path / "rbp_run"
    (rbp_root / "data" / "rbpnet").mkdir(parents=True)
    (rbp_root / "data" / "rbpnet" / "config.json").write_text("{}", encoding="utf-8")
    rbp_ckpt = rbp_root / "experiments" / "profile_only" / "cv" / "fold0" / "model" / "best.pt"
    rbp_ckpt.parent.mkdir(parents=True)
    rbp_ckpt.touch()

    (rbp_work / "rbpnet_config.sh").write_text(
        f"RUN_ROOT={shlex.quote(str(rbp_root))}\n"
        f"source {shlex.quote(str(SCRIPTS / 'rbpnet' / 'rbpnet_config.sh'))}\n"
        "setup_transcriptml_env() { :; }\n"
        'transcriptml() { printf "%s\\n" "$@"; }\n',
        encoding="utf-8",
    )

    rbp_res = subprocess.run(
        ["bash", str(rbp_work / "eval_cv_fold.sh"), "--exp", "profile_only", "--eval-split", "val", "0"],
        env=_clean_env(),
        text=True,
        capture_output=True,
    )
    assert rbp_res.returncode == 0, rbp_res.stderr
    rbp_args = rbp_res.stdout.splitlines()
    assert rbp_args[rbp_args.index("--checkpoint") + 1] == str(rbp_ckpt)
    assert rbp_args[rbp_args.index("--dataset") + 1] == f"{rbp_root}/data/rbpnet"
    assert rbp_args[rbp_args.index("--out-dir") + 1] == f"{rbp_root}/experiments/profile_only/cv/evaluation/fold0/val"
