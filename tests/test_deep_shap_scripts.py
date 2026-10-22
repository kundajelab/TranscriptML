"""Exercise DeepSHAP shell jobs without activating conda or submitting jobs."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(params=("saluki", "mpra"))
def deep_shap_jobs(tmp_path: Path, request):
    flavor = str(request.param)
    source = SCRIPTS if flavor == "saluki" else SCRIPTS / "mpra"
    config_name = "sherlock_config.sh" if flavor == "saluki" else "mpra_config.sh"
    work = tmp_path / flavor
    work.mkdir()
    for name in ("run_deep_shap.sh", "submit_deep_shap.sh"):
        shutil.copyfile(source / name, work / name)
        subprocess.run(["bash", "-n", str(work / name)], check=True)

    (work / config_name).write_text(
        f"source {shlex.quote(str(source / config_name))}\n"
        "setup_transcriptml_env() { :; }\n"
        'transcriptml() { printf "%s\\n" "$@"; }\n'
        'sbatch() { printf "%s\\n" "$@"; }\n',
        encoding="utf-8",
    )

    cv_root = tmp_path / "models with spaces"
    dataset = tmp_path / "input bundle"
    (cv_root / "fold2" / "model").mkdir(parents=True)
    (cv_root / "fold2" / "model" / "best.pt").touch()
    dataset.mkdir()
    (dataset / "X.npy").touch()
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("DEEP_SHAP_", "SLURM_", "TRANSCRIPTML_"))
        and key not in {"FOLD", "SHARD"}
    }
    env.update(
        CV_ROOT=str(cv_root),
        INTERPRET_ROOT=str(tmp_path / "results"),
        INTERPRET_DATASET_DIR=str(dataset),
        N_FOLDS="3",
        DEEP_SHAP_SEQUENCE_SHARDS="2",
        DEEP_SHAP_MAX_CONCURRENT="2",
        SLURM_ARRAY_TASK_ID="5",
        DEVICE="cpu",
        PRED_BATCH_SIZE="7",
    )
    return flavor, work, env


def _invoke(case, script: str, *arguments: str, **updates: str):
    _, work, env = case
    return subprocess.run(
        ["bash", str(work / script), *arguments],
        env={**env, **updates},
        text=True,
        capture_output=True,
    )


def test_deep_shap_worker_maps_fold_and_shard(deep_shap_jobs):
    _, _, env = deep_shap_jobs
    result = _invoke(deep_shap_jobs, "run_deep_shap.sh")
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    assert args[0] == "deep-shap"
    assert args[args.index("--checkpoint") + 1] == (
        f"{env['CV_ROOT']}/fold2/model/best.pt"
    )
    assert args[args.index("--dataset") + 1] == env["INTERPRET_DATASET_DIR"]
    assert args[args.index("--out-dir") + 1] == (
        f"{env['INTERPRET_ROOT']}/deep_shap/full-mono/fold2/shard1"
    )
    assert args[args.index("--sequence-shard-index") + 1] == "1"
    assert args[args.index("--sequence-shards") + 1] == "2"
    assert args[args.index("--reference-strategy") + 1] == "full-mono"
    assert args[args.index("--n-references") + 1] == "20"
    assert args[args.index("--attribution-batch-size") + 1] == "1"
    assert args[args.index("--batch-size") + 1] == "7"


def test_deep_shap_submit_combines_fold_and_shard_arrays(deep_shap_jobs):
    result = _invoke(deep_shap_jobs, "submit_deep_shap.sh")
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    assert args[0] == "--array=0-5%2"
    assert args[1].endswith("/run_deep_shap.sh")


def test_saluki_worker_forwards_optional_reference_controls(deep_shap_jobs):
    flavor, _, _ = deep_shap_jobs
    if flavor != "saluki":
        pytest.skip("Saluki-only codon reference controls")
    result = _invoke(
        deep_shap_jobs,
        "run_deep_shap.sh",
        DEEP_SHAP_REFERENCE_STRATEGY="cds-synonymous",
        DEEP_SHAP_RUN_NAME="synonymous_test",
        DEEP_SHAP_MINIMUM_EFFECT_SIZE="0.02",
        DEEP_SHAP_CDS_CHANNEL="4",
        DEEP_SHAP_INCLUDE_STOP_CODONS="true",
        DEEP_SHAP_RECODE_INITIATION_CODON="yes",
        DEEP_SHAP_RECODE_TERMINAL_STOP="1",
        DEEP_SHAP_REFERENCE_PREDICTIONS_FORMAT="csv",
    )
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    assert args[args.index("--reference-strategy") + 1] == "cds-synonymous"
    assert args[args.index("--minimum-effect-size") + 1] == "0.02"
    assert args[args.index("--cds-channel") + 1] == "4"
    assert "--include-stop-codons" in args
    assert "--recode-initiation-codon" in args
    assert "--recode-terminal-stop" in args
    assert args[args.index("--reference-predictions-format") + 1] == "csv"


def test_mpra_worker_rejects_regional_references(deep_shap_jobs):
    flavor, _, _ = deep_shap_jobs
    if flavor != "mpra":
        pytest.skip("MPRA-only strategy restriction")
    result = _invoke(
        deep_shap_jobs,
        "run_deep_shap.sh",
        DEEP_SHAP_REFERENCE_STRATEGY="3utr-mono",
    )
    assert result.returncode == 2
    assert "only full-mono and full-dinuc" in result.stderr
