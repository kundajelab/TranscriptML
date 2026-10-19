"""Exercise shell jobs with command capture; never activate conda or submit jobs."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture
def shell_jobs(tmp_path):
    for name in ("region_ablation_by_fold.sh", "region_ablation_custom_by_fold.sh",
                 "region_ablation_args.sh", "submit_region_ablation_custom_by_fold.sh"):
        shutil.copyfile(SCRIPTS / name, tmp_path / name)
        subprocess.run(["bash", "-n", str(tmp_path / name)], check=True)
    # Read the actual defaults, but replace environment activation and external commands.
    (tmp_path / "sherlock_config.sh").write_text(
        f'source {shlex.quote(str(SCRIPTS / "sherlock_config.sh"))}\n'
        'setup_transcriptml_env() { :; }\n'
        'transcriptml() { printf "%s\\n" "$@"; }\n'
        'sbatch() { printf "%s\\n" "$@"; }\n'
    )
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("REGION_", "SLURM_", "TRANSCRIPTML_"))}
    env.update(SLURM_ARRAY_TASK_ID="2", CV_ROOT="/models with spaces", INTERPRET_ROOT="/results",
               INTERPRET_DATASET_DIR="/input", N_FOLDS="10")
    return tmp_path, env


def invoke(shell_jobs, script, **updates):
    path, env = shell_jobs
    return subprocess.run(["bash", str(path / script)], env={**env, **updates},
                          text=True, capture_output=True)


def test_all_and_custom_fold_arguments(shell_jobs):
    full = invoke(shell_jobs, "region_ablation_by_fold.sh",
                  REGION_JUNCTION_DENSITIES="0,2", REGION_JUNCTION_DENSITY_QUANTILES="0.1,0.5")
    assert full.returncode == 0, full.stderr
    args = full.stdout.splitlines()
    assert args[0] == "region-ablation"
    assert args[args.index("--checkpoint") + 1] == "/models with spaces/fold2/model/best.pt"
    assert args[args.index("--families") + 1] == "all"
    assert args[args.index("--junction-densities") + 1] == "0,2"
    assert args[args.index("--junction-density-quantiles") + 1] == "0.1,0.5"
    assert args[args.index("--out-dir") + 1] == "/results/region_ablation/fold2"
    custom = invoke(shell_jobs, "region_ablation_custom_by_fold.sh",
                    REGION_ABLATION_FAMILIES="junction_density", REGION_ABLATION_CUSTOM_RUN_NAME="sweep")
    assert custom.returncode == 0, custom.stderr
    args = custom.stdout.splitlines()
    assert args[args.index("--families") + 1] == "junction_density"
    assert args[args.index("--out-dir") + 1] == "/results/region_ablation_custom/sweep/fold2"
    assert "--junction-densities" not in args
    assert "--junction-density-quantiles" not in args


@pytest.mark.parametrize("setting", [
    'REGION_ABLATION_DISABLED=("3utr_shuffle")',
    'REGION_ABLATION_N_FOR=("3utr_shuffle=0")',
    'REGION_ABLATION_N_FOR=("3utr_shuffle= -0 ")',
])
def test_all_wrapper_rejects_disabled_families_custom_accepts(shell_jobs, setting):
    path, _ = shell_jobs
    config = path / "user.sh"
    config.write_text(setting + "\n")
    result = invoke(shell_jobs, "region_ablation_by_fold.sh", TRANSCRIPTML_RUN_CONFIG=str(config))
    assert result.returncode != 0
    assert "region_ablation_custom_by_fold.sh" in result.stderr
    result = invoke(shell_jobs, "region_ablation_custom_by_fold.sh", TRANSCRIPTML_RUN_CONFIG=str(config))
    assert result.returncode == 0, result.stderr


def test_custom_submit_uses_matching_array_job(shell_jobs):
    result = invoke(shell_jobs, "submit_region_ablation_custom_by_fold.sh")
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    assert args[0] == "--array=0-9"
    assert args[1].endswith("/region_ablation_custom_by_fold.sh")
