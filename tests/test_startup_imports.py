"""Tests verifying lazy startup imports and job-scoped Matplotlib cache handling."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


REPO_SRC = Path(__file__).resolve().parents[1] / "src"


def _run_fresh_python(
    code: str,
    *,
    extra_env: dict[str, str | None] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_SRC)
    if extra_env:
        for key, value in extra_env.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def test_cli_build_parser_does_not_import_heavy_modules():
    _run_fresh_python(
        """
import sys
from transcriptml.cli.main import build_parser

parser = build_parser()
assert parser.prog == "transcriptml"
forbidden = ("torch", "tangermeme", "numba", "llvmlite", "matplotlib", "numpy")
loaded = [name for name in forbidden if name in sys.modules]
assert not loaded, f"Heavy modules loaded during build_parser(): {loaded}"
"""
    )


def test_cli_deep_shap_strategies_match_deep_shap_module():
    from transcriptml.cli.main import _DEEP_SHAP_REFERENCE_STRATEGIES
    from transcriptml.interpret.deep_shap import REFERENCE_STRATEGIES

    assert _DEEP_SHAP_REFERENCE_STRATEGIES == REFERENCE_STRATEGIES


def test_interpret_package_imports_lazily_and_exposes_all_symbols():
    _run_fresh_python(
        """
import sys
import transcriptml.interpret as interp

assert "torch" not in sys.modules
assert "tangermeme" not in sys.modules
assert "numba" not in sys.modules
assert "Predictor" in interp.__all__
assert "run_deep_shap" in interp.__all__
"""
    )


def test_plotting_training_defers_matplotlib_and_scopes_mplconfigdir(tmp_path: Path):
    lscratch = tmp_path / "lscratch_job"
    lscratch.mkdir()
    _run_fresh_python(
        """
import os, sys
from pathlib import Path
from transcriptml.plotting import training

assert "matplotlib" not in sys.modules
assert "matplotlib.pyplot" not in sys.modules

training._ensure_mpl_config_dir()
mpl_dir = Path(os.environ["MPLCONFIGDIR"])
assert mpl_dir.is_dir()
assert str(mpl_dir).startswith(os.environ["L_SCRATCH_JOB"])
assert "987654" in mpl_dir.name
""",
        extra_env={
            "L_SCRATCH_JOB": str(lscratch),
            "SLURM_JOB_ID": "987654",
            "MPLCONFIGDIR": None,
        },
    )
