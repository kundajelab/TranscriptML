from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _train_config(workflow: str) -> dict[str, Any]:
    if workflow == "saluki":
        return {
            "dataset": "__EDIT_ME_DATASET_DIR__",
            "output_dir": "__EDIT_ME_RUN_DIR__/model",
            "model": {
                "name": "saluki_exact",
                "params": {},
            },
            "batch_size": 64,
            "epochs": 250,
            "learning_rate": 0.0001,
            "weight_decay": 0.0,
            "gradient_clip_norm": 0.5,
            "patience": 10,
            "monitor": ["val_loss", "val_pearson"],
            "loss": {"name": "mse"},
            "device": "auto",
            "num_workers": 0,
            "mmap_mode": "r",
            "seed": 42,
            "head_layernorm": False,
            "split_source": "auto",
            "split": {"method": "random", "val_frac": 0.1, "test_frac": 0.1},
        }
    if workflow == "legnet":
        return {
            "dataset": "__EDIT_ME_DATASET_DIR__",
            "output_dir": "__EDIT_ME_RUN_DIR__/model",
            "model": {
                "name": "legnet",
                "params": {},
            },
            "batch_size": 64,
            "epochs": 20,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "patience": 5,
            "monitor": "val_loss",
            "loss": {"name": "mse"},
            "device": "auto",
            "seed": 123,
            "split_source": "auto",
            "split": {"method": "random", "val_frac": 0.1, "test_frac": 0.1},
        }
    if workflow == "rbpnet":
        return {
            "dataset": "__EDIT_ME_RBPNET_BUNDLE_DIR__",
            "output_dir": "__EDIT_ME_RUN_DIR__/model",
            "model": {
                "name": "rbpnet",
                "params": {
                    "profile_length": 300,
                    "enrichment_head_type": "none",
                },
            },
            "batch_size": 64,
            "epochs": 100,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "optimizer": {"name": "adamw"},
            "lr_scheduler": {"name": "reduce_on_plateau", "patience": 3},
            "mixed_precision": False,
            "gradient_clip_norm": 0.5,
            "patience": 10,
            "monitor": "val_loss",
            "loss": {
                "name": "rbpnet",
                "lambda_ip_profile": 1.0,
                "lambda_sm_profile": 1.0,
                "lambda_enrichment": 1.0,
            },
            "device": "auto",
            "num_workers": 0,
            "mmap_mode": "r",
            "seed": 123,
            "max_train_jitter": 0,
            "deduplicate_loci": True,
            "split_source": "config",
            "split": {
                "method": "group",
                "group_col": "group_gene_id",
                "val_frac": 0.1,
                "test_frac": 0.1,
            },
        }
    raise ValueError("workflow must be one of: saluki, legnet, rbpnet")


def _readme(workflow: str) -> str:
    return f"""# TranscriptML {workflow} Run

This directory was created by `transcriptml init-run`.

Edit `train_config.json`, then run:

```sh
transcriptml train train_config.json
```

The checked-in `scripts/` workflows remain unchanged; this directory is only a
starter config bundle.
"""


def init_run(workflow: str, out_dir: str | Path, *, force: bool = False) -> Path:
    """Write starter configs for a TranscriptML run.

    Args:
        workflow: Workflow template name: ``saluki``, ``legnet``, or ``rbpnet``.
        out_dir: Directory to create or populate.
        force: Allow writing into a non-empty output directory.
    """

    workflow = str(workflow).strip().lower()
    if workflow not in {"saluki", "legnet", "rbpnet"}:
        raise ValueError("workflow must be one of: saluki, legnet, rbpnet")
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()) and not force:
        raise FileExistsError(f"Output directory is not empty: {out}. Use --force to overwrite template files.")
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_config.json").write_text(json.dumps(_train_config(workflow), indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(_readme(workflow), encoding="utf-8")
    return out
