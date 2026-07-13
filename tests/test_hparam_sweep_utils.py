from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


def _load_sweep_utils():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "hparam_sweep_utils.py"
    spec = importlib.util.spec_from_file_location("hparam_sweep_utils", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_apply_hparams_merges_top_level_model_params_and_dotted_paths():
    utils = _load_sweep_utils()
    config = {
        "model": {"name": "saluki_exact", "params": {"filters": 64}},
        "learning_rate": 0.1,
        "batch_size": 32,
        "seed": 7,
    }
    row = {
        "learning_rate": "0.001",
        "batch_size": "16",
        "dropout": "0.25",
        "model.params.kernel_size": "7",
        "model": "saluki_like",
        "monitor": '["val_loss","val_pearson"]',
        "progress": "false",
        "combo_label": "fast-pass",
    }

    merged = utils.apply_hparams(json.loads(json.dumps(config)), row)

    assert merged["learning_rate"] == 0.001
    assert merged["batch_size"] == 16
    assert merged["monitor"] == ["val_loss", "val_pearson"]
    assert merged["progress"] is False
    assert merged["model"]["name"] == "saluki_like"
    assert merged["model"]["params"]["filters"] == 64
    assert merged["model"]["params"]["dropout"] == 0.25
    assert merged["model"]["params"]["kernel_size"] == 7
    assert "combo_label" not in merged["model"]["params"]


def test_param_parser_allows_json_lists_as_values():
    utils = _load_sweep_utils()

    name, values = utils.parse_param_spec("ef_block_sizes=[64,96,128],[32,64,96]")

    assert name == "ef_block_sizes"
    assert values == [[64, 96, 128], [32, 64, 96]]


def test_write_fold_artifacts_for_sweep_combo(tmp_path):
    utils = _load_sweep_utils()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    np.save(dataset / "X.npy", np.zeros((6, 4, 8), dtype=np.float32))
    np.save(dataset / "y.npy", np.arange(6, dtype=np.float32))
    (dataset / "ids.txt").write_text("\n".join(f"id{i}" for i in range(6)), encoding="utf-8")
    (dataset / "schema.json").write_text("{}", encoding="utf-8")
    (dataset / "config.json").write_text("{}", encoding="utf-8")

    base_config = tmp_path / "base.json"
    base_config.write_text(
        json.dumps(
            {
                "dataset": "placeholder",
                "output_dir": "placeholder",
                "model": {"name": "legnet", "params": {"in_ch": 4}},
                "learning_rate": 0.1,
                "seed": 10,
            }
        ),
        encoding="utf-8",
    )
    sweep_table = tmp_path / "sweep.tsv"
    sweep_table.write_text(
        "learning_rate\tstem_ch\tmodel.params.head_dropout\n0.001\t32\t0.2\n",
        encoding="utf-8",
    )

    config_path = utils.write_fold_artifacts(
        dataset=dataset,
        base_config=base_config,
        cv_root=tmp_path / "sweep_root",
        fold=1,
        n_folds=3,
        seed=123,
        val_offset=1,
        sweep_table=sweep_table,
        combo_index=0,
        default_model_name="legnet",
    )

    assert config_path == tmp_path / "sweep_root" / "combo_0000" / "fold1" / "train_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["dataset"].endswith("combo_0000/fold1/dataset")
    assert config["output_dir"].endswith("combo_0000/fold1/model")
    assert config["learning_rate"] == 0.001
    assert config["seed"] == 11
    assert config["model"]["params"]["in_ch"] == 4
    assert config["model"]["params"]["stem_ch"] == 32
    assert config["model"]["params"]["head_dropout"] == 0.2
    assert (tmp_path / "sweep_root" / "combo_0000" / "hparams.json").exists()
    assert (tmp_path / "sweep_root" / "combo_0000" / "fold1" / "dataset" / "X.npy").is_symlink()

    splits = json.loads(
        (tmp_path / "sweep_root" / "combo_0000" / "fold1" / "dataset" / "splits.json").read_text(
            encoding="utf-8"
        )
    )
    assert sorted(splits) == ["test", "train", "val"]
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == 6


def test_generate_and_summarize_sweep(tmp_path):
    utils = _load_sweep_utils()
    grid_path = tmp_path / "saluki_hparams.tsv"
    utils.generate_grid("saluki", "smoke", [], grid_path)
    headers, rows = utils.read_table(grid_path)
    assert headers == ["filters", "learning_rate", "epochs", "patience"]
    assert len(rows) == 2

    sweep_root = tmp_path / "sweep"
    utils.normalize_table(grid_path, sweep_root / "sweep_table.tsv")
    for combo, fold_metrics in {
        0: [(0.5, 0.4, 0.45), (0.7, 0.3, 0.35)],
        1: [(0.4, 0.2, 0.25)],
    }.items():
        for fold, (pearson, mse, loss) in enumerate(fold_metrics):
            summary_dir = sweep_root / f"combo_{combo:04d}" / f"fold{fold}" / "model"
            summary_dir.mkdir(parents=True)
            (summary_dir / "summary.json").write_text(
                json.dumps({"test_pearson": pearson, "test_mse": mse, "test_loss": loss}),
                encoding="utf-8",
            )

    summary_path = utils.summarize_sweep(sweep_root=sweep_root)

    with summary_path.open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle, delimiter="\t"))
    by_combo = {int(row["combo_index"]): row for row in summary_rows}
    assert by_combo[0]["rank"] == "1"
    assert by_combo[0]["completed_folds"] == "2"
    assert by_combo[0]["mean_test_pearson"] == "0.6"
    assert by_combo[1]["completed_folds"] == "1"
    assert (sweep_root / "combo_0000" / "combo_summary.json").exists()
