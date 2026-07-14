import json

import numpy as np
import pytest

from transcriptml.cli.main import main
from transcriptml.workflows.cv import cv_splits, prepare_cv_fold


def _write_dataset(path, n=8):
    path.mkdir()
    np.save(path / "X.npy", np.zeros((n, 4, 8), dtype=np.float32))
    np.save(path / "y.npy", np.arange(n, dtype=np.float32))
    (path / "ids.txt").write_text("\n".join(f"id{i}" for i in range(n)) + "\n", encoding="utf-8")
    (path / "schema.json").write_text('{"name":"rna4","channels":["A","C","G","U"]}', encoding="utf-8")
    (path / "metadata.json").write_text(json.dumps([{"i": i} for i in range(n)]), encoding="utf-8")
    (path / "config.json").write_text("{}", encoding="utf-8")


def _write_base_config(path):
    path.write_text(
        json.dumps(
            {
                "dataset": "placeholder",
                "output_dir": "placeholder",
                "model": {"name": "old_model", "params": {"in_ch": 4}},
                "learning_rate": 0.01,
                "seed": 100,
            }
        ),
        encoding="utf-8",
    )


def test_prepare_cv_fold_writes_bundle_view_and_model_config(tmp_path):
    dataset = tmp_path / "dataset"
    _write_dataset(dataset, n=8)
    base_config = tmp_path / "base.json"
    _write_base_config(base_config)

    config_path = prepare_cv_fold(
        dataset=dataset,
        base_config=base_config,
        cv_root=tmp_path / "cv",
        fold=2,
        model="saluki_exact",
        n_folds=4,
        seed=7,
        val_offset=1,
    )

    assert config_path == tmp_path / "cv" / "fold2" / "train_config.json"
    fold_dataset = tmp_path / "cv" / "fold2" / "dataset"
    assert (fold_dataset / "X.npy").is_symlink()
    assert (fold_dataset / "y.npy").is_symlink()
    assert (fold_dataset / "metadata.json").is_symlink()
    assert (tmp_path / "cv" / "fold2" / "model").is_dir()

    splits = json.loads((fold_dataset / "splits.json").read_text(encoding="utf-8"))
    assert sorted(splits) == ["test", "train", "val"]
    assert set(splits["train"]).isdisjoint(splits["val"])
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert set(splits["val"]).isdisjoint(splits["test"])
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == 8

    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["dataset"].endswith("cv/fold2/dataset")
    assert config["output_dir"].endswith("cv/fold2/model")
    assert config["model"] == {"name": "saluki_exact", "params": {"in_ch": 4}}
    assert config["seed"] == 102
    assert config["mmap_mode"] == "r"


def test_cv_prepare_fold_cli_requires_model(tmp_path):
    dataset = tmp_path / "dataset"
    _write_dataset(dataset)
    base_config = tmp_path / "base.json"
    _write_base_config(base_config)

    with pytest.raises(SystemExit):
        main(
            [
                "cv",
                "prepare-fold",
                "--dataset",
                str(dataset),
                "--base-config",
                str(base_config),
                "--cv-root",
                str(tmp_path / "cv"),
                "--fold",
                "0",
            ]
        )


def test_cv_prepare_fold_cli_prints_config_path(tmp_path, capsys):
    dataset = tmp_path / "dataset"
    _write_dataset(dataset)
    base_config = tmp_path / "base.json"
    _write_base_config(base_config)

    main(
        [
            "cv",
            "prepare-fold",
            "--dataset",
            str(dataset),
            "--base-config",
            str(base_config),
            "--cv-root",
            str(tmp_path / "cv"),
            "--fold",
            "0",
            "--model",
            "legnet",
            "--n-folds",
            "4",
        ]
    )

    out = capsys.readouterr().out.strip()
    assert out.endswith("cv/fold0/train_config.json")


def test_cv_splits_validate_fold_and_val_offset():
    with pytest.raises(ValueError, match="fold"):
        cv_splits(8, fold=4, n_folds=4)
    with pytest.raises(ValueError, match="val_offset"):
        cv_splits(8, fold=0, n_folds=4, val_offset=4)
