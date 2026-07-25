import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from transcriptml.cli.main import main
from transcriptml.data.bundle import DatasetBundle, save_bundle
from transcriptml.training import evaluation
from transcriptml.workflows.cv import find_fold_checkpoints, load_fold_test_indices


class OffsetModel(torch.nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = float(offset)

    def forward(self, x):
        return x[:, 0, 0] + self.offset


def _write_dataset(path, *, targets=True):
    X = np.zeros((2, 4, 3), dtype=np.float32)
    X[:, 0, 0] = [1.0, 2.0]
    bundle = DatasetBundle(
        X=X,
        y=np.array([5.0, 7.0], dtype=np.float32) if targets else None,
        ids=["tx1", "tx2"],
        schema="rna4",
    )
    save_bundle(bundle, path)


def _write_checkpoint(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test checkpoint placeholder")


def _write_fold_test_split(checkpoint_path, test_indices):
    split_path = checkpoint_path.parent.parent / "dataset" / "splits.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        json.dumps({"train": [], "val": [], "test": list(test_indices)}),
        encoding="utf-8",
    )


def test_evaluate_fold_checkpoints_writes_average_predictions_and_residuals(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    _write_dataset(dataset)
    checkpoints = [tmp_path / "fold0.pt", tmp_path / "fold1.pt"]
    for path in checkpoints:
        _write_checkpoint(path)

    offsets = {"fold0.pt": 0.0, "fold1.pt": 2.0}
    monkeypatch.setattr(
        evaluation,
        "load_checkpoint",
        lambda path, map_location: (OffsetModel(offsets[path.name]), {}),
    )

    out_csv = tmp_path / "ensemble_predictions.csv"
    result = evaluation.evaluate_fold_checkpoints(
        checkpoints,
        dataset,
        out_csv,
        batch_size=1,
        progress=False,
    )

    np.testing.assert_allclose(result["average_predictions"], [2.0, 3.0])
    np.testing.assert_allclose(result["targets"], [5.0, 7.0])
    np.testing.assert_allclose(result["average_residuals"], [3.0, 4.0])
    assert result["fold_count"] == 2
    assert result["prediction_scope"] == "full_dataset"
    assert result["predictions_per_example"] == 2
    assert result["mse"] == pytest.approx(12.5)
    assert result["pearson"] == pytest.approx(1.0)
    assert result["mean_residual"] == pytest.approx(3.5)

    with out_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == ["index", "id", "target", "average_prediction", "average_residual"]
    assert rows == [
        {
            "index": "0",
            "id": "tx1",
            "target": "5.0",
            "average_prediction": "2.0",
            "average_residual": "3.0",
        },
        {
            "index": "1",
            "id": "tx2",
            "target": "7.0",
            "average_prediction": "3.0",
            "average_residual": "4.0",
        },
    ]

    summary = json.loads(out_csv.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert summary["analysis"] == "fold_checkpoint_ensemble"
    assert summary["fold_count"] == 2
    assert summary["prediction_scope"] == "full_dataset"
    assert summary["predictions_per_example"] == 2
    assert summary["residual_definition"] == (
        "mean(truth - fold_prediction) = truth - average_prediction"
    )
    assert summary["mse"] == pytest.approx(12.5)
    assert summary["mean_residual"] == pytest.approx(3.5)


def test_evaluate_fold_checkpoints_supports_targetless_dataset(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    _write_dataset(dataset, targets=False)
    checkpoint = tmp_path / "fold0.pt"
    _write_checkpoint(checkpoint)
    monkeypatch.setattr(
        evaluation,
        "load_checkpoint",
        lambda path, map_location: (OffsetModel(1.0), {}),
    )

    out_csv = tmp_path / "predictions.csv"
    result = evaluation.evaluate_fold_checkpoints(
        [checkpoint],
        dataset,
        out_csv,
        progress=False,
    )

    np.testing.assert_allclose(result["average_predictions"], [2.0, 3.0])
    assert "average_residuals" not in result
    assert "targets" not in result
    with out_csv.open(newline="", encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == ["index", "id", "average_prediction"]

    summary = json.loads(out_csv.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert summary["target_available"] is False
    assert "mean_residual" not in summary


def test_evaluate_fold_checkpoints_validates_checkpoint_inputs(tmp_path):
    dataset = tmp_path / "dataset"
    _write_dataset(dataset)

    with pytest.raises(ValueError, match="at least one"):
        evaluation.evaluate_fold_checkpoints([], dataset, progress=False)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        evaluation.evaluate_fold_checkpoints([tmp_path / "missing.pt"], dataset, progress=False)


def test_cv_ensemble_predict_discovers_checkpoints_in_natural_order(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "dataset"
    _write_dataset(dataset)
    cv_root = tmp_path / "cv"
    for fold in (10, 2):
        _write_checkpoint(cv_root / f"fold{fold}" / "model" / "best.pt")

    checkpoint_paths = find_fold_checkpoints(cv_root)
    assert [path.parent.parent.name for path in checkpoint_paths] == ["fold2", "fold10"]

    def load_checkpoint(path, map_location):
        offset = 0.0 if path.parent.parent.name == "fold2" else 2.0
        return OffsetModel(offset), {}

    monkeypatch.setattr(evaluation, "load_checkpoint", load_checkpoint)
    out_csv = tmp_path / "ensemble.csv"
    main(
        [
            "cv",
            "ensemble-predict",
            "--cv-root",
            str(cv_root),
            "--dataset",
            str(dataset),
            "--out-csv",
            str(out_csv),
            "--device",
            "cpu",
        ]
    )

    captured = capsys.readouterr()
    assert captured.out.strip() == str(out_csv)
    assert "ensemble-predict: discovered 2 checkpoints" in captured.err
    assert "ensemble: dataset ready: 2 examples; targets available" in captured.err
    assert "ensemble: checkpoint 1/2 complete" in captured.err
    assert "estimated remaining" in captured.err
    assert "ensemble: metrics: mse=" in captured.err
    assert "ensemble: done: 2 examples across 2 checkpoints" in captured.err
    with out_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [float(row["average_prediction"]) for row in rows] == [2.0, 3.0]
    summary = json.loads(out_csv.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert [Path(path).parent.parent.name for path in summary["checkpoint_paths"]] == ["fold2", "fold10"]


def test_cv_ensemble_predict_test_only_uses_each_folds_test_indices(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "dataset"
    _write_dataset(dataset)
    cv_root = tmp_path / "cv"
    fold2 = cv_root / "fold2" / "model" / "best.pt"
    fold10 = cv_root / "fold10" / "model" / "best.pt"
    for checkpoint, test_indices in [(fold2, [0]), (fold10, [1])]:
        _write_checkpoint(checkpoint)
        _write_fold_test_split(checkpoint, test_indices)

    assert load_fold_test_indices([fold2, fold10]) == [[0], [1]]

    def load_checkpoint(path, map_location):
        offset = 0.0 if path.parent.parent.name == "fold2" else 2.0
        return OffsetModel(offset), {}

    monkeypatch.setattr(evaluation, "load_checkpoint", load_checkpoint)
    out_csv = tmp_path / "test_only.csv"
    main(
        [
            "cv",
            "ensemble-predict",
            "--cv-root",
            str(cv_root),
            "--dataset",
            str(dataset),
            "--out-csv",
            str(out_csv),
            "--test-only",
        ]
    )

    captured = capsys.readouterr()
    assert "scope=fold test sets" in captured.err
    assert "1 prediction(s) per example" in captured.err
    with out_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [float(row["average_prediction"]) for row in rows] == [1.0, 4.0]
    assert [float(row["average_residual"]) for row in rows] == [4.0, 3.0]

    summary = json.loads(out_csv.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert summary["prediction_scope"] == "test_only"
    assert summary["predictions_per_example"] == 1


def test_test_only_requires_complete_nonoverlapping_fold_coverage(tmp_path):
    dataset = tmp_path / "dataset"
    _write_dataset(dataset)
    cv_root = tmp_path / "cv"
    checkpoints = [
        cv_root / "fold0" / "model" / "best.pt",
        cv_root / "fold1" / "model" / "best.pt",
    ]
    for checkpoint in checkpoints:
        _write_checkpoint(checkpoint)
        _write_fold_test_split(checkpoint, [0])

    with pytest.raises(ValueError, match=r"missing=1, repeated=1"):
        evaluation.evaluate_fold_checkpoints(
            checkpoints,
            dataset,
            test_only=True,
            progress=False,
        )


def test_load_fold_test_indices_requires_split_file(tmp_path):
    checkpoint = tmp_path / "cv" / "fold0" / "model" / "best.pt"
    _write_checkpoint(checkpoint)

    with pytest.raises(FileNotFoundError, match="requires fold split file"):
        load_fold_test_indices([checkpoint])
