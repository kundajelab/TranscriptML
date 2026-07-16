import json

import numpy as np
import pytest

from transcriptml.analysis.ism_summary import summarize_ism_folds
from transcriptml.cli.main import main
from transcriptml.data.bundle import DatasetBundle, save_bundle


def _write_fold_deltas(root, arrays):
    for i, arr in enumerate(arrays):
        fold_dir = root / f"fold{i}"
        fold_dir.mkdir(parents=True)
        np.save(fold_dir / "deltas.npy", arr.astype(np.float32))


def _tiny_bundle(root):
    X = np.zeros((2, 4, 3), dtype=np.uint8)
    X[0, [0, 1, 2], [0, 1, 2]] = 1
    X[1, [3, 2, 1], [0, 1, 2]] = 1
    save_bundle(DatasetBundle(X=X, ids=["seq0", "seq1"], schema="rna4"), root)
    return X


def test_summarize_ism_folds_writes_average_centered_std_and_projected(tmp_path):
    ism_dir = tmp_path / "ism"
    base = (np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3) / 10.0)
    _write_fold_deltas(ism_dir, [base, base + 2.0])
    X = _tiny_bundle(tmp_path / "data")

    out_dir = tmp_path / "summary"
    summary = summarize_ism_folds(
        input_dir=ism_dir,
        out_dir=out_dir,
        dataset=tmp_path / "data",
        batch_size=1,
        write_fold_std=True,
        write_projected=True,
    )

    average = np.load(out_dir / "average_deltas.npy")
    centered = np.load(out_dir / "average_mean_centered_deltas.npy")
    std = np.load(out_dir / "fold_std_deltas.npy")
    projected = np.load(out_dir / "average_projected_mean_centered_deltas.npy")

    expected_average = base + 1.0
    expected_centered = expected_average - expected_average.mean(axis=1, keepdims=True)
    assert np.allclose(average, expected_average)
    assert np.allclose(centered, expected_centered)
    assert np.allclose(centered.mean(axis=1), 0.0, atol=1e-6)
    assert np.allclose(std, 1.0)
    assert np.allclose(projected, expected_centered * X)
    assert (out_dir / "ids.txt").read_text(encoding="utf-8").splitlines() == ["seq0", "seq1"]
    assert summary["fold_count"] == 2
    assert summary["outputs"]["average_mean_centered_deltas"].endswith("average_mean_centered_deltas.npy")

    summary_json = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_json["same_sequence_order_required"] is True


def test_summarize_ism_cli_smoke(tmp_path):
    ism_dir = tmp_path / "ism"
    arr0 = np.zeros((1, 4, 2), dtype=np.float32)
    arr1 = np.ones((1, 4, 2), dtype=np.float32)
    _write_fold_deltas(ism_dir, [arr0, arr1])

    out_dir = tmp_path / "summary"
    main(["summarize-ism", "--input-dir", str(ism_dir), "--out-dir", str(out_dir), "--batch-size", "1"])

    assert (out_dir / "average_deltas.npy").exists()
    assert np.allclose(np.load(out_dir / "average_deltas.npy"), 0.5)
    assert np.allclose(np.load(out_dir / "average_mean_centered_deltas.npy"), 0.0)


def test_summarize_ism_requires_dataset_for_projected(tmp_path):
    ism_dir = tmp_path / "ism"
    _write_fold_deltas(ism_dir, [np.zeros((1, 4, 2), dtype=np.float32)])

    with pytest.raises(ValueError, match="requires --dataset"):
        summarize_ism_folds(input_dir=ism_dir, out_dir=tmp_path / "summary", write_projected=True)
