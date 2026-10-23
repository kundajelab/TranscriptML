from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from transcriptml.analysis.attribution_average import (
    average_attributions,
    find_fold_attribution_paths,
)
from transcriptml.cli.main import build_parser, main
from transcriptml.data.bundle import DatasetBundle, save_bundle


def _write_fold_array(root: Path, fold: int, name: str, values: np.ndarray) -> Path:
    fold_dir = root / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    path = fold_dir / name
    np.save(path, values)
    return path


def test_average_deep_shap_attributions_and_preserve_alignment(tmp_path: Path):
    indices = np.asarray([2, 5], dtype=np.int64)
    ids = "tx2\ntx5\n"
    for attribution_kind in ("hypothetical", "actual"):
        name = f"high_confidence_{attribution_kind}_attributions.npy"
        for fold, value in enumerate((1.0, 3.0)):
            path = _write_fold_array(
                tmp_path,
                fold,
                name,
                np.full((2, 4, 3), value, dtype=np.float32),
            )
            np.save(path.with_name("high_confidence_indices.npy"), indices)
            path.with_name("high_confidence_ids.txt").write_text(ids, encoding="utf-8")

    X = np.arange(6 * 6 * 3, dtype=np.float32).reshape(6, 6, 3)
    dataset = tmp_path / "dataset"
    save_bundle(
        DatasetBundle(
            X=X,
            ids=tuple(f"tx{index}" for index in range(6)),
            schema="saluki6",
            metadata=tuple({"row": index} for index in range(6)),
        ),
        dataset,
    )

    out = tmp_path / "average"
    summary = average_attributions(
        tmp_path,
        out_dir=out,
        confidence="high",
        dataset=dataset,
        batch_size=1,
        progress=False,
    )

    for attribution_kind in ("hypothetical", "actual"):
        np.testing.assert_array_equal(
            np.load(out / f"average_high_confidence_{attribution_kind}_attributions.npy"),
            np.full((2, 4, 3), 2.0, dtype=np.float32),
        )
    np.testing.assert_array_equal(np.load(out / "sequences.npy"), X[indices])
    np.testing.assert_array_equal(np.load(out / "sequence_indices.npy"), indices)
    assert (out / "ids.txt").read_text(encoding="utf-8") == ids
    assert json.loads((out / "metadata.json").read_text(encoding="utf-8")) == [
        {"row": 2},
        {"row": 5},
    ]
    np.testing.assert_array_equal(np.load(out / "high_confidence_indices.npy"), indices)
    assert (out / "high_confidence_ids.txt").read_text(encoding="utf-8") == ids
    assert summary["attribution_type"] == "deep_shap"
    assert summary["confidence"] == "high"
    assert summary["fold_count"] == 2
    assert summary["sequence_shape"] == [2, 6, 3]


def test_deep_shap_partition_rows_are_aligned_by_index_intersection(tmp_path: Path):
    name = "low_confidence_actual_attributions.npy"
    for fold, indices in enumerate(([0, 1], [0, 2])):
        path = _write_fold_array(
            tmp_path,
            fold,
            name,
            np.full((2, 4, 3), float(fold * 2), dtype=np.float32),
        )
        np.save(path.with_name("low_confidence_indices.npy"), indices)
        path.with_name("low_confidence_ids.txt").write_text(
            f"tx0\ntx{indices[1]}\n", encoding="utf-8"
        )

    out = tmp_path / "intersection"
    summary = average_attributions(
        tmp_path,
        input_file=name,
        out_dir=out,
        progress=False,
    )
    np.testing.assert_array_equal(
        np.load(out / f"average_{name}"),
        np.ones((1, 4, 3), dtype=np.float32),
    )
    np.testing.assert_array_equal(np.load(out / "low_confidence_indices.npy"), [0])
    assert (out / "low_confidence_ids.txt").read_text(encoding="utf-8") == "tx0\n"
    assert summary["fold_row_counts"] == [2, 2]
    assert summary["aligned_row_count"] == 1

    with pytest.raises(ValueError, match="not identically aligned"):
        average_attributions(
            tmp_path,
            input_file=name,
            out_dir=tmp_path / "exact",
            alignment="exact",
            progress=False,
        )


@pytest.mark.parametrize(
    ("alignment", "expected_indices", "expected_support"),
    (
        ("intersection", [2], [3]),
        ("union", [0, 1, 2, 3, 4], [1, 2, 3, 2, 1]),
        ("mode", [1, 2, 3], [2, 3, 2]),
    ),
)
def test_deep_shap_fold_membership_policies(
    tmp_path: Path,
    alignment: str,
    expected_indices: list[int],
    expected_support: list[int],
):
    fold_indices = ([0, 1, 2], [1, 2, 3], [2, 3, 4])
    name = "high_confidence_actual_attributions.npy"
    for fold, indices in enumerate(fold_indices):
        values = np.stack(
            [
                np.full((4, 2), 10.0 * fold + index, dtype=np.float32)
                for index in indices
            ]
        )
        path = _write_fold_array(tmp_path, fold, name, values)
        np.save(path.with_name("high_confidence_indices.npy"), indices)
        path.with_name("high_confidence_ids.txt").write_text(
            "".join(f"tx{index}\n" for index in indices),
            encoding="utf-8",
        )

    out = tmp_path / alignment
    summary = average_attributions(
        tmp_path,
        input_file=name,
        out_dir=out,
        alignment=alignment,
        progress=False,
    )

    expected_values = {
        0: 0.0,
        1: 6.0,
        2: 12.0,
        3: 18.0,
        4: 24.0,
    }
    expected = np.stack(
        [np.full((4, 2), expected_values[index], dtype=np.float32) for index in expected_indices]
    )
    np.testing.assert_array_equal(np.load(out / f"average_{name}"), expected)
    np.testing.assert_array_equal(
        np.load(out / "high_confidence_indices.npy"), expected_indices
    )
    np.testing.assert_array_equal(
        np.load(out / "high_confidence_fold_support.npy"), expected_support
    )
    assert summary["alignment"] == alignment
    assert summary["fold_support_min"] == min(expected_support)
    assert summary["fold_support_max"] == max(expected_support)


def test_deep_shap_mode_threshold_is_strict_and_configurable(tmp_path: Path):
    name = "low_confidence_actual_attributions.npy"
    for fold, indices in enumerate(([0, 1], [0], [0, 1])):
        path = _write_fold_array(
            tmp_path,
            fold,
            name,
            np.full((len(indices), 4, 2), fold, dtype=np.float32),
        )
        np.save(path.with_name("low_confidence_indices.npy"), indices)
        path.with_name("low_confidence_ids.txt").write_text(
            "".join(f"tx{index}\n" for index in indices),
            encoding="utf-8",
        )

    out = tmp_path / "strict_mode"
    average_attributions(
        tmp_path,
        input_file=name,
        out_dir=out,
        alignment="mode",
        mode_threshold=2 / 3,
        progress=False,
    )
    # Index 1 occurs in exactly two of three folds and therefore does not
    # satisfy the deliberately strict greater-than threshold.
    np.testing.assert_array_equal(np.load(out / "low_confidence_indices.npy"), [0])


def test_deep_shap_empty_consensus_writes_empty_aligned_outputs(tmp_path: Path):
    for attribution_kind in ("hypothetical", "actual"):
        name = f"high_confidence_{attribution_kind}_attributions.npy"
        for fold, index in enumerate((0, 1)):
            path = _write_fold_array(
                tmp_path,
                fold,
                name,
                np.ones((1, 4, 2), dtype=np.float32),
            )
            np.save(path.with_name("high_confidence_indices.npy"), [index])
            path.with_name("high_confidence_ids.txt").write_text(
                f"tx{index}\n", encoding="utf-8"
            )
    dataset = tmp_path / "dataset"
    save_bundle(
        DatasetBundle(
            X=np.ones((2, 4, 2), dtype=np.int8),
            ids=("tx0", "tx1"),
            schema="rna4",
        ),
        dataset,
    )

    out = tmp_path / "empty"
    summary = average_attributions(
        tmp_path,
        out_dir=out,
        confidence="high",
        dataset=dataset,
        progress=False,
    )

    assert np.load(out / "average_high_confidence_hypothetical_attributions.npy").shape == (
        0,
        4,
        2,
    )
    assert np.load(out / "sequences.npy").shape == (0, 4, 2)
    assert summary["aligned_row_count"] == 0


def test_average_all_deep_shap_combines_partitions_before_fold_mean(tmp_path: Path):
    fold_partitions = (
        {"high_confidence": [0, 2], "low_confidence": [1, 3]},
        {"high_confidence": [1, 2], "low_confidence": [0, 3]},
    )
    offsets = {"hypothetical": 0.0, "actual": 100.0}
    for fold, partitions in enumerate(fold_partitions):
        for partition, indices in partitions.items():
            for attribution_kind, kind_offset in offsets.items():
                values = np.stack(
                    [
                        np.full(
                            (4, 2),
                            kind_offset + 10.0 * fold + index,
                            dtype=np.float32,
                        )
                        for index in indices
                    ]
                )
                path = _write_fold_array(
                    tmp_path,
                    fold,
                    f"{partition}_{attribution_kind}_attributions.npy",
                    values,
                )
                np.save(path.with_name(f"{partition}_indices.npy"), indices)
                path.with_name(f"{partition}_ids.txt").write_text(
                    "".join(f"tx{index}\n" for index in indices),
                    encoding="utf-8",
                )

    X = np.arange(5 * 4 * 2, dtype=np.float32).reshape(5, 4, 2)
    dataset = tmp_path / "dataset"
    save_bundle(
        DatasetBundle(
            X=X,
            ids=tuple(f"tx{index}" for index in range(5)),
            schema="rna4",
        ),
        dataset,
    )
    out = tmp_path / "average_all"
    summary = average_attributions(
        tmp_path,
        out_dir=out,
        confidence="all",
        dataset=dataset,
        alignment="exact",
        batch_size=2,
        progress=False,
    )

    indices = np.arange(4, dtype=np.int64)
    for attribution_kind, kind_offset in offsets.items():
        expected = np.stack(
            [
                np.full((4, 2), kind_offset + 5.0 + index, dtype=np.float32)
                for index in indices
            ]
        )
        np.testing.assert_array_equal(
            np.load(out / f"average_all_{attribution_kind}_attributions.npy"),
            expected,
        )
    np.testing.assert_array_equal(np.load(out / "all_indices.npy"), indices)
    assert (out / "all_ids.txt").read_text(encoding="utf-8") == (
        "tx0\ntx1\ntx2\ntx3\n"
    )
    np.testing.assert_array_equal(np.load(out / "sequences.npy"), X[indices])
    np.testing.assert_array_equal(np.load(out / "sequence_indices.npy"), indices)
    assert summary["confidence"] == "all"
    assert summary["fold_row_counts"] == [4, 4]
    assert summary["aligned_row_count"] == 4


def test_average_all_deep_shap_excludes_rows_ineligible_in_any_fold(tmp_path: Path):
    for attribution_kind in ("hypothetical", "actual"):
        for fold, low_indices in enumerate(([1, 2], [1])):
            high_path = _write_fold_array(
                tmp_path,
                fold,
                f"high_confidence_{attribution_kind}_attributions.npy",
                np.full((1, 4, 2), fold + 1.0, dtype=np.float32),
            )
            np.save(high_path.with_name("high_confidence_indices.npy"), [0])
            high_path.with_name("high_confidence_ids.txt").write_text(
                "tx0\n", encoding="utf-8"
            )
            low_path = _write_fold_array(
                tmp_path,
                fold,
                f"low_confidence_{attribution_kind}_attributions.npy",
                np.full((len(low_indices), 4, 2), fold + 3.0, dtype=np.float32),
            )
            np.save(low_path.with_name("low_confidence_indices.npy"), low_indices)
            low_path.with_name("low_confidence_ids.txt").write_text(
                "".join(f"tx{index}\n" for index in low_indices),
                encoding="utf-8",
            )
    dataset = tmp_path / "dataset"
    save_bundle(
        DatasetBundle(
            X=np.ones((3, 4, 2), dtype=np.int8),
            ids=("tx0", "tx1", "tx2"),
            schema="rna4",
        ),
        dataset,
    )

    out = tmp_path / "intersection_all"
    summary = average_attributions(
        tmp_path,
        out_dir=out,
        confidence="all",
        dataset=dataset,
        progress=False,
    )

    np.testing.assert_array_equal(np.load(out / "all_indices.npy"), [0, 1])
    assert summary["aligned_row_count"] == 2

    union_out = tmp_path / "union_all"
    union_summary = average_attributions(
        tmp_path,
        out_dir=union_out,
        confidence="all",
        dataset=dataset,
        alignment="union",
        progress=False,
    )
    np.testing.assert_array_equal(np.load(union_out / "all_indices.npy"), [0, 1, 2])
    np.testing.assert_array_equal(
        np.load(union_out / "all_fold_support.npy"), [2, 2, 1]
    )
    assert union_summary["aligned_row_count"] == 3


def test_deep_shap_use_all_folds_selects_by_confidence_then_uses_both_partitions(
    tmp_path: Path,
):
    high_by_fold = ([0, 1, 2], [1, 2, 3], [2, 3, 4])
    all_indices = list(range(5))
    for fold, high_indices in enumerate(high_by_fold):
        low_indices = [index for index in all_indices if index not in high_indices]
        for partition, indices in (
            ("high_confidence", high_indices),
            ("low_confidence", low_indices),
        ):
            for attribution_kind, offset in (("hypothetical", 0.0), ("actual", 100.0)):
                values = np.stack(
                    [
                        np.full(
                            (4, 2),
                            offset + 10.0 * fold + index,
                            dtype=np.float32,
                        )
                        for index in indices
                    ]
                )
                path = _write_fold_array(
                    tmp_path,
                    fold,
                    f"{partition}_{attribution_kind}_attributions.npy",
                    values,
                )
                np.save(path.with_name(f"{partition}_indices.npy"), indices)
                path.with_name(f"{partition}_ids.txt").write_text(
                    "".join(f"tx{index}\n" for index in indices),
                    encoding="utf-8",
                )

    X = np.arange(5 * 4 * 2, dtype=np.float32).reshape(5, 4, 2)
    dataset = tmp_path / "all_fold_dataset"
    save_bundle(
        DatasetBundle(
            X=X,
            ids=tuple(f"tx{index}" for index in all_indices),
            schema="rna4",
        ),
        dataset,
    )
    out = tmp_path / "high_union_all_folds"
    summary = average_attributions(
        tmp_path,
        out_dir=out,
        confidence="high",
        use_all_folds=True,
        dataset=dataset,
        alignment="union",
        batch_size=2,
        progress=False,
    )

    for attribution_kind, offset in (("hypothetical", 0.0), ("actual", 100.0)):
        expected = np.stack(
            [
                np.full((4, 2), offset + 10.0 + index, dtype=np.float32)
                for index in all_indices
            ]
        )
        np.testing.assert_array_equal(
            np.load(
                out
                / f"average_high_confidence_{attribution_kind}_attributions.npy"
            ),
            expected,
        )
    np.testing.assert_array_equal(
        np.load(out / "high_confidence_confidence_support.npy"),
        [1, 2, 3, 2, 1],
    )
    np.testing.assert_array_equal(
        np.load(out / "high_confidence_fold_support.npy"),
        [3, 3, 3, 3, 3],
    )
    np.testing.assert_array_equal(np.load(out / "sequences.npy"), X)
    assert summary["use_all_folds"] is True
    assert summary["confidence_support_min"] == 1
    assert summary["fold_support_min"] == 3


@pytest.mark.parametrize("method", ("mean", "median", "conservative"))
def test_ism_combination_and_channel_centering(tmp_path: Path, method: str):
    values = np.asarray(
        [
            [[[1.0], [4.0], [-3.0], [8.0]]],
            [[[3.0], [0.0], [-1.0], [6.0]]],
            [[[9.0], [2.0], [-2.0], [7.0]]],
        ],
        dtype=np.float32,
    )[:, 0]
    for fold, value in enumerate(values):
        _write_fold_array(tmp_path, fold, "deltas.npy", value[None, ...])

    out = tmp_path / method
    summary = average_attributions(
        tmp_path,
        out_dir=out,
        ism=True,
        method=method,
        batch_size=1,
        progress=False,
    )
    stacked = values[:, None, ...]
    if method == "mean":
        expected = np.mean(stacked, axis=0)
    elif method == "median":
        expected = np.median(stacked, axis=0)
    else:
        index = np.argmin(np.abs(stacked), axis=0, keepdims=True)
        expected = np.take_along_axis(stacked, index, axis=0)[0]
    centered = expected - expected.mean(axis=1, keepdims=True)

    np.testing.assert_allclose(np.load(out / "combined_deltas.npy"), expected)
    np.testing.assert_allclose(np.load(out / "combined_centered_deltas.npy"), centered)
    np.testing.assert_allclose(
        np.load(out / "combined_centered_deltas.npy").mean(axis=1),
        0.0,
        atol=2e-7,
    )
    assert summary["channel_centered"] is True


def test_deep_shap_rejects_non_mean_combination(tmp_path: Path):
    with pytest.raises(ValueError, match="--method requires --ism"):
        average_attributions(
            tmp_path,
            input_file="attributions.npy",
            method="median",
            progress=False,
        )


def test_fold_discovery_rejects_gaps(tmp_path: Path):
    _write_fold_array(tmp_path, 0, "deltas.npy", np.zeros((1, 4, 1)))
    _write_fold_array(tmp_path, 2, "deltas.npy", np.zeros((1, 4, 1)))
    with pytest.raises(ValueError, match=r"missing fold\(s\): \[1\]"):
        find_fold_attribution_paths(tmp_path, "deltas.npy")


def test_average_attributions_cli_ism(tmp_path: Path):
    _write_fold_array(tmp_path, 0, "deltas.npy", np.ones((1, 4, 2)))
    _write_fold_array(tmp_path, 1, "deltas.npy", np.full((1, 4, 2), 3.0))
    out = tmp_path / "out"

    main(
        [
            "average-attributions",
            str(tmp_path),
            "--ism",
            "--method",
            "median",
            "--out-dir",
            str(out),
        ]
    )

    np.testing.assert_array_equal(
        np.load(out / "combined_deltas.npy"),
        np.full((1, 4, 2), 2.0, dtype=np.float32),
    )


def test_average_attributions_cli_deep_shap_options():
    args = build_parser().parse_args(
        [
            "average-attributions",
            "deep-shap-root",
            "--confidence",
            "low",
            "--dataset",
            "dataset",
            "--all-channels",
        ]
    )
    assert args.ism is False
    assert args.confidence == "low"
    assert args.dataset == Path("dataset")
    assert args.all_channels is True
    assert args.alignment == "intersection"

    base_only_args = build_parser().parse_args(
        [
            "average-attributions",
            "deep-shap-root",
            "--confidence",
            "low",
            "--dataset",
            "dataset",
            "--base-channels-only",
        ]
    )
    assert base_only_args.all_channels is False

    all_args = build_parser().parse_args(
        [
            "average-attributions",
            "deep-shap-root",
            "--confidence",
            "all",
            "--dataset",
            "dataset",
        ]
    )
    assert all_args.confidence == "all"
    assert all_args.all_channels is True

    mode_args = build_parser().parse_args(
        [
            "average-attributions",
            "deep-shap-root",
            "--confidence",
            "high",
            "--dataset",
            "dataset",
            "--alignment",
            "mode",
            "--mode-threshold",
            "0.7",
        ]
    )
    assert mode_args.alignment == "mode"
    assert mode_args.mode_threshold == 0.7

    all_fold_args = build_parser().parse_args(
        [
            "average-attributions",
            "deep-shap-root",
            "--confidence",
            "high",
            "--dataset",
            "dataset",
            "--alignment",
            "union",
            "--use-all-folds",
        ]
    )
    assert all_fold_args.use_all_folds is True
