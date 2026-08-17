import json

import numpy as np
import pytest

from transcriptml.cli.main import build_parser, main
from transcriptml.data.bundle import DatasetBundle, save_bundle
from transcriptml.training.trainer import TrainConfig, _select_splits
from transcriptml.workflows.chromosome_cv import (
    create_chromosome_cv_plan,
    load_chromosome_cv_plan,
    resolve_chromosome_cv_plan,
    save_chromosome_cv_plan,
)


def _metadata():
    counts = {
        "chr1": 23,
        "chr2": 17,
        "chr3": 15,
        "chr4": 11,
        "chr5": 9,
        "chr6": 8,
        "chr7": 6,
        "chr8": 5,
        "chr9": 4,
        "chr10": 2,
    }
    return [
        {"group_chromosome": chromosome, "row": row}
        for chromosome, count in counts.items()
        for row in range(count)
    ]


def test_chromosome_plan_is_deterministic_balanced_and_partitions_groups():
    metadata = _metadata()
    first = create_chromosome_cv_plan(metadata, n_folds=5)
    second = create_chromosome_cv_plan(metadata, n_folds=5)
    assert first == second
    assert first.plan_id == second.plan_id
    flattened = [chrom for group in first.fold_groups for chrom in group]
    assert len(flattened) == len(set(flattened)) == 10
    assert set(flattened) == {row["group_chromosome"] for row in metadata}
    assert sum(first.fold_example_counts) == len(metadata)
    # Largest-first greedy balancing cannot guarantee equal folds, but this
    # realistic skew remains substantially tighter than one largest group.
    assert max(first.fold_example_counts) - min(first.fold_example_counts) <= 6


def test_chromosome_plan_rotation_has_no_leakage_and_complete_cv_coverage():
    metadata = _metadata()
    plan = create_chromosome_cv_plan(metadata, n_folds=5)
    test_occurrences = {chrom: 0 for chrom in plan.chromosome_counts}
    val_occurrences = {chrom: 0 for chrom in plan.chromosome_counts}
    for fold in range(plan.n_folds):
        resolution = resolve_chromosome_cv_plan(plan, metadata, fold=fold)
        split_groups = {name: set(values) for name, values in resolution.groups.items()}
        assert split_groups["train"].isdisjoint(split_groups["val"])
        assert split_groups["train"].isdisjoint(split_groups["test"])
        assert split_groups["val"].isdisjoint(split_groups["test"])
        assigned = [index for values in resolution.indices.values() for index in values]
        assert sorted(assigned) == list(range(len(metadata)))
        assert len(assigned) == len(set(assigned))
        for chromosome in split_groups["test"]:
            test_occurrences[chromosome] += 1
        for chromosome in split_groups["val"]:
            val_occurrences[chromosome] += 1
    assert set(test_occurrences.values()) == {1}
    assert set(val_occurrences.values()) == {1}


def test_chromosome_plan_roundtrip_and_dataset_mismatch_detection(tmp_path):
    metadata = _metadata()
    plan = create_chromosome_cv_plan(metadata, n_folds=5)
    path = save_chromosome_cv_plan(plan, tmp_path / "cv5.json")
    loaded = load_chromosome_cv_plan(path)
    assert loaded == plan
    assert loaded.to_dict() == json.loads(path.read_text(encoding="utf-8"))
    changed = list(metadata)
    changed[0] = {**changed[0], "group_chromosome": "chrX"}
    with pytest.raises(ValueError, match="differ from the saved CV plan"):
        resolve_chromosome_cv_plan(loaded, changed, fold=0)


def test_chromosome_plan_integrates_with_training_split_selection(tmp_path):
    metadata = _metadata()
    plan_path = save_chromosome_cv_plan(
        create_chromosome_cv_plan(metadata, n_folds=5), tmp_path / "cv5.json"
    )
    bundle = DatasetBundle(
        X=np.zeros((len(metadata), 4, 8), dtype=np.float32),
        y=np.zeros(len(metadata), dtype=np.float32),
        metadata=metadata,
    )
    cfg = TrainConfig(
        dataset="unused",
        output_dir=str(tmp_path / "model"),
        cv_plan=str(plan_path),
        fold=3,
    )
    splits, source = _select_splits(bundle, cfg)
    expected = resolve_chromosome_cv_plan(
        load_chromosome_cv_plan(plan_path), metadata, fold=3
    )
    assert source == "cv_plan"
    assert splits == expected.indices


def test_chromosome_plan_cli_create_and_resolve(tmp_path, capsys):
    metadata = _metadata()
    bundle_dir = tmp_path / "bundle"
    save_bundle(
        DatasetBundle(
            X=np.zeros((len(metadata), 4, 8), dtype=np.uint8),
            ids=[f"id{i}" for i in range(len(metadata))],
            metadata=metadata,
        ),
        bundle_dir,
    )
    plan_path = tmp_path / "cv5.json"
    main(
        [
            "cv",
            "create-chromosome-plan",
            "--dataset",
            str(bundle_dir),
            "--output",
            str(plan_path),
            "--n-folds",
            "5",
        ]
    )
    assert capsys.readouterr().out.strip() == str(plan_path)
    splits_path = tmp_path / "fold2.json"
    main(
        [
            "cv",
            "resolve-plan",
            "--dataset",
            str(bundle_dir),
            "--cv-plan",
            str(plan_path),
            "--fold",
            "2",
            "--output",
            str(splits_path),
        ]
    )
    assert capsys.readouterr().out.strip() == str(splits_path)
    resolved = json.loads(splits_path.read_text(encoding="utf-8"))
    assert resolved["fold"] == 2
    assert resolved["validation_fold"] == 3
    assert sum(len(values) for values in resolved["indices"].values()) == len(metadata)


def test_chromosome_plan_requires_enough_chromosomes():
    with pytest.raises(ValueError, match="only 1 chromosomes"):
        create_chromosome_cv_plan(
            [{"group_chromosome": "chr21"}] * 10,
            n_folds=5,
        )


def test_train_cli_accepts_plan_and_job_array_overrides():
    args = build_parser().parse_args(
        [
            "train",
            "train.json",
            "--cv-plan",
            "cv5.json",
            "--fold",
            "3",
            "--dataset",
            "bundle",
            "--output-dir",
            "runs/fold3",
        ]
    )
    assert args.cv_plan == "cv5.json"
    assert args.fold == 3
    assert args.dataset == "bundle"
    assert args.output_dir == "runs/fold3"
