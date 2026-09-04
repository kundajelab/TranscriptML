from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from transcriptml.cli.main import main
from transcriptml.data.bundle import DatasetBundle, save_bundle
from transcriptml.data.encoding import encode_saluki_transcript
from transcriptml.interpret.legnet_scan import (
    generate_scan_windows,
    normalize_scan_regions,
    save_legnet_scan_result,
    scan_legnet_windows,
)
from transcriptml.interpret.predictor import Predictor
from transcriptml.models.registry import build_model, save_checkpoint


def _sum_called_bases(batch: np.ndarray) -> np.ndarray:
    return np.asarray(batch).sum(axis=(1, 2), dtype=np.float32)


def test_generate_scan_windows_full_short_and_single_tail():
    assert generate_scan_windows(0, 8, 8, 2) == ((0, 8),)
    assert generate_scan_windows(3, 7, 8, 2) == ((3, 7),)
    assert generate_scan_windows(0, 11, 8, 2) == ((0, 8), (2, 10), (4, 11))
    assert generate_scan_windows(5, 5, 8, 2) == ()

    with pytest.raises(ValueError, match="stride"):
        generate_scan_windows(0, 10, 8, 9)


def test_normalize_scan_regions_expands_all_and_rejects_transcript_mix():
    assert normalize_scan_regions(None) == ("3utr",)
    assert normalize_scan_regions("3' UTR,CDS,3utr") == ("3utr", "cds")
    assert normalize_scan_regions("all") == ("5utr", "cds", "3utr")
    assert normalize_scan_regions("full") == ("transcript",)

    with pytest.raises(ValueError, match="cannot be combined"):
        normalize_scan_regions("transcript,3utr")


def test_scan_legnet_windows_padding_coordinates_metadata_and_skips(tmp_path):
    X = np.stack(
        [
            encode_saluki_transcript("A" * 20, length=20, cds_positions=[4, 7, 10]),
            encode_saluki_transcript("C" * 10, length=20),
        ]
    )
    metadata = [
        {
            "transcript_length": 20,
            "cds_start": 4,
            "cds_end": 13,
            "gene": "G1",
            "nested": {"source": "test"},
        },
        {"transcript_length": 10, "gene": "G2"},
    ]
    result = scan_legnet_windows(
        X,
        Predictor(_sum_called_bases, batch_size=2),
        window_size=8,
        regions="3utr",
        schema="saluki6",
        sequence_ids=["tx1", "tx2"],
        metadata=metadata,
        save_sequences=True,
        storage_dir=tmp_path,
        progress=False,
    )

    assert result.stride == 2
    assert result.scores.tolist() == [7.0]
    assert result.sequences is not None
    assert result.sequences.shape == (1, 4, 8)
    assert result.sequences[0, :, 7].sum() == 0
    instance = result.instances[0]
    assert (instance.region, instance.transcript_start, instance.transcript_end) == ("3utr", 13, 20)
    assert (instance.encoded_start, instance.encoded_end) == (13, 20)
    assert (instance.unpadded_length, instance.padding_length) == (7, 1)
    assert result.region_outcomes[1].skip_reason == "unresolved_region_annotation"

    save_legnet_scan_result(result, tmp_path, checkpoint="model.pt", dataset="bundle", progress=False)
    np.testing.assert_array_equal(np.load(tmp_path / "scores.npy"), [7.0])
    with (tmp_path / "instances.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["transcript_id"] == "tx1"
    assert rows[0]["metadata.gene"] == "G1"
    assert rows[0]["metadata.nested"] == '{"source":"test"}'

    with (tmp_path / "transcript_scores.csv").open(newline="", encoding="utf-8") as handle:
        transcript_rows = list(csv.DictReader(handle))
    assert transcript_rows[0]["status"] == "scored"
    assert transcript_rows[0]["predicted_stability_std"] == "0.0"
    assert transcript_rows[1]["status"] == "skipped"
    assert transcript_rows[1]["n_windows"] == "0"
    assert transcript_rows[1]["predicted_stability_mean"] == ""

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_windows"] == 1
    assert summary["skip_reason_counts"] == {"unresolved_region_annotation": 1}


def test_scan_legnet_all_regions_stay_within_bounds_and_aggregate():
    X = encode_saluki_transcript("A" * 20, length=20, cds_positions=[4, 7, 10])[None]
    result = scan_legnet_windows(
        X,
        Predictor(_sum_called_bases),
        window_size=8,
        stride=2,
        regions="all",
        schema="saluki6",
        sequence_ids=["tx"],
        metadata=[{"transcript_length": 20, "cds_start": 4, "cds_end": 13}],
        progress=False,
    )

    intervals = [(x.region, x.encoded_start, x.encoded_end) for x in result.instances]
    assert intervals == [
        ("5utr", 0, 4),
        ("cds", 4, 12),
        ("cds", 6, 13),
        ("3utr", 13, 20),
    ]
    assert result.scores.tolist() == [4.0, 8.0, 7.0, 7.0]


def test_scan_legnet_reports_original_coordinates_for_five_prime_truncation():
    X = encode_saluki_transcript("A" * 20, length=12, cds_positions=[4, 7, 10])[None]
    result = scan_legnet_windows(
        X,
        Predictor(_sum_called_bases),
        window_size=8,
        regions="3utr",
        schema="saluki6",
        metadata=[{"transcript_length": 20, "cds_start": 4, "cds_end": 13}],
        progress=False,
    )

    instance = result.instances[0]
    assert instance.coordinate_offset == 8
    assert instance.coordinate_source == "transcript_metadata"
    assert (instance.encoded_start, instance.encoded_end) == (5, 12)
    assert (instance.transcript_start, instance.transcript_end) == (13, 20)


def test_scan_legnet_reports_original_coordinates_for_three_prime_truncation():
    X = encode_saluki_transcript(
        "A" * 20,
        length=12,
        cds_positions=[4, 7, 10, 13, 16],
        truncate_from="3prime",
    )[None]
    result = scan_legnet_windows(
        X,
        Predictor(_sum_called_bases),
        window_size=8,
        regions="cds",
        schema="saluki6",
        metadata=[
            {
                "transcript_length": 20,
                "represented_length": 12,
                "encoded_offset": 0,
                "cds_start": 4,
                "cds_end": 19,
            }
        ],
        progress=False,
    )

    instance = result.instances[0]
    assert instance.coordinate_offset == 0
    assert instance.coordinate_source == "transcript_metadata"
    assert (instance.encoded_start, instance.encoded_end) == (4, 12)
    assert (instance.transcript_start, instance.transcript_end) == (4, 12)


def test_scan_legnet_cli_writes_complete_output(tmp_path):
    model_config = {
        "name": "legnet",
        "params": {
            "in_ch": 4,
            "stem_ch": 2,
            "stem_ks": 3,
            "ef_ks": 3,
            "ef_block_sizes": [2],
            "pool_sizes": [1],
            "resize_factor": 1,
            "head_dropout": 0.0,
            "output_dim": 1,
        },
    }
    checkpoint = tmp_path / "legnet.pt"
    save_checkpoint(checkpoint, build_model(model_config), model_config)
    bundle = DatasetBundle(
        X=encode_saluki_transcript("ACGUACGUACGU", length=12, cds_positions=[2, 5])[None],
        ids=["tx1"],
        schema="saluki6",
        metadata=[{"transcript_length": 12, "cds_start": 2, "cds_end": 8, "gene": "G1"}],
    )
    dataset = tmp_path / "bundle"
    save_bundle(bundle, dataset)
    out = tmp_path / "scan"

    main(
        [
            "scan-legnet",
            "--checkpoint",
            str(checkpoint),
            "--dataset",
            str(dataset),
            "--out-dir",
            str(out),
            "--window-size",
            "4",
            "--save-sequences",
            "--device",
            "cpu",
        ]
    )

    assert np.load(out / "scores.npy").shape == (1,)
    assert np.load(out / "sequences.npy").shape == (1, 4, 4)
    assert (out / "instances.csv").exists()
    assert (out / "transcript_scores.csv").exists()
    assert (out / "transcript_region_scores.csv").exists()
    assert json.loads((out / "summary.json").read_text(encoding="utf-8"))["regions"] == ["3utr"]


def test_scan_legnet_cli_rejects_non_legnet_checkpoint(tmp_path):
    model_config = {
        "name": "small_cnn",
        "params": {"in_ch": 4, "n_filters": 2, "kernel_size": 3, "n_layers": 1},
    }
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, build_model(model_config), model_config)
    bundle = DatasetBundle(
        X=encode_saluki_transcript("ACGU", length=4)[None],
        schema="saluki6",
    )
    dataset = tmp_path / "bundle"
    save_bundle(bundle, dataset)

    with pytest.raises(ValueError, match="requires a checkpoint"):
        main(
            [
                "scan-legnet",
                "--checkpoint",
                str(checkpoint),
                "--dataset",
                str(dataset),
                "--out-dir",
                str(tmp_path / "out"),
                "--window-size",
                "4",
                "--regions",
                "transcript",
            ]
        )
