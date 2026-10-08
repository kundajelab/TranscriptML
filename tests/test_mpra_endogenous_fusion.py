from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from transcriptml.data.builders import build_saluki_dataset_from_gtf
from transcriptml.data.encoding import decode_rna_one_hot, encode_saluki_transcript


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "mpra_endogenous_fusion.py"
SPEC = importlib.util.spec_from_file_location("mpra_endogenous_fusion", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
fusion = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fusion)


def _write_inputs(tmp_path: Path) -> dict[str, Path]:
    fasta = tmp_path / "genome.fa"
    fasta.write_text(">chr1\nAAACCCGGGTTT\n", encoding="utf-8")
    gtf = tmp_path / "transcripts.gtf"
    gtf.write_text(
        "\n".join(
            [
                'chr1\ttest\texon\t1\t3\t.\t+\t.\tgene_id "G1"; transcript_id "tx1";',
                'chr1\ttest\texon\t7\t9\t.\t+\t.\tgene_id "G1"; transcript_id "tx1";',
                'chr1\ttest\tCDS\t1\t3\t.\t+\t0\tgene_id "G1"; transcript_id "tx1";',
                'chr1\ttest\tCDS\t7\t9\t.\t+\t0\tgene_id "G1"; transcript_id "tx1";',
                'chrZ\ttest\texon\t1\t3\t.\t+\t.\tgene_id "G2"; transcript_id "tx_chrZ";',
                "",
            ]
        ),
        encoding="utf-8",
    )
    endogenous = tmp_path / "endogenous.csv"
    endogenous.write_text(
        "transcript_id,target,split,note\n"
        "tx1,1.0,train,kept\n"
        "tx_chrZ,9.0,test,no_fasta\n"
        "missing,8.0,,no_gtf\n",
        encoding="utf-8",
    )
    mpra = tmp_path / "mpra.csv"
    mpra.write_text(
        "variant,insert,target,split,batch\n"
        "v1,AC,2.0,val,b1\n"
        "v2,T,3.0,,b2\n",
        encoding="utf-8",
    )
    construct = tmp_path / "construct.txt"
    construct.write_text(
        "5utr\tAAA\n"
        "orf\tATGCCCTAG\n"
        "3utr.1\tGG\n"
        "3utr.2\tTT\n",
        encoding="utf-8",
    )
    return {
        "fasta": fasta,
        "gtf": gtf,
        "endogenous": endogenous,
        "mpra": mpra,
        "construct": construct,
    }


def _args(paths: dict[str, Path], out_dir: Path, *extra: str):
    return fusion.build_parser().parse_args(
        [
            "--endogenous-table",
            str(paths["endogenous"]),
            "--gtf",
            str(paths["gtf"]),
            "--fasta",
            str(paths["fasta"]),
            "--endogenous-target-col",
            "target",
            "--endogenous-split-col",
            "split",
            "--mpra-table",
            str(paths["mpra"]),
            "--mpra-sequence-col",
            "insert",
            "--mpra-target-col",
            "target",
            "--mpra-id-col",
            "variant",
            "--mpra-split-col",
            "split",
            "--construct",
            str(paths["construct"]),
            "--out-dir",
            str(out_dir),
            "--length",
            "24",
            *extra,
        ]
    )


def test_fusion_bundle_contents_encoding_metadata_splits_and_parity(tmp_path):
    paths = _write_inputs(tmp_path)
    out_dir = tmp_path / "fusion"
    fusion.build_fusion(_args(paths, out_dir))

    assert {path.name for path in out_dir.iterdir()} == {
        "X.npy",
        "y.npy",
        "ids.txt",
        "metadata.json",
        "schema.json",
        "config.json",
        "splits.json",
    }
    X = np.load(out_dir / "X.npy")
    y = np.load(out_dir / "y.npy")
    ids = (out_dir / "ids.txt").read_text(encoding="utf-8").splitlines()
    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    schema = json.loads((out_dir / "schema.json").read_text(encoding="utf-8"))
    splits = json.loads((out_dir / "splits.json").read_text(encoding="utf-8"))
    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))

    assert X.shape == (3, 6, 24)
    np.testing.assert_allclose(y, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert ids == ["endogenous:tx1", "mpra:v1", "mpra:v2"]
    assert schema["name"] == "saluki6"
    assert splits == {"train": [0], "val": [1], "test": []}

    parity_dir = tmp_path / "endogenous_only"
    parity = build_saluki_dataset_from_gtf(
        gtf_path=paths["gtf"],
        fasta_path=paths["fasta"],
        out_dir=parity_dir,
        targets_path=paths["endogenous"],
        target_col="target",
        target_id_col="transcript_id",
        split_col="split",
        length=24,
        progress=False,
    )
    np.testing.assert_array_equal(X[0], parity.X[0])

    expected_reporter = "AAAATGCCCTAGGGACTT"
    assert decode_rna_one_hot(X[1, :4]) == expected_reporter.replace("T", "U") + "N" * 6
    assert np.flatnonzero(X[1, 4]).tolist() == [3, 6, 9]
    assert not np.any(X[1, 5])
    assert metadata[0]["data_source"] == "endogenous"
    assert metadata[0]["source_id"] == "tx1"
    assert metadata[1]["data_source"] == "mpra"
    assert metadata[1]["source_id"] == "v1"
    assert metadata[1]["five_prime_utr_start"] == 0
    assert metadata[1]["five_prime_utr_end"] == 3
    assert metadata[1]["cds_start"] == 3
    assert metadata[1]["cds_end"] == 12
    assert metadata[1]["three_prime_utr_1_start"] == 12
    assert metadata[1]["three_prime_utr_1_end"] == 14
    assert metadata[1]["insert_start"] == 14
    assert metadata[1]["insert_end"] == 16
    assert metadata[1]["three_prime_utr_2_start"] == 16
    assert metadata[1]["three_prime_utr_2_end"] == 18
    assert metadata[1]["transcript_length"] == 18
    assert metadata[1]["represented_length"] == 18
    assert metadata[1]["encoded_offset"] == 0
    assert metadata[1]["batch"] == "b1"
    assert config["builder"] == "mpra_endogenous_fusion"
    assert config["n_endogenous"] == 1
    assert config["n_mpra"] == 2
    assert config["endogenous"]["filtering"]["n_missing_gtf_transcripts"] == 1
    assert config["endogenous"]["filtering"]["n_skipped_missing_fasta_chromosome"] == 1
    assert config["construct"]["orf_includes_terminal_stop"] is True


def test_fusion_reports_progress_for_both_sources_and_publication(tmp_path, capsys):
    paths = _write_inputs(tmp_path)
    out_dir = tmp_path / "fusion"
    fusion.build_fusion(_args(paths, out_dir))

    stderr = capsys.readouterr().err
    assert "build-fusion: build endogenous source" in stderr
    assert "build-fusion: allocate fused array for 3 examples (1 endogenous, 2 MPRA)" in stderr
    assert "build-fusion: copy endogenous encodings: started (0/1 examples)" in stderr
    assert "build-fusion: copy endogenous encodings: finished (1/1 examples)" in stderr
    assert "build-fusion: encode MPRA reporters: started (0/2 reporters)" in stderr
    assert "build-fusion: encode MPRA reporters: finished (2/2 reporters)" in stderr
    assert "build-fusion: write targets and bundle metadata" in stderr
    assert f"build-fusion: publish bundle to {out_dir}" in stderr
    assert "build-fusion: done (1 endogenous + 2 MPRA = 3 examples)" in stderr


@pytest.mark.parametrize(
    ("truncate_from", "expected_offset"),
    [("5prime", 10), ("3prime", 0)],
)
def test_reporter_truncation_matches_saluki_encoder(tmp_path, truncate_from, expected_offset):
    paths = _write_inputs(tmp_path)
    out_dir = tmp_path / f"fusion_{truncate_from}"
    args = _args(
        paths,
        out_dir,
        "--length",
        "8",
        "--truncate-from",
        truncate_from,
    )
    fusion.build_fusion(args)

    X = np.load(out_dir / "X.npy")
    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    reporter = "AAAATGCCCTAGGGACTT"
    expected = encode_saluki_transcript(
        reporter,
        length=8,
        cds_positions=range(3, 12, 3),
        splice_positions=(),
        truncate_from=truncate_from,
    )
    np.testing.assert_array_equal(X[1], expected)
    assert metadata[1]["represented_length"] == 8
    assert metadata[1]["encoded_offset"] == expected_offset


def test_duplicate_mpra_ids_are_rejected(tmp_path):
    paths = _write_inputs(tmp_path)
    paths["mpra"].write_text(
        "variant,insert,target,split\n"
        "dup,AC,2.0,train\n"
        "dup,T,3.0,test\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate MPRA ID"):
        fusion.build_fusion(_args(paths, tmp_path / "fusion"))


@pytest.mark.parametrize(
    ("construct_text", "message"),
    [
        ("5utr\tAAA\norf\tATG\n3utr.1\tGG\n", "Missing construct segment"),
        (
            "5utr\tAAA\norf\tATGA\n3utr.1\tGG\n3utr.2\tTT\n",
            "ORF length must be divisible by three",
        ),
        (
            "5utr\tAAA\norf\tATG\n3utr.1\tGG\n3utr.2\tTT\nextra\tA\n",
            "Unexpected construct segment",
        ),
    ],
)
def test_malformed_construct_is_rejected(tmp_path, construct_text, message):
    paths = _write_inputs(tmp_path)
    paths["construct"].write_text(construct_text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        fusion.build_fusion(_args(paths, tmp_path / "fusion"))


def test_missing_requested_column_is_rejected(tmp_path):
    paths = _write_inputs(tmp_path)
    paths["mpra"].write_text("variant,target\nv1,2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"Missing column\(s\).+insert"):
        fusion.build_fusion(_args(paths, tmp_path / "fusion"))


def test_nonempty_output_directory_is_rejected_without_changes(tmp_path):
    paths = _write_inputs(tmp_path)
    out_dir = tmp_path / "fusion"
    out_dir.mkdir()
    marker = out_dir / "keep.txt"
    marker.write_text("do not replace", encoding="utf-8")

    with pytest.raises(ValueError, match="not empty"):
        fusion.build_fusion(_args(paths, out_dir))
    assert marker.read_text(encoding="utf-8") == "do not replace"
