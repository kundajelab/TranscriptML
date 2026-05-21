import numpy as np

from transcriptml.data.bundle import DatasetBundle, load_bundle, save_bundle
from transcriptml.data.builders import build_saluki_dataset_from_gtf
from transcriptml.data.encoding import (
    decode_rna_one_hot,
    encode_rna_sequence,
    encode_saluki_transcript,
    infer_valid_length,
)
from transcriptml.data.genomics import extract_transcript_records, parse_gtf_attributes
from transcriptml.data.schemas import get_schema


def test_rna_one_hot_t_as_u_unknown_zero():
    x = encode_rna_sequence("ACGTUNX")
    assert x.shape == (4, 7)
    assert x[:, 0].tolist() == [1, 0, 0, 0]
    assert x[:, 1].tolist() == [0, 1, 0, 0]
    assert x[:, 2].tolist() == [0, 0, 1, 0]
    assert x[:, 3].tolist() == [0, 0, 0, 1]
    assert x[:, 4].tolist() == [0, 0, 0, 1]
    assert x[:, 5].sum() == 0
    assert x[:, 6].sum() == 0


def test_saluki_padding_truncation_and_valid_length():
    short = encode_saluki_transcript("AC", length=5)
    assert short.shape == (6, 5)
    assert infer_valid_length(short) == 2
    assert short[:, 2:].sum() == 0

    long = encode_saluki_transcript(
        "ACGUAC",
        length=4,
        cds_positions=[0, 2, 4, 5],
        splice_positions=[1, 5],
    )
    assert infer_valid_length(long) == 4
    assert long[2, 0] == 1
    assert long[3, 1] == 1
    assert long[0, 2] == 1
    assert long[1, 3] == 1
    assert long[4].tolist() == [1, 0, 1, 1]
    assert long[5].tolist() == [0, 0, 0, 1]


def test_schema_definitions():
    assert get_schema("rna4").channels == ("A", "C", "G", "U")
    assert get_schema("saluki6").n_channels == 6
    assert get_schema("saluki6").annotation_channels == ("CDS_codon_start", "splice_junction")


def test_bundle_roundtrip(tmp_path):
    bundle = DatasetBundle(
        X=np.zeros((2, 4, 3), dtype=np.uint8),
        y=np.array([1.0, 2.0], dtype=np.float32),
        ids=["a", "b"],
        schema="rna4",
        metadata=[{"split": "train"}, {"split": "test"}],
        splits={"train": [0], "test": [1]},
    )
    save_bundle(bundle, tmp_path)
    loaded = load_bundle(tmp_path)
    assert loaded.ids == ["a", "b"]
    assert loaded.schema.name == "rna4"
    np.testing.assert_array_equal(loaded.y, bundle.y)


def test_gtf_attribute_parser_handles_gtf_and_gff3_styles():
    attrs = parse_gtf_attributes('gene_id "GENE1"; transcript_id "TX1"; gene_name "ABC";')
    assert attrs["gene_id"] == "GENE1"
    assert attrs["transcript_id"] == "TX1"
    assert attrs["gene_name"] == "ABC"

    attrs = parse_gtf_attributes("gene_id=GENE1;transcript_id=TX1")
    assert attrs == {"gene_id": "GENE1", "transcript_id": "TX1"}


def test_gtf_fasta_extraction_and_saluki_builder(tmp_path):
    fasta = tmp_path / "genome.fa"
    fasta.write_text(">chr1\nAAACCCGGGTTT\n", encoding="utf-8")
    gtf = tmp_path / "tx.gtf"
    gtf.write_text(
        "\n".join(
            [
                'chr1\ttest\texon\t1\t3\t.\t+\t.\tgene_id "G1"; transcript_id "tx_pos";',
                'chr1\ttest\texon\t7\t9\t.\t+\t.\tgene_id "G1"; transcript_id "tx_pos";',
                'chr1\ttest\tCDS\t1\t3\t.\t+\t0\tgene_id "G1"; transcript_id "tx_pos";',
                'chr1\ttest\tCDS\t7\t9\t.\t+\t0\tgene_id "G1"; transcript_id "tx_pos";',
                'chr1\ttest\texon\t1\t3\t.\t-\t.\tgene_id "G2"; transcript_id "tx_neg";',
                'chr1\ttest\texon\t10\t12\t.\t-\t.\tgene_id "G2"; transcript_id "tx_neg";',
                'chr1\ttest\tCDS\t1\t3\t.\t-\t0\tgene_id "G2"; transcript_id "tx_neg";',
                'chr1\ttest\tCDS\t10\t12\t.\t-\t0\tgene_id "G2"; transcript_id "tx_neg";',
                "",
            ]
        ),
        encoding="utf-8",
    )

    records = {r.transcript_id: r for r in extract_transcript_records(gtf, fasta)}
    assert records["tx_pos"].sequence == "AAAGGG"
    assert records["tx_pos"].splice_positions == (2,)
    assert records["tx_pos"].cds_positions == (0, 3)
    assert records["tx_neg"].sequence == "AAATTT"
    assert records["tx_neg"].splice_positions == (2,)
    assert records["tx_neg"].cds_positions == (0, 3)

    targets = tmp_path / "targets.csv"
    targets.write_text(
        "transcript_id,log_kdeg,split\n"
        "tx_neg,2.5,test\n"
        "tx_pos,1.5,train\n"
        "missing,9.0,train\n",
        encoding="utf-8",
    )
    bundle = build_saluki_dataset_from_gtf(
        gtf,
        fasta,
        tmp_path / "bundle",
        targets_path=targets,
        target_col="log_kdeg",
        split_col="split",
        length=8,
    )
    assert bundle.ids == ["tx_neg", "tx_pos"]
    np.testing.assert_allclose(bundle.y, np.array([2.5, 1.5], dtype=np.float32))
    assert bundle.splits == {"train": [1], "val": [], "test": [0]}
    assert bundle.X.shape == (2, 6, 8)
    assert decode_rna_one_hot(bundle.X[0, :4]) == "AAAUUUNN"
    assert bundle.X[0, 4].tolist() == [1, 0, 0, 1, 0, 0, 0, 0]
    assert bundle.X[0, 5].tolist() == [0, 0, 1, 0, 0, 0, 0, 0]

    loaded = load_bundle(tmp_path / "bundle", mmap_mode="r")
    assert loaded.ids == ["tx_neg", "tx_pos"]
    assert loaded.config["builder"] == "saluki_gtf"
    assert loaded.config["n_missing_transcripts"] == 1
