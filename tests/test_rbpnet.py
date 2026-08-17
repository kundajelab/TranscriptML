import csv
import gzip
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
import pysam
import pytest
from scipy.stats import poisson

from transcriptml.cli.main import build_parser
from transcriptml.data.encoding import encode_rna_sequence
from transcriptml.rbpnet.bundle import (
    RBPNetBundleConfig,
    _shifted_materialized_interval,
    jitter_crop_offset,
    load_rbpnet_bundle,
    make_rbpnet_bundle,
)
from transcriptml.rbpnet.coordinates import Exon, Region, Transcript, annotate_regions
from transcriptml.rbpnet.experiment import ProcessedECLIPDataset
from transcriptml.rbpnet.fasta import reverse_complement, transcript_sequence
from transcriptml.rbpnet.preprocessing import PipelineConfig, Sample, preprocess_eclip
from transcriptml.rbpnet.selection import SelectionConfig, load_selection_manifest, select_regions
from transcriptml.rbpnet.serialization import calculate_tpm
from transcriptml.rbpnet.signals import (
    ExonBinIndex,
    alignment_is_transcript_compatible,
    create_signal_store,
    extract_bam_to_store,
    five_prime_reference_position,
    read1_rna_strand,
    write_sparse_signal_chunkwise,
)
from transcriptml.rbpnet.windows import WindowScanConfig, generate_window_bounds, scan_windows


def _tx(strand="+"):
    tx = Transcript(
        f"tx_{strand}", f"gene_{strand}", "G", "T", "protein_coding", "chr1", strand,
        [Exon("chr1", 100, 105, strand), Exon("chr1", 200, 204, strand)],
    )
    tx.finalize()
    return tx


def test_transcript_coordinates_regions_and_minus_sequence(tmp_path):
    plus = _tx("+")
    minus = _tx("-")
    assert [(e.tx_start, e.tx_end) for e in plus.exons] == [(0, 5), (5, 9)]
    assert plus.genome_to_transcript("chr1", 200) == 5
    assert plus.transcript_to_genome(5) == ("chr1", 200, "+")
    assert [(e.start, e.tx_start, e.tx_end) for e in minus.exons] == [(200, 0, 4), (100, 4, 9)]
    assert minus.genome_to_transcript("chr1", 203) == 0
    assert minus.transcript_to_genome(4) == ("chr1", 104, "-")
    minus.feature_intervals = {"CDS": [(102, 105), (200, 202)]}
    assert annotate_regions(minus) == [
        Region(0, 2, "5putr"), Region(2, 7, "cds"), Region(7, 9, "3putr")
    ]

    fasta_path = tmp_path / "genome.fa"
    fasta_path.write_text(">chr1\n" + "A" * 100 + "ACGTA" + "N" * 95 + "CCGG" + "A" * 20 + "\n")
    pysam.faidx(str(fasta_path))
    with pysam.FastaFile(str(fasta_path)) as fasta:
        assert transcript_sequence(fasta, plus) == "ACGTACCGG"
        assert transcript_sequence(fasta, minus) == "CCGGTACGT"


def _gene_tx(strand="+", *, coding=False):
    tx = Transcript(
        f"gene_tx_{strand}", f"gene_{strand}", "", "",
        "protein_coding" if coding else "lncRNA", "chr1", strand,
        [Exon("chr1", 100, 110, strand), Exon("chr1", 200, 210, strand)],
        coordinate_space="gene",
    )
    if coding:
        tx.feature_intervals = {"CDS": [(105, 110), (200, 205)]}
    tx.finalize()
    tx.regions = annotate_regions(tx)
    return tx


def test_gene_coordinates_regions_sequence_and_round_trips(tmp_path):
    plus = _gene_tx("+", coding=True)
    minus = _gene_tx("-", coding=True)
    expected_regions = [
        Region(0, 5, "5putr"),
        Region(5, 10, "cds"),
        Region(10, 100, "intron"),
        Region(100, 105, "cds"),
        Region(105, 110, "3putr"),
    ]
    assert plus.length == minus.length == 110
    assert plus.regions == minus.regions == expected_regions
    assert plus.genome_to_transcript("chr1", 150) == 50
    assert minus.genome_to_transcript("chr1", 150) == 59
    assert plus.transcript_to_genome(50) == ("chr1", 150, "+")
    assert minus.transcript_to_genome(59) == ("chr1", 150, "-")
    for tx in (plus, minus):
        for coordinate in (0, 9, 10, 50, 99, 100, 109):
            chrom, genomic, strand = tx.transcript_to_genome(coordinate)
            assert strand == tx.strand
            assert tx.genome_to_transcript(chrom, genomic) == coordinate

    fasta_path = tmp_path / "genome.fa"
    genomic = ("ACGT" * 80)[:300]
    fasta_path.write_text(">chr1\n" + genomic + "\n")
    pysam.faidx(str(fasta_path))
    with pysam.FastaFile(str(fasta_path)) as fasta:
        expected = genomic[100:210]
        assert transcript_sequence(fasta, plus) == expected
        assert transcript_sequence(fasta, minus) == reverse_complement(expected)


def _alignment(start, cigar, reverse=False):
    read = pysam.AlignedSegment()
    read.query_name = "compatibility"
    read.flag = 16 if reverse else 0
    read.reference_start = start
    read.cigartuples = cigar
    return read


def _junction_tx(strand="+"):
    tx = Transcript(
        f"junction_{strand}", f"g_{strand}", "", "", "protein_coding", "chr1", strand,
        [Exon("chr1", 100, 110, strand), Exon("chr1", 200, 210, strand)],
    )
    tx.finalize()
    return tx


def test_alignment_compatibility_and_library_orientation():
    assert five_prime_reference_position(_alignment(102, ((0, 6),))) == 102
    reverse = _alignment(102, ((0, 6),), reverse=True)
    assert five_prime_reference_position(reverse) == 107
    assert read1_rna_strand(True, "opposite") == "+"
    assert read1_rna_strand(False, "same") == "+"
    assert read1_rna_strand(False, "unstranded") is None
    contained = _alignment(102, ((pysam.CMATCH, 6),))
    junction = _alignment(105, ((pysam.CMATCH, 5), (pysam.CREF_SKIP, 90), (pysam.CMATCH, 5)))
    intronic = _alignment(105, ((pysam.CMATCH, 100),))
    wrong_junction = _alignment(
        104, ((pysam.CMATCH, 5), (pysam.CREF_SKIP, 91), (pysam.CMATCH, 5))
    )
    assert alignment_is_transcript_compatible(contained, _junction_tx())
    assert alignment_is_transcript_compatible(junction, _junction_tx())
    assert alignment_is_transcript_compatible(junction, _junction_tx("-"))
    assert not alignment_is_transcript_compatible(intronic, _junction_tx())
    assert not alignment_is_transcript_compatible(wrong_junction, _junction_tx())

    gene = _gene_tx("+")
    assert alignment_is_transcript_compatible(_alignment(102, ((pysam.CMATCH, 6),)), gene)
    assert alignment_is_transcript_compatible(_alignment(140, ((pysam.CMATCH, 20),)), gene)
    assert alignment_is_transcript_compatible(junction, gene)
    assert not alignment_is_transcript_compatible(
        _alignment(205, ((pysam.CMATCH, 10),)), gene
    )


def test_bam_assignment_reports_transcript_incompatibility(tmp_path):
    tx = _junction_tx()
    bam_path = tmp_path / "reads.bam"
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": 1000}]}
    with pysam.AlignmentFile(bam_path, "wb", header=header) as bam:
        for name, start, cigar, length in [
            ("contained", 101, ((pysam.CMATCH, 4),), 4),
            ("intronic", 105, ((pysam.CMATCH, 100),), 100),
        ]:
            read = pysam.AlignedSegment()
            read.query_name = name
            read.query_sequence = "A" * length
            read.flag = 81
            read.reference_id = 0
            read.reference_start = start
            read.mapping_quality = 60
            read.cigar = cigar
            read.query_qualities = pysam.qualitystring_to_array("I" * length)
            bam.write(read)
    pysam.index(str(bam_path))
    with create_signal_store(tmp_path / "signals.h5", [tx], ["ip"], ["ip"]) as store:
        qc, counts = extract_bam_to_store(
            bam_path, 0, store["counts"], [tx], ExonBinIndex([tx]),
            "opposite", 1, True, tmp_path, progress=False,
        )
    assert qc["retained"] == 1
    assert qc["transcript_incompatible"] == 1
    assert counts.tolist() == [1]


@pytest.mark.parametrize(
    ("compression", "level"),
    [("gzip", 1), ("gzip", 4), ("lzf", None), (None, None)],
)
def test_chunkwise_sparse_signal_write_matches_fancy_indexing_and_skips_empty_chunks(
    tmp_path, compression, level
):
    positions = np.asarray([1, 7, 8, 15, 33, 39], dtype=np.int64)
    values = np.asarray([2, 4, 1, 8, 3, 9], dtype=np.uint64)
    expected = np.zeros(40, dtype=np.uint32)
    expected[positions] = values.astype(np.uint32)
    path = tmp_path / f"signal_{compression or 'none'}_{level}.h5"
    with h5py.File(path, "w") as store:
        kwargs = {"compression": compression, "shuffle": compression is not None}
        if compression == "gzip":
            kwargs["compression_opts"] = level
        dataset = store.create_dataset(
            "counts",
            shape=(1, 40),
            dtype=np.uint32,
            chunks=(1, 8),
            fillvalue=0,
            **kwargs,
        )
        batches = [
            (positions[:3], values[:3]),
            (positions[3:5], values[3:5]),
            (positions[5:], values[5:]),
        ]
        assert write_sparse_signal_chunkwise(dataset, 0, batches) == len(positions)
        np.testing.assert_array_equal(dataset[0], expected)
        # Positions touch chunks 0, 1, and 4. Chunks 2 and 3 stay at the
        # HDF5 fill value and are never explicitly allocated.
        if hasattr(dataset.id, "get_num_chunks"):
            assert dataset.id.get_num_chunks() == 3


def test_chunkwise_sparse_signal_write_validates_sorted_unique_positions(tmp_path):
    with h5py.File(tmp_path / "invalid.h5", "w") as store:
        dataset = store.create_dataset(
            "counts", shape=(1, 16), dtype=np.uint32, chunks=(1, 8), fillvalue=0
        )
        with pytest.raises(ValueError, match="strictly increasing"):
            write_sparse_signal_chunkwise(
                dataset,
                0,
                [(np.asarray([4, 3]), np.asarray([1, 1]))],
            )


def test_gene_space_bam_assignment_retains_intronic_and_spliced_reads(tmp_path):
    gene = _gene_tx("+")
    mature = _junction_tx("+")
    bam_path = tmp_path / "gene_reads.bam"
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": 1000}]}
    records = [
        ("exonic", 101, ((pysam.CMATCH, 4),), 4),
        ("junction", 105, ((pysam.CMATCH, 5), (pysam.CREF_SKIP, 90), (pysam.CMATCH, 5)), 10),
        ("intronic", 150, ((pysam.CMATCH, 4),), 4),
    ]
    with pysam.AlignmentFile(bam_path, "wb", header=header) as bam:
        for name, start, cigar, query_length in records:
            read = pysam.AlignedSegment()
            read.query_name = name
            read.query_sequence = "A" * query_length
            read.flag = 81  # read1, reverse; opposite-strand RNA is plus
            read.reference_id = 0
            read.reference_start = start
            read.mapping_quality = 60
            read.cigar = cigar
            read.query_qualities = pysam.qualitystring_to_array("I" * query_length)
            bam.write(read)
    pysam.index(str(bam_path))

    with create_signal_store(tmp_path / "gene.h5", [gene], ["ip"], ["ip"]) as store:
        gene_qc, gene_counts = extract_bam_to_store(
            bam_path, 0, store["counts"], [gene], ExonBinIndex([gene]),
            "opposite", 1, True, tmp_path, progress=False,
        )
    with create_signal_store(tmp_path / "mature.h5", [mature], ["ip"], ["ip"]) as store:
        mature_qc, mature_counts = extract_bam_to_store(
            bam_path, 0, store["counts"], [mature], ExonBinIndex([mature]),
            "opposite", 1, True, tmp_path, progress=False,
        )
    assert gene_qc["retained"] == 3
    assert mature_qc["retained"] == 2
    assert mature_qc["no_compatible_transcript"] == 1
    assert gene_counts.tolist() == [3]
    assert mature_counts.tolist() == [2]


def test_preprocess_pipeline_manifest_tpm_effective_sizes_and_missing_contig(tmp_path):
    fasta = tmp_path / "genome.fa"
    fasta.write_text(">chr1\n" + "ACGT" * 20 + "\n")
    gtf = tmp_path / "annotation.gtf"
    gtf.write_text(
        'chr1\ttest\ttranscript\t1\t20\t.\t+\t.\tgene_id "g1"; transcript_id "t1"; transcript_type "lncRNA";\n'
        'chr1\ttest\texon\t1\t20\t.\t+\t.\tgene_id "g1"; transcript_id "t1"; exon_number 1;\n'
        'chrPatch\ttest\ttranscript\t1\t20\t.\t+\t.\tgene_id "g2"; transcript_id "t2"; transcript_type "lncRNA";\n'
        'chrPatch\ttest\texon\t1\t20\t.\t+\t.\tgene_id "g2"; transcript_id "t2"; exon_number 1;\n'
    )
    bam_path = tmp_path / "reads.bam"
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": 80}]}
    with pysam.AlignmentFile(bam_path, "wb", header=header) as bam:
        read = pysam.AlignedSegment()
        read.query_name = "retained_read1"
        read.query_sequence = "AAAA"
        read.flag = 81
        read.reference_id = 0
        read.reference_start = 4
        read.mapping_quality = 60
        read.cigar = ((pysam.CMATCH, 4),)
        read.query_qualities = pysam.qualitystring_to_array("IIII")
        bam.write(read)
    pysam.index(str(bam_path))
    output = tmp_path / "processed"
    qc = preprocess_eclip(PipelineConfig(
        genome_fasta=fasta,
        gtf=gtf,
        sminput=Sample("sminput", bam_path, "sminput"),
        ips=(Sample("ip1", bam_path, "ip"), Sample("ip2", bam_path, "ip")),
        output_dir=output,
        progress=False,
    ))
    assert qc["annotation"]["transcripts"] == 1
    assert qc["annotation"]["transcripts_skipped_missing_fasta_contig"] == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert [sample["effective_library_size"] for sample in manifest["samples"]] == [1, 1, 1]
    assert manifest["derived_signals"]["ip_pooled"]["effective_library_size"] == 2
    with (output / "transcripts.tsv").open() as handle:
        row = next(csv.DictReader(handle, delimiter="\t"))
    assert int(row["sm_input_raw_count"]) == 1
    assert float(row["sm_input_tpm"]) == pytest.approx(1_000_000)
    with ProcessedECLIPDataset(output) as ds:
        assert ds.get_sequence("t1", 0, 20) == ("ACGT" * 5)
        assert int(ds.get_pooled_ip_profile("t1", 0, 20).sum()) == 2


def test_full_gene_preprocessing_reader_orientation_introns_and_mature_regression(tmp_path):
    fasta = tmp_path / "genome.fa"
    genomic = ("ACGT" * 300)[:1000]
    fasta.write_text(">chr1\n" + genomic + "\n")
    gtf = tmp_path / "annotation.gtf"
    gtf.write_text(
        'chr1\ttest\ttranscript\t101\t210\t.\t+\t.\tgene_id "gp"; transcript_id "tp"; transcript_type "lncRNA";\n'
        'chr1\ttest\texon\t101\t110\t.\t+\t.\tgene_id "gp"; transcript_id "tp"; exon_number 1;\n'
        'chr1\ttest\texon\t201\t210\t.\t+\t.\tgene_id "gp"; transcript_id "tp"; exon_number 2;\n'
        'chr1\ttest\ttranscript\t401\t510\t.\t-\t.\tgene_id "gm"; transcript_id "tm"; transcript_type "lncRNA";\n'
        'chr1\ttest\texon\t401\t410\t.\t-\t.\tgene_id "gm"; transcript_id "tm"; exon_number 2;\n'
        'chr1\ttest\texon\t501\t510\t.\t-\t.\tgene_id "gm"; transcript_id "tm"; exon_number 1;\n'
    )
    bam_path = tmp_path / "reads.bam"
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": 1000}]}
    records = [
        ("plus_junction", 105, ((pysam.CMATCH, 5), (pysam.CREF_SKIP, 90), (pysam.CMATCH, 5)), 10, 81),
        ("plus_intron", 150, ((pysam.CMATCH, 4),), 4, 81),
        ("minus_junction", 405, ((pysam.CMATCH, 5), (pysam.CREF_SKIP, 90), (pysam.CMATCH, 5)), 10, 65),
        ("minus_intron", 450, ((pysam.CMATCH, 4),), 4, 65),
    ]
    with pysam.AlignmentFile(bam_path, "wb", header=header) as bam:
        for name, start, cigar, query_length, flag in records:
            read = pysam.AlignedSegment()
            read.query_name = name
            read.query_sequence = "A" * query_length
            read.flag = flag
            read.reference_id = 0
            read.reference_start = start
            read.mapping_quality = 60
            read.cigar = cigar
            read.query_qualities = pysam.qualitystring_to_array("I" * query_length)
            bam.write(read)
    pysam.index(str(bam_path))

    def run(space, output):
        return preprocess_eclip(PipelineConfig(
            genome_fasta=fasta,
            gtf=gtf,
            sminput=Sample("sminput", bam_path, "sminput"),
            ips=(Sample("ip1", bam_path, "ip"),),
            output_dir=output,
            coordinate_space=space,
            progress=False,
        ))

    mature_dir = tmp_path / "mature"
    gene_dir = tmp_path / "gene"
    mature_qc = run("mature_transcript", mature_dir)
    gene_qc = run("gene", gene_dir)
    assert mature_qc["annotation"]["transcriptome_bases"] == 40
    assert gene_qc["annotation"]["coordinate_space_bases"] == 220
    assert gene_qc["annotation"]["intronic_bases"] == 180
    assert mature_qc["samples"]["sminput"]["retained"] == 2
    assert gene_qc["samples"]["sminput"]["retained"] == 4

    with ProcessedECLIPDataset(gene_dir) as ds:
        assert ds.coordinate_space == "gene"
        assert ds.get_sequence("tp", 0, 110) == genomic[100:210]
        assert ds.get_sequence("tm", 0, 110) == reverse_complement(genomic[400:510])
        assert ds.genome_to_coordinate("tp", "chr1", 150) == 50
        assert ds.genome_to_coordinate("tm", "chr1", 450) == 59
        assert ds.coordinate_to_genome("tp", 50) == ("chr1", 150, "+")
        assert ds.coordinate_to_genome("tm", 59) == ("chr1", 450, "-")
        assert [(b.start, b.end) for b in ds.get_genomic_blocks("tm", 10, 100)] == [(410, 500)]
        assert int(ds.get_profile("tp", 50, 60, "sminput").sum()) == 1
        assert int(ds.get_profile("tm", 50, 60, "sminput").sum()) == 1

    windows = tmp_path / "gene_windows"
    summary = scan_windows(WindowScanConfig(
        processed_dir=gene_dir,
        output_prefix=windows,
        window_size=10,
        stride=10,
        progress=False,
    ))
    assert summary["coordinate_space"] == "gene"
    assert summary["region_type_windows"]["intron"] == 18
    selected = tmp_path / "gene_selected"
    select_regions(SelectionConfig(
        processed_dir=gene_dir,
        windows=windows,
        output_prefix=selected,
        strategy="broad_coverage",
        replicate_mode="combined",
        min_total_count=0,
        progress=False,
    ))
    bundle = make_rbpnet_bundle(RBPNetBundleConfig(
        processed_dir=gene_dir,
        selection_manifest=selected,
        output_dir=tmp_path / "gene_bundle",
        input_length=20,
        profile_length=20,
        max_jitter=2,
        transcript_end_policy="shift_to_fit",
        progress=False,
    ))
    assert bundle.config["coordinate_space"] == "gene"
    with ProcessedECLIPDataset(gene_dir) as ds:
        for index, item in enumerate(bundle.metadata):
            start, end = item["sequence_materialized_start"], item["sequence_materialized_end"]
            np.testing.assert_array_equal(
                bundle.X[index],
                encode_rna_sequence(ds.get_sequence(item["transcript_id"], start, end)),
            )


def _write_processed_fixture(root: Path, *, length=12, profiles=None):
    root.mkdir(parents=True)
    sequence = ("ACGTGCGTAAAA" * ((length + 11) // 12))[:length]
    (root / "transcripts.fa").write_text(f">tx1\n{sequence}\n")
    pysam.faidx(str(root / "transcripts.fa"))
    if length == 12:
        regions = [
            {"start": 0, "end": 4, "type": "5putr"},
            {"start": 4, "end": 9, "type": "cds"},
            {"start": 9, "end": 12, "type": "3putr"},
        ]
        exons = [(0, 5, 100, 105), (5, 12, 200, 207)]
    else:
        regions = [{"start": 0, "end": length, "type": "noncoding_exon"}]
        exons = [(0, length, 100, 100 + length)]
    with (root / "transcripts.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "transcript_id", "gene_id", "chrom", "strand", "transcript_length",
                "signal_offset", "sm_input_tpm", "region_annotations",
            ],
            delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow({
            "transcript_id": "tx1", "gene_id": "gene1", "chrom": "chr1", "strand": "+",
            "transcript_length": length, "signal_offset": 0, "sm_input_tpm": 5.0,
            "region_annotations": json.dumps(regions, separators=(",", ":")),
        })
    with gzip.open(root / "exons.tsv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "transcript_id", "tx_start", "tx_end", "chrom", "genomic_start",
                "genomic_end", "strand",
            ],
            delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        for tx_start, tx_end, genomic_start, genomic_end in exons:
            writer.writerow({
                "transcript_id": "tx1", "tx_start": tx_start, "tx_end": tx_end,
                "chrom": "chr1", "genomic_start": genomic_start, "genomic_end": genomic_end,
                "strand": "+",
            })
    if profiles is None:
        profiles = np.asarray([
            [0, 1, 0, 2, 0, 0, 3, 0, 0, 0, 1, 0],
            [1, 0, 0, 1, 0, 4, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1],
        ], dtype=np.uint32)
    with h5py.File(root / "signals.h5", "w") as h5:
        strings = h5py.string_dtype("utf-8")
        h5.create_dataset("counts", data=profiles)
        h5.create_dataset("ip_pooled", data=profiles[1:].sum(axis=0, dtype=np.uint32))
        h5.create_dataset("sample_names", data=np.asarray(["sminput", "ipA", "ipB"], dtype=object), dtype=strings)
        h5.create_dataset("sample_roles", data=np.asarray(["sminput", "ip", "ip"], dtype=object), dtype=strings)
        h5.create_dataset("transcript_ids", data=np.asarray(["tx1"], dtype=object), dtype=strings)
        h5.create_dataset("transcript_offsets", data=np.asarray([0], dtype=np.int64))
        h5.create_dataset("transcript_lengths", data=np.asarray([length], dtype=np.int64))
    manifest = {
        "format": "transcriptml-rbpnet-experiment", "format_version": "1",
        "files": {
            "metadata": "transcripts.tsv", "exon_mapping": "exons.tsv.gz",
            "sequences": "transcripts.fa", "signals": "signals.h5",
        },
        "samples": [
            {"name": "sminput", "role": "sminput", "effective_library_size": 7},
            {"name": "ipA", "role": "ip", "effective_library_size": 6},
            {"name": "ipB", "role": "ip", "effective_library_size": 6},
        ],
        "derived_signals": {"ip_pooled": {"effective_library_size": 12}},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return sequence, profiles


def test_tpm_reader_window_counts_mixed_regions_and_normalization(tmp_path):
    tx1 = Transcript("a", "g1", "", "", "lncRNA", "chr1", "+", [Exon("chr1", 0, 100, "+")])
    tx2 = Transcript("b", "g2", "", "", "lncRNA", "chr1", "+", [Exon("chr1", 0, 200, "+")])
    tx1.finalize()
    tx2.finalize()
    np.testing.assert_allclose(calculate_tpm(np.array([10, 10]), [tx1, tx2]), [2 / 3 * 1e6, 1 / 3 * 1e6])

    root = tmp_path / "processed"
    sequence, profiles = _write_processed_fixture(root)
    with ProcessedECLIPDataset(root) as ds:
        assert ds.get_sequence("tx1", 4, 8) == sequence[4:8]
        np.testing.assert_array_equal(ds.get_profile("tx1", 4, 8, "ipA"), profiles[1, 4:8])
        assert [(b.start, b.end) for b in ds.get_genomic_blocks("tx1", 4, 8)] == [(104, 105), (200, 203)]
    assert list(generate_window_bounds(10, 4, 4, True)) == [(0, 4), (4, 8)]
    assert list(generate_window_bounds(10, 4, 4, False)) == [(0, 4), (4, 8), (8, 10)]

    prefix = tmp_path / "windows"
    summary = scan_windows(WindowScanConfig(
        processed_dir=root, output_prefix=prefix, window_size=4, stride=4, progress=False,
    ))
    assert summary["region_type_windows"] == {"5putr": 1, "cds": 1, "mixed": 1}
    rows = pq.read_table(str(prefix) + ".parquet").to_pylist()
    junction = rows[1]
    np.testing.assert_array_equal(
        [junction[f"{name}_count"] for name in ("sminput", "ipA", "ipB")],
        profiles[:, 4:8].sum(axis=1),
    )
    assert junction["ip_pooled_count"] == int(profiles[1:, 4:8].sum())
    assert junction["sminput_cpm"] == pytest.approx(3 / 7 * 1e6)
    assert junction["ip_pooled_cpm"] == pytest.approx(6 / 12 * 1e6)
    assert junction["max_ipA_5pend"] == 4
    assert rows[2]["region_type"] == "mixed"
    assert rows[2]["region_cds_nt"] == 1 and rows[2]["region_3putr_nt"] == 3


def test_selection_strategies_ids_serialization_and_stitching(tmp_path):
    root = tmp_path / "processed"
    _write_processed_fixture(root)
    windows = tmp_path / "windows"
    scan_windows(WindowScanConfig(
        processed_dir=root, output_prefix=windows, window_size=4, stride=2, progress=False,
    ))
    broad = tmp_path / "broad"
    summary = select_regions(SelectionConfig(
        processed_dir=root, windows=windows, output_prefix=broad, strategy="broad_coverage",
        min_total_count=1, min_sminput_count=0, min_ip_count=0, progress=False,
    ))
    loaded = load_selection_manifest(broad)
    assert summary["n_examples"] == len(loaded.rows) > 0
    assert summary["configuration"]["replicate_mode"] == "per_ip"
    assert len({row["example_id"] for row in loaded.rows}) == len(loaded.rows)
    broad2 = tmp_path / "broad2"
    select_regions(SelectionConfig(
        processed_dir=root, windows=windows, output_prefix=broad2, strategy="broad_coverage",
        min_total_count=1, min_sminput_count=0, min_ip_count=0, progress=False,
    ))
    assert [r["example_id"] for r in loaded.rows] == [
        r["example_id"] for r in load_selection_manifest(broad2).rows
    ]
    per_ip = tmp_path / "broad_per_ip"
    select_regions(SelectionConfig(
        processed_dir=root, windows=windows, output_prefix=per_ip, strategy="broad_coverage",
        min_total_count=0, min_sminput_count=0, min_ip_count=0,
        replicate_mode="per_ip", progress=False,
    ))
    assert {row["replicate_id"] for row in load_selection_manifest(per_ip).rows} == {"ipA", "ipB"}

    classified = tmp_path / "classified"
    classified_summary = select_regions(SelectionConfig(
        processed_dir=root, windows=windows, output_prefix=classified,
        strategy="peak_gray_negative", min_total_count=1, min_sminput_count=0,
        min_ip_count=0, peak_fdr=1.0, negative_fdr=1.0,
        peak_min_log2_ratio=100.0, negative_max_log2_ratio=-100.0,
        progress=False,
    ))
    assert classified_summary["n_examples"] > 0
    states = set(classified_summary["state_counts"])
    assert states <= {"peak", "gray", "confident_negative"}
    assert any(row["source_window_count"] > 1 for row in load_selection_manifest(classified).rows)


def test_region_type_filtering_includes_matching_mixed_by_default(tmp_path):
    root = tmp_path / "processed"
    _write_processed_fixture(root)
    windows = tmp_path / "windows"
    scan_windows(WindowScanConfig(
        processed_dir=root, output_prefix=windows, window_size=4, stride=2,
        progress=False,
    ))

    broad = tmp_path / "broad_regions"
    broad_summary = select_regions(SelectionConfig(
        processed_dir=root,
        windows=windows,
        output_prefix=broad,
        strategy="broad_coverage",
        replicate_mode="combined",
        min_total_count=0,
        region_types=("5putr", "cds"),
        progress=False,
    ))
    broad_manifest = load_selection_manifest(broad)
    assert [row["region_type"] for row in broad_manifest.rows] == [
        "5putr", "mixed", "cds", "mixed", "mixed"
    ]
    assert broad_summary["region_filter"] == {
        "mode": "overlap",
        "requested_region_types": ["5putr", "cds"],
        "mixed_policy": "include_matching",
        "source_window_counts": {"5putr": 1, "cds": 1, "mixed": 3},
        "eligible_window_counts": {"5putr": 1, "cds": 1, "mixed": 3},
    }
    assert broad_summary["selected_example_region_counts"] == {
        "5putr": 1,
        "cds": 1,
        "mixed": 3,
    }
    assert broad_manifest.metadata["configuration"]["region_types"] == [
        "5putr",
        "cds",
    ]

    matching_3putr = tmp_path / "broad_3putr"
    matching_summary = select_regions(SelectionConfig(
        processed_dir=root,
        windows=windows,
        output_prefix=matching_3putr,
        strategy="broad_coverage",
        replicate_mode="combined",
        min_total_count=0,
        region_types="3putr",
        progress=False,
    ))
    assert matching_summary["n_examples"] == 2
    assert {row["region_type"] for row in load_selection_manifest(matching_3putr).rows} == {
        "mixed"
    }

    pure = tmp_path / "broad_pure"
    pure_summary = select_regions(SelectionConfig(
        processed_dir=root,
        windows=windows,
        output_prefix=pure,
        strategy="broad_coverage",
        replicate_mode="combined",
        min_total_count=0,
        region_types=("5putr", "cds"),
        discard_mixed=True,
        progress=False,
    ))
    assert [row["region_type"] for row in load_selection_manifest(pure).rows] == [
        "5putr", "cds"
    ]
    assert pure_summary["region_filter"]["mixed_policy"] == "discard"

    only_mixed = tmp_path / "broad_only_mixed"
    only_mixed_summary = select_regions(SelectionConfig(
        processed_dir=root,
        windows=windows,
        output_prefix=only_mixed,
        strategy="broad_coverage",
        replicate_mode="combined",
        min_total_count=0,
        region_types=("3putr",),
        only_mixed=True,
        progress=False,
    ))
    assert only_mixed_summary["n_examples"] == 2
    assert only_mixed_summary["region_filter"]["mixed_policy"] == "only"
    assert {row["region_type"] for row in load_selection_manifest(only_mixed).rows} == {
        "mixed"
    }

    classified = tmp_path / "classified_cds"
    select_regions(SelectionConfig(
        processed_dir=root,
        windows=windows,
        output_prefix=classified,
        strategy="peak_gray_negative",
        min_total_count=0,
        peak_fdr=1.0,
        negative_fdr=1.0,
        peak_min_log2_ratio=100.0,
        negative_max_log2_ratio=-100.0,
        region_types=("cds",),
        discard_mixed=True,
        progress=False,
    ))
    classified_rows = load_selection_manifest(classified).rows
    assert len(classified_rows) == 1
    assert classified_rows[0]["region_type"] == "cds"
    # CDS contributes one adequately measured test, so BH adjustment over the
    # requested region universe leaves each exact-tail p-value unchanged.
    assert classified_rows[0]["source_min_enrichment_qvalue"] == pytest.approx(
        classified_rows[0]["source_min_enrichment_pvalue"]
    )
    assert classified_rows[0]["source_min_depletion_qvalue"] == pytest.approx(
        classified_rows[0]["source_min_depletion_pvalue"]
    )

    classified_boundary = tmp_path / "classified_3putr_boundary"
    select_regions(SelectionConfig(
        processed_dir=root,
        windows=windows,
        output_prefix=classified_boundary,
        strategy="peak_gray_negative",
        min_total_count=0,
        peak_fdr=1.0,
        negative_fdr=1.0,
        peak_min_log2_ratio=100.0,
        negative_max_log2_ratio=-100.0,
        region_types=("3putr",),
        only_mixed=True,
        progress=False,
    ))
    boundary_rows = load_selection_manifest(classified_boundary).rows
    assert boundary_rows
    assert all(row["region_type"] == "mixed" for row in boundary_rows)
    assert all(row["region_3putr_nt"] > 0 for row in boundary_rows)

    with pytest.raises(ValueError, match="unsupported region_types: mixed"):
        select_regions(SelectionConfig(
            processed_dir=root,
            windows=windows,
            output_prefix=tmp_path / "invalid_region",
            strategy="broad_coverage",
            region_types=("mixed",),
            progress=False,
        ))
    with pytest.raises(ValueError, match="only_mixed requires"):
        select_regions(SelectionConfig(
            processed_dir=root,
            windows=windows,
            output_prefix=tmp_path / "missing_regions",
            strategy="broad_coverage",
            only_mixed=True,
            progress=False,
        ))
    with pytest.raises(ValueError, match="mutually exclusive"):
        select_regions(SelectionConfig(
            processed_dir=root,
            windows=windows,
            output_prefix=tmp_path / "conflicting_mixed",
            strategy="broad_coverage",
            region_types=("cds",),
            discard_mixed=True,
            only_mixed=True,
            progress=False,
        ))


def test_select_regions_cli_parses_comma_separated_region_types():
    args = build_parser().parse_args([
        "rbpnet",
        "select-regions",
        "--processed-dir", "processed",
        "--windows", "windows.parquet",
        "--output-prefix", "selected",
        "--strategy", "broad_coverage",
        "--region-types", "3putr,cds",
        "--only-mixed",
    ])
    assert args.region_types == ("3putr", "cds")
    assert args.only_mixed is True
    assert args.discard_mixed is False


def test_zero_count_peak_negative_edges_and_broad_coverage_defaults(tmp_path):
    profiles = np.zeros((3, 12), dtype=np.uint32)
    profiles[0, 0] = 30       # informative input-only window
    profiles[1, 4] = 30       # informative IP-only window
    root = tmp_path / "processed"
    _write_processed_fixture(root, profiles=profiles)
    windows = tmp_path / "windows"
    scan_windows(WindowScanConfig(
        processed_dir=root, output_prefix=windows, window_size=4, stride=4, progress=False,
    ))

    classified = tmp_path / "classified"
    summary = select_regions(SelectionConfig(
        processed_dir=root,
        windows=windows,
        output_prefix=classified,
        strategy="peak_gray_negative",
        progress=False,
    ))
    rows = load_selection_manifest(classified).rows
    assert summary["configuration"]["min_total_count"] == 8
    assert summary["configuration"]["min_ip_count"] == 0
    assert summary["configuration"]["min_sminput_count"] == 0
    by_start = {row["selection_start"]: row for row in rows}
    assert by_start[0]["selection_state"] == "confident_negative"
    assert by_start[0]["ip_pooled_count"] == 0
    assert by_start[0]["sminput_count"] == 30
    assert by_start[4]["selection_state"] == "peak"
    assert by_start[4]["ip_pooled_count"] == 30
    assert by_start[4]["sminput_count"] == 0
    assert all(np.isfinite(row["log2_ip_pooled_vs_sminput"]) for row in rows)

    combined = tmp_path / "broad_combined"
    combined_summary = select_regions(SelectionConfig(
        processed_dir=root, windows=windows, output_prefix=combined,
        strategy="broad_coverage", replicate_mode="combined", progress=False,
    ))
    assert combined_summary["n_examples"] == 2
    per_ip = tmp_path / "broad_per_ip"
    per_ip_summary = select_regions(SelectionConfig(
        processed_dir=root, windows=windows, output_prefix=per_ip,
        strategy="broad_coverage", replicate_mode="per_ip", progress=False,
    ))
    assert per_ip_summary["n_examples"] == 3
    assert {row["replicate_id"] for row in load_selection_manifest(per_ip).rows} == {"ipA", "ipB"}


def test_original_rbpnet_poisson_selection_and_50nt_advance(tmp_path):
    length = 500
    profiles = np.zeros((3, length), dtype=np.uint32)
    profiles[0, ::50] = 1
    profiles[1, 200] = 10
    profiles[2, 202] = 10
    root = tmp_path / "processed"
    _write_processed_fixture(root, length=length, profiles=profiles)
    windows = tmp_path / "windows_v1"
    scan_windows(WindowScanConfig(
        processed_dir=root, output_prefix=windows, window_size=100, stride=1, progress=False,
    ))
    selected = tmp_path / "original"
    summary = select_regions(SelectionConfig(
        processed_dir=root, windows=windows, output_prefix=selected,
        strategy="original_rbpnet", progress=False,
    ))
    rows = load_selection_manifest(selected).rows
    assert summary["n_examples"] > 0
    assert all(row["selection_length"] == 100 for row in rows)
    starts = [row["selection_start"] for row in rows]
    assert all(right - left >= 50 for left, right in zip(starts, starts[1:]))
    assert all(row["selection_pvalue"] < 0.01 for row in rows)

    explicit = tmp_path / "original_explicit"
    select_regions(SelectionConfig(
        processed_dir=root, windows=windows, output_prefix=explicit,
        strategy="original_rbpnet", poisson_null="ip_locus_density", progress=False,
    ))
    explicit_rows = load_selection_manifest(explicit).rows
    assert [row["example_id"] for row in explicit_rows] == [row["example_id"] for row in rows]
    np.testing.assert_allclose(
        [row["selection_pvalue"] for row in explicit_rows],
        [row["selection_pvalue"] for row in rows],
    )

    matching_region = tmp_path / "original_noncoding"
    matching_summary = select_regions(SelectionConfig(
        processed_dir=root,
        windows=windows,
        output_prefix=matching_region,
        strategy="original_rbpnet",
        region_types=("noncoding_exon",),
        progress=False,
    ))
    assert [row["example_id"] for row in load_selection_manifest(matching_region).rows] == [
        row["example_id"] for row in rows
    ]
    assert matching_summary["region_filter"]["eligible_window_counts"] == {
        "noncoding_exon": 401
    }

    excluded_region = tmp_path / "original_cds"
    excluded_summary = select_regions(SelectionConfig(
        processed_dir=root,
        windows=windows,
        output_prefix=excluded_region,
        strategy="original_rbpnet",
        region_types=("cds",),
        progress=False,
    ))
    assert excluded_summary["n_examples"] == 0
    assert excluded_summary["region_filter"]["eligible_window_counts"] == {}

    sminput_null = tmp_path / "original_sminput"
    sminput_summary = select_regions(SelectionConfig(
        processed_dir=root, windows=windows, output_prefix=sminput_null,
        strategy="original_rbpnet", poisson_null="sminput", progress=False,
    ))
    sminput_rows = load_selection_manifest(sminput_null).rows
    assert sminput_summary["configuration"]["poisson_null"] == "sminput"
    assert sminput_rows
    first = sminput_rows[0]
    expected_mu = (first["sminput_count"] + 1.0) * (12 / 7)
    assert first["selection_null_mean"] == pytest.approx(expected_mu)
    assert first["selection_pvalue"] == pytest.approx(
        poisson.sf(first["ip_pooled_count"] - 1, expected_mu)
    )


def test_materialized_bundle_exact_arrays_jitter_padding_and_mmap(tmp_path):
    root = tmp_path / "processed"
    sequence, profiles = _write_processed_fixture(root)
    windows = tmp_path / "windows"
    scan_windows(WindowScanConfig(
        processed_dir=root, output_prefix=windows, window_size=4, stride=4, progress=False,
    ))
    selection = tmp_path / "selection"
    select_regions(SelectionConfig(
        processed_dir=root, windows=windows, output_prefix=selection, strategy="broad_coverage",
        min_total_count=0, min_sminput_count=0, min_ip_count=0, progress=False,
    ))
    out = tmp_path / "bundle"
    built = make_rbpnet_bundle(RBPNetBundleConfig(
        processed_dir=root, selection_manifest=selection, output_dir=out,
        input_length=4, profile_length=4, max_jitter=2,
        transcript_end_policy="pad", progress=False,
    ))
    assert built.X.dtype == np.uint8 and built.X.shape[1:] == (4, 8)
    assert built.arrays["sminput_profiles"].dtype == np.uint32
    assert built.arrays["ip_profiles"].shape == (len(built.ids), 2, 8)
    assert built.arrays["profile_ip_totals"].dtype == np.uint64
    first = built.metadata[0]
    start, end = first["sequence_context_start"], first["sequence_context_end"]
    src_start, src_end = max(0, start), min(len(sequence), end)
    destination = src_start - start
    expected = np.zeros((4, 8), dtype=np.uint8)
    expected[:, destination:destination + src_end - src_start] = encode_rna_sequence(
        sequence[src_start:src_end]
    )
    np.testing.assert_array_equal(built.X[0], expected)
    pstart, pend = first["profile_context_start"], first["profile_context_end"]
    psrc_start, psrc_end = max(0, pstart), min(len(sequence), pend)
    pdestination = psrc_start - pstart
    expected_profiles = np.zeros((3, 8), dtype=np.uint32)
    expected_profiles[:, pdestination:pdestination + psrc_end - psrc_start] = profiles[:, psrc_start:psrc_end]
    np.testing.assert_array_equal(built.arrays["sminput_profiles"][0], expected_profiles[0])
    np.testing.assert_array_equal(built.arrays["ip_profiles"][0], expected_profiles[1:])
    np.testing.assert_array_equal(
        built.arrays["profile_ip_totals"][0], expected_profiles[1:].sum(axis=1, dtype=np.uint64)
    )
    loaded = load_rbpnet_bundle(out, mmap_mode="r")
    assert isinstance(loaded.X, np.memmap)
    assert isinstance(loaded.arrays["ip_profiles"], np.memmap)
    assert loaded.config["max_jitter"] == 2
    assert loaded.config["sample_metadata"]["ip_axis_order"] == ["ipA", "ipB"]
    assert (out / "examples.parquet").is_file()

    dropped_out = tmp_path / "bundle_drop"
    dropped = make_rbpnet_bundle(RBPNetBundleConfig(
        processed_dir=root, selection_manifest=selection, output_dir=dropped_out,
        input_length=4, profile_length=4, max_jitter=1,
        transcript_end_policy="drop", progress=False,
    ))
    assert len(dropped.ids) < len(built.ids)
    assert all(m["sequence_left_pad"] == m["sequence_right_pad"] == 0 for m in dropped.metadata)


def test_shift_to_fit_boundaries_jitter_coordinates_and_short_loci(tmp_path):
    assert _shifted_materialized_interval(0, 4, 2, 12) == (0, 8)
    assert _shifted_materialized_interval(6, 4, 2, 12) == (2, 10)
    assert _shifted_materialized_interval(11, 4, 2, 12) == (4, 12)
    assert _shifted_materialized_interval(6, 10, 2, 12) is None
    root = tmp_path / "processed"
    sequence, profiles = _write_processed_fixture(root)
    windows = tmp_path / "windows"
    scan_windows(WindowScanConfig(
        processed_dir=root, output_prefix=windows, window_size=4, stride=4, progress=False,
    ))
    selection = tmp_path / "selection"
    select_regions(SelectionConfig(
        processed_dir=root, windows=windows, output_prefix=selection,
        strategy="broad_coverage", min_total_count=0, progress=False,
    ))
    out = tmp_path / "shifted"
    bundle = make_rbpnet_bundle(RBPNetBundleConfig(
        processed_dir=root,
        selection_manifest=selection,
        output_dir=out,
        input_length=4,
        profile_length=4,
        max_jitter=2,
        transcript_end_policy="shift_to_fit",
        progress=False,
    ))
    by_selection_start = {item["selection_start"]: (index, item) for index, item in enumerate(bundle.metadata)}
    assert by_selection_start[0][1]["sequence_materialized_start"] == 0
    assert by_selection_start[4][1]["sequence_materialized_start"] == 2
    assert by_selection_start[8][1]["sequence_materialized_start"] == 4
    assert all(
        item["sequence_left_pad"] == item["sequence_right_pad"] == 0
        for item in bundle.metadata
    )
    for index, item in enumerate(bundle.metadata):
        start, end = item["sequence_materialized_start"], item["sequence_materialized_end"]
        np.testing.assert_array_equal(bundle.X[index], encode_rna_sequence(sequence[start:end]))
        pstart, pend = item["profile_materialized_start"], item["profile_materialized_end"]
        np.testing.assert_array_equal(bundle.arrays["sminput_profiles"][index], profiles[0, pstart:pend])
        np.testing.assert_array_equal(bundle.arrays["ip_profiles"][index], profiles[1:, pstart:pend])
        assert item["sequence_anchor_offset"] == item["transcript_anchor"] - start

    left_index, left = by_selection_start[0]
    offsets = [
        jitter_crop_offset(
            anchor=left["transcript_anchor"],
            materialized_start=left["sequence_materialized_start"],
            locus_length=left["locus_length"],
            crop_length=4,
            jitter_shift=shift,
        )
        for shift in range(-2, 3)
    ]
    assert offsets == [0, 0, 0, 1, 2]
    for shift, offset in zip(range(-2, 3), offsets):
        desired = left["transcript_anchor"] - 2 + shift
        actual = min(max(desired, 0), len(sequence) - 4)
        np.testing.assert_array_equal(
            bundle.X[left_index, :, offset:offset + 4],
            encode_rna_sequence(sequence[actual:actual + 4]),
        )

    right = by_selection_start[8][1]
    assert jitter_crop_offset(
        anchor=right["transcript_anchor"],
        materialized_start=right["sequence_materialized_start"],
        locus_length=right["locus_length"],
        crop_length=4,
        jitter_shift=2,
    ) == 4
    assert bundle.config["transcript_end_policy"] == "shift_to_fit"
    assert bundle.config["n_dropped_short_loci"] == 0

    with pytest.raises(ValueError, match="no examples remain"):
        make_rbpnet_bundle(RBPNetBundleConfig(
            processed_dir=root,
            selection_manifest=selection,
            output_dir=tmp_path / "too_short",
            input_length=10,
            profile_length=10,
            max_jitter=2,
            transcript_end_policy="shift_to_fit",
            progress=False,
        ))
