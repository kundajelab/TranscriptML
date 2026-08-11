import csv
import gzip
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
import pysam
import pytest

from transcriptml.data.encoding import encode_rna_sequence
from transcriptml.rbpnet.bundle import RBPNetBundleConfig, load_rbpnet_bundle, make_rbpnet_bundle
from transcriptml.rbpnet.coordinates import Exon, Region, Transcript, annotate_regions
from transcriptml.rbpnet.experiment import ProcessedECLIPDataset
from transcriptml.rbpnet.fasta import transcript_sequence
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
    yeo = tmp_path / "yeo"
    summary = select_regions(SelectionConfig(
        processed_dir=root, windows=windows, output_prefix=yeo, strategy="yeo_2026",
        min_total_count=1, min_sminput_count=0, min_ip_count=0, progress=False,
    ))
    loaded = load_selection_manifest(yeo)
    assert summary["n_examples"] == len(loaded.rows) > 0
    assert len({row["example_id"] for row in loaded.rows}) == len(loaded.rows)
    yeo2 = tmp_path / "yeo2"
    select_regions(SelectionConfig(
        processed_dir=root, windows=windows, output_prefix=yeo2, strategy="yeo_2026",
        min_total_count=1, min_sminput_count=0, min_ip_count=0, progress=False,
    ))
    assert [r["example_id"] for r in loaded.rows] == [
        r["example_id"] for r in load_selection_manifest(yeo2).rows
    ]
    per_ip = tmp_path / "yeo_per_ip"
    select_regions(SelectionConfig(
        processed_dir=root, windows=windows, output_prefix=per_ip, strategy="yeo_2026",
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


def test_materialized_bundle_exact_arrays_jitter_padding_and_mmap(tmp_path):
    root = tmp_path / "processed"
    sequence, profiles = _write_processed_fixture(root)
    windows = tmp_path / "windows"
    scan_windows(WindowScanConfig(
        processed_dir=root, output_prefix=windows, window_size=4, stride=4, progress=False,
    ))
    selection = tmp_path / "selection"
    select_regions(SelectionConfig(
        processed_dir=root, windows=windows, output_prefix=selection, strategy="yeo_2026",
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
