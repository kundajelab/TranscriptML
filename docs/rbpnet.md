# RBPNet/eCLIP data workflow

TranscriptML includes the complete data path needed before implementing an
RBPNet model. It starts from ordinary eCLIP alignments and ends with fixed-shape,
memory-mappable NumPy arrays. It does **not** implement an RBPNet architecture,
loss, trainer, peak caller intended for general use, GC matching, or training
example sampling.

```text
FASTA + one-transcript-per-gene GTF + IP BAM(s) + SMInput BAM
                              |
                              v
                         preprocess
                              |
                              v
       canonical transcript-space HDF5 + FASTA + metadata
                              |
                              v
                     descriptive window scan
                              |
                              v
                       region selection
                              |
                              v
                 versioned selection manifest
                              |
                              v
                     RBPNet DatasetBundle
                              |
                              v
              fixed-shape, memory-mappable .npy arrays
```

Install the optional assay dependencies with:

```bash
python -m pip install -e ".[rbpnet]"
```

## 1. Canonical preprocessing

The GTF must select one transcript per gene. GTF coordinates are converted to
zero-based, half-open intervals. Exons are spliced in transcript 5′→3′ order;
minus-strand exon sequence is reverse-complemented and minus-strand transcript
position zero is therefore the highest-genomic-coordinate mature RNA base.
Protein-coding transcripts are partitioned into `5putr`, `cds`, and `3putr`.
Transcripts without a CDS are `noncoding_exon` throughout.

```bash
transcriptml rbpnet preprocess \
  --genome-fasta ../RBPNet2/Data/chr21_test/chr21.fa \
  --gtf ../RBPNet2/Data/chr21_test/gencode_v50_MANE_select_chr21.gtf \
  --ip-bam ip1=../RBPNet2/Data/chr21_test/ip1_chr21.bam \
  --ip-bam ip2=../RBPNet2/Data/chr21_test/ip2_chr21.bam \
  --sminput-bam sminput=../RBPNet2/Data/chr21_test/sminput_chr21.bam \
  --output-dir processed/chr21
```

Only read1 alignments are considered. The crosslink-position signal is the
aligned 5′ reference base: `reference_start` for forward alignments and
`reference_end - 1` for reverse alignments. In the eCLIP libraries used during
development, read1 aligns opposite the RNA strand, so `--read1-rna-strand
opposite` is the default. Use `same` or `unstranded` for libraries with another
convention. This read1 convention describes the supplied BAMs and should not be
confused with papers that name the crosslink-bearing FASTQ mate R2 before BAM
construction.

Unmapped, secondary, supplementary, QC-failed, duplicate (by default), and
low-MAPQ alignments are filtered. An event is assigned only when its 5′ base
maps to one strand-compatible selected transcript and the complete alignment is
compatible with that mature transcript. Aligned/deleted reference segments
must remain in exons and each CIGAR `N` must exactly match a selected adjacent
exon junction. Intronic/pre-mRNA alignments are reported separately as
`transcript_incompatible` in `qc.json`.

Missing `.fai`/`.bai` indexes are created when the source files and their
directories are writable. GTF transcripts on contigs absent from the analysis
FASTA are reported and skipped; retained FASTA contigs must be represented in
every BAM.

The canonical directory contains:

| File | Contract |
| --- | --- |
| `manifest.json` | Format version, exact sample order/roles, input file stat records, configuration, and effective library sizes |
| `qc.json` | Annotation, sequence, read-filter, assignment, and per-sample totals |
| `transcripts.tsv` | Gene/transcript metadata, length, strand, raw counts, SMInput TPM, and compact region annotations |
| `exons.tsv.gz` | Transcript interval ↔ genomic exon block mappings |
| `regions.tsv.gz` | Long-form transcript region intervals |
| `transcripts.fa` + `.fai` | Indexed mature-transcript sequences |
| `signals.h5` | Canonical base-resolution retained read1 5′ counts |

SMInput TPM is calculated from retained transcript counts divided by mature
transcript length, followed by normalization of those rates to one million.
Each sample's `effective_library_size` is exactly the number of retained read1
5′ events used to construct its HDF5 track. That field is the CPM denominator.
Pooled-IP CPM uses the sum of IP counts divided by the sum of IP effective
library sizes.

### HDF5 signal layout

HDF5 is a hierarchical binary container: datasets behave like typed,
multidimensional arrays stored inside a file, can be compressed and chunked,
and can be sliced without loading the entire array. `signals.h5` uses a compact
concatenated-transcript layout:

| Dataset | Shape/dtype | Meaning |
| --- | --- | --- |
| `counts` | `(S, total_transcript_bases)`, `uint32` | One row per manifest sample |
| `ip_pooled` | `(total_transcript_bases,)`, `uint32` | Chunkwise sum of all IP rows |
| `sample_names` / `sample_roles` | `(S,)`, UTF-8 | HDF5 row identity |
| `transcript_ids` | `(T,)`, UTF-8 | Transcript order |
| `transcript_offsets` / `transcript_lengths` | `(T,)`, `int64` | Slice boundaries in concatenated space |

Users normally do not need to calculate flat offsets. The lazy reader handles
FASTA, HDF5, sample order, and exon blocks:

```python
from transcriptml.rbpnet import ProcessedECLIPDataset

with ProcessedECLIPDataset("processed/chr21") as ds:
    print(ds.transcripts)
    print(ds.samples)
    seq = ds.get_sequence("ENST...", 100, 400)
    input_profile = ds.get_profile("ENST...", 100, 400, sample="sminput")
    ip1_profile = ds.get_profile("ENST...", 100, 400, sample="ip1")
    pooled = ds.get_pooled_ip_profile("ENST...", 100, 400)
    blocks = ds.get_genomic_blocks("ENST...", 100, 400)
```

## 2. Descriptive window scanning

The scanner reads the canonical experiment, not BAMs. A new window size or
stride therefore requires only a new scan:

```bash
transcriptml rbpnet scan-windows \
  --processed-dir processed/chr21 \
  --window-size 100 \
  --stride 50 \
  --min-sminput-tpm 0 \
  --pseudocount 1 \
  --output-prefix processed/chr21_windows_100nt
```

It writes equivalent `*.tsv.gz` and `*.parquet` tables plus `*.scan.json`.
Incomplete terminal windows are omitted unless
`--include-incomplete-terminal-windows` is given. Per-transcript cumulative
sums make count aggregation efficient even for stride-1 scans.

Rows include transcript/genomic coordinates and exon blocks; exact overlap
counts/fractions for every region class; GC; SMInput TPM; dynamic per-sample
counts, CPM, and maximum positional counts; pooled-IP values; combined
coverage; and a CPM-scale pooled-IP/SMInput log ratio. Boundary-crossing
windows are `mixed`, never silently assigned to one region.

The scanner is deliberately descriptive. It does not label peaks, negatives,
or training examples.

## 3. Region selection

Selection asks which experimental loci are eligible and why. It writes a
versioned `*.parquet` manifest, equivalent `*.tsv.gz`, and a
`*.selection.json` provenance sidecar.

### Published RBPNet v1

First make the published 100-nt, stride-1 descriptive scan, then select:

```bash
transcriptml rbpnet scan-windows \
  --processed-dir processed/chr21 --window-size 100 --stride 1 \
  --output-prefix processed/chr21_windows_v1

transcriptml rbpnet select-regions \
  --processed-dir processed/chr21 \
  --windows processed/chr21_windows_v1.parquet \
  --strategy original_rbpnet \
  --output-prefix processed/chr21_original
```

The default preset follows [Horlacher et al. 2023](https://doi.org/10.1186/s13059-023-03015-7)
and its [reference implementation](https://github.com/mhorlacher/rbpnet): a one-sided Poisson test
against the transcript-level pooled-IP rate (`p < 0.01`), at least 8 pooled
counts, a maximum positional count of at least 2, and a 50-nt advance after an
accepted candidate. The selected interval remains 100 nt; a later 300-nt
bundle context is independent. All thresholds are CLI-configurable.

### Broad measured windows (`yeo_2026`)

This strategy applies coverage thresholds without a peak test:

```bash
transcriptml rbpnet select-regions \
  --processed-dir processed/chr21 \
  --windows processed/chr21_windows_100nt.parquet \
  --strategy yeo_2026 \
  --min-total-count 8 --min-sminput-count 1 --min-ip-count 1 \
  --output-prefix processed/chr21_measured
```

`--replicate-mode combined` retains one row while preserving every replicate
column. `per_ip` emits a replicate-identified row for each IP satisfying the
thresholds. This is a configurable implementation of the broad inclusion
philosophy; its defaults are not an assertion that one coverage cutoff is
universally optimal.

### Peak / gray / confident negative

```bash
transcriptml rbpnet select-regions \
  --processed-dir processed/chr21 \
  --windows processed/chr21_windows_100nt.parquet \
  --strategy peak_gray_negative \
  --min-total-count 8 --min-sminput-count 1 --min-ip-count 1 \
  --peak-fdr 0.05 --peak-min-log2-ratio 1 \
  --negative-fdr 0.05 --negative-max-log2-ratio -0.5 \
  --stitch-gap 0 \
  --output-prefix processed/chr21_peak_gray_negative
```

Adequately measured windows are tested by conditioning on pooled-IP + SMInput
counts. The null IP probability is determined by their effective library
sizes. One-sided exact binomial enrichment/depletion p-values are
Benjamini–Hochberg corrected. Peaks require enrichment plus a minimum log2
effect; confident negatives require depletion plus a maximum log2 effect; the
remaining adequate windows are gray. Low-information windows are omitted.
Overlapping/nearby windows are stitched only when transcript, state, and region
type agree. Peak anchors are pooled-signal maxima; negative and gray anchors
are interval midpoints. These defaults are transparent starting choices, not a
definitive CLIP peak caller.

### Selection manifest v1

Every row has a deterministic content-derived `example_id`; gene, transcript,
chromosome, strand, anchor, and half-open selection interval; region overlap;
strategy/state and optional replicate identity; sample/pooled signal summaries;
statistical fields; and explicit gene/transcript/chromosome grouping columns.
Parquet metadata and the sidecar preserve the complete selection and scan
configuration. Consumers must key by `example_id`, not row order.

## 4. Materialized RBPNet bundle

Bundle construction separates the selected biological interval from future
model context:

```bash
transcriptml rbpnet make-bundle \
  --processed-dir processed/chr21 \
  --selection-manifest processed/chr21_measured.parquet \
  --output-dir data/rbpnet_chr21 \
  --input-length 300 \
  --profile-length 300 \
  --max-jitter 32 \
  --transcript-end-policy pad
```

For requested input length `L`, profile length `P`, and maximum future jitter
`J`, the bundle writes:

| File | Shape/dtype |
| --- | --- |
| `X.npy` | `(N, 4, L + 2J)`, `uint8`, TranscriptML RNA4 A/C/G/U encoding |
| `sminput_profiles.npy` | `(N, P + 2J)`, `uint32` |
| `ip_profiles.npy` | `(N, R, P + 2J)`, `uint32`; `R` follows `ip_axis_order` |
| `sequence_valid_mask.npy` | `(N, L + 2J)`, `uint8` |
| `profile_valid_mask.npy` | `(N, P + 2J)`, `uint8` |
| `profile_sminput_totals.npy` | `(N,)`, `uint64` |
| `profile_ip_totals.npy` | `(N, R)`, `uint64` |
| `selection_sminput_counts.npy` | `(N,)`, `uint64` |
| `selection_ip_counts.npy` | `(N, R)`, `uint64` |

Pooled IP is `ip_profiles.sum(axis=1)` and is not the only stored
representation. The ordinary TranscriptML sidecars (`ids.txt`,
`metadata.json`, `schema.json`, `config.json`) accompany the arrays, and the
full selected-example metadata is copied to `examples.parquet`. All arrays can
be loaded with NumPy `mmap_mode="r"`.

With `--transcript-end-policy drop`, examples whose full materialized sequence
or profile context crosses a transcript end are removed and counted. With
`pad`, fixed widths are preserved using all-zero sequence/profile padding and
the validity masks distinguish padding from valid ambiguous sequence or true
zero signal.

A future jitter shift `s` in `[-J, +J]` takes each requested crop starting at
`J + s` in its materialized array. Bundle construction does not perform random
augmentation. Original v1 can use `J=0`; a jitter-ready workflow can use
`J=32`. Candidate-scan advance and training-time jitter are unrelated.

```python
from transcriptml.rbpnet.bundle import load_rbpnet_bundle

bundle = load_rbpnet_bundle("data/rbpnet_chr21", mmap_mode="r")
X = bundle.X
ip = bundle.arrays["ip_profiles"]
input_profile = bundle.arrays["sminput_profiles"]
print(bundle.config["sample_metadata"]["ip_axis_order"])
```

The canonical experiment uses HDF5 because it provides compressed lazy slicing
over an entire transcriptome. The model bundle uses separate `.npy` files
because its selected fixed-shape arrays are simple to inspect and memory-map.
Changing selection, context, or jitter does not require reprocessing BAMs.
