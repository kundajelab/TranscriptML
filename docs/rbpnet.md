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
       canonical locus-coordinate HDF5 + FASTA + metadata
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
zero-based, half-open intervals. `--coordinate-space mature_transcript` is the
default and preserves the original behavior: exons are spliced in transcript
5′→3′ order. `--coordinate-space gene` instead retains the contiguous genomic
span from the first through last selected exon, including introns. Both spaces
are represented in annotated RNA 5′→3′ orientation. Thus, position zero on a
minus-strand locus is its highest-genomic-coordinate base and gene-space
sequence is reverse-complemented.

Protein-coding exon sequence is partitioned into `5putr`, `cds`, and `3putr`;
exons without a CDS are `noncoding_exon`. Gene space additionally labels the
gaps between exons as `intron`. The labels exhaustively partition each stored
locus, so a boundary-crossing window is reported as `mixed`.

```bash
transcriptml rbpnet preprocess \
  --genome-fasta ../RBPNet2/Data/chr21_test/chr21.fa \
  --gtf ../RBPNet2/Data/chr21_test/gencode_v50_MANE_select_chr21.gtf \
  --ip-bam ip1=../RBPNet2/Data/chr21_test/ip1_chr21.bam \
  --ip-bam ip2=../RBPNet2/Data/chr21_test/ip2_chr21.bam \
  --sminput-bam sminput=../RBPNet2/Data/chr21_test/sminput_chr21.bam \
  --coordinate-space mature_transcript \
  --output-dir processed/chr21
```

Use `--coordinate-space gene --output-dir processed/chr21_gene` to build the
corresponding full-gene experiment. Coordinate space is recorded in
`manifest.json`, `signals.h5`, scan metadata, selection provenance, and bundle
metadata. Downstream stages reject mismatched inputs.

For backward compatibility, table columns such as `transcript_id`, `tx_start`,
`tx_end`, and `transcript_anchor` retain their established names. In a
gene-space experiment, their numeric values are gene-space coordinates in the
selected transcript's 5′→3′ orientation; `coordinate_space` removes the
ambiguity.

Only read1 alignments are considered. The crosslink-position signal is the
aligned 5′ reference base: `reference_start` for forward alignments and
`reference_end - 1` for reverse alignments. In the eCLIP libraries used during
development, read1 aligns opposite the RNA strand, so `--read1-rna-strand
opposite` is the default. Use `same` or `unstranded` for libraries with another
convention. This read1 convention describes the supplied BAMs and should not be
confused with papers that name the crosslink-bearing FASTQ mate R2 before BAM
construction.

Unmapped, secondary, supplementary, QC-failed, duplicate (by default), and
low-MAPQ alignments are filtered. Assignment then depends on coordinate space:

- In mature-transcript space, the 5′ base must map uniquely to a
  strand-compatible selected exon. Aligned/deleted reference segments must
  remain in exons and each CIGAR `N` must exactly match an adjacent selected
  exon junction. Intronic/pre-mRNA alignments are not retained.
- In gene space, the 5′ base must map uniquely to a strand-compatible selected
  gene span. Every reference-consuming CIGAR operation must remain inside that
  span. Intronic alignments are retained, and `N` operations may represent any
  splice within the selected gene span; they need not match the one selected
  mature isoform. Reads overlapping multiple same-strand gene spans remain
  ambiguous rather than being assigned arbitrarily.

The same read filtering and unique-gene rule apply in both modes. Rejections
are itemized in `qc.json`.

Missing `.fai`/`.bai` indexes are created when the source files and their
directories are writable. GTF transcripts on contigs absent from the analysis
FASTA are reported and skipped; retained FASTA contigs must be represented in
every BAM.

The canonical directory contains:

| File | Contract |
| --- | --- |
| `manifest.json` | Format version, exact sample order/roles, input file stat records, configuration, and effective library sizes |
| `qc.json` | Annotation, sequence, read-filter, assignment, and per-sample totals |
| `transcripts.tsv` | Gene/transcript metadata, coordinate space, genomic span, length, strand, raw counts, SMInput TPM, and compact region annotations |
| `exons.tsv.gz` | Transcript interval ↔ genomic exon block mappings |
| `regions.tsv.gz` | Long-form transcript region intervals |
| `transcripts.fa` + `.fai` | Indexed 5′→3′ selected-locus sequences |
| `signals.h5` | Canonical base-resolution retained read1 5′ counts |

SMInput TPM is calculated from retained locus counts divided by stored locus
length, followed by normalization of those rates to one million. Therefore,
gene-space TPM uses gene-space retained events and full gene-span length; it is
not numerically interchangeable with mature-transcript TPM.
Each sample's `effective_library_size` is exactly the number of retained read1
5′ events used to construct its HDF5 track. That field is the CPM denominator.
Pooled-IP CPM uses the sum of IP counts divided by the sum of IP effective
library sizes.

### HDF5 signal layout

HDF5 is a hierarchical binary container: datasets behave like typed,
multidimensional arrays stored inside a file, can be compressed and chunked,
and can be sliced without loading the entire array. `signals.h5` uses a compact
concatenated-locus layout:

| Dataset | Shape/dtype | Meaning |
| --- | --- | --- |
| `counts` | `(S, total_locus_bases)`, `uint32` | One row per manifest sample |
| `ip_pooled` | `(total_locus_bases,)`, `uint32` | Chunkwise sum of all IP rows |
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
    print(ds.coordinate_space)
    seq = ds.get_sequence("ENST...", 100, 400)
    input_profile = ds.get_profile("ENST...", 100, 400, sample="sminput")
    ip1_profile = ds.get_profile("ENST...", 100, 400, sample="ip1")
    pooled = ds.get_pooled_ip_profile("ENST...", 100, 400)
    blocks = ds.get_genomic_blocks("ENST...", 100, 400)
    genomic = ds.coordinate_to_genome("ENST...", 100)
    coordinate = ds.genome_to_coordinate("ENST...", "chr21", genomic[1])
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
  --poisson-null ip_locus_density \
  --output-prefix processed/chr21_original
```

The default preset follows [Horlacher et al. 2023](https://doi.org/10.1186/s13059-023-03015-7)
and its [reference implementation](https://github.com/mhorlacher/rbpnet): a
one-sided Poisson test against the whole-locus pooled-IP rate. For a window of
length `W` in a locus of length `T`, the published/default null is:

```text
mu = pooled_IP_locus_total / T * W
```

A candidate requires uncorrected `p < 0.01`, at least 8 pooled counts, and a
maximum positional count of at least 2. After accepting one, scanning advances
50 nt. No multiple-testing correction is applied: this is the intentionally
lenient published candidate generator, not a calibrated peak caller. The
selected interval remains 100 nt; later model context is independent.

An explicitly experimental alternative uses the matched input:

```bash
transcriptml rbpnet select-regions \
  --processed-dir processed/chr21 \
  --windows processed/chr21_windows_v1.parquet \
  --strategy original_rbpnet \
  --poisson-null sminput \
  --sminput-poisson-pseudocount 1 \
  --output-prefix processed/chr21_original_sminput_null
```

Its exact expectation is:

```text
mu = (SMInput_window_count + 1)
     * (pooled_IP_effective_library_size / SMInput_effective_library_size)
```

The add-one term is applied in SMInput count space and then exposure-scaled to
the pooled-IP library. This asks whether the observed pooled-IP count exceeds
the background expected at the IP sequencing depth. It retains the same
uncorrected `p < 0.01`, count/height requirements, and 50-nt advance. The
manifest records `poisson_null`, the pseudocount, formula, and each selected
window's `selection_null_mean`. This mode is not attributed to Horlacher et al.

### Broad measured windows (`broad_coverage`)

This strategy applies coverage thresholds without a peak test:

```bash
transcriptml rbpnet select-regions \
  --processed-dir processed/chr21 \
  --windows processed/chr21_windows_100nt.parquet \
  --strategy broad_coverage \
  --min-total-count 6 --min-sminput-count 0 --min-ip-count 0 \
  --replicate-mode per_ip \
  --output-prefix processed/chr21_measured
```

The defaults implement `IP + SMInput > 5`, including cases where either track
is zero. The default `--replicate-mode per_ip` applies that criterion independently as
`IP_replicate_j + SMInput >= 6` and emits a replicate-identified row.
`combined` instead applies it to pooled IP and emits one row while preserving
every replicate column. This selector is inspired by the broad-coverage
training philosophy of Yeo et al.; it works on arbitrary TranscriptML scan
tables and is **not** an exact reproduction of their non-overlapping,
annotation-aware Skipper window generation. A locked exact-window preset can
be added separately later.

### Peak / gray / confident negative

```bash
transcriptml rbpnet select-regions \
  --processed-dir processed/chr21 \
  --windows processed/chr21_windows_100nt.parquet \
  --strategy peak_gray_negative \
  --min-total-count 8 --min-sminput-count 0 --min-ip-count 0 \
  --peak-fdr 0.05 --peak-min-log2-ratio 1 \
  --negative-fdr 0.05 --negative-max-log2-ratio -0.5 \
  --stitch-gap 0 \
  --output-prefix processed/chr21_peak_gray_negative
```

By default, combined pooled-IP + SMInput count of at least 8 is sufficient to
enter classification; neither individual track must be nonzero. Optional
`--min-ip-count` and `--min-sminput-count` knobs remain available. Thus both
`IP=0, SMInput=N` and `IP=N, SMInput=0` are valid, informative cases.
Adequately measured windows are tested by conditioning on pooled-IP + SMInput
counts. The null IP probability is determined by effective library sizes.
One-sided exact binomial enrichment/depletion p-values naturally handle both
extremes and are Benjamini–Hochberg corrected. The log2 effect uses the
scanner's explicit CPM-scale pseudocount, so it remains finite at zero. Peaks
require enrichment plus a minimum log2 effect; confident negatives require
depletion plus a maximum log2 effect; the remaining adequate windows are gray.
Low-total-information windows are omitted.
Overlapping/nearby windows are stitched only when transcript, state, and region
type agree. Peak anchors are pooled-signal maxima; negative and gray anchors
are interval midpoints. These defaults are transparent starting choices, not a
definitive CLIP peak caller.

### Selection manifest v1

Every row has a deterministic content-derived `example_id`; gene, transcript,
chromosome, strand, coordinate space, anchor, and half-open selection interval; region overlap;
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
  --transcript-end-policy shift_to_fit
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

Boundary handling is configurable:

- `shift_to_fit` is the default. It first centers the full stored width around
  the anchor, then shifts the interval right or left until it lies entirely in
  the locus. The anchor need not remain centered, but every stored position is
  real sequence/signal. A locus shorter than either requested stored width is
  genuinely insufficient; its examples are dropped and counted as
  `n_dropped_short_loci`, never silently padded.
- `drop` preserves the old behavior of discarding any example whose centered
  stored sequence or profile crosses a boundary.
- `pad` preserves fixed widths with all-zero sequence/profile padding. Validity
  masks distinguish padding from ambiguous sequence or true zero signal.

Each example's metadata records `locus_length`, biological anchor,
`sequence_materialized_start/end`, `profile_materialized_start/end`, anchor
offsets, padding amounts, and the crop offsets at both jitter extremes.

For `shift_to_fit`, a future jitter shift `s` in `[-J,+J]` must be resolved in
biological coordinates—not universally as `J+s`:

```text
desired_crop_start = anchor - crop_length//2 + s
actual_crop_start  = clip(desired_crop_start, 0, locus_length-crop_length)
crop_offset        = actual_crop_start - materialized_start
```

Near a boundary, multiple requested shifts can collapse to the same closest
legal crop. The shifted stored interval of width `L+2J` contains every such
legal `L`-nt crop when the locus is long enough. Sequence and profile starts
are recorded separately because their requested lengths may differ. The helper
`transcriptml.rbpnet.bundle.jitter_crop_offset` implements this formula.
Legacy `pad` bundles retain their centered `J+s` crop convention. Bundle
construction does not itself perform random augmentation. Candidate-scan
advance and training-time jitter remain unrelated.

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
