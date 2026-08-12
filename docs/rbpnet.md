# RBPNet/eCLIP data workflow

TranscriptML includes an eCLIP path from ordinary alignments through fixed-shape,
memory-mappable arrays and structured RBPNet training. The scanner is
descriptive and the selectors prepare model examples; none is intended as a
general-purpose peak caller. GC matching and post-selection sampling are not
implemented.

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
                              |
                              v
           structured RBPNet profile/enrichment training
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

Signal rows are sparse-aggregated and then written one complete touched HDF5
chunk at a time. Zero-only chunks retain the HDF5 fill value and are not
allocated. The default is shuffled gzip level 1, chosen to reduce write and
slice-decompression time while retaining compact sparse storage. Use
`--signal-compression gzip --signal-compression-level 4` for smaller but slower
files, `--signal-compression lzf` for faster/larger files, or
`--signal-compression none` when storage is unimportant. Compression and chunk
length are recorded as `signals.h5` attributes and in preprocessing provenance.

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

Every selector can restrict its candidate universe to exact scanner
annotations without regenerating the descriptive window table:

```bash
transcriptml rbpnet select-regions \
  --processed-dir processed/chr21 \
  --windows processed/chr21_windows_100nt.parquet \
  --strategy peak_gray_negative \
  --region-types 5putr,cds,3putr \
  --output-prefix processed/chr21_exonic_selection
```

Valid values are `5putr`, `cds`, `3putr`, `noncoding_exon`, `intron`, and
`mixed`. The default is all types. Filtering is exact: for example,
`--region-types 3putr` accepts only windows wholly contained in 3' UTR and
does not accept a boundary-crossing `mixed` window. Include `mixed`
explicitly when desired.

The restriction is applied before each strategy's signal/statistical rules.
Consequently, excluded windows do not affect coverage eligibility, the
original selector's testing/50-nt advance, or the `peak_gray_negative` BH
correction universe. The published IP locus-density Poisson null remains based
on the complete locus; this option restricts which windows are tested, not how
that published null is defined. The selection provenance records the requested
types and source/eligible window counts, while the sidecar also reports
selected-example counts by region type.

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

## 5. RBPNet model and training

Create a native starter config and train it with the same TranscriptML command
used by other registered models:

```bash
transcriptml init-run --workflow rbpnet --out-dir configs/rbpnet
# Edit dataset/output paths and profile_length, then:
transcriptml train configs/rbpnet/train_config.json

transcriptml evaluate \
  --checkpoint runs/rbpnet/model/best.pt \
  --dataset data/rbpnet_chr21 \
  --out-csv runs/rbpnet/predictions.csv
```

`transcriptml models show rbpnet --json` prints every architectural default.
The first model family intentionally requires equal sequence and profile crop
lengths. It uses no observed-control input, pooling, reverse-complement
augmentation, valid convolution, absolute-count head, or larger sequence
context than prediction context.

### Default architecture

The RNA4 sequence alone enters a same-padded 1D convolution with 128 filters
and kernel 12 followed by ReLU, then five residual blocks. Each block is a dilated
kernel-6 convolution, BatchNorm, ReLU, dropout 0.25, and residual addition. The
dilations are `[2, 4, 8, 16, 32]`; no positional pooling occurs. Two independent
kernel-25, stride-one transposed-convolution heads produce target and control
positional logits. A global-average-pooled linear head produces the scalar
mixture logit. Important dimensions, normalization, biases, kernels, dilation
schedule, dropout, head kinds, and profile length are configurable.

The trunk receptive field is reported in checkpoints and `summary.json` and is

```text
RF = 1 + (initial_kernel - 1)
       + sum((residual_kernel - 1) * dilation)
```

which is 322 bases for the defaults (160 indexed positions to the left and 161
to the right under the documented asymmetric even-kernel padding). Explicit
left/right padding preserves output index `i` as index `i`; the extra base of
an even effective kernel is placed on the right.

| Model parameter | Default | Meaning |
| --- | --- | --- |
| `in_ch` | `4` | RNA4 input channels. |
| `n_filters` | `128` | Shared positional hidden width. |
| `initial_kernel_size` | `12` | Initial same-padded convolution kernel. |
| `n_residual_blocks` | `5` | Number of residual convolutions. |
| `residual_kernel_size` | `6` | Residual convolution kernel. |
| `dilations` | `null` | Explicit schedule; `null` resolves powers of two starting at 2. |
| `normalization` | `batch` | `batch`, `layer`, or `none`. |
| `dropout` | `0.25` | Dropout in each residual branch. |
| `profile_head_type` | `transpose_conv` | `transpose_conv` or ordinary same-padded `conv`. |
| `profile_head_kernel_size` | `25` | Kernel shared by the two separately parameterized profile heads. |
| `profile_head_bias` | `true` | Whether profile heads include a bias. |
| `enrichment_head_type` | `none` | `none`, `linear`, or `mlp`. |
| `enrichment_hidden` | `64` | Hidden width for the optional MLP only. |
| `enrichment_dropout` | `0` | Optional MLP dropout. |
| `profile_length` | `300` | Required input/output crop length, or `null` to accept any length. |

### Profile model: target, control, and pi

The two heads define independently normalized distributions
`p_target=softmax(target_logits)` and
`p_control=softmax(control_logits)`. With global mixing logit `a`,
`pi=sigmoid(a)` and the predicted IP distribution is

```text
p_IP = pi * p_target + (1 - pi) * p_control
```

The mixture is evaluated with `logsigmoid` and `logaddexp` for stability. `pi`
is the latent fraction of the **positional IP profile** assigned to the target
component. It is not IP/SMInput enrichment and is never given an IP-vs-SMInput
binomial loss. SMInput is a profile training target, not a neural-network
input.

By default, individual IP profiles are summed once across their replicate axis
and the complete multinomial NLL is calculated for the pooled IP counts under
`p_IP`. A second complete multinomial NLL compares SMInput counts with
`p_control`. The `lgamma` combinatorial constant is included by default and can
be disabled. A zero-total profile has no positional information, so that locus
is excluded from that profile component's mean instead of producing a NaN.
Components are reduced over informative loci and weighted by
`lambda_ip_profile` and `lambda_sm_profile` (both 1 by default).

### Optional enrichment model

Set `"enrichment_head_type": "linear"` to enable the default enrichment head,
or `"mlp"` for a configurable two-layer head. The head average-pools the shared
hidden representation only over the biological selection interval, using the
coordinate-derived mask for the current jittered crop. It supports variable
measurement widths. The result `eta_i` is a sequence-predicted log enrichment,
independent of `pi`.

For replicate `j`, effective retained-event library sizes supply the known
offset and the observed selection-interval counts supply the binomial data:

```text
depth_offset_j = log(L_IP_j / L_SM)
logit(p_ij)    = eta_i + depth_offset_j
N_ij           = IP_ij + SM_i
IP_ij          ~ Binomial(N_ij, p_ij)
```

The logits-based complete binomial NLL is evaluated independently for each
valid locus-replicate pair and averaged over those pairs. One sequence row and
one `eta_i` therefore use every IP replicate without duplicating the locus.
`IP=0` and `SMInput=0` edge cases are exact and require no pseudocount; only a
pair with both counts zero is excluded because it has no information. Enabling
the head adds `lambda_enrichment * L_enrichment`, with weight 1 by default.

### Jitter-ready structured batches

`RBPNetDataset` memory-maps the bundle arrays and returns sequence, pooled and
individual IP profiles, SMInput profile, exact selection counts, effective
library sizes/depth offsets, selection mask, valid-position masks, coordinates,
and identifiers. With `max_train_jitter=J`, a deterministic RNG keyed by
seed/epoch/example samples a shift from `[-J,+J]` for training. Sequence and all
profiles use the same biological crop. The crop start is derived from anchor,
actual materialized start, and locus bounds, so boundary-shifted contexts do
not incorrectly assume offset `J+s`. Evaluation always uses shift zero.

When enrichment is enabled, every allowed jittered crop must fully contain its
selection interval; invalid bundle/context combinations fail before training.
`max_train_jitter` cannot exceed the materialized bundle margin.

### Splits, optimization, and outputs

RBPNet starter configs use a `group` split on `group_gene_id`, keeping all
overlapping loci from one gene together. Transcript, chromosome, metadata, and
explicit predefined groups are also usable through their metadata columns.
Every non-random split is checked for group overlap. Row-random splitting is
rejected unless `allow_random_window_split=true` explicitly acknowledges the
leakage risk. Replicate-specific selection rows describing an identical locus
are deduplicated by default while retaining the complete replicate axis.

For chromosome cross-validation, create one plan from the final bundle and
reuse it for every training job:

```bash
transcriptml cv create-chromosome-plan \
  --dataset data/rbpnet \
  --group-col group_chromosome \
  --n-folds 5 \
  --output runs/rbpnet/cv5.json

transcriptml train configs/rbpnet/train_config.json \
  --dataset data/rbpnet \
  --cv-plan runs/rbpnet/cv5.json \
  --fold 0 \
  --output-dir runs/rbpnet/fold0/model
```

Chromosomes are sorted by decreasing example count and greedily assigned to
the currently smallest fold group, with deterministic ties. Run `k` uses group
`k` for test, `(k+1) mod N` for validation, and every remaining group for
training. Resolution verifies that the dataset's chromosome membership and
counts still exactly match the immutable, content-hashed plan. Training records
both the plan path and its validated `plan_id` in summaries and checkpoints.

AdamW, Adam, and SGD; plateau, cosine, and step schedulers; clipping; early
stopping; device selection; DataLoader workers; seeds; and mixed precision are
configurable. `history.json` logs total, pooled-IP profile, SMInput profile, and
enrichment losses independently. `best.pt` and `last.pt` retain model, loss,
optimizer, samples, coordinate space, split, receptive-field, and training
provenance. Evaluation CSVs contain `pi`, optional `eta`, and each replicate's
depth-adjusted predicted IP fraction. The raw structured tensors remain
available through the Python model output for future attribution work.

Profile-only model block:

```json
{
  "model": {
    "name": "rbpnet",
    "params": {"profile_length": 300, "enrichment_head_type": "none"}
  },
  "loss": {"name": "rbpnet"},
  "max_train_jitter": 0
}
```

To train profiles plus enrichment, change only the head and, if desired, the
independent component weights:

```json
{
  "model": {
    "name": "rbpnet",
    "params": {"profile_length": 300, "enrichment_head_type": "linear"}
  },
  "loss": {
    "name": "rbpnet",
    "lambda_ip_profile": 1.0,
    "lambda_sm_profile": 1.0,
    "lambda_enrichment": 1.0
  },
  "max_train_jitter": 32
}
```
