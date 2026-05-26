# TranscriptML

Training and interpreting RNA sequence-to-function models.

TranscriptML currently implements reusable APIs for RNA one-hot encoding (Saluki-style 6-channel and basic 4-channel for MPRAs), dataset
bundles, model training/evaluation (Saluki and an MPRA model (LegNet) currently supported; RiboNN and potentially RBPNet in the future), and interpretation.

## Install

For full-functionality model training and tests:

```bash
git clone https://github.com/kundajelab/TranscriptML.git
cd TranscriptML
python -m pip install -e ".[dev,genomics]"
```

If you don't need the GTF/FASTA transcript extraction functionality, install with:

```bash
python -m pip install -e ".[dev]"
```

### Optional Dependencies

TranscriptML keeps a small core install: NumPy, PyTorch, and TOML support on
older Python versions. Optional extras enable workflows that need additional
packages:

- `dev`: installs `pytest` for running the test suite.
- `genomics`: installs `pyfaidx` for GTF/FASTA transcript extraction used by
  `transcriptml build-saluki-gtf`.
- `arrow`: installs `pyarrow` for streaming codon-ISM mutation tables to
  Parquet or Arrow IPC files.

Common combinations:

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[dev,genomics]"
python -m pip install -e ".[dev,genomics,arrow]"
```

## Quickstart

This example starts from a genome FASTA, transcript annotations, and a target
table with transcript-level RNA stability measurements. Replace the example
paths and column names with the ones from your dataset.

Required inputs:

- `genome.fa`: genomic FASTA
- `annotations.gtf`: transcript annotation GTF with `exon` rows and
  `transcript_id` attributes
- `targets.csv`: table containing `transcript_id`, `log_kdeg`, and optionally
  `split` values such as `train`, `val`, and `test`

Build a Saluki-style dataset bundle:

```bash
transcriptml build-saluki-gtf \
  --gtf annotations.gtf \
  --fasta genome.fa \
  --out-dir data/saluki \
  --targets targets.csv \
  --target-id-col transcript_id \
  --target-col log_kdeg \
  --split-col split \
  --length 12288
```

The commands here assume `targets.csv` has a `split` column. If it does not,
omit `--split-col split` above and `--split test` during evaluation; training
will create a random train/validation/test split and write test predictions
under the run directory. The build command writes `data/saluki/X.npy`,
`data/saluki/y.npy`, transcript IDs, metadata, and bundle configuration files.

Create a small training config:

```bash
cat > train_saluki.json <<'JSON'
{
  "dataset": "data/saluki",
  "output_dir": "runs/saluki_exact",
  "model": {"name": "saluki_exact"},
  "epochs": 50,
  "device": "auto"
}
JSON
```

Train, evaluate, and run single-nucleotide ISM:

```bash
transcriptml train train_saluki.json

transcriptml evaluate runs/saluki_exact/best.pt data/saluki predictions.csv \
  --split test \
  --device auto

transcriptml ism runs/saluki_exact/best.pt data/saluki interpret/ism \
  --device auto \
  --mutation-batch-size 512
```

Training writes checkpoints and summaries under `runs/saluki_exact/`. Evaluation
writes `predictions.csv` and `predictions.summary.json`. ISM writes NumPy arrays
under `interpret/ism/`.

## Details

### Input Data

TranscriptML writes processed datasets as bundles containing `X.npy`, optional
`y.npy`, `ids.txt`, `schema.json`, `config.json`, optional `metadata.json`, and
optional `splits.json`.

For Saluki-style transcript models, provide:

- A genome FASTA. Contig names should match the GTF, although common `chr`/no
  `chr` aliases are tried.
- A standard 9-column GTF with `exon` rows and `transcript_id` attributes. `CDS`
  rows are optional; transcripts without CDS get an all-zero CDS channel.
- A target CSV/TSV when training supervised models. It should contain one row
  per transcript, a transcript id column such as `transcript_id`, and a numeric
  target column such as `log_kdeg`. A `split` column with `train`, `val`, and
  `test` is optional.

Transcripts on GTF chromosomes that are absent from the FASTA are skipped and
reported in `config.json` under `skipped_missing_fasta_chromosomes`.

Saluki arrays have shape `(N, 6, L)`: A/C/G/U, CDS codon-start positions, and
splice-junction positions. Short transcripts are padded with all-zero columns.
Long transcripts are truncated from the 5-prime side, preserving the 3-prime
window, matching the legacy Saluki-style preprocessing.

### Build Datasets

Build Saluki input directly from GTF/FASTA:

```bash
transcriptml build-saluki-gtf \
  --gtf annotations.gtf \
  --fasta genome.fa \
  --out-dir data/saluki \
  --targets targets.csv \
  --target-id-col transcript_id \
  --target-col log_kdeg \
  --split-col split \
  --length 12288
```

Only transcripts present in both the GTF and target table are kept. If no target
table is provided, all transcript isoforms with exon annotations are encoded and
`y.npy` is omitted.

If you already have transcript sequences and transcript-coordinate annotations,
use the table builder instead:

```bash
transcriptml build-saluki \
  --table transcripts.csv \
  --out-dir data/saluki_table \
  --sequence-col sequence \
  --id-col transcript_id \
  --target-col log_kdeg \
  --cds-positions-col cds_starts \
  --splice-positions-col splice_sites \
  --split-col split \
  --length 12288
```

For MPRA-style sequence/target tables:

```bash
transcriptml build-mpra mpra.csv data/mpra \
  --sequence-col sequence \
  --target-col activity \
  --id-col reporter_id \
  --split-col split
```

### Train Saluki

Training is config-driven. Example `train_saluki.json`:

```json
{
  "dataset": "data/saluki",
  "output_dir": "runs/saluki_exact",
  "model": {
    "name": "saluki_exact",
    "params": {
      "seq_depth": 6,
      "filters": 64,
      "kernel_size": 5,
      "num_layers": 6,
      "dropout": 0.3,
      "augment_shift": 3
    }
  },
  "batch_size": 64,
  "epochs": 50,
  "learning_rate": 0.001,
  "weight_decay": 0.0,
  "patience": 8,
  "monitor": "val_loss",
  "device": "auto",
  "num_workers": 0,
  "mmap_mode": "r",
  "split": {"method": "metadata", "split_col": "split"}
}
```

Then run:

```bash
transcriptml train train_saluki.json
```

The `"saluki_exact"` model trains the closer PyTorch reproduction of the
Basenji/Saluki architecture. Checkpoints include model config and weights, so
they can be reloaded without restating hyperparameters.

### Evaluate

```bash
transcriptml evaluate runs/saluki_exact/best.pt data/saluki predictions.csv --split test --device auto
```

This writes per-transcript predictions and a companion summary JSON.

### Interpret Saluki

Single-nucleotide ISM:

```bash
transcriptml ism runs/saluki_exact/best.pt data/saluki interpret/ism \
  --device auto \
  --mutation-batch-size 512
```

Full-length ISM over many 12,288 nt transcripts can be expensive because it
evaluates three mutants per valid base. For first-pass Saluki interpretation,
motif-centered analyses are usually much cheaper.

CDS codon ISM writes a long-form mutation table with one row per codon
mutation and effect `mutant_prediction - reference_prediction`. By default it
only scans synonymous alternatives. Use `--mutation-policy all-codons` to scan
all 63 alternatives per reference codon; stop codons are included unless
`--exclude-stop-codons` is set. Stop codon ISMs are often not interpretable (Saluki does not learn canonical NMD unless trained on PTC-containing isoforms and thus won't predict PTCs to be destabilizing) or observe weird artifacts (fun example: UGA looks to be very stabilizing as it is only found within the CDS in selenocysteine protein mRNAs where it is decoded as selenocysteine; these mRNAs are on average highly stable, leading to a strong spurious correlation between UGA presence and stabilization), but I like to include them as a nice example of out-of-distribution hallucination.

```bash
transcriptml codon-ism runs/saluki_exact/best.pt data/saluki interpret/codon_ism \
  --device auto \
  --mutation-policy synonymous-only \
  --table-format npz \
  --mutation-batch-size 512
```

For large jobs, `codon-ism` streams the mutation table as CSV, chunked NPZ,
Parquet, or Arrow IPC. Parquet and Arrow output require installing the optional
`arrow` extra. Add `--position-scores` to also save the legacy-style derived
max-absolute codon effect projected back onto reference nucleotide channels.

Motif ablation, with effect `A - R`:

```bash
transcriptml motif-ablation runs/saluki_exact/best.pt data/saluki interpret/pre_ablation \
  --motif "UGUA[A|U|C]AUA" \
  --n-scrambles 10 \
  --device auto
```

Motif context specificity, with effect `(MA - M) - (A - R)`:

```bash
transcriptml motif-context runs/saluki_exact/best.pt data/saluki interpret/pre_context \
  --motif "UGUA[A|U|C]AUA" \
  --window-size 5 \
  --context-width 100 \
  --n-scrambles 10 \
  --n-window-scrambles 5 \
  --device auto
```

Pairwise motif epistasis, with effect `A12 - A1 - A2 + R`:

```bash
transcriptml epistasis runs/saluki_exact/best.pt data/saluki interpret/pre_epistasis \
  --motif "UGUA[A|U|C]AUA" \
  --n-scrambles 10 \
  --max-pairs 5000 \
  --device auto
```

Interpretation outputs are saved as `.npy` arrays plus CSV tables describing the
motif instances or motif pairs that were tested.

### Performance Notes

- Saluki builders write `X.npy` directly as a NumPy `.npy` memmap instead of
  holding an extra full-size array in RAM.
- `train_from_config` loads bundles with `mmap_mode: "r"` by default and reads
  batches lazily during training.
- Evaluation and prediction convert data to float32 per batch, not for the
  whole dataset at once.
- Motif scanning is vectorized over possible start positions.
