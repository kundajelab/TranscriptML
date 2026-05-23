# TranscriptML

Training and interpreting RNA sequence-to-function models.

TranscriptML uses a `src/` layout and the import name `transcriptml`. The first
package version focuses on reusable APIs for RNA one-hot encoding, dataset
bundles, compact neural models, training/evaluation, and interpretation.

## Install

For core model training and tests:

```bash
git clone https://github.com/kundajelab/TranscriptML.git
cd TranscriptML
python -m pip install -e ".[dev]"
```

For direct GTF/FASTA transcript extraction, install the genomics extra:

```bash
python -m pip install -e ".[dev,genomics]"
```

## Quickstart

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
transcriptml build-saluki-gtf annotations.gtf genome.fa data/saluki \
  --targets targets.csv \
  --target-id-col transcript_id \
  --target-col log_kdeg \
  --split-col split \
  --length 12880
```

Only transcripts present in both the GTF and target table are kept. If no target
table is provided, all transcript isoforms with exon annotations are encoded and
`y.npy` is omitted.

If you already have transcript sequences and transcript-coordinate annotations,
use the table builder instead:

```bash
transcriptml build-saluki transcripts.csv data/saluki_table \
  --sequence-col sequence \
  --id-col transcript_id \
  --target-col log_kdeg \
  --cds-positions-col cds_starts \
  --splice-positions-col splice_sites \
  --split-col split \
  --length 12880
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
  "output_dir": "runs/saluki_gru",
  "model": {
    "name": "saluki_gru",
    "params": {
      "in_ch": 6,
      "base_ch": 64,
      "n_convs": 4,
      "gru_hidden": 64,
      "head_hidden": 64,
      "dropout": 0.2
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

Use `"name": "saluki_exact"` to train the closer PyTorch reproduction of the
Basenji/Saluki architecture, or `"saluki_gru"` for the lighter Saluki-inspired
Conv/GRU model. Checkpoints include model config and weights, so they can be
reloaded without restating hyperparameters.

### Evaluate

```bash
transcriptml evaluate runs/saluki_gru/best.pt data/saluki predictions.csv --split test --device auto
```

This writes per-transcript predictions and a companion summary JSON.

### Interpret Saluki

Single-nucleotide ISM:

```bash
transcriptml ism runs/saluki_gru/best.pt data/saluki interpret/ism \
  --device auto \
  --mutation-batch-size 512
```

Full-length ISM over many 12,880 nt transcripts can be expensive because it
evaluates three mutants per valid base. For first-pass Saluki interpretation,
motif-centered analyses are usually much cheaper.

Motif ablation, with effect `A - R`:

```bash
transcriptml motif-ablation runs/saluki_gru/best.pt data/saluki interpret/pre_ablation \
  --motif "UGUA[A|U|C]AUA" \
  --n-scrambles 10 \
  --device auto
```

Motif context specificity, with effect `(MA - M) - (A - R)`:

```bash
transcriptml motif-context runs/saluki_gru/best.pt data/saluki interpret/pre_context \
  --motif "UGUA[A|U|C]AUA" \
  --window-size 5 \
  --context-width 100 \
  --n-scrambles 10 \
  --n-window-scrambles 5 \
  --device auto
```

Pairwise motif epistasis, with effect `A12 - A1 - A2 + R`:

```bash
transcriptml epistasis runs/saluki_gru/best.pt data/saluki interpret/pre_epistasis \
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
