# Usage

TranscriptML stores processed data in a *dataset bundle*. A bundle contains the
encoded sequences in `X.npy`, optional targets in `y.npy`, sequence identifiers,
a channel schema, and optional metadata and split definitions. Training and
interpretation commands consume the same bundle, which keeps sequence encoding
consistent across a project.

Run `transcriptml --help` for the full command list and
`transcriptml <command> --help` for command-specific options.
Training configurations accept `"device": "auto"`. Evaluation and
interpretation commands take an explicit Torch device such as `cpu` or `cuda`;
the local examples below use `cpu` for portability.

## Saluki workflow

Use this workflow when each example is an annotated transcript paired with a
transcriptome-derived scalar estimate, such as an RNA degradation rate. Saluki
uses six input channels: A, C, G, U, CDS codon starts, and splice junctions.

### 1. Build Saluki input

The most direct inputs are:

- a genome FASTA;
- a GTF containing `exon` rows and `transcript_id` attributes, with `CDS` rows
  when CDS-aware analyses are needed; and
- a CSV or TSV with one row per transcript, a matching transcript identifier,
  and a numeric target.

For example:

```text
transcript_id,log_kdeg,log_kdeg_se
ENST00000335137.4,-1.28,0.11
ENST00000423372.3,-0.76,0.09
```

Build a 12,288 nt Saluki bundle:

```bash
transcriptml build-saluki-gtf \
  --gtf annotations.gtf \
  --fasta genome.fa \
  --targets targets.csv \
  --target-id-col transcript_id \
  --target-col log_kdeg \
  --out-dir data/saluki \
  --length 12288
```

Only transcript identifiers present in both the annotation and target table are
kept. Short transcripts are padded on the 3-prime side. Long transcripts retain
the 3-prime-most 12,288 nt. Transcripts without `CDS` rows receive an all-zero
CDS channel.

If the table already contains transcript sequence, CDS-start positions, and
splice positions in transcript coordinates, use `build-saluki` instead. See its
help output for the expected columns.

### 2. Train with 10-fold cross-validation

Ten-fold cross-validation is the recommended default for reporting model
performance. Fold `i` is held out for testing, fold `i + 1` is used for
validation, and the other eight folds are used for training. The repository
includes a helper that creates lightweight fold bundles using symbolic links,
so the encoded array is not copied ten times.

Copy and edit the supplied base configuration:

```bash
cp scripts/example_train_config.json train_saluki.json
```

The helper replaces `dataset` and `output_dir`; edit the model and training
settings in `train_saluki.json`. Then train the folds sequentially on a local
machine:

```bash
for fold in $(seq 0 9); do
  config=$(python scripts/write_cv_fold_artifacts.py \
    --dataset data/saluki \
    --base-config train_saluki.json \
    --cv-root runs/saluki_cv10 \
    --fold "${fold}" \
    --n-folds 10 \
    --seed 42)

  transcriptml train "${config}"
  mkdir -p "runs/saluki_cv10/fold${fold}/eval"
  transcriptml evaluate \
    "runs/saluki_cv10/fold${fold}/model/best.pt" \
    "runs/saluki_cv10/fold${fold}/dataset" \
    "runs/saluki_cv10/fold${fold}/eval/test_predictions.csv" \
    --split test \
    --device cpu
done
```

Each fold writes a checkpoint, training history, summary, and held-out
predictions under `runs/saluki_cv10/fold*/`. Combine the ten held-out prediction
tables to report performance across the full dataset.

The supplied helper assigns rows randomly. Create grouped folds upstream when
related isoforms, homologous transcripts, or other biological groups must stay
together. A predefined `split` column is also supported for a single
train/validation/test run by passing `--split-col split` during the build.

### 3. Run single-nucleotide ISM

ISM substitutes each valid reference base with the other three bases and saves
`mutant_prediction - reference_prediction` in `deltas.npy`. Run it with one
fold checkpoint as follows:

```bash
transcriptml ism \
  runs/saluki_cv10/fold0/model/best.pt \
  data/saluki \
  interpret/saluki_ism/fold0 \
  --device cpu \
  --batch-size 128 \
  --mutation-batch-size 512
```

Repeat this for all fold checkpoints when comparing interpretation results
across models. Full-transcript ISM evaluates three mutants per valid nucleotide,
so start with a small sequence set or one fold when checking a new setup.

Plot a transcript or a short window from the result:

```bash
transcriptml plot-ism \
  --ism interpret/saluki_ism/fold0/deltas.npy \
  --dataset data/saluki \
  --seq-index 0 \
  --start 4000 \
  --end 4300 \
  --out interpret/saluki_ism/fold0/transcript0.png
```

### 4. Other interpretation analyses

Motif ablation replaces each motif occurrence with scrambled alternatives. It
is a focused and substantially cheaper test than full-length ISM:

```bash
transcriptml motif-ablation \
  runs/saluki_cv10/fold0/model/best.pt \
  data/saluki \
  interpret/pre_ablation/fold0 \
  --motif 'UGUA[A|U|C]AUA' \
  --region 3utr \
  --n-scrambles 10 \
  --device cpu
```

Motif epistasis tests whether ablating two occurrences together has a
non-additive effect. Use `--motif2` for two different motifs or omit it for
pairs of the same motif:

```bash
transcriptml epistasis \
  runs/saluki_cv10/fold0/model/best.pt \
  data/saluki \
  interpret/pre_are_epistasis/fold0 \
  --motif 'UGUA[A|U|C]AUA' \
  --motif2 'UUAUUUAUU' \
  --region 3utr \
  --max-pairs 5000 \
  --device cpu
```

`motif-context` tests whether nearby sequence changes a motif's effect. Codon
ISM substitutes CDS codons while preserving the rest of the transcript. The
default synonymous scan is usually the clearest first analysis:

```bash
transcriptml codon-ism \
  runs/saluki_cv10/fold0/model/best.pt \
  data/saluki \
  interpret/codon_ism/fold0 \
  --mutation-policy synonymous-only \
  --exclude-stop-codons \
  --table-format npz \
  --device cpu
```

Use `--mutation-policy all-codons` for all alternatives. Large all-codon scans
can be sharded with `--sequence-shard-index` and `--sequence-shards`. Parquet or
Arrow output requires the `arrow` or `analysis` installation extra.

### HPC-optimized Saluki workflow

The checked-in `scripts/` directory provides SLURM job arrays for dataset
building, 10-fold training, ISM, motif analyses, and codon ISM. The defaults
target Stanford Sherlock and include the `akundaje` partition, 48 GB GPU
constraints, and Kundaje lab module versions. Change these values for another
cluster.

Keep the repository checkout clean and copy the scripts to a run directory:

```bash
TRANSCRIPTML_REPO=/home/users/isvock/TranscriptML
RUN_WORKDIR=/scratch/users/isvock/transcriptml_runs/human_kdeg

mkdir -p "${RUN_WORKDIR}"
cp -R "${TRANSCRIPTML_REPO}/scripts" "${RUN_WORKDIR}/scripts"
cd "${RUN_WORKDIR}"
```

Edit the user settings in `scripts/sherlock_config.sh`. A typical Sherlock
configuration is:

```bash
TRANSCRIPTML_REPO=/home/users/isvock/TranscriptML
CONDA_ENV=transcript-ml
SHERLOCK_CONDA_ROOT=/home/groups/akundaje/miniconda

GTF=/oak/stanford/groups/akundaje/refs/gencode.v44.annotation.gtf
FASTA=/oak/stanford/groups/akundaje/refs/GRCh38.primary_assembly.genome.fa
TARGETS=/scratch/users/isvock/rna_decay/targets.csv
TARGET_ID_COL=transcript_id
TARGET_COL=log_kdeg

RUN_ROOT=/scratch/users/isvock/transcriptml_runs/human_kdeg/results
DATASET_DIR=${RUN_ROOT}/data/saluki
CV_ROOT=${RUN_ROOT}/cv10
INTERPRET_ROOT=${RUN_ROOT}/interpret
N_FOLDS=10
DEVICE=cuda
```

Also edit `scripts/example_train_config.json`, then submit the stages:

```bash
sbatch scripts/build_saluki_gtf.sh
bash scripts/submit_train_eval_cv.sh
bash scripts/submit_ism_by_fold.sh
bash scripts/submit_motif_ablation_by_fold.sh
bash scripts/submit_motif_epistasis_by_fold.sh
bash scripts/submit_codon_ism_by_fold.sh
```

Wait for each dependent stage to finish before submitting the next one. For an
all-codon scan, `submit_all_codon_ism_shard_by_fold.sh` creates an additional
job array within each fold. Review the `#SBATCH` lines in the copied worker
scripts and adjust partitions, GPU constraints, memory, time limits, modules,
and paths for the local system. See the full
[Saluki script reference](https://github.com/kundajelab/TranscriptML/blob/main/scripts/README.md)
for every configuration option and expected output path.

## MPRA workflow

The MPRA workflow accepts any assay that can be reduced to one variable sequence
insert and one numeric measurement per example. Suitable measurements include
RNA abundance or stability, translation, ribosome recruitment, protein output,
and RNA localization. Examples include 3-prime UTR stability assays such as
[UTR-Seq](https://doi.org/10.1016/j.molcel.2017.11.014),
[RESA](https://doi.org/10.1038/nmeth.4121), and the 2022
[fast-UTR assay](https://doi.org/10.1093/g3journal/jkab404). Translation
examples include a large 5-prime UTR polysome-profiling
[MPRA](https://doi.org/10.1038/s41587-019-0164-5) and the human DART study from
[Lewis and Xie et al. (2025)](https://doi.org/10.1016/j.molcel.2024.11.030).
Sequence-to-localization measurements, such as this neuronal 3-prime UTR
[MPRA](https://doi.org/10.1093/nar/gkac779), also fit this representation.

Keep the experimental context consistent across rows. Aggregate technical
barcodes or replicates before training, or keep them in the same fold. Designs
with multiple independently varying inserts require custom preprocessing and
possibly a custom model.

### 1. Build MPRA input

Prepare a CSV or TSV with one row per distinct insert. DNA and RNA alphabets are
both accepted; `T` is encoded as `U`.

```text
variant_id,insert_sequence,activity
var_0001,ACTGGTAATTAA,-0.21
var_0002,TGTGCATACTGA,0.34
var_0003,ATTTGGACTTAC,0.08
```

Build the four-channel A/C/G/U bundle:

```bash
transcriptml build-mpra mpra.csv data/mpra \
  --sequence-col insert_sequence \
  --target-col activity \
  --id-col variant_id
```

By default, the encoded length is the longest input sequence. Shorter inserts
are right-padded with all-zero columns. Set `--length` when the reporter design
has a defined input width. Inserts longer than that width retain their
3-prime-most bases, so trim or align sequences upstream if another convention
is biologically appropriate.

### 2. Train LegNet with 10-fold cross-validation

Copy and edit the LegNet base configuration:

```bash
cp scripts/mpra/example_legnet_train_config.json train_legnet.json
```

Run the same held-out-fold design used for Saluki:

```bash
for fold in $(seq 0 9); do
  config=$(python scripts/mpra/write_cv_fold_artifacts.py \
    --dataset data/mpra \
    --base-config train_legnet.json \
    --cv-root runs/mpra_cv10 \
    --fold "${fold}" \
    --n-folds 10 \
    --seed 42)

  transcriptml train "${config}"
  mkdir -p "runs/mpra_cv10/fold${fold}/eval"
  transcriptml evaluate \
    "runs/mpra_cv10/fold${fold}/model/best.pt" \
    "runs/mpra_cv10/fold${fold}/dataset" \
    "runs/mpra_cv10/fold${fold}/eval/test_predictions.csv" \
    --split test \
    --device cpu
done
```

Allelic pairs, overlapping tiles, and inserts derived from the same native
region can be easy for a model to recognize. Keep such groups in one fold when
the goal is to measure generalization to new sequence contexts.

### 3. Interpret the MPRA model

Single-nucleotide ISM uses the same command as the Saluki workflow:

```bash
transcriptml ism \
  runs/mpra_cv10/fold0/model/best.pt \
  data/mpra \
  interpret/mpra_ism/fold0 \
  --device cpu \
  --mutation-batch-size 512
```

Motif ablation, motif context, and motif epistasis also work with four-channel
MPRA bundles. Region filters such as `--region 3utr` require Saluki annotation
channels, so omit them for ordinary MPRA input:

```bash
transcriptml motif-ablation \
  runs/mpra_cv10/fold0/model/best.pt \
  data/mpra \
  interpret/mpra_are_ablation/fold0 \
  --motif 'UUAUUUAUU' \
  --n-scrambles 10 \
  --device cpu
```

Interpret MPRA results in the assay's exact reporter context. For example,
single-base effects may reflect cryptic splice sites, unintended promoter
activity, or other construct-specific behavior in addition to the intended RNA
regulatory mechanism.

### HPC-optimized MPRA workflow

The `scripts/mpra/` directory supplies Sherlock-oriented jobs for input
building, LegNet CV, hyperparameter sweeps, and ISM. Copy it into a run
directory:

```bash
TRANSCRIPTML_REPO=/home/users/isvock/TranscriptML
RUN_WORKDIR=/scratch/users/isvock/transcriptml_runs/mpra_legnet

mkdir -p "${RUN_WORKDIR}/scripts"
cp -R "${TRANSCRIPTML_REPO}/scripts/mpra" "${RUN_WORKDIR}/scripts/mpra"
cd "${RUN_WORKDIR}"
```

Edit `scripts/mpra/mpra_config.sh`:

```bash
TRANSCRIPTML_REPO=/home/users/isvock/TranscriptML
CONDA_ENV=transcript-ml
SHERLOCK_CONDA_ROOT=/home/groups/akundaje/miniconda

MPRA_TABLE=/scratch/users/isvock/mpra/mpra_activity.csv
SEQUENCE_COL=insert_sequence
TARGET_COL=activity
ID_COL=variant_id

RUN_ROOT=/scratch/users/isvock/transcriptml_runs/mpra_legnet/results
DATASET_DIR=${RUN_ROOT}/data/mpra
CV_ROOT=${RUN_ROOT}/cv10
INTERPRET_ROOT=${RUN_ROOT}/interpret
N_FOLDS=10
DEVICE=cuda
```

Edit `scripts/mpra/example_legnet_train_config.json`, then run:

```bash
sbatch scripts/mpra/build_legnet_input.sh
bash scripts/mpra/submit_train_eval_cv.sh
bash scripts/mpra/submit_ism_by_fold.sh
```

The last two commands submit ten-task SLURM arrays. Use
`make_legnet_hparam_grid.py` and `submit_hparam_sweep_cv.sh` before the final CV
run when tuning is needed. As with the Saluki scripts, change the checked-in
Sherlock partitions, 48 GB GPU constraints, modules, resources, and filesystem
paths for another cluster. See the full
[MPRA script reference](https://github.com/kundajelab/TranscriptML/blob/main/scripts/mpra/README.md)
for the complete set of settings and outputs.
