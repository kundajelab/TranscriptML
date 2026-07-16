# Usage

This page walks through the two main TranscriptML workflows:

- **Saluki**, for transcriptome-derived RNA stability measurements.
- **MPRA-LegNet**, for single-insert MPRA-style measurements.

For each workflow, the basic pattern is the same:

1. Build a TranscriptML dataset bundle.
2. Train models, usually with cross-validation.
3. Evaluate held-out predictions.
4. Run interpretation analyses, such as single-nucleotide ISM or motif ablation.

At the end, this page briefly points to additional analyses that are available
once a trained model and dataset bundle exist.

## Saluki Workflow

Saluki is a model described in [Agarwal et al.
2022](https://link.springer.com/article/10.1186/s13059-022-02811-x). It predicts
transcriptome-wide RNA stability from transcript sequence plus transcript
structure annotations. In TranscriptML, a Saluki input has six channels:
A/C/G/U sequence channels, one CDS annotation channel, and one splice-junction
annotation channel.

<details>
<summary>How TranscriptML's Saluki differs from the original</summary>

TranscriptML's Saluki implementation is intentionally single-task. The original
Saluki model jointly predicted mouse and human stability measurements.
Multi-task models can be powerful, but they also make it easier for a model to
share information between tasks in ways that are hard to diagnose and may not reflect actual biology. A single-task
model is a cleaner default when the goal is model interpretation.

</details>

### 1. Build Saluki Input

TranscriptML stores processed data in a *dataset bundle*. A bundle is a
directory containing encoded arrays, targets, transcript IDs, schema metadata,
and optional split information. Most downstream commands take a bundle directory
as input.

The most direct inputs for building a Saluki bundle are:

- a genome FASTA;
- a GTF containing `exon` rows and `transcript_id` attributes, with `CDS` rows
  when CDS-aware analyses are needed; and
- a CSV or TSV with one row per transcript, a matching transcript identifier,
  and a numeric target.

The target table might look something like this:

```text
transcript_id,log_kdeg
ENST00000335137.4,-1.28
ENST00000423372.3,-0.76
```

Build the bundle with `build-saluki-gtf`:

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

The `--length` value controls the encoded transcript width. The original Saluki
model used `12288`. Longer transcripts are truncated from the 5-prime end, and
shorter transcripts are padded at the 3-prime end.

Only transcript IDs present in both the GTF and the target table are kept.
Transcripts without `CDS` rows receive an all-zero CDS channel, which means they
can still be used for ordinary training and prediction, but CDS-specific
analyses will not find codons in those transcripts.

The output directory contains files such as:

```text
data/saluki/
  X.npy
  y.npy
  ids.txt
  metadata.json
  schema.json
  config.json
```

If your table already contains transcript sequences, CDS positions, and splice
positions in transcript coordinates, use `transcriptml build-saluki` instead.
Run `transcriptml build-saluki --help` for the expected column names.

### 2. Train Saluki With 10-Fold Cross-Validation

N-fold cross-validation (CV) is the recommended default for estimating model
performance. For fold `i`, TranscriptML uses fold `i` as the test split, fold
`i + 1` as the validation split, and the remaining folds for training. Saluki
originally used 10-fold CV, so that is what we will demonstrate here.

<details>
<summary>The Benefits and Challenges of Cross-Validation</summary>

For model interpretation analyses, cross-validation is a useful internal
control. By training the model from scratch on several slightly different
training sets, you can ask whether a motif, nucleotide effect, codon effect, or
other conclusion is reproducible across independently trained models.

Fold-to-fold variation can also give a practical sense of how sensitive model
predictions or interpretation scores are to the training split. This is not a
fully calibrated uncertainty estimate, but it is often a valuable stability
check.

The downside is that many downstream analyses need to be run once per fold. For
example, interpreting a 10-fold CV run usually means running the same ISM or
motif analysis 10 separate times, one per checkpoint. This workflow is much
easier on an HPC system with a scheduler, where each fold can run as a separate
job. See the HPC workflow below for that pattern.

</details>

First write a starter config. This JSON file specifies the model and training
settings:

```bash
transcriptml init-run --workflow saluki --out-dir configs/saluki
```

Edit `configs/saluki/train_config.json` to adjust settings such as
`batch_size`, `epochs`, `learning_rate`, `patience`, and model parameters. For
cross-validation, the fold-preparation command will replace `dataset` and
`output_dir`, so those fields can stay as placeholders in the base config.

Run the folds sequentially:

```bash
for fold in $(seq 0 9); do
  # This writes the fold-specific config and captures its path.
  config=$(
    transcriptml cv prepare-fold \
      --dataset data/saluki \
      --base-config configs/saluki/train_config.json \
      --cv-root runs/saluki_cv10 \
      --fold "${fold}" \
      --model saluki_exact \
      --n-folds 10 \
      --seed 42 \
      --val-offset 1
  )

  transcriptml train "${config}"

  mkdir -p "runs/saluki_cv10/fold${fold}/eval"
  transcriptml evaluate \
    --checkpoint "runs/saluki_cv10/fold${fold}/model/best.pt" \
    --dataset "runs/saluki_cv10/fold${fold}/dataset" \
    --out-csv "runs/saluki_cv10/fold${fold}/eval/test_predictions.csv" \
    --split test \
    --device auto
done
```

Each fold writes:

```text
runs/saluki_cv10/fold0/
  dataset/splits.json
  train_config.json
  model/best.pt
  model/last.pt
  model/history.json
  model/summary.json
  model/test_predictions.csv
  eval/test_predictions.csv
  eval/test_predictions.summary.json
```

The `model/summary.json` file records the best epoch, monitored validation
metrics under `best_monitor_values`, split source information, split counts,
and test metrics. The `eval/test_predictions.csv` file contains held-out
predictions for that fold. Concatenate the ten fold-level prediction tables to
measure performance across the full dataset.

The built-in fold assignment is random and transcript-level. If related
isoforms, homologous transcripts, or other biological groups must stay
together, create grouped folds upstream and run one predefined split per fold.

### 3. Run Single-Nucleotide ISM

Single-nucleotide ISM mutates each valid reference base to the other three
bases and records:

```text
mutant_prediction - reference_prediction
```

Run ISM once per fold checkpoint:

```bash
for fold in $(seq 0 9); do
  transcriptml ism \
    --checkpoint "runs/saluki_cv10/fold${fold}/model/best.pt" \
    --dataset data/saluki \
    --out-dir "interpret/saluki_ism/fold${fold}" \
    --device auto \
    --batch-size 128 \
    --mutation-batch-size 512
done
```

Each ISM output directory contains:

```text
interpret/saluki_ism/fold0/
  deltas.npy
  reference_predictions.npy
  valid_lengths.npy
  max_abs_effect.npy
  summary.json
```

The gold-standard downstream workflow is to average the fold-level ISM arrays
and mean-center them across nucleotide channels. The averaged deltas preserve
the model-effect scale, while the mean-centered version behaves more like a
feature-attribution track and is the better input for MoDISco-style motif
discovery. This assumes each fold's ISM was run on the same dataset in the same
sequence order, as in the loop above.

```bash
transcriptml summarize-ism \
  --input-dir interpret/saluki_ism \
  --out-dir interpret/saluki_ism_summary \
  --dataset data/saluki \
  --write-fold-std \
  --write-projected
```

This writes:

```text
interpret/saluki_ism_summary/
  average_deltas.npy
  average_mean_centered_deltas.npy
  average_projected_mean_centered_deltas.npy
  fold_std_deltas.npy
  ids.txt
  summary.json
```

Use `average_mean_centered_deltas.npy` when you want a four-channel
hypothetical-effect style score, and use
`average_projected_mean_centered_deltas.npy` when you want scores only on the
observed reference bases.

Full-transcript ISM can be expensive. A Saluki transcript of length `12288`
requires up to `3 * 12288` mutant predictions per transcript. Start with one
fold or a small subset when checking a new setup. On an L40S, one-fold ISM of a
standard full dataset of about 10,000 transcripts usually takes between 4 and 8
hours.

Plot one transcript or a short window from the averaged, mean-centered ISM:

```bash
transcriptml plot-ism \
  --ism interpret/saluki_ism_summary/average_mean_centered_deltas.npy \
  --dataset data/saluki \
  --seq-index 0 \
  --start 4000 \
  --end 4300 \
  --out interpret/saluki_ism_summary/transcript0_4000_4300.png
```

Use `--gene-id`, or `--metadata-field` and `--metadata-value`, when the bundle
metadata provides a more meaningful lookup than sequence index.

When `--dataset` is provided, `plot-ism` automatically uses `X.npy` to draw a
reference-base logo track, and Saluki-style six-channel inputs also get a compact
CDS/splice isoform track. The logo is useful for short windows where individual
bases are readable, especially around motifs or strong local effects. For long
windows or whole transcripts, the letters become dense and distracting; use
`--no-logo` in those cases. Use `--no-isoform` if the annotation track is not
needed for a particular figure.

### 4. Run Motif Analyses

Motif analyses are usually much cheaper than full-transcript ISM and are designed
to hone in on the context specificity and syntax of a particular motif or set of motifs.

Motif ablation replaces each motif occurrence with scrambled alternatives and
measures the change in prediction:

```bash
transcriptml motif-ablation \
  --checkpoint runs/saluki_cv10/fold0/model/best.pt \
  --dataset data/saluki \
  --out-dir interpret/pre_ablation/fold0 \
  --motif 'UGUA[A|U|C]AUA' \
  --region 3utr \
  --n-scrambles 10 \
  --device auto
```

Motif context asks whether nearby sequence changes a motif's effect:

```bash
transcriptml motif-context \
  --checkpoint runs/saluki_cv10/fold0/model/best.pt \
  --dataset data/saluki \
  --out-dir interpret/pre_context/fold0 \
  --motif 'UGUA[A|U|C]AUA' \
  --region 3utr \
  --window-size 5 \
  --context-width 100 \
  --n-window-scrambles 5 \
  --device auto
```

Motif epistasis tests whether ablating two motif occurrences together has a
non-additive effect. Use `--motif2` for two different motifs, or omit it to test
pairs of the same motif:

```bash
transcriptml epistasis \
  --checkpoint runs/saluki_cv10/fold0/model/best.pt \
  --dataset data/saluki \
  --out-dir interpret/pre_are_epistasis/fold0 \
  --motif 'UGUA[A|U|C]AUA' \
  --motif2 'UUAUUUAUU' \
  --region 3utr \
  --max-pairs 5000 \
  --device auto
```

The `--region` flag can be `5utr`, `cds`, or `3utr`. Omit it to analyze motif
instances across the full transcript. Region-aware analyses require Saluki-style
annotation channels.

### 5. Run Codon Analyses

Lots of work has shown that the coding sequence of an mRNA strongly influences its
stability. Codon analyses are designed to dissect Saluki's understanding of this influence.

Synonymous codon ISM substitutes CDS codons while preserving the rest of the
transcript. This is the clearest first codon-level perturbation, and can help you
assess trends related to within-amino-acid-family optimality:

```bash
transcriptml codon-ism \
  --checkpoint runs/saluki_cv10/fold0/model/best.pt \
  --dataset data/saluki \
  --out-dir interpret/codon_ism/fold0 \
  --mutation-policy synonymous-only \
  --exclude-stop-codons \
  --table-format npz \
  --device auto
```

Use `--mutation-policy all-codons` when you want all alternative codons rather
than synonymous alternatives only. Large all-codon scans can be split with
`--sequence-shard-index` and `--sequence-shards`. Parquet and Arrow output need
the `arrow` or `analysis` installation extra. All-codon scans take considerably
more time than synonymous codon scans. Synonymous codon scans are typically a bit more
efficient than single-nt ISM, taking on the order of hours, whereas all-codon scans may take
multiple days. In practice, I like to split all-codon scans up into several indepdenent jobs 
per fold; see the TranscriptML [scripts](https://github.com/kundajelab/TranscriptML/blob/main/scripts/submit_all_codon_ism_shard_by_fold.sh) for an example.

### HPC-Optimized Saluki Workflow

The `scripts/` directory contains Sherlock-oriented SLURM jobs for Saluki input
building, 10-fold CV, hyperparameter sweeps, ISM, motif analyses, and codon ISM.
The scripts are intentionally editable. Copy them to a run directory, edit the
copied config, and leave the clean repository checkout alone.

```bash
TRANSCRIPTML_REPO=/home/users/isvock/TranscriptML
RUN_WORKDIR=/scratch/users/isvock/transcriptml_runs/human_kdeg

mkdir -p "${RUN_WORKDIR}"
cp -R "${TRANSCRIPTML_REPO}/scripts" "${RUN_WORKDIR}/scripts"
cd "${RUN_WORKDIR}"
```

Edit `scripts/sherlock_config.sh`:

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
CV_MODEL=saluki_exact
DEVICE=cuda
```

Edit `scripts/example_train_config.json` for model and training
hyperparameters, then submit stages:

```bash
sbatch scripts/build_saluki_gtf.sh
bash scripts/submit_train_eval_cv.sh
bash scripts/submit_ism_by_fold.sh
bash scripts/submit_motif_ablation_by_fold.sh
bash scripts/submit_motif_epistasis_by_fold.sh
bash scripts/submit_codon_ism_by_fold.sh
```

Wait for each dependent stage to finish before submitting the next one. For
all-codon ISM, `submit_all_codon_ism_shard_by_fold.sh` creates a shard array
inside each fold. Review the `#SBATCH` lines in the copied worker scripts and
adjust partition, GPU constraints, memory, time limits, modules, and paths for
your cluster.

See the [Saluki script reference](https://github.com/kundajelab/TranscriptML/blob/main/scripts/README.md)
for all script configuration options and expected output paths.

## MPRA-LegNet Workflow

Massively parallel reporter assays (MPRAs) are powerful experimental tools for
dissecting the sequence grammar of biochemical processes. Training a deep learning
model on this data and interpreting it can be a uniquely powerful strategy for 
identifying regulatory motifs and their context specificity. It can also help uncover
assay-specific biases that may plague a particular dataset.

In my opinion, one of the best general MPRA models is MPRA-LegNet, described in
[Agarwal et al. 2025](https://www.nature.com/articles/s41586-024-08430-9) (Another
Vikram Agarwal model; the guy is GOATed).

The TranscriptML MPRA workflow accepts assays that can be reduced to one variable sequence
insert and one numeric measurement per example. Suitable targets include RNA
abundance or stability, translation, ribosome recruitment, protein output, and
RNA localization.

Keep the experimental context consistent across rows. Aggregate technical
barcodes or replicates before training, or keep related rows in the same fold.
Allelic pairs, overlapping tiles, and inserts derived from the same native
region can be easy for a model to recognize, so grouped splits may be important
when measuring generalization to new sequence contexts.

### 1. Build MPRA Input

Prepare a CSV or TSV with one row per distinct insert. DNA and RNA alphabets are
both accepted; `T` is encoded as `U`.

```text
variant_id,insert_sequence,activity
var_0001,ACTGGTAATTAA,-0.21
var_0002,TGTGCATACTGA,0.34
var_0003,ATTTGGACTTAC,0.08
```

Build a four-channel A/C/G/U bundle:

```bash
transcriptml build-mpra mpra.csv data/mpra \
  --sequence-col insert_sequence \
  --target-col activity \
  --id-col variant_id
```

By default, the encoded length is the longest input sequence. Shorter inserts
are right-padded with all-zero columns. Set `--length` when the reporter design
has a fixed input width. Inserts longer than that width keep their 3-prime-most
bases, so trim or align sequences upstream if another convention is
biologically appropriate.

### 2. Train LegNet With 10-Fold Cross-Validation

Write a starter LegNet config:

```bash
transcriptml init-run --workflow legnet --out-dir configs/legnet
```

Edit `configs/legnet/train_config.json`, then run CV:

```bash
for fold in $(seq 0 9); do
  config=$(
    transcriptml cv prepare-fold \
      --dataset data/mpra \
      --base-config configs/legnet/train_config.json \
      --cv-root runs/mpra_cv10 \
      --fold "${fold}" \
      --model legnet \
      --n-folds 10 \
      --seed 42 \
      --val-offset 1
  )

  transcriptml train "${config}"

  mkdir -p "runs/mpra_cv10/fold${fold}/eval"
  transcriptml evaluate \
    --checkpoint "runs/mpra_cv10/fold${fold}/model/best.pt" \
    --dataset "runs/mpra_cv10/fold${fold}/dataset" \
    --out-csv "runs/mpra_cv10/fold${fold}/eval/test_predictions.csv" \
    --split test \
    --device auto
done
```

This writes the same fold structure as the Saluki CV workflow, but with
four-channel MPRA input and a LegNet model.

### 3. Interpret the MPRA Model

Single-nucleotide ISM uses the same command shape as the Saluki workflow. Run it
once per fold checkpoint:

```bash
for fold in $(seq 0 9); do
  transcriptml ism \
    --checkpoint "runs/mpra_cv10/fold${fold}/model/best.pt" \
    --dataset data/mpra \
    --out-dir "interpret/mpra_ism/fold${fold}" \
    --device auto \
    --mutation-batch-size 512
done
```

Then average and mean-center the fold-level arrays:

```bash
transcriptml summarize-ism \
  --input-dir interpret/mpra_ism \
  --out-dir interpret/mpra_ism_summary \
  --dataset data/mpra \
  --write-fold-std \
  --write-projected
```

Plot one insert from the averaged, mean-centered ISM:

```bash
transcriptml plot-ism \
  --ism interpret/mpra_ism_summary/average_mean_centered_deltas.npy \
  --dataset data/mpra \
  --seq-index 0 \
  --out interpret/mpra_ism_summary/insert0.png
```

Motif ablation, motif context, and motif epistasis also work with four-channel
MPRA bundles. Region filters such as `--region 3utr` require Saluki annotation
channels, so omit them for ordinary MPRA input:

```bash
transcriptml motif-ablation \
  --checkpoint runs/mpra_cv10/fold0/model/best.pt \
  --dataset data/mpra \
  --out-dir interpret/mpra_are_ablation/fold0 \
  --motif 'UUAUUUAUU' \
  --n-scrambles 10 \
  --device auto
```

Interpret MPRA results in the assay's exact reporter context. Single-base
effects may reflect cryptic splice sites, unintended promoter activity, or
other construct-specific behavior in addition to the intended RNA regulatory
mechanism.

### HPC-Optimized MPRA Workflow

The `scripts/mpra/` directory contains Sherlock-oriented jobs for MPRA bundle
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
CV_MODEL=legnet
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
run when tuning is needed. As with the Saluki scripts, adjust checked-in
Sherlock partitions, GPU constraints, modules, resources, and filesystem paths
for your cluster.

See the [MPRA script reference](https://github.com/kundajelab/TranscriptML/blob/main/scripts/mpra/README.md)
for all settings and outputs.

## Other Useful Commands

List available registered models:

```bash
transcriptml models list
```

Inspect default parameters for one model:

```bash
transcriptml models show saluki_exact --json
```

Create a starter config directory:

```bash
transcriptml init-run --workflow saluki --out-dir configs/saluki
transcriptml init-run --workflow legnet --out-dir configs/legnet
```

These commands are intentionally small. They are meant to make the first run
easier, not to replace project-specific judgment about splits, targets, and
biological grouping.
