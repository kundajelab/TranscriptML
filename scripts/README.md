# Sherlock Scripts

These scripts are intentionally Sherlock-specific and deliberately small. For a new run, edit `sherlock_config.sh`, optionally edit `example_train_config.json`, then submit the workflow pieces you need.

## Files

- `sherlock_config.sh`: one place to set paths, conda environment, fold count, motifs, and runtime knobs.
- `example_train_config.json`: base TranscriptML training config used by the CV scripts. The CV script replaces `dataset` and `output_dir` for each fold.
- `build_saluki_gtf.sh`: builds a Saluki-style dataset bundle with `transcriptml build-saluki-gtf`.
- `train_eval_cv_fold.sh` and `submit_train_eval_cv.sh`: 10-fold CV as a SLURM job array, one job per fold.
- `ism_by_fold.sh` and `submit_ism_by_fold.sh`: single-nucleotide ISM, one job per trained fold.
- `codon_ism_by_fold.sh` and `submit_codon_ism_by_fold.sh`: synonymous codon ISM, one job per trained fold.
- `motif_ablation_by_fold.sh` and `motif_ablation_all_folds.sh`: motif ablations across the configured motif list.
- `motif_epistasis_by_fold.sh` and `motif_epistasis_all_folds.sh`: motif epistasis across the configured motif-pair list.

## Configure A Run

Edit `scripts/sherlock_config.sh`:

```bash
TRANSCRIPTML_REPO="/home/users/isvock/TranscriptML"
CONDA_ENV="transcript-ml"
SHERLOCK_CONDA_ROOT="$GROUP_HOME/miniconda"

GTF="/oak/stanford/groups/akundaje/refs/gencode.v44.annotation.gtf"
FASTA="/oak/stanford/groups/akundaje/refs/GRCh38.primary_assembly.genome.fa"
TARGETS="/scratch/users/isvock/rna_decay/targets.csv"
TARGET_ID_COL="transcript_id"
TARGET_COL="log_kdeg"
SPLIT_COL=""

RUN_NAME="human_kdeg_saluki_exact"
RUN_ROOT="/scratch/users/isvock/TranscriptML/${RUN_NAME}"
DATASET_DIR="${RUN_ROOT}/data/saluki"
CV_ROOT="${RUN_ROOT}/cv10"
INTERPRET_ROOT="${RUN_ROOT}/interpret"
DEVICE="cuda"
```

Then edit `scripts/example_train_config.json` for model and training hyperparameters. For example, to train a smaller fast pass:

```json
{
  "dataset": "REPLACED_BY_CV_SCRIPT",
  "output_dir": "REPLACED_BY_CV_SCRIPT",
  "model": {"name": "saluki_exact", "params": {"seq_depth": 6, "filters": 32}},
  "batch_size": 64,
  "epochs": 10,
  "learning_rate": 0.001,
  "patience": 3,
  "monitor": "val_loss",
  "device": "auto",
  "mmap_mode": "r",
  "seed": 42
}
```

## Build The Dataset

Submit the data processing job:

```bash
cd /home/users/isvock/TranscriptML
sbatch scripts/build_saluki_gtf.sh
```

This writes the bundle under `${DATASET_DIR}`, including `X.npy`, `y.npy`, `ids.txt`, `schema.json`, and sidecar metadata.

## Train And Evaluate 10-Fold CV

Submit one training/evaluation job per fold:

```bash
bash scripts/submit_train_eval_cv.sh
```

Each fold writes:

```text
${CV_ROOT}/fold0/dataset/splits.json
${CV_ROOT}/fold0/train_config.json
${CV_ROOT}/fold0/model/best.pt
${CV_ROOT}/fold0/eval/test_predictions.csv
${CV_ROOT}/fold0/eval/test_predictions.summary.json
```

The fold split is deterministic from `CV_SEED`. For fold `i`, fold `i` is the test split, fold `i + 1` is validation, and the remaining folds are training.

## Run ISM

After CV finishes:

```bash
bash scripts/submit_ism_by_fold.sh
```

Outputs go to:

```text
${INTERPRET_ROOT}/ism/fold0/
${INTERPRET_ROOT}/ism/fold1/
...
```

## Run Synonymous Codon ISM

```bash
bash scripts/submit_codon_ism_by_fold.sh
```

The script uses:

```bash
--mutation-policy synonymous-only
--exclude-stop-codons
--table-format npz
```

Outputs go to `${INTERPRET_ROOT}/codon_ism/fold*/`.

## Run Motif Ablations

The default motif list in `sherlock_config.sh` includes PRE, ARE-nonamer, GGACU, a let-7 7mer-m8 target site, and a miR-16 7mer-m8 target site.

To run one SLURM job per fold:

```bash
bash scripts/submit_motif_ablation_by_fold.sh
```

To run all folds in one job:

```bash
sbatch scripts/motif_ablation_all_folds.sh
```

Outputs go to `${INTERPRET_ROOT}/motif_ablation/<motif_label>/fold*/`.

## Run Motif Epistasis

The default motif-pair list includes same-motif pairs and PRE/ARE/GGACU cross-pairs, plus let-7/miR-16.

To run one SLURM job per fold:

```bash
bash scripts/submit_motif_epistasis_by_fold.sh
```

To run all folds in one job:

```bash
sbatch scripts/motif_epistasis_all_folds.sh
```

Outputs go to `${INTERPRET_ROOT}/motif_epistasis/<pair_label>/fold*/`.

## Common Tweaks

Change the ablation motif list in `sherlock_config.sh`:

```bash
MOTIF_ABLATION_SPECS=(
  "PRE|UGUANAUA"
  "ARE_nonamer|UUAUUUAUU"
  "GGACU|GGACU"
  "let7_7mer_m8|CUACCUC"
  "miR16_7mer_m8|UGCUGCU"
)
```

Change epistasis pairs:

```bash
MOTIF_EPISTASIS_SPECS=(
  "PRE_ARE|UGUANAUA|UUAUUUAUU"
  "PRE_PRE|UGUANAUA|"
)
```

The motif syntax supports `A/C/G/U/T`, `N` wildcards, and bracket alternatives like `UGUA[A|U|C]AUA`.
