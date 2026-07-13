# Sherlock Scripts

These scripts are intentionally Sherlock-specific and deliberately small. For a new run, keep the cloned TranscriptML repo clean, copy the full `scripts/` directory to a run/work directory, edit the copied config files there, and submit jobs from the copied scripts.

## Files

- `sherlock_config.sh`: per-run shell config after you copy `scripts/`; set paths, conda environment, fold count, motifs, and runtime knobs in the copied file.
- `example_train_config.json`: per-run training config after you copy `scripts/`; edit model and training hyperparameters in the copied file. The training scripts replace `dataset` and `output_dir` in generated configs.
- `build_saluki_gtf.sh`: builds a Saluki-style dataset bundle with `transcriptml build-saluki-gtf`.
- `train_eval_split.sh` and `submit_train_eval_split.sh`: train and evaluate one predefined train/validation/test split from the dataset bundle.
- `train_eval_cv_fold.sh` and `submit_train_eval_cv.sh`: 10-fold CV as a SLURM job array, one job per fold.
- `make_saluki_hparam_grid.py`, `submit_hparam_sweep_cv.sh`, and `hparam_sweep_cv_combo.sh`: generate or consume a hyperparameter table and run one CV sweep combo per SLURM array task.
- `summarize_hparam_sweep.py`: aggregate completed sweep combo/fold summaries into `${SWEEP_ROOT}/sweep_summary.tsv`.
- `ism_by_fold.sh` and `submit_ism_by_fold.sh`: single-nucleotide ISM, one job per trained fold.
- `codon_ism_by_fold.sh` and `submit_codon_ism_by_fold.sh`: synonymous codon ISM, one job per trained fold.
- `all_codon_ism_shard_by_fold.sh` and `submit_all_codon_ism_shard_by_fold.sh`: all-codon ISM, one 10-task job array per fold by default.
- `motif_ablation_by_fold.sh` and `motif_ablation_all_folds.sh`: motif ablations across the configured motif list.
- `motif_epistasis_by_fold.sh` and `motif_epistasis_all_folds.sh`: motif epistasis across the configured motif-pair list.
- `mpra/`: MPRA 3-prime UTR insert workflows for building 4-channel LegNet input, training LegNet, and running single-nucleotide ISM. See `mpra/README.md`.

## Configure A Run

All job scripts source `scripts/sherlock_config.sh` before doing work. For a
normal run, edit the copied config values and ignore the internal helper code.
The important knobs are paths, your conda environment, and a few run settings.

From a clean TranscriptML checkout, copy the full scripts directory to a run/work directory:

```bash
TRANSCRIPTML_REPO="/home/users/isvock/TranscriptML"
RUN_WORKDIR="/scratch/users/isvock/transcriptml_runs/human_kdeg_saluki_exact"

mkdir -p "${RUN_WORKDIR}"
cp -R "${TRANSCRIPTML_REPO}/scripts" "${RUN_WORKDIR}/scripts"
cd "${RUN_WORKDIR}"
```

### What To Edit In `sherlock_config.sh`

Edit the copied `scripts/sherlock_config.sh`, not the one in the clean repo.
For the usual "clean repo in home/OAK, copied scripts in scratch" setup, set
these values:

```bash
# /scratch/users/isvock/transcriptml_runs/human_kdeg_saluki_exact/scripts/sherlock_config.sh

# 1. TranscriptML checkout and conda environment.
# Set TRANSCRIPTML_REPO because these copied scripts live outside the repo.
TRANSCRIPTML_REPO="/home/users/isvock/TranscriptML"
CONDA_ENV="transcript-ml"
SHERLOCK_CONDA_ROOT="${GROUP_HOME:-${HOME}}/miniconda"

# 2. Input files and column names for build_saluki_gtf.sh.
GTF="/oak/stanford/groups/akundaje/refs/gencode.v44.annotation.gtf"
FASTA="/oak/stanford/groups/akundaje/refs/GRCh38.primary_assembly.genome.fa"
TARGETS="/scratch/users/isvock/rna_decay/targets.csv"
TARGET_ID_COL="transcript_id"
TARGET_COL="log_kdeg"
SPLIT_COL=""

# 3. Output directories for this run.
RUN_NAME="human_kdeg_saluki_exact"
RUN_ROOT="/scratch/users/isvock/TranscriptML/${RUN_NAME}"
DATASET_DIR="${RUN_ROOT}/data/saluki"
TRAIN_OUTPUT_ROOT="${RUN_ROOT}/train_eval"
CV_ROOT="${RUN_ROOT}/cv10"
INTERPRET_ROOT="${RUN_ROOT}/interpret"

# 4. Runtime choices.
N_FOLDS="10"
CV_SEED="42"
DEVICE="cuda"
```

What each group means:

| Variable(s) | When to change |
| --- | --- |
| `TRANSCRIPTML_REPO` | Set this to the clean TranscriptML checkout when the copied `scripts/` directory is outside the repo. This is the normal scratch-run case. If `transcriptml` is already installed in `CONDA_ENV`, you can leave it empty. |
| `CONDA_ENV`, `SHERLOCK_CONDA_ROOT` | Set these to the conda environment and conda install used on Sherlock. The job setup loads `gcc/10.1.0` and `openblas/0.3.10` before activating conda. |
| `GTF`, `FASTA`, `TARGETS` | Set these to your annotation GTF, genome FASTA, and target table. |
| `TARGET_ID_COL`, `TARGET_COL`, `SPLIT_COL`, `METADATA_COLS` | Match these to columns in `TARGETS`. Leave `SPLIT_COL=""` if your target table does not already define train/val/test splits. |
| `RUN_NAME`, `RUN_ROOT` | Pick a run name and scratch/OAK location where outputs should be written. |
| `DATASET_DIR`, `TRAIN_OUTPUT_ROOT`, `CV_ROOT`, `INTERPRET_ROOT` | Usually leave these derived from `RUN_ROOT`. Change them only if you want outputs split across custom locations. |
| `SWEEP_TABLE`, `SWEEP_ROOT`, `SWEEP_MAX_CONCURRENT`, `SWEEP_SKIP_COMPLETED` | Hyperparameter sweep controls. By default the sweep table is `scripts/saluki_hparams.tsv`, outputs go under `${RUN_ROOT}/hparam_sweep`, and completed folds are skipped on rerun. |
| `N_FOLDS`, `CV_SEED`, `CV_VAL_OFFSET`, `EVAL_SPLIT` | Change these if you do not want the default 10-fold CV behavior. |
| `MODEL_DIR`, `EVAL_DIR`, `GENERATED_TRAIN_CONFIG`, `TRAIN_SEED`, `REQUIRE_SPLIT_FILE` | Optional controls for `train_eval_split.sh`. By default it writes under `${TRAIN_OUTPUT_ROOT}` and requires `${DATASET_DIR}/splits.json`. |
| `PRED_BATCH_SIZE`, `MUTATION_BATCH_SIZE`, `DEVICE` | Runtime controls for GPU/CPU and prediction/ISM batch sizes. |
| `MOTIF_REGION` | Region for motif ablation and epistasis jobs. Defaults to `3utr`; use `5utr`, `cds`, `3utr`, or leave empty for whole-transcript analyses. |
| `MOTIF_ABLATION_SPECS`, `MOTIF_EPISTASIS_SPECS` | Edit only when running motif ablation or motif epistasis with a custom motif list. |

You normally do not need to edit:

- `_TRANSCRIPTML_SCRIPT_DIR`, `SCRIPT_CONFIG_DIR`, or `_TRANSCRIPTML_REPO_CANDIDATE`: internal path discovery for the copied scripts.
- `TRANSCRIPTML_RUN_CONFIG`: optional advanced override file. Leave it unset for ordinary runs.
- `parse_motif_ablation_spec`, `parse_motif_epistasis_spec`, or `setup_transcriptml_env`: helper functions used by the job scripts.

Then edit the copied `scripts/example_train_config.json` for model and training hyperparameters. For example, a smaller fast pass could use:

```json
{
  "dataset": "SCRIPT_OVERWRITES_THIS_FROM_DATASET_DIR",
  "output_dir": "SCRIPT_OVERWRITES_THIS_OUTPUT_DIR",
  "model": {"name": "saluki_exact", "params": {"seq_depth": 6, "filters": 32}},
  "batch_size": 64,
  "epochs": 10,
  "learning_rate": 0.001,
  "gradient_clip_norm": 0.5,
  "patience": 3,
  "monitor": ["val_loss", "val_pearson"],
  "device": "auto",
  "mmap_mode": "r",
  "seed": 42
}
```

For the script workflows, leave `dataset` and `output_dir` as placeholders in `scripts/example_train_config.json`. `train_eval_split.sh` and `train_eval_cv_fold.sh` both generate a per-run train config and overwrite those two fields. If you edit those two keys in the copied base config, the script-generated configs still replace them. Edit them only when running `transcriptml train` directly outside these workflows.

Training early stopping can monitor one metric or a list of metrics. With
`"monitor": ["val_loss", "val_pearson"]`, an epoch counts as improved if the
validation loss decreases or the validation Pearson correlation increases.

### Alternative Loss Functions

The training scripts preserve any `loss` block you add to the copied
`scripts/example_train_config.json`. `dataset` and `output_dir` are still
overwritten by `train_eval_split.sh` and `train_eval_cv_fold.sh`; the loss
configuration is not overwritten.

TranscriptML uses ordinary unweighted MSE when the config has no `loss` block.
To train with weighted MSE, first make sure the target table column named by
`TARGET_COL` is the scalar target you want to predict, usually `log_kdeg`.
Then make sure the weight or standard-error column is written to bundle
metadata:

```bash
# scripts/sherlock_config.sh
TARGET_COL="log_kdeg"

# Leave METADATA_COLS empty to keep all non-target target-table columns, or list
# the exact columns you need.
METADATA_COLS="log_kdeg_se,split"
```

If your target table already contains the final per-transcript weights, add a
`weight_col` loss block to `scripts/example_train_config.json`:

```json
{
  "loss": {
    "name": "weighted_mse",
    "weight_col": "log_kdeg_weight",
    "min_weight": 0.01,
    "max_weight": 100.0
  }
}
```

If your target table contains a standard error for `log_kdeg`, use `se_col`
instead. TranscriptML derives weights as `1 / (se^2 + eps)` and clips them to
avoid extreme influence from tiny or huge uncertainty estimates:

```json
{
  "loss": {
    "name": "weighted_mse",
    "se_col": "log_kdeg_se",
    "eps": 1e-8,
    "min_weight": 0.01,
    "max_weight": 100.0
  }
}
```

To train with the binomial count likelihood, the target table must contain one
row per transcript with total reads, new reads, and pulse duration in hours.
The model output is interpreted as natural-log `kdeg`, and the likelihood uses
`new_reads ~ Binomial(total_reads, 1 - exp(-kdeg * pulse_hours))`.

```bash
# scripts/sherlock_config.sh
TARGET_COL="log_kdeg"
METADATA_COLS="total_reads,new_reads,pulse_hours,split"
```

Keeping `TARGET_COL="log_kdeg"` is recommended because it writes `y.npy` for
Pearson/MSE reporting, even though `binomial_nll` trains from the count columns.
Then add this to `scripts/example_train_config.json`:

```json
{
  "loss": {
    "name": "binomial_nll",
    "total_reads_col": "total_reads",
    "new_reads_col": "new_reads",
    "pulse_hours_col": "pulse_hours"
  }
}
```

After changing `METADATA_COLS` or any target-table columns, rerun the dataset
build job before submitting training jobs:

```bash
sbatch scripts/build_saluki_gtf.sh
bash scripts/submit_train_eval_split.sh
# or
bash scripts/submit_train_eval_cv.sh
```

## Build The Dataset

Submit the data processing job:

```bash
cd /scratch/users/isvock/transcriptml_runs/human_kdeg_saluki_exact
sbatch scripts/build_saluki_gtf.sh
```

Submit `sbatch` commands from the run directory that contains the copied
`scripts/` directory. The job scripts use SLURM's submit directory to find the
matching copied `scripts/sherlock_config.sh`.

This writes the bundle under `${DATASET_DIR}`, including `X.npy`, `y.npy`, `ids.txt`, `schema.json`, and sidecar metadata.

If you want to train from a target-table split column, set `SPLIT_COL` in the
copied `scripts/sherlock_config.sh` before building the dataset. The split
column values should include `train`, `val` or `validation`, and `test`. The
builder writes those row assignments to `${DATASET_DIR}/splits.json`, and
`transcriptml train` uses that file automatically.

## Train And Evaluate One Predefined Split

Use this workflow when the dataset bundle already has a `splits.json`, for
example because you built it from a target table with `SPLIT_COL="split"`.

```bash
cd /scratch/users/isvock/transcriptml_runs/human_kdeg_saluki_exact
bash scripts/submit_train_eval_split.sh
```

This submits one SLURM job. It writes:

```text
${TRAIN_OUTPUT_ROOT}/train_config.json
${MODEL_DIR}/best.pt
${EVAL_DIR}/test_predictions.csv
${EVAL_DIR}/test_predictions.summary.json
```

For a manual 4-fold CV workaround, make four run directories, copy `scripts/`
into each, and give each directory a different target table whose `split`
column contains that fold's train/val/test labels. In each copied
`scripts/sherlock_config.sh`, set `TARGETS` to that fold's target table,
`SPLIT_COL` to the split column name, and choose a fold-specific `RUN_ROOT`.
Then run the build job and `submit_train_eval_split.sh` in each directory.

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

## Hyperparameter Sweep CV

Use this when you want one SLURM job per hyperparameter combo, with all CV
folds run sequentially inside that job.

Generate a starter table, or provide your own CSV/TSV with one row per combo:

```bash
python scripts/make_saluki_hparam_grid.py --preset smoke --out scripts/saluki_hparams.tsv
```

Sweep-table columns can be top-level training config fields such as
`learning_rate`, `batch_size`, `epochs`, `patience`, or `weight_decay`; Saluki
model params such as `filters`, `dropout`, and `kernel_size`; or dotted config
paths such as `model.params.filters`. Values are parsed as JSON when possible,
so arrays and booleans can be written as `[64,96,128]` and `true`.

Submit the sweep:

```bash
bash scripts/submit_hparam_sweep_cv.sh
```

Each SLURM array task writes one combo under `${SWEEP_ROOT}`:

```text
${SWEEP_ROOT}/sweep_table.tsv
${SWEEP_ROOT}/combo_0000/hparams.json
${SWEEP_ROOT}/combo_0000/fold0/train_config.json
${SWEEP_ROOT}/combo_0000/fold0/model/summary.json
${SWEEP_ROOT}/combo_0000/fold0/eval/test_predictions.summary.json
${SWEEP_ROOT}/combo_0000/combo_summary.json
${SWEEP_ROOT}/sweep_summary.tsv
```

To refresh the aggregate table after manual edits or partial reruns:

```bash
python scripts/summarize_hparam_sweep.py --sweep-root "${SWEEP_ROOT}"
```

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

## Run Sharded All-Codon ISM

```bash
bash scripts/submit_all_codon_ism_shard_by_fold.sh
```

This submits one SLURM job array per fold. Each fold array has
`${CODON_ISM_SHARDS_PER_FOLD:-10}` tasks, and task `i` runs all-codon ISM on a
contiguous transcript shard using:

```bash
--mutation-policy all
--table-format parquet
--sequence-shard-index "${SLURM_ARRAY_TASK_ID}"
--sequence-shards "${CODON_ISM_SHARDS_PER_FOLD:-10}"
```

Outputs go to `${INTERPRET_ROOT}/all_codon_ism/fold*/shard*/`.

## Run Motif Ablations

The default motif list in `sherlock_config.sh` includes PRE (`UGUA[A|U|C]AUA`), ARE-nonamer, GGACU, a let-7 7mer-m8 target site, and a miR-16 7mer-m8 target site. Sherlock motif jobs default to `MOTIF_REGION="3utr"`, so only motif instances fully inside the 3-prime UTR are analyzed. Set `MOTIF_REGION=""`, `5utr`, or `cds` in the copied `sherlock_config.sh` to change this.

To run one SLURM job per fold:

```bash
bash scripts/submit_motif_ablation_by_fold.sh
```

To run all folds in one job:

```bash
sbatch scripts/motif_ablation_all_folds.sh
```

Outputs go to `${INTERPRET_ROOT}/motif_ablation/<motif_label>/fold*/`.

To make one box-and-points summary plot per fold after motif ablations finish,
run this in an environment with NumPy and Matplotlib:

```bash
python scripts/plot_motif_ablation_effects.py \
  "${INTERPRET_ROOT}/motif_ablation" \
  --out-dir "${INTERPRET_ROOT}/motif_ablation/plots" \
  --ylim -0.45 0.45
```

The plotting script reads each
`${INTERPRET_ROOT}/motif_ablation/<motif_label>/fold*/effects.npy` file and
writes one figure per fold.

## Run Motif Epistasis

The default motif-pair list includes same-motif pairs and PRE/ARE/GGACU cross-pairs, plus let-7/miR-16. Like motif ablations, Sherlock epistasis jobs use `MOTIF_REGION="3utr"` by default.

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

Change the ablation motif list in the copied `scripts/sherlock_config.sh`:

```bash
MOTIF_ABLATION_SPECS=(
  "PRE|UGUA[A|U|C]AUA"
  "ARE_nonamer|UUAUUUAUU"
  "GGACU|GGACU"
  "let7_7mer_m8|CUACCUC"
  "miR16_7mer_m8|UGCUGCU"
)
```

Change epistasis pairs in the copied `scripts/sherlock_config.sh`:

```bash
MOTIF_EPISTASIS_SPECS=(
  "PRE_ARE|UGUA[A|U|C]AUA|UUAUUUAUU"
  "PRE_PRE|UGUA[A|U|C]AUA|"
)
```

The motif syntax supports `A/C/G/U/T`, `N` wildcards, and bracket alternatives like `UGUA[A|U|C]AUA`. In motif specs, top-level `|` characters separate fields; `|` characters inside bracket alternatives stay part of the motif.
