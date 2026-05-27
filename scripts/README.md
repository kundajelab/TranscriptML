# Sherlock Scripts

These scripts are intentionally Sherlock-specific and deliberately small. For a new run, keep the cloned TranscriptML repo clean, copy the full `scripts/` directory to a run/work directory, edit the copied config files there, and submit jobs from the copied scripts.

## Files

- `sherlock_config.sh`: per-run shell config after you copy `scripts/`; set paths, conda environment, fold count, motifs, and runtime knobs in the copied file.
- `example_train_config.json`: per-run training config after you copy `scripts/`; edit model and training hyperparameters in the copied file. The CV script always replaces `dataset` and `output_dir` in each generated fold config.
- `build_saluki_gtf.sh`: builds a Saluki-style dataset bundle with `transcriptml build-saluki-gtf`.
- `train_eval_cv_fold.sh` and `submit_train_eval_cv.sh`: 10-fold CV as a SLURM job array, one job per fold.
- `ism_by_fold.sh` and `submit_ism_by_fold.sh`: single-nucleotide ISM, one job per trained fold.
- `codon_ism_by_fold.sh` and `submit_codon_ism_by_fold.sh`: synonymous codon ISM, one job per trained fold.
- `motif_ablation_by_fold.sh` and `motif_ablation_all_folds.sh`: motif ablations across the configured motif list.
- `motif_epistasis_by_fold.sh` and `motif_epistasis_all_folds.sh`: motif epistasis across the configured motif-pair list.

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
| `DATASET_DIR`, `CV_ROOT`, `INTERPRET_ROOT` | Usually leave these derived from `RUN_ROOT`. Change them only if you want outputs split across custom locations. |
| `N_FOLDS`, `CV_SEED`, `EVAL_SPLIT` | Change these if you do not want the default 10-fold CV behavior. |
| `PRED_BATCH_SIZE`, `MUTATION_BATCH_SIZE`, `DEVICE` | Runtime controls for GPU/CPU and prediction/ISM batch sizes. |
| `MOTIF_ABLATION_SPECS`, `MOTIF_EPISTASIS_SPECS` | Edit only when running motif ablation or motif epistasis with a custom motif list. |

You normally do not need to edit:

- `_TRANSCRIPTML_SCRIPT_DIR`, `SCRIPT_CONFIG_DIR`, or `_TRANSCRIPTML_REPO_CANDIDATE`: internal path discovery for the copied scripts.
- `TRANSCRIPTML_RUN_CONFIG`: optional advanced override file. Leave it unset for ordinary runs.
- `parse_motif_ablation_spec`, `parse_motif_epistasis_spec`, or `setup_transcriptml_env`: helper functions used by the job scripts.

Then edit the copied `scripts/example_train_config.json` for model and training hyperparameters. For example, a smaller fast pass could use:

```json
{
  "dataset": "CV_SCRIPT_OVERWRITES_THIS_FROM_DATASET_DIR",
  "output_dir": "CV_SCRIPT_OVERWRITES_THIS_PER_FOLD",
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

For the CV workflow, leave `dataset` and `output_dir` as placeholders in `scripts/example_train_config.json`. `train_eval_cv_fold.sh` calls `write_cv_fold_artifacts.py`, which reads the copied `scripts/example_train_config.json`, sets `dataset` to `${CV_ROOT}/foldN/dataset`, sets `output_dir` to `${CV_ROOT}/foldN/model`, and writes `${CV_ROOT}/foldN/train_config.json`. If you edit those two keys in the copied base config, the CV scripts still overwrite them in the generated fold configs. Edit them only when running `transcriptml train` directly outside this CV workflow.

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

The default motif list in `sherlock_config.sh` includes PRE (`UGUA[A|U|C]AUA`), ARE-nonamer, GGACU, a let-7 7mer-m8 target site, and a miR-16 7mer-m8 target site.

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
