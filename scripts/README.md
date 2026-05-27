# Sherlock Scripts

These scripts are intentionally Sherlock-specific and deliberately small. For a new run, keep this cloned repo clean, put run-specific settings in a separate config file, and pass that file with `TRANSCRIPTML_RUN_CONFIG`.

## Files

- `sherlock_config.sh`: shared defaults plus the `TRANSCRIPTML_RUN_CONFIG` loader. Edit this only when changing the workflow defaults.
- `example_train_config.json`: base TranscriptML training config template. Copy it into a run directory for per-run hyperparameters. The CV script always replaces `dataset` and `output_dir` in each generated fold config.
- `build_saluki_gtf.sh`: builds a Saluki-style dataset bundle with `transcriptml build-saluki-gtf`.
- `train_eval_cv_fold.sh` and `submit_train_eval_cv.sh`: 10-fold CV as a SLURM job array, one job per fold.
- `ism_by_fold.sh` and `submit_ism_by_fold.sh`: single-nucleotide ISM, one job per trained fold.
- `codon_ism_by_fold.sh` and `submit_codon_ism_by_fold.sh`: synonymous codon ISM, one job per trained fold.
- `motif_ablation_by_fold.sh` and `motif_ablation_all_folds.sh`: motif ablations across the configured motif list.
- `motif_epistasis_by_fold.sh` and `motif_epistasis_all_folds.sh`: motif epistasis across the configured motif-pair list.

## Configure A Run

Create a run directory outside the repo, copy the train-config template there if you want run-specific model or training hyperparameters, and create a run config:

```bash
RUN_CONFIG="/scratch/users/isvock/transcriptml_runs/human_kdeg_saluki_exact/config.sh"
mkdir -p "$(dirname "${RUN_CONFIG}")"
cp scripts/example_train_config.json "$(dirname "${RUN_CONFIG}")/train_config.json"
```

Edit the copied `train_config.json` for model and training hyperparameters. Leave `dataset` and `output_dir` as placeholders for CV runs; they are rewritten per fold.

Then create `config.sh`:

```bash
# /scratch/users/isvock/transcriptml_runs/human_kdeg_saluki_exact/config.sh
CONDA_ENV="transcript-ml"
SHERLOCK_CONDA_ROOT="${GROUP_HOME:-${HOME}}/miniconda"

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

BASE_TRAIN_CONFIG="${TRANSCRIPTML_RUN_CONFIG_DIR}/train_config.json"
```

`TRANSCRIPTML_RUN_CONFIG_DIR` is set by `sherlock_config.sh` before it sources your run config, so paths can be relative to the run config file. You usually do not need to set `TRANSCRIPTML_REPO`; it defaults to the parent of `scripts/`. Set it only if these scripts are copied/symlinked somewhere unusual or if a run should use a different TranscriptML checkout.

For example, the copied `train_config.json` could be a smaller fast pass:

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

For the CV workflow, `train_eval_cv_fold.sh` calls `write_cv_fold_artifacts.py`, which reads `BASE_TRAIN_CONFIG`, sets `dataset` to `${CV_ROOT}/foldN/dataset`, sets `output_dir` to `${CV_ROOT}/foldN/model`, and writes `${CV_ROOT}/foldN/train_config.json`. If you edit those two keys in the base config, the CV scripts still overwrite them in the generated fold configs. Edit them only when running `transcriptml train` directly outside this CV workflow.

Pass the run config when launching any workflow piece:

```bash
TRANSCRIPTML_RUN_CONFIG="${RUN_CONFIG}" bash scripts/submit_train_eval_cv.sh
```

Variables inside `config.sh` can be plain shell assignments; they do not need `export` because each script sources the file itself.

## Build The Dataset

Submit the data processing job:

```bash
cd /home/users/isvock/TranscriptML
TRANSCRIPTML_RUN_CONFIG="${RUN_CONFIG}" sbatch scripts/build_saluki_gtf.sh
```

This writes the bundle under `${DATASET_DIR}`, including `X.npy`, `y.npy`, `ids.txt`, `schema.json`, and sidecar metadata.

## Train And Evaluate 10-Fold CV

Submit one training/evaluation job per fold:

```bash
TRANSCRIPTML_RUN_CONFIG="${RUN_CONFIG}" bash scripts/submit_train_eval_cv.sh
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
TRANSCRIPTML_RUN_CONFIG="${RUN_CONFIG}" bash scripts/submit_ism_by_fold.sh
```

Outputs go to:

```text
${INTERPRET_ROOT}/ism/fold0/
${INTERPRET_ROOT}/ism/fold1/
...
```

## Run Synonymous Codon ISM

```bash
TRANSCRIPTML_RUN_CONFIG="${RUN_CONFIG}" bash scripts/submit_codon_ism_by_fold.sh
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
TRANSCRIPTML_RUN_CONFIG="${RUN_CONFIG}" bash scripts/submit_motif_ablation_by_fold.sh
```

To run all folds in one job:

```bash
TRANSCRIPTML_RUN_CONFIG="${RUN_CONFIG}" sbatch scripts/motif_ablation_all_folds.sh
```

Outputs go to `${INTERPRET_ROOT}/motif_ablation/<motif_label>/fold*/`.

## Run Motif Epistasis

The default motif-pair list includes same-motif pairs and PRE/ARE/GGACU cross-pairs, plus let-7/miR-16.

To run one SLURM job per fold:

```bash
TRANSCRIPTML_RUN_CONFIG="${RUN_CONFIG}" bash scripts/submit_motif_epistasis_by_fold.sh
```

To run all folds in one job:

```bash
TRANSCRIPTML_RUN_CONFIG="${RUN_CONFIG}" sbatch scripts/motif_epistasis_all_folds.sh
```

Outputs go to `${INTERPRET_ROOT}/motif_epistasis/<pair_label>/fold*/`.

## Common Tweaks

Override the ablation motif list in your run config:

```bash
MOTIF_ABLATION_SPECS=(
  "PRE|UGUA[A|U|C]AUA"
  "ARE_nonamer|UUAUUUAUU"
  "GGACU|GGACU"
  "let7_7mer_m8|CUACCUC"
  "miR16_7mer_m8|UGCUGCU"
)
```

Override epistasis pairs in your run config:

```bash
MOTIF_EPISTASIS_SPECS=(
  "PRE_ARE|UGUA[A|U|C]AUA|UUAUUUAUU"
  "PRE_PRE|UGUA[A|U|C]AUA|"
)
```

The motif syntax supports `A/C/G/U/T`, `N` wildcards, and bracket alternatives like `UGUA[A|U|C]AUA`. In motif specs, top-level `|` characters separate fields; `|` characters inside bracket alternatives stay part of the motif.
