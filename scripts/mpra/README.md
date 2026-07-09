# MPRA LegNet Scripts

These scripts are intentionally small and Sherlock-oriented, following the
same pattern as the transcriptome-derived training scripts in `scripts/`.
For a new run, keep the cloned TranscriptML repo clean, copy `scripts/mpra/`
to a run/work directory, edit the copied config files there, and submit jobs
from the copied scripts.

## Files

- `mpra_config.sh`: per-run shell config after you copy `scripts/mpra/`; set paths, conda environment, MPRA table columns, and runtime knobs in the copied file.
- `example_legnet_train_config.json`: per-run training config after you copy `scripts/mpra/`; edit LegNet and training hyperparameters in the copied file. The training scripts replace `dataset` and `output_dir` in generated configs.
- `build_legnet_input.sh`: builds a 4-channel RNA MPRA dataset bundle with `transcriptml build-mpra`.
- `train_eval_cv_fold.sh` and `submit_train_eval_cv.sh`: 10-fold CV as a SLURM job array, one LegNet training/evaluation job per fold.
- `ism_by_fold.sh` and `submit_ism_by_fold.sh`: single-nucleotide ISM, one job per trained fold.
- `train_legnet.sh` and `submit_train_legnet.sh`: optional helper to train one LegNet model from the MPRA bundle.
- `run_legnet_ism.sh` and `submit_legnet_ism.sh`: optional helper to run ISM on one trained checkpoint.
- `write_cv_fold_artifacts.py` and `write_legnet_train_config.py`: internal helpers used by the training scripts to generate run-specific bundles and train configs.

## Configure A Run

For a normal Sherlock run, copy this directory to scratch and edit the copied
files:

```bash
TRANSCRIPTML_REPO="/home/users/isvock/TranscriptML"
RUN_WORKDIR="/scratch/users/isvock/transcriptml_runs/mpra_legnet"

mkdir -p "${RUN_WORKDIR}/scripts"
cp -R "${TRANSCRIPTML_REPO}/scripts/mpra" "${RUN_WORKDIR}/scripts/mpra"
cd "${RUN_WORKDIR}"
```

Edit `scripts/mpra/mpra_config.sh`:

```bash
TRANSCRIPTML_REPO="/home/users/isvock/TranscriptML"
CONDA_ENV="transcript-ml"
SHERLOCK_CONDA_ROOT="${GROUP_HOME:-${HOME}}/miniconda"

MPRA_TABLE="/scratch/users/isvock/mpra/mpra_3utr_stability.csv"
SEQUENCE_COL="utr_insert"
TARGET_COL="rna_stability"
ID_COL="variant_id"
SPLIT_COL=""

RUN_NAME="mpra_3utr_legnet"
RUN_ROOT="/scratch/users/isvock/TranscriptML/${RUN_NAME}"
DATASET_DIR="${RUN_ROOT}/data/mpra"
CV_ROOT="${RUN_ROOT}/cv10"
INTERPRET_ROOT="${RUN_ROOT}/interpret"
N_FOLDS="10"
CV_SEED="42"
DEVICE="cuda"
```

Important config values:

| Variable(s) | When to change |
| --- | --- |
| `TRANSCRIPTML_REPO` | Set this to the clean TranscriptML checkout when the copied `scripts/mpra/` directory is outside the repo. |
| `CONDA_ENV`, `SHERLOCK_CONDA_ROOT` | Set these to the conda environment and conda install used on Sherlock. |
| `MPRA_TABLE` | Input CSV/TSV table with one row per MPRA construct. |
| `SEQUENCE_COL`, `TARGET_COL` | Column names for the 3-prime UTR insert sequence and scalar RNA stability target. |
| `ID_COL` | Optional stable construct identifier column. Row numbers are used when this is empty. |
| `SPLIT_COL` | Optional train/val/test split column. Leave empty to let the training config make a random split. |
| `METADATA_COLS` | Optional comma-separated metadata columns to keep. Leave empty to keep all non-sequence, non-target, and non-id columns. |
| `MPRA_LENGTH` | Optional fixed encoded length. Leave empty to use the longest input sequence length. Shorter sequences are padded; longer sequences are truncated from the 5-prime side. |
| `DELIMITER` | Optional delimiter override. When empty, `.tsv`/`.tab` use tabs and other files use commas. |
| `RUN_ROOT`, `DATASET_DIR`, `CV_ROOT`, `INTERPRET_ROOT` | Output locations for the dataset, fold model/evaluation files, and ISM files. |
| `BASE_TRAIN_CONFIG` | Base JSON training config. Defaults to `scripts/mpra/example_legnet_train_config.json`. |
| `N_FOLDS`, `CV_SEED`, `CV_VAL_OFFSET` | CV controls. By default, fold `i` is the test split and fold `i + 1` is validation. |
| `RUN_SPLIT_EVALUATION` | `auto` runs the extra `transcriptml evaluate --split` step only when the built dataset has `splits.json`. Set `1` to require it or `0` to skip it. |
| `PRED_BATCH_SIZE`, `MUTATION_BATCH_SIZE`, `DEVICE` | Runtime controls for prediction and ISM. |

Then edit `scripts/mpra/example_legnet_train_config.json` for model and
training hyperparameters. For first passes, the main values to tune are usually
`batch_size`, `epochs`, `learning_rate`, `patience`, and the LegNet channel
sizes in `model.params`.

## Expected Input Table

The build script expects a delimited table with a header row. At minimum it
needs:

- one sequence column, named by `SEQUENCE_COL`, containing the MPRA 3-prime UTR insert as RNA or DNA letters. `T` is treated as `U`; unknown bases are encoded as all-zero columns.
- one target column, named by `TARGET_COL`, containing a numeric RNA stability value.

Optional columns:

- an identifier column named by `ID_COL`
- a split column named by `SPLIT_COL`, with values `train`, `val`/`valid`/`validation`, and `test`
- any metadata columns you want copied into `metadata.json`

Example:

```csv
variant_id,utr_insert,rna_stability,split
var_0001,ACUGGUAUUUAA,-0.21,train
var_0002,UGUGCAUACUGA,0.34,val
var_0003,AUUUGGACUUAC,0.08,test
```

## Build LegNet Input

Submit the build job:

```bash
cd /scratch/users/isvock/transcriptml_runs/mpra_legnet
sbatch scripts/mpra/build_legnet_input.sh
```

This runs:

```bash
transcriptml build-mpra "${MPRA_TABLE}" "${DATASET_DIR}" \
  --sequence-col "${SEQUENCE_COL}" \
  --target-col "${TARGET_COL}"
```

with optional `--id-col`, `--length`, `--metadata-cols`, `--split-col`, and
`--delimiter` arguments when the matching config variables are set.

Output under `${DATASET_DIR}`:

```text
X.npy
y.npy
ids.txt
schema.json
config.json
metadata.json
splits.json  # only when SPLIT_COL is set
```

`X.npy` has shape `(N, 4, L)` with channels `A`, `C`, `G`, and `U`. `y.npy`
contains the RNA stability target values from `TARGET_COL`.

## Train And Evaluate 10-Fold CV

Submit one LegNet training/evaluation job per fold:

```bash
bash scripts/mpra/submit_train_eval_cv.sh
```

This submits a SLURM array from `0` through `N_FOLDS - 1`. Each task writes a
fold-specific dataset bundle with symlinked arrays and its own `splits.json`,
then trains and evaluates one LegNet model.

Each fold writes:

```text
${CV_ROOT}/fold0/dataset/splits.json
${CV_ROOT}/fold0/train_config.json
${CV_ROOT}/fold0/model/best.pt
${CV_ROOT}/fold0/model/last.pt
${CV_ROOT}/fold0/model/history.json
${CV_ROOT}/fold0/model/summary.json
${CV_ROOT}/fold0/model/test_predictions.csv
${CV_ROOT}/fold0/eval/test_predictions.csv
${CV_ROOT}/fold0/eval/test_predictions.summary.json
```

The fold split is deterministic from `CV_SEED`. For fold `i`, fold `i` is the
test split, fold `i + CV_VAL_OFFSET` modulo `N_FOLDS` is validation, and the
remaining folds are training.

The CV workflow ignores any `SPLIT_COL` from the source MPRA table. If you want
to train/evaluate a single predefined split instead, use the optional
`train_legnet.sh` helper below.

## Run ISM By Fold

After CV finishes, submit one ISM job per trained fold:

```bash
bash scripts/mpra/submit_ism_by_fold.sh
```

This submits a SLURM array from `0` through `N_FOLDS - 1`. Task `i` loads:

```text
${CV_ROOT}/fold${i}/model/best.pt
```

and runs ISM on `${INTERPRET_DATASET_DIR}`, which defaults to the full MPRA
bundle at `${DATASET_DIR}`.

Outputs go to:

```text
${INTERPRET_ROOT}/ism/fold0/
${INTERPRET_ROOT}/ism/fold1/
...
```

Each fold directory contains:

```text
deltas.npy
reference_predictions.npy
valid_lengths.npy
max_abs_effect.npy
summary.json
```

`deltas.npy` has shape `(N, 4, L)`. At each valid sequence position, the three
alternative base channels store `mutant_prediction - reference_prediction`; the
reference base channel remains zero. `max_abs_effect.npy` summarizes each
position by the maximum absolute single-base substitution effect.

## Optional Single-Split Helpers

For a single model instead of CV:

```bash
bash scripts/mpra/submit_train_legnet.sh
```

The script writes a run-specific training config to
`${GENERATED_TRAIN_CONFIG}` and outputs under `${MODEL_DIR}`. If `SPLIT_COL`
was set before building the dataset, it also writes split evaluation files under
`${EVAL_DIR}`. If `SPLIT_COL` is empty, training uses the random split defined
in `example_legnet_train_config.json` and writes its test predictions to
`${MODEL_DIR}/test_predictions.csv`.

After that single-model training finishes, run:

```bash
bash scripts/mpra/submit_legnet_ism.sh
```

By default this single-checkpoint helper runs:

```bash
transcriptml ism \
  "${MODEL_DIR}/best.pt" \
  "${DATASET_DIR}" \
  "${ISM_OUT_DIR}" \
  --device "${DEVICE}" \
  --batch-size "${PRED_BATCH_SIZE}" \
  --mutation-batch-size "${MUTATION_BATCH_SIZE}"
```

Set `LEGNET_CHECKPOINT` to interpret a different checkpoint, or
`INTERPRET_DATASET_DIR` to run ISM on a different MPRA bundle encoded with the
same sequence length and schema.
