# MPRA LegNet Scripts

These scripts are intentionally small and Sherlock-oriented, following the
same pattern as the transcriptome-derived training scripts in `scripts/`.
For a new run, keep the cloned TranscriptML repo clean, copy `scripts/mpra/`
to a run/work directory, edit the copied config files there, and submit jobs
from the copied scripts.

## Files

- `mpra_config.sh`: per-run shell config after you copy `scripts/mpra/`; set paths, conda environment, MPRA table columns, and runtime knobs in the copied file.
- `example_legnet_train_config.json`: per-run training config after you copy `scripts/mpra/`; edit LegNet and training hyperparameters in the copied file. `train_legnet.sh` replaces `dataset` and `output_dir` in the generated config.
- `build_legnet_input.sh`: builds a 4-channel RNA MPRA dataset bundle with `transcriptml build-mpra`.
- `train_legnet.sh` and `submit_train_legnet.sh`: train one LegNet model from the MPRA bundle, and optionally evaluate a predefined split.
- `run_legnet_ism.sh` and `submit_legnet_ism.sh`: run single-nucleotide ISM on the trained LegNet checkpoint.
- `write_legnet_train_config.py`: internal helper used by `train_legnet.sh` to generate the run-specific train config.

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
TRAIN_OUTPUT_ROOT="${RUN_ROOT}/train_eval"
INTERPRET_ROOT="${RUN_ROOT}/interpret"
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
| `RUN_ROOT`, `DATASET_DIR`, `TRAIN_OUTPUT_ROOT`, `INTERPRET_ROOT` | Output locations for the dataset, model/evaluation files, and ISM files. |
| `BASE_TRAIN_CONFIG` | Base JSON training config. Defaults to `scripts/mpra/example_legnet_train_config.json`. |
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

## Train LegNet

Submit the training job:

```bash
bash scripts/mpra/submit_train_legnet.sh
```

or submit the job script directly:

```bash
sbatch scripts/mpra/train_legnet.sh
```

The script writes a run-specific training config to
`${GENERATED_TRAIN_CONFIG}`, then runs:

```bash
transcriptml train "${GENERATED_TRAIN_CONFIG}"
```

Outputs:

```text
${GENERATED_TRAIN_CONFIG}
${MODEL_DIR}/best.pt
${MODEL_DIR}/last.pt
${MODEL_DIR}/history.json
${MODEL_DIR}/splits.json
${MODEL_DIR}/summary.json
${MODEL_DIR}/test_predictions.csv
```

If `SPLIT_COL` was set before building the dataset, `train_legnet.sh` also
runs `transcriptml evaluate` on `${EVAL_SPLIT}` and writes:

```text
${EVAL_DIR}/${EVAL_SPLIT}_predictions.csv
${EVAL_DIR}/${EVAL_SPLIT}_predictions.summary.json
```

If `SPLIT_COL` is empty, training uses the random split defined in
`example_legnet_train_config.json`. In that case, use
`${MODEL_DIR}/splits.json` and `${MODEL_DIR}/test_predictions.csv` as the
record of the split and test predictions.

## Run Single-Nucleotide ISM

After training finishes, submit ISM:

```bash
bash scripts/mpra/submit_legnet_ism.sh
```

or submit the job script directly:

```bash
sbatch scripts/mpra/run_legnet_ism.sh
```

By default this runs:

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

Outputs under `${ISM_OUT_DIR}`:

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
