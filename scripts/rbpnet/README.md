# RBPNet Sherlock workflow

> **Warning — experimental workflow:** All preprocessing, scanning, selection,
> bundle construction, training, and evaluation described here are minimally
> tested. The workflow has only been confirmed to process data successfully and
> train reasonable models on PUM2 eCLIP data. It needs substantially more
> validation than other TranscriptML functionality and should not be treated as
> production-ready or broadly validated.

This directory implements the staged workflow:

```text
FASTA/GTF/IP BAMs/SMInput BAM
  -> canonical eCLIP preprocessing
  -> window scan, selection, and RBPNet bundle
  -> immutable balanced chromosome CV plan
  -> one independent training job per fold
  -> one deterministic scientific evaluation report per fold
```

Copy this directory to a writable run directory and edit `rbpnet_config.sh`.
At minimum set `TRANSCRIPTML_REPO`, `GENOME_FASTA`, `GTF`, `SMINPUT_BAM`, and
the `IP_BAMS` array. Use `sample_name=path` values, for example:

```bash
SMINPUT_BAM="sminput=/oak/project/PUM2_sminput.bam"
IP_BAMS=(
  "ip1=/oak/project/PUM2_ip1.bam"
  "ip2=/oak/project/PUM2_ip2.bam"
)
```

Then run or submit each stage in order:

```bash
mkdir -p slurm_output
sbatch scripts/rbpnet/preprocess.sh
sbatch scripts/rbpnet/scan_select_bundle.sh
sbatch scripts/rbpnet/create_chromosome_cv_plan.sh
bash scripts/rbpnet/submit_train_cv.sh
bash scripts/rbpnet/submit_eval_cv.sh
```

The data-construction script chooses stride 1 automatically for
`original_rbpnet` and stride 50 for the other selectors unless
`WINDOW_STRIDE` is explicitly set. Its default selector is
`peak_gray_negative`; all thresholds remain editable in the config.
Set `REGION_TYPES` to a comma-separated selection universe such as `3putr` or
`cds,3putr`; matching boundary-crossing windows are included by default. Set
`DISCARD_MIXED=1` to keep only pure windows or `ONLY_MIXED=1` to keep only
matching boundary-crossing windows.

The CV stage counts examples per chromosome, greedily balances whole
chromosomes across `N_FOLDS`, and writes `CV_PLAN` once. Every fold job loads
that same file. Fold `k` uses group `k` as test, group `(k+1) mod N` as
validation, and all other groups for training. `train_cv_fold.sh 0` can be run
interactively for one fold without Slurm.

After every fold has a `${CV_ROOT}/fold<N>/model/best.pt`, run
`submit_eval_cv.sh`. It submits one `eval_cv_fold.sh` task per fold and writes:

```text
${EVAL_ROOT}/fold0/test/
  summary.json
  examples.parquet
  stratified_metrics.parquet
  calibration.parquet
  plots/
```

Set `EVAL_SAVE_PROFILES=1` to additionally save memory-mappable predicted
target, control, and IP profiles. `EVAL_SPLIT` accepts `train`, `val`, `test`,
or `all` and defaults to `test`. Evaluation is deterministic with zero jitter
and resolves indices exclusively from each checkpoint, so the shared bundle's
own `splits` value is never used. `eval_cv_fold.sh 0` runs one fold
interactively. Other evaluation controls—including batch size, device,
calibration bins, enrichment pseudocount, and representative-example sampling—
are documented directly in `rbpnet_config.sh`.

## Interpretation status

The generic `transcriptml ism` and `transcriptml motif-ablation` commands do
not currently support RBPNet checkpoints. Those tools assume one scalar model
prediction per sequence. RBPNet instead returns structured target, control,
and IP positional distributions, `pi`, and optional `eta`; an interpretation
run therefore needs an explicit scientific objective such as profile
log-likelihood, regional mass, `pi`, or `eta`. The generic predictor also feeds
the materialized bundle width directly to the model and does not apply
RBPNet's coordinate-derived zero-jitter crop and validity/measurement masks,
which is incorrect for jitter-margin bundles. No ISM or motif-ablation Slurm
scripts are provided until a structured RBPNet attribution API defines those
choices explicitly.

`example_train_config.json` enables the independent linear enrichment head and
32-nt training jitter. Change `enrichment_head_type` to `none` for profile-only
RBPNet, and keep `profile_length`/`max_train_jitter` consistent with the bundle.
