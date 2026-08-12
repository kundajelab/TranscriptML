# RBPNet Sherlock workflow

This directory implements the staged workflow:

```text
FASTA/GTF/IP BAMs/SMInput BAM
  -> canonical eCLIP preprocessing
  -> window scan, selection, and RBPNet bundle
  -> immutable balanced chromosome CV plan
  -> one independent training job per fold
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
```

The data-construction script chooses stride 1 automatically for
`original_rbpnet` and stride 50 for the other selectors unless
`WINDOW_STRIDE` is explicitly set. Its default selector is
`peak_gray_negative`; all thresholds remain editable in the config.

The CV stage counts examples per chromosome, greedily balances whole
chromosomes across `N_FOLDS`, and writes `CV_PLAN` once. Every fold job loads
that same file. Fold `k` uses group `k` as test, group `(k+1) mod N` as
validation, and all other groups for training. `train_cv_fold.sh 0` can be run
interactively for one fold without Slurm.

`example_train_config.json` enables the independent linear enrichment head and
32-nt training jitter. Change `enrichment_head_type` to `none` for profile-only
RBPNet, and keep `profile_length`/`max_train_jitter` consistent with the bundle.
