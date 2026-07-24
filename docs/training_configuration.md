# Training Configuration

TranscriptML model training is controlled by a JSON or TOML file. The same top-level
training settings are used for Saluki and MPRA-LegNet runs; the main difference
between the workflows is the model selected under `model`.

Create a starter JSON config with:

```bash
transcriptml init-run --workflow saluki --out-dir configs/saluki
transcriptml init-run --workflow legnet --out-dir configs/legnet
```

Then train directly:

```bash
transcriptml train configs/saluki/train_config.json
```

For cross-validation, pass the starter config to `transcriptml cv
prepare-fold`. Fold preparation preserves training, loss, and sequence-control
settings, while replacing `dataset`, `output_dir`, the model name, and the
fold-specific training seed.

## Saluki Starter Configuration

`transcriptml init-run --workflow saluki` writes the following training
defaults:

```json
{
  "dataset": "__EDIT_ME_DATASET_DIR__",
  "output_dir": "__EDIT_ME_RUN_DIR__/model",
  "model": {
    "name": "saluki_exact",
    "params": {}
  },
  "batch_size": 64,
  "epochs": 250,
  "learning_rate": 0.0001,
  "weight_decay": 0.0,
  "gradient_clip_norm": 0.5,
  "patience": 10,
  "monitor": ["val_loss", "val_pearson"],
  "loss": {
    "name": "mse"
  },
  "device": "auto",
  "num_workers": 0,
  "mmap_mode": "r",
  "seed": 42,
  "split_source": "auto",
  "split": {
    "method": "random",
    "val_frac": 0.1,
    "test_frac": 0.1
  }
}
```

An empty `model.params` mapping means that the model constructor defaults
described below are used. You only need to add parameters that you want to
change.

The Sherlock workflow uses `scripts/example_train_config.json` as its base
config and starts from the same optimization settings. Its shell helpers
replace the two path fields, so leave `dataset` and `output_dir` as placeholders
when using those scripts.

## MPRA-LegNet Starter Configuration

`transcriptml init-run --workflow legnet` selects LegNet and writes:

```json
{
  "dataset": "__EDIT_ME_DATASET_DIR__",
  "output_dir": "__EDIT_ME_RUN_DIR__/model",
  "model": {
    "name": "legnet",
    "params": {}
  },
  "batch_size": 64,
  "epochs": 20,
  "learning_rate": 0.001,
  "weight_decay": 0.0,
  "patience": 5,
  "monitor": "val_loss",
  "loss": {
    "name": "mse"
  },
  "device": "auto",
  "seed": 123,
  "split_source": "auto",
  "split": {
    "method": "random",
    "val_frac": 0.1,
    "test_frac": 0.1
  }
}
```

Fields omitted from this starter, such as `gradient_clip_norm`,
`num_workers`, and `mmap_mode`, use the generic trainer fallbacks in the next
section. The Sherlock MPRA workflow has its own editable base config at
`scripts/mpra/example_legnet_train_config.json`.

## Top-Level Training Settings

The following fields are accepted by `transcriptml train`. The default column
shows the generic trainer fallback when a field is omitted. Workflow starter
configs can deliberately override those fallbacks, as the Saluki starter does
above.

| Field | Type | Default when omitted | Meaning |
| --- | --- | --- | --- |
| `dataset` | path | required | Dataset bundle containing `X.npy` and its sidecar files. |
| `output_dir` | path | required | Directory for checkpoints, history, split information, predictions, and the run summary. |
| `model` | mapping or string | `small_cnn` | Registered model name and optional constructor parameters. Saluki and MPRA starters explicitly select their workflow model. |
| `batch_size` | integer | `64` | Number of examples per optimizer or evaluation batch. A final singleton training batch is dropped because batch-normalized models cannot train on it reliably. |
| `epochs` | integer | `20` | Maximum number of training epochs before early stopping. |
| `learning_rate` | float | `0.001` | Learning rate passed to the AdamW optimizer. |
| `weight_decay` | float | `0.0` | AdamW weight-decay coefficient. |
| `gradient_clip_norm` | float or `null` | `0.5` | Maximum global gradient norm. Set to `null`, `0`, or a negative value to disable clipping. |
| `patience` | integer | `5` | Number of consecutive non-improving epochs tolerated by early stopping. A negative value disables early stopping. |
| `monitor` | string or list | `"val_loss"` | Validation metric or metrics used to select `best.pt` and reset early-stopping patience. |
| `loss` | string, mapping, or `null` | `{"name": "mse"}` | Training objective. Available loss configurations are described below. |
| `device` | string | `"cpu"` | PyTorch device such as `cpu`, `cuda`, or `cuda:0`. `auto` selects CUDA when available and otherwise uses CPU. |
| `num_workers` | integer | `0` | Number of worker processes used by each PyTorch DataLoader. Workers remain alive between epochs when this is greater than zero. |
| `mmap_mode` | string or `null` | `"r"` | NumPy memory-map mode used when loading bundle arrays. Use `"r"` to avoid reading the complete input into memory, or `null` to load it normally. |
| `seed` | integer | `123` | Seeds Python, NumPy, and PyTorch. It also seeds a config-defined random split unless `split.seed` is set. |
| `progress` | boolean | `true` | Whether to print data-processing, batch, epoch, and evaluation progress. |
| `sequence_controls` | mapping, list, or `null` | `null` | Optional sequence ablations applied before split selection. |
| `split_source` | string | `"auto"` | Whether splits come from the bundle or from the `split` block. |
| `split` | mapping | random 80/10/10 | Config-defined split settings, used according to `split_source`. |

The canonical model mapping contains a registered `name` and a `params`
mapping:

```json
{
  "model": {
    "name": "saluki_exact",
    "params": {
      "filters": 64
    }
  }
}
```

Parameters that are absent from `params` use the selected model's defaults.

### Checkpoint Selection And Early Stopping

The metrics available to `monitor` are:

- `train_loss`
- `train_pearson`
- `val_loss`
- `val_pearson`

Loss metrics improve when they decrease; Pearson metrics improve when they
increase. A string can name one metric:

```json
{"monitor": "val_loss"}
```

A list uses an OR rule. With:

```json
{"monitor": ["val_loss", "val_pearson"]}
```

an epoch is considered improved when validation loss decreases or validation
Pearson correlation increases. That epoch replaces `best.pt` and resets
patience. `last.pt` is written after every epoch.

## Saluki Model Parameters

Saluki dataset bundles normally have six channels: A, C, G, U, CDS codon
starts, and splice junctions. TranscriptML provides the close architecture
reproduction `saluki_exact` and the smaller configurable alternative
`saluki_like`.

Inspect the installed defaults at any time:

```bash
transcriptml models show saluki_exact --json
transcriptml models show saluki_like --json
```

### `saluki_exact`

This is the model used by the standard Saluki workflow.

```json
{
  "model": {
    "name": "saluki_exact",
    "params": {
      "seq_depth": 6,
      "filters": 64,
      "kernel_size": 5,
      "num_layers": 6,
      "dropout": 0.3,
      "augment_shift": 3,
      "ln_epsilon": 0.007,
      "keras_bn_momentum": 0.9,
      "bn_eps": 0.001
    }
  }
}
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| `seq_depth` | `6` | Number of input channels. Keep this at six for an ordinary Saluki bundle. |
| `filters` | `64` | Width of the convolutional stack, GRU, and dense hidden layer. |
| `kernel_size` | `5` | Width of the initial and repeated one-dimensional convolutions. |
| `num_layers` | `6` | Number of convolution, dropout, and max-pooling blocks after the initial convolution. |
| `dropout` | `0.3` | Dropout probability in convolutional blocks and the dense head. |
| `augment_shift` | `3` | Maximum random right shift applied during training. The sampled shift is between zero and this value; set to zero to disable it. |
| `ln_epsilon` | `0.007` | Numerical epsilon used by channel layer normalization. |
| `keras_bn_momentum` | `0.9` | Batch-normalization momentum expressed using the Keras convention reproduced by this model. |
| `bn_eps` | `0.001` | Numerical epsilon used by batch-normalization layers in the head. |

### `saluki_like`

`saluki_like` is a compact convolutional/GRU model inspired by Saluki rather
than an exact reproduction. `saluki_gru` is an alias for the same
implementation.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `in_ch` | `6` | Number of input channels. |
| `base_ch` | `64` | Number of channels in every convolutional block. |
| `kernel_size` | `5` | Width of each one-dimensional convolution. |
| `n_convs` | `4` | Number of convolutional blocks; must be at least one. |
| `pool_size` | `2` | Max-pooling factor after each convolution. Values at or below one disable pooling. |
| `dropout` | `0.2` | Dropout probability in the encoder, recurrent stack when applicable, and regression head. |
| `gru_hidden` | `64` | GRU hidden-state width. |
| `gru_layers` | `1` | Number of stacked GRU layers. |
| `bidirectional` | `false` | Whether the GRU reads the sequence in both directions. |
| `head_hidden` | `64` | Width of the hidden layer in the regression head. |
| `output_dim` | `1` | Number of outputs. Leave this at one for TranscriptML's scalar training workflow. |

## MPRA-LegNet Model Parameters

MPRA bundles have four A/C/G/U channels. The standard MPRA workflow selects the
registered `legnet` model:

```json
{
  "model": {
    "name": "legnet",
    "params": {
      "in_ch": 4,
      "stem_ch": 64,
      "stem_ks": 7,
      "ef_ks": 5,
      "ef_block_sizes": [64, 96, 128],
      "pool_sizes": [2, 2, 2],
      "resize_factor": 4,
      "block_dropout": 0.0,
      "head_dropout": 0.1,
      "stem_dropout": 0.0,
      "output_dim": 1
    }
  }
}
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| `in_ch` | `4` | Number of input channels. Keep this at four for an ordinary MPRA bundle. |
| `stem_ch` | `64` | Number of channels produced by the initial convolutional stem. |
| `stem_ks` | `7` | Kernel size of the stem convolution. |
| `ef_ks` | `5` | Kernel size used by efficient and local blocks after the stem. |
| `ef_block_sizes` | `[64, 96, 128]` | Output-channel width of each successive LegNet stage. |
| `pool_sizes` | `[2, 2, 2]` | Pooling factor after each stage. This list must have the same length as `ef_block_sizes`; values at or below one disable pooling for that stage. |
| `resize_factor` | `4` | Internal channel expansion factor in each efficient block. |
| `block_dropout` | `0.0` | Channel-dropout probability inside efficient and local stage blocks. |
| `head_dropout` | `0.1` | Dropout probability in the regression head. |
| `stem_dropout` | `0.0` | Channel-dropout probability after the stem activation. |
| `output_dim` | `1` | Number of outputs. Leave this at one for TranscriptML's scalar training workflow. |

Run `transcriptml models show legnet --json` to inspect these defaults from the
installed package.

## Loss Configuration

TranscriptML supports unweighted MSE, metadata-weighted MSE, and a binomial
count likelihood. Losses that refer to metadata columns require those columns
to be present in the dataset bundle's `metadata.json`.

### Unweighted MSE

This is the default for both workflows:

```json
{
  "loss": {
    "name": "mse"
  }
}
```

The string form `"loss": "mse"` and a missing or `null` loss block have the
same effect.

### Weighted MSE

Use exactly one of `weight_col` or `se_col`.

If metadata already contains final per-example weights:

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

If metadata instead contains standard errors:

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

| Field | Default | Meaning |
| --- | --- | --- |
| `name` | required | Use `"weighted_mse"`. |
| `weight_col` | none | Metadata column containing final nonnegative weights. Mutually exclusive with `se_col`. |
| `se_col` | none | Metadata column containing nonnegative standard errors. Weights are calculated as `1 / (se^2 + eps)`. |
| `eps` | `1e-8` | Stabilizer used only when deriving weights from `se_col`. |
| `min_weight` | `0.01` | Lower clipping bound. Set to `null` for no lower bound. |
| `max_weight` | `100.0` | Upper clipping bound. Set to `null` for no upper bound. |

### Binomial Count Likelihood

This loss is intended for pulse-labeling measurements with total reads, new
reads, and pulse duration:

```json
{
  "loss": {
    "name": "binomial_nll",
    "total_reads_col": "total_reads",
    "new_reads_col": "new_reads",
    "pulse_hours_col": "pulse_hours",
    "eps": 1e-7,
    "max_rate_time": 80.0,
    "log_base": "e"
  }
}
```

The model output is interpreted as `log(kdeg)`. The likelihood uses:

```text
new_reads ~ Binomial(total_reads, 1 - exp(-kdeg * pulse_hours))
```

| Field | Default | Meaning |
| --- | --- | --- |
| `total_reads_col` | `"total_reads"` | Metadata column containing positive total-read counts. |
| `new_reads_col` | `"new_reads"` | Metadata column containing new-read counts between zero and total reads. |
| `pulse_hours_col` | `"pulse_hours"` | Metadata column containing positive pulse durations in hours. |
| `eps` | `1e-7` | Numerical lower bound used in probability and rate calculations. |
| `max_rate_time` | `80.0` | Upper bound on `kdeg * pulse_hours` used for numerical stability. |
| `log_base` | `"e"` | Base of the model's log-rate output. Accepted values include `"e"`, `"10"`, `"2"`, or another positive numeric base other than one. |

This loss can train without `y.npy`, but keeping a scalar target is useful
because TranscriptML can then report Pearson correlation and MSE alongside the
count likelihood.

## Split Configuration

`split_source` determines whether training uses splits already stored in the
bundle or constructs them from the `split` block.

| `split_source` | Behavior |
| --- | --- |
| `"auto"` | Use bundle splits when present; otherwise use the config-defined `split` block. |
| `"bundle"` | Require and use the bundle's `splits.json`. |
| `"config"` | Ignore bundle splits and construct splits from the config. |

Cross-validation fold preparation writes `splits.json` inside each fold bundle.
The usual CV workflow therefore uses those fold assignments under the default
`"auto"` setting.

### Random Splits

```json
{
  "split_source": "config",
  "split": {
    "method": "random",
    "val_frac": 0.1,
    "test_frac": 0.1,
    "seed": 42
  }
}
```

`val_frac` and `test_frac` must each be between zero and one, and their sum must
be less than one. `split.seed` overrides the top-level training seed for split
assignment.

### Metadata Splits

```json
{
  "split_source": "config",
  "split": {
    "method": "metadata",
    "split_col": "split"
  }
}
```

The selected metadata column may use `train`, `val`, `valid`, `validation`, or
`test` labels.

### Explicit Split Indices

```json
{
  "split_source": "config",
  "split": {
    "method": "predefined",
    "splits": {
      "train": [0, 1, 2, 3],
      "val": [4],
      "test": [5]
    }
  }
}
```

No example index may occur in more than one split.

## Sequence Controls

Sequence controls perturb selected parts of the input before TranscriptML
chooses train, validation, and test examples. All splits in that training run
therefore use the same controlled representation.

The preferred form is an explicit operation list:

```json
{
  "sequence_controls": {
    "seed": 42,
    "operations": [
      {
        "operation": "randomize_nucleotides",
        "regions": ["cds"]
      }
    ]
  }
}
```

This example replaces every represented CDS nucleotide independently with a
uniformly sampled A, C, G, or U. It preserves UTR bases, region lengths,
padding, splice annotations, and the original CDS codon-start channel.
Transcripts without a detectable CDS are skipped and counted in the run
summary.

### Available Operations

| Operation | Regions | Effect |
| --- | --- | --- |
| `shuffle_nucleotides` | `5utr`, `cds`, `3utr`, or `transcript` | Permutes the existing nucleotide calls within each selected region, preserving its nucleotide composition. |
| `shuffle_codons` | `cds` only | Permutes complete existing CDS codons as three-nucleotide units, preserving the multiset of codons. |
| `randomize_nucleotides` | `5utr`, `cds`, `3utr`, or `transcript` | Replaces every selected position independently with uniformly random A, C, G, or U. |
| `cds_frameshift` | `cds` only | Shifts the CDS/codon-start annotation channel by one or two positions while leaving nucleotide and splice channels unchanged. |

The first three operations rewrite only A/C/G/U base channels. `cds_frameshift`
rewrites only the CDS annotation channel, so it can be combined with one
base-editing operation on the CDS.

Use `shuffle_codons` when the control should preserve CDS codon composition.
Use `randomize_nucleotides` when the sequence itself should be replaced rather
than rearranged.

### Regions

The canonical region names are:

- `5utr`
- `cds`
- `3utr`
- `transcript`

`all` expands to the three annotated regions independently. A
whole-`transcript` base operation cannot be combined with separate 5-prime UTR,
CDS, or 3-prime UTR base operations.

When `regions` is omitted, `shuffle_nucleotides` and
`randomize_nucleotides` select all three annotated regions.
`shuffle_codons` and `cds_frameshift` select the CDS.

Annotated regions require a schema with a resolvable CDS channel, such as a
Saluki bundle. An ordinary four-channel MPRA bundle can use transcript-wide
`shuffle_nucleotides` or `randomize_nucleotides`, but it cannot identify
5-prime UTR, CDS, or 3-prime UTR boundaries.

### Frameshift Controls

The shift must be one or two:

```json
{
  "sequence_controls": {
    "operations": [
      {
        "operation": "cds_frameshift",
        "shift": 1
      }
    ]
  }
}
```

This tests the annotation frame supplied to the model. It does not insert,
delete, or move nucleotide bases.

### Sequence-Control Settings

| Field | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` | Set to `false` to disable the complete block without deleting it. |
| `seed` | `0` | Seed used for sequence controls. This is separate from the top-level training seed. |
| `operations` | none | List of operation mappings. Each entry contains `operation`, optional `regions`, and `shift` for `cds_frameshift`. |
| `save` | `false` | Save the controlled bundle under `<output_dir>/sequence_controlled_dataset`. |
| `save_dir` | none | Save to an explicit bundle directory. Providing this field also enables saving. |
| `cds_channel` | inferred | CDS channel name or zero-based channel index. Usually unnecessary for a standard Saluki schema. |

Randomization is deterministic for a given sequence-control seed, sequence
index, operation, and region. In CV, keeping `sequence_controls.seed` fixed
therefore gives every fold the same controlled input, even though the
top-level training seed changes by fold.

Setting `save` writes a complete bundle plus `sequence_controls.json`:

```json
{
  "sequence_controls": {
    "seed": 42,
    "save": true,
    "operations": [
      {
        "operation": "randomize_nucleotides",
        "regions": ["cds"]
      }
    ]
  }
}
```

This is useful when a separate `transcriptml evaluate` or interpretation
command must consume the controlled inputs. Those commands do not
automatically reconstruct controls from the checkpoint; point `--dataset` at
the saved controlled bundle.

In the Sherlock CV script, for example, a saved fold bundle is located at:

```text
${CV_ROOT}/foldN/model/sequence_controlled_dataset
```

Evaluating `${CV_ROOT}/foldN/dataset` instead uses the original unmodified
inputs.

### Shortcut And Legacy Forms

Top-level operation shortcuts are accepted inside `sequence_controls`:

```json
{
  "sequence_controls": {
    "seed": 42,
    "randomize_nucleotides": ["cds"]
  }
}
```

`shuffle_nucleotides`, `shuffle_codons`, `randomize_nucleotides`, and
`cds_frameshift` all have shortcut forms. For the three base operations, a
boolean `true` selects the operation's default regions; an explicit region list
is clearer.

The value of `sequence_controls` can also be a bare operation list:

```json
{
  "sequence_controls": [
    {
      "operation": "randomize_nucleotides",
      "regions": ["transcript"]
    }
  ]
}
```

This compact form uses sequence-control seed zero and cannot set `save`,
`save_dir`, or `cds_channel`; use the mapping form when those settings matter.

The older keys `5pUTR_ablation`, `CDS_ablation`, and `3pUTR_ablation` remain
available for compatibility. Their `"scramble"` mode shuffles nucleotides in
UTRs but shuffles codon units in the CDS, while `"ablate"` selects independent
random nucleotides. The older `"true_scramble"` mode shuffles nucleotides,
including within the CDS. New configs should use the explicit operation names
so that the intended control is unambiguous.

## Using These Settings On Sherlock

The Saluki scripts read:

```text
scripts/example_train_config.json
```

The MPRA-LegNet scripts read:

```text
scripts/mpra/example_legnet_train_config.json
```

Edit the copied file in your Sherlock run directory. The split and CV wrappers
preserve model parameters, training settings, losses, and sequence controls.
They replace `dataset` and `output_dir`; CV preparation also selects the
configured CV model and adds the fold number to the top-level training seed.

For example:

```bash
# Saluki
bash scripts/submit_train_eval_cv.sh

# MPRA-LegNet
bash scripts/mpra/submit_train_eval_cv.sh
```

After a fold starts, inspect the fully resolved config at:

```text
${CV_ROOT}/foldN/train_config.json
```

That file is the clearest record of the exact settings used for one trained
model.
