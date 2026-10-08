# Training Configuration

TranscriptML model training is controlled by a JSON or TOML file. The same
top-level training settings are used across Saluki, MPRA-LegNet, and structured
RBPNet runs; the model and loss determine the batch contract.

```{warning}
RBPNet preprocessing and modeling are experimental. They have been minimally
tested and have only been confirmed to preprocess data successfully and train
reasonable models on PUM2 eCLIP data. They need substantially more validation
than other TranscriptML functionality.
```

Create a starter JSON config with:

```bash
transcriptml init-run --workflow saluki --out-dir configs/saluki
transcriptml init-run --workflow legnet --out-dir configs/legnet
transcriptml init-run --workflow rbpnet --out-dir configs/rbpnet
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
  "head_layernorm": false,
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

## RBPNet Starter Configuration

```{warning}
This starter config is part of the experimental RBPNet workflow. Successful
execution and reasonable training behavior have been checked on PUM2 eCLIP,
but the preprocessing, model, losses, and evaluation require broader
validation before scientific or production use.
```

`transcriptml init-run --workflow rbpnet` selects the structured RBPNet trainer.
Edit the bundle path, output path, and `profile_length` to match the bundle:

```json
{
  "dataset": "__EDIT_ME_RBPNET_BUNDLE_DIR__",
  "output_dir": "__EDIT_ME_RUN_DIR__/model",
  "model": {
    "name": "rbpnet",
    "params": {
      "profile_length": 300,
      "enrichment_head_type": "none"
    }
  },
  "batch_size": 64,
  "epochs": 100,
  "learning_rate": 0.001,
  "weight_decay": 0.0,
  "optimizer": {"name": "adamw"},
  "lr_scheduler": {"name": "reduce_on_plateau", "patience": 3},
  "mixed_precision": false,
  "gradient_clip_norm": 0.5,
  "patience": 10,
  "monitor": "val_loss",
  "loss": {
    "name": "rbpnet",
    "lambda_ip_profile": 1.0,
    "lambda_sm_profile": 1.0,
    "lambda_enrichment": 1.0
  },
  "device": "auto",
  "num_workers": 0,
  "mmap_mode": "r",
  "seed": 123,
  "max_train_jitter": 0,
  "deduplicate_loci": true,
  "split_source": "config",
  "split": {
    "method": "group",
    "group_col": "group_gene_id",
    "val_frac": 0.1,
    "test_frac": 0.1
  }
}
```

Set `enrichment_head_type` to `linear` (or `mlp`) to add the independent
replicate-aware enrichment likelihood. The three RBPNet component weights are
independent; `lambda_enrichment` has no effect when the head is disabled. See
the {ref}`RBPNet guide <rbpnet-model-and-training>` for the equations, bundle
fields, and jitter semantics.

## Top-Level Training Settings

The following fields are accepted by `transcriptml train`. The default column
shows the generic trainer fallback when a field is omitted. Workflow starter
configs can deliberately override those fallbacks, as the Saluki starter does
above.

| Field | Type | Default when omitted | Meaning |
| --- | --- | --- | --- |
| `dataset` | path | required | Dataset bundle containing `X.npy` and its sidecar files. |
| `output_dir` | path | required | Directory for checkpoints, history, split information, predictions, and the run summary. |
| `model` | mapping or string | `small_cnn` | Registered model name and optional constructor parameters. Workflow starters explicitly select their model. |
| `batch_size` | integer | `64` | Number of examples per optimizer or evaluation batch. A final singleton training batch is dropped because batch-normalized models cannot train on it reliably. |
| `epochs` | integer | `20` | Maximum number of training epochs before early stopping. |
| `learning_rate` | float | `0.001` | Learning rate. Structured RBPNet passes this to the selected optimizer unless overridden there. |
| `weight_decay` | float | `0.0` | Weight-decay coefficient. |
| `optimizer` | string or mapping | `"adamw"` | Structured RBPNet supports AdamW, Adam, and SGD, with optional optimizer parameters. Scalar workflows retain AdamW. |
| `lr_scheduler` | string, mapping, or `null` | `null` | Structured RBPNet supports plateau, cosine, and step schedulers. |
| `mixed_precision` | boolean | `false` | Enable autocast; CUDA also uses gradient scaling. |
| `gradient_clip_norm` | float or `null` | `0.5` | Maximum global gradient norm. Set to `null`, `0`, or a negative value to disable clipping. |
| `patience` | integer | `5` | Number of consecutive non-improving epochs tolerated by early stopping. A negative value disables early stopping. |
| `monitor` | string or list | `"val_loss"` | Validation metric or metrics used to select `best.pt` and reset early-stopping patience. |
| `loss` | string, mapping, or `null` | `{"name": "mse"}` | Training objective. Available loss configurations are described below. |
| `device` | string | `"cpu"` | PyTorch device such as `cpu`, `cuda`, or `cuda:0`. `auto` selects CUDA when available and otherwise uses CPU. |
| `num_workers` | integer | `0` | Number of worker processes used by each PyTorch DataLoader. Workers remain alive between epochs when this is greater than zero. |
| `mmap_mode` | string or `null` | `"r"` | NumPy memory-map mode used when loading bundle arrays. Use `"r"` to avoid reading the complete input into memory, or `null` to load it normally. |
| `seed` | integer | `123` | Seeds Python, NumPy, and PyTorch. It also seeds a config-defined random split unless `split.seed` is set. |
| `progress` | boolean | `true` | Whether to print data-processing, batch, epoch, and evaluation progress. |
| `debug_epoch_predictions` | boolean | `false` | Save deterministic end-of-epoch train and validation predictions to `debug_epoch_predictions.csv`. This adds one evaluation pass over the training split per epoch. |
| `head_layernorm` | boolean | `false` | For `saluki_exact`, replace both dense-head BatchNorm layers with per-example LayerNorm. Other model types reject `true`. |
| `sequence_controls` | mapping, list, or `null` | `null` | Optional sequence ablations applied before split selection. |
| `split_source` | string | `"auto"` | Whether splits come from the bundle or from the `split` block. |
| `split` | mapping | random 80/10/10 | Config-defined split settings, used according to `split_source`. |
| `max_train_jitter` | integer | `0` | Structured RBPNet shift range; cannot exceed the bundle's materialized margin. Evaluation remains shift zero. |
| `deduplicate_loci` | boolean | `true` | Collapse repeated eligibility rows for an identical RBPNet locus while retaining all replicate arrays. |
| `allow_random_window_split` | boolean | `false` | Explicitly permit unsafe RBPNet row-random splitting. Grouped splitting is the safe default. |
| `cv_plan` | path or `null` | `null` | Saved balanced chromosome CV plan. Must be provided together with `fold`; it takes precedence over `split_source`. |
| `fold` | integer or `null` | `null` | Zero-based chromosome CV test-fold index. Validation is the following fold modulo `n_folds`. |

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

Structured RBPNet runs instead expose `train_loss`, `val_loss`, and the
corresponding `train_`/`val_` forms of `ip_profile_loss`, `sm_profile_loss`, and
`enrichment_loss`. Profile-only runs report enrichment loss as zero.

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

### Epoch Prediction Debugging

Set `"debug_epoch_predictions": true` to write
`debug_epoch_predictions.csv` in the training output directory. The file has
one row per train or validation example per completed epoch. Predictions are
made in evaluation mode using the model at the end of the epoch, so dropout
and stochastic augmentation are disabled and all training examples are
included.

The columns include `epoch`, `split`, original dataset `index`, `id`, `target`,
`prediction`, `residual`, `squared_error`, configured split-level `loss` and
`pearson`, the corresponding online `history_loss` and `history_pearson`,
`loss_name`, and checkpoint-monitoring context. Split-level metrics are
repeated on each example row to keep the CSV self-contained.

For the training split, `loss` and `pearson` can differ from the
`history_loss` and `history_pearson` columns. History metrics are collected
while batches are being trained and the model parameters are changing, with
augmentation and dropout active when configured. The debug metrics instead
evaluate the final model state for that epoch deterministically.

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
      "pooling": "max",
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
| `num_layers` | `6` | Number of convolution, dropout, and pooling blocks after the initial convolution. |
| `pooling` | `"max"` | Downsampling operation in each convolutional block. Use `"max"` or `"average"`. |
| `dropout` | `0.3` | Dropout probability in convolutional blocks and the dense head. |
| `augment_shift` | `3` | Maximum random right shift applied during training. The sampled shift is between zero and this value; set to zero to disable it. |
| `ln_epsilon` | `0.007` | Numerical epsilon used by channel layer normalization. |
| `keras_bn_momentum` | `0.9` | Batch-normalization momentum expressed using the Keras convention reproduced by this model. |
| `bn_eps` | `0.001` | Numerical epsilon used by normalization layers in the head. |
| `head_layernorm` | `false` | Checkpoint-level record of whether the head uses LayerNorm. During training, set the top-level `head_layernorm` field instead. |

Set the top-level training option to enable the experimental head:

```json
{
  "model": {"name": "saluki_exact", "params": {}},
  "head_layernorm": true
}
```

This preserves the original `Normalization → ReLU → Linear → Dropout →
Normalization → ReLU → Linear` ordering, but makes both head normalization
layers independent of batch and running statistics. LayerNorm uses `bn_eps`
so enabling the option changes the normalization behavior without also
changing its numerical epsilon. The resolved value is saved in checkpoint
`model_config.params`, allowing the checkpoint loader to reconstruct the
correct head, and is also recorded in `summary.json`.

To use average pooling and disable stochastic shift augmentation, set the
corresponding model parameters:

```json
{
  "model": {
    "name": "saluki_exact",
    "params": {
      "pooling": "average",
      "augment_shift": 0
    }
  }
}
```

During each training forward pass, a positive `augment_shift` samples one
integer offset from zero through the configured maximum. All channels are
shifted right together, zeros are inserted at the left boundary, and the same
number of positions are removed from the right boundary. Evaluation mode never
applies the shift. Setting `augment_shift` to zero bypasses the operation in
training mode as well.

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

## Convolution Universe Model

`convolution_universe` is a configurable convolutional or convolutional/recurrent
model for controlled architecture experiments. It uses the same flat
`model.params` interface and checkpoint handling as every other registered
TranscriptML model.

Inspect the installed defaults with:

```bash
transcriptml models show convolution_universe --json
```

The default is a six-channel, fully convolutional residual network. It has a
kernel-11 stem followed by six kernel-7 blocks with dilations
`[1, 2, 4, 8, 16, 32]`, no internal pooling, and global average-plus-maximum
pooling before the MLP head. An empty parameter mapping therefore selects the
complete default:

```json
{
  "model": {
    "name": "convolution_universe",
    "params": {}
  }
}
```

### Pooling CNN example

Set dilation to one and pool after every block to build a CNN whose receptive
field grows through downsampling rather than dilation:

```json
{
  "model": {
    "name": "convolution_universe",
    "params": {
      "dilations": 1,
      "pooling": "max",
      "pool_size": 2
    }
  }
}
```

`filters`, `kernel_size`, `dilations`, `pooling`, and `pool_size` may each be a
single value or a list with exactly `num_layers` entries. This permits staged
widths and selective downsampling without changing the ordinary flat model
configuration:

```json
{
  "model": {
    "name": "convolution_universe",
    "params": {
      "num_layers": 4,
      "filters": [64, 64, 128, 128],
      "kernel_size": [7, 7, 5, 5],
      "dilations": [1, 2, 1, 2],
      "pooling": ["none", "max", "none", "max"],
      "pool_size": [2, 2, 2, 2]
    }
  }
}
```

### Exact Saluki configuration

`ConvolutionUniverseConfig.saluki()` returns the following complete
configuration. It matches `saluki_exact`'s layer order, reverse GRU traversal,
head, shift augmentation, and initialization while leaving `saluki_exact`
itself and its checkpoints unchanged:

```json
{
  "model": {
    "name": "convolution_universe",
    "params": {
      "in_ch": 6,
      "output_dim": 1,
      "augment_shift": 3,
      "weight_init": "saluki",
      "filters": 64,
      "num_layers": 6,
      "stem_kernel_size": 5,
      "stem_padding": "valid",
      "stem_bias": false,
      "block_type": "conv",
      "kernel_size": 5,
      "dilations": 1,
      "padding": "valid",
      "conv_bias": true,
      "norm": "layer",
      "norm_position": "pre",
      "norm_eps": 0.007,
      "activation": "relu",
      "dropout": 0.3,
      "use_residual": false,
      "pooling": "max",
      "pool_size": 2,
      "mlp_ratio": 4,
      "layer_scale_init": 0.000001,
      "drop_path_rate": 0.0,
      "pre_rnn_norm": "layer",
      "pre_rnn_activation": "relu",
      "rnn_type": "gru",
      "rnn_hidden": 64,
      "rnn_layers": 1,
      "rnn_bidirectional": false,
      "rnn_reverse": true,
      "rnn_summary": "last",
      "rnn_dropout": 0.0,
      "global_pool": "avgmax",
      "global_pool_size": 1,
      "head_type": "mlp",
      "head_hidden": 64,
      "head_norm": "batch",
      "head_norm_position": "pre",
      "head_activation": "relu",
      "head_dropout": 0.3,
      "head_norm_eps": 0.001,
      "head_bn_momentum": 0.1
    }
  }
}
```

The class method can also be used directly in Python:

```python
from transcriptml.models import ConvolutionUniverse, ConvolutionUniverseConfig

config = ConvolutionUniverseConfig.saluki()
model = ConvolutionUniverse(**config.to_kwargs())
```

Once this configuration is in a training JSON file, Saluki ablations are
ordinary parameter edits. For example:

- set `rnn_type` to `none` to replace the GRU with `global_pool`;
- set `pooling` to `average` or to a per-layer list;
- set `padding` to `same`, enable `use_residual`, and supply a dilation schedule;
- set `block_type` to `convnext` for ConvNeXt-v1-style blocks;
- set `head_norm` to `layer` to remove batch-dependent head normalization; or
- select an LSTM, multiple recurrent layers, or bidirectionality with the
  corresponding `rnn_*` fields.

When enabling residuals, convolutional `padding` must be `same`. The stochastic
shift, recurrent reverse traversal, and Saluki initialization are independent
knobs and should be disabled explicitly when an ablation is intended to remove
them.

### Parameter groups

| Group | Parameters |
| --- | --- |
| Input and stem | `in_ch`, `output_dim`, `augment_shift`, `weight_init`, `filters`, `stem_kernel_size`, `stem_padding`, `stem_bias` |
| Convolutional tower | `num_layers`, `block_type`, `kernel_size`, `dilations`, `padding`, `conv_bias`, `norm`, `norm_position`, `norm_eps`, `activation`, `dropout`, `use_residual`, `pooling`, `pool_size` |
| ConvNeXt-specific | `mlp_ratio`, `layer_scale_init`, `drop_path_rate` |
| Recurrent encoder | `pre_rnn_norm`, `pre_rnn_activation`, `rnn_type`, `rnn_hidden`, `rnn_layers`, `rnn_bidirectional`, `rnn_reverse`, `rnn_summary`, `rnn_dropout` |
| CNN-only readout | `global_pool`, `global_pool_size` |
| Prediction head | `head_type`, `head_hidden`, `head_norm`, `head_norm_position`, `head_activation`, `head_dropout`, `head_norm_eps`, `head_bn_momentum` |

`block_type` accepts `conv` or `convnext`. Standard convolution blocks honor the
tower normalization and activation fields. ConvNeXt blocks use their defining
depthwise convolution, channel-last LayerNorm, GELU pointwise MLP, optional
LayerScale, and stochastic depth. `rnn_type` accepts `none`, `gru`, or `lstm`;
`rnn_summary` accepts `last`, `mean`, or `max`; and `global_pool` accepts `avg`,
`max`, or `avgmax`.

This first implementation intentionally treats the complete fixed-width tensor
as valid sequence. It does not infer padding lengths, mask convolutions, or pack
recurrent inputs.

## Saluki–Simple-CNN Hybrid Model

`saluki_cnn_hybrid` trains an exact Saluki feature encoder and an exact
SmallCNN feature encoder in parallel. Both branches receive the same
six-channel tensor and the same stochastic right shift. The Saluki branch
contributes its final reversed-GRU state; the CNN branch contributes the
global-max-plus-global-mean vector that normally enters `SmallCNN.head`.

The default embeddings have widths 64 and 128, respectively. They are
concatenated in Saluki-then-CNN order into a 192-wide vector and passed through
a Saluki-style pre-normalized head. The complete default is selected with:

```json
{
  "model": {
    "name": "saluki_cnn_hybrid",
    "params": {}
  }
}
```

Inspect every resolved nested default with:

```bash
transcriptml models show saluki_cnn_hybrid --json
```

The model accepts independent partial dictionaries for the two encoders and
for fusion. Missing values inside each section retain their defaults:

```json
{
  "model": {
    "name": "saluki_cnn_hybrid",
    "params": {
      "saluki": {
        "filters": 32,
        "dropout": 0.2
      },
      "cnn": {
        "n_filters": 96,
        "n_layers": 3
      },
      "fusion": {
        "head_hidden": 128
      }
    }
  }
}
```

`saluki.seq_depth` and `cnn.in_ch` must match because both branches consume the
same input; both default to six. `saluki.augment_shift` is sampled once per
training forward pass and applied before branching, so the two embeddings
always describe the same augmented sequence. Evaluation does not shift either
branch.

### Projected concatenation

Raw concatenation preserves both native embeddings. To learn independent
branch projections before concatenation, select `project_concat`:

```json
{
  "model": {
    "name": "saluki_cnn_hybrid",
    "params": {
      "fusion": {
        "mode": "project_concat",
        "saluki_projection_dim": 64,
        "cnn_projection_dim": 96,
        "projection_norm": "layer",
        "projection_activation": "gelu",
        "projection_dropout": 0.1
      }
    }
  }
}
```

Each projection applies `Linear → normalization → activation → dropout` and
the projected vectors are concatenated in the same Saluki-then-CNN order.
Projection normalization accepts `none`, `batch`, or `layer`; activation
accepts `none`, `relu`, `gelu`, or `silu`.

### Hybrid head normalization

The default final head matches Saluki's
`Normalization → ReLU → Linear → Dropout → Normalization → ReLU → Linear`
order and uses BatchNorm. Select LayerNorm in the nested fusion section for a
head that can train on singleton batches:

```json
{
  "model": {
    "name": "saluki_cnn_hybrid",
    "params": {
      "fusion": {
        "head_norm": "layer"
      }
    }
  }
}
```

The top-level training option `head_layernorm` remains specific to
`saluki_exact`; it does not alter the hybrid model.

| Section | Parameters |
| --- | --- |
| `saluki` | `seq_depth`, `filters`, `kernel_size`, `num_layers`, `pooling`, `dropout`, `augment_shift`, `ln_epsilon` |
| `cnn` | `in_ch`, `n_filters`, `kernel_size`, `n_layers`, `dropout` |
| `fusion` | `mode`, `saluki_projection_dim`, `cnn_projection_dim`, `projection_norm`, `projection_activation`, `projection_dropout`, `head_hidden`, `head_norm`, `head_activation`, `head_dropout`, `head_norm_eps`, `head_bn_momentum`, `output_dim`, `weight_init` |

The prediction head and optional projections use Saluki truncated-He
initialization by default. Set `fusion.weight_init` to `pytorch` to retain the
standard PyTorch initialization for those new layers. Each feature branch
always retains the initialization of its standalone architecture.

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

### Balanced chromosome CV plans

Create one content-hashed plan from a dataset's chromosome grouping metadata:

```bash
transcriptml cv create-chromosome-plan \
  --dataset data/rbpnet \
  --group-col group_chromosome \
  --n-folds 5 \
  --output cv/cv5.json
```

The JSON records the grouping column, example count on every chromosome,
chromosome membership and total examples for every fold group, algorithm and
tie-breaking rules, format and TranscriptML versions, and a SHA-256 `plan_id`.
Generation sorts chromosomes from largest to smallest and assigns each to the
group with the smallest current example count; ties use chromosome name and
then fold index deterministically. This balances examples rather than numbers
of chromosomes. Plans require at least three folds and at least one chromosome
per fold. The plan path and validated `plan_id` are recorded in training
summaries and checkpoints.

For run `k`, test is group `k`, validation is `(k+1) mod N`, and training is all
remaining groups. Inspect or materialize one resolution with:

```bash
transcriptml cv resolve-plan \
  --dataset data/rbpnet --cv-plan cv/cv5.json --fold 0 \
  --output cv/fold0_splits.json
```

Training accepts overrides suitable for a Slurm job array:

```bash
transcriptml train configs/rbpnet/train_config.json \
  --dataset data/rbpnet \
  --cv-plan cv/cv5.json \
  --fold "${SLURM_ARRAY_TASK_ID}" \
  --output-dir "cv/fold${SLURM_ARRAY_TASK_ID}/model"
```

Resolution rejects missing chromosomes, new chromosomes, or changed example
counts rather than silently applying an obsolete plan.

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
