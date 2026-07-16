# Installation

TranscriptML requires Python 3.10 or newer and makes extensive use of PyTorch.
The suggested workflow is to create a basic conda environment, install the
correct PyTorch build for your machine, and then install TranscriptML:

```bash
# 1. Create and activate an environment.
conda create -n transcript-ml python=3.12 pip
conda activate transcript-ml

# 2. Install PyTorch using the command from:
# https://pytorch.org/get-started/locally/
# 
# On the PyTorch website, choose:
#   Package: Pip
#   Language: Python
#   Compute Platform: your CUDA version, or CPU if needed
#
# You only need torch. torchvision and torchaudio are not required by TranscriptML.

# 3. Confirm PyTorch installation.
python - <<'PY'
import torch
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA build:", torch.version.cuda)
PY

# 4. Install TranscriptML
git clone https://github.com/kundajelab/TranscriptML.git
cd TranscriptML
python -m pip install -e .
```

If you will not use GPUs, pre-installing PyTorch this way is less important.
However, model training and most interpretation analyses are much faster on a
GPU, so a GPU-enabled PyTorch install is strongly recommended for most
TranscriptML use.

Optional extras are available for a few heavier workflows:

| Use case | Install |
| --- | --- |
| Write codon-ISM tables as Parquet or Arrow | `python -m pip install -e ".[arrow]"` |
| Summarize and plot codon-ISM tables | `python -m pip install -e ".[analysis]"` |
| Run the test suite | `python -m pip install -e ".[dev]"` |

Extras can be combined. The `analysis` extra already includes `pyarrow`, so you
do not need to install both `analysis` and `arrow`. A full analysis and
development install is:

```bash
python -m pip install -e ".[analysis,dev]"
```
