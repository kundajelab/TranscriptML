# Installation

TranscriptML requires Python 3.10 or newer. From a cloned repository, the core
install is enough for MPRA input building, table-based Saluki input building,
training, evaluation, single-nucleotide ISM, motif analyses, and NumPy codon-ISM
output:

```bash
git clone https://github.com/kundajelab/TranscriptML.git
cd TranscriptML
python -m pip install -e .
```

Install an optional extra only when your workflow needs it:

| Use case | Install |
| --- | --- |
| Build Saluki input from a large GTF and indexed genome FASTA | `python -m pip install -e ".[genomics]"` |
| Write codon-ISM tables as Parquet or Arrow | `python -m pip install -e ".[arrow]"` |
| Summarize and plot codon-ISM tables | `python -m pip install -e ".[analysis]"` |
| Run the test suite | `python -m pip install -e ".[dev]"` |

The `genomics` extra installs `pyfaidx` for indexed, memory-efficient FASTA
access. `build-saluki-gtf` has an in-memory fallback, so this extra is optional
for small FASTA files but recommended for whole genomes. The `analysis` extra
already includes the Arrow dependencies.

Extras can be combined. A full analysis and development install is:

```bash
python -m pip install -e ".[genomics,analysis,dev]"
```

The core install includes PyTorch. On a GPU system, install the PyTorch build
that matches the local CUDA driver first, following the PyTorch installation
instructions, and then install TranscriptML.
