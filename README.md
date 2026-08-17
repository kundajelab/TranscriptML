# TranscriptML

TranscriptML is a toolkit for training, evaluating, and interpreting RNA
sequence-to-function models. It provides command-line tools and reusable Python
APIs for preparing sequence datasets, training models, evaluating held-out
predictions, and investigating learned sequence features with analyses such as
in silico mutagenesis, transcript-region and exon-junction ablation, motif
ablation, context scans, etc.

TranscriptML currently supports three main workflows:

- **Saluki** predicts transcriptome-wide RNA stability from transcript sequence,
  coding-frame annotations, and splice sites.
- **MPRA-LegNet** models MPRA measurements from variable sequence inserts and
  supports targets such as RNA stability, translation, protein
  output, etc.
- **RBPNet/eCLIP** converts FASTA/GTF/BAM inputs into a canonical
  mature-transcript or full-gene coordinate experiment, descriptive windows, explicit selection
  manifests, and memory-mappable model-ready NumPy bundles, then trains a
  sequence-only target/control profile model with an optional replicate-aware
  enrichment likelihood. **This entire workflow is experimental:** it has been
  minimally tested and has only been confirmed to preprocess data successfully
  and train reasonable models on PUM2 eCLIP data. It needs substantially more
  validation than the other TranscriptML workflows.

In the future, I plan to also support [RiboNN](https://www.nature.com/articles/s41587-025-02712-x) modeling of translation efficiency measurements
and extend [RBPNet](https://link.springer.com/article/10.1186/s13059-023-03015-7) interpretation and model variants.

## Installation

TranscriptML requires Python 3.10 or newer and PyTorch. Install the appropriate
PyTorch build for your system using the [official PyTorch installation
guide](https://pytorch.org/get-started/locally/), then install TranscriptML from
source:

```bash
git clone https://github.com/kundajelab/TranscriptML.git
cd TranscriptML
python -m pip install -e .
```

Optional dependencies and development installation instructions are described
in the [installation guide](https://kundajelab.github.io/TranscriptML/installation.html).

## Documentation

Full documentation, including usage guides and the Python API reference, is
available at <https://kundajelab.github.io/TranscriptML/>.

This package is under active development, and as such I am actively working to expand
and evolve TranscriptML's core functionalities and documentation.

## Citation

If you use either implemented model, please cite the corresponding publication:

- **MPRA-LegNet:** Agarwal, V., Inoue, F., Schubach, M. *et al.* Massively
  parallel characterization of transcriptional regulatory elements. *Nature*
  **639**, 411–420 (2025). [doi:10.1038/s41586-024-08430-9](https://doi.org/10.1038/s41586-024-08430-9)
- **Saluki:** Agarwal, V. & Kelley, D. R. The genetic and biochemical
  determinants of mRNA degradation rates in mammals. *Genome Biology* **23**,
  245 (2022). [doi:10.1186/s13059-022-02811-x](https://doi.org/10.1186/s13059-022-02811-x)
