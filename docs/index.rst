TranscriptML
============

TranscriptML provides data preparation, model training, evaluation, and
interpretation tools for RNA sequence-to-function models. It is designed for
two common starting points:

* transcript-level measurements paired with annotated transcript sequences,
  modeled with Saluki; and
* MPRA measurements paired with one variable RNA insert per construct, modeled
  with LegNet.

Interpretation tools include single-nucleotide in silico mutagenesis (ISM),
motif ablations, motif context scans, motif epistasis analyses, and
Saluki-specific codon ISM. These analyses can expose learned regulatory
sequence features as well as technical artifacts in the model or assay.

RiboNN support for translation measurements and RBPNet support for RBP binding
measurements are planned but not yet implemented.

Start here
----------

See :doc:`installation` for the smallest install that covers your use case.
The :doc:`usage` guide walks through Saluki and MPRA projects from input tables
through cross-validation and interpretation. Use the :doc:`api` reference when
calling TranscriptML from Python.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   usage
   api
