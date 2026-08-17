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
region/junction ablations, motif ablations, motif context scans, motif epistasis analyses, and
Saluki-specific codon ISM. These analyses can expose learned regulatory
sequence features as well as technical artifacts in the model or assay.

RBPNet/eCLIP preprocessing, descriptive scanning, region selection,
materialized dataset construction, and structured profile/enrichment training
are supported.

.. warning::

   The entire RBPNet/eCLIP workflow is experimental, including preprocessing,
   scanning and selection, bundle construction, modeling, and evaluation. It
   has been minimally tested and has only been confirmed to preprocess data
   successfully and train reasonable models on PUM2 eCLIP data. It needs
   substantially more validation than other TranscriptML functionality.

Start here
----------

See :doc:`installation` for the smallest install that covers your use case.
The :doc:`usage` guide walks through Saluki and MPRA projects from input tables
through cross-validation and interpretation. The :doc:`training_configuration`
guide describes every Saluki, LegNet, and shared training option. Use the
:doc:`api` reference when calling TranscriptML from Python.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   usage
   rbpnet
   training_configuration
   api
