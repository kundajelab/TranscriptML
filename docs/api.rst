API reference
=============

The command-line workflows in :doc:`usage` cover most projects. The interfaces
below support custom Python workflows.

Data
----

Schemas and encoding
~~~~~~~~~~~~~~~~~~~~

.. automodule:: transcriptml.data.schemas
   :members: SequenceSchema, RNA4, SALUKI6, get_schema
   :member-order: bysource

.. automodule:: transcriptml.data.encoding
   :members: fixed_length_sequence, encode_rna_sequence, encode_sequences, encode_saluki_transcript, infer_valid_length, infer_valid_lengths, decode_rna_one_hot
   :member-order: bysource

Dataset bundles and builders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: transcriptml.data.bundle
   :members: DatasetBundle, save_bundle_metadata, save_bundle, load_bundle
   :member-order: bysource

.. automodule:: transcriptml.data.builders
   :members: build_mpra_dataset, build_saluki_dataset, build_saluki_dataset_from_gtf
   :member-order: bysource

Transcript annotation
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: transcriptml.data.genomics
   :members: GTFRecord, TranscriptFeature, TranscriptRecord, reverse_complement, parse_gtf_attributes, iter_gtf_records, load_transcript_features, extract_transcript_records, write_saluki_memmap
   :member-order: bysource

Sequence controls
~~~~~~~~~~~~~~~~~

.. automodule:: transcriptml.data.controls
   :members: SequenceControlOperation, SequenceControlConfig, normalize_sequence_control_config, apply_sequence_controls_array, apply_sequence_controls_to_bundle
   :member-order: bysource

Models
------

.. automodule:: transcriptml.models.registry
   :members: ModelConfig, list_models, model_default_params, build_model, save_checkpoint, load_checkpoint
   :member-order: bysource

.. automodule:: transcriptml.models.reproduce
   :members: SalukiExactConfig, SalukiExact
   :member-order: bysource

.. automodule:: transcriptml.models.saluki
   :members: SalukiLikeConfig, SalukiLike
   :member-order: bysource

.. automodule:: transcriptml.models.legnet
   :members: LegNetConfig, LegNet
   :member-order: bysource

.. automodule:: transcriptml.models.cnn
   :members: SmallCNNConfig, SmallCNN
   :member-order: bysource

Training and evaluation
-----------------------

.. automodule:: transcriptml.training.trainer
   :members: TrainConfig, train_model, train_from_config
   :member-order: bysource

.. automodule:: transcriptml.training.losses
   :members: LossOutput, TrainingLoss, RegressionMSELoss, WeightedMSELoss, BinomialNLLLoss, build_training_loss
   :member-order: bysource

.. automodule:: transcriptml.training.evaluation
   :members: predict_array, evaluate_model, predict_to_csv, evaluate_checkpoint
   :member-order: bysource

.. automodule:: transcriptml.training.splits
   :members: random_split_indices, predefined_split_indices, normalize_splits
   :member-order: bysource

.. automodule:: transcriptml.training.metrics
   :members: mse, pearson_corr
   :member-order: bysource

Interpretation
--------------

.. automodule:: transcriptml.interpret.predictor
   :members: Predictor, EnsemblePredictor
   :member-order: bysource

.. automodule:: transcriptml.interpret.ism
   :members: ISMResult, compute_ism, max_abs_effect_per_position, save_ism_result
   :member-order: bysource

.. automodule:: transcriptml.interpret.codon_ism
   :members: CodonISMResult, compute_codon_ism, mutation_table_writer, save_codon_ism_result
   :member-order: bysource

.. automodule:: transcriptml.interpret.ablation
   :members: MotifAblationResult, motif_ablation, save_motif_ablation_result
   :member-order: bysource

.. automodule:: transcriptml.interpret.context
   :members: MotifContextResult, motif_context_scan, save_motif_context_result
   :member-order: bysource

.. automodule:: transcriptml.interpret.epistasis
   :members: EpistasisResult, motif_epistasis, save_epistasis_result
   :member-order: bysource

.. automodule:: transcriptml.interpret.motifs
   :members: parse_motif, motif_length, base_indices_from_ohe, region_matches_motif, find_motif_starts, intervals_overlap
   :member-order: bysource

Plotting
--------

.. automodule:: transcriptml.plotting.single_nt_ism
   :members: plot_single_nt_ism
   :member-order: bysource

Run setup
---------

.. automodule:: transcriptml.workflows.init_run
   :members: init_run
   :member-order: bysource
