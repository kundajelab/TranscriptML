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

RBPNet/eCLIP data
-----------------

.. automodule:: transcriptml.rbpnet.preprocessing
   :members: Sample, PipelineConfig, preprocess_eclip
   :member-order: bysource

.. automodule:: transcriptml.rbpnet.experiment
   :members: ProcessedECLIPDataset, TranscriptRecord, SampleRecord, RegionRecord, GenomicBlock
   :member-order: bysource

.. automodule:: transcriptml.rbpnet.windows
   :members: WindowScanConfig, generate_window_bounds, calculate_gc_fraction, summarize_regions, scan_windows
   :member-order: bysource

.. automodule:: transcriptml.rbpnet.selection
   :members: SelectionConfig, SelectionManifest, select_regions, load_selection_manifest
   :member-order: bysource

.. automodule:: transcriptml.rbpnet.bundle
   :members: RBPNetBundleConfig, jitter_crop_offset, make_rbpnet_bundle, load_rbpnet_bundle
   :member-order: bysource

.. automodule:: transcriptml.rbpnet.dataset
   :members: RBPNetBatch, RBPNetDataset, collate_rbpnet, deduplicate_locus_indices
   :member-order: bysource

.. automodule:: transcriptml.rbpnet.losses
   :members: RBPNetLossConfig, RBPNetLossOutput, multinomial_nll, replicate_binomial_nll, RBPNetObjective
   :member-order: bysource

.. automodule:: transcriptml.rbpnet.training
   :members: train_rbpnet_model, evaluate_rbpnet_model, write_rbpnet_predictions
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

.. automodule:: transcriptml.models.rbpnet
   :members: RBPNetConfig, RBPNetOutput, RBPNet, SamePadConv1d, SameLengthConvTranspose1d, theoretical_receptive_field
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
   :members: predict_array, evaluate_model, predict_to_csv, evaluate_checkpoint, evaluate_fold_checkpoints
   :member-order: bysource

.. automodule:: transcriptml.training.splits
   :members: random_split_indices, predefined_split_indices, group_split_indices, validate_group_disjoint, normalize_splits
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

.. automodule:: transcriptml.interpret.window_ism
   :members: WindowISMResult, generate_window_starts, compute_window_ism, save_window_ism_result
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
