from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_SALUKI_LENGTH = 12288


def _csv_list(value: str | None) -> list[str] | None:
    """Parse a comma-separated CLI value into a list of strings.

    Args:
        value: Optional raw CLI string containing comma-separated values. ``None``
            is returned unchanged.
    """

    if value is None:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def _maybe_int(value: str | None) -> str | int | None:
    """Return an integer for digit-only CLI values, otherwise the original value."""

    if value is None:
        return None
    return int(value) if value.isdigit() else value


def _analysis_install_message() -> str:
    return "This command requires the analysis extra: pip install 'TranscriptML[analysis]'"


def _resolve_named_or_positional_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    *,
    command: str,
    specs: list[tuple[str, str, str, str]],
) -> dict[str, str]:
    """Resolve values from named flags or legacy positional arguments."""

    resolved: dict[str, str] = {}
    for dest, flag_dest, flag, metavar in specs:
        flagged = getattr(args, flag_dest)
        positional = getattr(args, dest)
        if flagged is not None and positional is not None and str(flagged) != str(positional):
            parser.error(f"{command} got both {flag} and positional {metavar}; use only one")
        value = flagged if flagged is not None else positional
        if value is None:
            parser.error(f"{command} requires {flag} or legacy positional {metavar}")
        resolved[dest] = value
    return resolved


def _resolve_evaluate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict[str, str]:
    """Resolve evaluate paths from named flags or legacy positional arguments."""

    return _resolve_named_or_positional_args(
        args,
        parser,
        command="evaluate",
        specs=[
            ("checkpoint", "checkpoint_flag", "--checkpoint", "CHECKPOINT"),
            ("dataset", "dataset_flag", "--dataset", "DATASET"),
            ("out_csv", "out_csv_flag", "--out-csv", "OUT_CSV"),
        ],
    )


def _resolve_interpret_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict[str, str]:
    """Resolve interpretation paths from named flags or legacy positional arguments."""

    return _resolve_named_or_positional_args(
        args,
        parser,
        command=str(args.command),
        specs=[
            ("checkpoint", "checkpoint_flag", "--checkpoint", "CHECKPOINT"),
            ("dataset", "dataset_flag", "--dataset", "DATASET"),
            ("out_dir", "out_dir_flag", "--out-dir", "OUT_DIR"),
        ],
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the TranscriptML command-line argument parser."""

    parser = argparse.ArgumentParser(prog="transcriptml")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init-run", help="Write starter run configuration files")
    p.add_argument("--workflow", required=True, choices=["saluki", "legnet"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("models", help="Inspect registered model defaults")
    model_sub = p.add_subparsers(dest="models_command", required=True)
    p_list = model_sub.add_parser("list", help="List registered models")
    p_list.add_argument("--json", action="store_true", help="Print JSON instead of a table")
    p_show = model_sub.add_parser("show", help="Show default parameters for a model")
    p_show.add_argument("model")
    p_show.add_argument("--json", action="store_true", help="Print JSON instead of a table")

    p = sub.add_parser("cv", help="Cross-validation workflow helpers")
    cv_sub = p.add_subparsers(dest="cv_command", required=True)
    p_fold = cv_sub.add_parser("prepare-fold", help="Write one fold dataset bundle and train config")
    p_fold.add_argument("--dataset", required=True, help="Original TranscriptML dataset bundle")
    p_fold.add_argument("--base-config", required=True, help="Base TranscriptML train config JSON")
    p_fold.add_argument("--cv-root", required=True, help="Directory containing fold*/ outputs")
    p_fold.add_argument("--fold", type=int, required=True, help="Zero-based fold index")
    p_fold.add_argument("--model", required=True, help="Registered model name for this fold config")
    p_fold.add_argument("--n-folds", type=int, default=10)
    p_fold.add_argument("--seed", type=int, default=42)
    p_fold.add_argument("--val-offset", type=int, default=1)

    p = sub.add_parser("build-mpra", help="Build an RNA4 MPRA dataset bundle")
    p.add_argument("table")
    p.add_argument("out_dir")
    p.add_argument("--sequence-col", required=True)
    p.add_argument("--target-col")
    p.add_argument("--id-col")
    p.add_argument("--length", type=int)
    p.add_argument("--metadata-cols")
    p.add_argument("--split-col")
    p.add_argument("--delimiter")

    p = sub.add_parser("build-saluki", help="Build a Saluki-style transcript dataset bundle")
    p.add_argument("--table", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sequence-col", required=True)
    p.add_argument("--id-col", required=True)
    p.add_argument("--target-col")
    p.add_argument("--cds-positions-col")
    p.add_argument("--splice-positions-col")
    p.add_argument("--length", type=int, default=DEFAULT_SALUKI_LENGTH)
    p.add_argument("--metadata-cols")
    p.add_argument("--split-col")
    p.add_argument("--delimiter")

    p = sub.add_parser("build-saluki-gtf", help="Build a Saluki dataset from GTF + genome FASTA")
    p.add_argument("--gtf", required=True)
    p.add_argument("--fasta", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--targets")
    p.add_argument("--target-col")
    p.add_argument("--target-id-col", default="transcript_id")
    p.add_argument("--length", type=int, default=DEFAULT_SALUKI_LENGTH)
    p.add_argument("--metadata-cols")
    p.add_argument("--split-col")
    p.add_argument("--delimiter")

    p = sub.add_parser("train", help="Train from a JSON/TOML config")
    p.add_argument("config")

    p = sub.add_parser("evaluate", help="Evaluate a checkpoint on a dataset bundle")
    p.add_argument("checkpoint", nargs="?", metavar="CHECKPOINT", help="Checkpoint path; prefer --checkpoint")
    p.add_argument("dataset", nargs="?", metavar="DATASET", help="Dataset bundle; prefer --dataset")
    p.add_argument("out_csv", nargs="?", metavar="OUT_CSV", help="Prediction CSV path; prefer --out-csv")
    p.add_argument("--checkpoint", dest="checkpoint_flag", help="Checkpoint path")
    p.add_argument("--dataset", dest="dataset_flag", help="Dataset bundle directory")
    p.add_argument("--out-csv", dest="out_csv_flag", help="Prediction CSV output path")
    p.add_argument("--split")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cpu")

    for name, help_text in [
        ("ism", "Run single-nucleotide ISM"),
        ("codon-ism", "Run CDS codon-level ISM"),
        ("motif-ablation", "Run motif ablation"),
        ("motif-context", "Run motif context scan"),
        ("epistasis", "Run pairwise motif epistasis"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("checkpoint", nargs="?", metavar="CHECKPOINT", help="Checkpoint path; prefer --checkpoint")
        p.add_argument("dataset", nargs="?", metavar="DATASET", help="Dataset bundle; prefer --dataset")
        p.add_argument("out_dir", nargs="?", metavar="OUT_DIR", help="Output directory; prefer --out-dir")
        p.add_argument("--checkpoint", dest="checkpoint_flag", help="Checkpoint path")
        p.add_argument("--dataset", dest="dataset_flag", help="Dataset bundle directory")
        p.add_argument("--out-dir", dest="out_dir_flag", help="Output directory")
        p.add_argument("--device", default="cpu")
        p.add_argument("--batch-size", type=int, default=128)
        if name not in {"ism", "codon-ism"}:
            p.add_argument("--motif", required=True)
            p.add_argument(
                "--region",
                help="Optional region filter for motif sites: 5utr, cds, or 3utr",
            )
            p.add_argument("--cds-channel", help="CDS annotation channel name or integer index")
            p.add_argument("--n-scrambles", type=int, default=10)
            p.add_argument(
                "--strategy",
                default="random_different",
                choices=["random_different", "shuffle", "dinuc_shuffle"],
            )
            p.add_argument("--seed", type=int, default=123)
        if name in {"ism", "codon-ism"}:
            p.add_argument("--mutation-batch-size", type=int, default=512)
        if name == "codon-ism":
            p.add_argument(
                "--mutation-policy",
                default="synonymous-only",
                choices=["synonymous-only", "synonymous", "all-codons", "all"],
            )
            p.add_argument("--exclude-stop-codons", action="store_true")
            p.add_argument("--cds-channel", help="CDS annotation channel name or integer index")
            p.add_argument("--position-scores", action="store_true", help="Also save max-absolute codon effects")
            p.add_argument(
                "--sequence-start",
                type=int,
                help="Inclusive transcript index for a contiguous codon-ISM slice",
            )
            p.add_argument(
                "--sequence-end",
                type=int,
                help="Exclusive transcript index for a contiguous codon-ISM slice",
            )
            p.add_argument(
                "--sequence-shard-index",
                type=int,
                help="Zero-based transcript shard index for codon ISM",
            )
            p.add_argument(
                "--sequence-shards",
                type=int,
                help="Total transcript shards for codon ISM",
            )
            p.add_argument(
                "--table-format",
                default="npz",
                choices=["csv", "npz", "parquet", "arrow"],
                help="Streaming long-form mutation table format",
            )
            p.add_argument("--rows-per-shard", type=int, default=100_000)
        if name == "motif-context":
            p.add_argument("--window-size", type=int, default=5)
            p.add_argument("--context-width", type=int)
            p.add_argument("--n-window-scrambles", type=int, default=5)
        if name == "epistasis":
            p.add_argument("--motif2")
            p.add_argument("--include-overlaps", action="store_true")
            p.add_argument("--max-pairs", type=int)

    p = sub.add_parser("plot-ism", help="Plot a single sequence from single-nucleotide ISM arrays")
    p.add_argument("--ism", type=Path, help="Path to .npy or .npz ISM array with shape (N, 4, L)")
    p.add_argument("--ism-key", help="Array key to read from --ism when it is a .npz file")
    p.add_argument("--dataset", type=Path, help="Dataset bundle directory used to default X.npy and metadata.json")
    p.add_argument("--seq-features", type=Path, help="Optional .npy or .npz sequence/features array")
    p.add_argument("--seq-features-key", help="Array key to read from --seq-features when it is a .npz file")
    p.add_argument("--seq-index", type=int, help="Zero-based sequence index to plot")
    p.add_argument("--metadata", type=Path, help="Optional metadata JSON list with one record per sequence")
    p.add_argument("--metadata-field", help="Metadata field to match for sequence lookup")
    p.add_argument("--metadata-value", help="Metadata value to match for sequence lookup")
    p.add_argument("--gene-id", help="Shortcut for --metadata-field gene_id --metadata-value VALUE")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int)
    p.add_argument("--out", type=Path, help="Output figure path")
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--show", action="store_true")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--title")
    p.add_argument("--gene-name")
    p.add_argument("--base-labels", default="A,C,G,U")
    p.add_argument("--cmap", default="RdBu_r")
    p.add_argument("--center", type=float, default=0.0)
    p.add_argument("--vmin", type=float)
    p.add_argument("--vmax", type=float)
    p.add_argument("--logo-vlim", type=float)
    p.add_argument("--logo-font", default="DejaVu Sans")
    p.add_argument("--figsize", type=float, nargs=2, metavar=("WIDTH", "HEIGHT"))
    p.add_argument("--max-xticks", type=int, default=12)
    p.add_argument("--xtick-rotation", type=int, default=0)
    p.add_argument("--pad-atol", type=float, default=0.0)
    p.add_argument("--mean-center", action="store_true")
    p.add_argument("--no-trim-padding", action="store_true")
    p.add_argument("--no-logo", action="store_true")
    p.add_argument("--no-isoform", action="store_true")
    p.add_argument("--no-cbar", action="store_true")
    p.add_argument("--no-robust", action="store_true")
    p.add_argument("--no-symmetric", action="store_true")

    p = sub.add_parser("codon-usage", help="Compute and plot CDS-position codon usage")
    p.add_argument("--data-dir", type=Path, required=True, help="Directory containing X.npy, metadata.json, schema.json")
    p.add_argument("--out-dir", type=Path, required=True, help="Directory for tables, plots, and README")
    p.add_argument("--n-bins", type=int, default=20)
    p.add_argument("--plots", choices=["all", "none"], default="all")
    p.add_argument("--dpi", type=int, default=300)

    p = sub.add_parser("summarize-codon-ism", help="Summarize codon-ISM mutation tables")
    p.add_argument("--mode", required=True, choices=["synonymous", "all-codons"])
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-bins", type=int, default=20)
    p.add_argument("--plots", choices=["all", "average-only", "folds-only", "none"], default="all")
    p.add_argument("--flip-sign", action="store_true")
    p.add_argument("--keep-input-position", action="store_true")
    p.add_argument("--dpi", type=int, default=300)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the TranscriptML command-line interface.

    Args:
        argv: Optional argument vector to parse instead of ``sys.argv``. Pass
            ``None`` to use the process command-line arguments.
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "init-run":
        from transcriptml.workflows import init_run

        out = init_run(args.workflow, args.out_dir, force=args.force)
        print(f"Wrote starter run configs to {out}")
        return
    if args.command == "models":
        from transcriptml.models.registry import list_models, model_default_params

        if args.models_command == "list":
            models = list_models()
            if args.json:
                print(json.dumps(models, indent=2))
            else:
                for name, config_cls in models.items():
                    print(f"{name}\t{config_cls}")
            return
        if args.models_command == "show":
            params = model_default_params(args.model)
            payload = {"name": args.model, "params": params}
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                print(args.model)
                for key, value in params.items():
                    print(f"{key}\t{value}")
            return
    if args.command == "cv":
        from transcriptml.workflows import prepare_cv_fold

        if args.cv_command == "prepare-fold":
            config_path = prepare_cv_fold(
                dataset=args.dataset,
                base_config=args.base_config,
                cv_root=args.cv_root,
                fold=args.fold,
                model=args.model,
                n_folds=args.n_folds,
                seed=args.seed,
                val_offset=args.val_offset,
            )
            print(config_path)
            return
    if args.command == "plot-ism":
        from transcriptml.plotting.single_nt_ism import plot_ism_from_args

        plot_ism_from_args(args)
        return
    if args.command == "codon-usage":
        from transcriptml.analysis.codon_usage import run_codon_usage_from_args

        run_codon_usage_from_args(args)
        return
    if args.command == "summarize-codon-ism":
        common = [
            "--out-dir",
            str(args.out_dir),
            "--n-bins",
            str(args.n_bins),
            "--plots",
            args.plots,
            "--dpi",
            str(args.dpi),
        ]
        if args.flip_sign:
            common.append("--flip-sign")
        if args.keep_input_position:
            common.append("--keep-input-position")
        try:
            if args.mode == "synonymous":
                from transcriptml.analysis.codon_ism_synonymous import main as summarize_synonymous

                fold_dirs = sorted(args.input_dir.glob("fold[0-9]*"))
                if not fold_dirs:
                    raise SystemExit(f"No fold directories found under {args.input_dir}")
                summarize_synonymous(["--fold-dirs", *[str(path) for path in fold_dirs], *common])
            else:
                from transcriptml.analysis.codon_ism_all import main as summarize_all

                summarize_all(["--input-dir", str(args.input_dir), *common])
        except ModuleNotFoundError as exc:
            missing = exc.name or "analysis dependency"
            raise SystemExit(f"Missing {missing}. {_analysis_install_message()}") from exc
        return
    if args.command == "build-mpra":
        from transcriptml.data.builders import build_mpra_dataset

        build_mpra_dataset(
            args.table,
            args.out_dir,
            sequence_col=args.sequence_col,
            target_col=args.target_col,
            id_col=args.id_col,
            length=args.length,
            metadata_cols=_csv_list(args.metadata_cols),
            split_col=args.split_col,
            delimiter=args.delimiter,
        )
        return
    if args.command == "build-saluki":
        from transcriptml.data.builders import build_saluki_dataset

        build_saluki_dataset(
            table_path=args.table,
            out_dir=args.out_dir,
            sequence_col=args.sequence_col,
            id_col=args.id_col,
            target_col=args.target_col,
            cds_positions_col=args.cds_positions_col,
            splice_positions_col=args.splice_positions_col,
            length=args.length,
            metadata_cols=_csv_list(args.metadata_cols),
            split_col=args.split_col,
            delimiter=args.delimiter,
        )
        return
    if args.command == "build-saluki-gtf":
        from transcriptml.data.builders import build_saluki_dataset_from_gtf

        build_saluki_dataset_from_gtf(
            gtf_path=args.gtf,
            fasta_path=args.fasta,
            out_dir=args.out_dir,
            targets_path=args.targets,
            target_col=args.target_col,
            target_id_col=args.target_id_col,
            length=args.length,
            metadata_cols=_csv_list(args.metadata_cols),
            split_col=args.split_col,
            delimiter=args.delimiter,
        )
        return
    if args.command == "train":
        from transcriptml.training.trainer import train_from_config

        train_from_config(args.config)
        return
    if args.command == "evaluate":
        from transcriptml.progress import log_progress
        from transcriptml.training.evaluation import evaluate_checkpoint

        evaluate_paths = _resolve_evaluate_args(args, parser)
        result = evaluate_checkpoint(
            evaluate_paths["checkpoint"],
            evaluate_paths["dataset"],
            evaluate_paths["out_csv"],
            split=args.split,
            batch_size=args.batch_size,
            device=args.device,
        )
        metrics = {k: v for k, v in result.items() if k not in {"predictions", "targets", "indices"}}
        summary_path = Path(evaluate_paths["out_csv"]).with_suffix(".summary.json")
        log_progress(f"evaluate: writing summary to {summary_path}")
        summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return

    from transcriptml.data.bundle import load_bundle
    from transcriptml.interpret.predictor import Predictor
    from transcriptml.progress import log_progress

    interpret_paths = _resolve_interpret_args(args, parser)
    checkpoint = interpret_paths["checkpoint"]
    dataset = interpret_paths["dataset"]
    out_dir = interpret_paths["out_dir"]

    log_progress(f"{args.command}: loading dataset {dataset}")
    bundle = load_bundle(dataset, mmap_mode="r" if args.command == "codon-ism" else None)
    log_progress(f"{args.command}: loading checkpoint {checkpoint}")
    predictor = Predictor.from_checkpoint(checkpoint, device=args.device, batch_size=args.batch_size)
    cds_channel = _maybe_int(getattr(args, "cds_channel", None))
    if args.command == "ism":
        from transcriptml.interpret.ism import compute_ism, save_ism_result

        result = compute_ism(bundle.X, predictor, mutation_batch_size=args.mutation_batch_size)
        save_ism_result(result, out_dir)
    elif args.command == "codon-ism":
        from transcriptml.interpret.codon_ism import compute_codon_ism, mutation_table_writer, save_codon_ism_result

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        if args.table_format == "csv":
            table_path = out_path / "mutations.csv"
        elif args.table_format == "parquet":
            table_path = out_path / "mutations.parquet"
        elif args.table_format == "arrow":
            table_path = out_path / "mutations.arrow"
        else:
            table_path = out_path / "mutations_npz"
        writer = mutation_table_writer(table_path, format=args.table_format, rows_per_shard=args.rows_per_shard)
        result = compute_codon_ism(
            bundle.X,
            predictor,
            schema=bundle.schema,
            cds_channel=cds_channel,
            mutation_policy=args.mutation_policy,
            include_stop_codons=not args.exclude_stop_codons,
            mutation_batch_size=args.mutation_batch_size,
            compute_position_scores=args.position_scores,
            writer=writer,
            collect=False,
            sequence_start=args.sequence_start,
            sequence_end=args.sequence_end,
            sequence_shard_index=args.sequence_shard_index,
            sequence_shards=args.sequence_shards,
        )
        save_codon_ism_result(result, out_dir, save_mutations=False)
    elif args.command == "motif-ablation":
        from transcriptml.interpret.ablation import motif_ablation, save_motif_ablation_result

        result = motif_ablation(
            bundle.X,
            predictor,
            motif=args.motif,
            n_scrambles=args.n_scrambles,
            strategy=args.strategy,
            seed=args.seed,
            region=args.region,
            schema=bundle.schema,
            cds_channel=cds_channel,
        )
        save_motif_ablation_result(result, out_dir)
    elif args.command == "motif-context":
        from transcriptml.interpret.context import motif_context_scan, save_motif_context_result

        result = motif_context_scan(
            bundle.X,
            predictor,
            motif=args.motif,
            window_size=args.window_size,
            context_width=args.context_width,
            n_motif_scrambles=args.n_scrambles,
            n_window_scrambles=args.n_window_scrambles,
            strategy=args.strategy,
            seed=args.seed,
            region=args.region,
            schema=bundle.schema,
            cds_channel=cds_channel,
        )
        save_motif_context_result(result, out_dir)
    elif args.command == "epistasis":
        from transcriptml.interpret.epistasis import motif_epistasis, save_epistasis_result

        result = motif_epistasis(
            bundle.X,
            predictor,
            motif=args.motif,
            motif2=args.motif2,
            n_scrambles=args.n_scrambles,
            strategy=args.strategy,
            seed=args.seed,
            skip_overlaps=not args.include_overlaps,
            max_pairs=args.max_pairs,
            region=args.region,
            schema=bundle.schema,
            cds_channel=cds_channel,
        )
        save_epistasis_result(result, out_dir)


if __name__ == "__main__":
    main()
