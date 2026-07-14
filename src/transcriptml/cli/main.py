from __future__ import annotations

import argparse
import json
from pathlib import Path

from transcriptml.data.bundle import load_bundle
from transcriptml.data.builders import build_mpra_dataset, build_saluki_dataset, build_saluki_dataset_from_gtf
from transcriptml.data.encoding import DEFAULT_SALUKI_LENGTH
from transcriptml.interpret.ablation import motif_ablation, save_motif_ablation_result
from transcriptml.interpret.context import motif_context_scan, save_motif_context_result
from transcriptml.interpret.codon_ism import compute_codon_ism, mutation_table_writer, save_codon_ism_result
from transcriptml.interpret.epistasis import motif_epistasis, save_epistasis_result
from transcriptml.interpret.ism import compute_ism, save_ism_result
from transcriptml.interpret.predictor import Predictor
from transcriptml.models.registry import list_models, model_default_params
from transcriptml.progress import log_progress
from transcriptml.training.evaluation import evaluate_checkpoint
from transcriptml.training.trainer import train_from_config
from transcriptml.workflows import init_run


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
    p.add_argument("checkpoint")
    p.add_argument("dataset")
    p.add_argument("out_csv")
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
        p.add_argument("checkpoint")
        p.add_argument("dataset")
        p.add_argument("out_dir")
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

    args = build_parser().parse_args(argv)
    if args.command == "init-run":
        out = init_run(args.workflow, args.out_dir, force=args.force)
        print(f"Wrote starter run configs to {out}")
        return
    if args.command == "models":
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
        train_from_config(args.config)
        return
    if args.command == "evaluate":
        result = evaluate_checkpoint(
            args.checkpoint,
            args.dataset,
            args.out_csv,
            split=args.split,
            batch_size=args.batch_size,
            device=args.device,
        )
        metrics = {k: v for k, v in result.items() if k not in {"predictions", "targets", "indices"}}
        log_progress(f"evaluate: writing summary to {Path(args.out_csv).with_suffix('.summary.json')}")
        Path(args.out_csv).with_suffix(".summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return

    log_progress(f"{args.command}: loading dataset {args.dataset}")
    bundle = load_bundle(args.dataset, mmap_mode="r" if args.command == "codon-ism" else None)
    log_progress(f"{args.command}: loading checkpoint {args.checkpoint}")
    predictor = Predictor.from_checkpoint(args.checkpoint, device=args.device, batch_size=args.batch_size)
    cds_channel = _maybe_int(getattr(args, "cds_channel", None))
    if args.command == "ism":
        result = compute_ism(bundle.X, predictor, mutation_batch_size=args.mutation_batch_size)
        save_ism_result(result, args.out_dir)
    elif args.command == "codon-ism":
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.table_format == "csv":
            table_path = out_dir / "mutations.csv"
        elif args.table_format == "parquet":
            table_path = out_dir / "mutations.parquet"
        elif args.table_format == "arrow":
            table_path = out_dir / "mutations.arrow"
        else:
            table_path = out_dir / "mutations_npz"
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
        save_codon_ism_result(result, args.out_dir, save_mutations=False)
    elif args.command == "motif-ablation":
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
        save_motif_ablation_result(result, args.out_dir)
    elif args.command == "motif-context":
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
        save_motif_context_result(result, args.out_dir)
    elif args.command == "epistasis":
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
        save_epistasis_result(result, args.out_dir)


if __name__ == "__main__":
    main()
