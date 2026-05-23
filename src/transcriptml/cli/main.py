from __future__ import annotations

import argparse
import json
from pathlib import Path

from transcriptml.data.bundle import load_bundle
from transcriptml.data.builders import build_mpra_dataset, build_saluki_dataset, build_saluki_dataset_from_gtf
from transcriptml.data.encoding import DEFAULT_SALUKI_LENGTH
from transcriptml.interpret.ablation import motif_ablation, save_motif_ablation_result
from transcriptml.interpret.context import motif_context_scan, save_motif_context_result
from transcriptml.interpret.epistasis import motif_epistasis, save_epistasis_result
from transcriptml.interpret.ism import compute_ism, save_ism_result
from transcriptml.interpret.predictor import Predictor
from transcriptml.training.evaluation import evaluate_checkpoint
from transcriptml.training.trainer import train_from_config


def _csv_list(value: str | None) -> list[str] | None:
    """Parse a comma-separated CLI value into a list of strings."""

    if value is None:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    """Construct the TranscriptML command-line argument parser."""

    parser = argparse.ArgumentParser(prog="transcriptml")
    sub = parser.add_subparsers(dest="command", required=True)

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
        if name != "ism":
            p.add_argument("--motif", required=True)
            p.add_argument("--n-scrambles", type=int, default=10)
            p.add_argument(
                "--strategy",
                default="random_different",
                choices=["random_different", "shuffle", "dinuc_shuffle"],
            )
            p.add_argument("--seed", type=int, default=123)
        if name == "ism":
            p.add_argument("--mutation-batch-size", type=int, default=512)
        if name == "motif-context":
            p.add_argument("--window-size", type=int, default=5)
            p.add_argument("--context-width", type=int)
            p.add_argument("--n-window-scrambles", type=int, default=5)
        if name == "epistasis":
            p.add_argument("--motif2")
            p.add_argument("--include-overlaps", action="store_true")
            p.add_argument("--max-pairs", type=int)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the TranscriptML command-line interface."""

    args = build_parser().parse_args(argv)
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
        Path(args.out_csv).with_suffix(".summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return

    bundle = load_bundle(args.dataset)
    predictor = Predictor.from_checkpoint(args.checkpoint, device=args.device, batch_size=args.batch_size)
    if args.command == "ism":
        result = compute_ism(bundle.X, predictor, mutation_batch_size=args.mutation_batch_size)
        save_ism_result(result, args.out_dir)
    elif args.command == "motif-ablation":
        result = motif_ablation(
            bundle.X,
            predictor,
            motif=args.motif,
            n_scrambles=args.n_scrambles,
            strategy=args.strategy,
            seed=args.seed,
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
        )
        save_epistasis_result(result, args.out_dir)


if __name__ == "__main__":
    main()
