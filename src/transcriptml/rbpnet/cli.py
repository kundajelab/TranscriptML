"""TranscriptML-native CLI wiring for the RBPNet data workflow."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_VALID_SAMPLE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


def add_rbpnet_parser(subparsers) -> None:
    """Add the nested ``transcriptml rbpnet`` command family."""

    root = subparsers.add_parser("rbpnet", help="Prepare eCLIP data for future RBPNet models")
    commands = root.add_subparsers(dest="rbpnet_command", required=True)

    preprocess = commands.add_parser(
        "preprocess", help="Build a canonical mature-transcript or full-gene eCLIP experiment"
    )
    preprocess.add_argument("--genome-fasta", required=True, type=Path)
    preprocess.add_argument("--gtf", required=True, type=Path, help="one-transcript-per-gene GTF")
    preprocess.add_argument(
        "--ip-bam", action="append", required=True, metavar="[LABEL=]PATH",
        help="IP BAM; repeat once per replicate",
    )
    preprocess.add_argument("--sminput-bam", required=True, metavar="[LABEL=]PATH")
    preprocess.add_argument("--output-dir", required=True, type=Path)
    preprocess.add_argument(
        "--coordinate-space",
        choices=("mature_transcript", "gene"),
        default="mature_transcript",
        help="canonical locus coordinates (default: mature_transcript)",
    )
    preprocess.add_argument(
        "--read1-rna-strand", choices=("opposite", "same", "unstranded"), default="opposite",
        help="RNA strand relative to read1 alignment (default: opposite for eCLIP)",
    )
    preprocess.add_argument("--min-mapq", type=int, default=1)
    preprocess.add_argument("--include-duplicates", action="store_true")
    preprocess.add_argument("--overwrite", action="store_true")
    preprocess.add_argument("--no-progress", action="store_true")

    scan = commands.add_parser(
        "scan-windows", help="Create descriptive transcript-window TSV and Parquet tables"
    )
    scan.add_argument("--processed-dir", required=True, type=Path)
    scan.add_argument("--window-size", type=int, default=100)
    scan.add_argument("--stride", type=int, default=50)
    scan.add_argument("--min-sminput-tpm", type=float, default=0.0)
    scan.add_argument("--pseudocount", type=float, default=1.0)
    terminal = scan.add_mutually_exclusive_group()
    terminal.add_argument(
        "--omit-incomplete-terminal-windows", dest="omit_incomplete", action="store_true",
        default=True,
    )
    terminal.add_argument(
        "--include-incomplete-terminal-windows", dest="omit_incomplete", action="store_false",
    )
    scan.add_argument("--output-prefix", required=True, type=Path)
    scan.add_argument("--overwrite", action="store_true")
    scan.add_argument("--no-progress", action="store_true")

    select = commands.add_parser(
        "select-regions", help="Select eligible loci and write a lightweight manifest"
    )
    select.add_argument("--processed-dir", required=True, type=Path)
    select.add_argument("--windows", required=True, type=Path, help="window Parquet path or prefix")
    select.add_argument("--output-prefix", required=True, type=Path)
    select.add_argument(
        "--strategy", required=True,
        choices=("original_rbpnet", "broad_coverage", "peak_gray_negative"),
    )
    select.add_argument("--original-min-pvalue", type=float, default=0.01)
    select.add_argument("--original-min-count", type=int, default=8)
    select.add_argument("--original-min-height", type=int, default=2)
    select.add_argument("--original-advance", type=int, default=50)
    select.add_argument(
        "--poisson-null", choices=("ip_locus_density", "sminput"),
        default="ip_locus_density",
        help="v1 Poisson null; sminput is an experimental library-scaled alternative",
    )
    select.add_argument(
        "--sminput-poisson-pseudocount", type=float, default=1.0,
        help="additive SMInput window-count pseudocount for the experimental null (default: 1)",
    )
    select.add_argument(
        "--min-total-count", type=int, default=None,
        help="minimum IP+SMInput count (default: 6 for broad_coverage, 8 for peak_gray_negative)",
    )
    select.add_argument(
        "--min-sminput-count", type=int, default=0,
        help="optional per-window SMInput minimum (default: 0)",
    )
    select.add_argument(
        "--min-ip-count", type=int, default=0,
        help="optional pooled/per-replicate IP minimum (default: 0)",
    )
    select.add_argument("--min-sminput-tpm", type=float, default=0.0)
    select.add_argument(
        "--replicate-mode", choices=("combined", "per_ip"), default="per_ip",
        help="broad_coverage eligibility mode (default: per_ip)",
    )
    select.add_argument("--peak-fdr", type=float, default=0.05)
    select.add_argument("--peak-min-log2-ratio", type=float, default=1.0)
    select.add_argument("--negative-fdr", type=float, default=0.05)
    select.add_argument("--negative-max-log2-ratio", type=float, default=-0.5)
    select.add_argument("--stitch-gap", type=int, default=0)
    select.add_argument("--overwrite", action="store_true")
    select.add_argument("--no-progress", action="store_true")

    bundle = commands.add_parser(
        "make-bundle", help="Materialize selected loci as NumPy RBPNet arrays"
    )
    bundle.add_argument("--processed-dir", required=True, type=Path)
    bundle.add_argument("--selection-manifest", required=True, type=Path)
    bundle.add_argument("--output-dir", required=True, type=Path)
    bundle.add_argument("--input-length", type=int, default=300)
    bundle.add_argument("--profile-length", type=int, default=300)
    bundle.add_argument("--max-jitter", type=int, default=0)
    bundle.add_argument(
        "--transcript-end-policy",
        choices=("drop", "pad", "shift_to_fit"),
        default="shift_to_fit",
        help="locus-boundary handling (default: shift_to_fit)",
    )
    bundle.add_argument("--overwrite", action="store_true")
    bundle.add_argument("--no-progress", action="store_true")


def _sample(value: str, role: str):
    from transcriptml.rbpnet.preprocessing import Sample

    if "=" in value:
        name, path_text = value.split("=", 1)
    else:
        path_text = value
        name = Path(value).name.removesuffix(".bam")
    if not _VALID_SAMPLE.fullmatch(name):
        raise ValueError(
            f"invalid sample name {name!r}; use LABEL=/path/file.bam with letters/numbers/._-"
        )
    return Sample(name=name, path=Path(path_text), role=role)


def _install_message() -> str:
    return "This command requires the RBPNet extra: pip install 'TranscriptML[rbpnet]'"


def run_rbpnet_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Dispatch one parsed RBPNet subcommand with consistent errors."""

    try:
        if args.rbpnet_command == "preprocess":
            from transcriptml.rbpnet.preprocessing import PipelineConfig, preprocess_eclip

            qc = preprocess_eclip(PipelineConfig(
                genome_fasta=args.genome_fasta,
                gtf=args.gtf,
                sminput=_sample(args.sminput_bam, "sminput"),
                ips=tuple(_sample(value, "ip") for value in args.ip_bam),
                output_dir=args.output_dir,
                coordinate_space=args.coordinate_space,
                read1_rna_strand=args.read1_rna_strand,
                min_mapq=args.min_mapq,
                exclude_duplicates=not args.include_duplicates,
                overwrite=args.overwrite,
                progress=not args.no_progress,
            ))
            result = {
                "output_dir": str(args.output_dir),
                "transcripts": qc["annotation"]["transcripts"],
                "samples": {name: values["retained"] for name, values in qc["samples"].items()},
            }
        elif args.rbpnet_command == "scan-windows":
            from transcriptml.rbpnet.windows import WindowScanConfig, scan_windows

            result = scan_windows(WindowScanConfig(
                processed_dir=args.processed_dir,
                output_prefix=args.output_prefix,
                window_size=args.window_size,
                stride=args.stride,
                min_sminput_tpm=args.min_sminput_tpm,
                pseudocount=args.pseudocount,
                omit_incomplete_terminal_windows=args.omit_incomplete,
                overwrite=args.overwrite,
                progress=not args.no_progress,
            ))
        elif args.rbpnet_command == "select-regions":
            from transcriptml.rbpnet.selection import SelectionConfig, select_regions

            result = select_regions(SelectionConfig(
                processed_dir=args.processed_dir,
                windows=args.windows,
                output_prefix=args.output_prefix,
                strategy=args.strategy,
                original_min_pvalue=args.original_min_pvalue,
                original_min_count=args.original_min_count,
                original_min_height=args.original_min_height,
                original_advance=args.original_advance,
                poisson_null=args.poisson_null,
                sminput_poisson_pseudocount=args.sminput_poisson_pseudocount,
                min_total_count=args.min_total_count,
                min_sminput_count=args.min_sminput_count,
                min_ip_count=args.min_ip_count,
                min_sminput_tpm=args.min_sminput_tpm,
                replicate_mode=args.replicate_mode,
                peak_fdr=args.peak_fdr,
                peak_min_log2_ratio=args.peak_min_log2_ratio,
                negative_fdr=args.negative_fdr,
                negative_max_log2_ratio=args.negative_max_log2_ratio,
                stitch_gap=args.stitch_gap,
                overwrite=args.overwrite,
                progress=not args.no_progress,
            ))
        else:
            from transcriptml.rbpnet.bundle import RBPNetBundleConfig, make_rbpnet_bundle

            built = make_rbpnet_bundle(RBPNetBundleConfig(
                processed_dir=args.processed_dir,
                selection_manifest=args.selection_manifest,
                output_dir=args.output_dir,
                input_length=args.input_length,
                profile_length=args.profile_length,
                max_jitter=args.max_jitter,
                transcript_end_policy=args.transcript_end_policy,
                overwrite=args.overwrite,
                progress=not args.no_progress,
            ))
            result = {
                "output_dir": str(args.output_dir),
                "n_examples": len(built.ids),
                "X_shape": list(built.X.shape),
                "ip_profiles_shape": list(built.arrays["ip_profiles"].shape),
            }
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing optional dependency {exc.name!r}. {_install_message()}") from exc
    except (FileNotFoundError, FileExistsError, KeyError, IndexError, ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
