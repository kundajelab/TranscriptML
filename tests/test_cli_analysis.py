import json

import numpy as np
import pytest

from transcriptml.cli.main import _resolve_evaluate_args, _resolve_interpret_args, build_parser, main
from transcriptml.data.bundle import DatasetBundle, save_bundle
from transcriptml.data.encoding import encode_saluki_transcript


def test_models_cli_list_and_show_json(capsys):
    main(["models", "list"])
    out = capsys.readouterr().out
    assert "saluki_exact" in out
    assert "legnet" in out

    main(["models", "show", "saluki_exact", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "saluki_exact"
    assert payload["params"]["filters"] == 64


def test_init_run_cli_writes_templates(tmp_path):
    out_dir = tmp_path / "run"
    main(["init-run", "--workflow", "saluki", "--out-dir", str(out_dir)])

    train_config = json.loads((out_dir / "train_config.json").read_text(encoding="utf-8"))
    assert train_config["model"]["name"] == "saluki_exact"
    assert train_config["split_source"] == "auto"
    assert not (out_dir / "run_config.json").exists()
    assert (out_dir / "README.md").exists()


def test_plot_ism_cli_demo_writes_png(tmp_path):
    out_path = tmp_path / "ism.png"
    main(["plot-ism", "--demo", "--out", str(out_path), "--no-logo"])
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_evaluate_cli_resolves_named_and_legacy_positional_args():
    parser = build_parser()

    named = parser.parse_args(
        [
            "evaluate",
            "--checkpoint",
            "model/best.pt",
            "--dataset",
            "data/saluki",
            "--out-csv",
            "eval/predictions.csv",
        ]
    )
    assert _resolve_evaluate_args(named, parser) == {
        "checkpoint": "model/best.pt",
        "dataset": "data/saluki",
        "out_csv": "eval/predictions.csv",
    }

    positional = parser.parse_args(["evaluate", "model/best.pt", "data/saluki", "eval/predictions.csv"])
    assert _resolve_evaluate_args(positional, parser) == {
        "checkpoint": "model/best.pt",
        "dataset": "data/saluki",
        "out_csv": "eval/predictions.csv",
    }

    mixed = parser.parse_args(["evaluate", "model/best.pt", "data/saluki", "--out-csv", "eval/predictions.csv"])
    assert _resolve_evaluate_args(mixed, parser)["out_csv"] == "eval/predictions.csv"


def test_evaluate_cli_rejects_conflicting_named_and_positional_args():
    parser = build_parser()
    args = parser.parse_args(
        [
            "evaluate",
            "old.pt",
            "data/saluki",
            "eval/predictions.csv",
            "--checkpoint",
            "new.pt",
        ]
    )

    with pytest.raises(SystemExit):
        _resolve_evaluate_args(args, parser)


def test_interpret_cli_resolves_named_and_legacy_positional_args():
    parser = build_parser()
    motif_extra = {
        "motif-ablation": ["--motif", "AUG"],
        "motif-context": ["--motif", "AUG"],
        "epistasis": ["--motif", "AUG"],
    }

    for command in ["ism", "codon-ism", "motif-ablation", "motif-context", "epistasis"]:
        named = parser.parse_args(
            [
                command,
                "--checkpoint",
                "model/best.pt",
                "--dataset",
                "data/saluki",
                "--out-dir",
                f"interpret/{command}",
                *motif_extra.get(command, []),
            ]
        )
        assert _resolve_interpret_args(named, parser) == {
            "checkpoint": "model/best.pt",
            "dataset": "data/saluki",
            "out_dir": f"interpret/{command}",
        }

        positional = parser.parse_args(
            [
                command,
                "model/best.pt",
                "data/saluki",
                f"interpret/{command}",
                *motif_extra.get(command, []),
            ]
        )
        assert _resolve_interpret_args(positional, parser) == {
            "checkpoint": "model/best.pt",
            "dataset": "data/saluki",
            "out_dir": f"interpret/{command}",
        }

        mixed = parser.parse_args(
            [
                command,
                "model/best.pt",
                "data/saluki",
                "--out-dir",
                f"interpret/{command}",
                *motif_extra.get(command, []),
            ]
        )
        assert _resolve_interpret_args(mixed, parser)["out_dir"] == f"interpret/{command}"


def test_interpret_cli_rejects_conflicting_named_and_positional_args():
    parser = build_parser()
    args = parser.parse_args(
        [
            "ism",
            "old.pt",
            "data/saluki",
            "interpret/ism",
            "--checkpoint",
            "new.pt",
        ]
    )

    with pytest.raises(SystemExit):
        _resolve_interpret_args(args, parser)


def test_codon_usage_cli_tiny_bundle(tmp_path):
    X = encode_saluki_transcript("AUGGCU", length=6, cds_positions=[0, 3])[None]
    bundle = DatasetBundle(
        X=X,
        ids=["tx1"],
        schema="saluki6",
        metadata=[{"gene_id": "G1", "cds_length": 6}],
    )
    data_dir = tmp_path / "data"
    save_bundle(bundle, data_dir)

    out_dir = tmp_path / "usage"
    main(["codon-usage", "--data-dir", str(data_dir), "--out-dir", str(out_dir), "--plots", "none"])

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_included_full_cds_transcripts"] == 1
    assert (out_dir / "tables" / "global_codon_usage.tsv").exists()
    table_text = (out_dir / "tables" / "global_codon_usage.tsv").read_text(encoding="utf-8")
    assert "AUG" in table_text
    assert "GCU" in table_text
