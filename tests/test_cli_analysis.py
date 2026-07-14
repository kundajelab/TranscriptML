import json

import numpy as np

from transcriptml.cli.main import main
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
