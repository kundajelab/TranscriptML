import csv
import json

import numpy as np
import pytest

from transcriptml.cli.main import main


IDENTITY = {
    "sequence_index": [0],
    "codon_start": [3],
    "cds_codon_index": [1],
    "cds_start": [0],
    "cds_end": [8],
    "cds_length": [9],
    "cds_relative_position": [0.8],
    "reference_codon": ["GCU"],
    "alternate_codon": ["GCC"],
    "reference_amino_acid": ["A"],
    "alternate_amino_acid": ["A"],
    "synonymous": [True],
}


def _require_analysis_deps():
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("polars")
    pytest.importorskip("seaborn")
    return pa, pq


def _read_tsv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_synonymous_parquet(path, *, delta):
    pa, pq = _require_analysis_deps()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(IDENTITY)
    data.update(
        {
            "reference_prediction": [10.0],
            "mutant_prediction": [10.0 + float(delta)],
            "delta": [float(delta)],
        }
    )
    pq.write_table(pa.table(data), path)


def test_summarize_codon_ism_synonymous_tiny_parquet_and_position_mode(tmp_path):
    _require_analysis_deps()
    input_dir = tmp_path / "codon_ism"
    _write_synonymous_parquet(input_dir / "fold0" / "mutations.parquet", delta=1.0)
    _write_synonymous_parquet(input_dir / "fold1" / "mutations.parquet", delta=3.0)

    out_default = tmp_path / "summary_default"
    main(
        [
            "summarize-codon-ism",
            "--mode",
            "synonymous",
            "--input-dir",
            str(input_dir),
            "--out-dir",
            str(out_default),
            "--n-bins",
            "2",
            "--plots",
            "none",
        ]
    )

    assert (out_default / "average_mutations.parquet").exists()
    global_rows = _read_tsv(out_default / "tables" / "average" / "global_codon_effects.tsv")
    assert global_rows[0]["reference_codon"] == "GCU"
    assert float(global_rows[0]["mean_delta"]) == pytest.approx(2.0)
    default_bins = _read_tsv(out_default / "tables" / "average" / "position_bins_by_amino_acid.tsv")
    assert default_bins[0]["position_bin"] == "0"

    out_keep = tmp_path / "summary_keep"
    main(
        [
            "summarize-codon-ism",
            "--mode",
            "synonymous",
            "--input-dir",
            str(input_dir),
            "--out-dir",
            str(out_keep),
            "--n-bins",
            "2",
            "--plots",
            "none",
            "--keep-input-position",
        ]
    )
    keep_bins = _read_tsv(out_keep / "tables" / "average" / "position_bins_by_amino_acid.tsv")
    assert keep_bins[0]["position_bin"] == "1"


def _write_all_codon_npz(fold_dir, *, delta):
    mutations_dir = fold_dir / "shard0" / "mutations_npz"
    mutations_dir.mkdir(parents=True, exist_ok=True)
    arrays = dict(IDENTITY)
    arrays["delta"] = [float(delta)]
    part = mutations_dir / "part0.npz"
    np.savez(part, **{key: np.asarray(value) for key, value in arrays.items()})
    manifest = {
        "columns": list(arrays),
        "n_rows": 1,
        "parts": [{"path": part.name, "n_rows": 1}],
    }
    (mutations_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_summarize_codon_ism_all_codons_tiny_npz(tmp_path):
    _require_analysis_deps()
    input_dir = tmp_path / "all_codon_ism"
    _write_all_codon_npz(input_dir / "fold0", delta=1.0)
    _write_all_codon_npz(input_dir / "fold1", delta=3.0)

    out_dir = tmp_path / "summary"
    main(
        [
            "summarize-codon-ism",
            "--mode",
            "all-codons",
            "--input-dir",
            str(input_dir),
            "--out-dir",
            str(out_dir),
            "--n-bins",
            "2",
            "--plots",
            "none",
        ]
    )

    assert (out_dir / "README.md").exists()
    rows = _read_tsv(out_dir / "tables" / "average" / "global_reference_codon_effects.tsv")
    assert rows[0]["reference_codon"] == "GCU"
    assert float(rows[0]["mean_delta"]) == pytest.approx(2.0)
