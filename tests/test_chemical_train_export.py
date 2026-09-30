import csv
import gzip
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from molgap import pcqm_gptrans_v4, pcqm_official_edge_state
from molgap.chemical_aux_cache import export_fixed_training_smiles
from molgap.training_reproducibility import sha256_file


def _write_archive(path, rows):
    content = io.StringIO()
    writer = csv.writer(content, lineterminator="\n")
    writer.writerow(["idx", "smiles", "homolumogap"])
    writer.writerows(rows)
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr(
            pcqm_official_edge_state.CSV_MEMBER,
            gzip.compress(content.getvalue().encode("utf-8"), mtime=0),
        )
    return path


def _install_export_mocks(
    monkeypatch,
    tmp_path,
    archive,
    *,
    train_rows,
    official_train,
    expected_archive_sha256=None,
):
    monkeypatch.setattr(pcqm_gptrans_v4, "TRAIN_ROWS", train_rows)

    graph_root = tmp_path / "accepted_graphs"
    graph_root.mkdir()
    graph_path = graph_root / "train_shard_0000.pt"
    graph_path.write_bytes(b"synthetic graph acceptance fixture")

    if expected_archive_sha256 is None:
        expected_archive_sha256 = sha256_file(archive)
    manifest_path = tmp_path / "fixed_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source": {
                    "official_archive_sha256": expected_archive_sha256,
                    "official_row_manifest_sha256": "a" * 64,
                }
            }
        ),
        encoding="utf-8",
    )

    validation_calls = []
    split_calls = []

    def validate_fixed_assets(actual_graph_root, actual_manifest_path, *, verify_content):
        validation_calls.append(
            (actual_graph_root, actual_manifest_path, verify_content)
        )
        return SimpleNamespace(train_paths=[graph_path])

    def load_official_splits(actual_archive):
        split_calls.append(actual_archive)
        return {"train": np.asarray(official_train, dtype=np.int64)}

    monkeypatch.setattr(
        pcqm_gptrans_v4, "validate_fixed_assets", validate_fixed_assets
    )
    monkeypatch.setattr(
        pcqm_official_edge_state, "load_official_splits", load_official_splits
    )
    return graph_root, manifest_path, validation_calls, split_calls, graph_path


def test_export_reads_only_fixed_train_prefix_and_not_target_column(
    monkeypatch, tmp_path
):
    archive = _write_archive(
        tmp_path / "official.zip",
        [
            (0, "CCO", "target-column-sentinel-0"),
            (1, "CCN", "target-column-sentinel-1"),
            (2, "CCC", "target-column-sentinel-2"),
            (3, "COC", "target-column-sentinel-tail-3"),
            (4, "CNC", "target-column-sentinel-tail-4"),
        ],
    )
    train_rows = 3
    graph_root, manifest_path, validation_calls, split_calls, graph_path = (
        _install_export_mocks(
            monkeypatch,
            tmp_path,
            archive,
            train_rows=train_rows,
            official_train=[0, 1, 2, 3],
        )
    )

    original_read_csv = pd.read_csv
    read_options = []

    def guarded_read_csv(*args, **kwargs):
        read_options.append(kwargs.copy())
        assert kwargs.get("usecols") == ["idx", "smiles"]
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded_read_csv)
    output = tmp_path / "export"
    result = export_fixed_training_smiles(
        archive, graph_root, manifest_path, output
    )

    lines = Path(result["rows"]).read_text(encoding="utf-8").splitlines()
    assert len(lines) == train_rows
    exported = [json.loads(line) for line in lines]
    role = json.loads(Path(result["role"]).read_text(encoding="utf-8"))

    assert exported == [
        {"source_index": 0, "smiles": "CCO"},
        {"source_index": 1, "smiles": "CCN"},
        {"source_index": 2, "smiles": "CCC"},
    ]
    assert read_options[0]["nrows"] == 3
    assert "homolumogap" not in read_options[0]["usecols"]
    assert len(split_calls) == 1
    assert validation_calls == [(graph_root, manifest_path, True)]
    assert role["source_indices"] == [0, 1, 2]
    assert role["official_train_membership_verified"] is True
    assert role["protected_target_columns_read"] is False
    assert role["train_graph_sha256"] == [sha256_file(graph_path)]
    assert role["rows_sha256"] == result["rows_sha256"]


def test_export_rejects_archive_sha_mismatch_before_reading_splits(
    monkeypatch, tmp_path
):
    archive = _write_archive(
        tmp_path / "official.zip",
        [(0, "CCO", "target-sentinel"), (1, "CCN", "target-sentinel")],
    )
    graph_root, manifest_path, _, split_calls, _ = _install_export_mocks(
        monkeypatch,
        tmp_path,
        archive,
        train_rows=2,
        official_train=[0, 1],
        expected_archive_sha256="0" * 64,
    )

    output = tmp_path / "export"
    with pytest.raises(ValueError, match="archive differs"):
        export_fixed_training_smiles(archive, graph_root, manifest_path, output)

    assert split_calls == []
    assert not output.exists()


def test_export_rejects_prefix_outside_official_train_before_csv_read(
    monkeypatch, tmp_path
):
    archive = _write_archive(
        tmp_path / "official.zip",
        [(0, "CCO", "target-sentinel"), (1, "CCN", "target-sentinel"),
         (2, "CCC", "target-sentinel")],
    )
    graph_root, manifest_path, _, _, _ = _install_export_mocks(
        monkeypatch,
        tmp_path,
        archive,
        train_rows=3,
        official_train=[0, 2],
    )

    def forbidden_read_csv(*args, **kwargs):
        raise AssertionError("CSV must not be opened for a non-train prefix")

    monkeypatch.setattr(pd, "read_csv", forbidden_read_csv)
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="not exclusively official train"):
        export_fixed_training_smiles(archive, graph_root, manifest_path, output)

    assert not output.exists()


def test_export_rejects_misaligned_official_csv_indices(monkeypatch, tmp_path):
    archive = _write_archive(
        tmp_path / "official.zip",
        [(0, "CCO", "target-sentinel"), (2, "CCN", "target-sentinel"),
         (1, "CCC", "target-sentinel")],
    )
    graph_root, manifest_path, _, _, _ = _install_export_mocks(
        monkeypatch,
        tmp_path,
        archive,
        train_rows=3,
        official_train=[0, 1, 2],
    )

    output = tmp_path / "export"
    with pytest.raises(ValueError, match="row mapping differs"):
        export_fixed_training_smiles(archive, graph_root, manifest_path, output)

    assert not output.exists()



