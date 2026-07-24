import csv

import pytest
import torch
from torch_geometric.data import Data

from scripts.phase8.training.train_encoder import _explicit_split


def graph(source_idx: int) -> Data:
    return Data(source_idx=torch.tensor([source_idx]))


def write_split(path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_idx", "split"])
        writer.writeheader()
        writer.writerows(rows)


def test_explicit_split_preserves_manifest_order(tmp_path) -> None:
    path = tmp_path / "split.csv"
    write_split(
        path,
        [
            {"source_idx": 3, "split": "train"},
            {"source_idx": 1, "split": "validation"},
            {"source_idx": 2, "split": "test"},
        ],
    )
    split, report = _explicit_split([graph(1), graph(2), graph(3)], path)

    assert [int(item.source_idx) for item in split["train"]] == [3]
    assert report["rows"] == {"train": 1, "validation": 1, "test": 1}


def test_explicit_split_rejects_missing_graph(tmp_path) -> None:
    path = tmp_path / "split.csv"
    write_split(
        path,
        [
            {"source_idx": 1, "split": "train"},
            {"source_idx": 2, "split": "validation"},
            {"source_idx": 9, "split": "test"},
        ],
    )
    with pytest.raises(ValueError, match="unavailable graph"):
        _explicit_split([graph(1), graph(2), graph(3)], path)
