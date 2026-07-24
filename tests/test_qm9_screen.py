import numpy as np
import torch

from molgap.qm9_payloads import combine_embedding_payloads_on_intersection
from molgap.qm9_screen import fixed_split, target_tensor


def test_fixed_split_is_deterministic_and_disjoint():
    first = fixed_split(100, 60, 20, 20, seed=42)
    second = fixed_split(100, 60, 20, 20, seed=42)

    assert np.array_equal(first.all_indices, second.all_indices)
    assert len(set(first.train).intersection(first.validation)) == 0
    assert len(set(first.train).intersection(first.test)) == 0
    assert len(set(first.validation).intersection(first.test)) == 0


def test_qm9_target_selection_uses_electron_volt_columns():
    values = torch.arange(19, dtype=torch.float32).view(1, -1)
    selected = target_tensor({"y": values})

    assert selected.tolist() == [2.0, 3.0, 4.0]


def test_payload_intersection_builds_fair_single_and_dual_views(tmp_path):
    targets = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    primary = {
        role: {
            "embeddings": torch.tensor([[1.0, 3.0], [5.0, 7.0]]),
            "predictions": targets.clone(),
            "targets": targets.clone(),
            "source_idx": torch.tensor([10, 20]),
        }
        for role in ("train", "validation", "test")
    }
    secondary = {
        role: {
            "embeddings": torch.tensor([[9.0, 11.0], [3.0, 5.0]]),
            "predictions": torch.stack((targets[0] + 10.0, targets[1])),
            "targets": torch.stack((targets[0] + 10.0, targets[1])),
            "source_idx": torch.tensor([30, 20]),
        }
        for role in ("train", "validation", "test")
    }
    primary_path = tmp_path / "primary.pt"
    secondary_path = tmp_path / "secondary.pt"
    torch.save(primary, primary_path)
    torch.save(secondary, secondary_path)

    result = combine_embedding_payloads_on_intersection(
        primary_path, secondary_path, tmp_path / "combined"
    )
    average = torch.load(
        result["outputs"]["average"], map_location="cpu", weights_only=False
    )
    concat = torch.load(
        result["outputs"]["concat"], map_location="cpu", weights_only=False
    )

    assert result["rows"] == {"train": 1, "validation": 1, "test": 1}
    assert average["test"]["source_idx"].tolist() == [20]
    assert average["test"]["embeddings"].tolist() == [[4.0, 6.0]]
    assert concat["test"]["embeddings"].tolist() == [[5.0, 7.0, 3.0, 5.0]]
