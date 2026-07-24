import pytest
import torch

from molgap.qm9_fusion import _aligned_fusion_payloads


def _payload(target):
    return {
        role: {
            "embeddings": torch.ones(1, 2),
            "predictions": torch.zeros(1, 3),
            "targets": torch.tensor([target]),
            "source_idx": torch.tensor([7]),
        }
        for role in ("train", "validation", "test")
    }


def test_fusion_alignment_rejects_mismatched_targets(tmp_path):
    left = tmp_path / "left.pt"
    right = tmp_path / "right.pt"
    torch.save(_payload([1.0, 2.0, 3.0]), left)
    torch.save(_payload([1.0, 2.0, 4.0]), right)

    with pytest.raises(ValueError, match="Mismatched fusion targets"):
        _aligned_fusion_payloads(left, right)
