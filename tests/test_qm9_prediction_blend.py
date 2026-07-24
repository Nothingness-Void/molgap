import torch

from molgap.qm9_fusion import validation_selected_prediction_blend


def test_prediction_blend_selects_validation_weight(tmp_path):
    source_idx = torch.tensor([3, 4])
    targets = torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    target_payload = {
        role: {"source_idx": source_idx, "targets": targets}
        for role in ("validation", "test")
    }
    first = {
        f"{role}_source_idx": source_idx
        for role in ("validation", "test")
    }
    second = dict(first)
    for role in ("validation", "test"):
        first[f"{role}_predictions"] = targets + 1.0
        second[f"{role}_predictions"] = targets - 1.0
    first_path = tmp_path / "first.pt"
    second_path = tmp_path / "second.pt"
    target_path = tmp_path / "targets.pt"
    torch.save(first, first_path)
    torch.save(second, second_path)
    torch.save(target_payload, target_path)

    result = validation_selected_prediction_blend(
        first_path, second_path, target_path, grid_points=11
    )

    assert result["first_weight"] == 0.5
    assert result["test_metrics"]["blend"]["average"]["mae"] == 0.0
