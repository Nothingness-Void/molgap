import pytest
import torch

from molgap.pcqm_geometry_audit import aligned_payload


def test_alignment_reorders_and_rejects_identity_errors():
    reference = {"source_idx": torch.tensor([1, 2]), "target_eV": torch.tensor([3., 4.]),
                 "prediction_eV": torch.tensor([3.1, 4.1])}
    reversed_payload = {key: value.flip(0) for key, value in reference.items()}
    aligned = aligned_payload(reversed_payload, reference)
    assert torch.equal(aligned["payload"]["source_idx"], reference["source_idx"])
    duplicate = {**reference, "source_idx": torch.tensor([1, 1])}
    with pytest.raises(RuntimeError, match="Duplicate"):
        aligned_payload(duplicate, reference)
    missing = {**reference, "source_idx": torch.tensor([1, 3])}
    with pytest.raises(RuntimeError, match="coverage"):
        aligned_payload(missing, reference)
    labels = {**reference, "target_eV": torch.tensor([3., 5.])}
    with pytest.raises(AssertionError):
        aligned_payload(labels, reference)
    nan = {**reference, "prediction_eV": torch.tensor([float("nan"), 4.])}
    with pytest.raises(RuntimeError, match="Non-finite"):
        aligned_payload(nan, reference)
