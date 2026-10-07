import pytest
import torch
from molgap.k1_weight_average_diagnostic import mean_parameters


def test_average_preserves_buffers_and_means_only_parameters():
    a, b = torch.nn.BatchNorm1d(3), torch.nn.BatchNorm1d(3)
    with torch.no_grad():
        a.weight.fill_(2); b.weight.fill_(6)
        a.running_mean.fill_(9); b.running_mean.fill_(-7)
        a.num_batches_tracked.fill_(17); b.num_batches_tracked.fill_(80)
    mean_parameters(a, b)
    assert torch.equal(a.weight, torch.full((3,), 4.))
    assert torch.equal(a.running_mean, torch.full((3,), 9.))
    assert int(a.num_batches_tracked) == 17


def test_incompatible_endpoints_rejected_before_any_write():
    a, b = torch.nn.Linear(2, 3), torch.nn.Linear(2, 4)
    before = a.weight.clone()
    with pytest.raises(ValueError):
        mean_parameters(a, b)
    assert torch.equal(a.weight, before)
