"""CPU-only contract probes; no dataset roles or optimizer training."""
import torch

from molgap.k1_portability_dual import MODES
from molgap.pcqm_k1_variants import ARCHITECTURE_CONFIGS, make_encoder


def _fixture():
    x = torch.zeros((8, 9), dtype=torch.long)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7],
        [1, 0, 2, 1, 3, 2, 5, 4, 6, 5, 7, 6],
    ])
    edge_attr = torch.zeros((edge_index.shape[1], 3), dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    rwse = torch.linspace(0.0, 1.0, 8 * 16).reshape(8, 16)
    return x, edge_index, edge_attr, batch, rwse


def test_two_independent_modes_are_registered():
    assert len(MODES) == 2
    assert all(mode in ARCHITECTURE_CONFIGS for mode in MODES)
    assert ARCHITECTURE_CONFIGS[MODES[0]]["change"] != ARCHITECTURE_CONFIGS[MODES[1]]["change"]


def test_zero_start_exact_k1_and_distinct_parameter_counts():
    arguments = _fixture()
    torch.manual_seed(42)
    baseline = make_encoder("neural_atom_k1_v4").eval()
    with torch.no_grad():
        expected = baseline(*arguments)
    counts = []
    for mode in MODES:
        torch.manual_seed(42)
        model = make_encoder(mode).eval()
        with torch.no_grad():
            actual = model(*arguments)
        assert torch.equal(expected, actual), mode
        counts.append(sum(parameter.numel() for parameter in model.parameters()))
    assert counts == [3_668_033, 3_660_545]


def test_intervention_receives_gradient_without_cross_arm_module():
    arguments = _fixture()
    for mode in MODES:
        torch.manual_seed(42)
        model = make_encoder(mode).train()
        loss = model(*arguments).square().sum()
        loss.backward()
        if mode == MODES[0]:
            assert not hasattr(model, "degree_balance")
            parameters = model.rwse_refresh.parameters()
        else:
            assert not hasattr(model, "rwse_refresh")
            parameters = [model.degree_balance]
        assert any(
            value.grad is not None and torch.isfinite(value.grad).all()
            and torch.count_nonzero(value.grad).item() > 0
            for value in parameters
        ), mode
