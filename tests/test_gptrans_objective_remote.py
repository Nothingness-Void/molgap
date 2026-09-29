"""Opt-in remote GPU engineering checks; never part of local model execution."""
import os
import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("MOLGAP_REMOTE_OBJECTIVE_TEST") != "1" or not torch.cuda.is_available(),
    reason="requires an explicitly selected remote CUDA preflight",
)


def test_head_rng_gradients_and_export_equivalence():
    from types import SimpleNamespace
    from molgap.gptrans import OGBGPTransTiny
    from molgap.gptrans_objective import GPTransObjective, GPTransObjectiveConfig, export_gap_state_dict
    from molgap.pcqm_gptrans_v4 import _forward, ExponentialMovingAverage, _optimizer_step

    def model_factory():
        return OGBGPTransTiny(node_channels=16, pair_channels=4, num_heads=4,
                              num_layers=1, dropout=0, drop_path=0).cuda()
    model = model_factory()
    rng = torch.random.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state().clone()
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    objective = GPTransObjective(model, GPTransObjectiveConfig(auxiliary_hidden_dim=8, descriptor_weight=.1, fingerprint_weight=.1))
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(gpu_rng, torch.cuda.get_rng_state())
    for name, value in initial.items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
    batch = SimpleNamespace(x=torch.zeros(4, 9, dtype=torch.long, device="cuda"),
                            edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], device="cuda"),
                            edge_attr=torch.zeros(4, 3, dtype=torch.long, device="cuda"),
                            batch=torch.tensor([0, 0, 1, 1], device="cuda"),
                            y=torch.tensor([.1, .2], device="cuda"),
                            chemical_descriptors=torch.ones(2, 200, device="cuda"),
                            chemical_fingerprint=torch.ones(2, 512, device="cuda"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ema = ExponentialMovingAverage(model)
    _optimizer_step(model, optimizer, ema, batch, 0., 1., check_finite=True, objective=objective)
    assert any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in model._chemical_aux_head.parameters())
    assert not model.readout._forward_pre_hooks
    exported = model_factory()
    exported.load_state_dict(export_gap_state_dict(model.state_dict()), strict=True)
    model.eval(); exported.eval()
    with torch.no_grad():
        torch.testing.assert_close(_forward(model, batch), _forward(exported, batch), rtol=0, atol=0)
    assert not hasattr(exported, "_chemical_aux_head")
    restored = model_factory()
    GPTransObjective(restored, objective.config)
    restored.load_state_dict(model.state_dict(), strict=True)
    restored_ema = ExponentialMovingAverage(restored)
    restored_ema.load_state_dict(ema.state_dict())
    for key, value in ema.state_dict().items():
        torch.testing.assert_close(value, restored_ema.state_dict()[key], rtol=0, atol=0)
