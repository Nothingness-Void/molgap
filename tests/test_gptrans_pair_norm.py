from __future__ import annotations

import torch
import pytest

from molgap.gptrans import GPTransBlock
from molgap.gptrans_variants import MODES, PairStateNormalizationBlock, normalize_pair


@pytest.fixture(autouse=True)
def cpu_scope():
    from molgap.fast_sandbox import _cpu_scope
    with _cpu_scope(42, 1):
        yield


def test_pair_normalization_modes_are_registered() -> None:
    assert {"pair_update_norm", "pair_post_norm"}.issubset(MODES)


@pytest.mark.parametrize("wrapped", [False, True])
def test_parameter_free_block_preserves_parameter_identity(wrapped) -> None:
    if wrapped:
        from molgap.gptrans_bond_local import make_bond_local_model
        from molgap.gptrans_bond_pair_norm import construct_family_model
        from molgap.gptrans_execution_backend import bind_model_layout, prepare_layout, prepared_forward
        from molgap.fast_sandbox import synthetic_batch
        from io import BytesIO
        reference = make_bond_local_model()
        rng = torch.get_rng_state().clone()
        candidate = construct_family_model(None, None)
        assert torch.equal(rng, torch.get_rng_state())
        candidate.load_state_dict(reference.state_dict(), strict=True)
        assert sum(p.numel() for p in candidate.parameters()) == 5871201
        assert all(torch.equal(a, b) for a, b in zip(reference.state_dict().values(), candidate.state_dict().values()))
        bind_model_layout(candidate)
        batch = prepare_layout(synthetic_batch())
        optimizer = torch.optim.AdamW(candidate.parameters(), lr=.001, foreach=True)
        before = candidate.blocks[0].core.attention.qkv.weight.detach().clone()
        prediction = prepared_forward(candidate, batch)
        prediction.abs().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in candidate.parameters())
        optimizer.step()
        assert not torch.equal(before, candidate.blocks[0].core.attention.qkv.weight)
        buffer = BytesIO()
        torch.save({"model": candidate.state_dict(), "optimizer": optimizer.state_dict()}, buffer)
        buffer.seek(0)
        restored = construct_family_model(None, None)
        restored.load_state_dict(torch.load(buffer, weights_only=True)["model"], strict=True)
        assert all(torch.equal(a, b) for a, b in zip(candidate.state_dict().values(), restored.state_dict().values()))
        return
    reference = GPTransBlock(32, 8, 4, 0.0, 0.0, 1.0)
    candidate = PairStateNormalizationBlock(
        32, 8, 4, 0.0, 0.0, 1.0, variant="pair_update_norm"
    )
    candidate.load_state_dict(reference.state_dict(), strict=True)
    assert list(reference.state_dict()) == list(candidate.state_dict())
    assert sum(p.numel() for p in reference.parameters()) == sum(
        p.numel() for p in candidate.parameters()
    )


def test_pair_normalization_is_per_pair_across_channels() -> None:
    pair = torch.arange(2 * 4 * 3 * 3, dtype=torch.float32).reshape(2, 4, 3, 3)
    normalized = normalize_pair(pair)
    channel_mean = normalized.permute(0, 2, 3, 1).mean(dim=-1)
    assert torch.allclose(channel_mean, torch.zeros_like(channel_mean), atol=1e-6)
