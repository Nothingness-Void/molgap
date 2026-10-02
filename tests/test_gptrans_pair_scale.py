"""Synthetic recurrence checks; no molecular model inference or training."""
import inspect
import math
import pytest
import torch
from molgap.gptrans_pair_scale import pair_residual, apply_pair_depth_scale, DepthScaledPairBlock
from molgap.gptrans import GPTransBlock
from molgap.pcqm_gptrans_v4 import _ema_decay, _scientific_fields


def test_scale_preserves_anchor_and_gradients():
    previous = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    delta = torch.ones_like(previous, requires_grad=True)
    result = pair_residual(previous, previous + delta, 12)
    torch.testing.assert_close(result - previous, delta / math.sqrt(12))
    result.sum().backward()
    torch.testing.assert_close(delta.grad, torch.full_like(delta, 1 / math.sqrt(12)))
    assert torch.equal(pair_residual(previous, previous, 12), previous)
    for depth in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            pair_residual(previous, previous, depth)


def test_existing_tensors_and_rng_are_untouched():
    # Tiny blocks only; deliberately no forward or molecular model factory.
    class Toy:
        blocks = [GPTransBlock(8, 2, 2, 0, 0, 1) for _ in range(12)]
    model = Toy()
    tensors = [{k: v.clone() for k, v in b.state_dict().items()} for b in model.blocks]
    rng = torch.get_rng_state().clone()
    apply_pair_depth_scale(model)
    assert torch.equal(rng, torch.get_rng_state())
    for block, initial in zip(model.blocks, tensors):
        assert type(block) is DepthScaledPairBlock
        assert initial.keys() == block.state_dict().keys()
        assert all(torch.equal(v, block.state_dict()[k]) for k, v in initial.items())
    with pytest.raises(ValueError):
        apply_pair_depth_scale(model)


def test_new_arm_changes_no_optimization_fields():
    assert _ema_decay("degree_pair_depth_scale_ema999") == .999
    assert _scientific_fields("degree_pair_depth_scale_ema999") == _scientific_fields("degree_scale_ema999")
    source = inspect.getsource(DepthScaledPairBlock.forward)
    assert "super().forward" in source and "key_padding_mask.transpose" in source
