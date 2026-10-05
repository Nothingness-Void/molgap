"""Synthetic module/constructor checks; no dataset, training or remote access."""
import pytest
import torch
from torch import nn

from molgap.gptrans import OGBGPTransTiny
from molgap.gptrans_author_variants import apply_author_variant
from molgap.gptrans_pair_transition import (
    MODE, PARAMETER_CAP, INSERTIONS, PairTransitionBlock, architecture_identity,
    construct, load_initial,
)
from molgap.pcqm_gptrans_v4 import _state_sha256, _verify_model_identity, _ema_decay
from molgap.training_reproducibility import atomic_torch_save


def test_zero_return_valid_pair_mask_and_gradients():
    block = PairTransitionBlock(nn.Identity())
    pair = torch.randn(2, 32, 5, 5)
    padding = torch.tensor([[[[False, False, False, True, True]]],
                            [[[False, False, False, False, False]]]])
    before = torch.get_rng_state().clone()
    assert torch.equal(block.transition(pair, padding), pair)
    assert torch.equal(torch.get_rng_state(), before)
    with torch.no_grad():
        block.output.weight.fill_(.01)
        block.output.bias.fill_(.1)
    result = block.transition(pair, padding)
    assert torch.equal(result[:, :, 0, :], pair[:, :, 0, :])
    assert torch.equal(result[:, :, :, 0], pair[:, :, :, 0])
    assert torch.equal(result[0, :, 3:, :], pair[0, :, 3:, :])
    assert torch.equal(result[0, :, :, 3:], pair[0, :, :, 3:])
    assert not torch.equal(result[0, :, 1:3, 1:3], pair[0, :, 1:3, 1:3])
    result.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())


def test_pair_transition_is_permutation_equivariant():
    block = PairTransitionBlock(nn.Identity())
    nn.init.normal_(block.output.weight, std=.01)
    pair = torch.randn(1, 32, 4, 4)
    padding = torch.zeros(1, 1, 1, 4, dtype=torch.bool)
    order = torch.tensor([0, 3, 1, 2])
    expected = block.transition(pair, padding)[:, :, order][:, :, :, order]
    actual = block.transition(pair[:, :, order][:, :, :, order], padding)
    assert torch.equal(actual, expected)


def test_frozen_constructor_roundtrip_preserves_core_and_rng(tmp_path):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        base = apply_author_variant(OGBGPTransTiny(), "degree_scale").state_dict()
    before = torch.get_rng_state().clone()
    model = construct(MODE, base)
    assert torch.equal(torch.get_rng_state(), before)
    assert sum(p.numel() for p in model.parameters()) == PARAMETER_CAP == 5263841
    for index, block in enumerate(model.blocks, 1):
        assert isinstance(block, PairTransitionBlock) == (index in INSERTIONS)
    for name, value in base.items():
        parts = name.split(".")
        mapped = ".".join(parts[:2] + ["core"] + parts[2:]) if parts[0] == "blocks" and int(parts[1]) + 1 in INSERTIONS else name
        assert torch.equal(model.state_dict()[mapped], value)
    digest = _state_sha256(model)
    path = tmp_path / "initial.pt"
    atomic_torch_save(path, {"format": "molgap-gptrans-pair-transition-initial-v1",
        "variant": MODE, "architecture_identity": architecture_identity(MODE),
        "parameters": PARAMETER_CAP, "state_sha256": digest, "model_state": model.state_dict()})
    restored = load_initial(MODE, path)
    assert torch.equal(torch.get_rng_state(), before)
    assert _verify_model_identity(restored) == (PARAMETER_CAP, architecture_identity(MODE))
    assert _state_sha256(restored) == digest and _ema_decay(MODE) == .999
    with pytest.raises(ValueError, match="complete frozen"):
        apply_author_variant(OGBGPTransTiny(), MODE)


def test_new_mode_uses_native_screen_not_unqualified_modular_trainer():
    from molgap.experiment_spec import ADDONS
    from molgap.gptrans_author_screen import validate_arm_allocation
    assert ADDONS[(MODE, "1")].source_module == "molgap.gptrans_pair_transition"
    assert validate_arm_allocation({"arms": {MODE: {}}, "single_arm_reason": "one hypothesis"}) == (MODE,)
    with pytest.raises(ValueError, match="Single-arm"):
        validate_arm_allocation({"arms": {MODE: {}}})
