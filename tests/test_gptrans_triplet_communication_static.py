"""Static and synthetic tensor algebra only; no molecular model execution."""
import ast
from pathlib import Path
import pytest
import torch
from molgap.gptrans_triplet_communication import (
    MODES, PARAMETERS, INSERTIONS, HEADS, HEAD_DIM, configuration,
    projection_shapes, aggregate_direction, attention_direction, masked_weights,
)
from molgap.pcqm_gptrans_v4 import capacity_module, RELATION_DIAGNOSTIC_MODES, _ema_decay
from molgap.gptrans_author_screen import validate_arm_allocation
from molgap.experiment_spec import ADDONS


def test_inventory_registration_and_existing_owner():
    assert PARAMETERS[MODES[0]] == 5_880_961
    assert PARAMETERS[MODES[1]] == 5_889_409
    for mode in MODES:
        extra = 64 + sum(o * i + o for o, i in projection_shapes(mode).values())
        assert 5_871_201 + len(INSERTIONS) * extra == PARAMETERS[mode]
        assert ADDONS[(mode, "1")].single_addon
        assert capacity_module(mode).__name__ == "molgap.gptrans_triplet_communication"
        assert mode in RELATION_DIAGNOSTIC_MODES and _ema_decay(mode) == .999
    assert validate_arm_allocation({"arms": {m: {} for m in MODES}}) == MODES
    with pytest.raises(ValueError):
        configuration("degree_local_bond_return_ema999")


@pytest.mark.parametrize("outward", [False, True])
def test_attention_zero_matching_equals_aggregation(outward):
    generator = torch.Generator().manual_seed(17)
    value = torch.randn(2, 3, 3, HEADS, HEAD_DIM, generator=generator)
    bias = torch.randn(2, 3, 3, HEADS, generator=generator)
    gate = torch.randn(2, 3, 3, HEADS, generator=generator)
    valid = torch.ones(2, 3, 3, dtype=torch.bool)
    zero = torch.zeros_like(value)
    expected = aggregate_direction(value, bias, gate, valid, outward=outward)
    actual = attention_direction(value, zero, zero, bias, gate, valid, outward=outward)
    torch.testing.assert_close(actual, expected)
    weights = masked_weights(bias, gate, valid[..., None], dim=1 if outward else 2)
    manual = torch.zeros_like(value)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                w, v = (weights[:, k, i], value[:, k, j]) if outward else (weights[:, i, k], value[:, j, k])
                manual[:, i, j] += w[..., None] * v
    torch.testing.assert_close(expected, manual)


@pytest.mark.parametrize("outward", [False, True])
@pytest.mark.parametrize("attention", [False, True])
def test_padding_virtual_inclusion_and_permutation(outward, attention):
    generator = torch.Generator().manual_seed(18)
    value = torch.randn(1, 3, 3, HEADS, HEAD_DIM, generator=generator)
    bias = torch.randn(1, 3, 3, HEADS, generator=generator)
    gate = torch.randn(1, 3, 3, HEADS, generator=generator)
    valid = torch.ones(1, 3, 3, dtype=torch.bool)
    def evaluate(v, b, g, mask):
        return (attention_direction(v, v, v, b, g, mask, outward=outward) if attention else
                aggregate_direction(v, b, g, mask, outward=outward))
    expected = evaluate(value, bias, gate, valid)
    assert expected[:, 0, 0].abs().sum() > 0
    permutation = torch.tensor([0, 2, 1])
    permute = lambda x: x[:, permutation][:, :, permutation]
    torch.testing.assert_close(evaluate(permute(value), permute(bias), permute(gate), permute(valid)), permute(expected))
    pad_value = torch.zeros(1, 4, 4, HEADS, HEAD_DIM)
    pad_bias, pad_gate = torch.full((1, 4, 4, HEADS), 999.), torch.full((1, 4, 4, HEADS), 999.)
    pad_valid = torch.zeros(1, 4, 4, dtype=torch.bool)
    pad_value[:, :3, :3], pad_bias[:, :3, :3], pad_gate[:, :3, :3], pad_valid[:, :3, :3] = value, bias, gate, valid
    actual = evaluate(pad_value, pad_bias, pad_gate, pad_valid)
    torch.testing.assert_close(actual[:, :3, :3], expected)
    assert torch.isfinite(actual).all()
    # Downstream residual mask excludes padded output pairs for both methods.
    assert not actual.masked_fill(~pad_valid[..., None, None], 0)[:, 3].any()


def test_all_masked_softmax_finite_zero():
    actual = masked_weights(torch.zeros(1, 2, 2, 2), torch.zeros(1, 2, 2, 2),
                            torch.zeros(1, 2, 2, 1, dtype=torch.bool), dim=2)
    assert torch.isfinite(actual).all() and not actual.any()


def test_tensor_only_initial_and_virtual_connected_owner():
    tree = ast.parse(Path("src/molgap/gptrans_triplet_communication.py").read_text())
    freeze = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "freeze_initial")
    assert "construct(" not in ast.unparse(freeze) and "OGBGPTransTiny" not in ast.unparse(freeze)
    block = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TripletLocalBlock")
    source = ast.unparse(block)
    assert "pair + delta" in source and "valid_pairs(padding)" in source
    assert "dropout" not in source and "manual_seed" not in source
    acceptance = Path("src/molgap/gptrans_author_acceptance.py").read_text()
    assert "RELATION_DIAGNOSTIC_MODES" in acceptance and '"diagnostic_trace"' in acceptance
