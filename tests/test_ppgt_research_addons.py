"""Synthetic CPU FP32 contracts; execution belongs to the coordinator's Luna.

These fixtures establish neither native equivalence nor scientific improvement.
"""
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace

import pytest
import torch
from torch_geometric.data import Batch

from modern2d_fixtures import graph
from molgap.family_training_core import normalized_gap_update
from molgap.model_substrate import ModelSubstrate, ReadoutInput, SlotBinding, SlotName
from molgap.modern2d_execution_proof import bind_candidate
from molgap.ppgt_author import MODEL_VERSION, PPGTAuthor, PPGTAuthorConfig
from molgap.ppgt_research_addons import apply_addon, compact_config


ARMS = ("baseline", "mean_readout", "residual_scale", "both")


@pytest.fixture(autouse=True)
def cpu_fp32():
    threads, dtype = torch.get_num_threads(), torch.get_default_dtype()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        yield
    torch.set_default_dtype(dtype)
    torch.set_num_threads(threads)


@pytest.fixture
def batch():
    return Batch.from_data_list([graph(n, i) for i, n in enumerate((3, 2, 3, 1))])


def configured_model(variant):
    model = PPGTAuthor(compact_config())
    if variant != "baseline":
        apply_addon(model, {"variant": variant})
        model._molgap_substrate.seal()
    return model


def test_compact_config_is_typed_frozen_fresh_and_leaves_original_defaults():
    original = asdict(PPGTAuthorConfig())
    config = compact_config()
    assert type(config) is PPGTAuthorConfig
    assert config == compact_config() and config is not compact_config()
    expected = dict(hidden=192, blocks=8, heads=8, pair_dim=64,
                    pe_mlp_hidden=256, stem_blocks=2, rrwp_steps=8, spe_bases=5,
                    pooling="sum", layer_scale=1e-4, attention_dropout=0., stochastic_depth=0.)
    assert {key: asdict(config)[key] for key in expected} == expected
    with pytest.raises(FrozenInstanceError):
        config.pooling = "mean"
    assert asdict(PPGTAuthorConfig()) == original
    assert (original["blocks"], original["stem_blocks"], original["rrwp_steps"]) == (16, 8, 24)
    assert MODEL_VERSION == "2"


@pytest.mark.parametrize("variant", ARMS)
def test_factorial_changes_only_declared_state_with_same_parameter_objects(variant, batch):
    original = PPGTAuthor(compact_config()).eval()
    candidate = deepcopy(original)
    before = {key: value.detach().clone() for key, value in original.state_dict().items()}
    parameters = dict(candidate.named_parameters())
    rng = torch.get_rng_state().clone()
    if variant != "baseline":
        assert apply_addon(candidate, {"variant": variant}) is candidate
        owner = candidate._molgap_substrate
        assert owner.bindings == (type(owner.bindings[0])(
            "ppgt_research_" + variant, SlotBinding("ppgt_small", SlotName.PRE_READOUT)),)
        assert not owner.sealed
        owner.seal()
        declaration = candidate._molgap_addons["ppgt_research_" + variant]
        payload = ReadoutInput(torch.zeros(2, 192))
        assert declaration(payload) is payload
        assert not tuple(declaration.parameters()) and not tuple(declaration.buffers())
        assert not tuple(owner.named_addon_parameters())
        assert candidate._molgap_ppgt_research_variant == variant
    else:
        assert not hasattr(candidate, "_molgap_substrate")
    assert torch.equal(rng, torch.get_rng_state())
    assert parameters.keys() == dict(candidate.named_parameters()).keys()
    assert all(parameters[key] is value for key, value in candidate.named_parameters())
    assert sum(p.numel() for p in candidate.parameters()) == sum(p.numel() for p in original.parameters())
    assert candidate.state_dict().keys() == before.keys()
    changed = set()
    for key, value in candidate.state_dict().items():
        assert torch.equal(original.state_dict()[key], before[key]), key
        if variant in {"residual_scale", "both"} and key.endswith((".attention_scale", ".ffn_scale")):
            assert torch.equal(value, torch.full_like(value, 1e-2)), key
            changed.add(key)
        else:
            assert torch.equal(value, before[key]), key
    assert len(changed) == (16 if variant in {"residual_scale", "both"} else 0)
    expected = replace(original.config, pooling="mean") if variant in {"mean_readout", "both"} else original.config
    assert candidate.config == expected
    assert candidate.forward.__func__ is original.forward.__func__
    assert candidate.config.layer_scale == 1e-4
    if variant == "baseline":
        with torch.no_grad():
            torch.testing.assert_close(candidate(batch), original(batch), rtol=0, atol=0)


@pytest.mark.parametrize("variant", ["mean_readout", "both"])
@pytest.mark.parametrize("bucketed", [False, True])
def test_mean_pooling_occurs_once_before_head_for_single_graph(variant, bucketed):
    model = configured_model(variant).eval()
    if bucketed:
        bind_candidate(model, "ppgt_small")
    nodes, pooled = [], []
    norm_hook = model.post_backbone_norm.register_forward_hook(
        lambda module, args, output: nodes.append(output.detach().clone()))
    head_hook = model.head.register_forward_pre_hook(
        lambda module, args: pooled.append(args[0].detach().clone()))
    try:
        with torch.no_grad():
            prediction = model(Batch.from_data_list([graph(3)]))
            expected_embedding = nodes[0].mean(1 if bucketed else 0)
            torch.testing.assert_close(pooled[0], expected_embedding, rtol=0, atol=0)
            # The nonlinear head acts after pooling; averaging predictions is different.
            wanted = expected_embedding
            for layer in model.head:
                wanted = layer(wanted)
            torch.testing.assert_close(prediction, wanted.reshape(1), rtol=0, atol=0)
    finally:
        norm_hook.remove()
        head_hook.remove()


@pytest.mark.parametrize("variant", ARMS)
def test_bucketed_forward_honors_config_and_original_graph_order(variant, batch):
    model = configured_model(variant).eval()
    with torch.no_grad():
        expected = model(batch)
        bind_candidate(model, "ppgt_small")
        actual = model(batch)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=2e-5)
    assert actual.shape == (4,) and actual.dtype == torch.float32
    assert actual.device.type == "cpu"
    if variant != "baseline":
        assert model._molgap_substrate.sealed and not model._molgap_substrate._active


@pytest.mark.parametrize("variant", ARMS)
@pytest.mark.parametrize("bucketed", [False, True])
def test_call_backward_and_one_shared_optimizer_update(variant, bucketed, batch):
    model = configured_model(variant).train()
    if bucketed:
        bind_candidate(model, "ppgt_small")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    before = model.head[-1].bias.detach().clone()
    loss = normalized_gap_update(model, optimizer, batch, 0., 1.,
        forward=lambda model, batch: model(batch), clip_norm=5., check_finite=True,
        cast_target_float=True)
    assert torch.isfinite(torch.as_tensor(loss)).all()
    assert not torch.equal(before, model.head[-1].bias)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(int(state["step"]) == 1 for state in optimizer.state.values())
    assert all(block.attention_scale.grad.abs().sum() > 0 for block in model.blocks)
    if variant != "baseline":
        assert model._molgap_substrate.sealed and not model._molgap_substrate._active


@pytest.mark.parametrize("config", [{}, {"variant": "baseline"}, {"variant": "other"},
    {"variant": None}, {"variant": []}, {"variant": "mean_readout", "pooling": "sum"}, None])
def test_invalid_variant_rejects_without_mutation(config):
    model = PPGTAuthor(compact_config())
    before = deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="variant"):
        apply_addon(model, config)
    assert not hasattr(model, "_molgap_substrate")
    assert model.config == compact_config()
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())


@pytest.mark.parametrize("variant", ARMS[1:])
def test_reapplication_rejected_without_resetting_learned_scale(variant):
    model = configured_model(variant)
    with torch.no_grad():
        model.blocks[0].attention_scale.add_(0.03)
    before = deepcopy(model.state_dict())
    for second in ARMS[1:]:
        with pytest.raises(ValueError, match="already applied"):
            apply_addon(model, {"variant": second})
        assert len(model._molgap_substrate.bindings) == 1
        assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())


def test_residual_scale_rejects_noninitial_state_and_sealed_attachment_is_atomic():
    model = PPGTAuthor(compact_config())
    with torch.no_grad():
        model.blocks[-1].ffn_scale[0] = 0.3
    with pytest.raises(ValueError, match="fresh"):
        apply_addon(model, {"variant": "both"})
    assert model.config.pooling == "sum" and not hasattr(model, "_molgap_substrate")
    assert torch.equal(model.blocks[0].attention_scale, torch.full((192,), 1e-4))
    model = PPGTAuthor(compact_config())
    ModelSubstrate(model, "ppgt_small").seal()
    before = deepcopy(model.state_dict())
    with pytest.raises(RuntimeError, match="sealed"):
        apply_addon(model, {"variant": "both"})
    assert model.config.pooling == "sum" and not hasattr(model, "_molgap_ppgt_research_variant")
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
