"""Fixed-weight raw-author eager forward/backward on synthetic CPU graphs.

These checks grant no benchmark, initialization RNG, training or native parity.
Run only through the coordinating main agent's configured Luna leaf worker.
"""
from dataclasses import replace

import pytest
import torch
from torch_geometric.data import Batch, Data

from modern2d_fixtures import graph
from molgap.ppgt_author import PPGTAuthor, PPGTAuthorConfig
from ppgt_author_reference_harness import AuthorOracle, copy_weights


def small_config(**changes):
    return replace(PPGTAuthorConfig(hidden=16, heads=2, blocks=2, pair_dim=8,
        pe_mlp_hidden=12, stem_blocks=2, expansion=2, stem_expansion=2,
        rrwp_steps=3, spe_bases=2), **changes)


def fixed_weights(model):
    """Finite noninitial state; nonzero affine/scales avoid zero-start hiding."""
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            index = torch.arange(parameter.numel(), dtype=parameter.dtype).reshape(parameter.shape)
            phase = (sum(map(ord, name)) % 41) / 11
            value = 0.11 * torch.sin(index * 0.37 + phase)
            if name.endswith(".beta"):
                value = 0.8 + value
            if name.endswith(("attention_scale", "ffn_scale")):
                value = 0.3 + value
            parameter.copy_(value)


def varied_batch():
    a, b = graph(3, 8), graph(5, 19)
    # Unequal directional chemistry features expose the pair transpose. The
    # synthetic feature contract permits these without claiming molecular IO.
    a.edge_attr[:, 0] = torch.arange(a.edge_index.shape[1]) % 5
    b.edge_attr[:, 1] = torch.arange(b.edge_index.shape[1]) % 6
    return Batch.from_data_list([a, b])


@pytest.mark.parametrize("pooling", ["sum", "mean"])
@pytest.mark.parametrize("symmetric", [True, False])
def test_raw_author_padded_whole_forward_and_all_parameter_gradients(pooling, symmetric, record_property):
    torch.set_num_threads(1)
    config = small_config(pooling=pooling, sym_norm=symmetric)
    candidate, oracle = PPGTAuthor(config).eval(), AuthorOracle(config).eval()
    fixed_weights(candidate)
    mapped = copy_weights(candidate, oracle)
    batch = varied_batch()
    got, expected = candidate(batch), oracle(batch)
    torch.testing.assert_close(got, expected, atol=3e-6, rtol=2e-5)
    weights = torch.tensor([0.7, -0.4])
    ((got * weights).sum() + got.square().sum() * 0.03).backward()
    ((expected * weights).sum() + expected.square().sum() * 0.03).backward()
    errors = {}
    for name, parameter in candidate.named_parameters():
        wanted = mapped[name].grad
        assert parameter.grad is not None and wanted is not None, name
        assert torch.isfinite(parameter.grad).all() and torch.isfinite(wanted).all(), name
        torch.testing.assert_close(parameter.grad, wanted, atol=1e-5, rtol=8e-5, msg=name)
        errors[name] = float((parameter.grad - wanted).abs().max())
    report = {"prediction_max_abs": float((got - expected).abs().max()),
              "parameter_gradient_max_abs": max(errors.values()),
              "parameters_checked": len(errors), "scope": "SYNTHETIC_CPU_FIXED_WEIGHTS"}
    record_property("ppgt_author_parity", str(report))
    print("PPGT_AUTHOR_PARITY", report)


def test_raw_author_attention_input_and_pair_gradients():
    torch.set_num_threads(1)
    config = small_config()
    candidate, oracle = PPGTAuthor(config).eval(), AuthorOracle(config).eval()
    fixed_weights(candidate)
    copy_weights(candidate, oracle)
    local, raw = candidate.blocks[0].attention, oracle.blocks[0].attention
    x = torch.linspace(-0.7, 0.9, 3 * config.hidden).reshape(3, config.hidden).requires_grad_()
    pair = torch.linspace(-0.5, 0.8, 9 * config.pair_dim).reshape(3, 3, config.pair_dim).requires_grad_()
    x_raw, pair_raw = x.detach().clone().requires_grad_(), pair.detach().clone().requires_grad_()
    nodes = torch.arange(3)
    src, dst = torch.meshgrid(nodes, nodes, indexing="ij")
    data = Batch.from_data_list([Data(x=torch.zeros(3, config.hidden),
        edge_index=torch.stack((src.flatten(), dst.flatten())),
        edge_attr=pair_raw.transpose(0, 1).reshape(9, config.pair_dim))])
    got, expected = local(x, pair), raw(x_raw, x_raw, data)
    torch.testing.assert_close(got, expected, atol=2e-6, rtol=2e-5)
    cotangent = torch.cos(torch.arange(got.numel()).reshape(got.shape).float())
    (got * cotangent).sum().backward()
    (expected * cotangent).sum().backward()
    for observed, wanted in ((x.grad, x_raw.grad), (pair.grad, pair_raw.grad)):
        assert observed is not None and wanted is not None
        assert torch.count_nonzero(observed) > 0
        torch.testing.assert_close(observed, wanted, atol=2e-6, rtol=3e-5)


def test_raw_author_isolates_empty_edges_and_parallel_edges():
    config = small_config(pooling="mean")
    candidate, oracle = PPGTAuthor(config).eval(), AuthorOracle(config).eval()
    fixed_weights(candidate)
    copy_weights(candidate, oracle)
    isolate = graph(1, 0)
    bonded = graph(3, 1)
    bonded.edge_index = torch.cat((bonded.edge_index, bonded.edge_index[:, [0, 2]]), 1)
    bonded.edge_attr = torch.zeros(bonded.edge_index.shape[1], 3, dtype=torch.long)
    batch = Batch.from_data_list([isolate, bonded])
    torch.testing.assert_close(candidate(batch), oracle(batch), atol=3e-6, rtol=2e-5)
