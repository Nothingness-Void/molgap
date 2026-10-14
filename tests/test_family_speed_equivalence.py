"""Exact, bounded CPU regressions for operation removal, not speed claims."""
from copy import deepcopy
import os

import pytest
import torch

from molgap.gptrans import OGBGPTransTiny
from molgap.gptrans_execution_backend import prepared_shortest_path
from molgap.training_reproducibility import sum_scalar_metrics
from test_family_training_session import events, make_session


@pytest.mark.parametrize("cap", [1, 2, 20])
@pytest.mark.parametrize("maximum", [1, 7, 18])
def test_shortest_path_keeps_exact_codes_and_skips_impossible_expansions(monkeypatch, cap, maximum):
    generator = torch.Generator().manual_seed(42)
    adjacency = torch.rand(3, maximum, maximum, generator=generator) > .8
    mask = torch.ones_like(adjacency)
    if maximum > 1:
        mask[-1, -1, :] = False
        mask[-1, :, -1] = False
        adjacency[-1, -1, :] = False
        adjacency[-1, :, -1] = False
    model = type("DistanceOwner", (), {"shortest_path_cap": cap})()
    calls = []
    native = torch.bmm
    def bmm(left, right):
        calls.append(True)
        return native(left, right)
    monkeypatch.setattr(torch, "bmm", bmm)
    expected = OGBGPTransTiny._shortest_path(model, adjacency, mask)
    assert len(calls) == cap
    calls.clear()
    rng = torch.get_rng_state().clone()
    actual = prepared_shortest_path(model, adjacency, mask)
    assert len(calls) == max(0, min(cap, maximum - 1) - 1)
    assert torch.equal(expected, actual) and torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_metric_prefix_only_transfers_new_scalars_and_keeps_reporting_order(monkeypatch, dtype):
    session = make_session("neural_atom_k1", development=False, schedule_kind="cosine_step", events=events())
    values = [torch.tensor(value, dtype=dtype) for value in (1e12, .125, 3.25, 1e-6, .5)]
    calls = []
    native = torch.Tensor.cpu
    def transfer(value, *args, **kwargs):
        calls.append(value.numel())
        return native(value, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "cpu", transfer)
    try:
        session.pending_absolute = values[:2]
        first = session._cursor(session.kernel.optimizer.state_dict())
        assert calls == [2]
        first["pending_absolute"][0].zero_()
        session.pending_absolute.extend(values[2:])
        second = session._cursor(session.kernel.optimizer.state_dict())
        assert calls == [2, 3]
        assert torch.equal(torch.stack(second["pending_absolute"]), torch.stack(values))
        assert sum_scalar_metrics(session._metric_host) == sum_scalar_metrics(values)
        assert len(session._metric_host) == 5 and session.pending_absolute == []
    finally:
        session.close()


def test_layout_private_transfers_do_not_request_blocking_copies(monkeypatch):
    from torch_geometric.data import Batch, Data
    from molgap.k1_execution_layout import layout_from_cpu_batch
    batch = Batch.from_data_list([Data(x=torch.zeros(n, 9), edge_index=torch.empty(2, 0, dtype=torch.long))
        for n in (2, 3)])
    calls = []
    native = torch.Tensor.to
    def transfer(value, *args, **kwargs):
        calls.append(kwargs.get("non_blocking"))
        return native(value, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", transfer)
    layout = layout_from_cpu_batch(batch, "cpu")
    assert calls == [True, True]
    assert torch.equal(layout.flat_index, torch.tensor([0, 1, 3, 4, 5]))
    assert torch.equal(layout.valid, torch.tensor([[True, True, False], [True, True, True]]))


@pytest.mark.skipif(os.environ.get("MOLGAP_NATIVE_UNIT_TESTS") != "1" or not torch.cuda.is_available(),
                   reason="Separate explicitly enabled native unit qualification")
def test_native_layout_transport_keeps_output_gradient_optimizer_and_rng(monkeypatch):
    from copy import copy
    from torch_geometric.data import Batch, Data
    import molgap.k1_execution_layout as owner
    from molgap.k1_screen_training import _model_batch_to_device, _optimizer_step
    from molgap.qm9_neural_atom import make_encoder
    from molgap.training_reproducibility import configure_fp32_determinism, deterministic_memory_fill_context
    from molgap.v4_runtime import state_dict_sha256
    from test_family_training_session import assert_equal
    native_layout = owner.layout_from_cpu_batch
    batch = Batch.from_data_list([Data(x=torch.zeros(n, 9, dtype=torch.long),
        edge_index=torch.tensor([[i for i in range(n - 1)] + [i + 1 for i in range(n - 1)],
                                 [i + 1 for i in range(n - 1)] + [i for i in range(n - 1)]]),
        edge_attr=torch.zeros(2 * (n - 1), 3, dtype=torch.long), random_walk_pe=torch.zeros(n, 16),
        y=torch.tensor([float(n) / 10])) for n in (3, 5)])
    def blocking_layout(cpu_batch, device):
        layout = native_layout(cpu_batch, "cpu")
        return owner.DenseLayout(cpu_batch, layout.flat_index.to(device), layout.valid.to(device),
            layout.nodes, layout.graphs, layout.max_nodes)
    snapshots = []
    for layout_fn in (blocking_layout, native_layout):
        configure_fp32_determinism(42)
        model = make_encoder("neural_atom_k1").to("cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-5)
        model.train()
        inputs = {"initial_model_sha256": state_dict_sha256(model.state_dict()),
                  "starting_cpu_rng": torch.get_rng_state().clone(),
                  "starting_gpu_rng": torch.cuda.get_rng_state().clone()}
        with monkeypatch.context() as scoped:
            scoped.setattr(owner, "layout_from_cpu_batch", layout_fn)
            cpu = copy(batch)
            with deterministic_memory_fill_context(enabled=False), owner.cpu_layout_context(model, cpu, "cuda"):
                device_batch = _model_batch_to_device(cpu, "cuda", training=True)
                loss, absolute, rows = _optimizer_step(model, optimizer, device_batch, 0., 1., check_finite=True)
        snapshots.append({**inputs, "loss": loss.cpu(), "absolute": absolute.cpu(), "rows": rows,
            "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            "gradient": {key: value.grad.detach().cpu().clone() for key, value in model.named_parameters() if value.grad is not None},
            "optimizer": deepcopy(optimizer.state_dict()), "cpu_rng": torch.get_rng_state().clone(),
            "gpu_rng": torch.cuda.get_rng_state().clone()})
        del model, optimizer
    assert_equal(snapshots[0], snapshots[1])
