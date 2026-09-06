import random

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from molgap.pcqm_geometry_scratch import (
    ScratchConfig, capture_rng, restore_rng, make_matched_model, forward, validate_resume,
)
from molgap.pcqm_wedge import WedgeData, directed_nonbacktracking_wedges


def test_rng_roundtrip_and_changed_contract_rejected():
    state = capture_rng()
    expected = (random.random(), np.random.random(), torch.rand(3))
    restore_rng(state)
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert expected[0:2] == actual[0:2]
    torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)
    validate_resume({"identity": {"arm": "triangle"}}, {"arm": "triangle"})
    with pytest.raises(RuntimeError, match="identity"):
        validate_resume({"identity": {"arm": "triangle"}}, {"arm": "geometry"})


def test_scratch_pair_has_identical_common_state_and_initial_function():
    triangle = make_matched_model("triangle", 42).eval()
    geometry = make_matched_model("geometry", 42).eval()
    for name, value in triangle.state_dict().items():
        torch.testing.assert_close(value, geometry.state_dict()[name], rtol=0, atol=0)
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    wedge = directed_nonbacktracking_wedges(edges)
    graph = WedgeData(x=torch.zeros(3, 9, dtype=torch.long), edge_index=edges,
                     edge_attr=torch.zeros(4, 3, dtype=torch.long), random_walk_pe=torch.zeros(3, 16),
                     wedge_edge_ids=wedge, edge_distance=torch.ones(4, 1),
                     wedge_angle_cos=torch.zeros(len(wedge), 1), geometry_valid=torch.tensor([True]))
    batch = Batch.from_data_list([graph, graph])
    with torch.no_grad():
        torch.testing.assert_close(forward(triangle, batch, "triangle"), forward(geometry, batch, "geometry"), atol=2e-6, rtol=2e-6)
    loss = forward(geometry.train(), batch, "geometry").sum()
    loss.backward()
    assert geometry.distance_initial.weight.grad is not None
    assert torch.isfinite(geometry.distance_initial.weight.grad).all()
    assert ScratchConfig().precision == "fp32_tf32_off"


def test_optimizer_scheduler_resume_matches_next_step(tmp_path):
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12, eta_min=1e-6)

    def step(m, opt, sched):
        opt.zero_grad()
        m(torch.rand(4, 3)).sum().backward()
        opt.step()
        sched.step()

    step(model, optimizer, scheduler)
    path = tmp_path / "last.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "rng": capture_rng()}, path)
    step(model, optimizer, scheduler)
    restored = torch.nn.Linear(3, 1)
    opt = torch.optim.AdamW(restored.parameters(), lr=4e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=12, eta_min=1e-6)
    saved = torch.load(path, weights_only=False)
    restored.load_state_dict(saved["model"])
    opt.load_state_dict(saved["optimizer"])
    sched.load_state_dict(saved["scheduler"])
    restore_rng(saved["rng"])
    step(restored, opt, sched)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], atol=0, rtol=0)
    assert scheduler.get_last_lr() == sched.get_last_lr()
