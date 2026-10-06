"""Synthetic end-to-end derivatives and frozen scope; no retained models/data."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from molgap.gptrans import GPTransBlock
from molgap.gptrans_pair_transition import PairTransitionBlock
from molgap.gptrans_capacity import BondLocalBlock
from molgap.gptrans_bottleneck import derivative_probe, portability_indices, validate_release_contract


class TinyGraphToken(nn.Module):
    def __init__(self, local=False):
        super().__init__()
        self.blocks = nn.ModuleList([GPTransBlock(16, 32, 4, 0, 0, 1) for _ in range(2)])
        wrapper = BondLocalBlock if local else PairTransitionBlock
        self.blocks = nn.ModuleList([wrapper(block, channels=16) if local else wrapper(block) for block in self.blocks])
        self.readout = nn.Linear(48, 1)

    def forward(self, node, pair, mask):
        edges = (torch.tensor([0,0,1,1]), torch.tensor([1,2,1,2]), torch.tensor([2,1,2,1]))
        for block in self.blocks:
            node, pair = block(node, pair, mask, edges) if isinstance(block, BondLocalBlock) else block(node, pair, mask)
        return self.readout(torch.cat((node[:,0], pair[:,:,0,0]), -1)).reshape(-1)


@pytest.mark.parametrize("local", [False, True])
def test_final_gap_loss_reachability_and_nonmutation(local):
    torch.manual_seed(42)
    model = TinyGraphToken(local=local).eval()
    node = torch.randn(2, 4, 16, requires_grad=True)
    pair = torch.randn(2, 32, 4, 4, requires_grad=True)
    mask = torch.zeros(2,1,1,4, dtype=torch.bool)
    before = {k:v.clone() for k,v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    probes = derivative_probe(model, lambda:model(node,pair,mask), torch.tensor([1.,2.]))
    assert probes[0]["loss_gradient_connected"]
    assert probes[1]["loss_gradient_connected"] is local
    assert probes[1]["branch_parameter_gradient_l2"] == 0 if not local else probes[1]["branch_parameter_gradient_l2"] > 0
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    assert torch.equal(rng, torch.get_rng_state())
    assert all(p.grad is None for p in model.parameters())
    assert all(not block._forward_hooks and not block._forward_pre_hooks for block in model.blocks)


def test_train_mode_probe_rejected():
    with pytest.raises(ValueError, match="eval"):
        derivative_probe(TinyGraphToken(), lambda:None, torch.zeros(2))


def test_fixed_later_cohort_is_unique_sorted_and_not_training():
    ids = portability_indices()
    assert len(ids) == len(set(ids)) == 10000
    assert ids.min() >= 0 and ids.max() < 50000
    assert (ids[1:] > ids[:-1]).all()
    assert (ids == portability_indices()).all()


def test_scope_fails_closed_and_retains_required_roles():
    root = Path(__file__).resolve().parents[1]
    contract = json.loads((root/"experiments/pcqm_gptrans_bottleneck_audit/contract.json").read_text())
    if not contract["model_assets"]:
        pytest.skip("Mechanical assets have not been frozen")
    release = {"files":{name:spec["sha256"] for name,spec in {**contract["model_assets"],**contract["reference_payloads"]}.items()}}
    inventory = {p:{"sha256":d} for p,d in contract["source_identities"].items()}
    metadata = json.loads((root/"experiments/pcqm_gptrans_bottleneck_audit/kernel-metadata.json").read_text())
    validate_release_contract(contract, release, inventory, metadata)
    for key,value in (("optimizer_steps",1), ("protected_roles_read",True), ("portability_rows",50000), ("source_identities",{})):
        bad = deepcopy(contract)
        bad[key] = value
        with pytest.raises(ValueError):
            validate_release_contract(bad, release, inventory, metadata)


def test_diagnostic_source_has_no_optimizer_or_geometry_input():
    import ast
    from molgap import gptrans_bottleneck
    tree = ast.parse(Path(gptrans_bottleneck.__file__).read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert not any(isinstance(n.func, ast.Attribute) and n.func.attr in {"step", "clip_grad_norm_", "train"} for n in calls)
    assert not any(isinstance(n, ast.Attribute) and n.attr in {"pos", "edge_distance", "wedge_angle_cos"} for n in ast.walk(tree))
