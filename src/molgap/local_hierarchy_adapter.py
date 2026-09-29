"""Local reconstruction on existing K1/GPTrans encoders, without model forks.

Objective semantics follow pcqm_local_hierarchy.py at server commit
67a6237895ce94d502991d251f22654df3ab06c7. Lifecycle remains with the trainer.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class HierarchyConfig:
    mask_rate: float = .15
    group_weight: float = .5
    head_seed: int = 42

    def __post_init__(self):
        if type(self.mask_rate) not in (int, float) or not 0 < self.mask_rate <= 1:
            raise ValueError("mask_rate must be in (0, 1]")
        if type(self.group_weight) not in (int, float) or not math.isfinite(self.group_weight) or self.group_weight < 0:
            raise ValueError("group_weight must be finite and nonnegative")
        if type(self.head_seed) is not int or not 0 <= self.head_seed < 2**63:
            raise ValueError("invalid head seed")


def undirected_mask(edge_index, generator, rate):
    if edge_index.device.type != "cpu" or edge_index.dtype != torch.long:
        raise ValueError("Mask generation requires CPU long edge indices")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("Expected [2, E] edges")
    if edge_index.shape[1] == 0:
        return torch.zeros(0, dtype=torch.bool)
    pairs = edge_index.t().sort(dim=1).values
    unique, inverse = torch.unique(pairs, dim=0, return_inverse=True)
    chosen = torch.rand(len(unique), generator=generator) < rate
    if not bool(chosen.any()):
        chosen[inverse[0]] = True
    return chosen[inverse]


def mask_batch(batch, generator, config):
    """Clone features; preserve topology, targets and caller's original graph."""
    if batch.x.device.type != "cpu" or batch.x.shape[0] == 0:
        raise ValueError("Mask a nonempty CPU batch before transfer")
    masked = batch.clone()
    nodes = torch.rand(batch.x.shape[0], generator=generator) < config.mask_rate
    if not bool(nodes.any()):
        nodes[0] = True
    edges = undirected_mask(batch.edge_index, generator, config.mask_rate)
    masked.x[nodes] = 0
    masked.edge_attr[edges] = 0
    return masked, nodes, edges


def sparse_gptrans_states(node, pair, batch):
    """Map real nodes and directed chemical bonds, excluding virtual/pad pairs."""
    from .gptrans import OGBGPTransTiny
    if batch.batch.numel() != batch.x.shape[0]:
        raise ValueError("Batch node identity mismatch")
    group = batch.batch
    boundaries = torch.cat((group.new_zeros(1), torch.where(group[1:] != group[:-1])[0] + 1))
    if len(boundaries) != node.shape[0]:
        raise ValueError("Unordered or incomplete graph batch")
    local = torch.arange(len(group), device=group.device) - boundaries[group]
    eb, es, et = OGBGPTransTiny._local_edges(batch.edge_index, group, len(group))
    if not torch.equal(group[batch.edge_index[0]], group[batch.edge_index[1]]):
        raise ValueError("Cross-graph edge")
    return node[group, local + 1], pair[eb, :, es + 1, et + 1]


class LocalHierarchyAdapter:
    def __init__(self, model, family, config=HierarchyConfig()):
        from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
        if family not in ("k1", "gptrans_joint"):
            raise ValueError("Unsupported hierarchy family")
        self.model, self.family, self.config = model, family, config
        nd, ed = (192, 64) if family == "k1" else (256, 32)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.head_seed)
            self.heads = nn.ModuleDict({
                "atom": nn.ModuleList(nn.Linear(nd, n) for n in get_atom_feature_dims()),
                "bond": nn.ModuleList(nn.Linear(ed, n) for n in get_bond_feature_dims()),
                "group": nn.Linear(nd, 12),
            })
        parameter = next(model.parameters())
        self.heads.to(device=parameter.device, dtype=parameter.dtype)

    @contextmanager
    def capture(self):
        state, handles = {}, []
        noise = getattr(self.model, "noise_std", None)
        def save_node(_module, _args, value): state["node"] = value
        def save_edge(_module, _args, value): state["edge"] = value
        def save_dense(_module, _args, value): state["dense"] = value
        try:
            if self.family == "k1":
                handles.append(self.model.local_blocks[-1].register_forward_hook(save_node))
                handles.append(self.model.edge_updates[-1].register_forward_hook(save_edge))
                # Layer9 molecular exchange follows the last local block.
                if "9" in self.model.neural_atom_mixers:
                    handles.append(self.model.neural_atom_mixers["9"].register_forward_hook(save_node))
            else:
                if noise is None:
                    raise ValueError("GPTrans joint model has no noise configuration")
                self.model.noise_std = 0.0
                handles.append(self.model.blocks[-1].register_forward_hook(save_dense))
            yield state
        finally:
            for handle in handles:
                handle.remove()
            if noise is not None:
                self.model.noise_std = noise

    def loss(self, masked, original, node_mask, edge_mask):
        with self.capture() as state:
            self.model(masked.x, masked.edge_index, masked.edge_attr, masked.batch,
                       getattr(masked, "random_walk_pe", None))
        if self.family == "k1":
            node, edge = state["node"], state["edge"]
        else:
            node, edge = sparse_gptrans_states(*state["dense"], masked)
        return reconstruction_loss(self.heads, node, edge, original, node_mask, edge_mask, self.config)

    def checkpoint_state(self):
        return {"family": self.family, "config": asdict(self.config), "heads": self.heads.state_dict()}

    def restore(self, state):
        if state.get("family") != self.family or state.get("config") != asdict(self.config):
            raise ValueError("Hierarchy checkpoint identity changed")
        self.heads.load_state_dict(state["heads"], strict=True)


def reconstruction_loss(heads, node, edge, original, node_mask, edge_mask, config):
    if node_mask.dtype != torch.bool or node_mask.shape != (node.shape[0],) or not bool(node_mask.any()):
        raise ValueError("At least one real atom must be masked")
    if edge_mask.dtype != torch.bool or edge_mask.shape != (edge.shape[0],):
        raise ValueError("Bond mask identity mismatch")
    groups = original.functional_group_y
    if groups.shape != (node.shape[0], 12) or not bool(((groups == 0) | (groups == 1)).all()):
        raise ValueError("Invalid atom-attached functional group targets")
    atom = sum(F.cross_entropy(head(node[node_mask]), original.x[node_mask, col])
               for col, head in enumerate(heads["atom"])) / len(heads["atom"])
    bond = (sum(F.cross_entropy(head(edge[edge_mask]), original.edge_attr[edge_mask, col])
                for col, head in enumerate(heads["bond"])) / len(heads["bond"])
            if bool(edge_mask.any()) else atom.new_zeros(()))
    group = F.binary_cross_entropy_with_logits(heads["group"](node[node_mask]), groups[node_mask].float())
    loss = atom + bond + config.group_weight * group
    return loss, {"atom_ce": atom.detach(), "bond_ce": bond.detach(), "group_bce": group.detach()}
