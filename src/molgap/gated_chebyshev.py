"""Sparse gated molecular MPNN with four-hop or matched local diffusion."""
import torch
from torch import nn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


MODES = ("gated_chebyshev4", "gated_local_matched")


def normalized_adjacency(node, source, target, weight):
    return torch.zeros_like(node).index_add(0, target, node[source] * weight[:, None])


class GatedDiffusionBlock(nn.Module):
    def __init__(self, channels, four_hop):
        super().__init__()
        self.four_hop = four_hop
        self.norm = nn.LayerNorm(channels)
        self.gate = nn.Linear(3 * channels, channels)
        self.message = nn.Linear(channels, channels)
        self.local_output = nn.Linear(channels, channels)
        self.taps = nn.ModuleList(nn.Linear(channels, channels) for _ in range(5))
        self.output_norm = nn.LayerNorm(channels)

    def forward(self, node, bond, source, target, weight):
        clean = self.norm(node)
        gate = torch.sigmoid(self.gate(torch.cat((clean[source], clean[target], bond), -1)))
        messages = gate * self.message(clean[source])
        summed = torch.zeros_like(node).index_add(0, target, messages)
        gates = torch.zeros_like(node).index_add(0, target, gate)
        local = self.local_output(summed / gates.clamp_min(1e-6))
        first = normalized_adjacency(clean, source, target, weight)
        terms = [clean, first]
        for hop in range(2, 5):
            terms.append(2 * normalized_adjacency(terms[-1], source, target, weight) - terms[-2]
                         if self.four_hop else terms[hop % 2])
        # Both modes use all five independent, identically initialized tap projections.
        diffusion = sum(tap(term) for tap, term in zip(self.taps, terms)) / len(self.taps)
        return node + torch.nn.functional.silu(self.output_norm(local + diffusion))


class GatedChebyshev(nn.Module):
    accepts_prepared_batch = True

    def __init__(self, mode, channels=192, layers=5):
        super().__init__()
        if mode not in MODES:
            raise ValueError("Unknown gated diffusion mode")
        self.atom_encoder = AtomEncoder(channels)
        self.bond_encoder = BondEncoder(channels)
        self.blocks = nn.ModuleList(GatedDiffusionBlock(channels, mode == MODES[0])
                                    for _ in range(layers))
        self.readout = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, channels),
                                     nn.SiLU(), nn.Linear(channels, 1))

    def forward(self, batch):
        node, bond = self.atom_encoder(batch.x), self.bond_encoder(batch.edge_attr)
        source, target = batch.edge_index
        degree = torch.bincount(target, minlength=len(node)).to(node.dtype)
        inverse_degree = degree.clamp_min(1).rsqrt()
        weight = inverse_degree[source] * inverse_degree[target]
        for block in self.blocks:
            node = block(node, bond, source, target, weight)
        pooled = node.new_zeros(batch.num_graphs, node.shape[-1]).index_add(0, batch.batch, node)
        counts = torch.bincount(batch.batch, minlength=batch.num_graphs).to(node.dtype)
        return self.readout(pooled / counts[:, None]).view(-1)


def construct_family_model(recipe, state):
    return GatedChebyshev(recipe.mode)


def policy(mode):
    from .family_training_policy import FamilyModePolicy
    if mode not in MODES:
        raise ValueError("Unknown gated diffusion mode")
    return FamilyModePolicy(construct=construct_family_model, standalone_model=True,
                            objective_name="degree_bond_local_ema999")
