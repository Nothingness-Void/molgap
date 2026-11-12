"""Official GPTrans multi-hop path edge encoding on the BondLocal + pair-update-norm core.

For every ordered pair (i, j) at shortest-path length d <= HOPS, the k-th bond on one
shortest path (k < d) is embedded (mean over the three OGB bond slots), mixed by a
position-specific 32x32 matrix, summed over k and divided by d, then added to the
pair stream.  Official source: GPTrans models/gptrans.py L107-134 (multi_hop_max_dist=20).
Starting `path_mix` at zero keeps the candidate's start state functionally equal to the
reference; the frozen candidate state is the reference state plus that zero tensor.
Pairs farther than HOPS (the model's distance cap) and unreachable pairs get no term.
Path choice among ties: the lowest-index neighbour that is one hop nearer the target.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .gptrans_bond_local import BondLocalGPTrans
from .gptrans_bond_pair_norm import construct_family_model as _reference

HOPS = 20


def _path_term(adjacency, bond_type, distance, table):
    """Sum over path positions of the table row of that bond, divided by hop count."""
    size, nodes = distance.shape[:2]
    # next_hop[b, i, j]: lowest-index neighbour m of i with d(m, j) = d(i, j) - 1.
    nearer = adjacency[:, :, :, None] & (distance[:, None] == (distance - 1)[:, :, None])
    next_hop = nearer.to(torch.uint8).argmax(2)
    current = torch.arange(nodes, device=distance.device)[None, :, None].expand(size, nodes, nodes)
    flat_type = bond_type.reshape(size, -1)
    rows = table.shape[1]
    total = table.new_zeros(size, nodes, nodes, table.shape[-1])
    for k in range(HOPS):
        following = next_hop.gather(1, current)
        code = flat_type.gather(1, (current * nodes + following).reshape(size, -1)).view_as(current)
        # Inactive pairs read the appended zero row; no mask product is kept for backward.
        code = torch.where((distance > k) & (distance <= HOPS), code, rows - 1)
        total = total + F.embedding(code, table[k])
        current = following
    return total / distance.clamp(1, HOPS)[..., None].to(total.dtype)


class MultiHopBondLocalGPTrans(BondLocalGPTrans):
    def _bond_table(self):
        """Per bond-type-combination, per-position mixed embedding, plus a zero row."""
        encoders = self.bond_encoder.bond_embedding_list
        sizes = [encoder.num_embeddings for encoder in encoders]
        codes = torch.arange(sizes[0] * sizes[1] * sizes[2], device=self.path_mix.device)
        embedding = 0
        stride = codes.numel()
        for encoder, size in zip(encoders, sizes):
            stride //= size
            embedding = embedding + encoder((codes // stride) % size)
        mixed = torch.einsum("cd,kde->kce", embedding / len(encoders), self.path_mix)
        return F.pad(mixed, (0, 0, 0, 1)), sizes

    def _multihop(self, edge_attr, edges, padding):
        """Dense (batch, nodes, nodes, pair) path term for the real atoms."""
        graph, source, target = edges
        size, nodes = padding.shape[0], padding.shape[-1] - 1
        table, sizes = self._bond_table()
        attributes = edge_attr.long()
        code = (attributes[:, 0] * sizes[1] + attributes[:, 1]) * sizes[2] + attributes[:, 2]
        device = edge_attr.device
        adjacency = torch.zeros(size, nodes, nodes, dtype=torch.bool, device=device)
        adjacency[graph, source, target] = True
        bond_type = torch.zeros(size, nodes, nodes, dtype=torch.long, device=device)
        bond_type[graph, source, target] = code
        valid = ~padding[:, 0, 0, 1:]
        distance = self._shortest_path(adjacency, valid[:, None, :] & valid[:, :, None])
        return _path_term(adjacency, bond_type, distance, table)

    def forward(self, x, edge_index, edge_attr, batch, random_walk_pe=None):
        node, pair, padding = self._dense_inputs(x, edge_index, edge_attr, batch)
        graph, source, target = self._local_edges(edge_index, batch, len(x))
        term = self._multihop(edge_attr, (graph, source, target), padding)
        pair = pair + F.pad(term.permute(0, 3, 1, 2), (1, 0, 1, 0))
        edges = (graph, source + 1, target + 1)
        flat_target = graph * node.shape[1] + edges[2]
        for block in self.blocks:
            node, pair = block(node, pair, padding, edges, flat_target=flat_target)
        return self.readout(torch.cat((node[:, 0], pair[:, :, 0, 0]), -1))


def construct_family_model(recipe, state):
    model = _reference(recipe, state)
    model.path_mix = nn.Parameter(torch.zeros(HOPS, model.pair_channels, model.pair_channels))
    model.__class__ = MultiHopBondLocalGPTrans
    return model


def policy(mode):
    from .family_training_policy import FamilyModePolicy
    if mode != "degree_bond_local_pair_update_norm_multihop_edge_ema999":
        raise ValueError("Unknown multi-hop edge mode")
    return FamilyModePolicy(construct=construct_family_model,
                            objective_name="degree_bond_local_ema999")
