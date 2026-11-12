"""Official GPTrans multi-hop path edge encoding on the BondLocal + pair-update-norm core.

For every ordered pair (i, j) at shortest-path length d <= HOPS, the k-th bond on one
shortest path (k < d) is embedded (mean over the three OGB bond slots), mixed by a
position-specific 32x32 matrix, summed over k and divided by d, then added to the
pair stream.  Official source: GPTrans models/gptrans.py L107-134 (multi_hop_max_dist=20).
The term is added inside `spatial_encoder` (a substrate submodule replacement mounted after
the frozen state loads), so both arms share one initial state; `path_mix` starts at zero and the
candidate's initial output equals the reference bit for bit.
Pairs farther than HOPS (the model's distance cap) and unreachable pairs get no term.
Path choice among ties: the lowest-index neighbour that is one hop nearer the target.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .models.substrate import ModelSubstrate
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


class MultiHopSpatial(nn.Module):
    """Distance embedding plus the path term; bond data arrives through a model pre-hook."""

    def __init__(self, base, model):
        super().__init__()
        self.base = base
        self.path_mix = nn.Parameter(torch.zeros(HOPS, base.embedding_dim, base.embedding_dim))
        self._owner, self.args = [model], None

    def _table(self):
        encoders = self._owner[0].bond_encoder.bond_embedding_list
        sizes = [encoder.num_embeddings for encoder in encoders]
        codes = torch.arange(sizes[0] * sizes[1] * sizes[2], device=self.path_mix.device)
        embedding, stride = 0, codes.numel()
        for encoder, size in zip(encoders, sizes):
            stride //= size
            embedding = embedding + encoder((codes // stride) % size)
        mixed = torch.einsum("cd,kde->kce", embedding / len(encoders), self.path_mix)
        return F.pad(mixed, (0, 0, 0, 1)), sizes

    def forward(self, spatial):
        x, edge_index, edge_attr, batch = self.args
        graph, source, target = self._owner[0]._local_edges(edge_index, batch, len(x))
        table, sizes = self._table()
        a = edge_attr.long()
        code = (a[:, 0] * sizes[1] + a[:, 1]) * sizes[2] + a[:, 2]
        bond_type = torch.zeros_like(spatial)
        bond_type[graph, source, target] = code
        # Distance 1 is exactly a bonded pair; padding carries the unreachable sentinel.
        term = _path_term(spatial == 1, bond_type, spatial, table)
        return self.base(spatial) + term


def mount(recipe, model):
    """Runs after the frozen state loads, before optimizer creation."""
    wrapper = MultiHopSpatial(model.spatial_encoder, model)
    ModelSubstrate(model, "gptrans_t").replace_submodule("multihop_edge", "spatial_encoder", wrapper)
    model.register_forward_pre_hook(lambda module, args: setattr(wrapper, "args", args[:4]))
    return model


def policy(mode):
    from .family_training_policy import FamilyModePolicy
    if mode != "degree_bond_local_pair_update_norm_multihop_edge_ema999":
        raise ValueError("Unknown multi-hop edge mode")
    return FamilyModePolicy(construct=_reference, mount=mount,
                            objective_name="degree_bond_local_ema999")
