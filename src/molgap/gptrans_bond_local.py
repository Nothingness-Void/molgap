"""Reviewed local-bond model component from server bdc4d238.

BondLocalBlock/BondLocalGPTrans are retained from gptrans_capacity.py.
The shared family session owns all training and output logic.
"""
import torch
from torch import nn
from .gptrans import OGBGPTransTiny

class BondLocalBlock(nn.Module):
    """Unnormalized true-bond sum before GPA, with an identity initial bypass.

Only accepted real directed bonds carry messages. There is no all-pairs edge
MLP and no new coordinate, path or feature cache.
"""
    def __init__(self, block, channels=256, pair_channels=32):
        super().__init__()
        self.core = block
        self.norm = nn.LayerNorm(channels)
        self.message = nn.Sequential(nn.Linear(2 * channels + pair_channels, 64), nn.GELU())
        self.output = nn.Linear(64, channels)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, node, pair, padding, edges):
        graph, source, target = edges
        clean = self.norm(node)
        message = self.output(self.message(torch.cat((clean[graph, source], clean[graph, target],
                            pair[graph, :, source, target]), -1)))
        # Strict deterministic mode selects Torch's deterministic CUDA reduction;
        # the owning preflight verifies two optimizer-inclusive repeats before use.
        flat_target = graph * node.shape[1] + target
        update = node.new_zeros(node.shape[0] * node.shape[1], node.shape[-1])
        update = torch.index_add(update, 0, flat_target, message)
        update = update.view_as(node)
        return self.core(node + update, pair, padding)


class BondLocalGPTrans(OGBGPTransTiny):
    def forward(self, x, edge_index, edge_attr, batch, random_walk_pe=None):
        node, pair, padding = self._dense_inputs(x, edge_index, edge_attr, batch)
        graph, source, target = self._local_edges(edge_index, batch, len(x))
        edges = (graph, source + 1, target + 1)
        for block in self.blocks:
            node, pair = block(node, pair, padding, edges)
        return self.readout(torch.cat((node[:, 0], pair[:, :, 0, 0]), -1))


def make_bond_local_model():
    """Construct the retained seed42 shape; the family loader supplies frozen tensors."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = OGBGPTransTiny()
        with torch.no_grad():
            model.in_degree_encoder.weight.mul_(.0897)
            model.out_degree_encoder.weight.mul_(.0897)
        model.__class__ = BondLocalGPTrans
        model.blocks = nn.ModuleList(BondLocalBlock(block) for block in model.blocks)
    if sum(p.numel() for p in model.parameters()) != 5_871_201:
        raise ValueError("Retained local-bond parameter count differs")
    return model
