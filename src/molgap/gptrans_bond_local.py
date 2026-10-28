"""Reviewed local-bond model component from server bdc4d238.

BondLocalBlock/BondLocalGPTrans are retained from gptrans_capacity.py.
The shared family session owns all training and output logic.
"""
import torch
from torch import nn
from .gptrans import OGBGPTransTiny


def _flat_bond_targets(graph, target, node_count):
    return graph * node_count + target


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

    def forward(self, node, pair, padding, edges, *, flat_target=None):
        graph, source, target = edges
        clean = self.norm(node)
        message = self.output(self.message(torch.cat((clean[graph, source], clean[graph, target],
                            pair[graph, :, source, target]), -1)))
        # Strict deterministic mode selects Torch's deterministic CUDA reduction;
        # the owning preflight verifies two optimizer-inclusive repeats before use.
        if flat_target is None:
            flat_target = _flat_bond_targets(graph, target, node.shape[1])
        update = node.new_zeros(node.shape[0] * node.shape[1], node.shape[-1])
        # This fresh zero buffer has no other readers; do not copy it before sum.
        update.index_add_(0, flat_target, message)
        update = update.view_as(node)
        return self.core(node + update, pair, padding)


class BondLocalGPTrans(OGBGPTransTiny):
    def forward(self, x, edge_index, edge_attr, batch, random_walk_pe=None):
        node, pair, padding = self._dense_inputs(x, edge_index, edge_attr, batch)
        graph, source, target = self._local_edges(edge_index, batch, len(x))
        edges = (graph, source + 1, target + 1)
        # Topology and dense width are unchanged across all propagation blocks.
        flat_target = _flat_bond_targets(graph, edges[2], node.shape[1])
        for block in self.blocks:
            node, pair = block(node, pair, padding, edges, flat_target=flat_target)
        return self.readout(torch.cat((node[:, 0], pair[:, :, 0, 0]), -1))


def make_bond_local_model(*, node_channels=256, pair_channels=32, num_layers=12,
                          num_heads=8, pair_update_norm=False):
    """Default tensors stay frozen-compatible; explicit sizes support bounded profiles."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = OGBGPTransTiny(node_channels=node_channels, pair_channels=pair_channels,
                              num_layers=num_layers, num_heads=num_heads)
        with torch.no_grad():
            model.in_degree_encoder.weight.mul_(.0897)
            model.out_degree_encoder.weight.mul_(.0897)
        if pair_update_norm:
            from .gptrans_variants import apply_variant
            apply_variant(model, "pair_update_norm")
        model.__class__ = BondLocalGPTrans
        model.blocks = nn.ModuleList(BondLocalBlock(block, node_channels, pair_channels)
                                     for block in model.blocks)
    retained_shape = (node_channels, pair_channels, num_layers, num_heads) == (256, 32, 12, 8)
    if retained_shape and sum(p.numel() for p in model.parameters()) != 5_871_201:
        raise ValueError("Retained local-bond parameter count differs")
    return model


def construct_family_model(recipe, state):
    """Keep the retained architecture; the shared kernel loads frozen tensors."""
    if recipe.mode in APPROVED_SHAPES:
        layers, nodes, pairs, heads = APPROVED_SHAPES[recipe.mode]
        return make_bond_local_model(num_layers=layers, node_channels=nodes,
                                     pair_channels=pairs, num_heads=heads,
                                     pair_update_norm=True)
    return make_bond_local_model()


APPROVED_SHAPES = {
    "degree_bond_local_pairnorm_12x384_ema999": (12, 384, 48, 12),
    "degree_bond_local_pairnorm_16x256_ema999": (16, 256, 32, 8),
}


def policy(mode):
    from .family_training_policy import FamilyModePolicy
    if mode != "degree_bond_local_ema999" and mode not in APPROVED_SHAPES:
        raise ValueError("Unknown retained local-bond mode")
    return FamilyModePolicy(construct=construct_family_model,
                            objective_name="reference" if mode in APPROVED_SHAPES else None)
