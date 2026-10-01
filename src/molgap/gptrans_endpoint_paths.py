"""Permutation-covariant endpoint chemistry over accepted graph tensors.

All eligible first bonds are pooled, so equal shortest-path ties never select
an arbitrary atom-index-dependent route. This represents endpoint contrast,
not a full bond sequence or inferred 3D geometry.
"""
import torch
from .gptrans import OGBGPTransTiny


def endpoint_path_contrast(distance, edge_batch, source, target, bond, cap=20):
    batch_size, nodes, _ = distance.shape
    eligible = (distance[edge_batch, target, :] + 1 == distance[edge_batch, source, :])
    eligible &= (distance[edge_batch, source, :] >= 2) & (distance[edge_batch, source, :] <= cap)
    flat_source = edge_batch * nodes + source
    counts = bond.new_zeros(batch_size * nodes, nodes)
    counts.index_add_(0, flat_source, eligible.to(bond.dtype))
    values = bond.new_zeros(batch_size * nodes, bond.shape[1], nodes)
    values.index_add_(0, flat_source, bond[:, :, None] * eligible[:, None, :])
    first = (values / counts.clamp_min(1)[:, None, :]).reshape(batch_size, nodes, bond.shape[1], nodes)
    first = first.permute(0, 2, 1, 3)
    return 0.5 * (first - first.transpose(-1, -2))


class EndpointPathGPTrans(OGBGPTransTiny):
    def _shortest_path(self, adjacency, pair_mask):
        result = super()._shortest_path(adjacency, pair_mask)
        self._endpoint_distance = result
        return result

    def _dense_inputs(self, x, edge_index, edge_attr, batch):
        if hasattr(self, "_endpoint_distance"):
            raise RuntimeError("Nested endpoint path call")
        try:
            node, pair, mask = super()._dense_inputs(x, edge_index, edge_attr, batch)
            edge_batch, source, target = self._local_edges(edge_index, batch, len(x))
            chemistry = self.bond_encoder(edge_attr.long())
            pair[:, :, 1:, 1:] += endpoint_path_contrast(
                self._endpoint_distance, edge_batch, source, target, chemistry, self.shortest_path_cap)
            return node, pair, mask
        finally:
            if hasattr(self, "_endpoint_distance"):
                del self._endpoint_distance


def parameter_groups(model, weight_decay):
    """No-decay only for biases and 1D tensors; embeddings remain decayed."""
    decay, no_decay, inventory = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        exempt = parameter.ndim == 1 or name.endswith(".bias")
        (no_decay if exempt else decay).append(parameter)
        inventory.append({"name": name, "shape": list(parameter.shape),
                          "weight_decay": 0.0 if exempt else weight_decay})
    if not decay or not no_decay:
        raise ValueError("Both optimizer groups must be populated")
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}], inventory
