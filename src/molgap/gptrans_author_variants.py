"""Single-factor input hypotheses on the unchanged GPTrans propagation core.

The path arm pools existing categorical bond tables; it is an author-inspired
input experiment, not a reproduction of GPTrans's learned hop-position mixing.
"""
from __future__ import annotations

import torch

from .gptrans import OGBGPTransTiny

MODES = ("degree_scale", "path_bond_mean", "degree_path_bond_mean", "degree_scale_ema999",
         "degree_group_decay_ema999", "degree_path_endpoints_ema999", "degree_pair_depth_scale_ema999", "degree_path_bond_mean_ema999",
         "degree_node_mean_readout_ema999", "degree_bond_mean_readout_ema999", "degree_decay001_ema999")
MODES += ("degree_node352_ema999", "degree_pair64_ema999", "degree_ffn2_ema999", "degree_bond_local_ema999")
MODES += ("degree_pair_transition_ema999",)
PATH_MODES = frozenset({"path_bond_mean", "degree_path_bond_mean", "degree_path_bond_mean_ema999"})
SCALED_MODES = frozenset(set(MODES) - {"path_bond_mean"})
G1_ARCHITECTURE_ID = "f156359acf2bcd121c04234c22195a12d4e605c17b1129c91c8a17a91c555896"
DEGREE_SCALE = 0.0897
DEGREE_INITIAL_SHA256 = "d10738efd0851fd53513b51c0a85c42327f87ee0ea35a0af1b6a381b8ce179d1"


def pooled_path_bonds(counts, lengths, tables):
    """Encode path category histograms without materializing every hop vector."""
    if counts.ndim != 2 or lengths.ndim != 1 or len(counts) != len(lengths):
        raise ValueError("Path histogram/length shape mismatch")
    if counts.shape[1] != sum(table.num_embeddings for table in tables):
        raise ValueError("Path bond feature cardinalities differ")
    if bool((lengths < 2).any()) or bool((lengths > 20).any()):
        raise ValueError("Only nonbonded shortest paths of length 2..20 are allowed")
    if bool((counts < 0).any()):
        raise ValueError("Negative path category counts")
    result = counts.new_zeros((len(counts), tables[0].embedding_dim), dtype=tables[0].weight.dtype)
    offset = 0
    for table in tables:
        values = counts[:, offset:offset + table.num_embeddings]
        if not torch.equal(values.sum(1).long(), lengths.long()):
            raise ValueError("Each categorical path histogram must sum to path length")
        result = result + values.to(table.weight.dtype) @ table.weight
        offset += table.num_embeddings
    return result / lengths.to(result.dtype).unsqueeze(1)


class PathInputGPTrans(OGBGPTransTiny):
    """Add nonbonded path-mean encoding, retaining all original tensors."""

    def forward_batch(self, batch):
        if hasattr(self, "_author_paths"):
            raise RuntimeError("Nested path forward is not supported")
        required = ("path_pair_index", "path_counts", "path_lengths")
        if any(not hasattr(batch, name) for name in required):
            raise RuntimeError("Accepted CPU path sidecar is required")
        self._author_paths = tuple(getattr(batch, name) for name in required)
        try:
            return self(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                        getattr(batch, "random_walk_pe", None))
        finally:
            del self._author_paths

    def _dense_inputs(self, x, edge_index, edge_attr, batch):
        if not hasattr(self, "_author_paths"):
            raise RuntimeError("Use forward_batch with accepted path inputs")
        node, pair, mask = super()._dense_inputs(x, edge_index, edge_attr, batch)
        indices, counts, lengths = self._author_paths
        if indices.ndim != 2 or indices.shape[0] != 2 or indices.shape[1] != len(lengths):
            raise ValueError("Path pair indices differ from sidecar")
        if bool((batch[indices[0]] != batch[indices[1]]).any()):
            raise ValueError("Path pair crosses molecules")
        molecules, source, target = self._local_edges(indices, batch, len(x))
        values = pooled_path_bonds(counts, lengths, self.bond_encoder.bond_embedding_list)
        pair[molecules, :, source + 1, target + 1] += values
        return node, pair, mask


def apply_author_variant(model, variant):
    if variant not in MODES:
        raise ValueError(variant)
    if variant in {"degree_node352_ema999", "degree_ffn2_ema999", "degree_bond_local_ema999", "degree_pair64_ema999", "degree_pair_transition_ema999"}:
        raise ValueError("Capacity arms require their complete frozen initialization factory")
    before = set(model.state_dict())
    if variant in SCALED_MODES:
        with torch.no_grad():
            model.in_degree_encoder.weight.mul_(DEGREE_SCALE)
            model.out_degree_encoder.weight.mul_(DEGREE_SCALE)
    if variant in PATH_MODES:
        # No constructor or RNG consumption: only the input method changes.
        model.__class__ = PathInputGPTrans
    elif variant == "degree_path_endpoints_ema999":
        from .gptrans_endpoint_paths import EndpointPathGPTrans
        model.__class__ = EndpointPathGPTrans
    elif variant == "degree_pair_depth_scale_ema999":
        from .gptrans_pair_scale import apply_pair_depth_scale
        apply_pair_depth_scale(model)
    elif variant in {"degree_node_mean_readout_ema999", "degree_bond_mean_readout_ema999"}:
        from .gptrans_readout import apply_readout_variant
        apply_readout_variant(model, variant)
    if set(model.state_dict()) != before:
        raise RuntimeError("Author input arm unexpectedly changed tensor keys")
    model._molgap_author_variant = variant
    return model
