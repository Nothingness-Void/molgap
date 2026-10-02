"""Parameter-free final readout interventions; unchanged GPTrans propagation."""
from __future__ import annotations

import torch

from .gptrans import OGBGPTransTiny

READOUT_MODES = ("degree_node_mean_readout_ema999", "degree_bond_mean_readout_ema999")


def select_readout(node, pair, node_mask, bond_mask, mode):
    """Pool only real atoms or real directed bonds, never virtual/padded entries."""
    if mode not in READOUT_MODES:
        raise ValueError(mode)
    if node_mask.shape != node[:, 1:].shape[:2] or bond_mask.shape != pair[:, 0, 1:, 1:].shape:
        raise ValueError("Readout masks do not match real molecular states")
    atom_count = node_mask.sum(1, keepdim=True)
    if bool((atom_count == 0).any()):
        raise ValueError("Empty molecule cannot have an atom readout")
    if bool((bond_mask & ~(node_mask[:, :, None] & node_mask[:, None, :])).any()):
        raise ValueError("Bond readout includes padded atoms")
    node_summary, pair_summary = node[:, 0], pair[:, :, 0, 0]
    if mode == READOUT_MODES[0]:
        node_summary = (node[:, 1:] * node_mask[:, :, None]).sum(1) / atom_count
    else:
        count = bond_mask.sum((1, 2)).unsqueeze(1)
        pooled = (pair[:, :, 1:, 1:] * bond_mask[:, None]).sum((2, 3)) / count.clamp_min(1)
        # Bondless molecules retain the reference relation readout, not NaN/zeros.
        pair_summary = torch.where(count > 0, pooled, pair_summary)
    return torch.cat((node_summary, pair_summary), dim=-1)


class MolecularReadoutGPTrans(OGBGPTransTiny):
    def forward(self, x, edge_index, edge_attr, batch, random_walk_pe=None):
        del random_walk_pe
        node, pair, key_padding_mask = self._dense_inputs(x, edge_index, edge_attr, batch)
        for block in self.blocks:
            node, pair = block(node, pair, key_padding_mask)
        node_mask = ~key_padding_mask[:, 0, 0, 1:]
        bond_mask = torch.zeros((len(node), node_mask.shape[1], node_mask.shape[1]),
                                dtype=torch.bool, device=node.device)
        molecules, src, dst = self._local_edges(edge_index, batch, len(x))
        bond_mask[molecules, src, dst] = True
        return self.readout(select_readout(node, pair, node_mask, bond_mask, self._molgap_author_variant))


def apply_readout_variant(model, mode):
    if mode not in READOUT_MODES:
        raise ValueError(mode)
    model.__class__ = MolecularReadoutGPTrans
    model._molgap_author_variant = mode
    return model
