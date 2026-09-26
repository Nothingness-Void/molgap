"""Two isolated input/local-path additions to the frozen random GPTrans core.

The caller first verifies and loads the seed-42 *random initialization* of the
unchanged pair32 GPTrans model.  The additions below use a separate fixed RNG
stream, so constructing them cannot change the frozen core or its training RNG.
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.utils import to_dense_batch

from .gptrans import OGBGPTransTiny
from .gps import _PersistentEdgeUpdate


MODES = ("rwse16", "rwse16_local_edge")
RWSE_CHANNELS = 16
EDGE_STATE_CHANNELS = 64
ADDON_SEED = 420_016


class LocalInductiveBiasGPTrans(nn.Module):
    """Add RWSE16 to nodes, optionally with persistent true-bond edge states.

    The local path only indexes bonds in ``edge_index``.  Its node-message
    projections and the RWSE projection start at zero, making both variants
    initially functionally equal to the verified random GPTrans core while
    allowing the new paths to learn during training.
    """

    def __init__(self, base: OGBGPTransTiny, mode: str) -> None:
        super().__init__()
        if type(base) is not OGBGPTransTiny or mode not in MODES:
            raise ValueError("Expected frozen OGBGPTransTiny and a supported mode")
        if base.node_channels != 256 or base.pair_channels != 32 or len(base.blocks) != 12:
            raise ValueError("Local bias variants require the frozen GPTrans-T core")
        self.base = base
        self.mode = mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(ADDON_SEED)
            self.rwse_to_node = nn.Linear(RWSE_CHANNELS, base.node_channels, bias=False)
            nn.init.zeros_(self.rwse_to_node.weight)
            if mode == "rwse16_local_edge":
                self.bond_to_edge = nn.Linear(base.pair_channels, EDGE_STATE_CHANNELS)
                self.edge_updates = nn.ModuleList(
                    _PersistentEdgeUpdate(base.node_channels, EDGE_STATE_CHANNELS, 0.1)
                    for _ in base.blocks
                )
                self.edge_to_node = nn.ModuleList(
                    nn.Linear(EDGE_STATE_CHANNELS, base.node_channels, bias=False)
                    for _ in base.blocks
                )
                for projection in self.edge_to_node:
                    nn.init.zeros_(projection.weight)

    def _validate_rwse(self, x: torch.Tensor, pe: torch.Tensor | None) -> torch.Tensor:
        if (pe is None or pe.shape != (x.shape[0], RWSE_CHANNELS)
                or pe.dtype != torch.float32 or pe.device != x.device
                or not bool(torch.isfinite(pe).all())):
            raise ValueError("Expected finite, aligned FP32 RWSE16")
        return pe

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        random_walk_pe: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pe = self._validate_rwse(x, random_walk_pe)
        node, pair, key_padding_mask = self.base._dense_inputs(x, edge_index, edge_attr, batch)
        dense_pe, atom_mask = to_dense_batch(pe, batch)
        if atom_mask.shape != node[:, 1:].shape[:2] or not torch.equal(
            atom_mask, ~key_padding_mask[:, 0, 0, 1:]
        ):
            raise RuntimeError("RWSE and GPTrans atom padding disagree")
        node = torch.cat((node[:, :1], node[:, 1:] + self.rwse_to_node(dense_pe)), dim=1)

        if self.mode == "rwse16_local_edge":
            if edge_index.numel() and not torch.equal(batch[edge_index[0]], batch[edge_index[1]]):
                raise ValueError("A real bond crosses graph boundaries")
            edge_batch, edge_source, edge_target = self.base._local_edges(
                edge_index, batch, int(x.shape[0])
            )
            edge_source = edge_source + 1  # reserve dense position zero for graph token
            edge_target = edge_target + 1
            edge_state = self.bond_to_edge(self.base.bond_encoder(edge_attr.long()))

        for index, block in enumerate(self.base.blocks):
            if self.mode == "rwse16_local_edge":
                edge_state = self.edge_updates[index](
                    node[:, 1:][atom_mask], edge_index, edge_state
                )
                message = self.edge_to_node[index](edge_state)
                flat_target = edge_batch * node.shape[1] + edge_target
                accumulated = torch.zeros_like(node).view(-1, node.shape[-1])
                accumulated.index_add_(0, flat_target, message)
                degree = node.new_zeros(node.shape[0] * node.shape[1], 1)
                degree.index_add_(0, flat_target, node.new_ones((flat_target.numel(), 1)))
                node = node + (accumulated / degree.clamp_min(1)).view_as(node)
            node, pair = block(node, pair, key_padding_mask)
        graph_state = torch.cat((node[:, 0], pair[:, :, 0, 0]), dim=-1)
        return self.base.readout(graph_state)


def apply_local_inductive_bias(base: OGBGPTransTiny, mode: str) -> LocalInductiveBiasGPTrans:
    """Wrap a separately verified, randomly initialized frozen GPTrans core."""
    return LocalInductiveBiasGPTrans(base, mode)
