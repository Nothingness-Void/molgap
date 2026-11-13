"""Zero-start rooted h=2 induced-subgraph MPNN on a K1/3 pre-message slot."""
from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_substrate import K1MessageInput, ModelSubstrate, SlotBinding, SlotName


class _SubgraphLayer(nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.src_proj = nn.Linear(hidden_channels, hidden_channels)
        self.edge_proj = nn.Linear(hidden_channels, hidden_channels)
        self.upd_norm = nn.LayerNorm(hidden_channels * 2)
        self.upd_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
            nn.SiLU(),
            nn.Linear(hidden_channels * 2, hidden_channels),
        )

    def forward(self, h, b_e, w_e, e_emb, e_active, flat_dst, b_n_n, n_max):
        m_node = self.src_proj(self.norm1(h))
        agg = h.new_zeros((b_n_n, h.shape[-1]))
        if b_e.numel() > 0:
            m_edge = F.silu(m_node[b_e, w_e] + self.edge_proj(e_emb).unsqueeze(1))
            m_edge = m_edge * e_active.unsqueeze(-1)
            agg.index_add_(0, flat_dst, m_edge.reshape(-1, h.shape[-1]))
        agg = agg.reshape(h.shape)
        return h + self.upd_mlp(self.upd_norm(torch.cat((h, agg), dim=-1)))


class NestedSubgraphMessage(nn.Module):
    """Rooted h=2 induced-subgraph MPNN; zero return projection preserves step 0."""

    def __init__(
        self,
        hidden_channels: int = 64,
        node_channels: int = 192,
        edge_channels: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.node_proj = nn.Linear(node_channels, hidden_channels)
        self.edge_in = nn.Linear(edge_channels, hidden_channels)
        self.dist_emb = nn.Embedding(4, hidden_channels)
        self.layers = nn.ModuleList(
            [_SubgraphLayer(hidden_channels) for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(hidden_channels * 2)
        self.return_projection = nn.Linear(hidden_channels * 2, node_channels, bias=False)
        nn.init.zeros_(self.return_projection.weight)

    def forward(self, payload: K1MessageInput) -> K1MessageInput:
        node = payload.node
        n_nodes = node.shape[0]
        if n_nodes == 0:
            return payload
        edge_index, edge_attr, graph_index = payload.edge_index, payload.edge_attr, payload.graph_index
        if graph_index.ndim != 1 or graph_index.numel() != n_nodes:
            raise ValueError("Malformed graph_index for NestedSubgraphMessage")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("Malformed edge_index for NestedSubgraphMessage")
        if int(graph_index.min().item()) < 0:
            raise ValueError("Negative graph_index in NestedSubgraphMessage")
        device = node.device
        num_graphs = int(graph_index.max().item()) + 1
        counts = torch.zeros(num_graphs, dtype=torch.long, device=device)
        counts.index_add_(0, graph_index, torch.ones(n_nodes, dtype=torch.long, device=device))
        if (counts <= 0).any():
            raise ValueError("Empty graph in NestedSubgraphMessage batch")
        offsets = counts.cumsum(0) - counts
        local_idx = torch.arange(n_nodes, device=device, dtype=torch.long) - offsets[graph_index]
        if (local_idx < 0).any() or (local_idx >= counts[graph_index]).any():
            raise ValueError("Non-contiguous graph_index in NestedSubgraphMessage")
        n_max = int(counts.max().item())

        mask = torch.arange(n_max, device=device).unsqueeze(0) < counts.unsqueeze(1)
        pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
        src, dst = edge_index[0], edge_index[1]
        b_e, w_e, u_e = graph_index[src], local_idx[src], local_idx[dst]
        adj = torch.zeros((num_graphs, n_max, n_max), dtype=torch.bool, device=device)
        if edge_index.shape[1] > 0:
            adj[b_e, w_e, u_e] = True
            adj[b_e, u_e, w_e] = True
        eye = torch.eye(n_max, dtype=torch.bool, device=device).unsqueeze(0).expand(num_graphs, -1, -1)
        adj = adj & ~eye
        a0 = eye & pair_mask
        a1 = adj & ~a0 & pair_mask
        adj_f = adj.to(dtype=node.dtype)
        a2 = (torch.bmm(adj_f, adj_f) > 0.5) & ~a0 & ~a1 & pair_mask
        in_sub = a0 | a1 | a2
        dist_id = torch.full((num_graphs, n_max, n_max), 3, dtype=torch.long, device=device)
        dist_id = dist_id.masked_fill(a2, 2).masked_fill(a1, 1).masked_fill(a0, 0)

        node_h = node.new_zeros((num_graphs, n_max, self.hidden_channels))
        node_h[graph_index, local_idx] = self.node_proj(node)
        in_sub_f = in_sub.unsqueeze(-1).to(dtype=node.dtype)
        h = (node_h.unsqueeze(2) + self.dist_emb(dist_id)) * in_sub_f
        e_emb = self.edge_in(edge_attr)
        e_active = (in_sub[b_e, w_e, :] & in_sub[b_e, u_e, :]).to(dtype=h.dtype)
        r_range = torch.arange(n_max, device=device, dtype=torch.long)
        flat_dst = ((b_e * n_max + u_e).unsqueeze(1) * n_max + r_range.unsqueeze(0)).reshape(-1)
        b_n_n = num_graphs * n_max * n_max

        for layer in self.layers:
            h = layer(h, b_e, w_e, e_emb, e_active, flat_dst, b_n_n, n_max) * in_sub_f

        root_feat = torch.diagonal(h, dim1=1, dim2=2).permute(0, 2, 1)
        sub_mean = h.sum(dim=1) / in_sub.sum(dim=1).clamp_min(1).to(dtype=h.dtype).unsqueeze(-1)
        node_out = torch.cat((root_feat, sub_mean), dim=-1)[graph_index, local_idx]
        delta = self.return_projection(F.silu(self.out_norm(node_out)))
        return replace(payload, node=node + delta)


def apply_addon(model, config):
    owner = getattr(model, "_molgap_substrate", None)
    if owner is None:
        owner = ModelSubstrate(model, "neural_atom_k1")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(config["seed"]))
        module = NestedSubgraphMessage()
    owner.attach(
        config["name"],
        module,
        SlotBinding("neural_atom_k1", SlotName.PRE_MESSAGE, int(config["layer"])),
    )
    return model
