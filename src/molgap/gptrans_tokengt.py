"""Permutation-invariant TokenGT architecture on the reviewed GPTrans family session."""
from __future__ import annotations

import torch
from torch import nn

from .gptrans import DropPath, OGBGPTransTiny


class TokenGTAttention(nn.Module):
    """Multi-head self-attention with additive per-head structural incidence bias."""

    def __init__(self, d_model: int = 256, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, channels = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        logits = (q @ k.transpose(-2, -1)) * self.scale + attn_bias
        logits = logits.masked_fill(key_padding_mask, float("-inf"))
        attn = self.attn_drop(torch.softmax(logits, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(bsz, seq_len, channels)
        return self.out_drop(self.out_proj(out))


class TokenGTBlock(nn.Module):
    """Pre-LN Transformer block with DropPath and LayerScale."""

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        d_ff: int = 424,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        layer_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = TokenGTAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path)
        self.attn_scale = nn.Parameter(torch.full((d_model,), float(layer_scale)))
        self.ffn_scale = nn.Parameter(torch.full((d_model,), float(layer_scale)))

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.drop_path(self.attn_scale * self.attn(self.norm1(x), attn_bias, key_padding_mask))
        x = x + self.drop_path(self.ffn_scale * self.ffn(self.norm2(x)))
        return x


class TokenGTGPTrans(OGBGPTransTiny):
    """Permutation-invariant TokenGT with Chebyshev spectral and incidence encodings."""

    def __init__(self) -> None:
        super().__init__(
            node_channels=256,
            pair_channels=32,
            num_layers=12,
            num_heads=8,
            dropout=0.1,
            drop_path=0.1,
            layer_scale=1.0,
            n_targets=1,
        )
        from ogb.graphproppred.mol_encoder import BondEncoder

        for attr in (
            "in_degree_encoder",
            "out_degree_encoder",
            "spatial_encoder",
            "virtual_pair",
            "node_norm",
            "input_dropout",
            "blocks",
            "readout",
            "bond_encoder",
        ):
            delattr(self, attr)

        self.bond_encoder = BondEncoder(256)
        self.spec_node_mlp = nn.Sequential(nn.Linear(16, 256), nn.LayerNorm(256), nn.GELU())
        self.spec_graph_mlp = nn.Sequential(nn.Linear(16, 256), nn.LayerNorm(256), nn.GELU())
        self.spec_edge_mlp = nn.Sequential(nn.Linear(48, 256), nn.LayerNorm(256), nn.GELU())
        self.token_type_emb = nn.Embedding(3, 256)
        self.input_norm = nn.LayerNorm(256)
        self.input_drop = nn.Dropout(0.1)
        self.incidence_bias = nn.Embedding(4, 8)
        path_rates = torch.linspace(0.0, 0.1, 12).tolist()
        self.blocks = nn.ModuleList(
            TokenGTBlock(256, 8, 424, 0.1, path_rates[idx], 1.0) for idx in range(12)
        )
        self.out_norm = nn.LayerNorm(256)
        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    @staticmethod
    def _resolve_layout(x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, layout):
        if layout is not None:
            return (
                layout["batch_size"],
                layout["max_nodes"],
                layout["mask"],
                layout["flat_index"],
                layout["edges"],
            )
        node_count = int(x.shape[0])
        bsz = int(batch.max().item()) + 1 if batch.numel() else 0
        positions = torch.arange(node_count, device=batch.device)
        boundaries = torch.cat((positions.new_zeros(1), torch.where(batch[1:] != batch[:-1])[0] + 1))
        counts = torch.cat((boundaries[1:], positions.new_tensor([node_count]))) - boundaries
        max_nodes = int(counts.max().item()) if bsz else 0
        local = positions - boundaries[batch]
        mask = torch.arange(max_nodes, device=batch.device)[None, :] < counts[:, None]
        flat_index = batch * max_nodes + local
        edges = (batch[edge_index[0]], local[edge_index[0]], local[edge_index[1]])
        return bsz, max_nodes, mask, flat_index, edges

    @staticmethod
    def _chebyshev_spectral(
        bsz: int,
        max_nodes: int,
        mask: torch.Tensor,
        edges: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        device = mask.device
        graph, src, dst = edges
        adj = torch.zeros((bsz, max_nodes, max_nodes), dtype=dtype, device=device)
        adj[graph, src, dst] = 1.0
        adj[graph, dst, src] = 1.0
        pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        adj = adj * pair_mask.to(dtype)
        deg = adj.sum(dim=-1)
        d_inv_sqrt = torch.where(deg > 0, deg.clamp_min(1.0).rsqrt(), torch.zeros_like(deg))
        norm_lap = -adj * d_inv_sqrt.unsqueeze(-1) * d_inv_sqrt.unsqueeze(-2)
        t0 = torch.eye(max_nodes, dtype=dtype, device=device).unsqueeze(0).expand(bsz, -1, -1)
        t0 = t0 * pair_mask.to(dtype)
        polys = [t0, norm_lap]
        for _ in range(2, 16):
            polys.append(2.0 * torch.bmm(norm_lap, polys[-1]) - polys[-2])
        return torch.stack(polys, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        random_walk_pe: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del random_walk_pe
        bsz, max_nodes, mask, flat_index, edges = self._resolve_layout(
            x, edge_index, batch, getattr(self, "_execution_layout", None)
        )
        atom_emb = self.atom_encoder(x.long())
        dtype, device = atom_emb.dtype, atom_emb.device
        spectral = self._chebyshev_spectral(bsz, max_nodes, mask, edges, dtype)
        k_diag = spectral.diagonal(dim1=-2, dim2=-1).permute(0, 2, 1)
        valid_counts = mask.sum(dim=-1).clamp_min(1).to(dtype).unsqueeze(-1)
        k_graph = (k_diag * mask.unsqueeze(-1).to(dtype)).sum(dim=1) / valid_counts

        graph_tok = self.graph_token.expand(bsz, 1, -1) + self.spec_graph_mlp(k_graph).unsqueeze(1)
        dense_atoms = atom_emb.new_zeros((bsz * max_nodes, 256))
        dense_atoms[flat_index] = atom_emb
        node_tok = dense_atoms.view(bsz, max_nodes, 256) + self.spec_node_mlp(k_diag)

        graph, src, dst = edges
        lo = torch.minimum(src, dst)
        hi = torch.maximum(src, dst)
        bond_emb = self.bond_encoder(edge_attr.long())
        pair_idx = (graph * max_nodes + lo) * max_nodes + hi
        bond_grid = bond_emb.new_zeros((bsz * max_nodes * max_nodes, 256))
        bond_grid.index_add_(0, pair_idx, bond_emb)
        mult_grid = bond_emb.new_zeros((bsz * max_nodes * max_nodes,))
        mult_grid.index_add_(0, pair_idx, bond_emb.new_ones((edge_attr.shape[0],)))
        bond_grid = bond_grid.view(bsz, max_nodes, max_nodes, 256)
        mult_grid = mult_grid.view(bsz, max_nodes, max_nodes)
        upper = torch.triu(torch.ones((max_nodes, max_nodes), dtype=torch.bool, device=device), diagonal=1)
        has_undir = (mult_grid > 0) & upper.unsqueeze(0)
        bond_grid = bond_grid / mult_grid.clamp_min(1.0).unsqueeze(-1)

        flat_has = has_undir.view(bsz, max_nodes * max_nodes)
        m_max = int(flat_has.sum(dim=-1).max().item()) if bsz else 0
        if m_max > 0:
            pos = flat_has.long().cumsum(dim=-1) - 1
            b_idx, p_idx = torch.where(flat_has)
            m_idx = pos[b_idx, p_idx]
            edge_lo_idx = torch.div(p_idx, max_nodes, rounding_mode="floor")
            edge_hi_idx = p_idx % max_nodes

            edge_tok = atom_emb.new_zeros((bsz, m_max, 256))
            edge_mask = mask.new_zeros((bsz, m_max))
            edge_lo = torch.zeros((bsz, m_max), dtype=torch.long, device=device)
            edge_hi = torch.zeros((bsz, m_max), dtype=torch.long, device=device)

            k_lo = k_diag[b_idx, edge_lo_idx]
            k_hi = k_diag[b_idx, edge_hi_idx]
            k_pair = spectral[b_idx, :, edge_lo_idx, edge_hi_idx]
            spec_edge = torch.cat((0.5 * (k_lo + k_hi), (k_lo - k_hi).abs(), k_pair), dim=-1)
            edge_tok[b_idx, m_idx] = bond_grid[b_idx, edge_lo_idx, edge_hi_idx] + self.spec_edge_mlp(spec_edge)
            edge_mask[b_idx, m_idx] = True
            edge_lo[b_idx, m_idx] = edge_lo_idx
            edge_hi[b_idx, m_idx] = edge_hi_idx
        else:
            edge_tok = atom_emb.new_zeros((bsz, 0, 256))
            edge_mask = mask.new_zeros((bsz, 0))
            edge_lo = torch.zeros((bsz, 0), dtype=torch.long, device=device)
            edge_hi = torch.zeros((bsz, 0), dtype=torch.long, device=device)

        tokens = torch.cat((graph_tok, node_tok, edge_tok), dim=1)
        seq_len = 1 + max_nodes + m_max
        type_ids = torch.empty((seq_len,), dtype=torch.long, device=device)
        type_ids[0] = 0
        type_ids[1 : 1 + max_nodes] = 1
        if m_max > 0:
            type_ids[1 + max_nodes :] = 2
        tokens = self.input_drop(self.input_norm(tokens + self.token_type_emb(type_ids).unsqueeze(0)))

        full_mask = torch.cat((torch.ones((bsz, 1), dtype=torch.bool, device=device), mask, edge_mask), dim=1)
        key_padding_mask = ~full_mask[:, None, None, :]

        bias_idx = torch.zeros((bsz, seq_len, seq_len), dtype=torch.long, device=device)
        if m_max > 0:
            atom_ids = torch.arange(max_nodes, device=device)[None, :, None]
            inc = (atom_ids == edge_lo.unsqueeze(1)) | (atom_ids == edge_hi.unsqueeze(1))
            inc = inc & mask.unsqueeze(2) & edge_mask.unsqueeze(1)
            bias_idx[:, 1 : 1 + max_nodes, 1 + max_nodes :] = inc.long() * 2
            bias_idx[:, 1 + max_nodes :, 1 : 1 + max_nodes] = inc.transpose(1, 2).long() * 2
            shared = (
                (edge_lo.unsqueeze(2) == edge_lo.unsqueeze(1))
                | (edge_lo.unsqueeze(2) == edge_hi.unsqueeze(1))
                | (edge_hi.unsqueeze(2) == edge_lo.unsqueeze(1))
                | (edge_hi.unsqueeze(2) == edge_hi.unsqueeze(1))
            ) & edge_mask.unsqueeze(2) & edge_mask.unsqueeze(1)
            bias_idx[:, 1 + max_nodes :, 1 + max_nodes :] = shared.long() * 3
        diag = torch.eye(seq_len, dtype=torch.bool, device=device).unsqueeze(0)
        bias_idx = bias_idx.masked_fill(diag, 1)
        attn_bias = self.incidence_bias(bias_idx).permute(0, 3, 1, 2)

        for block in self.blocks:
            tokens = block(tokens, attn_bias, key_padding_mask)
        return self.head(self.out_norm(tokens[:, 0]))


def make_tokengt_model() -> TokenGTGPTrans:
    """Construct the deterministic seed42 TokenGT model."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        return TokenGTGPTrans()


def construct_family_model(recipe, state) -> TokenGTGPTrans:
    """Construct TokenGT for the shared GPTrans family session."""
    del recipe, state
    return make_tokengt_model()


def policy(mode: str):
    from .family_training_policy import FamilyModePolicy

    if mode != "tokengt_ema999":
        raise ValueError("Unknown TokenGT mode")
    return FamilyModePolicy(construct=construct_family_model)
