"""CutlassGT-2026: Zero-4D-Tensor Fused Graph Transformer with WIRE RoPE, FlashBias, ResFormer, and Gated SDPA."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch import nn

NUM_LAYERS = 5
D_MODEL = 256
N_HEADS = 8
D_ROPE = 32
D_BIAS = 16
D_QK = D_ROPE + D_BIAS  # 48, 48 % 4 == 0
D_V = 48
TOPO_DIM = 9  # diag(L^+) + diag(P^1..P^8)


def _compute_wire_topo_and_pad(batch_idx: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Compute sign-free permutation-equivariant topological features and padded indices."""
    bsz = int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 1
    counts = torch.bincount(batch_idx, minlength=bsz)
    max_real = max(int(counts.max().item()), 1)
    n_pad = ((max_real + 1 + 7) // 8) * 8

    offsets = counts.cumsum(0) - counts
    local_idx = torch.arange(num_nodes, device=batch_idx.device) - offsets[batch_idx] + 1  # 1-based (0 is virtual)

    topo_flat = torch.zeros((num_nodes, TOPO_DIM), dtype=torch.float32, device=batch_idx.device)
    with torch.no_grad():
        adj = torch.zeros((bsz, max_real, max_real), dtype=torch.float32, device=batch_idx.device)
        if edge_index.numel() > 0:
            u, v = edge_index[0], edge_index[1]
            eg = batch_idx[u]
            adj[eg, local_idx[u] - 1, local_idx[v] - 1] = 1.0
        deg = adj.sum(dim=-1)
        # Sign-invariant resistance/biharmonic diagonal diag(L^+) via CPU/GPU pinv in no_grad
        lap = torch.diag_embed(deg) - adj
        # Use eigh-free pinv or damped inverse (L + 1/N * 11^T)^{-1} - 1/N * 11^T for exact L^+ without SVD non-determinism
        n_vec = counts.clamp_min(1).to(torch.float32).view(-1, 1, 1)
        valid_r = (torch.arange(max_real, device=batch_idx.device).unsqueeze(0) < counts.unsqueeze(1)).to(torch.float32)
        proj = (valid_r.unsqueeze(2) * valid_r.unsqueeze(1)) / n_vec
        eye_pad = torch.diag_embed(1.0 - valid_r)
        l_reg = lap + proj + eye_pad
        l_pinv = torch.linalg.inv(l_reg) - proj
        diag_lpinv = torch.diagonal(l_pinv, dim1=-2, dim2=-1).clamp(0.0, 20.0) * valid_r

        # Sign-free multi-scale random walk return probabilities diag(P^1..P^8)
        trans = adj / deg.clamp_min(1.0).unsqueeze(-1)
        walk = trans
        diags = [diag_lpinv]
        for k in range(8):
            if k > 0:
                walk = torch.bmm(walk, trans)
            diags.append(torch.diagonal(walk, dim1=-2, dim2=-1) * valid_r)
        topo_dense = torch.stack(diags, dim=-1)  # [B, max_real, 9]
        topo_flat = topo_dense[batch_idx, local_idx - 1]

    key_mask = torch.zeros((bsz, n_pad), dtype=torch.bool, device=batch_idx.device)
    key_mask[:, 0] = True
    key_mask[batch_idx, local_idx] = True
    return local_idx, topo_flat, key_mask, counts, n_pad


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack((out1, out2), dim=-1).flatten(-2)


class CutlassGTBlock(nn.Module):
    """Fused sparse BondLocal + Cutlass SDPA (WIRE RoPE + FlashBias + ResFormer + Gated Attn) + SwiGLU."""

    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS, bond_hidden: int = 96, ffn_hidden: int = 384) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.norm_bond = nn.LayerNorm(d_model)
        self.bond_proj = nn.Linear(64, 64)
        self.bond_msg = nn.Sequential(nn.Linear(2 * d_model + 64, bond_hidden), nn.SiLU(), nn.Linear(bond_hidden, d_model))
        self.bond_gate = nn.Linear(2 * d_model + 64, d_model)

        self.norm_attn = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, n_heads * D_ROPE)
        self.k_proj = nn.Linear(d_model, n_heads * D_ROPE)
        self.phi_q = nn.Linear(d_model, n_heads * D_BIAS)
        self.phi_k = nn.Linear(d_model, n_heads * D_BIAS)
        self.v_proj = nn.Linear(d_model, n_heads * D_V)
        self.gate_proj = nn.Linear(d_model, n_heads * D_V)
        self.out_proj = nn.Linear(n_heads * D_V, d_model)

        self.norm_ffn = nn.LayerNorm(d_model)
        self.swiglu_in = nn.Linear(d_model, 2 * ffn_hidden)
        self.swiglu_out = nn.Linear(ffn_hidden, d_model)

    def forward(self, x_pad: torch.Tensor, batch_idx: torch.Tensor, local_idx: torch.Tensor, edge_index: torch.Tensor,
                e_emb: torch.Tensor, key_mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                v1_res: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, n_pad, d_model = x_pad.shape
        # 1. Sparse O(|E|) directed 1-hop BondLocal covalent message
        if edge_index.numel() > 0:
            xn = self.norm_bond(x_pad)
            u, v = edge_index[0], edge_index[1]
            gu, gv = batch_idx[u], batch_idx[v]
            xu = xn[gu, local_idx[u]]
            xv = xn[gv, local_idx[v]]
            ef = F.silu(self.bond_proj(e_emb))
            cat_e = torch.cat((xu, xv, ef), dim=-1)
            msg = self.bond_msg(cat_e) * torch.sigmoid(self.bond_gate(cat_e))
            upd = x_pad.new_zeros((bsz * n_pad, d_model))
            flat_dst = gv * n_pad + local_idx[v]
            upd.index_add_(0, flat_dst, msg)
            x_pad = x_pad + upd.view(bsz, n_pad, d_model)

        # 2. 8-head Cutlass SDPA with WIRE RoPE + FlashBias + ResFormer + Gated Attention
        h = self.norm_attn(x_pad)
        q_r = self.q_proj(h).view(bsz, n_pad, self.n_heads, D_ROPE).transpose(1, 2)
        k_r = self.k_proj(h).view(bsz, n_pad, self.n_heads, D_ROPE).transpose(1, 2)
        q_r = _apply_rope(q_r, cos, sin)
        k_r = _apply_rope(k_r, cos, sin)

        pq = self.phi_q(h).view(bsz, n_pad, self.n_heads, D_BIAS).transpose(1, 2)
        pk = self.phi_k(h).view(bsz, n_pad, self.n_heads, D_BIAS).transpose(1, 2)
        q = torch.cat((q_r, pq), dim=-1).contiguous()
        k = torch.cat((k_r, pk), dim=-1).contiguous()

        v_cur = self.v_proj(h).view(bsz, n_pad, self.n_heads, D_V).transpose(1, 2).contiguous()
        if v1_res is None:
            v1_out = v_cur
            v_in = v_cur
        else:
            v1_out = v1_res
            v_in = (0.5 * v_cur + 0.5 * v1_res).contiguous()

        attn_out = F.scaled_dot_product_attention(q, k, v_in, attn_mask=key_mask[:, None, None, :])
        attn_out = attn_out.transpose(1, 2).reshape(bsz, n_pad, self.n_heads * D_V)
        attn_out = attn_out * torch.sigmoid(self.gate_proj(h))
        x_pad = x_pad + self.out_proj(attn_out)

        # 3. Pre-LN SwiGLU FFN
        u_g, v_g = self.swiglu_in(self.norm_ffn(x_pad)).chunk(2, dim=-1)
        x_pad = x_pad + self.swiglu_out(F.silu(u_g) * v_g)
        x_pad = x_pad * key_mask.unsqueeze(-1).to(x_pad.dtype)
        return x_pad, v1_out


class CutlassGT2026(nn.Module):
    """Zero-4D-Tensor Fused Graph Transformer."""

    accepts_prepared_batch = True

    def __init__(self) -> None:
        super().__init__()
        self.atom_encoder = AtomEncoder(D_MODEL)
        self.bond_encoder = BondEncoder(64)
        self.virt_token = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.topo_proj = nn.Linear(TOPO_DIM, D_MODEL)
        self.rope_freq = nn.Sequential(nn.Linear(TOPO_DIM, 64), nn.SiLU(), nn.Linear(64, N_HEADS * (D_ROPE // 2)))
        self.blocks = nn.ModuleList([CutlassGTBlock() for _ in range(NUM_LAYERS)])
        self.final_norm = nn.LayerNorm(D_MODEL)
        self.readout = nn.Sequential(nn.Linear(2 * D_MODEL, D_MODEL), nn.SiLU(), nn.Linear(D_MODEL, 1))

    def forward(self, batch) -> torch.Tensor:
        x_emb = self.atom_encoder(batch.x.long())
        e_emb = self.bond_encoder(batch.edge_attr.long()) if batch.edge_attr.numel() > 0 else x_emb.new_zeros((0, 64))
        batch_idx = batch.batch
        bsz = int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 1

        local_idx, topo_flat, key_mask, counts, n_pad = _compute_wire_topo_and_pad(batch_idx, batch.edge_index, x_emb.shape[0])
        x_nodes = x_emb + self.topo_proj(topo_flat)
        x_pad = x_emb.new_zeros((bsz, n_pad, D_MODEL))
        x_pad[:, 0:1] = self.virt_token.expand(bsz, 1, D_MODEL)
        x_pad[batch_idx, local_idx] = x_nodes

        angles = x_emb.new_zeros((bsz, n_pad, N_HEADS * (D_ROPE // 2)))
        angles[batch_idx, local_idx] = self.rope_freq(topo_flat)
        angles = angles.view(bsz, n_pad, N_HEADS, D_ROPE // 2).transpose(1, 2)
        cos, sin = torch.cos(angles), torch.sin(angles)

        v1_res = None
        for blk in self.blocks:
            x_pad, v1_res = blk(x_pad, batch_idx, local_idx, batch.edge_index, e_emb, key_mask, cos, sin, v1_res)

        x_pad = self.final_norm(x_pad) * key_mask.unsqueeze(-1).to(x_pad.dtype)
        real_mask = key_mask.clone()
        real_mask[:, 0] = False
        real_sum = (x_pad * real_mask.unsqueeze(-1).to(x_pad.dtype)).sum(dim=1)
        real_mean = real_sum / counts.clamp_min(1).unsqueeze(-1).to(x_pad.dtype)
        return self.readout(torch.cat((x_pad[:, 0], real_mean), dim=-1)).squeeze(-1)


def construct_family_model(recipe, state):
    return CutlassGT2026()


def policy(mode):
    from .family_training_policy import FamilyModePolicy

    if mode != "cutlass_gt":
        raise ValueError("Unknown CutlassGT mode")
    return FamilyModePolicy(
        standalone_model=True,
        construct=construct_family_model,
        objective_name="degree_bond_local_ema999",
    )
