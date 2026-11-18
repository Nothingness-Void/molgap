"""Pure-2D GRIT-SPSE Graph Transformer for GPTrans-T."""
from __future__ import annotations

from dataclasses import replace
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch, Dataset
from torch_geometric.utils import to_dense_adj, to_dense_batch
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

EXTRA_BATCH_FIELDS = ("atom_spse",)


def _atom_spse_features(n_nodes: int, edge_index: torch.Tensor) -> torch.Tensor:
    if n_nodes == 0:
        return torch.empty((0, 6), dtype=torch.float32)
    adj = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)
    if edge_index.numel() > 0:
        u, v = edge_index[0].tolist(), edge_index[1].tolist()
        for a, b in zip(u, v):
            if a != b:
                adj[a, b] = 1.0
                adj[b, a] = 1.0
    p = adj @ adj
    cols = []
    for _ in range(3, 9):
        p = p @ adj
        cols.append(torch.log1p(torch.diagonal(p).clamp_min(0.0)))
    return torch.stack(cols, dim=-1)


class _SpseDataset(Dataset):
    def __init__(self, base) -> None:
        super().__init__()
        self.base = base

    def len(self) -> int:
        return len(self.base)

    def get(self, idx: int):
        g = self.base[idx].clone()
        g.atom_spse = _atom_spse_features(int(g.num_nodes), g.edge_index)
        return g


def attach_inputs(input_root, sidecar, data):
    return replace(
        data,
        train=_SpseDataset(data.train),
        development=_SpseDataset(data.development) if data.development is not None else None,
    )


def prepare_batch(batch, *, training: bool = False):
    if hasattr(batch, "atom_spse"):
        return batch
    items = []
    for g in batch.to_data_list():
        g = g.clone()
        g.atom_spse = _atom_spse_features(int(g.num_nodes), g.edge_index)
        items.append(g)
    tmp = Batch.from_data_list(items).to(batch.x.device)
    for field in EXTRA_BATCH_FIELDS:
        setattr(batch, field, getattr(tmp, field))
    return batch


class _SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.gate_up = nn.Linear(dim, 2 * hidden)
        self.down = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(g) * u)


class _GRITBlock(nn.Module):
    def __init__(self, d_node: int = 256, d_pair: int = 48, n_heads: int = 8, d_head: int = 16) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_attn = n_heads * d_head
        self.d_v = d_node // n_heads
        self.norm_n = nn.LayerNorm(d_node)
        self.norm_e = nn.LayerNorm(d_pair)
        self.w_q = nn.Linear(d_node, self.d_attn)
        self.w_k = nn.Linear(d_node, self.d_attn)
        self.w_e = nn.Linear(d_pair, self.d_attn, bias=False)
        self.w_v = nn.Linear(d_node, d_node)
        self.w_a = nn.Parameter(torch.randn(n_heads, d_head) * 0.02)
        self.theta_1 = nn.Parameter(torch.ones(d_node))
        self.theta_2 = nn.Parameter(torch.zeros(d_node))
        self.w_o = nn.Linear(d_node, d_node)
        self.norm_ehat = nn.LayerNorm(self.d_attn)
        self.w_e_out = nn.Linear(self.d_attn, d_pair)
        self.norm_ffn = nn.LayerNorm(d_node)
        self.ffn = _SwiGLU(d_node, 2 * d_node)

    def forward(self, node: torch.Tensor, pair: torch.Tensor, valid: torch.Tensor, deg: torch.Tensor):
        b, n, _ = node.shape
        h_n = self.norm_n(node)
        h_e = self.norm_e(pair)
        q = self.w_q(h_n).view(b, n, self.n_heads, self.d_head)
        k = self.w_k(h_n).view(b, n, self.n_heads, self.d_head)
        e_in = self.w_e(h_e).view(b, n, n, self.n_heads, self.d_head)
        u = q.unsqueeze(2) + k.unsqueeze(1) + e_in
        e_hat = (torch.sqrt(F.relu(u) + 1e-6) - 1e-3) - (torch.sqrt(F.relu(-u) + 1e-6) - 1e-3)
        logits = (e_hat * self.w_a).sum(dim=-1)
        logits = logits.masked_fill(~valid[:, None, :, None], -1e9)
        attn = torch.softmax(logits, dim=2)
        v = self.w_v(h_n).view(b, n, self.n_heads, self.d_v)
        m = torch.einsum("bijh,bjhd->bihd", attn, v).reshape(b, n, -1)
        m = m * self.theta_1 + torch.log1p(deg).unsqueeze(-1) * (m * self.theta_2)
        node = node + self.w_o(m)
        pair_mask = (valid.unsqueeze(1) & valid.unsqueeze(2)).unsqueeze(-1)
        pair = (pair + self.w_e_out(self.norm_ehat(e_hat.reshape(b, n, n, self.d_attn)))).masked_fill(~pair_mask, 0.0)
        node = (node + self.ffn(self.norm_ffn(node))).masked_fill(~valid.unsqueeze(-1), 0.0)
        return node, pair


class GRITSPSEModel(nn.Module):
    """Pure-2D GRIT-SPSE graph transformer with SignedSqrt edge attention and ADS."""

    accepts_prepared_batch = True

    def __init__(self, d_node: int = 256, d_pair: int = 48, n_layers: int = 8) -> None:
        super().__init__()
        self.d_node = d_node
        self.d_pair = d_pair
        self.atom_emb = AtomEncoder(d_node)
        self.bond_emb = BondEncoder(d_pair)
        self.spse_proj = nn.Linear(6, d_node)
        self.rrwp_proj = nn.Linear(16, d_pair)
        self.virt_node = nn.Parameter(torch.zeros(1, 1, d_node))
        self.virt_row = nn.Parameter(torch.zeros(1, 1, d_pair))
        self.virt_col = nn.Parameter(torch.zeros(1, 1, d_pair))
        self.virt_self = nn.Parameter(torch.zeros(1, 1, d_pair))
        self.blocks = nn.ModuleList([_GRITBlock(d_node, d_pair) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(d_node)
        self.readout = nn.Sequential(
            nn.Linear(3 * d_node, d_node),
            nn.SiLU(),
            nn.Linear(d_node, 1),
        )

    def forward(self, batch) -> torch.Tensor:
        batch = prepare_batch(batch)
        x_dense, real_valid = to_dense_batch(
            self.atom_emb(batch.x.long()) + self.spse_proj(batch.atom_spse), batch.batch
        )
        bsz, n_max, _ = x_dense.shape
        with torch.no_grad():
            if batch.edge_index.numel() > 0:
                adj = to_dense_adj(batch.edge_index, batch=batch.batch, max_num_nodes=n_max)
            else:
                adj = x_dense.new_zeros((bsz, n_max, n_max))
            adj = (adj + adj.transpose(1, 2) > 0).float()
            adj = adj * (1.0 - torch.eye(n_max, device=adj.device).unsqueeze(0))
            deg_real = adj.sum(dim=-1)
            p1 = adj / deg_real.clamp_min(1.0).unsqueeze(-1)
            walks = [p1]
            cur = p1
            for _ in range(15):
                cur = torch.bmm(cur, p1)
                walks.append(cur)
            rrwp = torch.stack(walks, dim=-1)
        node = torch.cat((self.virt_node.expand(bsz, 1, -1), x_dense), dim=1)
        valid = torch.cat((torch.ones((bsz, 1), dtype=torch.bool, device=node.device), real_valid), dim=1)
        deg = F.pad(deg_real, (1, 0), value=0.0)
        pair = node.new_zeros((bsz, n_max + 1, n_max + 1, self.d_pair))
        pair[:, 1:, 1:, :] = self.rrwp_proj(rrwp)
        if batch.edge_index.numel() > 0:
            edge_feat = self.bond_emb(batch.edge_attr.long())
            b_adj = to_dense_adj(batch.edge_index, batch=batch.batch, edge_attr=edge_feat, max_num_nodes=n_max)
            pair[:, 1:, 1:, :] = pair[:, 1:, 1:, :] + b_adj
        pair[:, 0, 1:, :] = self.virt_row.expand(bsz, n_max, -1)
        pair[:, 1:, 0, :] = self.virt_col.expand(bsz, n_max, -1)
        pair[:, 0, 0, :] = self.virt_self.squeeze(1)
        pair_mask = (valid.unsqueeze(1) & valid.unsqueeze(2)).unsqueeze(-1)
        pair = pair.masked_fill(~pair_mask, 0.0)
        for block in self.blocks:
            node, pair = block(node, pair, valid, deg)
        h = self.norm_out(node)
        h_real = h[:, 1:, :] * real_valid.unsqueeze(-1)
        cnt = real_valid.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        h_mean = h_real.sum(dim=1) / cnt
        h_sum = h_real.sum(dim=1) * 0.1
        return self.readout(torch.cat((h[:, 0, :], h_mean, h_sum), dim=-1))


def construct_family_model(recipe, _initial_state=None):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(recipe.seed))
        return GRITSPSEModel()


def policy(mode):
    from .family_training_policy import FamilyModePolicy

    if mode != "grit_spse":
        raise ValueError("Unknown GRIT-SPSE mode")
    return FamilyModePolicy(
        construct=construct_family_model,
        prepare_batch=prepare_batch,
        attach_inputs=attach_inputs,
        objective_name="degree_bond_local_ema999",
        standalone_model=True,
        extra_batch_fields=EXTRA_BATCH_FIELDS,
    )
