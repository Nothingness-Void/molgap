"""Self-contained high-throughput GPTrans-T (BondLocal + PairNorm) + optional Fast-3D + GPU-resident store."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import torch
from torch import nn
import torch.nn.functional as F

FROZEN_GPTRANS_INIT_SHA256 = "27590c1a928cafa8b633998e2291916b8f171d345ed5cb3b0234f0e9a37c3320"
EXPECTED_POS_MANIFEST_SHA256 = "81c445a0d8aaeba27f94c3df6d42c3989f292b789edc6458a64a20381f2e4526"
PROTECTED_ROW_START = 3_250_000
TRAIN_100K_MEAN_EV = 5.338300266634226
TRAIN_100K_STD_EV = 1.2750902063230574


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class FastDropPath(nn.Module):
    """Inductor-friendly per-graph stochastic depth."""

    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = float(probability)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.probability == 0.0 or not self.training:
            return value
        keep = 1.0 - self.probability
        shape = (value.shape[0],) + (1,) * (value.ndim - 1)
        mask = (torch.rand(shape, device=value.device) < keep).to(value.dtype)
        return value * (mask / keep)


class FastGraphPropagationAttention(nn.Module):
    """NHWC Linear GraphPropagationAttention mathematically identical to GraphPropagationAttention."""

    def __init__(
        self,
        node_channels: int,
        pair_channels: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if node_channels % num_heads:
            raise ValueError("node_channels must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_channels = node_channels // num_heads
        self.scale = self.head_channels**-0.5
        self.qkv = nn.Linear(node_channels, node_channels * 3)
        self.pair_to_attention = nn.Linear(pair_channels, num_heads)
        self.attention_to_pair = nn.Linear(num_heads, pair_channels)
        self.pair_to_node = nn.Linear(pair_channels, node_channels)
        self.output = nn.Linear(node_channels, node_channels)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        node: torch.Tensor,
        pair: torch.Tensor,
        key_padding_mask_b11n: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, node_count, channels = node.shape
        qkv = self.qkv(node).reshape(
            batch_size, node_count, 3, self.num_heads, self.head_channels
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        logits = (query @ key.transpose(-2, -1)) * self.scale
        bias = self.pair_to_attention(pair).permute(0, 3, 1, 2)
        logits = logits + bias
        residual_logits = logits

        logits_masked = logits.masked_fill(key_padding_mask_b11n, float("-inf"))
        attention = self.attention_dropout(
            torch.softmax(logits_masked, dim=-1, dtype=torch.float32).to(logits.dtype)
        )
        node_update = (attention @ value).transpose(1, 2).reshape(
            batch_size, node_count, channels
        )

        attn_plus_res = (attention + residual_logits).permute(0, 2, 3, 1)
        pair_update = self.attention_to_pair(attn_plus_res)
        kpm_nhwc = key_padding_mask_b11n.squeeze(1).unsqueeze(-1)  # (B, 1, N, 1)
        pw_logits = pair_update.masked_fill(kpm_nhwc, float("-inf"))
        pair_weights = torch.softmax(pw_logits, dim=2, dtype=torch.float32).to(pair_update.dtype)
        pair_message = (pair_weights * pair_update).sum(dim=2)
        node_update = node_update + self.pair_to_node(pair_message)
        return self.output_dropout(self.output(node_update)), pair_update


class FastGPTransPairNormBlock(nn.Module):
    """GPTransBlock with parameter-free pair_update_norm in NHWC layout."""

    def __init__(
        self,
        node_channels: int,
        pair_channels: int,
        num_heads: int,
        dropout: float,
        drop_path: float,
        layer_scale: float,
    ) -> None:
        super().__init__()
        self.pair_channels = int(pair_channels)
        self.node_norm1 = nn.LayerNorm(node_channels)
        self.attention = FastGraphPropagationAttention(
            node_channels, pair_channels, num_heads, dropout
        )
        self.node_norm2 = nn.LayerNorm(node_channels)
        self.ffn = nn.Sequential(
            nn.Linear(node_channels, node_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(node_channels, node_channels),
            nn.Dropout(dropout),
        )
        self.drop_path = FastDropPath(drop_path)
        self.attention_scale = nn.Parameter(
            torch.full((node_channels,), float(layer_scale))
        )
        self.ffn_scale = nn.Parameter(
            torch.full((node_channels,), float(layer_scale))
        )

    def forward(
        self,
        node: torch.Tensor,
        pair: torch.Tensor,
        key_padding_mask_b11n: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_update, pair_update = self.attention(
            self.node_norm1(node), pair, key_padding_mask_b11n
        )
        pair_normed = F.layer_norm(
            pair_update.float(), (self.pair_channels,)
        ).to(pair_update.dtype)
        pair = pair + pair_normed
        node = node + self.drop_path(self.attention_scale * node_update)
        node = node + self.drop_path(self.ffn_scale * self.ffn(self.node_norm2(node)))
        return node, pair


class FastBondLocalPairNormBlock(nn.Module):
    """BondLocal wrapper around FastGPTransPairNormBlock with static-shape trash-row scatter."""

    def __init__(self, core: FastGPTransPairNormBlock, channels: int = 256, pair_channels: int = 32) -> None:
        super().__init__()
        self.core = core
        self.norm = nn.LayerNorm(channels)
        self.message = nn.Sequential(nn.Linear(2 * channels + pair_channels, 64), nn.GELU())
        self.output = nn.Linear(64, channels)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        node: torch.Tensor,
        pair: torch.Tensor,
        key_padding_mask_b11n: torch.Tensor,
        edge_src_flat: torch.Tensor,
        edge_dst_flat: torch.Tensor,
        edge_pair_flat: torch.Tensor,
        edge_target_scatter: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, n_full, channels = node.shape
        clean_flat = self.norm(node).reshape(bsz * n_full, channels)
        pair_flat = pair.reshape(bsz * n_full * n_full, pair.shape[-1])
        msg_in = torch.cat(
            (clean_flat[edge_src_flat], clean_flat[edge_dst_flat], pair_flat[edge_pair_flat]),
            dim=-1,
        )
        message = self.output(self.message(msg_in))
        update = node.new_zeros((bsz * n_full + 1, channels))
        update.index_add_(0, edge_target_scatter, message.to(update.dtype))
        node = node + update[:-1].reshape(bsz, n_full, channels)
        return self.core(node, pair, key_padding_mask_b11n)


class Fast3DPairEncoder(nn.Module):
    """Zero-initialized pair-type affine Gaussian RBF + 3D neighbor centrality encoder."""

    def __init__(
        self,
        node_channels: int = 256,
        pair_channels: int = 32,
        rbf_bins: int = 64,
        rbf_max_a: float = 10.0,
        pos_noise_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.rbf_bins = int(rbf_bins)
        self.pos_noise_std = float(pos_noise_std)
        self.gamma_emb = nn.Embedding(119, 1)
        self.beta_emb = nn.Embedding(119, 1)
        nn.init.zeros_(self.gamma_emb.weight)
        nn.init.zeros_(self.beta_emb.weight)

        self.register_buffer("centers", torch.linspace(0.0, float(rbf_max_a), self.rbf_bins))
        self.sigma = float(rbf_max_a) / float(self.rbf_bins - 1)

        self.pair_mlp = nn.Sequential(
            nn.Linear(self.rbf_bins, pair_channels),
            nn.GELU(),
            nn.Linear(pair_channels, pair_channels),
        )
        nn.init.zeros_(self.pair_mlp[2].weight)
        nn.init.zeros_(self.pair_mlp[2].bias)

        self.centrality_proj = nn.Linear(self.rbf_bins, node_channels)
        nn.init.zeros_(self.centrality_proj.weight)
        nn.init.zeros_(self.centrality_proj.bias)

    def forward(
        self,
        atomic_nums: torch.Tensor,
        pos: torch.Tensor,
        atom_mask: torch.Tensor,
        geom_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, n_pad = atomic_nums.shape
        pos_f = pos.float()
        if self.training and self.pos_noise_std > 0.0:
            atom_gmask = (atom_mask & geom_valid[:, None]).unsqueeze(-1).to(pos_f.dtype)
            pos_f = pos_f + torch.randn_like(pos_f) * (self.pos_noise_std * atom_gmask)

        diff = pos_f.unsqueeze(2) - pos_f.unsqueeze(1)
        dist = torch.sqrt((diff * diff).sum(dim=-1).clamp_min(1e-8))

        z = atomic_nums.clamp(0, 118)
        g_raw = self.gamma_emb(z)  # (B, N, 1)
        b_raw = self.beta_emb(z)
        gamma = 1.0 + 0.5 * torch.tanh(g_raw.unsqueeze(2) + g_raw.unsqueeze(1))
        beta = b_raw.unsqueeze(2) + b_raw.unsqueeze(1)
        d_tilde = (gamma * dist.unsqueeze(-1) + beta).clamp_min(0.0)

        rbf = torch.exp(-0.5 * ((d_tilde - self.centers) / self.sigma) ** 2)
        eye = torch.eye(n_pad, dtype=torch.bool, device=atomic_nums.device).unsqueeze(0)
        pair_valid = atom_mask.unsqueeze(2) & atom_mask.unsqueeze(1) & (~eye) & geom_valid[:, None, None]
        rbf = rbf * pair_valid.unsqueeze(-1).to(rbf.dtype)

        pair_delta = self.pair_mlp(rbf) * pair_valid.unsqueeze(-1).to(rbf.dtype)
        centrality = rbf.sum(dim=2)
        atom_valid = (atom_mask & geom_valid[:, None]).unsqueeze(-1).to(rbf.dtype)
        node_delta = self.centrality_proj(centrality) * atom_valid
        return node_delta, pair_delta


class FastBondLocalPairNormGPTrans(nn.Module):
    """High-throughput NHWC static-bucketed BondLocal+PairNorm GPTrans-T with optional Fast-3D."""

    def __init__(
        self,
        node_channels: int = 256,
        pair_channels: int = 32,
        num_layers: int = 12,
        num_heads: int = 8,
        shortest_path_cap: int = 20,
        dropout: float = 0.1,
        drop_path: float = 0.1,
        layer_scale: float = 1.0,
        use_3d: bool = False,
        pos_noise_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.node_channels = int(node_channels)
        self.pair_channels = int(pair_channels)
        self.shortest_path_cap = int(shortest_path_cap)
        self.use_3d = bool(use_3d)

        self.atom_encoder = AtomEncoder(node_channels)
        self.bond_encoder = BondEncoder(pair_channels)
        self.in_degree_encoder = nn.Embedding(512, node_channels)
        self.out_degree_encoder = nn.Embedding(512, node_channels)
        self.spatial_encoder = nn.Embedding(shortest_path_cap + 2, pair_channels)
        self.graph_token = nn.Parameter(torch.empty(1, 1, node_channels))
        self.virtual_pair = nn.Parameter(torch.empty(1, 1, 1, pair_channels))
        self.node_norm = nn.LayerNorm(node_channels)
        self.input_dropout = nn.Dropout(dropout)

        path_rates = torch.linspace(0.0, drop_path, num_layers).tolist()
        self.blocks = nn.ModuleList(
            FastBondLocalPairNormBlock(
                FastGPTransPairNormBlock(
                    node_channels, pair_channels, num_heads, dropout, path_rates[i], layer_scale
                ),
                channels=node_channels,
                pair_channels=pair_channels,
            )
            for i in range(num_layers)
        )
        self.readout = nn.Sequential(
            nn.Linear(node_channels + pair_channels, node_channels),
            nn.LayerNorm(node_channels),
            nn.GELU(),
            nn.Linear(node_channels, 1),
        )
        nn.init.normal_(self.graph_token, std=0.02)
        nn.init.normal_(self.virtual_pair, std=0.02)

        self.encoder_3d = (
            Fast3DPairEncoder(
                node_channels=node_channels,
                pair_channels=pair_channels,
                pos_noise_std=pos_noise_std,
            )
            if self.use_3d
            else None
        )

    def load_from_gptrans_checkpoint(self, ckpt_path: Path) -> None:
        raw = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
        sd = raw.get("model_state_dict", raw.get("state_dict", raw.get("model", raw)))
        mapped = {}
        for k, v in sd.items():
            if k == "virtual_pair" and v.ndim == 4:
                mapped[k] = v.permute(0, 2, 3, 1).contiguous()
            elif (
                k.endswith("attention.pair_to_attention.weight")
                or k.endswith("attention.attention_to_pair.weight")
            ) and v.ndim == 4:
                mapped[k] = v.squeeze(-1).squeeze(-1).contiguous()
            else:
                mapped[k] = v
        missing, unexpected = self.load_state_dict(mapped, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected checkpoint keys: {unexpected}")
        non_3d_missing = [k for k in missing if not k.startswith("encoder_3d.")]
        if non_3d_missing:
            raise ValueError(f"Missing backbone keys from checkpoint: {non_3d_missing}")

    def forward(
        self,
        dense_x: torch.Tensor,
        degree: torch.Tensor,
        atom_mask: torch.Tensor,
        spd: torch.Tensor,
        edge_attr: torch.Tensor,
        bond_pair_idx: torch.Tensor,
        edge_src_flat: torch.Tensor,
        edge_dst_flat: torch.Tensor,
        edge_pair_flat: torch.Tensor,
        edge_target_scatter: torch.Tensor,
        pos: torch.Tensor | None = None,
        geom_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, n_pad, _ = dense_x.shape
        embedded = self.atom_encoder(dense_x.reshape(-1, dense_x.shape[-1])).reshape(
            bsz, n_pad, self.node_channels
        )
        embedded = embedded + self.in_degree_encoder(degree) + self.out_degree_encoder(degree)

        pair_atom = self.spatial_encoder(spd)  # (B, N_pad, N_pad, 32)
        bond_emb = self.bond_encoder(edge_attr)  # (E_pad, 32)
        pair_flat = pair_atom.new_zeros((bsz * n_pad * n_pad + 1, self.pair_channels))
        pair_flat.index_add_(0, bond_pair_idx, bond_emb.to(pair_flat.dtype))
        pair_atom = pair_atom + pair_flat[:-1].reshape(bsz, n_pad, n_pad, self.pair_channels)

        if self.encoder_3d is not None and pos is not None and geom_valid is not None:
            node_3d, pair_3d = self.encoder_3d(dense_x[:, :, 0], pos, atom_mask, geom_valid)
            embedded = embedded + node_3d.to(embedded.dtype)
            pair_atom = pair_atom + pair_3d.to(pair_atom.dtype)

        embedded = embedded * atom_mask.unsqueeze(-1).to(embedded.dtype)
        token = self.graph_token.expand(bsz, -1, -1).to(embedded.dtype)
        node = self.input_dropout(self.node_norm(torch.cat((token, embedded), dim=1)))

        full_pair = F.pad(pair_atom, (0, 0, 1, 0, 1, 0))
        vp = self.virtual_pair.to(full_pair.dtype)
        full_pair[:, 0:1, :, :] = full_pair[:, 0:1, :, :] + vp
        full_pair[:, 1:, 0:1, :] = full_pair[:, 1:, 0:1, :] + vp

        full_mask = torch.cat(
            (torch.ones((bsz, 1), dtype=torch.bool, device=atom_mask.device), atom_mask), dim=1
        )
        kpm = (~full_mask)[:, None, None, :]

        for block in self.blocks:
            node, full_pair = block(
                node, full_pair, kpm, edge_src_flat, edge_dst_flat, edge_pair_flat, edge_target_scatter
            )

        graph_state = torch.cat((node[:, 0], full_pair[:, 0, 0, :]), dim=-1)
        return self.readout(graph_state).squeeze(-1)


@torch.no_grad()
def _compute_spd_batch_gpu(
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_graph: torch.Tensor,
    n_counts: torch.Tensor,
    cap: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact shortest-path matrices on GPU for a chunk of graphs."""
    bsz = int(n_counts.shape[0])
    m = int(n_counts.max().item())
    dev = n_counts.device
    adj = torch.zeros((bsz, m, m), dtype=torch.bool, device=dev)
    if edge_u.numel() > 0:
        adj[edge_graph, edge_u, edge_v] = True
    mask = torch.arange(m, device=dev)[None, :] < n_counts[:, None]
    pair_mask = mask[:, None, :] & mask[:, :, None]
    unreachable = cap + 1
    dist = torch.full((bsz, m, m), unreachable, dtype=torch.long, device=dev)
    diag = torch.eye(m, dtype=torch.bool, device=dev).unsqueeze(0)
    dist.masked_fill_(diag & pair_mask, 0)
    frontier = adj
    adj_f = adj.float()
    last_step = min(cap, max(1, m - 1))
    for step in range(1, last_step + 1):
        dist.masked_fill_(frontier & (dist == unreachable) & pair_mask, step)
        if step < last_step:
            frontier = torch.bmm(frontier.float(), adj_f) > 0
    return dist.to(torch.uint8), pair_mask


class GPUResidentGraphStore:
    """Compact GPU-resident PCQM store with precomputed uint8 SPD and optional ETKDG 3D coordinates."""

    def __init__(
        self,
        shard_paths: list[Path],
        *,
        device: str | torch.device = "cuda",
        expected_source_start: int | None = None,
        expected_source_stop: int | None = None,
        pos_tables: list[tuple[int, int, dict]] | None = None,
        shortest_path_cap: int = 20,
    ) -> None:
        self.device = torch.device(device)
        xs, degrees, ys, src_idxs = [], [], [], []
        eus, evs, eas = [], [], []
        n_counts_list, e_counts_list = [], []
        pos_chunks, gv_chunks = [], []
        has_pos = pos_tables is not None

        for spath in shard_paths:
            data, slices = torch.load(Path(spath), map_location="cpu", weights_only=False)
            n_graphs = int(slices["x"].shape[0] - 1)
            sidx = data.source_idx.view(-1).long()
            if int(sidx.max().item()) >= PROTECTED_ROW_START:
                raise ValueError(
                    f"Protected split boundary violated: max source_idx={int(sidx.max())} >= {PROTECTED_ROW_START}"
                )
            src_idxs.append(sidx)
            ys.append(data.y.view(-1).float())

            x_ptr = slices["x"].long()
            e_ptr = slices["edge_index"].long()
            nc = x_ptr[1:] - x_ptr[:-1]
            ec = e_ptr[1:] - e_ptr[:-1]
            n_counts_list.append(nc)
            e_counts_list.append(ec)

            xs.append(data.x.to(torch.int16))
            ea_shard = data.edge_attr.to(torch.int16)
            eas.append(ea_shard)

            # Edge indices inside PyG shards are already graph-local [0..n_i-1].
            node_offset_per_edge = torch.repeat_interleave(x_ptr[:-1], ec)
            u_local = data.edge_index[0].long()
            v_local = data.edge_index[1].long()
            eus.append(u_local.to(torch.int16))
            evs.append(v_local.to(torch.int16))

            deg = torch.zeros((data.x.shape[0],), dtype=torch.int16)
            if data.edge_index.shape[1] > 0:
                deg.scatter_add_(0, u_local + node_offset_per_edge, torch.ones((data.edge_index.shape[1],), dtype=torch.int16))
            degrees.append(deg)

            if has_pos:
                pos_list_shard = []
                gv_list_shard = []
                # Vectorized fast lookup when possible, row loop fallback exact
                for r in range(n_graphs):
                    sid = int(sidx[r].item())
                    n_r = int(nc[r].item())
                    found = False
                    for t_start, t_stop, tbl in pos_tables:
                        if t_start <= sid < t_stop:
                            row = sid - t_start
                            ps, pe = int(tbl["node_ptr"][row].item()), int(tbl["node_ptr"][row + 1].item())
                            p_t = tbl["pos"][ps:pe].float()
                            if p_t.shape[0] != n_r:
                                raise ValueError(f"pos count mismatch at source_idx={sid}: {p_t.shape[0]} vs {n_r}")
                            pos_list_shard.append(p_t)
                            gv_list_shard.append(bool(tbl["pos_valid"][row].item()))
                            found = True
                            break
                    if not found:
                        raise KeyError(f"source_idx={sid} outside pos_manifest bounds")
                pos_chunks.append(torch.cat(pos_list_shard, dim=0))
                gv_chunks.append(torch.tensor(gv_list_shard, dtype=torch.bool))

        self.source_idx = torch.cat(src_idxs, dim=0)
        if expected_source_start is not None and int(self.source_idx.min().item()) < expected_source_start:
            raise ValueError(f"Unexpected source_idx start {int(self.source_idx.min())} < {expected_source_start}")
        if expected_source_stop is not None and int(self.source_idx.max().item()) >= expected_source_stop:
            raise ValueError(f"Unexpected source_idx stop {int(self.source_idx.max())} >= {expected_source_stop}")

        self.n_graphs = int(self.source_idx.shape[0])
        n_counts_cpu = torch.cat(n_counts_list, dim=0)
        e_counts_cpu = torch.cat(e_counts_list, dim=0)
        x_cpu = torch.cat(xs, dim=0)
        deg_cpu = torch.cat(degrees, dim=0)
        eu_cpu = torch.cat(eus, dim=0)
        ev_cpu = torch.cat(evs, dim=0)
        ea_cpu = torch.cat(eas, dim=0)
        y_cpu = torch.cat(ys, dim=0)

        node_ptr_cpu = torch.zeros(self.n_graphs + 1, dtype=torch.long)
        node_ptr_cpu[1:] = torch.cumsum(n_counts_cpu, 0)
        edge_ptr_cpu = torch.zeros(self.n_graphs + 1, dtype=torch.long)
        edge_ptr_cpu[1:] = torch.cumsum(e_counts_cpu, 0)
        spd_counts_cpu = n_counts_cpu * n_counts_cpu
        spd_ptr_cpu = torch.zeros(self.n_graphs + 1, dtype=torch.long)
        spd_ptr_cpu[1:] = torch.cumsum(spd_counts_cpu, 0)

        # Precompute flattened per-molecule uint8 SPD matrices on GPU in chunks
        spd_flat_cpu = torch.empty((int(spd_ptr_cpu[-1].item()),), dtype=torch.uint8)
        chunk_g = 4096
        for start in range(0, self.n_graphs, chunk_g):
            stop = min(self.n_graphs, start + chunk_g)
            nc_c = n_counts_cpu[start:stop].to(self.device)
            ec_c = e_counts_cpu[start:stop].to(self.device)
            es, ee = int(edge_ptr_cpu[start].item()), int(edge_ptr_cpu[stop].item())
            g_edge = torch.repeat_interleave(torch.arange(stop - start, device=self.device), ec_c)
            dist_u8, pmask = _compute_spd_batch_gpu(
                eu_cpu[es:ee].long().to(self.device),
                ev_cpu[es:ee].long().to(self.device),
                g_edge,
                nc_c,
                cap=shortest_path_cap,
            )
            ss, se = int(spd_ptr_cpu[start].item()), int(spd_ptr_cpu[stop].item())
            spd_flat_cpu[ss:se] = dist_u8[pmask].cpu()

        # Upload everything to GPU once
        self.y = y_cpu.to(self.device)
        self.x = x_cpu.to(self.device)
        self.degree = deg_cpu.to(self.device)
        self.edge_u = eu_cpu.to(self.device)
        self.edge_v = ev_cpu.to(self.device)
        self.edge_attr = ea_cpu.to(self.device)
        self.spd_flat = spd_flat_cpu.to(self.device)

        self.node_ptr = node_ptr_cpu.to(self.device)
        self.n_counts = n_counts_cpu.to(self.device)
        self.edge_ptr = edge_ptr_cpu.to(self.device)
        self.e_counts = e_counts_cpu.to(self.device)
        self.spd_ptr = spd_ptr_cpu.to(self.device)
        self.spd_counts = spd_counts_cpu.to(self.device)

        self.has_pos = has_pos
        if has_pos:
            self.pos = torch.cat(pos_chunks, dim=0).to(self.device)
            self.geom_valid = torch.cat(gv_chunks, dim=0).to(self.device)
        else:
            self.pos = None
            self.geom_valid = None
        self.max_store_n = int(n_counts_cpu.max().item())

    def __len__(self) -> int:
        return self.n_graphs

    def nbytes(self) -> int:
        tensors = [
            self.y, self.x, self.degree, self.edge_u, self.edge_v, self.edge_attr,
            self.spd_flat, self.node_ptr, self.n_counts, self.edge_ptr, self.e_counts,
            self.spd_ptr, self.spd_counts,
        ]
        if self.pos is not None:
            tensors.extend([self.pos, self.geom_valid])
        return sum(t.numel() * t.element_size() for t in tensors)

    @torch.no_grad()
    def get_batch(
        self,
        indices: torch.Tensor,
        *,
        pad_to_n: int = 20,
        e_bucket: int = 256,
        unreachable_val: int = 21,
    ) -> dict[str, torch.Tensor | None]:
        idx = indices.to(self.device, dtype=torch.long)
        bsz = int(idx.numel())
        nc = self.n_counts[idx]
        ec = self.e_counts[idx]
        max_n_obs = int(nc.max().item())
        n_pad = max(int(pad_to_n), max_n_obs)

        total_n = int(nc.sum().item())
        total_e = int(ec.sum().item())
        e_pad = max(int(e_bucket), ((max(total_e, 1) + e_bucket - 1) // e_bucket) * e_bucket)

        # Nodes placement
        n_start = self.node_ptr[idx]
        n_off = torch.cumsum(nc, 0) - nc
        n_src = torch.repeat_interleave(n_start - n_off, nc, output_size=total_n) + torch.arange(total_n, device=self.device)
        graph_of_node = torch.repeat_interleave(torch.arange(bsz, device=self.device), nc, output_size=total_n)
        local_of_node = torch.arange(total_n, device=self.device) - torch.repeat_interleave(n_off, nc, output_size=total_n)
        node_dst_flat = graph_of_node * n_pad + local_of_node

        dense_x = torch.zeros((bsz * n_pad, 9), dtype=torch.long, device=self.device)
        dense_x[node_dst_flat] = self.x[n_src].long()
        dense_x = dense_x.view(bsz, n_pad, 9)

        degree = torch.zeros((bsz * n_pad,), dtype=torch.long, device=self.device)
        degree[node_dst_flat] = self.degree[n_src].long().clamp_max(511)
        degree = degree.view(bsz, n_pad)

        atom_mask = torch.arange(n_pad, device=self.device)[None, :] < nc[:, None]

        if self.has_pos and self.pos is not None and self.geom_valid is not None:
            pos_dense = torch.zeros((bsz * n_pad, 3), dtype=torch.float32, device=self.device)
            pos_dense[node_dst_flat] = self.pos[n_src]
            pos_dense = pos_dense.view(bsz, n_pad, 3)
            gv = self.geom_valid[idx]
        else:
            pos_dense, gv = None, None

        # SPD placement (flat n_i*n_i per molecule -> (B, N_pad, N_pad))
        sc = self.spd_counts[idx]
        total_s = int(sc.sum().item())
        s_start = self.spd_ptr[idx]
        s_off = torch.cumsum(sc, 0) - sc
        s_src = torch.repeat_interleave(s_start - s_off, sc, output_size=total_s) + torch.arange(total_s, device=self.device)
        s_graph = torch.repeat_interleave(torch.arange(bsz, device=self.device), sc, output_size=total_s)
        s_n = torch.repeat_interleave(nc, sc, output_size=total_s)
        s_local = torch.arange(total_s, device=self.device) - torch.repeat_interleave(s_off, sc, output_size=total_s)
        s_row = torch.div(s_local, s_n, rounding_mode="floor")
        s_col = s_local - s_row * s_n
        spd_dst_flat = s_graph * (n_pad * n_pad) + s_row * n_pad + s_col

        spd_dense = torch.full((bsz * n_pad * n_pad,), unreachable_val, dtype=torch.long, device=self.device)
        spd_dense[spd_dst_flat] = self.spd_flat[s_src].long()
        spd_dense = spd_dense.view(bsz, n_pad, n_pad)

        # Edge placement into static bucket E_pad
        edge_attr = torch.zeros((e_pad, 3), dtype=torch.long, device=self.device)
        bond_pair_idx = torch.full((e_pad,), bsz * n_pad * n_pad, dtype=torch.long, device=self.device)
        edge_src_flat = torch.zeros((e_pad,), dtype=torch.long, device=self.device)
        edge_dst_flat = torch.zeros((e_pad,), dtype=torch.long, device=self.device)
        edge_pair_flat = torch.zeros((e_pad,), dtype=torch.long, device=self.device)
        edge_target_scatter = torch.full((e_pad,), bsz * (n_pad + 1), dtype=torch.long, device=self.device)

        if total_e > 0:
            e_start = self.edge_ptr[idx]
            e_off = torch.cumsum(ec, 0) - ec
            e_src = torch.repeat_interleave(e_start - e_off, ec, output_size=total_e) + torch.arange(total_e, device=self.device)
            e_graph = torch.repeat_interleave(torch.arange(bsz, device=self.device), ec, output_size=total_e)
            eu = self.edge_u[e_src].long()
            ev = self.edge_v[e_src].long()
            n_full = n_pad + 1

            edge_attr[:total_e] = self.edge_attr[e_src].long()
            bond_pair_idx[:total_e] = e_graph * (n_pad * n_pad) + eu * n_pad + ev
            edge_src_flat[:total_e] = e_graph * n_full + (eu + 1)
            edge_dst_flat[:total_e] = e_graph * n_full + (ev + 1)
            edge_pair_flat[:total_e] = e_graph * (n_full * n_full) + (eu + 1) * n_full + (ev + 1)
            edge_target_scatter[:total_e] = e_graph * n_full + (ev + 1)

        return {
            "dense_x": dense_x,
            "degree": degree,
            "atom_mask": atom_mask,
            "spd": spd_dense,
            "edge_attr": edge_attr,
            "bond_pair_idx": bond_pair_idx,
            "edge_src_flat": edge_src_flat,
            "edge_dst_flat": edge_dst_flat,
            "edge_pair_flat": edge_pair_flat,
            "edge_target_scatter": edge_target_scatter,
            "pos": pos_dense,
            "geom_valid": gv,
            "y": self.y[idx],
        }


def load_pos_tables(pos_manifest_path: Path, expected_manifest_sha256: str | None = EXPECTED_POS_MANIFEST_SHA256) -> list[tuple[int, int, dict]]:
    pos_manifest_path = Path(pos_manifest_path)
    if expected_manifest_sha256 is not None and sha256_file(pos_manifest_path) != expected_manifest_sha256:
        raise ValueError(f"pos_manifest.json SHA256 mismatch: {pos_manifest_path}")
    manifest = json.loads(pos_manifest_path.read_text(encoding="utf-8"))
    pos_dir = pos_manifest_path.parent
    tables: list[tuple[int, int, dict]] = []
    for role_name, meta in manifest["roles"].items():
        pt_path = pos_dir / meta["file"]
        if sha256_file(pt_path) != meta["sha256"]:
            raise ValueError(f"SHA256 mismatch for {pt_path}")
        tbl = torch.load(pt_path, map_location="cpu", weights_only=False)
        tables.append((int(meta["source_idx_start"]), int(meta["source_idx_stop"]), tbl))
    return tables
