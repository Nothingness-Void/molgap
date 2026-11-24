"""Self-contained regularized & accelerated DeMol-CIN (Pure-2D + optional Fast-3D) with precomputed scatter denominators."""
from __future__ import annotations

from collections import deque
from pathlib import Path
from types import SimpleNamespace

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data

PROTECTED_ROW_START = 3_250_000
MAX_RING_SIZE = 8
MAX_RINGS = 16


def _find_chordless_rings(n_nodes: int, bonds: list[tuple[int, int]], bond_map: dict[tuple[int, int], int]) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(n_nodes)]
    for u, v in bonds:
        adj[u].append(v)
        adj[v].append(u)
    seen: set[frozenset[int]] = set()
    rings: list[list[int]] = []
    for u, v in bonds:
        dist = [-1] * n_nodes
        parent = [-1] * n_nodes
        dist[u] = 0
        q: deque[int] = deque([u])
        while q:
            cur = q.popleft()
            if dist[cur] >= MAX_RING_SIZE - 1:
                continue
            for nxt in adj[cur]:
                if (cur == u and nxt == v) or (cur == v and nxt == u):
                    continue
                if dist[nxt] == -1:
                    dist[nxt] = dist[cur] + 1
                    parent[nxt] = cur
                    if nxt == v:
                        q.clear()
                        break
                    q.append(nxt)
        if dist[v] < 2:
            continue
        cyc, cur = [], v
        while cur != -1:
            cyc.append(cur)
            cur = parent[cur]
        k = len(cyc)
        if not (3 <= k <= MAX_RING_SIZE):
            continue
        key = frozenset(cyc)
        if key in seen:
            continue
        if not any(
            (min(cyc[i], cyc[j]), max(cyc[i], cyc[j])) in bond_map
            for i in range(k)
            for j in range(i + 2, k)
            if not (i == 0 and j == k - 1)
        ):
            seen.add(key)
            rings.append(cyc)
    rings.sort(key=lambda r: (len(r), tuple(sorted(r))))
    return rings[:MAX_RINGS]


def _spse_counts(n_nodes: int, bonds: list[tuple[int, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    if n_nodes == 0:
        return torch.empty((0, 6), dtype=torch.float32), torch.empty((0, 6), dtype=torch.float32)
    adj = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)
    for u, v in bonds:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    p = adj @ adj
    a_cols, b_cols = [], []
    u_idx = torch.tensor([b[0] for b in bonds], dtype=torch.long) if bonds else None
    v_idx = torch.tensor([b[1] for b in bonds], dtype=torch.long) if bonds else None
    for _ in range(3, 9):
        if u_idx is not None:
            b_cols.append(torch.log1p(p[u_idx, v_idx].clamp_min(0.0)))
        p = p @ adj
        a_cols.append(torch.log1p(torch.diagonal(p).clamp_min(0.0)))
    atom_spse = torch.stack(a_cols, dim=-1)
    bond_spse = torch.stack(b_cols, dim=-1) if b_cols else torch.empty((0, 6), dtype=torch.float32)
    return atom_spse, bond_spse


def extract_cellular_parts(x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> dict[str, torch.Tensor]:
    n_nodes = int(x.shape[0])
    ei = edge_index.tolist() if edge_index.numel() else [[], []]
    ea = edge_attr.tolist() if edge_attr is not None and edge_attr.numel() else []
    undir: dict[tuple[int, int], list[int]] = {}
    for idx, (u, v) in enumerate(zip(ei[0], ei[1])):
        if u != v:
            pair = (u, v) if u < v else (v, u)
            if pair not in undir:
                undir[pair] = ea[idx] if idx < len(ea) else [0, 0, 0]
    bonds = sorted(undir.keys())
    bond_map = {b: i for i, b in enumerate(bonds)}
    e_u = len(bonds)
    atom_spse, bond_spse = _spse_counts(n_nodes, bonds)
    if e_u > 0:
        bond_attr = torch.tensor([undir[b] for b in bonds], dtype=torch.int16)
        bond_to_atom = torch.tensor([[b[0], b[1]] for b in bonds], dtype=torch.int16)
    else:
        bond_attr = torch.empty((0, 3), dtype=torch.int16)
        bond_to_atom = torch.empty((0, 2), dtype=torch.int16)

    inc_bonds: list[list[int]] = [[] for _ in range(n_nodes)]
    for b_id, (u, v) in enumerate(bonds):
        inc_bonds[u].append(b_id)
        inc_bonds[v].append(b_id)
    lg_rows, lg_c = [], []
    for c, blist in enumerate(inc_bonds):
        for b1 in blist:
            for b2 in blist:
                if b1 != b2:
                    lg_rows.append((b1, b2))
                    lg_c.append(c)
    lg_edge_index = torch.tensor(lg_rows, dtype=torch.int16) if lg_rows else torch.empty((0, 2), dtype=torch.int16)
    lg_center_atom = torch.tensor(lg_c, dtype=torch.int16).unsqueeze(1) if lg_c else torch.empty((0, 1), dtype=torch.int16)

    rings = _find_chordless_rings(n_nodes, bonds, bond_map)
    if rings:
        ring_size = torch.tensor([[len(r)] for r in rings], dtype=torch.int8)
        r_a, r_b = [], []
        for r_id, cyc in enumerate(rings):
            k = len(cyc)
            for i, a in enumerate(cyc):
                nxt = cyc[(i + 1) % k]
                r_a.append((a, r_id))
                r_b.append((bond_map[(a, nxt) if a < nxt else (nxt, a)], r_id))
        ring_to_atom = torch.tensor(r_a, dtype=torch.int16)
        ring_to_bond = torch.tensor(r_b, dtype=torch.int16)
    else:
        ring_size = torch.empty((0, 1), dtype=torch.int8)
        ring_to_atom = torch.empty((0, 2), dtype=torch.int16)
        ring_to_bond = torch.empty((0, 2), dtype=torch.int16)

    return {
        "x": x.to(torch.int16),
        "atom_spse": atom_spse.float(),
        "bond_attr": bond_attr,
        "bond_spse": bond_spse.float(),
        "bond_to_atom": bond_to_atom,
        "lg_edge_index": lg_edge_index,
        "lg_center_atom": lg_center_atom,
        "ring_size": ring_size,
        "ring_to_atom": ring_to_atom,
        "ring_to_bond": ring_to_bond,
    }


def _scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = src.new_zeros((dim_size, src.shape[1]))
    if src.numel() > 0:
        out.index_add_(0, index, src.to(out.dtype))
    return out


def _inv_count(index: torch.Tensor, dim_size: int, device: torch.device) -> torch.Tensor:
    cnt = torch.zeros((dim_size, 1), dtype=torch.float32, device=device)
    if index.numel() > 0:
        cnt.index_add_(0, index, torch.ones((index.shape[0], 1), dtype=torch.float32, device=device))
    return 1.0 / cnt.clamp_min(1.0)


class _RegMLP(nn.Module):
    def __init__(self, in_d: int, hid_d: int, out_d: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_d, hid_d)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hid_d, out_d)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.drop1(F.silu(self.fc1(x)))))


class _RegSwiGLU(nn.Module):
    def __init__(self, d: int, hid: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.w12 = nn.Linear(d, 2 * hid)
        self.w3 = nn.Linear(hid, d)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = self.w12(self.norm(x.float()).to(x.dtype)).chunk(2, dim=-1)
        return self.drop2(self.w3(self.drop1(F.silu(u) * v)))


class FastRegCellBlock(nn.Module):
    """Regularized GNN+ Pre-LN SwiGLU cellular block with precomputed scatter denominators and optional 3D LG."""

    def __init__(
        self,
        d_a: int = 144,
        d_b: int = 112,
        d_r: int = 112,
        d_v: int = 144,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        use_3d: bool = False,
    ) -> None:
        super().__init__()
        self.drop_path_prob = float(drop_path)
        self.use_3d = bool(use_3d)
        self.na = nn.LayerNorm(d_a)
        self.nb = nn.LayerNorm(d_b)
        self.nr = nn.LayerNorm(d_r)
        self.nv = nn.LayerNorm(d_v)
        self.b2a = _RegMLP(d_b + 2 * d_a, d_a, d_a, dropout=dropout)
        self.a2b = _RegMLP(d_b + 2 * d_a, d_b, d_b, dropout=dropout)
        self.lg_msg = nn.Linear(d_b + d_a, d_b)
        self.lg_gate = nn.Sequential(nn.Linear(2 * d_b + 2 * d_a, d_b), nn.Sigmoid())
        self.lg_out = nn.Sequential(nn.Linear(d_b, d_b), nn.Dropout(dropout))
        if self.use_3d:
            self.lg_3d_gate = nn.Linear(48, d_b)
            nn.init.zeros_(self.lg_3d_gate.weight)
            nn.init.zeros_(self.lg_3d_gate.bias)
        else:
            self.lg_3d_gate = None
        self.r_in = _RegMLP(d_r + d_a + d_b, d_r, d_r, dropout=dropout)
        self.r2a = nn.Linear(d_r, d_a)
        self.r2b = nn.Linear(d_r, d_b)
        self.v_upd = _RegMLP(d_v + d_a + d_b + d_r, d_v, d_v, dropout=dropout)
        self.v2a = nn.Linear(d_v, d_a)
        self.v2b = nn.Linear(d_v, d_b)
        self.v2r = nn.Linear(d_v, d_r)
        self.ffn_a = _RegSwiGLU(d_a, int(1.5 * d_a), dropout=dropout)
        self.ffn_b = _RegSwiGLU(d_b, int(1.5 * d_b), dropout=dropout)
        self.ffn_r = _RegSwiGLU(d_r, int(1.5 * d_r), dropout=dropout)

    def _graph_drop(self, bsz: int, batch_a: torch.Tensor, batch_b: torch.Tensor, batch_r: torch.Tensor, device: torch.device, dtype: torch.dtype):
        if self.drop_path_prob == 0.0 or not self.training:
            return None
        keep = 1.0 - self.drop_path_prob
        g = ((torch.rand((bsz, 1), device=device) < keep).to(dtype)) / keep
        return g[batch_a], g[batch_b], g[batch_r], g

    def forward(self, h_a, h_b, h_r, h_v, batch, denoms: dict[str, torch.Tensor], lg_3d: torch.Tensor | None):
        bsz = denoms["bsz"]
        na = self.na(h_a.float()).to(h_a.dtype)
        nb = self.nb(h_b.float()).to(h_b.dtype)
        nr = self.nr(h_r.float()).to(h_r.dtype)
        nv = self.nv(h_v.float()).to(h_v.dtype)
        n_a, n_b, n_r = na.shape[0], nb.shape[0], nr.shape[0]
        dp = self._graph_drop(bsz, batch.batch, batch.bond_batch, batch.ring_batch, h_a.device, h_a.dtype)

        if n_b > 0:
            u, v = batch.bond_to_atom[0], batch.bond_to_atom[1]
            pair_a = torch.cat((nb, na[u] + na[v], (na[u] - na[v]).abs()), dim=-1)
            mb = self.b2a(pair_a)
            da = _scatter_add(mb, u, n_a) + _scatter_add(mb, v, n_a)
            db = self.a2b(pair_a)
            if dp is not None:
                da = da * dp[0]
                db = db * dp[1]
            h_a = h_a + da
            h_b = h_b + db

        if batch.lg_edge_index.numel() > 0:
            src, dst = batch.lg_edge_index[0], batch.lg_edge_index[1]
            c = batch.lg_center_atom
            other_src = u[src] + v[src] - c
            other_dst = u[dst] + v[dst] - c
            dir_diff = na[other_dst] - na[other_src]
            gate_in = torch.cat((nb[src], nb[dst], na[c], dir_diff), dim=-1)
            gate = self.lg_gate(gate_in)
            msg = self.lg_msg(torch.cat((nb[src], na[other_src] - na[c]), dim=-1))
            if self.lg_3d_gate is not None and lg_3d is not None:
                msg = msg + self.lg_3d_gate(lg_3d).to(msg.dtype)
            dlg = self.lg_out(_scatter_add(gate * msg, dst, n_b))
            if dp is not None:
                dlg = dlg * dp[1]
            h_b = h_b + dlg

        if n_r > 0:
            ra_a, ra_r = batch.ring_to_atom[0], batch.ring_to_atom[1]
            rb_b, rb_r = batch.ring_to_bond[0], batch.ring_to_bond[1]
            mr_a = (_scatter_add(na[ra_a], ra_r, n_r) * denoms["ra_r"]).to(nr.dtype)
            mr_b = (_scatter_add(nb[rb_b], rb_r, n_r) * denoms["rb_r"]).to(nr.dtype)
            dr = self.r_in(torch.cat((nr, mr_a, mr_b), dim=-1))
            if dp is not None:
                dr = dr * dp[2]
            h_r = h_r + dr
            nr2 = self.nr(h_r.float()).to(h_r.dtype)
            dra = (_scatter_add(self.r2a(nr2[ra_r]), ra_a, n_a) * denoms["ra_a"]).to(h_a.dtype)
            drb = (_scatter_add(self.r2b(nr2[rb_r]), rb_b, n_b) * denoms["rb_b"]).to(h_b.dtype)
            if dp is not None:
                dra = dra * dp[0]
                drb = drb * dp[1]
            h_a = h_a + dra
            h_b = h_b + drb

        va = (_scatter_add(na, batch.batch, bsz) * denoms["va"]).to(nv.dtype)
        vb = (_scatter_add(nb, batch.bond_batch, bsz) * denoms["vb"]).to(nv.dtype)
        vr = (_scatter_add(nr, batch.ring_batch, bsz) * denoms["vr"]).to(nv.dtype)
        dv = self.v_upd(torch.cat((nv, va, vb, vr), dim=-1))
        if dp is not None:
            dv = dv * dp[3]
        h_v = h_v + dv
        nv2 = self.nv(h_v.float()).to(h_v.dtype)
        dva = self.v2a(nv2)[batch.batch]
        if dp is not None:
            dva = dva * dp[0]
        h_a = h_a + dva
        if n_b > 0:
            dvb = self.v2b(nv2)[batch.bond_batch]
            if dp is not None:
                dvb = dvb * dp[1]
            h_b = h_b + dvb
        if n_r > 0:
            dvr = self.v2r(nv2)[batch.ring_batch]
            if dp is not None:
                dvr = dvr * dp[2]
            h_r = h_r + dvr

        fa = self.ffn_a(h_a)
        fb = self.ffn_b(h_b)
        fr = self.ffn_r(h_r)
        if dp is not None:
            fa = fa * dp[0]
            fb = fb * dp[1]
            fr = fr * dp[2]
        return h_a + fa, h_b + fb, h_r + fr, h_v


class FastDeMolCINReg(nn.Module):
    """Regularized & accelerated DeMol-CIN (Dropout + DropPath + cached scatters + optional Fast-3D)."""

    def __init__(
        self,
        d_a: int = 144,
        d_b: int = 112,
        d_r: int = 112,
        d_v: int = 144,
        num_layers: int = 8,
        dropout: float = 0.1,
        drop_path: float = 0.1,
        use_3d: bool = False,
        pos_noise_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.use_3d = bool(use_3d)
        self.pos_noise_std = float(pos_noise_std)
        self.atom_emb = AtomEncoder(d_a)
        self.bond_emb = BondEncoder(d_b)
        self.atom_spse_proj = nn.Linear(6, d_a)
        self.bond_spse_proj = nn.Linear(6, d_b)
        self.ring_emb = nn.Embedding(MAX_RING_SIZE + 1, d_r)
        self.virt_init = nn.Parameter(torch.zeros(1, d_v))
        nn.init.normal_(self.virt_init, std=0.02)
        self.input_drop = nn.Dropout(dropout)

        if self.use_3d:
            self.register_buffer("bond_rbf_c", torch.linspace(0.5, 4.0, 32))
            self.bond_rbf_inv_2s2 = -0.5 / ((3.5 / 31.0) ** 2)
            self.bond_3d_proj = nn.Linear(32, d_b)
            nn.init.zeros_(self.bond_3d_proj.weight)
            nn.init.zeros_(self.bond_3d_proj.bias)

            self.register_buffer("ang_rbf_c", torch.linspace(-1.0, 1.0, 16))
            self.ang_rbf_inv_2s2 = -0.5 / ((2.0 / 15.0) ** 2)
            self.register_buffer("nb13_rbf_c", torch.linspace(1.0, 6.0, 32))
            self.nb13_rbf_inv_2s2 = -0.5 / ((5.0 / 31.0) ** 2)

            self.register_buffer("env_rbf_c", torch.linspace(0.5, 5.0, 32))
            self.env_rbf_inv_2s2 = -0.5 / ((4.5 / 31.0) ** 2)
            self.atom_3d_proj = nn.Linear(32, d_a)
            nn.init.zeros_(self.atom_3d_proj.weight)
            nn.init.zeros_(self.atom_3d_proj.bias)

        dp_rates = torch.linspace(0.0, drop_path, num_layers).tolist()
        self.blocks = nn.ModuleList(
            FastRegCellBlock(d_a, d_b, d_r, d_v, dropout=dropout, drop_path=dp_rates[i], use_3d=use_3d)
            for i in range(num_layers)
        )
        self.norm_a = nn.LayerNorm(d_a)
        self.norm_b = nn.LayerNorm(d_b)
        self.norm_r = nn.LayerNorm(d_r)
        self.norm_v = nn.LayerNorm(d_v)
        self.readout = nn.Sequential(
            nn.LayerNorm(2 * d_a + d_b + d_r + d_v),
            nn.Linear(2 * d_a + d_b + d_r + d_v, d_a),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_a, 1),
        )

    def forward(self, batch) -> torch.Tensor:
        bsz = int(batch.num_graphs)
        h_a = self.atom_emb(batch.x.long()) + self.atom_spse_proj(batch.atom_spse)
        if batch.bond_attr.shape[0] > 0:
            h_b = self.bond_emb(batch.bond_attr.long()) + self.bond_spse_proj(batch.bond_spse)
        else:
            h_b = h_a.new_zeros((0, self.bond_spse_proj.out_features))
        h_r = (
            self.ring_emb(batch.ring_size.long().clamp(0, MAX_RING_SIZE))
            if batch.ring_size.shape[0] > 0
            else h_a.new_zeros((0, self.ring_emb.embedding_dim))
        )
        h_v = self.virt_init.expand(bsz, -1).to(h_a.dtype)

        lg_3d = None
        if self.use_3d and getattr(batch, "pos", None) is not None and getattr(batch, "geom_valid", None) is not None:
            pos = batch.pos.float()
            gv = batch.geom_valid
            gv_node = gv[batch.batch].unsqueeze(-1).to(pos.dtype)
            if self.training and self.pos_noise_std > 0.0:
                pos = pos + torch.randn_like(pos) * (self.pos_noise_std * gv_node)
            if h_b.shape[0] > 0:
                u, v = batch.bond_to_atom[0], batch.bond_to_atom[1]
                d_b = torch.sqrt(((pos[u] - pos[v]) ** 2).sum(-1).clamp_min(1e-8))
                rbf_b = torch.exp(((d_b.unsqueeze(-1) - self.bond_rbf_c) ** 2) * self.bond_rbf_inv_2s2)
                gv_b = gv[batch.bond_batch].unsqueeze(-1).to(rbf_b.dtype)
                h_b = h_b + self.bond_3d_proj(rbf_b * gv_b).to(h_b.dtype)
            if batch.lg_edge_index.numel() > 0:
                u, v = batch.bond_to_atom[0], batch.bond_to_atom[1]
                src, dst = batch.lg_edge_index[0], batch.lg_edge_index[1]
                c = batch.lg_center_atom
                osrc = u[src] + v[src] - c
                odst = u[dst] + v[dst] - c
                v1 = pos[osrc] - pos[c]
                v2 = pos[odst] - pos[c]
                n1 = torch.sqrt((v1 * v1).sum(-1).clamp_min(1e-8))
                n2 = torch.sqrt((v2 * v2).sum(-1).clamp_min(1e-8))
                cos_ang = ((v1 * v2).sum(-1) / (n1 * n2)).clamp(-1.0, 1.0)
                d13 = torch.sqrt(((pos[osrc] - pos[odst]) ** 2).sum(-1).clamp_min(1e-8))
                ang_f = torch.exp(((cos_ang.unsqueeze(-1) - self.ang_rbf_c) ** 2) * self.ang_rbf_inv_2s2)
                d13_f = torch.exp(((d13.unsqueeze(-1) - self.nb13_rbf_c) ** 2) * self.nb13_rbf_inv_2s2)
                gv_lg = gv[batch.batch[c]].unsqueeze(-1).to(ang_f.dtype)
                lg_3d = torch.cat((ang_f, d13_f), dim=-1) * gv_lg
            # Non-bonded intramolecular radial spectrum within 5.0 A (via graph-same mask on atom pairs)
            nc = batch.ptr[1:] - batch.ptr[:-1]
            m_n = max(int(nc.max().item()), 1)
            local = torch.arange(pos.shape[0], device=pos.device) - batch.ptr[:-1].to(pos.device)[batch.batch]
            pos_d = pos.new_zeros((bsz, m_n, 3))
            mask_d = torch.zeros((bsz, m_n), dtype=torch.bool, device=pos.device)
            pos_d[batch.batch, local] = pos
            mask_d[batch.batch, local] = True
            dist_mat = torch.sqrt(((pos_d[:, :, None, :] - pos_d[:, None, :, :]) ** 2).sum(-1).clamp_min(1e-8))
            eye = torch.eye(m_n, dtype=torch.bool, device=pos.device).unsqueeze(0)
            p_valid = mask_d[:, :, None] & mask_d[:, None, :] & (~eye) & (dist_mat < 5.0) & gv[:, None, None]
            env_rbf = torch.exp(((dist_mat.unsqueeze(-1) - self.env_rbf_c) ** 2) * self.env_rbf_inv_2s2) * p_valid.unsqueeze(-1).to(pos.dtype)
            env_sum = env_rbf.sum(dim=2)[batch.batch, local]
            h_a = h_a + self.atom_3d_proj(env_sum).to(h_a.dtype)

        h_a = self.input_drop(h_a)
        h_b = self.input_drop(h_b)
        if h_r.shape[0] > 0:
            h_r = self.input_drop(h_r)

        n_a, n_b, n_r = h_a.shape[0], h_b.shape[0], h_r.shape[0]
        denoms = {
            "bsz": bsz,
            "ra_r": _inv_count(batch.ring_to_atom[1], n_r, h_a.device) if n_r > 0 else h_a.new_empty((0, 1)),
            "rb_r": _inv_count(batch.ring_to_bond[1], n_r, h_a.device) if n_r > 0 else h_a.new_empty((0, 1)),
            "ra_a": _inv_count(batch.ring_to_atom[0], n_a, h_a.device) if n_r > 0 else h_a.new_empty((n_a, 1)),
            "rb_b": _inv_count(batch.ring_to_bond[0], n_b, h_a.device) if (n_r > 0 and n_b > 0) else h_a.new_empty((n_b, 1)),
            "va": _inv_count(batch.batch, bsz, h_a.device),
            "vb": _inv_count(batch.bond_batch, bsz, h_a.device),
            "vr": _inv_count(batch.ring_batch, bsz, h_a.device),
        }

        for block in self.blocks:
            h_a, h_b, h_r, h_v = block(h_a, h_b, h_r, h_v, batch, denoms, lg_3d)

        ha = self.norm_a(h_a.float()).to(h_a.dtype)
        hb = self.norm_b(h_b.float()).to(h_b.dtype)
        hr = self.norm_r(h_r.float()).to(h_r.dtype)
        hv = self.norm_v(h_v.float()).to(h_v.dtype)

        feats = torch.cat((
            (_scatter_add(ha, batch.batch, bsz) * denoms["va"]).to(hv.dtype),
            (_scatter_add(ha, batch.batch, bsz) * 0.1).to(hv.dtype),
            (_scatter_add(hb, batch.bond_batch, bsz) * denoms["vb"]).to(hv.dtype),
            (_scatter_add(hr, batch.ring_batch, bsz) * denoms["vr"]).to(hv.dtype),
            hv,
        ), dim=-1)
        return self.readout(feats.float()).squeeze(-1)


LEVEL_SPEC = {
    "node": {"x": (9, torch.int16), "atom_spse": (6, torch.float32)},
    "bond": {"bond_attr": (3, torch.int16), "bond_spse": (6, torch.float32), "bond_to_atom": (2, torch.int16)},
    "lg": {"lg_edge_index": (2, torch.int16), "lg_center_atom": (1, torch.int16)},
    "ring": {"ring_size": (1, torch.int8)},
    "ra": {"ring_to_atom": (2, torch.int16)},
    "rb": {"ring_to_bond": (2, torch.int16)},
}


class GPUResidentCellStore:
    """Compact GPU-resident cellular complex store (0/1/2 cells + SPSE + optional 3D pos)."""

    def __init__(
        self,
        shard_paths: list[Path],
        *,
        device: str | torch.device = "cuda",
        expected_source_start: int | None = None,
        expected_source_stop: int | None = None,
        pos_tables: list[tuple[int, int, dict]] | None = None,
    ) -> None:
        self.device = torch.device(device)
        chunks = {lvl: {f: [] for f in fields} for lvl, fields in LEVEL_SPEC.items()}
        counts = {lvl: [] for lvl in LEVEL_SPEC}
        ys, sidxs, pos_chunks, gv_chunks = [], [], [], []
        has_pos = pos_tables is not None

        for spath in shard_paths:
            data, slices = torch.load(Path(spath), map_location="cpu", weights_only=False)
            n_g = int(slices["x"].shape[0] - 1)
            sidx = data.source_idx.view(-1).long()
            if int(sidx.max().item()) >= PROTECTED_ROW_START:
                raise ValueError(f"Protected split boundary violated: max source_idx={int(sidx.max())}")
            sidxs.append(sidx)
            ys.append(data.y.view(-1).float())
            x_ptr = slices["x"].long()
            e_ptr = slices["edge_index"].long()

            for i in range(n_g):
                xs, xe = int(x_ptr[i].item()), int(x_ptr[i + 1].item())
                es, ee = int(e_ptr[i].item()), int(e_ptr[i + 1].item())
                ei = data.edge_index[:, es:ee]
                ea = data.edge_attr[es:ee]
                parts = extract_cellular_parts(data.x[xs:xe], ei, ea)
                for lvl, fields in LEVEL_SPEC.items():
                    for f in fields:
                        chunks[lvl][f].append(parts[f])
                    counts[lvl].append(int(parts[next(iter(fields))].shape[0]))
                if has_pos:
                    sid = int(sidx[i].item())
                    n_r = xe - xs
                    found = False
                    for t_start, t_stop, tbl in pos_tables:
                        if t_start <= sid < t_stop:
                            row = sid - t_start
                            ps, pe = int(tbl["node_ptr"][row].item()), int(tbl["node_ptr"][row + 1].item())
                            p_t = tbl["pos"][ps:pe].float()
                            if p_t.shape[0] != n_r:
                                raise ValueError(f"pos mismatch at {sid}")
                            pos_chunks.append(p_t)
                            gv_chunks.append(bool(tbl["pos_valid"][row].item()))
                            found = True
                            break
                    if not found:
                        raise KeyError(f"source_idx={sid} outside pos_manifest")

        self.source_idx = torch.cat(sidxs, dim=0)
        if expected_source_start is not None and int(self.source_idx.min().item()) < expected_source_start:
            raise ValueError("Unexpected source_idx start")
        if expected_source_stop is not None and int(self.source_idx.max().item()) >= expected_source_stop:
            raise ValueError("Unexpected source_idx stop")
        self.n_graphs = int(self.source_idx.shape[0])

        self.tensors = {
            lvl: {f: torch.cat(chunks[lvl][f], dim=0).to(self.device) for f in fields}
            for lvl, fields in LEVEL_SPEC.items()
        }
        self.ptr = {}
        self.cnt = {}
        for lvl in LEVEL_SPEC:
            c_t = torch.tensor(counts[lvl], dtype=torch.long)
            p_t = torch.zeros(self.n_graphs + 1, dtype=torch.long)
            p_t[1:] = torch.cumsum(c_t, 0)
            self.cnt[lvl] = c_t.to(self.device)
            self.ptr[lvl] = p_t.to(self.device)

        self.y = torch.cat(ys, dim=0).to(self.device)
        self.has_pos = has_pos
        if has_pos:
            self.pos = torch.cat(pos_chunks, dim=0).to(self.device)
            self.geom_valid = torch.tensor(gv_chunks, dtype=torch.bool, device=self.device)
        else:
            self.pos = None
            self.geom_valid = None

    def __len__(self) -> int:
        return self.n_graphs

    @torch.no_grad()
    def get_batch(self, indices: torch.Tensor) -> SimpleNamespace:
        dev = self.device
        idx = indices.to(dev, dtype=torch.long)
        bsz = int(idx.numel())
        c, off, total, rows = {}, {}, {}, {}
        for lvl in LEVEL_SPEC:
            p = self.ptr[lvl]
            start = p[idx]
            cnt = p[idx + 1] - start
            c[lvl] = cnt
            off[lvl] = torch.cumsum(cnt, 0) - cnt
            tot = int(cnt.sum().item())
            total[lvl] = tot
            if tot > 0:
                rows[lvl] = torch.repeat_interleave(start - off[lvl], cnt, output_size=tot) + torch.arange(tot, device=dev)
            else:
                rows[lvl] = torch.empty((0,), dtype=torch.long, device=dev)

        def rep(src_lvl: str, over: str) -> torch.Tensor:
            if total[over] == 0:
                return torch.empty((0,), dtype=torch.long, device=dev)
            return torch.repeat_interleave(off[src_lvl], c[over], output_size=total[over])

        T = self.tensors
        graph = torch.arange(bsz, device=dev)
        ptr_node = torch.zeros((bsz + 1,), dtype=torch.long)
        ptr_node[1:] = torch.cumsum(c["node"].cpu(), 0)

        out = SimpleNamespace()
        out.num_graphs = bsz
        out.ptr = ptr_node
        out.y = self.y[idx]
        out.x = T["node"]["x"][rows["node"]].long()
        out.atom_spse = T["node"]["atom_spse"][rows["node"]]
        out.batch = torch.repeat_interleave(graph, c["node"], output_size=total["node"])
        out.bond_attr = T["bond"]["bond_attr"][rows["bond"]].long()
        out.bond_spse = T["bond"]["bond_spse"][rows["bond"]]
        out.bond_to_atom = (T["bond"]["bond_to_atom"][rows["bond"]].long() + rep("node", "bond")[:, None]).t().contiguous()
        out.bond_batch = torch.repeat_interleave(graph, c["bond"], output_size=total["bond"])
        out.lg_edge_index = (T["lg"]["lg_edge_index"][rows["lg"]].long() + rep("bond", "lg")[:, None]).t().contiguous()
        out.lg_center_atom = T["lg"]["lg_center_atom"][rows["lg"]].long().reshape(-1) + rep("node", "lg")
        out.ring_size = T["ring"]["ring_size"][rows["ring"]].long().reshape(-1)
        out.ring_batch = torch.repeat_interleave(graph, c["ring"], output_size=total["ring"])
        ra = T["ra"]["ring_to_atom"][rows["ra"]].long()
        out.ring_to_atom = (
            torch.stack((ra[:, 0] + rep("node", "ra"), ra[:, 1] + rep("ring", "ra")))
            if total["ra"] > 0
            else torch.empty((2, 0), dtype=torch.long, device=dev)
        )
        rb = T["rb"]["ring_to_bond"][rows["rb"]].long()
        out.ring_to_bond = (
            torch.stack((rb[:, 0] + rep("bond", "rb"), rb[:, 1] + rep("ring", "rb")))
            if total["rb"] > 0
            else torch.empty((2, 0), dtype=torch.long, device=dev)
        )
        if self.has_pos and self.pos is not None and self.geom_valid is not None:
            out.pos = self.pos[rows["node"]]
            out.geom_valid = self.geom_valid[idx]
        else:
            out.pos = None
            out.geom_valid = None
        return out
