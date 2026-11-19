"""DeMol-CIN: Pure-2D sparse cellular complex network with GNN+ SwiGLU and SPSE."""
from __future__ import annotations

from collections import deque
from dataclasses import replace

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

EXTRA_BATCH_FIELDS = (
    "atom_spse",
    "bond_spse",
    "bond_attr",
    "bond_to_atom",
    "bond_batch",
    "lg_edge_index",
    "lg_center_atom",
    "ring_size",
    "ring_to_atom",
    "ring_to_bond",
    "ring_batch",
)
MAX_RING_SIZE = 8
MAX_RINGS = 16


class CellData(Data):
    """PyG Data carrying 1-cell, line-graph, SPSE, and 2-cell incidence indices."""

    def __inc__(self, key, value, *args, **kwargs):
        n_nodes = int(self.x.shape[0])
        n_bonds = int(self.bond_attr.shape[0])
        n_rings = int(self.ring_size.shape[0])
        if key in {"bond_to_atom", "lg_center_atom"}:
            return n_nodes
        if key == "lg_edge_index":
            return n_bonds
        if key == "ring_to_atom":
            return torch.tensor([[n_nodes], [n_rings]], dtype=torch.long)
        if key == "ring_to_bond":
            return torch.tensor([[n_bonds], [n_rings]], dtype=torch.long)
        if key in {"bond_batch", "ring_batch"}:
            return 1
        if key in {"atom_spse", "bond_spse"}:
            return 0
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in {"bond_to_atom", "lg_edge_index", "ring_to_atom", "ring_to_bond"}:
            return 1
        if key in {"atom_spse", "bond_spse", "bond_attr", "bond_batch", "lg_center_atom", "ring_size", "ring_batch"}:
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


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
            for i in range(k) for j in range(i + 2, k) if not (i == 0 and j == k - 1)
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
    a_cols = []
    b_cols = []
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


def extract_cellular_complex(graph: Data) -> CellData:
    """Extract 0/1/2-cell incidence, SPSE 3..8 cycle/walk counts, and line-graph."""
    n_nodes = int(graph.x.shape[0])
    ei = graph.edge_index.tolist() if graph.edge_index.numel() else [[], []]
    ea = graph.edge_attr.tolist() if getattr(graph, "edge_attr", None) is not None and graph.edge_attr.numel() else []
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
        bond_attr = torch.tensor([undir[b] for b in bonds], dtype=torch.long)
        bond_to_atom = torch.tensor([[b[0] for b in bonds], [b[1] for b in bonds]], dtype=torch.long)
        bond_batch = torch.zeros(e_u, dtype=torch.long)
    else:
        bond_attr = torch.empty((0, 3), dtype=torch.long)
        bond_to_atom = torch.empty((2, 0), dtype=torch.long)
        bond_batch = torch.empty((0,), dtype=torch.long)

    inc_bonds: list[list[int]] = [[] for _ in range(n_nodes)]
    for b_id, (u, v) in enumerate(bonds):
        inc_bonds[u].append(b_id)
        inc_bonds[v].append(b_id)
    lg_src, lg_dst, lg_c = [], [], []
    for c, blist in enumerate(inc_bonds):
        for b1 in blist:
            for b2 in blist:
                if b1 != b2:
                    lg_src.append(b1)
                    lg_dst.append(b2)
                    lg_c.append(c)
    lg_edge_index = torch.tensor([lg_src, lg_dst], dtype=torch.long) if lg_src else torch.empty((2, 0), dtype=torch.long)
    lg_center_atom = torch.tensor(lg_c, dtype=torch.long) if lg_c else torch.empty((0,), dtype=torch.long)

    rings = _find_chordless_rings(n_nodes, bonds, bond_map)
    if rings:
        ring_size = torch.tensor([len(r) for r in rings], dtype=torch.long)
        r_a, r_ar, r_b, r_br = [], [], [], []
        for r_id, cyc in enumerate(rings):
            k = len(cyc)
            for i, a in enumerate(cyc):
                nxt = cyc[(i + 1) % k]
                r_a.append(a); r_ar.append(r_id)
                r_b.append(bond_map[(a, nxt) if a < nxt else (nxt, a)]); r_br.append(r_id)
        ring_to_atom = torch.tensor([r_a, r_ar], dtype=torch.long)
        ring_to_bond = torch.tensor([r_b, r_br], dtype=torch.long)
        ring_batch = torch.zeros(len(rings), dtype=torch.long)
    else:
        ring_size = torch.empty((0,), dtype=torch.long)
        ring_to_atom = torch.empty((2, 0), dtype=torch.long)
        ring_to_bond = torch.empty((2, 0), dtype=torch.long)
        ring_batch = torch.empty((0,), dtype=torch.long)

    out = CellData(**graph.to_dict())
    out.atom_spse, out.bond_spse = atom_spse, bond_spse
    out.bond_attr, out.bond_to_atom, out.bond_batch = bond_attr, bond_to_atom, bond_batch
    out.lg_edge_index, out.lg_center_atom = lg_edge_index, lg_center_atom
    out.ring_size, out.ring_to_atom, out.ring_to_bond, out.ring_batch = ring_size, ring_to_atom, ring_to_bond, ring_batch
    return out


class _CellDataset(Dataset):
    def __init__(self, base) -> None:
        self.base = base
        self.transform = None
        self._items = [extract_cellular_complex(base[i]) for i in range(len(base))]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> CellData:
        return self._items[idx]


def attach_inputs(input_root, sidecar, data):
    return replace(
        data,
        train=_CellDataset(data.train),
        development=_CellDataset(data.development) if data.development is not None else None,
    )


def prepare_batch(batch, *, training: bool = False):
    if hasattr(batch, "bond_to_atom"):
        return batch
    tmp = Batch.from_data_list([extract_cellular_complex(g) for g in batch.to_data_list()]).to(batch.x.device)
    for field in EXTRA_BATCH_FIELDS:
        setattr(batch, field, getattr(tmp, field))
    return batch


def _scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = src.new_zeros((dim_size, src.shape[1]))
    if src.numel() > 0:
        out.index_add_(0, index, src)
    return out


def _scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = _scatter_add(src, index, dim_size)
    if src.numel() > 0:
        cnt = src.new_zeros((dim_size, 1))
        cnt.index_add_(0, index, src.new_ones((src.shape[0], 1)))
        out = out / cnt.clamp_min(1.0)
    return out


def _mlp(in_d: int, hid_d: int, out_d: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_d, hid_d), nn.SiLU(), nn.Linear(hid_d, out_d))


class _SwiGLU(nn.Module):
    def __init__(self, d: int, hid: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.w12 = nn.Linear(d, 2 * hid)
        self.w3 = nn.Linear(hid, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = self.w12(self.norm(x)).chunk(2, dim=-1)
        return self.w3(F.silu(u) * v)


class CellBlock(nn.Module):
    """GNN+ Pre-LN SwiGLU cellular block: 0<->1, directed NB 1<->1, 0/1<->2, virtual hub."""

    def __init__(self, d_a: int = 144, d_b: int = 112, d_r: int = 112, d_v: int = 144) -> None:
        super().__init__()
        self.na, self.nb, self.nr, self.nv = nn.LayerNorm(d_a), nn.LayerNorm(d_b), nn.LayerNorm(d_r), nn.LayerNorm(d_v)
        self.b2a = _mlp(d_b + 2 * d_a, d_a, d_a)
        self.a2b = _mlp(d_b + 2 * d_a, d_b, d_b)
        self.lg_msg = nn.Linear(d_b + d_a, d_b)
        self.lg_gate = nn.Sequential(nn.Linear(2 * d_b + 2 * d_a, d_b), nn.Sigmoid())
        self.lg_out = nn.Linear(d_b, d_b)
        self.r_in = _mlp(d_r + d_a + d_b, d_r, d_r)
        self.r2a, self.r2b = nn.Linear(d_r, d_a), nn.Linear(d_r, d_b)
        self.v_upd = _mlp(d_v + d_a + d_b + d_r, d_v, d_v)
        self.v2a, self.v2b, self.v2r = nn.Linear(d_v, d_a), nn.Linear(d_v, d_b), nn.Linear(d_v, d_r)
        self.ffn_a = _SwiGLU(d_a, int(1.5 * d_a))
        self.ffn_b = _SwiGLU(d_b, int(1.5 * d_b))
        self.ffn_r = _SwiGLU(d_r, int(1.5 * d_r))

    def forward(self, h_a, h_b, h_r, h_v, batch, bsz: int):
        na, nb, nr, nv = self.na(h_a), self.nb(h_b), self.nr(h_r), self.nv(h_v)
        n_a, n_b, n_r = na.shape[0], nb.shape[0], nr.shape[0]
        if n_b > 0:
            u, v = batch.bond_to_atom[0], batch.bond_to_atom[1]
            pair_a = torch.cat((nb, na[u] + na[v], (na[u] - na[v]).abs()), dim=-1)
            mb = self.b2a(pair_a)
            h_a = h_a + _scatter_add(mb, u, n_a) + _scatter_add(mb, v, n_a)
            h_b = h_b + self.a2b(pair_a)
        if batch.lg_edge_index.numel() > 0:
            src, dst = batch.lg_edge_index[0], batch.lg_edge_index[1]
            c = batch.lg_center_atom
            other_src = u[src] + v[src] - c
            other_dst = u[dst] + v[dst] - c
            dir_diff = na[other_dst] - na[other_src]
            gate = self.lg_gate(torch.cat((nb[src], nb[dst], na[c], dir_diff), dim=-1))
            msg = self.lg_msg(torch.cat((nb[src], na[other_src] - na[c]), dim=-1))
            h_b = h_b + self.lg_out(_scatter_add(gate * msg, dst, n_b))
        if n_r > 0:
            ra_a, ra_r = batch.ring_to_atom[0], batch.ring_to_atom[1]
            rb_b, rb_r = batch.ring_to_bond[0], batch.ring_to_bond[1]
            h_r = h_r + self.r_in(torch.cat((nr, _scatter_mean(na[ra_a], ra_r, n_r), _scatter_mean(nb[rb_b], rb_r, n_r)), dim=-1))
            nr2 = self.nr(h_r)
            h_a = h_a + _scatter_mean(self.r2a(nr2[ra_r]), ra_a, n_a)
            h_b = h_b + _scatter_mean(self.r2b(nr2[rb_r]), rb_b, n_b)
        va, vb, vr = _scatter_mean(na, batch.batch, bsz), _scatter_mean(nb, batch.bond_batch, bsz), _scatter_mean(nr, batch.ring_batch, bsz)
        h_v = h_v + self.v_upd(torch.cat((nv, va, vb, vr), dim=-1))
        nv2 = self.nv(h_v)
        h_a = h_a + self.v2a(nv2)[batch.batch]
        if n_b > 0:
            h_b = h_b + self.v2b(nv2)[batch.bond_batch]
        if n_r > 0:
            h_r = h_r + self.v2r(nv2)[batch.ring_batch]
        return h_a + self.ffn_a(h_a), h_b + self.ffn_b(h_b), h_r + self.ffn_r(h_r), h_v


class DeMolCIN(nn.Module):
    """Standalone Pure-2D cellular complex network with GNN+ SwiGLU & SPSE."""

    accepts_prepared_batch = True

    def __init__(self, d_a: int = 144, d_b: int = 112, d_r: int = 112, d_v: int = 144, num_layers: int = 8) -> None:
        super().__init__()
        self.atom_emb = AtomEncoder(d_a)
        self.bond_emb = BondEncoder(d_b)
        self.atom_spse_proj = nn.Linear(6, d_a)
        self.bond_spse_proj = nn.Linear(6, d_b)
        self.ring_emb = nn.Embedding(MAX_RING_SIZE + 1, d_r)
        self.virt_init = nn.Parameter(torch.zeros(1, d_v))
        nn.init.normal_(self.virt_init, std=0.02)
        self.blocks = nn.ModuleList([CellBlock(d_a, d_b, d_r, d_v) for _ in range(num_layers)])
        self.norm_a, self.norm_b, self.norm_r, self.norm_v = nn.LayerNorm(d_a), nn.LayerNorm(d_b), nn.LayerNorm(d_r), nn.LayerNorm(d_v)
        self.readout = nn.Sequential(
            nn.LayerNorm(2 * d_a + d_b + d_r + d_v),
            nn.Linear(2 * d_a + d_b + d_r + d_v, d_a),
            nn.SiLU(),
            nn.Linear(d_a, 1),
        )

    def forward(self, batch) -> torch.Tensor:
        batch = prepare_batch(batch)
        bsz = int(batch.num_graphs)
        h_a = self.atom_emb(batch.x.long()) + self.atom_spse_proj(batch.atom_spse)
        if batch.bond_attr.shape[0] > 0:
            h_b = self.bond_emb(batch.bond_attr.long()) + self.bond_spse_proj(batch.bond_spse)
        else:
            h_b = h_a.new_zeros((0, self.bond_spse_proj.out_features))
        h_r = self.ring_emb(batch.ring_size.long().clamp(0, MAX_RING_SIZE)) if batch.ring_size.shape[0] > 0 else h_a.new_zeros((0, self.ring_emb.embedding_dim))
        h_v = self.virt_init.expand(bsz, -1)
        for block in self.blocks:
            h_a, h_b, h_r, h_v = block(h_a, h_b, h_r, h_v, batch, bsz)
        ha, hb, hr, hv = self.norm_a(h_a), self.norm_b(h_b), self.norm_r(h_r), self.norm_v(h_v)
        return self.readout(torch.cat((
            _scatter_mean(ha, batch.batch, bsz),
            _scatter_add(ha, batch.batch, bsz) * 0.1,
            _scatter_mean(hb, batch.bond_batch, bsz),
            _scatter_mean(hr, batch.ring_batch, bsz),
            hv,
        ), dim=-1)).squeeze(-1)


def construct_family_model(recipe, state):
    return DeMolCIN()


def policy(mode):
    from .family_training_policy import FamilyModePolicy

    if mode != "demol_cin":
        raise ValueError("Unknown DeMol-CIN mode")
    return FamilyModePolicy(
        construct=construct_family_model,
        prepare_batch=prepare_batch,
        attach_inputs=attach_inputs,
        objective_name="degree_bond_local_ema999",
        standalone_model=True,
        extra_batch_fields=EXTRA_BATCH_FIELDS,
    )
