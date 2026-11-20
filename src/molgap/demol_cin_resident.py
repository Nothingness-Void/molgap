"""Opt-in compact, GPU-resident DeMol-CIN inputs (CSR-style), replacing the per-molecule CellData list.

Every molecule goes once through the unchanged ``extract_cellular_complex``; its 0/1/2-cell tensors are packed
into a few large tensors in small integer dtypes with per-molecule pointers (local indices, no batch offsets).
A batch is assembled on the device from molecule indices and equals ``Batch.from_data_list`` of the CellData
objects element for element (same dtypes, shapes, offsets). Fields DeMol never reads (random_walk_pe, split_code,
smiles_sanitized) are not carried. Switch: ``MOLGAP_DEMOL_GPU_RESIDENT=1`` (see ``demol_cin.attach_inputs``).
"""
from __future__ import annotations

import torch
from torch_geometric.data import Batch
from torch.utils.data import Dataset

from .demol_cin import extract_cellular_complex

I16, I8, F32 = torch.int16, torch.int8, torch.float32
# level -> field -> (columns, stored dtype, original dtype); rows of a level are counted per molecule.
LEVELS = {
    "node": {"x": (9, I16, torch.long), "atom_spse": (6, F32, F32)},
    "dedge": {"edge_index": (2, I16, torch.long), "edge_attr": (3, I16, torch.long)},
    "bond": {"bond_attr": (3, I16, torch.long), "bond_spse": (6, F32, F32), "bond_to_atom": (2, I16, torch.long)},
    "lg": {"lg_edge_index": (2, I16, torch.long), "lg_center_atom": (1, I16, torch.long)},
    "ring": {"ring_size": (1, I8, torch.long)},
    "ra": {"ring_to_atom": (2, I16, torch.long)},
    "rb": {"ring_to_bond": (2, I16, torch.long)},
}
BATCH_KEYS = ("x", "edge_index", "edge_attr", "y", "source_idx", "atom_spse", "bond_spse", "bond_attr", "bond_to_atom",
              "bond_batch", "lg_edge_index", "lg_center_atom", "ring_size", "ring_to_atom", "ring_to_bond",
              "ring_batch", "batch", "ptr")


def _parts(c):
    return {
        "node": {"x": c.x, "atom_spse": c.atom_spse},
        "dedge": {"edge_index": c.edge_index.t(), "edge_attr": c.edge_attr},
        "bond": {"bond_attr": c.bond_attr, "bond_spse": c.bond_spse, "bond_to_atom": c.bond_to_atom.t()},
        "lg": {"lg_edge_index": c.lg_edge_index.t(), "lg_center_atom": c.lg_center_atom[:, None]},
        "ring": {"ring_size": c.ring_size[:, None]},
        "ra": {"ring_to_atom": c.ring_to_atom.t()},
        "rb": {"ring_to_bond": c.ring_to_bond.t()},
    }


def _pack(value, dtype, original, name):
    if value.dtype != original or (value.ndim != 2):
        raise ValueError("Unexpected dtype/shape for " + name)
    packed = value.to(dtype)
    if dtype != F32 and not torch.equal(packed.to(original), value):
        raise ValueError("Value does not fit the compact dtype: " + name)
    return packed


class CompactCells:
    """Host (or device) tensors for N molecules; ``ptr[level]`` has N+1 entries."""

    def __init__(self, tensors, ptr, y, source_idx):
        self.tensors, self.ptr, self.y, self.source_idx = tensors, ptr, y, source_idx
        self.device = torch.device("cpu")

    @property
    def rows(self):
        return int(self.y.shape[0])

    @property
    def nbytes(self):
        t = sum(v.numel() * v.element_size() for d in self.tensors.values() for v in d.values())
        return t + self.y.numel() * 4

    @classmethod
    def build(cls, base, chunk=2048):
        """Stream ``base`` in chunks; host memory holds one chunk of CellData plus the compact result."""
        n = len(base)
        shards = getattr(base, "datasets", (base,))
        chunks = {lvl: {f: [] for f in fields} for lvl, fields in LEVELS.items()}
        counts = {lvl: [] for lvl in LEVELS}
        ys, srcs = [], []
        for start in range(0, n, chunk):
            parts = {lvl: {f: [] for f in fields} for lvl, fields in LEVELS.items()}
            for i in range(start, min(start + chunk, n)):
                c = extract_cellular_complex(base[i])
                if c.y.dtype != F32 or c.y.numel() != 1 or c.source_idx.dtype != torch.long or c.source_idx.numel() != 1:
                    raise ValueError("Unexpected y/source_idx layout")
                ys.append(c.y.reshape(1))
                srcs.append(c.source_idx.reshape(1))
                for lvl, fields in _parts(c).items():
                    for f, v in fields.items():
                        parts[lvl][f].append(v)
                    counts[lvl].append(next(iter(fields.values())).shape[0])
            for lvl, fields in LEVELS.items():
                for f, (cols, dtype, original) in fields.items():
                    joined = torch.cat(parts[lvl][f], 0)
                    if joined.shape[1] != cols:
                        raise ValueError("Unexpected width for " + f)
                    chunks[lvl][f].append(_pack(joined, dtype, original, f))
            for shard in shards:  # PyG InMemoryDataset keeps every fetched Data in _data_list
                if getattr(shard, "_data_list", None) is not None:
                    shard._data_list = None
        tensors = {lvl: {f: torch.cat(chunks[lvl].pop(f), 0) for f in list(chunks[lvl])} for lvl in LEVELS}
        ptr = {}
        for lvl in LEVELS:
            ptr[lvl] = torch.zeros(n + 1, dtype=torch.long)
            ptr[lvl][1:] = torch.cumsum(torch.tensor(counts[lvl], dtype=torch.long), 0)
        return cls(tensors, ptr, torch.cat(ys), torch.cat(srcs))

    def upload(self, device="cuda", max_fraction=0.8):
        """Move the payload to the device; fail loudly when it does not fit. Pointers stay on the host."""
        device = torch.device(device)
        if self.device == device:
            return self
        if device.type == "cuda" and self.nbytes > max_fraction * torch.cuda.mem_get_info(device)[0]:
            raise ValueError("GPU-resident DeMol inputs need %d bytes; device memory is insufficient" % self.nbytes)
        self.tensors = {l: {f: v.to(device) for f, v in d.items()} for l, d in self.tensors.items()}
        self.y_dev = self.y.to(device)
        self.device = device
        return self

    def release(self):
        self.tensors = {}
        self.y_dev = None

    def batch(self, index):
        """Assemble the batch for molecule indices ``index`` (a list, in order) on ``self.device``."""
        dev = self.device
        idx = torch.as_tensor(index, dtype=torch.long)
        size = int(idx.numel())
        meta, count, total = [], {}, {}
        for lvl in LEVELS:
            p = self.ptr[lvl]
            start = p[idx]
            c = p[idx + 1] - start
            off = torch.cumsum(c, 0) - c
            count[lvl], total[lvl] = c, int(c.sum())
            meta += [start - off, c, off]
        meta.append(idx)
        m = torch.stack(meta).to(dev)
        at = {lvl: (m[3 * k], m[3 * k + 1], m[3 * k + 2]) for k, lvl in enumerate(LEVELS)}
        didx = m[-1]
        arange = torch.arange

        def rows(lvl):
            base, c, _ = at[lvl]
            return torch.repeat_interleave(base, c, output_size=total[lvl]) + arange(total[lvl], device=dev)

        def rep(src_lvl, over):  # per-molecule start offset of src_lvl in the batch, repeated over a level's rows
            return torch.repeat_interleave(at[src_lvl][2], at[over][1], output_size=total[over])

        T = self.tensors
        r = {lvl: rows(lvl) for lvl in LEVELS}
        out = Batch()
        out["x"] = T["node"]["x"][r["node"]].long()
        out["atom_spse"] = T["node"]["atom_spse"][r["node"]]
        out["edge_index"] = (T["dedge"]["edge_index"][r["dedge"]].long() + rep("node", "dedge")[:, None]).t().contiguous()
        out["edge_attr"] = T["dedge"]["edge_attr"][r["dedge"]].long()
        out["bond_attr"] = T["bond"]["bond_attr"][r["bond"]].long()
        out["bond_spse"] = T["bond"]["bond_spse"][r["bond"]]
        out["bond_to_atom"] = (T["bond"]["bond_to_atom"][r["bond"]].long() + rep("node", "bond")[:, None]).t().contiguous()
        out["lg_edge_index"] = (T["lg"]["lg_edge_index"][r["lg"]].long() + rep("bond", "lg")[:, None]).t().contiguous()
        out["lg_center_atom"] = T["lg"]["lg_center_atom"][r["lg"]].long().reshape(-1) + rep("node", "lg")
        out["ring_size"] = T["ring"]["ring_size"][r["ring"]].long().reshape(-1)
        ra = T["ra"]["ring_to_atom"][r["ra"]].long()
        out["ring_to_atom"] = torch.stack((ra[:, 0] + rep("node", "ra"), ra[:, 1] + rep("ring", "ra")))
        rb = T["rb"]["ring_to_bond"][r["rb"]].long()
        out["ring_to_bond"] = torch.stack((rb[:, 0] + rep("bond", "rb"), rb[:, 1] + rep("ring", "rb")))
        graph = arange(size, device=dev)
        out["bond_batch"] = torch.repeat_interleave(graph, at["bond"][1], output_size=total["bond"])
        out["ring_batch"] = torch.repeat_interleave(graph, at["ring"][1], output_size=total["ring"])
        out["batch"] = membership = torch.repeat_interleave(graph, at["node"][1], output_size=total["node"])
        out["y"] = self.y_dev[didx]
        out["source_idx"] = self.source_idx[idx]
        ncount = count["node"]
        ptr = torch.zeros(size + 1, dtype=torch.long)
        ptr[1:] = torch.cumsum(ncount, 0)
        out["ptr"] = ptr
        out._num_graphs = size
        if size < 1 or bool((ncount <= 0).any()):
            raise ValueError("Graph membership must be contiguous and ordered")
        maximum = int(ncount.max())
        local = arange(total["node"], device=dev) - ptr.to(dev)[membership]
        ei = out["edge_index"]
        graph_of_edge = membership[ei[0]]
        out.execution_layout = {
            "batch_size": size, "max_nodes": maximum,
            "mask": arange(maximum, device=dev)[None, :] < at["node"][1][:, None],
            "flat_index": membership * maximum + local,
            "edges": (graph_of_edge, local[ei[0]], local[ei[1]]),
        }
        return out


class _TrainStream:
    def __init__(self, store, sampler):
        self.store, self.sampler = store, sampler

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        for index in self.sampler:
            yield self.store.batch(index)


class _SequentialStream:
    """Ordered, no drop_last: same batches as DataLoader(shuffle=False)."""
    def __init__(self, store, batch_size):
        self.store, self.batch_size = store, batch_size
        self.stats = dict(cold_passes=0, warm_passes=0, cached_tensor_storage_bytes=0)
        self.disabled_reason = "device-resident"

    def __len__(self):
        return (self.store.rows + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for start in range(0, self.store.rows, self.batch_size):
            yield self.store.batch(list(range(start, min(start + self.batch_size, self.store.rows))))

    def clear(self):
        self.store.release()


class ResidentCellDataset(Dataset):
    """Drop-in for ``_CellDataset``: compact store plus on-demand CellData for legacy consumers."""
    def __init__(self, base, chunk=2048) -> None:
        self.base = base
        self.transform = None
        self.store = CompactCells.build(base, chunk)

    def __len__(self) -> int:
        return self.store.rows

    def __getitem__(self, idx: int):
        return extract_cellular_complex(self.base[idx])

    def train_stream(self, sampler):
        return _TrainStream(self.store, sampler)

    def sequential_stream(self, batch_size):
        return _SequentialStream(self.store, batch_size)
