"""Pure-2D, device-resident data layer for the 100K direction screen.

Hard constraints enforced here (not just documented):
  * 3D / geometry fields are never read into the store; their presence in the shard is recorded and asserted away.
  * train rows are exactly source_idx 0..N-1 (N <= 100000); development rows are exactly 100000..149999 (or a prefix
    when a smoke ``dev_limit`` is given); the two never overlap.
  * any source_idx >= 3,250,000 aborts immediately (tail_holdout_v1).
  * official valid / test roles are never opened: only manifest roles "train" and "development" are accepted.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

DROP_KEYS = ("pos", "edge_distance", "wedge_edge_ids", "wedge_angle_cos", "geometry_valid", "mmff_converged")
TAIL_HOLDOUT_START = 3_250_000
TRAIN_MAX_ROWS = 100_000
DEV_START, DEV_STOP = 100_000, 150_000
ALLOWED_ROLES = ("train", "development")
EXPECTED_MANIFEST_SHA = "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d"


def sha256_file(path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def atomic_write(path, data: bytes) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_json(path, obj) -> None:
    atomic_write(path, json.dumps(obj, indent=1, sort_keys=True).encode("utf-8"))


def atomic_npz(path, **arrays) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:  # file object: np.savez must not append ".npz"
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_torch_save(path, obj) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def tail_guard(source_idx) -> None:
    mx = int(source_idx.max()) if len(source_idx) else -1
    if mx >= TAIL_HOLDOUT_START:
        raise SystemExit(f"tail_holdout_v1: source_idx {mx} >= {TAIL_HOLDOUT_START} is protected; aborting")


def read_manifest(export_root, verify_shards: bool = True):
    """Return (manifest, manifest_sha, shard_records). Accepts the fixed-subset layout (geometry_shards) and the
    store layout (assets.topology)."""
    root = Path(export_root)
    mpath = root / "manifest.json"
    if not mpath.is_file():
        raise SystemExit(f"manifest.json not found under {root}")
    msha = sha256_file(mpath)
    if msha != EXPECTED_MANIFEST_SHA:
        raise SystemExit(f"manifest sha256 {msha} != expected {EXPECTED_MANIFEST_SHA}; refusing (row identity not proven)")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    if "geometry_shards" in manifest:
        recs = list(manifest["geometry_shards"])
    else:
        recs = list(manifest.get("assets", {}).get("topology", []))
    if not recs:
        raise SystemExit("manifest lists no shards")
    for key in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"):
        if manifest.get(key) not in (False, None):
            raise SystemExit(f"manifest flag {key} is not false")
    recs = [r for r in recs if r.get("role") in ALLOWED_ROLES]
    for r in recs:
        p = root / r["file"]
        if not p.is_file():
            raise SystemExit(f"missing shard {p}")
        if r.get("bytes") is not None and p.stat().st_size != int(r["bytes"]):
            raise SystemExit(f"shard size mismatch {p}")
        if r.get("source_idx_max") is not None and int(r["source_idx_max"]) >= TAIL_HOLDOUT_START:
            raise SystemExit("tail_holdout_v1: manifest shard exceeds protected boundary")
    if verify_shards:
        for r in recs:
            got = sha256_file(root / r["file"])
            if got != r["sha256"]:
                raise SystemExit(f"shard sha256 mismatch for {r['file']}: {got} != {r['sha256']}")
    return manifest, msha, recs


class Batch:
    __slots__ = ("x", "edge_index", "edge_attr", "batch", "random_walk_pe", "y", "num_graphs", "ids",
                 "ecfp", "desc")

    def to(self, *a, **k):
        return self


class GraphStore:
    """Device-resident compact 2D graph store with deterministic, sync-light batch assembly."""

    def __init__(self, device):
        self.device = torch.device(device)
        self.n = 0
        self.source_idx = None
        self.y = None
        self.ecfp = self.desc = self.aux = self.tags = None  # optional feature tensors aligned to rows
        self.provenance = {}

    def _finalize(self, xs, eis, eas, rws, ys, n_ptr, e_ptr):
        d = self.device
        self.all_x = torch.cat(xs).to(torch.uint8).to(d)
        self.all_ei = torch.cat(eis, dim=1).to(torch.int16).to(d)
        self.all_ea = torch.cat(eas).to(torch.uint8).to(d)
        self.all_rw = torch.cat(rws).float().to(d)
        self.y_cpu = torch.cat(ys).float()
        self.y = self.y_cpu.to(d)
        self.node_ptr_cpu = torch.cat(n_ptr)
        self.edge_ptr_cpu = torch.cat(e_ptr)
        self.node_ptr = self.node_ptr_cpu.to(d)
        self.edge_ptr = self.edge_ptr_cpu.to(d)
        self.n = int(self.y_cpu.numel())

    def attach_features(self, feats: dict):
        d = self.device
        if feats.get("ecfp_bits") is not None:
            self.ecfp = torch.from_numpy(feats["ecfp_bits"]).to(d)  # uint8 [n,2048]
        if feats.get("desc") is not None:
            self.desc = torch.from_numpy(feats["desc"]).float().to(d)
        if feats.get("aux16") is not None:
            self.aux = torch.from_numpy(feats["aux16"]).float().to(d)
        if feats.get("tags") is not None:
            self.tags = np.asarray(feats["tags"])  # cpu int16 [n,6]

    def batch(self, ids_cpu: torch.Tensor) -> Batch:
        d = self.device
        ids = ids_cpu.to(torch.int64)
        B = int(ids.numel())
        ids_g = ids.to(d, non_blocking=True)
        b = Batch()
        b.ids, b.num_graphs = ids_g, B
        b.y = self.y.index_select(0, ids_g)
        b.ecfp = self.ecfp.index_select(0, ids_g) if self.ecfp is not None else None
        b.desc = self.desc.index_select(0, ids_g) if self.desc is not None else None
        n0, n1 = self.node_ptr_cpu[ids], self.node_ptr_cpu[ids + 1]
        e0, e1 = self.edge_ptr_cpu[ids], self.edge_ptr_cpu[ids + 1]
        nc, ec = n1 - n0, e1 - e0
        tn, te = int(nc.sum()), int(ec.sum())
        ar = torch.arange(B, dtype=torch.int64, device=d)
        nc_g, ec_g = nc.to(d), ec.to(d)
        nptr = torch.zeros(B + 1, dtype=torch.int64, device=d)
        torch.cumsum(nc_g, 0, out=nptr[1:])
        eptr = torch.zeros(B + 1, dtype=torch.int64, device=d)
        torch.cumsum(ec_g, 0, out=eptr[1:])
        bvec = torch.repeat_interleave(ar, nc_g, output_size=tn)
        gnode = n0.to(d).index_select(0, bvec) + (torch.arange(tn, device=d) - nptr[:-1].index_select(0, bvec))
        egraph = torch.repeat_interleave(ar, ec_g, output_size=te)
        gedge = e0.to(d).index_select(0, egraph) + (torch.arange(te, device=d) - eptr[:-1].index_select(0, egraph))
        b.x = self.all_x.index_select(0, gnode).to(torch.int64)
        b.random_walk_pe = self.all_rw.index_select(0, gnode)
        b.edge_index = self.all_ei.index_select(1, gedge).to(torch.int64) + nptr[:-1].index_select(0, egraph).unsqueeze(0)
        b.edge_attr = self.all_ea.index_select(0, gedge).to(torch.int64)
        b.batch = bvec
        return b


def _load_shard(path):
    data, slices = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return data, slices


def load_role(export_root, role: str, rows: int | None = None, device="cpu", *, verify_shards=True,
              _manifest=None) -> GraphStore:
    """Load the first `rows` rows of a role. train: rows<=100000, source_idx must equal arange(rows).
    development: source_idx must equal 100000..(100000+rows-1); rows=None means all 50,000."""
    if role not in ALLOWED_ROLES:
        raise SystemExit(f"role {role!r} is not readable by this screen (official valid/test are forbidden)")
    manifest, msha, recs = _manifest or read_manifest(export_root, verify_shards)
    root = Path(export_root)
    want_start = 0 if role == "train" else DEV_START
    full = TRAIN_MAX_ROWS if role == "train" else DEV_STOP - DEV_START
    rows = full if rows is None else int(rows)
    if rows > full or rows < 1:
        raise SystemExit(f"{role}: requested {rows} rows, allowed 1..{full}")
    shards = [r for r in recs if r["role"] == role]
    xs, eis, eas, rws, ys, nps, eps, sidx = [], [], [], [], [], [], [], []
    need, n_off, e_off = rows, 0, 0
    dropped = set()
    for r in shards:
        if need <= 0:
            break
        data, sl = _load_shard(root / r["file"])
        keys = set(data.keys())
        dropped |= (keys & set(DROP_KEYS))
        g = int(sl["x"].numel() - 1)
        take = min(g, need)
        nn = int(sl["x"][take]); ee = int(sl["edge_index"][take])
        si = data.source_idx[:take].clone()
        tail_guard(si)
        sidx.append(si)
        x = data.x[:nn]
        if int(x.min()) < 0 or int(x.max()) > 255:
            raise SystemExit("atom feature outside uint8 range")
        xs.append(x.clone()); eis.append(data.edge_index[:, :ee].clone()); eas.append(data.edge_attr[:ee].clone())
        rws.append(data.random_walk_pe[:nn].clone()); ys.append(data.y[:take].reshape(-1).clone())
        nps.append(sl["x"][1:take + 1].clone() + n_off if nps else sl["x"][:take + 1].clone() + n_off)
        eps.append(sl["edge_index"][1:take + 1].clone() + e_off if eps else sl["edge_index"][:take + 1].clone() + e_off)
        n_off += nn; e_off += ee; need -= take
        del data, sl
    si_all = torch.cat(sidx)
    if si_all.numel() != rows:
        raise SystemExit(f"{role}: manifest shards hold only {si_all.numel()} rows, need {rows}")
    expect = torch.arange(want_start, want_start + rows, dtype=torch.int64)
    if not torch.equal(si_all, expect):
        raise SystemExit(f"{role}: source_idx is not exactly {want_start}..{want_start + rows - 1}")
    if role == "train" and int(si_all.max()) >= TRAIN_MAX_ROWS:
        raise SystemExit("train rows must have source_idx < 100000")
    st = GraphStore(device)
    st._finalize(xs, eis, eas, rws, ys, nps, eps)
    st.source_idx = si_all.numpy()
    assert not (dropped and any(hasattr(st, k) for k in DROP_KEYS)), "3D field leaked into the store"
    for k in DROP_KEYS:
        assert not hasattr(st, k)
    st.provenance = {"role": role, "rows": rows, "manifest_sha256": msha, "dropped_3d_keys_present_in_shard": sorted(dropped),
                     "source_idx_first": int(si_all[0]), "source_idx_last": int(si_all[-1]), "pure_2d_asserted": True}
    return st


def assert_disjoint(train: GraphStore, dev: GraphStore) -> None:
    if len(np.intersect1d(train.source_idx, dev.source_idx)):
        raise SystemExit("train and development rows overlap")
    if int(train.source_idx.max()) >= DEV_START or int(dev.source_idx.min()) < DEV_START:
        raise SystemExit("train/dev role boundary violated")


def row_identity(store: GraphStore) -> dict:
    return {"rows": int(store.n),
            "source_idx_sha256": sha256_bytes(np.ascontiguousarray(store.source_idx.astype("<i8")).tobytes()),
            "y_sha256": sha256_bytes(np.ascontiguousarray(store.y_cpu.numpy().astype("<f4")).tobytes())}


def load_features(path, store: GraphStore, train_ref: GraphStore | None = None, *, need=("ecfp", "desc", "aux16", "tags")):
    """Align features_v1.npz rows to `store` by source_idx. desc is standardised with the TRAIN rows' mean/std
    (train_ref, or the store itself when it is the train store). Returns dict ready for attach_features."""
    z = np.load(path, allow_pickle=False)
    fs = z["source_idx"]
    order = np.argsort(fs)
    pos = np.searchsorted(fs[order], store.source_idx)
    if (pos >= len(fs)).any() or not np.array_equal(fs[order][pos], store.source_idx):
        raise SystemExit("features file does not cover the requested source_idx rows")
    sel = order[pos]
    out = {"aux16": None, "ecfp_bits": None, "desc": None, "tags": None}
    if "tags" in need:
        out["tags"] = z["tags"][sel]
    if "aux16" in need:
        out["aux16"] = np.clip(z["aux16"][sel].astype(np.float32), -6, 6)
    if "ecfp" in need:
        out["ecfp_bits"] = np.unpackbits(z["ecfp_packed"][sel], axis=1)
    if "desc" in need:
        ref = train_ref if train_ref is not None else store
        rsel = order[np.searchsorted(fs[order], ref.source_idx)]
        dref = z["desc"][rsel].astype(np.float64)
        mu, sd = dref.mean(0), dref.std(0)
        sd[sd < 1e-8] = 1.0
        d = (z["desc"][sel].astype(np.float64) - mu) / sd
        out["desc"] = np.clip(np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0), -6, 6).astype(np.float32)
    return out


class EpochSampler:
    """Deterministic per-epoch batches: order depends only on (seed, epoch) so resume mid-epoch is exact.
    weights=None: permutation; weights given: n draws with replacement (same step count)."""

    def __init__(self, n, batch=128, seed=42, drop_last=True, weights=None):
        self.n, self.batch, self.seed, self.drop_last = int(n), int(batch), int(seed), drop_last
        self.weights = None if weights is None else torch.as_tensor(weights, dtype=torch.float64)

    def steps(self):
        return self.n // self.batch if self.drop_last else -(-self.n // self.batch)

    def order(self, epoch):
        g = torch.Generator()
        g.manual_seed(self.seed * 1_000_003 + int(epoch))
        if self.weights is None:
            return torch.randperm(self.n, generator=g)
        return torch.multinomial(self.weights, self.n, replacement=True, generator=g)

    def ids(self, order, step):
        return order[step * self.batch:(step + 1) * self.batch]
