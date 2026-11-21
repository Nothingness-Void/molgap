"""Build an aligned DFT heavy-atom coordinate sidecar for official PCQM4Mv2 train rows.

Streams the official training SDF archive (never extracts it), matches each SDF
molecule to its fixed-store topology graph by element and bond set, and writes
per-shard coordinate files in graph atom order plus a manifest (format v1).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import multiprocessing as mp
import os
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

SHARD_ROWS = 50_000
MAX_ROWS = 3_250_000
CHUNK = 2000
MEMBER = "pcqm4m-v2-train.sdf"
BOND_RANGE = (0.9, 2.0)
SHARD_FORMAT = "molgap-pcqm-dft-geometry-shard-v1"
SIDECAR_FORMAT = "molgap-pcqm-dft-geometry-sidecar-v1"
IDENTITY, ISOMORPHIC, BAD_COUNT, BAD_ELEMENTS, BAD_BONDS, BAD_LABELS = range(6)
CLASS_NAMES = ["identity_order", "isomorphism_mapped", "atom_count_mismatch",
               "element_mismatch", "bond_set_mismatch", "label_mismatch"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def parse_molblock(lines):
    """Return heavy-atom Z, H counts, charges, coordinates and bond (a, b, order) rows from a V2000 block."""
    from rdkit import Chem
    table = Chem.GetPeriodicTable()
    n_atoms, n_bonds = int(lines[3][0:3]), int(lines[3][3:6])
    syms, pos = [], []
    for ln in lines[4:4 + n_atoms]:
        parts = ln.split()
        syms.append(parts[3])
        pos.append((float(parts[0]), float(parts[1]), float(parts[2])))
    keep = [i for i, s in enumerate(syms) if s != "H"]
    new = {old: k for k, old in enumerate(keep)}
    z = np.array([table.GetAtomicNumber(syms[i]) for i in keep], dtype=np.int16)
    nh, charge = np.zeros(len(keep), np.int16), np.zeros(len(keep), np.int16)
    bonds = []
    for ln in lines[4 + n_atoms:4 + n_atoms + n_bonds]:
        a, b, order = int(ln[0:3]) - 1, int(ln[3:6]) - 1, int(ln[6:9])
        if a in new and b in new:
            bonds.append((min(new[a], new[b]), max(new[a], new[b]), order))
        else:  # heavy-H bond
            for h, x in ((a, b), (b, a)):
                if x in new and syms[h] == "H":
                    nh[new[x]] += 1
    for ln in lines[4 + n_atoms + n_bonds:]:
        if ln.startswith("M  CHG"):
            f = ln.split()[3:]
            for i in range(0, len(f) - 1, 2):
                if int(f[i]) - 1 in new:
                    charge[new[int(f[i]) - 1]] = int(f[i + 1])
    xyz = np.array([pos[i] for i in keep], dtype=np.float32).reshape(-1, 3)
    return z, nh, charge, xyz, sorted(bonds)


def labelled_skeleton(z, nh, charge, bonds, labelled=True):
    """Single-bond graph; atom isotope encodes (Z, H count, charge) so matches must agree."""
    from rdkit import Chem
    m = Chem.RWMol()
    for zi, h, q in zip(z, nh, charge):
        atom = Chem.Atom(int(zi))
        if labelled:
            atom.SetIsotope(int(zi) * 1000 + int(h) * 20 + int(q) + 10)
        m.AddAtom(atom)
    for a, b, *_ in bonds:
        m.AddBond(int(a), int(b), Chem.BondType.SINGLE)
    return m


def orders_agree(order_by_pair, g_bonds, match):
    """Graph SINGLE/DOUBLE/TRIPLE (0/1/2) must equal SDF order 1/2/3; graph AROMATIC (3) takes any Kekule order."""
    return all(t == 3 or order_by_pair[tuple(sorted((match[a], match[b])))] == t + 1 for a, b, t in g_bonds)


def length_flags(pos, gz, g_bonds):
    """Failed bond-length checks (fixed range, covalent range, bond order vs length) and graph C=C lengths."""
    from rdkit import Chem
    table = Chem.GetPeriodicTable()
    e = np.array(g_bonds)
    d = np.linalg.norm(pos[e[:, 0]] - pos[e[:, 1]], axis=1)
    za, zb = gz[e[:, 0]], gz[e[:, 1]]
    rsum = np.array([table.GetRcovalent(int(i)) + table.GetRcovalent(int(j)) for i, j in zip(za, zb)])
    co, cc = ((za == 6) & (zb == 8)) | ((za == 8) & (zb == 6)), (za == 6) & (zb == 6)
    flags = {
        "bonded_distance_fixed_range_0.9_2.0": ((d < BOND_RANGE[0]) | (d > BOND_RANGE[1])).any(),
        "bonded_distance_out_of_range": ((d < 0.75 * rsum) | (d > 1.25 * rsum)).any(),
        "bol_double_CO_long": (co & (e[:, 2] == 1) & (d > 1.30)).any(),
        "bol_double_CC_long": (cc & (e[:, 2] == 1) & (d > 1.40)).any(),
        "bol_single_CO_short": (co & (e[:, 2] == 0) & (d < 1.28)).any(),
    }
    names = [k for k, v in flags.items() if v]
    gate = any(n in ("bol_double_CO_long", "bol_single_CO_short") for n in names)  # C=C long is report-only
    return names + (["bond_order_length_inconsistent"] if gate else []), d[cc & (e[:, 2] == 1)]


def align_row(task):
    """Return (code, positions in graph order or None, failed length checks) for one row."""
    from rdkit import Chem
    text, gz, gh, gq, gb = task
    s_z, s_h, s_q, s_xyz, s_bonds = parse_molblock(text.split("\n"))
    n = len(gz)
    if len(s_z) != n:
        return BAD_COUNT, None, ([], [])
    if sorted(s_z.tolist()) != sorted(gz.tolist()):
        return BAD_ELEMENTS, None, ([], [])
    # OpenBabel writes charge-separated groups (nitro, N-oxide) uncharged with extra double bonds, so SDF charges are
    # ignored and bonds touching a graph-charged atom accept any order.
    wild = gq != 0
    g_bonds = sorted((a, b, 3 if wild[a] or wild[b] else t) for a, b, t in gb.tolist())
    gq = s_q = np.zeros(n, np.int16)
    order_by_pair = {(a, b): o for a, b, o in s_bonds}
    if len(s_bonds) != len(g_bonds):
        return BAD_BONDS, None, ([], [])
    ident = (np.array_equal(s_z, gz) and np.array_equal(s_h, gh) and np.array_equal(s_q, gq)
             and [b[:2] for b in s_bonds] == [b[:2] for b in g_bonds] and orders_agree(order_by_pair, g_bonds, range(n)))
    if ident:
        code, pos = IDENTITY, s_xyz
    else:
        target = labelled_skeleton(s_z, s_h, s_q, s_bonds)
        params = Chem.SubstructMatchParameters()
        params.maxMatches = 1
        params.setExtraFinalCheck(lambda mol, match: orders_agree(order_by_pair, g_bonds, match))
        match = target.GetSubstructMatch(labelled_skeleton(gz, gh, gq, g_bonds), params)  # graph j -> sdf p(j)
        if len(match) != n:  # no chemistry-consistent mapping: is the bare skeleton at least isomorphic?
            bare = target.GetSubstructMatch(labelled_skeleton(gz, gh, gq, g_bonds, False))
            return (BAD_LABELS if len(bare) == n else BAD_BONDS), None, ([], [])
        pos, code = s_xyz[np.array(match)], ISOMORPHIC
    return code, pos.astype(np.float32), length_flags(pos, gz, g_bonds) if g_bonds else ([], [])


def align_chunk(tasks):
    return [align_row(t) for t in tasks]


def iter_records(archive, start, stop):
    """Yield (row, molblock text) for rows [start, stop), stopping right after row stop-1."""
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if member.name != MEMBER:
                continue
            buf, row = [], 0
            for line in tar.extractfile(member):
                if line.startswith(b"$$$$"):
                    if row >= start:
                        yield row, b"".join(buf).decode("ascii", "replace")
                    buf, row = [], row + 1
                    if row >= stop:
                        return
                else:
                    buf.append(line.replace(b"\r\n", b"\n"))
            return
    raise SystemExit(f"member {MEMBER} not found")


def load_shard(roots, k):
    import torch
    rel = Path("store/topology/train") / f"train_shard_{k:04d}.pt"
    for root in roots:
        path = Path(root) / rel
        if path.is_file():
            data, slices = torch.load(path, weights_only=False)
            return path, rel.as_posix(), data, slices
    raise SystemExit(f"topology shard {k:04d} not found in any root")


def row_tasks(data, slices, r0, r1, texts):
    """Per-row payloads: atomic numbers and undirected bond pairs (local indices)."""
    xs, es = slices["x"], slices["edge_index"]
    tasks = []
    for r, text in zip(range(r0 % SHARD_ROWS, r1 - r0 + r0 % SHARD_ROWS), texts):
        z = (data.x[xs[r]:xs[r + 1], 0].numpy() + 1).astype(np.int16)
        ei = data.edge_index[:, es[r]:es[r + 1]].numpy()
        if ei.size and ei.max() >= len(z):
            ei = ei - int(xs[r])  # global indices: shift to local
        keep = ei[0] < ei[1]
        bt = data.edge_attr[es[r]:es[r + 1], 0].numpy()
        gb = np.concatenate([ei.T[keep], bt[keep, None]], axis=1)  # (a, b, OGB bond type)
        tasks.append((text, z, data.x[xs[r]:xs[r + 1], 4].numpy().astype(np.int16),
                      data.x[xs[r]:xs[r + 1], 3].numpy().astype(np.int16) - 5, gb))
    return tasks


def write_atomic(path, obj, torch_save=False):
    tmp = Path(str(path) + ".tmp")
    if torch_save:
        import torch
        torch.save(obj, tmp)
    else:
        tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--expected-archive-sha256", required=True)
    ap.add_argument("--topology-root", action="append", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--provenance-note", default="")
    a = ap.parse_args(argv)
    if a.start % SHARD_ROWS or a.stop % SHARD_ROWS or not 0 <= a.start < a.stop <= MAX_ROWS:
        raise SystemExit(f"start/stop must be multiples of {SHARD_ROWS} within [0, {MAX_ROWS}]")
    out = Path(a.output_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    digest = sha256_file(a.archive)
    if digest != a.expected_archive_sha256:
        raise SystemExit(f"archive sha256 {digest} != expected")
    t_hash = time.time() - t0

    import torch
    t1 = time.time()
    counts = collections.Counter()
    examples = collections.defaultdict(list)
    topo, shards, cc_lengths = [], [], []
    records = iter_records(a.archive, a.start, a.stop)
    pending = collections.deque()
    pool = mp.get_context("spawn").Pool(min(a.workers, 8))

    def note(name, idx):
        counts[name] += 1
        if len(examples[name]) < 20:
            examples[name].append(idx)

    try:
        for k in range(a.start // SHARD_ROWS, a.stop // SHARD_ROWS):
            path, rel, data, slices = load_shard(a.topology_root, k)
            r0 = k * SHARD_ROWS
            if int(data.source_idx[0]) != r0 or len(data.source_idx) != SHARD_ROWS:
                raise SystemExit(f"shard {k:04d} does not hold rows {r0}..")
            topo.append({"file": rel, "sha256": sha256_file(path), "bytes": path.stat().st_size})
            res = []
            for c0 in range(r0, r0 + SHARD_ROWS, CHUNK):
                texts = []
                for row, text in records:
                    if row != c0 + len(texts):
                        raise SystemExit("SDF row order drift")
                    texts.append(text)
                    if len(texts) == CHUNK:
                        break
                if len(texts) != CHUNK:
                    raise SystemExit("SDF ended early")
                pending.append(pool.apply_async(align_chunk, (row_tasks(data, slices, c0, c0 + CHUNK, texts),)))
                while len(pending) > 2 * a.workers:
                    res += pending.popleft().get()
            while pending:
                res += pending.popleft().get()
            offsets = slices["x"].numpy().astype(np.int64)
            pos = np.zeros((int(offsets[-1]), 3), np.float32)
            valid = np.zeros(SHARD_ROWS, bool)
            for i, (code, p, (flags, cc_len)) in enumerate(res):
                cc_lengths.extend(cc_len)
                note(CLASS_NAMES[code], r0 + i)
                if p is not None:
                    pos[offsets[i]:offsets[i + 1]], valid[i] = p, True
                    for f in flags:
                        note(f, r0 + i)
            name = f"shard_{k:04d}.pt"
            write_atomic(out / name, {"format": SHARD_FORMAT, "source_idx": data.source_idx.long(),
                                      "atom_offsets": torch.from_numpy(offsets),
                                      "pos": torch.from_numpy(pos), "valid": torch.from_numpy(valid)}, True)
            shards.append({"file": name, "source_idx_start": r0, "source_idx_stop": r0 + SHARD_ROWS,
                           "rows": SHARD_ROWS, "sha256": sha256_file(out / name),
                           "bytes": (out / name).stat().st_size})
            print(f"shard {k:04d} done {time.time() - t1:.0f}s", flush=True)
    finally:
        pool.terminate()
    rows = a.stop - a.start
    invalid = sum(counts[n] for n in CLASS_NAMES[2:])
    oor = counts["bonded_distance_out_of_range"]
    bol = counts["bond_order_length_inconsistent"]
    ok = invalid / rows <= 0.01 and oor / rows <= 0.001 and bol / rows <= 0.005
    manifest = {
        "format": SIDECAR_FORMAT, "status": "complete" if ok else "rejected",
        "source": {"archive_name": Path(a.archive).name, "archive_sha256": digest, "member": MEMBER,
                   "provenance_note": a.provenance_note},
        "rows": {"source_idx_start": a.start, "source_idx_stop": a.stop, "rows": rows},
        "topology": topo, "shards": shards,
        "alignment": {"identity_order": counts["identity_order"], "isomorphism_mapped": counts["isomorphism_mapped"],
                      "invalid": invalid, "invalid_by_class": {n: counts[n] for n in CLASS_NAMES[2:]},
                      "bonded_distance_out_of_range_rows": oor,
                      "bond_order_length_inconsistent_rows": bol,
                      "bond_order_length_by_class": {k: v for k, v in counts.items() if k.startswith("bol_")},
                      "bond_order_length_note": "gating: C=O > 1.30 A, C-O < 1.28 A. report-only: matcher enforces non-aromatic "
                                                "bond order; long C=C (> 1.40 A) are conjugated ring bonds",
                      "graph_double_CC_length_A": {"count": len(cc_lengths), "median": float(np.median(cc_lengths)),
                                                  "p99": float(np.percentile(cc_lengths, 99))},
                      "bonded_distance_fixed_range_rows": counts["bonded_distance_fixed_range_0.9_2.0"],
                      "bonded_distance_rule": "out of range = any bond outside 0.75-1.25 x covalent-radius sum",
                      "examples": {n: examples[n] for n in examples if n != "identity_order"}},
        "timing_s": {"archive_hash": round(t_hash, 1), "parse_align_write": round(time.time() - t1, 1)},
        "mapping": "chem-labelled-v2",
        "builder": "platforms/local/pcqm_dft_geometry_sidecar.py",
    }
    write_atomic(out / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("status", "rows", "alignment", "timing_s")}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
