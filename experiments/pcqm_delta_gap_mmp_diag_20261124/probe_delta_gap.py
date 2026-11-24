"""Zero-training single-point substitution DeltaGap diagnostic (NO_TRAIN).

Four-stage blind separation:
  Stage 1: Pure structure parsing (rdMMPA 1-cut) & disjoint real pairing (zero y, zero yhat).
  Stage 2: Deterministic EMA@60 inference with dev replay checks (zero score y).
  Stage 3: Covariate-balanced non-MMP rewiring & fail-closed Balance Gate (zero yhat).
  Stage 4: Single-pass whole-block recovery scoring & series-clustered bootstrap (iff gate passes).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import copy
import csv
import gzip
import hashlib
import io
import json
import math
from multiprocessing import get_context
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter, process_time
from zipfile import ZipFile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import networkx as nx
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, rdMMPA
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import ks_2samp

SHARD_ROWS = 50_000
TAIL_START = 3_250_000
DEV_SHARD = 10
SCORING_SHARDS = (58, 59, 60, 61)
LOCAL_SMOKE_SHARDS = (63, 64)
FIT_IDX_ELIGIBLE_LO = (600_000, 1_850_000)
FIT_IDX_ELIGIBLE_HI = (1_900_000, 3_150_000)
FIT_IDX_COUNT = 120_000
FIT_IDX_SEED = 20261123

OFFICIAL_ARCHIVE_SHA256 = "628ae612a3de752160929d93d1584a75257ae156e7b13d91b133d8db62e6d701"
FULL_TRAIN_MANIFEST_SHA256 = "7d358a679d299bd5de6c5822e42dcef34b0e0b9fa9598c684709509c0b94021d"
SCORING_SHARD_SHA256 = {
    58: "621e11151b10f775f1cd266df18b91525d75412bca505f3f663abaf1c3a25acb",
    59: "ba5cb07de08ff6d47371b7715b6e7d3d8f6fc668876d1b44921178006f374146",
    60: "32551f291e2d10dc482e8f82d3955eeee014ca41e3f5a0e94ff454f4417616ed",
    61: "35272cfdb0e338cd9956fccb1ba72b0bdb17b3d1c64cb88f164e71270d36da12",
}
FIXED_500K_MANIFEST_SHA256 = "630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751"
FIXED_500K_DEV_SHARD_SHA256 = "1a37dc5d458396b590044d978526822e6d1cb35df223de913a837dc7277d64c1"
FULL_TRAIN_DEV_SHARD_SHA256 = "3879dc27106034c27c893bac7d182d54fbbe93906efb83108e9dac541e0894ae"

MODEL_PINS = {
    "s43": {
        "arm_dir": "init_s43",
        "init_file": "init_s43.pt",
        "init_sha256": "2ccd1eb79de4b82fffb58522b9e2325026d36f04f398be0d49d3eea55ee05f3c",
        "manifest_sha256": "9f5131f307df7ee373e066c36afa96f17c84767c8712ead844b8073ce77457ac",
        "contract_sha256": "471ad2fb6785a1af128fbb5cccbb9a81c56c0b1c90064b692d82a098285bbc28",
        "selected_model_sha256": "13369b0ac07b71177f8d4a7a4c29f98c4ad9b892ac65fbb6e93b97a1a24a06ba",
        "predictions_sha256": "818b0b041919f53e31667c201ad926d56bf7d8e59c352056c912a2d63e6f6a62",
        "resume_sha256": "af709cc655cbcbd33ede294f09576dca78caad1385e91d61520850ceb4b9c52a",
        "trace_sha256": "722f43effa1ffe856ba2ce72b81e5be06f7231703a4c92247011f9954d1ff4c5",
    },
    "s44": {
        "arm_dir": "init_s44",
        "init_file": "init_s44.pt",
        "init_sha256": "2415cdbfea332db158086b2ee457ce95bcb90a2e3f87c04564146ba12a2c3573",
        "manifest_sha256": "2a439ae42a10179897ebf51d11a8a6db03d45029647c15781ec84af86e7a0f0a",
        "contract_sha256": "594da69200aac0818af0fb2f4e9fb7b3938b5b5e48a235cb485f826373e4e7b1",
        "selected_model_sha256": "a680e311a04548e9440d4b68866501d6dd8c3b49e96c86315699621bd0457ceb",
        "predictions_sha256": "fc5a9f5e11c303a8c5c1d1603f75b8ba853c23755db8e48538c52593e193a29e",
        "resume_sha256": "170b62f8123c4d872cc3fb50189e92e996db27743261982854420ca2faffe511",
        "trace_sha256": "14792cc98fea96eae094f7d968680196e0ce336f603ea94d2482ed117a6228d1",
    },
}

SEED = 20261124
N_BOOTSTRAP = 2000
BATCH_SIZE = 128
TOP_K_GLOBAL = 64
TOP_K_SCAF = 32
REWIRING_LAMBDAS = (0.0, 0.04, 0.08, 0.12, 0.18, 0.25)
N_OPT_STEPS = 600_000
INVESTMENT_GATE_MEV = 3.0

_WORKER_MFG = None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_pinned(path: Path, expected_sha256: str, label: str = "") -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required pinned file ({label}): {path}")
    obs = sha256_file(path)
    if obs != expected_sha256:
        raise ValueError(f"SHA256 mismatch for {label or path}: expected {expected_sha256}, got {obs}")
    return path


def atomic_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}-", delete=False) as f:
            tmp = Path(f.name)
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    return path


def save_npz_once(path: Path, **arrays) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable artifact: {path}")
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)
    return sha256_file(path)


def compute_spread_fit_idx() -> np.ndarray:
    eligible = np.r_[np.arange(*FIT_IDX_ELIGIBLE_LO), np.arange(*FIT_IDX_ELIGIBLE_HI)]
    return np.sort(np.random.default_rng(FIT_IDX_SEED).choice(eligible, FIT_IDX_COUNT, replace=False))


def build_target_indices(shards: tuple[int, ...]) -> np.ndarray:
    fit_set = set(compute_spread_fit_idx().tolist())
    rows = []
    for s in shards:
        assert 12 <= s <= 64, f"Disallowed shard {s}"
        lo, hi = s * SHARD_ROWS, (s + 1) * SHARD_ROWS
        for idx in range(lo, hi):
            if idx not in fit_set:
                rows.append(idx)
    arr = np.asarray(rows, dtype=np.int64)
    assert np.all(np.diff(arr) > 0) and arr[-1] < TAIL_START
    return arr


def _read_csv_field(archive_path: Path, rows: np.ndarray, field: int) -> list[str]:
    rows = np.asarray(rows, dtype=np.int64)
    assert np.all(np.diff(rows) > 0) and rows[0] >= 0 and rows[-1] < TAIL_START
    wanted = iter(rows.tolist())
    target = next(wanted, None)
    values = []
    with ZipFile(verify_pinned(archive_path, OFFICIAL_ARCHIVE_SHA256, "pcqm4m-v2.zip")) as z:
        with z.open("pcqm4m-v2/raw/data.csv.gz") as raw_f:
            with gzip.open(raw_f, "rb") as gz, io.TextIOWrapper(gz, encoding="utf-8") as lines:
                header = next(lines).strip()
                assert header == "idx,smiles,homolumogap", f"Unexpected CSV header: {header}"
                for i, line in enumerate(lines):
                    if target is None or i >= int(rows[-1]) + 1:
                        break
                    if i != target:
                        continue
                    rec = next(csv.reader([line]))
                    assert int(rec[0]) == i and len(rec) == 3
                    values.append(rec[field])
                    target = next(wanted, None)
    assert len(values) == len(rows), f"Expected {len(rows)} rows, got {len(values)}"
    return values


def _worker_init():
    global _WORKER_MFG
    RDLogger.DisableLog("rdApp.*")
    _WORKER_MFG = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _parse_smiles_chunk(smiles_chunk: list[str]):
    out = []
    mfg = _WORKER_MFG
    for smi in smiles_chunk:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            out.append(("", "", 0, 0, np.empty(0, dtype=np.uint16), [], []))
            continue
        noiso = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        ha = mol.GetNumHeavyAtoms()
        scaf, scaf_nrings = "", 0
        try:
            smol = MurckoScaffold.GetScaffoldForMol(mol)
            if smol.GetNumAtoms() > 0:
                scaf = Chem.MolToSmiles(smol, isomericSmiles=False)
                scaf_nrings = int(smol.GetRingInfo().NumRings())
        except Exception:
            scaf, scaf_nrings = "", 0
        onbits = np.flatnonzero(mfg.GetFingerprintAsNumPy(mol)).astype(np.uint16)
        all_cores, acyc_cores = set(), set()
        frags = rdMMPA.FragmentMol(mol, maxCuts=1, resultsAsMols=True, maxCutBonds=30)
        for lhs, rhs in frags:
            if lhs is not None or rhs is None:
                continue
            parts = Chem.GetMolFrags(rhs, asMols=True, sanitizeFrags=False)
            if len(parts) != 2:
                continue
            fa, fb = parts
            ha_a, ha_b = fa.GetNumHeavyAtoms(), fb.GetNumHeavyAtoms()
            for cm, cha, rm, rha in ((fa, ha_a, fb, ha_b), (fb, ha_b, fa, ha_a)):
                if cha >= 6 and 1 <= rha <= 12 and cha >= rha and cha * 2 >= ha:
                    cs = Chem.MolToSmiles(cm, canonical=True)
                    all_cores.add(cs)
                    Chem.FastFindRings(rm)
                    if rm.GetRingInfo().NumRings() == 0:
                        acyc_cores.add(cs)
        out.append((noiso, scaf, scaf_nrings, ha, onbits, sorted(all_cores), sorted(acyc_cores)))
    return out


def stage1_structures_and_real_pairs(archive_path: Path, target_indices: np.ndarray, out_dir: Path,
                                     *, n_workers: int, chunk_size: int = 512) -> dict:
    """Parse SMILES, build forbidden MMP edges + disjoint real pairs, seal Stage 1 (zero y, zero yhat)."""
    t0, cpu0 = perf_counter(), process_time()
    n = len(target_indices)
    smiles_list = _read_csv_field(archive_path, target_indices, 1)

    noiso = [""] * n
    scafs = [""] * n
    scaf_nrings = np.zeros(n, dtype=np.int32)
    has = np.zeros(n, dtype=np.int32)
    fp_arr = np.zeros((n, 2048), dtype=np.uint8)
    c2m_all = defaultdict(list)
    c2m_acyc = defaultdict(list)

    chunks = [smiles_list[i:i + chunk_size] for i in range(0, n, chunk_size)]
    mp_ctx = get_context("spawn")
    cursor = 0
    parse_fail = 0
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx, initializer=_worker_init) as pool:
        for res_chunk in pool.map(_parse_smiles_chunk, chunks):
            for ni, sc, nr, ha, onbits, ac, ay in res_chunk:
                if not ni:
                    parse_fail += 1
                noiso[cursor] = ni
                scafs[cursor] = sc
                scaf_nrings[cursor] = nr
                has[cursor] = ha
                if len(onbits):
                    fp_arr[cursor, onbits] = 1
                for cs in ac:
                    c2m_all[cs].append(cursor)
                for cs in ay:
                    c2m_acyc[cs].append(cursor)
                cursor += 1
    assert cursor == n

    # Build ALL 1-cut candidate MMP edges (forbidden in rewiring & source for Mode-A real graph)
    all_mmp_adj = [set() for _ in range(n)]
    real_adj = [set() for _ in range(n)]
    mega_members = [[] for _ in range(n)]
    mega_buckets_skipped = 0
    mega_buckets_all_count = 0
    for cs, entries in c2m_all.items():
        if len(entries) < 2:
            continue
        by_ni = {}
        for idx in entries:
            ni = noiso[idx]
            if ni and ni not in by_ni:
                by_ni[ni] = idx
        if len(by_ni) >= 2:
            ul_ni = sorted(by_ni.values())
            if len(ul_ni) > 300:
                mega_buckets_skipped += 1
            else:
                for a_i in range(len(ul_ni)):
                    u = ul_ni[a_i]
                    for b_i in range(a_i + 1, len(ul_ni)):
                        v = ul_ni[b_i]
                        real_adj[u].add(v)
                        real_adj[v].add(u)
        ul_all = sorted(set(entries))
        if len(ul_all) <= 300:
            for a_i in range(len(ul_all)):
                u = ul_all[a_i]
                for b_i in range(a_i + 1, len(ul_all)):
                    v = ul_all[b_i]
                    all_mmp_adj[u].add(v)
                    all_mmp_adj[v].add(u)
        else:
            mid = mega_buckets_all_count
            mega_buckets_all_count += 1
            for u in ul_all:
                mega_members[u].append(mid)

    # Also build same-Murcko acyclic-R MMP connected components for hierarchical series clustering
    g_acyc = nx.Graph()
    for cs, entries in c2m_acyc.items():
        if len(entries) < 2:
            continue
        ul = sorted(set(entries))
        if len(ul) > 300:
            continue
        for a_i in range(len(ul)):
            u = ul[a_i]
            if not scafs[u]:
                continue
            for b_i in range(a_i + 1, len(ul)):
                v = ul[b_i]
                if scafs[u] == scafs[v]:
                    g_acyc.add_edge(u, v)
    cc_id_acyc = {}
    for cid, comp in enumerate(nx.connected_components(g_acyc)):
        for u in comp:
            cc_id_acyc[u] = cid

    # Hierarchical chemical-series cluster IDs & raw Murcko cluster IDs across all N rows
    series_map, murcko_map, scaf_map = {}, {}, {}
    series_clusters = np.empty(n, dtype=np.int32)
    murcko_clusters = np.empty(n, dtype=np.int32)
    scaf_ids = np.full(n, -1, dtype=np.int32)
    noiso_ids = np.full(n, -1, dtype=np.int32)
    noiso_map = {}

    for i in range(n):
        sc = scafs[i]
        if sc:
            if sc not in scaf_map:
                scaf_map[sc] = len(scaf_map)
            scaf_ids[i] = scaf_map[sc]
        ni = noiso[i]
        if ni:
            if ni not in noiso_map:
                noiso_map[ni] = len(noiso_map)
            noiso_ids[i] = noiso_map[ni]
        sk = ("M", sc) if (scaf_nrings[i] >= 2 and sc) else (("C", cc_id_acyc[i]) if i in cc_id_acyc else ("S", i))
        if sk not in series_map:
            series_map[sk] = len(series_map)
        series_clusters[i] = series_map[sk]
        mk = ("M", sc) if sc else ("S", i)
        if mk not in murcko_map:
            murcko_map[mk] = len(murcko_map)
        murcko_clusters[i] = murcko_map[mk]

    # Deterministic structure-only disjoint matching
    matched = np.zeros(n, dtype=bool)
    real_pairs = []
    deg = np.array([len(real_adj[i]) for i in range(n)], dtype=np.int32)
    cand_verts = np.flatnonzero(deg > 0)
    order = sorted(cand_verts.tolist(), key=lambda x: (int(deg[x]), x))
    for u in order:
        if matched[u]:
            continue
        nbrs = [v for v in real_adj[u] if not matched[v]]
        if not nbrs:
            continue
        v = min(nbrs, key=lambda x: (int(deg[x]), x))
        matched[u] = True
        matched[v] = True
        real_pairs.append((int(u), int(v)))

    real_pairs_arr = np.asarray(real_pairs, dtype=np.int32)
    max_series_cnt = max(Counter(series_clusters.tolist()).values())
    max_murcko_cnt = max(Counter(murcko_clusters.tolist()).values())

    # Pack forbidden adjacency CSR for paired vertices so Stage 3 needs zero SMILES
    # Actually save packed CSR over all paired vertices' forbidden edges cleanly
    forb_csr_indptr = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        forb_csr_indptr[i + 1] = forb_csr_indptr[i] + len(all_mmp_adj[i])
    forb_csr_indices = np.empty(int(forb_csr_indptr[-1]), dtype=np.int32)
    pos = 0
    for i in range(n):
        if all_mmp_adj[i]:
            vals = sorted(all_mmp_adj[i])
            k = len(vals)
            forb_csr_indices[pos:pos + k] = vals
            pos += k
    assert pos == len(forb_csr_indices)

    mega_csr_indptr = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        mega_csr_indptr[i + 1] = mega_csr_indptr[i] + len(mega_members[i])
    mega_csr_indices = np.empty(int(mega_csr_indptr[-1]), dtype=np.int32)
    pos_m = 0
    for i in range(n):
        if mega_members[i]:
            vals_m = mega_members[i]
            k_m = len(vals_m)
            mega_csr_indices[pos_m:pos_m + k_m] = vals_m
            pos_m += k_m
    assert pos_m == len(mega_csr_indices)

    npz_path = out_dir / "stage1_real_pairs.npz"
    npz_sha = save_npz_once(
        npz_path,
        target_indices=target_indices,
        real_pairs=real_pairs_arr,
        scaf_ids=scaf_ids,
        noiso_ids=noiso_ids,
        has=has,
        fp_arr=fp_arr,
        series_clusters=series_clusters,
        murcko_clusters=murcko_clusters,
        forb_csr_indptr=forb_csr_indptr,
        forb_csr_indices=forb_csr_indices,
        mega_csr_indptr=mega_csr_indptr,
        mega_csr_indices=mega_csr_indices,
        mega_bucket_count=np.int32(mega_buckets_all_count),
    )

    summary = {
        "stage": "stage1_structures_and_real_pairs",
        "rows_total": int(n),
        "parse_fail": int(parse_fail),
        "mega_buckets_skipped": int(mega_buckets_skipped),
        "mega_buckets_all_count": int(mega_buckets_all_count),
        "real_pairs_count": int(len(real_pairs_arr)),
        "molecule_coverage": float(2 * len(real_pairs_arr) / n),
        "candidate_mmp_vertices": int(len(cand_verts)),
        "all_mmp_directed_edges": int(forb_csr_indptr[-1]),
        "series_cluster_count": int(len(series_map)),
        "max_series_cluster_share": float(max_series_cnt / n),
        "murcko_cluster_count": int(len(murcko_map)),
        "max_murcko_cluster_share": float(max_murcko_cnt / n),
        "stage1_npz_sha256": npz_sha,
        "labels_used": False,
        "predictions_used": False,
        "wall_seconds": perf_counter() - t0,
        "parent_cpu_seconds": process_time() - cpu0,
    }
    atomic_json(out_dir / "stage1_receipt.json", summary)
    return summary


def _setup_determinism():
    import torch
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _strip_labels_inplace(graphs):
    """Strip .y from graph views inside Stage 2 inference so inference sees topology only."""
    for g in graphs:
        if hasattr(g, "y"):
            delattr(g, "y")
    return graphs


def _prepare_batches(graphs, policy):
    from torch_geometric.loader import DataLoader
    from molgap.gptrans_execution_backend import prepare_layout
    from molgap.gptrans_execution_hotpath import MODEL_FIELDS

    batches = []
    for batch in DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=False, num_workers=0):
        b = prepare_layout(copy(batch))
        if policy.prepare_batch is not None:
            b = policy.prepare_batch(b, training=False)
        extra = tuple(f for f in policy.extra_batch_fields if hasattr(b, f))
        batches.append((b, MODEL_FIELDS + extra))
    return batches


def _run_det(model, batches, mean, std, device):
    import torch
    from molgap.gptrans_execution_backend import prepared_forward

    model.eval()
    out = []
    with torch.no_grad():
        for b, fields in batches:
            out.append(prepared_forward(model, b.to(device, *fields)).double().cpu().numpy().ravel())
    return np.concatenate(out) * std + mean


def stage2_predict_ema60(
    target_indices: np.ndarray,
    scoring_shard_paths: dict[int, Path],
    dev_shard_path: Path,
    arm_paths: dict[str, dict[str, Path]],
    out_dir: Path,
    *,
    dev_shard_expected_sha: str = FIXED_500K_DEV_SHARD_SHA256,
) -> dict:
    """Run deterministic EMA@60 inference on untouched shards (score labels unread)."""
    import torch
    from molgap.family_training_kernel import load_family_model, mode_policy
    from molgap.family_training_runner import read_recipe
    from molgap.packed_graph_dataset import PackedGraphDataset

    t0, cpu0 = perf_counter(), process_time()
    _setup_determinism()
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "Stage 2 requires CUDA for deterministic GPTrans-T parity"

    # Load verified dev shard (500K development role)
    verify_pinned(dev_shard_path, dev_shard_expected_sha, "dev train_shard_0010.pt")
    dev_ds = PackedGraphDataset(dev_shard_path, mmap=True).topology_only()
    assert len(dev_ds) == SHARD_ROWS
    dev_graphs = [dev_ds[i] for i in range(SHARD_ROWS)]
    dev_idx = np.array([int(g.source_idx) for g in dev_graphs], dtype=np.int64)
    assert np.array_equal(dev_idx, np.arange(DEV_SHARD * SHARD_ROWS, (DEV_SHARD + 1) * SHARD_ROWS))
    dev_y = np.array([float(g.y.item()) for g in dev_graphs], dtype=np.float64)
    _strip_labels_inplace(dev_graphs)

    # Load untouched target topology graphs WITHOUT storing/inspecting target labels
    target_graphs = []
    loaded_idx = []
    for s in SCORING_SHARDS:
        sp = verify_pinned(scoring_shard_paths[s], SCORING_SHARD_SHA256[s], f"train_shard_{s:04d}.pt")
        lo, hi = s * SHARD_ROWS, (s + 1) * SHARD_ROWS
        sel_rows = target_indices[(target_indices >= lo) & (target_indices < hi)]
        ds = PackedGraphDataset(sp, mmap=True).topology_only()
        assert len(ds) == SHARD_ROWS
        for idx in sel_rows.tolist():
            g = ds[int(idx - lo)]
            assert int(g.source_idx) == idx
            if hasattr(g, "y"):
                delattr(g, "y")
            target_graphs.append(g)
            loaded_idx.append(idx)
        del ds
    assert np.array_equal(np.asarray(loaded_idx, dtype=np.int64), target_indices)

    _, sample_recipe = read_recipe(arm_paths["s43"]["contract"])
    policy = mode_policy(sample_recipe.family, sample_recipe.mode)
    dev_batches = _prepare_batches(dev_graphs, policy)
    target_batches = _prepare_batches(target_graphs, policy)
    del dev_graphs, target_graphs

    preds = {}
    replay_max_abs = {}
    dev_ema60_mae = {}
    trace_pass60_mae = {}
    dev_ema60_diff = {}

    for arm in ("s43", "s44"):
        pins = MODEL_PINS[arm]
        files = arm_paths[arm]
        verify_pinned(files["init"], pins["init_sha256"], f"{arm}.init")
        verify_pinned(files["manifest"], pins["manifest_sha256"], f"{arm}.manifest")
        verify_pinned(files["contract"], pins["contract_sha256"], f"{arm}.contract")
        verify_pinned(files["selected_model"], pins["selected_model_sha256"], f"{arm}.selected_model")
        verify_pinned(files["predictions"], pins["predictions_sha256"], f"{arm}.predictions")
        verify_pinned(files["resume"], pins["resume_sha256"], f"{arm}.resume")
        verify_pinned(files["trace"], pins["trace_sha256"], f"{arm}.trace")

        tr = json.loads(files["trace"].read_text(encoding="utf-8"))
        p60 = [pt for pt in tr["observations"] if pt.get("epoch_or_pass") == 60][-1]
        assert p60["checkpoint_identity"] == pins["resume_sha256"]
        trace_pass60_mae[arm] = float(p60["ema_dev_metric"])

        _, recipe = read_recipe(files["contract"])
        mean, std = float(recipe.target_mean_eV), float(recipe.target_std_eV)
        model, _ = load_family_model(recipe, files["init"])
        model = model.to(device).eval()

        # 1. Verify selected_model.pt replays first 128 saved dev predictions <= 1e-4 eV
        sel_state = torch.load(files["selected_model"], map_location="cpu", weights_only=True)["model"]
        model.load_state_dict(sel_state, strict=True)
        dev_saved = torch.load(files["predictions"], map_location="cpu", weights_only=True)
        ix = dev_saved["source_idx"].ravel().argsort()
        assert np.array_equal(dev_saved["source_idx"].ravel()[ix].numpy(), dev_idx)
        first128 = _run_det(model, dev_batches[:1], mean, std, device)
        max_diff_128 = float(np.max(np.abs(first128 - dev_saved["prediction_eV"].ravel()[ix[:BATCH_SIZE]].numpy())))
        assert max_diff_128 <= 1e-4, f"{arm} dev 128 replay mismatch: {max_diff_128}"
        replay_max_abs[arm] = max_diff_128

        # 2. Load pass-60 EMA weights and verify full-dev MAE against canonical_trace pass 60 <= 1e-4 eV
        ckpt = torch.load(files["resume"], map_location="cpu", weights_only=True)
        assert ckpt["optimizer_step"] == 234360 and ckpt["cursor"]["pass_index"] == 60
        model.load_state_dict(ckpt["ema"], strict=True)
        dev_pred_ema = _run_det(model, dev_batches, mean, std, device)
        mae_dev = float(np.abs(dev_pred_ema - dev_y).mean())
        dev_diff = abs(mae_dev - trace_pass60_mae[arm])
        assert dev_diff <= 1e-4, f"{arm} dev EMA@60 MAE mismatch: {mae_dev} vs {trace_pass60_mae[arm]}"
        dev_ema60_mae[arm] = mae_dev
        dev_ema60_diff[arm] = dev_diff

        # 3. Deterministic EMA@60 predictions on untouched scoring block
        p_target = _run_det(model, target_batches, mean, std, device)
        assert len(p_target) == len(target_indices) and np.isfinite(p_target).all()
        preds[arm] = p_target

        del model, ckpt, sel_state
        torch.cuda.empty_cache()

    npz_path = out_dir / "stage2_predictions.npz"
    npz_sha = save_npz_once(
        npz_path,
        target_indices=target_indices,
        pred_s43_ema60=preds["s43"],
        pred_s44_ema60=preds["s44"],
    )
    summary = {
        "stage": "stage2_predict_ema60",
        "device": torch.cuda.get_device_name(0),
        "rows_total": int(len(target_indices)),
        "replay_max_abs_eV": replay_max_abs,
        "trace_pass60_ema_dev_mae_eV": trace_pass60_mae,
        "dev_ema60_mae_eV": dev_ema60_mae,
        "dev_ema60_abs_diff_eV": dev_ema60_diff,
        "stage2_npz_sha256": npz_sha,
        "score_labels_used": False,
        "wall_seconds": perf_counter() - t0,
        "parent_cpu_seconds": process_time() - cpu0,
    }
    atomic_json(out_dir / "stage2_receipt.json", summary)
    return summary


def _smd(a: np.ndarray, b: np.ndarray) -> float:
    sd = math.sqrt(0.5 * (float(np.var(a)) + float(np.var(b))))
    return (float(np.mean(a)) - float(np.mean(b))) / max(sd, 1e-12)


def _cov_report(pairs_loc: np.ndarray, v_scaf: np.ndarray, v_has: np.ndarray, v_ys: np.ndarray,
                v_fp: np.ndarray, bits_cnt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a = pairs_loc[:, 0]
    b = pairs_loc[:, 1]
    ss = ((v_scaf[a] >= 0) & (v_scaf[a] == v_scaf[b])).astype(np.float64)
    inter = np.bitwise_and(v_fp[a], v_fp[b]).sum(axis=1).astype(np.float64)
    den = bits_cnt[a].astype(np.float64) + bits_cnt[b].astype(np.float64) - inter
    ta = np.where(den > 0.0, inter / np.maximum(den, 1.0), 0.0)
    dh = np.abs(v_has[a] - v_has[b])
    dy = np.abs(v_ys[a] - v_ys[b])
    return ss, ta, dh, dy


def _make_hist_edges(arr: np.ndarray) -> np.ndarray:
    qs = np.quantile(arr, np.linspace(0.0, 1.0, 11))
    qs[0] = -1e12
    qs[-1] = 1e12
    return qs


def _init_hist(arr: np.ndarray, edges: np.ndarray) -> np.ndarray:
    h = np.zeros(len(edges) - 1, dtype=np.int64)
    idx = np.clip(np.searchsorted(edges, arr, side="right") - 1, 0, len(h) - 1)
    for k in idx.tolist():
        h[k] += 1
    return h


def _balance_loss_from_arrays(ss, ta, dh, dy, t_ss, t_ta_m, t_ta_s, t_dh_m, t_dh_s, t_dy_m, t_dy_s,
                              e_ta, e_dh, e_dy, h_ta_t, h_dh_t, h_dy_t) -> float:
    p = len(ss)
    l_ss = 16.0 * abs(float(ss.mean()) - t_ss) / 0.02

    def ml(arr, tm, ts, edges, ht):
        m = float(arr.mean())
        s = float(arr.std())
        h = _init_hist(arr, edges)
        return abs(m - tm) / ts + 0.7 * abs(s - ts) / ts + 1.0 * float(np.abs(h - ht).sum()) / p

    return (l_ss
            + 5.0 * ml(ta, t_ta_m, t_ta_s, e_ta, h_ta_t)
            + 2.5 * ml(dh, t_dh_m, t_dh_s, e_dh, h_dh_t)
            + 2.5 * ml(dy, t_dy_m, t_dy_s, e_dy, h_dy_t))


def _phase1_greedy_lambda(
    V: int,
    eu_all: np.ndarray,
    ev_all: np.ndarray,
    es_tan: np.ndarray,
    e_same_scaf: np.ndarray,
    top_nbrs: np.ndarray,
    top_sims: np.ndarray,
    forb_loc: list[set[int]],
    v_mega_sets: list[set[int]],
    v_has: np.ndarray,
    v_ys: np.ndarray,
    v_fp: np.ndarray,
    bits_cnt: np.ndarray,
    lam: float,
) -> tuple[np.ndarray, int]:
    """Build complete non-MMP perfect matching on V vertices with score = Tanimoto + lam * same_scaf."""
    score_e = es_tan + lam * e_same_scaf
    order = np.argsort(-score_e)
    eu = eu_all[order]
    ev = ev_all[order]
    partner = np.full(V, -1, dtype=np.int32)
    for a, b in zip(eu.tolist(), ev.tolist()):
        if partner[a] == -1 and partner[b] == -1:
            partner[a] = b
            partner[b] = a

    def ok_loc(a: int, b: int) -> bool:
        return a != b and (b not in forb_loc[a]) and v_mega_sets[a].isdisjoint(v_mega_sets[b])

    def tan_loc(a: int, b: int) -> float:
        inter = int(np.bitwise_and(v_fp[a], v_fp[b]).sum())
        den = float(bits_cnt[a] + bits_cnt[b] - inter)
        return inter / den if den > 0.0 else 0.0

    # Length-1 and length-3 augmenting paths over Top-K non-MMP neighbors
    k_g = top_nbrs.shape[1]
    for _pass in range(3):
        unm = np.flatnonzero(partner == -1)
        if len(unm) == 0:
            break
        aug = 0
        for a in unm.tolist():
            if partner[a] != -1:
                continue
            found = False
            for k in range(k_g):
                w = int(top_nbrs[a, k])
                if w < 0 or top_sims[a, k] <= 0.0:
                    break
                if partner[w] == -1:
                    partner[a] = w
                    partner[w] = a
                    aug += 1
                    found = True
                    break
            if found:
                continue
            best_aug, best_gain = None, -1e9
            for k in range(min(32, k_g)):
                w = int(top_nbrs[a, k])
                tw = float(top_sims[a, k])
                if w < 0 or tw <= 0.0:
                    break
                z = int(partner[w])
                if z < 0 or z == a:
                    continue
                for k2 in range(min(32, k_g)):
                    y = int(top_nbrs[z, k2])
                    ty = float(top_sims[z, k2])
                    if y < 0 or ty <= 0.0:
                        break
                    if y != a and partner[y] == -1:
                        gain = tw + ty - tan_loc(w, z)
                        if gain > best_gain:
                            best_gain = gain
                            best_aug = (w, z, y)
                        break
            if best_aug is not None:
                w, z, y = best_aug
                partner[a] = w
                partner[w] = a
                partner[z] = y
                partner[y] = z
                aug += 1
        if aug == 0:
            break

    pairs_list = []
    seen = np.zeros(V, dtype=bool)
    for a in range(V):
        b = int(partner[a])
        if b >= 0 and not seen[a]:
            seen[a] = seen[b] = True
            pairs_list.append([a, b])

    leftovers = np.flatnonzero(partner == -1).tolist()
    n_left = len(leftovers)
    leftovers.sort(key=lambda a: (v_has[a], v_ys[a]))
    while len(leftovers) >= 2:
        a = leftovers.pop()
        bj = -1
        for idx_j in range(len(leftovers) - 1, max(-1, len(leftovers) - 41), -1):
            if ok_loc(a, leftovers[idx_j]):
                bj = idx_j
                break
        if bj == -1:
            for idx_j in range(len(leftovers) - 1, -1, -1):
                if ok_loc(a, leftovers[idx_j]):
                    bj = idx_j
                    break
        if bj != -1:
            b = leftovers.pop(bj)
            pairs_list.append([a, b])
        else:
            b = leftovers.pop()
            fixed = False
            for pi, (x, y) in enumerate(pairs_list):
                if ok_loc(a, x) and ok_loc(b, y):
                    pairs_list[pi] = [a, x]
                    pairs_list.append([b, y])
                    fixed = True
                    break
                if ok_loc(a, y) and ok_loc(b, x):
                    pairs_list[pi] = [a, y]
                    pairs_list.append([b, x])
                    fixed = True
                    break
            if not fixed:
                leftovers.extend([a, b])
                break
    assert not leftovers, f"Failed to match {len(leftovers)} leftover vertices"
    return np.asarray(pairs_list, dtype=np.int32), n_left


def stage3_rewire_and_gate(
    archive_path: Path,
    scoring_shard_paths: dict[int, Path] | None,
    stage1_npz_path: Path,
    stage1_expected_sha256: str,
    out_dir: Path,
    *,
    shards: tuple[int, ...],
) -> dict:
    """Build matched non-MMP rewiring on V_paired and enforce Balance Gate (ZERO yhat / residuals)."""
    import torch

    t0, cpu0 = perf_counter(), process_time()
    verify_pinned(stage1_npz_path, stage1_expected_sha256, "stage1_real_pairs.npz")
    with np.load(stage1_npz_path, allow_pickle=False) as s1:
        target_indices = s1["target_indices"]
        real_pairs = s1["real_pairs"]
        scaf_ids = s1["scaf_ids"]
        noiso_ids = s1["noiso_ids"]
        has = s1["has"]
        fp_arr = s1["fp_arr"]
        forb_indptr = s1["forb_csr_indptr"]
        forb_indices = s1["forb_csr_indices"]
        mega_indptr = s1["mega_csr_indptr"]
        mega_indices = s1["mega_csr_indices"]
        k_mega = int(s1["mega_bucket_count"])

    # Read true labels from CSV and verify float32 equality against topology shards when present
    y_true = np.asarray([float(v) for v in _read_csv_field(archive_path, target_indices, 2)], dtype=np.float64)
    assert np.isfinite(y_true).all()
    if scoring_shard_paths is not None:
        from molgap.packed_graph_dataset import PackedGraphDataset
        y_sh = []
        for s in shards:
            sp = verify_pinned(scoring_shard_paths[s], SCORING_SHARD_SHA256[s], f"train_shard_{s:04d}.pt")
            lo, hi = s * SHARD_ROWS, (s + 1) * SHARD_ROWS
            sel = target_indices[(target_indices >= lo) & (target_indices < hi)]
            ds = PackedGraphDataset(sp, mmap=True).topology_only()
            assert np.array_equal(ds._data.source_idx.reshape(-1).numpy(), np.arange(lo, hi))
            y_sh.append(ds._data.y.reshape(-1).numpy()[sel - lo].astype(np.float32))
            del ds
        assert np.array_equal(y_true.astype(np.float32), np.concatenate(y_sh))

    M = len(real_pairs)
    verts = real_pairs.reshape(-1)
    V = len(verts)
    assert len(np.unique(verts)) == V
    v2pos = np.full(len(target_indices), -1, dtype=np.int32)
    v2pos[verts] = np.arange(V, dtype=np.int32)

    v_fp = fp_arr[verts]
    bits_cnt = v_fp.sum(axis=1).astype(np.float32)
    v_scaf = scaf_ids[verts]
    v_noiso = noiso_ids[verts]
    v_has = has[verts].astype(np.float64)
    v_ys = y_true[verts]

    # Map forbidden edges into local coordinates 0..V-1
    forb_loc = [set() for _ in range(V)]
    for pi, u in enumerate(verts.tolist()):
        st_i, en_i = int(forb_indptr[u]), int(forb_indptr[u + 1])
        for w in forb_indices[st_i:en_i].tolist():
            pj = int(v2pos[w])
            if pj >= 0:
                forb_loc[pi].add(pj)
    ni_groups = defaultdict(list)
    for pi, ni in enumerate(v_noiso.tolist()):
        if ni >= 0:
            ni_groups[ni].append(pi)
    for grp in ni_groups.values():
        if len(grp) > 1:
            for a_i in grp:
                for b_i in grp:
                    if a_i != b_i:
                        forb_loc[a_i].add(b_i)

    v_mega_sets = [set(mega_indices[int(mega_indptr[u]):int(mega_indptr[u + 1])].tolist()) for u in verts.tolist()]

    def ok_loc(a: int, b: int) -> bool:
        return a != b and (b not in forb_loc[a]) and v_mega_sets[a].isdisjoint(v_mega_sets[b])

    # Real pairs in local 0..V-1 coordinates: [0,1], [2,3], ...
    p_loc = np.arange(V, dtype=np.int32).reshape(M, 2)
    r_ss, r_ta, r_dh, r_dy = _cov_report(p_loc, v_scaf, v_has, v_ys, v_fp, bits_cnt)
    t_ss = float(r_ss.mean())
    t_ta_m, t_ta_s = float(r_ta.mean()), float(r_ta.std()) + 1e-9
    t_dh_m, t_dh_s = float(r_dh.mean()), float(r_dh.std()) + 1e-9
    t_dy_m, t_dy_s = float(r_dy.mean()), float(r_dy.std()) + 1e-9

    e_ta = _make_hist_edges(r_ta)
    e_dh = _make_hist_edges(r_dh)
    e_dy = _make_hist_edges(r_dy)
    h_ta_t = _init_hist(r_ta, e_ta)
    h_dh_t = _init_hist(r_dh, e_dh)
    h_dy_t = _init_hist(r_dy, e_dy)

    # GPU exact fp16 all-pairs non-MMP Top-K Tanimoto
    tg0 = perf_counter()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "Stage 3 Top-K similarity requires CUDA"
    X_gpu = torch.from_numpy(v_fp).to(device=dev, dtype=torch.float16)
    cnt_gpu = torch.from_numpy(bits_cnt).to(device=dev, dtype=torch.float32)
    scaf_gpu = torch.from_numpy(v_scaf).to(device=dev)
    M_gpu = None
    if k_mega > 0:
        mega_mat = np.zeros((V, k_mega), dtype=np.float16)
        for pi, mset in enumerate(v_mega_sets):
            if mset:
                mega_mat[pi, list(mset)] = 1.0
        M_gpu = torch.from_numpy(mega_mat).to(device=dev)
        del mega_mat

    top_nbrs = np.full((V, TOP_K_GLOBAL), -1, dtype=np.int32)
    top_sims = np.zeros((V, TOP_K_GLOBAL), dtype=np.float32)
    scaf_nbrs = np.full((V, TOP_K_SCAF), -1, dtype=np.int32)
    scaf_sims = np.zeros((V, TOP_K_SCAF), dtype=np.float32)

    chunk = 512
    for st in range(0, V, chunk):
        en = min(V, st + chunk)
        dot = torch.mm(X_gpu[st:en], X_gpu.t()).float()
        den = cnt_gpu[st:en, None] + cnt_gpu[None, :] - dot
        sim = torch.where(den > 0.0, dot / den, torch.zeros_like(dot))
        if M_gpu is not None:
            sim.masked_fill_(torch.mm(M_gpu[st:en], M_gpu.t()) > 0.5, -1.0)
        for li in range(en - st):
            gi = st + li
            sim[li, gi] = -1.0
            if forb_loc[gi]:
                idx_f = torch.tensor(list(forb_loc[gi]), device=dev, dtype=torch.long)
                sim[li, idx_f] = -1.0
        vals, idxs = torch.topk(sim, k=min(TOP_K_GLOBAL, V - 1), dim=1)
        top_sims[st:en, :vals.shape[1]] = vals.cpu().numpy()
        top_nbrs[st:en, :idxs.shape[1]] = idxs.cpu().numpy()

        sc_eq = (scaf_gpu[st:en, None] >= 0) & (scaf_gpu[st:en, None] == scaf_gpu[None, :])
        sim_sc = torch.where(sc_eq, sim, torch.full_like(sim, -1.0))
        sv, si = torch.topk(sim_sc, k=min(TOP_K_SCAF, V - 1), dim=1)
        scaf_sims[st:en, :sv.shape[1]] = sv.cpu().numpy()
        scaf_nbrs[st:en, :si.shape[1]] = si.cpu().numpy()

    del X_gpu, cnt_gpu, scaf_gpu, M_gpu
    torch.cuda.empty_cache()
    t_gpu = perf_counter() - tg0

    # Combine unique candidate undirected edges from top_nbrs[:, :32] + scaf_nbrs[:, :24]
    tp1 = perf_counter()
    u1 = np.repeat(np.arange(V, dtype=np.int32), 32)
    v1 = top_nbrs[:, :32].ravel()
    s1 = top_sims[:, :32].ravel()
    u2 = np.repeat(np.arange(V, dtype=np.int32), min(24, TOP_K_SCAF))
    v2 = scaf_nbrs[:, :min(24, TOP_K_SCAF)].ravel()
    s2 = scaf_sims[:, :min(24, TOP_K_SCAF)].ravel()

    uc = np.concatenate((u1, u2))
    vc = np.concatenate((v1, v2))
    sc_e = np.concatenate((s1, s2))
    valid = (vc >= 0) & (sc_e > 0.0) & (uc != vc)
    uc, vc, sc_e = uc[valid], vc[valid], sc_e[valid]
    lo_e = np.minimum(uc, vc)
    hi_e = np.maximum(uc, vc)
    keys_e = lo_e.astype(np.int64) * np.int64(V) + hi_e.astype(np.int64)
    _, uniq_idx = np.unique(keys_e, return_index=True)
    eu_all = lo_e[uniq_idx]
    ev_all = hi_e[uniq_idx]
    es_tan = sc_e[uniq_idx].astype(np.float64)
    e_same_scaf = ((v_scaf[eu_all] >= 0) & (v_scaf[eu_all] == v_scaf[ev_all])).astype(np.float64)

    # Sweep Lagrange multiplier lambda to hit joint same_Murcko + Tanimoto + covariate initialization
    best_lam = None
    best_init_loss = 1e18
    best_rew = None
    best_left = 0
    sweep_log = []
    for lam in REWIRING_LAMBDAS:
        cand_rew, n_left = _phase1_greedy_lambda(
            V, eu_all, ev_all, es_tan, e_same_scaf, top_nbrs, top_sims,
            forb_loc, v_mega_sets, v_has, v_ys, v_fp, bits_cnt, lam,
        )
        css, cta, cdh, cdy = _cov_report(cand_rew, v_scaf, v_has, v_ys, v_fp, bits_cnt)
        loss = _balance_loss_from_arrays(
            css, cta, cdh, cdy, t_ss, t_ta_m, t_ta_s, t_dh_m, t_dh_s, t_dy_m, t_dy_s,
            e_ta, e_dh, e_dy, h_ta_t, h_dh_t, h_dy_t,
        )
        sweep_log.append({
            "lambda": float(lam),
            "loss": float(loss),
            "same_Murcko_rew": float(css.mean()),
            "Tanimoto_mean_rew": float(cta.mean()),
            "p1_leftovers": int(n_left),
        })
        if loss < best_init_loss:
            best_init_loss = loss
            best_lam = float(lam)
            best_rew = cand_rew
            best_left = int(n_left)
    t_p1 = perf_counter() - tp1

    # Phase 2: 600,000 neighbor-guided + random 2-opt steps (deterministic seed)
    tp2 = perf_counter()
    rew_pairs = best_rew
    P = len(rew_pairs)
    assert P == M
    v2pair = np.empty(V, dtype=np.int32)
    for pi, (a, b) in enumerate(rew_pairs.tolist()):
        v2pair[a] = pi
        v2pair[b] = pi

    cur_ss, cur_ta, cur_dh, cur_dy = _cov_report(rew_pairs, v_scaf, v_has, v_ys, v_fp, bits_cnt)
    h_ta = _init_hist(cur_ta, e_ta)
    h_dh = _init_hist(cur_dh, e_dh)
    h_dy = _init_hist(cur_dy, e_dy)

    s_ss = float(cur_ss.sum())
    s_ta, s2_ta = float(cur_ta.sum()), float((cur_ta ** 2).sum())
    s_dh, s2_dh = float(cur_dh.sum()), float((cur_dh ** 2).sum())
    s_dy, s2_dy = float(cur_dy.sum()), float((cur_dy ** 2).sum())

    def bidx(val: float, edges: np.ndarray) -> int:
        return int(np.searchsorted(edges, val, side="right")) - 1

    def tan_loc(a: int, b: int) -> float:
        inter = int(np.bitwise_and(v_fp[a], v_fp[b]).sum())
        den = float(bits_cnt[a] + bits_cnt[b] - inter)
        return inter / den if den > 0.0 else 0.0

    def loss_fn(ss_s, ta_s, ta_2, dh_s, dh_2, dy_s, dy_2, hta, hdh, hdy):
        def ml(s, s2, tm, ts, h, ht):
            m = s / P
            v = max(s2 / P - m * m, 1e-12)
            return abs(m - tm) / ts + 0.7 * abs(math.sqrt(v) - ts) / ts + 1.0 * float(np.abs(h - ht).sum()) / P

        l_ss = 16.0 * abs((ss_s / P) - t_ss) / 0.02
        return (l_ss
                + 5.0 * ml(ta_s, ta_2, t_ta_m, t_ta_s, hta, h_ta_t)
                + 2.5 * ml(dh_s, dh_2, t_dh_m, t_dh_s, hdh, h_dh_t)
                + 2.5 * ml(dy_s, dy_2, t_dy_m, t_dy_s, hdy, h_dy_t))

    cur_loss = loss_fn(s_ss, s_ta, s2_ta, s_dh, s2_dh, s_dy, s2_dy, h_ta, h_dh, h_dy)
    rng = np.random.default_rng(SEED)
    acc = 0
    for step in range(N_OPT_STEPS):
        p1 = int(rng.integers(P))
        u1, v1 = int(rew_pairs[p1, 0]), int(rew_pairs[p1, 1])
        m_pick = step % 5
        if m_pick < 3:
            anchor = u1 if (step & 1) == 0 else v1
            other1 = v1 if anchor == u1 else u1
            if m_pick == 0 and scaf_nbrs[anchor, 0] >= 0:
                w = int(scaf_nbrs[anchor, int(rng.integers(TOP_K_SCAF))])
            else:
                w = int(top_nbrs[anchor, int(rng.integers(TOP_K_GLOBAL))])
            if w < 0:
                continue
            p2 = int(v2pair[w])
            if p2 == p1:
                continue
            u2, v2 = int(rew_pairs[p2, 0]), int(rew_pairs[p2, 1])
            other2 = v2 if w == u2 else u2
            na1, nb1, na2, nb2 = anchor, w, other1, other2
        else:
            p2 = int(rng.integers(P - 1))
            if p2 >= p1:
                p2 += 1
            u2, v2 = int(rew_pairs[p2, 0]), int(rew_pairs[p2, 1])
            if rng.random() < 0.5:
                na1, nb1, na2, nb2 = u1, u2, v1, v2
            else:
                na1, nb1, na2, nb2 = u1, v2, v1, u2
        if not (ok_loc(na1, nb1) and ok_loc(na2, nb2)):
            continue
        nss1 = 1.0 if (v_scaf[na1] >= 0 and v_scaf[na1] == v_scaf[nb1]) else 0.0
        nss2 = 1.0 if (v_scaf[na2] >= 0 and v_scaf[na2] == v_scaf[nb2]) else 0.0
        oss = cur_ss[p1] + cur_ss[p2]
        nss = nss1 + nss2

        nta1, nta2 = tan_loc(na1, nb1), tan_loc(na2, nb2)
        ndh1, ndh2 = abs(v_has[na1] - v_has[nb1]), abs(v_has[na2] - v_has[nb2])
        ndy1, ndy2 = abs(v_ys[na1] - v_ys[nb1]), abs(v_ys[na2] - v_ys[nb2])

        ns_ss = s_ss - oss + nss
        ns_ta = s_ta - cur_ta[p1] - cur_ta[p2] + nta1 + nta2
        ns2_ta = s2_ta - cur_ta[p1] ** 2 - cur_ta[p2] ** 2 + nta1 ** 2 + nta2 ** 2
        ns_dh = s_dh - cur_dh[p1] - cur_dh[p2] + ndh1 + ndh2
        ns2_dh = s2_dh - cur_dh[p1] ** 2 - cur_dh[p2] ** 2 + ndh1 ** 2 + ndh2 ** 2
        ns_dy = s_dy - cur_dy[p1] - cur_dy[p2] + ndy1 + ndy2
        ns2_dy = s2_dy - cur_dy[p1] ** 2 - cur_dy[p2] ** 2 + ndy1 ** 2 + ndy2 ** 2

        b1o, b2o, b1n, b2n = bidx(cur_ta[p1], e_ta), bidx(cur_ta[p2], e_ta), bidx(nta1, e_ta), bidx(nta2, e_ta)
        d1o, d2o, d1n, d2n = bidx(cur_dh[p1], e_dh), bidx(cur_dh[p2], e_dh), bidx(ndh1, e_dh), bidx(ndh2, e_dh)
        y1o, y2o, y1n, y2n = bidx(cur_dy[p1], e_dy), bidx(cur_dy[p2], e_dy), bidx(ndy1, e_dy), bidx(ndy2, e_dy)

        h_ta[b1o] -= 1; h_ta[b2o] -= 1; h_ta[b1n] += 1; h_ta[b2n] += 1
        h_dh[d1o] -= 1; h_dh[d2o] -= 1; h_dh[d1n] += 1; h_dh[d2n] += 1
        h_dy[y1o] -= 1; h_dy[y2o] -= 1; h_dy[y1n] += 1; h_dy[y2n] += 1

        new_loss = loss_fn(ns_ss, ns_ta, ns2_ta, ns_dh, ns2_dh, ns_dy, ns2_dy, h_ta, h_dh, h_dy)
        if new_loss <= cur_loss:
            rew_pairs[p1, 0], rew_pairs[p1, 1] = na1, nb1
            rew_pairs[p2, 0], rew_pairs[p2, 1] = na2, nb2
            v2pair[na1] = p1; v2pair[nb1] = p1; v2pair[na2] = p2; v2pair[nb2] = p2
            cur_ss[p1], cur_ss[p2] = nss1, nss2
            cur_ta[p1], cur_ta[p2] = nta1, nta2
            cur_dh[p1], cur_dh[p2] = ndh1, ndh2
            cur_dy[p1], cur_dy[p2] = ndy1, ndy2
            s_ss, s_ta, s2_ta, s_dh, s2_dh, s_dy, s2_dy = ns_ss, ns_ta, ns2_ta, ns_dh, ns2_dh, ns_dy, ns2_dy
            cur_loss = new_loss
            acc += 1
        else:
            h_ta[b1o] += 1; h_ta[b2o] += 1; h_ta[b1n] -= 1; h_ta[b2n] -= 1
            h_dh[d1o] += 1; h_dh[d2o] += 1; h_dh[d1n] -= 1; h_dh[d2n] -= 1
            h_dy[y1o] += 1; h_dy[y2o] += 1; h_dy[y1n] -= 1; h_dy[y2n] -= 1
    t_p2 = perf_counter() - tp2

    # Final verification & balance gate
    flat_rew = rew_pairs.reshape(-1)
    assert np.array_equal(np.sort(flat_rew), np.arange(V, dtype=np.int32)), "Vertex multiset mismatch"
    unmatched = int(V - len(rew_pairs) * 2)
    mmp_leak = sum(1 for a, b in rew_pairs.tolist() if not ok_loc(a, b))

    w_ss, w_ta, w_dh, w_dy = _cov_report(rew_pairs, v_scaf, v_has, v_ys, v_fp, bits_cnt)

    def cont_stats(r_arr, w_arr, scale=1.0):
        ra, wa = r_arr * scale, w_arr * scale
        return {
            "real_mean": float(ra.mean()),
            "real_std": float(ra.std()),
            "real_q10_50_90": [float(x) for x in np.quantile(ra, [0.1, 0.5, 0.9])],
            "rew_mean": float(wa.mean()),
            "rew_std": float(wa.std()),
            "rew_q10_50_90": [float(x) for x in np.quantile(wa, [0.1, 0.5, 0.9])],
            "smd": _smd(ra, wa),
            "ks": float(ks_2samp(ra, wa).statistic),
        }

    same_scaf_diff = abs(float(r_ss.mean()) - float(w_ss.mean()))
    ta_stats = cont_stats(r_ta, w_ta, 1.0)
    dh_stats = cont_stats(r_dh, w_dh, 1.0)
    dy_stats = cont_stats(r_dy, w_dy, 1000.0)

    gate_checks = {
        "unmatched_eq_0": unmatched == 0,
        "mmp_leak_eq_0": mmp_leak == 0,
        "same_murcko_diff_le_0.02": same_scaf_diff <= 0.02,
        "tanimoto_smd_le_0.10": abs(ta_stats["smd"]) <= 0.10,
        "tanimoto_ks_le_0.10": ta_stats["ks"] <= 0.10,
        "dha_smd_le_0.10": abs(dh_stats["smd"]) <= 0.10,
        "dha_ks_le_0.10": dh_stats["ks"] <= 0.10,
        "dgap_smd_le_0.10": abs(dy_stats["smd"]) <= 0.10,
        "dgap_ks_le_0.10": dy_stats["ks"] <= 0.10,
    }
    balance_gate_passed = all(gate_checks.values())

    # Convert rew_pairs from 0..V-1 back to 0..N-1 block row coordinates
    rew_block_pairs = verts[rew_pairs]
    assert np.array_equal(np.sort(rew_block_pairs.reshape(-1)), np.sort(verts))

    npz_path = out_dir / "stage3_rewiring.npz"
    npz_sha = save_npz_once(
        npz_path,
        target_indices=target_indices,
        y_true=y_true,
        rew_pairs=rew_block_pairs,
    )
    summary = {
        "stage": "stage3_rewire_and_gate",
        "pairs_count": int(P),
        "unmatched": unmatched,
        "mmp_leak": mmp_leak,
        "selected_lambda": best_lam,
        "p1_leftovers": best_left,
        "lambda_sweep": sweep_log,
        "swaps_accepted": int(acc),
        "covariate_balance": {
            "same_Murcko_real": float(r_ss.mean()),
            "same_Murcko_rew": float(w_ss.mean()),
            "same_Murcko_abs_diff": float(same_scaf_diff),
            "Tanimoto": ta_stats,
            "abs_dHA": dh_stats,
            "abs_dGap_meV": dy_stats,
        },
        "gate_checks": gate_checks,
        "balance_gate_passed": balance_gate_passed,
        "stage3_npz_sha256": npz_sha,
        "predictions_used": False,
        "timings_seconds": {"gpu_topk": t_gpu, "phase1": t_p1, "phase2": t_p2},
        "wall_seconds": perf_counter() - t0,
        "parent_cpu_seconds": process_time() - cpu0,
    }
    atomic_json(out_dir / "stage3_receipt.json", summary)
    return summary


def _cluster_bootstrap_recovery(
    n_total: int,
    clusters: np.ndarray,
    r_mev: np.ndarray,
    real_pairs: np.ndarray,
    rew_pairs: np.ndarray,
    *,
    n_boot: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> dict:
    """Exact paired cluster-bootstrap ratio estimator on whole-block recoverable MAE (meV)."""
    e_real = np.abs(r_mev[real_pairs[:, 0]]) + np.abs(r_mev[real_pairs[:, 1]]) - np.abs(r_mev[real_pairs[:, 0]] + r_mev[real_pairs[:, 1]])
    e_rew = np.abs(r_mev[rew_pairs[:, 0]]) + np.abs(r_mev[rew_pairs[:, 1]]) - np.abs(r_mev[rew_pairs[:, 0]] + r_mev[rew_pairs[:, 1]])

    g_real = np.zeros(n_total, dtype=np.float64)
    g_rew = np.zeros(n_total, dtype=np.float64)
    np.add.at(g_real, real_pairs[:, 0], 0.5 * e_real)
    np.add.at(g_real, real_pairs[:, 1], 0.5 * e_real)
    np.add.at(g_rew, rew_pairs[:, 0], 0.5 * e_rew)
    np.add.at(g_rew, rew_pairs[:, 1], 0.5 * e_rew)

    val_real = float(g_real.sum() / n_total)
    val_rew = float(g_rew.sum() / n_total)
    val_diff = val_real - val_rew

    k_clusters = int(clusters.max()) + 1
    c_n = np.bincount(clusters, minlength=k_clusters).astype(np.float64)
    c_gr = np.bincount(clusters, weights=g_real, minlength=k_clusters)
    c_gw = np.bincount(clusters, weights=g_rew, minlength=k_clusters)

    rng = np.random.default_rng(seed)
    draws_r = np.empty(n_boot, dtype=np.float64)
    draws_w = np.empty(n_boot, dtype=np.float64)
    draws_d = np.empty(n_boot, dtype=np.float64)
    chunk = 200
    for st in range(0, n_boot, chunk):
        cnt = min(chunk, n_boot - st)
        idx = rng.integers(0, k_clusters, size=(cnt, k_clusters))
        den = c_n[idx].sum(axis=1)
        rr = c_gr[idx].sum(axis=1) / den
        rw = c_gw[idx].sum(axis=1) / den
        draws_r[st:st + cnt] = rr
        draws_w[st:st + cnt] = rw
        draws_d[st:st + cnt] = rr - rw

    def pack(val, draws):
        lo, hi = np.quantile(draws, [0.025, 0.975])
        return {
            "value": float(val),
            "ci95": [float(lo), float(hi)],
            "n_bootstrap": int(n_boot),
            "n_clusters": int(k_clusters),
            "seed": int(seed),
        }

    return {
        "R_real_meV": pack(val_real, draws_r),
        "R_rew_meV": pack(val_rew, draws_w),
        "diff_real_minus_rew_meV": pack(val_diff, draws_d),
    }


def _score_model(
    n_total: int,
    yhat_ev: np.ndarray,
    y_ev: np.ndarray,
    real_pairs: np.ndarray,
    rew_pairs: np.ndarray,
    series_clusters: np.ndarray,
    murcko_clusters: np.ndarray,
) -> dict:
    r_mev = (yhat_ev - y_ev) * 1000.0
    dy_true_mev = (y_ev[real_pairs[:, 0]] - y_ev[real_pairs[:, 1]]) * 1000.0
    dy_pred_mev = (yhat_ev[real_pairs[:, 0]] - yhat_ev[real_pairs[:, 1]]) * 1000.0
    dy_rew_true_mev = (y_ev[rew_pairs[:, 0]] - y_ev[rew_pairs[:, 1]]) * 1000.0
    dy_rew_pred_mev = (yhat_ev[rew_pairs[:, 0]] - yhat_ev[rew_pairs[:, 1]]) * 1000.0

    corr_real = float(np.corrcoef(r_mev[real_pairs[:, 0]], r_mev[real_pairs[:, 1]])[0, 1])
    corr_rew = float(np.corrcoef(r_mev[rew_pairs[:, 0]], r_mev[rew_pairs[:, 1]])[0, 1])

    return {
        "whole_block_raw_mae_meV": float(np.abs(r_mev).mean()),
        "paired_subset_raw_mae_meV": float(np.abs(r_mev[real_pairs.reshape(-1)]).mean()),
        "residual_corr_real": corr_real,
        "residual_corr_rew": corr_rew,
        "descriptive_pair_dgap_mae_meV": {
            "real_pairs": float(np.abs(dy_pred_mev - dy_true_mev).mean()),
            "rewired_pairs": float(np.abs(dy_rew_pred_mev - dy_rew_true_mev).mean()),
            "real_dgap_r": float(np.corrcoef(dy_pred_mev, dy_true_mev)[0, 1]),
            "rew_dgap_r": float(np.corrcoef(dy_rew_pred_mev, dy_rew_true_mev)[0, 1]),
        },
        "series_clustered": _cluster_bootstrap_recovery(n_total, series_clusters, r_mev, real_pairs, rew_pairs),
        "murcko_clustered_descriptive": _cluster_bootstrap_recovery(n_total, murcko_clusters, r_mev, real_pairs, rew_pairs),
    }


def stage4_score_and_decide(
    s1_receipt: dict,
    s2_receipt: dict,
    s3_receipt: dict,
    out_dir: Path,
) -> dict:
    """Single-pass scoring & verdict iff Balance Gate passed."""
    t0, cpu0 = perf_counter(), process_time()
    if not s3_receipt["balance_gate_passed"]:
        result = {
            "format": "molgap-delta-gap-diag-results-v1",
            "verdict": "insufficient_evidence",
            "reason": "Pre-registered Stage 3 rewiring Balance Gate failed; residual scoring halted.",
            "stage1": s1_receipt,
            "stage2": s2_receipt,
            "stage3": s3_receipt,
            "arms": None,
            "ensemble_2init_descriptive": None,
            "wall_seconds": perf_counter() - t0,
            "parent_cpu_seconds": process_time() - cpu0,
        }
        atomic_json(out_dir / "results.json", result)
        return result

    s1_npz = verify_pinned(out_dir / "stage1_real_pairs.npz", s1_receipt["stage1_npz_sha256"], "stage1_real_pairs.npz")
    s2_npz = verify_pinned(out_dir / "stage2_predictions.npz", s2_receipt["stage2_npz_sha256"], "stage2_predictions.npz")
    s3_npz = verify_pinned(out_dir / "stage3_rewiring.npz", s3_receipt["stage3_npz_sha256"], "stage3_rewiring.npz")

    with np.load(s1_npz, allow_pickle=False) as s1, np.load(s2_npz, allow_pickle=False) as s2, np.load(s3_npz, allow_pickle=False) as s3:
        idx1, idx2, idx3 = s1["target_indices"], s2["target_indices"], s3["target_indices"]
        assert np.array_equal(idx1, idx2) and np.array_equal(idx1, idx3)
        n_total = len(idx1)
        real_pairs = s1["real_pairs"]
        rew_pairs = s3["rew_pairs"]
        series_clusters = s1["series_clusters"]
        murcko_clusters = s1["murcko_clusters"]
        y_ev = s3["y_true"]
        p43 = s2["pred_s43_ema60"]
        p44 = s2["pred_s44_ema60"]

    arms = {
        "s43": _score_model(n_total, p43, y_ev, real_pairs, rew_pairs, series_clusters, murcko_clusters),
        "s44": _score_model(n_total, p44, y_ev, real_pairs, rew_pairs, series_clusters, murcko_clusters),
    }
    ens2 = _score_model(n_total, 0.5 * (p43 + p44), y_ev, real_pairs, rew_pairs, series_clusters, murcko_clusters)

    stop_hit = any(arms[a]["series_clustered"]["R_real_meV"]["ci95"][1] < INVESTMENT_GATE_MEV for a in ("s43", "s44"))
    pass_hit = all(
        arms[a]["series_clustered"]["R_real_meV"]["value"] >= INVESTMENT_GATE_MEV
        and arms[a]["series_clustered"]["diff_real_minus_rew_meV"]["ci95"][0] > 0.0
        for a in ("s43", "s44")
    )
    if stop_hit:
        verdict = "STOP_NO_TRAIN"
    elif pass_hit:
        verdict = "PASS_DIAGNOSTIC"
    else:
        verdict = "INCONCLUSIVE_DO_NOT_TRAIN"

    result = {
        "format": "molgap-delta-gap-diag-results-v1",
        "verdict": verdict,
        "investment_gate_meV": INVESTMENT_GATE_MEV,
        "rows_total": n_total,
        "molecule_coverage": s1_receipt["molecule_coverage"],
        "arms": arms,
        "ensemble_2init_descriptive": ens2,
        "stage1": s1_receipt,
        "stage2": s2_receipt,
        "stage3": s3_receipt,
        "wall_seconds": perf_counter() - t0,
        "parent_cpu_seconds": process_time() - cpu0,
    }
    atomic_json(out_dir / "results.json", result)
    return result


def discover_kaggle_paths(input_root: Path) -> dict:
    """Resolve mounted Kaggle inputs purely by filenames + SHA verification (no hardcoded mount slugs)."""
    by_name = defaultdict(list)
    for p in input_root.rglob("*"):
        if p.is_file():
            by_name[p.name].append(p)

    def pick_pinned(filename: str, expected_sha: str, path_hint: str = "") -> Path:
        cands = [c for c in by_name.get(filename, []) if "checkpoint_chunks" not in c.parts]
        if path_hint:
            hinted = [c for c in cands if path_hint in c.as_posix()]
            if hinted:
                cands = hinted
        matches = [c for c in cands if sha256_file(c) == expected_sha]
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected 1 SHA-matched '{filename}' ({expected_sha[:12]}), found {len(matches)} among {[str(x) for x in cands]}")
        return matches[0]

    archive_path = pick_pinned("pcqm4m_v2_zip.bin", OFFICIAL_ARCHIVE_SHA256)
    dev_shard_path = pick_pinned("train_shard_0010.pt", FIXED_500K_DEV_SHARD_SHA256, "fixed-500k")
    scoring_shards = {s: pick_pinned(f"train_shard_{s:04d}.pt", SCORING_SHARD_SHA256[s]) for s in SCORING_SHARDS}

    arm_paths = {}
    for arm, pins in MODEL_PINS.items():
        hint = f"/init_{arm}/"
        arm_paths[arm] = {
            "init": pick_pinned(pins["init_file"], pins["init_sha256"]),
            "manifest": pick_pinned("output_manifest.json", pins["manifest_sha256"], hint),
            "contract": pick_pinned("training_contract.json", pins["contract_sha256"], hint),
            "selected_model": pick_pinned("selected_model.pt", pins["selected_model_sha256"], hint),
            "predictions": pick_pinned("development_predictions.pt", pins["predictions_sha256"], hint),
            "resume": pick_pinned("last_checkpoint.pt", pins["resume_sha256"], hint),
            "trace": pick_pinned("canonical_trace.json", pins["trace_sha256"], hint),
        }
    return {"archive": archive_path, "dev_shard": dev_shard_path, "scoring_shards": scoring_shards, "arms": arm_paths}


def run_local_smoke(out_dir: Path, n_workers: int):
    """Verify Stage 1 + Stage 3 Balance Gate + synthetic cluster bootstrap on ALREADY-CONSUMED Shards 63-64."""
    # Verify deterministic fit_idx matches saved spread-fit artifact bit-for-bit
    sf_path = Path("D:/molgap-artifacts/diagnostics/pcqm-residual-spread-fit/predict.npz")
    if sf_path.is_file():
        with np.load(sf_path, allow_pickle=False) as sf:
            assert np.array_equal(compute_spread_fit_idx(), sf["fit_idx"]), "Deterministic fit_idx mismatch!"

    target_indices = build_target_indices(LOCAL_SMOKE_SHARDS)
    assert target_indices[-1] < TAIL_START
    archive_path = Path("D:/molgap-data/ogb_pcqm4mv2_raw/pcqm4m-v2.zip")

    out_dir.mkdir(parents=True, exist_ok=True)
    for f in ("stage1_real_pairs.npz", "stage3_rewiring.npz", "stage1_receipt.json", "stage3_receipt.json", "smoke_summary.json"):
        (out_dir / f).unlink(missing_ok=True)

    s1 = stage1_structures_and_real_pairs(archive_path, target_indices, out_dir, n_workers=n_workers)
    s3 = stage3_rewire_and_gate(archive_path, None, out_dir / "stage1_real_pairs.npz", s1["stage1_npz_sha256"],
                                out_dir, shards=LOCAL_SMOKE_SHARDS)

    # Synthetic zero-signal Gaussian bootstrap sanity check (NO model residuals read!)
    with np.load(out_dir / "stage1_real_pairs.npz", allow_pickle=False) as d1, np.load(out_dir / "stage3_rewiring.npz", allow_pickle=False) as d3:
        rng = np.random.default_rng(1)
        syn_r = rng.normal(0.0, 100.0, size=len(target_indices))
        boot = _cluster_bootstrap_recovery(len(target_indices), d1["series_clusters"], syn_r, d1["real_pairs"], d3["rew_pairs"])

    smoke = {
        "mode": "local-smoke-consumed-63-64",
        "rows_total": int(len(target_indices)),
        "stage1": s1,
        "stage3": s3,
        "synthetic_bootstrap_sanity": boot,
    }
    atomic_json(out_dir / "smoke_summary.json", smoke)
    print(json.dumps({
        "rows_total": s1["rows_total"],
        "real_pairs": s1["real_pairs_count"],
        "coverage": round(s1["molecule_coverage"], 4),
        "max_series_share": round(s1["max_series_cluster_share"], 4),
        "selected_lambda": s3["selected_lambda"],
        "balance_gate_passed": s3["balance_gate_passed"],
        "gate_checks": s3["gate_checks"],
        "covariate_balance": s3["covariate_balance"],
    }, indent=2), flush=True)
    if not s3["balance_gate_passed"]:
        raise SystemExit(2)


def _run_diagnostic(paths: dict, out_dir: Path, n_workers: int, *, dev_shard_expected_sha: str):
    target_indices = build_target_indices(SCORING_SHARDS)
    out_dir.mkdir(parents=True, exist_ok=True)

    s1 = stage1_structures_and_real_pairs(paths["archive"], target_indices, out_dir, n_workers=n_workers)
    print("STAGE1 DONE:", json.dumps({k: s1[k] for k in ("rows_total", "real_pairs_count", "molecule_coverage", "wall_seconds")}), flush=True)

    s2 = stage2_predict_ema60(
        target_indices,
        paths["scoring_shards"],
        paths["dev_shard"],
        paths["arms"],
        out_dir,
        dev_shard_expected_sha=dev_shard_expected_sha,
    )
    print("STAGE2 DONE:", json.dumps({k: s2[k] for k in ("rows_total", "replay_max_abs_eV", "dev_ema60_abs_diff_eV", "wall_seconds")}), flush=True)

    s3 = stage3_rewire_and_gate(paths["archive"], paths["scoring_shards"], out_dir / "stage1_real_pairs.npz",
                                s1["stage1_npz_sha256"], out_dir, shards=SCORING_SHARDS)
    print("STAGE3 DONE:", json.dumps({"balance_gate_passed": s3["balance_gate_passed"], "selected_lambda": s3["selected_lambda"], "gate_checks": s3["gate_checks"], "wall_seconds": s3["wall_seconds"]}), flush=True)

    res = stage4_score_and_decide(s1, s2, s3, out_dir)
    print("STAGE4 DONE:", json.dumps({"verdict": res["verdict"], "rows_total": res.get("rows_total")}), flush=True)


def run_kaggle(input_root: Path, out_dir: Path, n_workers: int):
    paths = discover_kaggle_paths(input_root)
    _run_diagnostic(paths, out_dir, n_workers, dev_shard_expected_sha=FIXED_500K_DEV_SHARD_SHA256)


def run_local_full(out_dir: Path, n_workers: int):
    full_root = Path("D:/molgap-data/pcqm_full_train_v1")
    verify_pinned(full_root / "manifest.json", FULL_TRAIN_MANIFEST_SHA256, "pcqm_full_train_v1 manifest.json")
    run_root = Path("D:/molgap-artifacts/retrieved/pcqm-gptrans-pairnorm-init-noise-500k-s43/v1/version1")
    init_root = Path("D:/molgap-artifacts/initializations/gptrans-pairnorm-init-noise-500k")
    paths = {
        "archive": Path("D:/molgap-data/ogb_pcqm4mv2_raw/pcqm4m-v2.zip"),
        "dev_shard": full_root / "store/topology/train/train_shard_0010.pt",
        "scoring_shards": {s: full_root / f"store/topology/train/train_shard_{s:04d}.pt" for s in SCORING_SHARDS},
        "arms": {
            arm: {
                "init": init_root / f"{arm}.pt",
                "manifest": run_root / f"init_{arm}" / "output_manifest.json",
                "contract": run_root / f"init_{arm}" / "training_contract.json",
                "selected_model": run_root / f"init_{arm}" / "selected_model.pt",
                "predictions": run_root / f"init_{arm}" / "development_predictions.pt",
                "resume": run_root / f"init_{arm}" / "last_checkpoint.pt",
                "trace": run_root / f"init_{arm}" / "canonical_trace.json",
            }
            for arm in ("s43", "s44")
        },
    }
    _run_diagnostic(paths, out_dir, n_workers, dev_shard_expected_sha=FULL_TRAIN_DEV_SHARD_SHA256)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("local-smoke", "local-full", "kaggle"), required=True)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.mode == "local-smoke":
        run_local_smoke(args.out_dir, args.workers)
    elif args.mode == "local-full":
        run_local_full(args.out_dir, args.workers)
    else:
        run_kaggle(args.input_root, args.out_dir, args.workers)


if __name__ == "__main__":
    main()
