"""Official PCQM4Mv2 validation MAE for finished neural_atom_k1 checkpoints.

Two phases, always separate processes:
  gate   rebuild train rows from SMILES, compare with the stored shards and compare predictions on
         stored vs rebuilt graphs. Reads train rows only and writes the gate evidence.
  valid  requires a passed gate, an authority note and --authorize-official-validation-read. Reads
         exactly the official valid rows once, scores every checkpoint, writes aggregate MAE only.
test-dev and test-challenge have no code path here; only the train and valid index sets are used.
"""
import argparse, copy, csv, gzip, hashlib, json, os, platform, sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

TRAIN_ROWS, VALID_ROWS, SHARD_ROWS = 3_378_606, 73_545, 50_000
GATE_SHARDS = (0, 22, 44, 66, 67)
GATE_ROWS_PER_SHARD = 1_000
RWPE_TOL, MISMATCH_RATE_LIMIT = 1e-6, 0.02
PRED_TOL_EV, MAE_TOL_EV = 1e-4, 1e-5
EXACT_FIELDS = ("x", "edge_index", "edge_attr", "y", "smiles_sanitized")
CHECKPOINTS = ("selected_model.pt", "ema_model.pt")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_rows(path):
    with gzip.open(path, "rt") as f:
        reader = csv.reader(f)
        assert next(reader) == ["idx", "smiles", "homolumogap"]
        yield from reader


def build_graph(smiles, label, pe):
    from rdkit import Chem
    from torch_geometric.data import Data
    from molgap.ogb_features import atom_to_ogb_feature_vector, bond_to_ogb_feature_vector
    mol = Chem.MolFromSmiles(smiles)
    x = torch.tensor([atom_to_ogb_feature_vector(a) for a in mol.GetAtoms()], dtype=torch.long)
    edges, attrs = [], []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feature = bond_to_ogb_feature_vector(bond)
        edges += [(a, b), (b, a)]
        attrs += [feature, feature]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty(2, 0, dtype=torch.long)
    edge_attr = torch.tensor(attrs, dtype=torch.long) if attrs else torch.empty(0, 3, dtype=torch.long)
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=x.shape[0])
    graph = pe(graph)
    graph.y = torch.tensor([float(label)], dtype=torch.float32)
    graph.smiles_sanitized = torch.tensor([1], dtype=torch.int8)
    return graph


def load_model(path, device):
    from molgap.qm9_neural_atom import make_encoder
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = make_encoder("neural_atom_k1")
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval()


def predict_ev(model, graphs, mean, std, device, batch_size=1024):
    from torch_geometric.loader import DataLoader
    from molgap.k1_execution_layout import cpu_layout_context
    from molgap.family_model_ops import k1_batch_to_device, k1_forward
    out = []
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
            batch = copy.copy(batch)
            with cpu_layout_context(model, batch, device):
                out.append(k1_forward(model, k1_batch_to_device(batch, device, training=False)).double().cpu())
    return torch.cat(out) * std + mean


def train_statistics(raw_csv):
    """Mean and unbiased std of the train labels, from the first TRAIN_ROWS csv rows only."""
    labels = torch.empty(TRAIN_ROWS, dtype=torch.float32)
    for i, row in enumerate(csv_rows(raw_csv)):
        if i == TRAIN_ROWS:
            break
        labels[i] = float(row[2])
    targets = labels.double()
    return float(targets.mean()), float(targets.std(correction=1)), labels


def environment():
    import numpy, rdkit, torch_geometric
    return {"torch": torch.__version__, "torch_geometric": torch_geometric.__version__, "rdkit": rdkit.__version__,
            "numpy": numpy.__version__, "python": platform.python_version(), "platform": platform.platform(),
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def setup_determinism():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def run_gate(args):
    from torch_geometric.transforms import AddRandomWalkPE
    from molgap.pcqm_topology import load_packed_topology_shard
    setup_determinism()
    device = torch.device(args.device)
    raw_csv = Path(args.raw_dir) / "pcqm4m-v2" / "raw" / "data.csv.gz"
    mean, std, _ = train_statistics(raw_csv)
    if abs(std - 1.161) > 0.01:
        raise SystemExit(f"STOP: train-label std {std} is not about 1.161")
    wanted = {}
    shards = {}
    for shard in GATE_SHARDS:
        path = Path(args.data_root) / "store" / "topology" / "train" / f"train_shard_{shard:04d}.pt"
        shards[shard] = load_packed_topology_shard(path)
        rows = len(shards[shard])
        start = (rows // 2) if shard in (22, 44) else 0
        for local in range(start, min(start + GATE_ROWS_PER_SHARD, rows)):
            wanted[shard * SHARD_ROWS + local] = (shard, local)
    last = max(wanted)
    assert last < TRAIN_ROWS
    smiles = {}
    for i, row in enumerate(csv_rows(raw_csv)):
        if i in wanted:
            smiles[i] = row
        if i >= last:
            break
    pe = AddRandomWalkPE(walk_length=16, attr_name="random_walk_pe")
    stored, rebuilt = [], []
    counts = {k: {"exact": 0, "within_tolerance": 0, "over_tolerance": 0} for k in (*EXACT_FIELDS, "random_walk_pe")}
    rwpe_bad, rwpe_max_abs, rwpe_max_rel, failures = [], 0.0, 0.0, []
    for idx in sorted(wanted):
        shard, local = wanted[idx]
        ref = shards[shard][local]
        assert int(ref.source_idx) == idx and int(smiles[idx][0]) == idx
        new = build_graph(smiles[idx][1], smiles[idx][2], pe)
        for field in EXACT_FIELDS:
            ok = torch.equal(new[field], ref[field])
            counts[field]["exact" if ok else "over_tolerance"] += 1
            if not ok:
                failures.append([idx, field])
        a, b = new.random_walk_pe, ref.random_walk_pe
        if a.shape != b.shape:
            counts["random_walk_pe"]["over_tolerance"] += 1
            failures.append([idx, "random_walk_pe_shape"])
        elif torch.equal(a, b):
            counts["random_walk_pe"]["exact"] += 1
        else:
            diff = (a - b).abs()
            rwpe_bad.append(idx)
            rwpe_max_abs = max(rwpe_max_abs, float(diff.max()))
            rwpe_max_rel = max(rwpe_max_rel, float((diff / b.abs().clamp_min(1e-12)).max()))
            counts["random_walk_pe"]["within_tolerance" if float(diff.max()) <= RWPE_TOL else "over_tolerance"] += 1
        stored.append(ref)
        rebuilt.append(new)
    total = len(wanted)
    over = sum(c["over_tolerance"] for c in counts.values())
    mismatch_rate = (len(rwpe_bad) + sum(1 for f in failures if f[1] != "random_walk_pe_shape")) / total
    passed = over == 0 and mismatch_rate <= MISMATCH_RATE_LIMIT
    predictions = {}
    for name in CHECKPOINTS:
        model = load_model(Path(args.checkpoints) / name, device)
        first = predict_ev(model, stored, mean, std, device)
        again = predict_ev(model, stored, mean, std, device)
        new = predict_ev(model, rebuilt, mean, std, device)
        truth = torch.cat([g.y for g in stored]).double()
        entry = {"mae_stored_ev": float((first - truth).abs().mean()), "mae_rebuilt_ev": float((new - truth).abs().mean()),
                 "max_abs_pred_diff_stored_vs_rebuilt_ev": float((first - new).abs().max()),
                 "noise_floor_max_abs_diff_stored_twice_ev": float((first - again).abs().max())}
        entry["mae_delta_ev"] = abs(entry["mae_stored_ev"] - entry["mae_rebuilt_ev"])
        entry["passed"] = (entry["max_abs_pred_diff_stored_vs_rebuilt_ev"] <= PRED_TOL_EV and entry["mae_delta_ev"] <= MAE_TOL_EV)
        predictions[name] = entry
        passed = passed and entry["passed"]
    evidence = {
        "format": "molgap-official-valid-gate-v1", "passed": passed, "protected_roles_read": {
            "official_validation": False, "test_dev": False, "test_challenge": False},
        "sample": {"rows": total, "shards": list(GATE_SHARDS), "rows_per_shard": GATE_ROWS_PER_SHARD,
                   "source_idx_ranges": {str(s): [min(i for i, (sh, _) in wanted.items() if sh == s),
                                                  max(i for i, (sh, _) in wanted.items() if sh == s)] for s in GATE_SHARDS}},
        "per_array_counts": counts, "random_walk_pe": {"mismatching_source_idx": rwpe_bad, "max_abs_diff": rwpe_max_abs,
                                                       "max_rel_diff": rwpe_max_rel, "tolerance": RWPE_TOL,
                                                       "tolerance_set_after_observing_4_of_1000_ulp_mismatches": True},
        "mismatch_rate_limit": MISMATCH_RATE_LIMIT, "other_failures": failures[:20],
        "prediction_check": predictions, "prediction_tolerances_ev": {"max_abs_pred_diff": PRED_TOL_EV, "mae_delta": MAE_TOL_EV},
        "normalisation": {"mean_eV": mean, "std_eV": std, "source": "train labels (first 3,378,606 csv rows), float32 labels, float64 mean, unbiased std"},
        "notes": ["The original shard builder source is unavailable, so bit-reproduction of random_walk_pe cannot be shown.",
                  "Graphs are built with molgap.ogb_features plus PyG AddRandomWalkPE(walk_length=16), float32.",
                  "Alternatives tried on 1,000 rows: dense float32 matmul 4/1000 mismatching (same as PyG); float64-then-cast and matrix_power 994/1000 mismatching, max 3.6e-07."],
        "environment": environment(), "checkpoint_sha256": {n: sha256(Path(args.checkpoints) / n) for n in CHECKPOINTS},
        "raw_data_csv_sha256": sha256(raw_csv), "script_sha256": sha256(__file__)}
    out = Path(args.evidence)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({"passed": passed, "rwpe_mismatching_rows": len(rwpe_bad), "rwpe_max_abs": rwpe_max_abs,
                      "std": std, "mean": mean, "prediction_check": predictions}, indent=2))
    return 0 if passed else 1


def run_valid(args):
    from torch_geometric.transforms import AddRandomWalkPE
    from ogb.lsc import PCQM4Mv2Evaluator
    out = Path(args.evidence)
    gate = json.loads((out / "gate_evidence.json").read_text(encoding="utf-8"))
    if not gate["passed"] or gate["script_sha256"] != sha256(__file__):
        raise SystemExit("STOP: a passed gate for this exact script is required")
    if not args.authorize_official_validation_read or not Path(args.authority_note).is_file():
        raise SystemExit("STOP: official_validation requires the authority note and --authorize-official-validation-read")
    for name in CHECKPOINTS:
        if sha256(Path(args.checkpoints) / name) != gate["checkpoint_sha256"][name]:
            raise SystemExit("STOP: checkpoint changed since the gate")
    setup_determinism()
    device = torch.device(args.device)
    raw_dir = Path(args.raw_dir) / "pcqm4m-v2"
    raw_csv = raw_dir / "raw" / "data.csv.gz"
    # Pickled numpy index arrays from the official archive, whose sha256 equals the manifest value.
    split = torch.load(raw_dir / "split_dict.pt", weights_only=False)
    valid_idx = torch.as_tensor(split["valid"]).long()
    train_idx = torch.as_tensor(split["train"]).long()
    del split
    if not (len(valid_idx) == VALID_ROWS and valid_idx.unique().numel() == VALID_ROWS and int(valid_idx.min()) >= TRAIN_ROWS
            and torch.equal(train_idx, torch.arange(TRAIN_ROWS))):
        raise SystemExit("STOP: official train/valid split is not the expected shape")
    wanted = set(valid_idx.tolist())
    last = int(valid_idx.max())
    mean, std = gate["normalisation"]["mean_eV"], gate["normalisation"]["std_eV"]
    pe = AddRandomWalkPE(walk_length=16, attr_name="random_walk_pe")
    graphs, digest = [], hashlib.sha256()
    for i, row in enumerate(csv_rows(raw_csv)):
        if i > last:
            break
        if i not in wanted:
            continue
        assert int(row[0]) == i
        digest.update(f"{row[0]},{row[1]}\n".encode())
        graphs.append(build_graph(row[1], row[2], pe))
    assert len(graphs) == VALID_ROWS
    truth = torch.cat([g.y for g in graphs]).double()
    evaluator = PCQM4Mv2Evaluator()
    results = {}
    for name in CHECKPOINTS:
        model = load_model(Path(args.checkpoints) / name, device)
        pred = predict_ev(model, graphs, mean, std, device)
        results[name] = {"official_valid_mae_ev": float(evaluator.eval(
            {"y_pred": pred.float(), "y_true": truth.float()})["mae"]), "rows": len(graphs),
            "mae_float64_ev": float((pred - truth).abs().mean())}
    evidence = {"format": "molgap-official-valid-eval-v1", "protected_roles_read": {
        "official_validation": True, "test_dev": False, "test_challenge": False}, "results": results,
        "valid_rows": VALID_ROWS, "valid_rows_idx_smiles_sha256": digest.hexdigest(),
        "manifest_official_row_manifest_sha256_comparable": False,
        "normalisation": gate["normalisation"], "authority_note_sha256": sha256(args.authority_note),
        "checkpoint_sha256": gate["checkpoint_sha256"], "gate_evidence_sha256": sha256(out / "gate_evidence.json"),
        "raw_data_csv_sha256": gate["raw_data_csv_sha256"], "split_dict_sha256": sha256(raw_dir / "split_dict.pt"),
        "script_sha256": sha256(__file__), "environment": environment(), "per_row_predictions_saved": False}
    (out / "official_valid_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(results, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("phase", choices=("gate", "valid"))
    parser.add_argument("--raw-dir", required=True, help="folder holding the extracted pcqm4m-v2 directory")
    parser.add_argument("--data-root", help="pcqm_full_train_v1 root (gate only)")
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--authority-note")
    parser.add_argument("--authorize-official-validation-read", action="store_true")
    parser.add_argument("--src", default=str(Path(__file__).resolve().parents[2] / "src"))
    args = parser.parse_args(argv)
    sys.path.insert(0, args.src)
    return run_gate(args) if args.phase == "gate" else run_valid(args)


if __name__ == "__main__":
    sys.exit(main())
