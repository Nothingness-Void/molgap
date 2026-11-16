"""Official PCQM4Mv2 validation MAE for an equal-weight ensemble of finished gptrans_t B1024-runner endpoints.

Same two phases as pcqm_official_valid_eval.py, always separate processes:
  gate   rebuild train rows (shards below the tail holdout) from SMILES, compare with the stored shards and compare
         ensemble predictions on stored vs rebuilt graphs. Reads train rows only; tail and protected roles are not read.
  valid  requires a passed gate for this exact script, an authority note and --authorize-official-validation-read.
         Reads exactly the official valid rows once and writes the ensemble aggregate MAE only (no per-member score,
         no per-row predictions). Refuses to run when an evidence file already exists, so the read happens once.
test-dev and test-challenge have no code path here.
Members: --member NAME=RUN_DIR,INIT_FILE, RUN_DIR holds ema_model.pt and run_config.json (weights key "model").
"""
import argparse, hashlib, json, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pcqm_official_valid_eval as base
import pcqm_tail_holdout_eval as tail

GATE_SHARDS = (0, 22, 44, 64)
CHECKPOINT = "ema_model.pt"


def load_members(specs, device):
    """[(name, model, sha)] and the shared predict function; every member must share one normalisation."""
    members, stats, predict = [], set(), None
    for spec in specs:
        name, rest = spec.split("=", 1)
        run_dir, init = rest.split(",", 1)
        run_dir = Path(run_dir)
        cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        assert cfg["train_rows"] <= tail.TAIL_START, f"{name} overlaps the tail holdout"
        stats.add((cfg["target_mean_eV"], cfg["target_std_eV"]))
        load, predict, _ = tail.gptrans_scorer(tail.runner_recipe(run_dir / "run_config.json"), init, "model")
        members.append((name, load(run_dir / CHECKPOINT, device), base.sha256(run_dir / CHECKPOINT)))
    assert len(stats) == 1, "members disagree on target normalisation"
    return members, predict, stats.pop()


def ensemble_ev(members, predict, graphs, mean, std, device):
    return sum(predict(model, graphs, mean, std, device) for _, model, _ in members) / len(members)


def run_gate(args):
    from torch_geometric.transforms import AddRandomWalkPE
    from molgap.pcqm_topology import load_packed_topology_shard
    base.setup_determinism()
    device = torch.device(args.device)
    raw_csv = Path(args.raw_dir) / "pcqm4m-v2" / "raw" / "data.csv.gz"
    members, predict, (mean, std) = load_members(args.member, device)
    wanted, stored = {}, []
    for shard in GATE_SHARDS:
        shard_graphs = load_packed_topology_shard(Path(args.data_root) / "store" / "topology" / "train" / f"train_shard_{shard:04d}.pt")
        start = len(shard_graphs) // 2 if shard in (22, 44) else 0
        for local in range(start, min(start + base.GATE_ROWS_PER_SHARD, len(shard_graphs))):
            wanted[shard * base.SHARD_ROWS + local] = shard_graphs[local]
    assert max(wanted) < tail.TAIL_START
    smiles = {}
    for i, row in enumerate(base.csv_rows(raw_csv)):
        if i in wanted:
            smiles[i] = row
        if i >= max(wanted):
            break
    pe = AddRandomWalkPE(walk_length=16, attr_name="random_walk_pe")
    rebuilt, counts, failures = [], {k: 0 for k in (*base.EXACT_FIELDS, "random_walk_pe")}, []
    for idx in sorted(wanted):
        ref = wanted[idx]
        assert int(ref.source_idx) == idx and int(smiles[idx][0]) == idx
        new = base.build_graph(smiles[idx][1], smiles[idx][2], pe)
        for field in (*base.EXACT_FIELDS, "random_walk_pe"):
            a, b = new[field], ref[field]
            same = torch.equal(a, b) if field != "random_walk_pe" else (a.shape == b.shape and float((a - b).abs().max()) <= base.RWPE_TOL)
            if not same:
                counts[field] += 1
                failures.append([idx, field])
        stored.append(ref)
        rebuilt.append(new)
    truth = torch.cat([g.y for g in stored]).double()
    first = ensemble_ev(members, predict, stored, mean, std, device)
    again = ensemble_ev(members, predict, stored, mean, std, device)
    new = ensemble_ev(members, predict, rebuilt, mean, std, device)
    check = {"mae_stored_ev": float((first - truth).abs().mean()), "mae_rebuilt_ev": float((new - truth).abs().mean()),
             "max_abs_pred_diff_stored_vs_rebuilt_ev": float((first - new).abs().max()),
             "noise_floor_max_abs_diff_stored_twice_ev": float((first - again).abs().max())}
    check["mae_delta_ev"] = abs(check["mae_stored_ev"] - check["mae_rebuilt_ev"])
    rate = len(failures) / (len(wanted) * len(counts))
    passed = (check["max_abs_pred_diff_stored_vs_rebuilt_ev"] <= base.PRED_TOL_EV and check["mae_delta_ev"] <= base.MAE_TOL_EV
              and rate <= base.MISMATCH_RATE_LIMIT)
    evidence = {"format": "molgap-gptrans-ensemble-valid-gate-v1", "passed": passed,
                "protected_roles_read": {"official_validation": False, "test_dev": False, "test_challenge": False},
                "sample": {"rows": len(wanted), "shards": list(GATE_SHARDS)}, "field_mismatch_counts": counts,
                "field_mismatch_rate": rate, "rate_limit": base.MISMATCH_RATE_LIMIT, "failures_head": failures[:20],
                "prediction_check": check, "prediction_tolerances_ev": {"max_abs_pred_diff": base.PRED_TOL_EV, "mae_delta": base.MAE_TOL_EV},
                "normalisation": {"mean_eV": mean, "std_eV": std}, "members": {n: s for n, _, s in members},
                "raw_data_csv_sha256": base.sha256(raw_csv), "script_sha256": base.sha256(__file__), "environment": base.environment()}
    out = Path(args.evidence)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({"passed": passed, "field_mismatch_counts": counts, "prediction_check": check}, indent=2))
    return 0 if passed else 1


def run_valid(args):
    from torch_geometric.transforms import AddRandomWalkPE
    from ogb.lsc import PCQM4Mv2Evaluator
    out = Path(args.evidence)
    gate = json.loads((out / "gate_evidence.json").read_text(encoding="utf-8"))
    if not gate["passed"] or gate["script_sha256"] != base.sha256(__file__):
        raise SystemExit("STOP: a passed gate for this exact script is required")
    if not args.authorize_official_validation_read or not Path(args.authority_note).is_file():
        raise SystemExit("STOP: official_validation requires the authority note and --authorize-official-validation-read")
    if (out / "official_valid_evidence.json").exists():
        raise SystemExit("STOP: official validation was already read for this evidence directory")
    base.setup_determinism()
    device = torch.device(args.device)
    members, predict, (mean, std) = load_members(args.member, device)
    if {n: s for n, _, s in members} != gate["members"] or (mean, std) != (gate["normalisation"]["mean_eV"], gate["normalisation"]["std_eV"]):
        raise SystemExit("STOP: members or normalisation changed since the gate")
    raw_dir = Path(args.raw_dir) / "pcqm4m-v2"
    raw_csv = raw_dir / "raw" / "data.csv.gz"
    split = torch.load(raw_dir / "split_dict.pt", weights_only=False)
    valid_idx = torch.as_tensor(split["valid"]).long()
    train_idx = torch.as_tensor(split["train"]).long()
    del split
    if not (len(valid_idx) == base.VALID_ROWS and valid_idx.unique().numel() == base.VALID_ROWS and int(valid_idx.min()) >= base.TRAIN_ROWS
            and torch.equal(train_idx, torch.arange(base.TRAIN_ROWS))):
        raise SystemExit("STOP: official train/valid split is not the expected shape")
    wanted, last = set(valid_idx.tolist()), int(valid_idx.max())
    pe = AddRandomWalkPE(walk_length=16, attr_name="random_walk_pe")
    graphs, digest = [], hashlib.sha256()
    for i, row in enumerate(base.csv_rows(raw_csv)):
        if i > last:
            break
        if i not in wanted:
            continue
        assert int(row[0]) == i
        digest.update(f"{row[0]},{row[1]}\n".encode())
        graphs.append(base.build_graph(row[1], row[2], pe))
    assert len(graphs) == base.VALID_ROWS
    truth = torch.cat([g.y for g in graphs]).double()
    pred = ensemble_ev(members, predict, graphs, mean, std, device)
    result = {"official_valid_mae_ev": float(PCQM4Mv2Evaluator().eval({"y_pred": pred.float(), "y_true": truth.float()})["mae"]),
              "rows": len(graphs), "mae_float64_ev": float((pred - truth).abs().mean())}
    evidence = {"format": "molgap-gptrans-ensemble-valid-eval-v1",
                "protected_roles_read": {"official_validation": True, "test_dev": False, "test_challenge": False},
                "result": result, "members": gate["members"], "weights": "equal", "valid_rows_idx_smiles_sha256": digest.hexdigest(),
                "normalisation": gate["normalisation"], "authority_note_sha256": base.sha256(args.authority_note),
                "gate_evidence_sha256": base.sha256(out / "gate_evidence.json"), "raw_data_csv_sha256": gate["raw_data_csv_sha256"],
                "split_dict_sha256": base.sha256(raw_dir / "split_dict.pt"), "script_sha256": base.sha256(__file__),
                "environment": base.environment(), "per_row_predictions_saved": False}
    (out / "official_valid_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("phase", choices=("gate", "valid"))
    parser.add_argument("--raw-dir", required=True, help="folder holding the extracted pcqm4m-v2 directory")
    parser.add_argument("--data-root", help="pcqm_full_train_v1 root (gate only)")
    parser.add_argument("--member", action="append", required=True, metavar="NAME=RUN_DIR,INIT_FILE")
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
