"""Dev-only scoring of the four finished serial-chain checkpoints, paired bootstrap, and head distance error.

Reads only TRAIN rows: development roles (100,000..149,999 and 500,000..549,999) and, for the distance head, sidecar rows.
It never reads the tail_holdout_v1 rows (>= 3,250,000), data.csv.gz, official valid or test. Truth is `g.y` from the stored
topology shards; normalisation comes from the frozen recipe. Scoring builds no sidecar and passes no coordinate field, so the
d-arm results also show pure-2D inference works. A reproduced dev MAE that differs from selected_eval.json by more than 1e-4
stops the run.

Usage (run with -I): score_dev_and_distance.py --retrieved DIR --fixed-root DIR --out DIR
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "platforms" / "local"))
SHARD_ROWS = 50_000
TAIL_START = 3_250_000
SEED, DRAWS = 20261123, 2000
ITEMS = {  # name: (recipe dir, init, dev shard, train rows)
    "c100-gptt-100k-s43": ("c100_gptrans_pun_s43", "s43.pt", 2, 100_000),
    "d100-dft-100k-s43": ("d100_dft_s43", "s43.pt", 2, 100_000),
    "c500-gptt-500k-s44": ("c500_gptrans_pun_s44", "s44.pt", 10, 500_000),
    "d500-dft-500k-s44": ("d500_dft_s44", "s44.pt", 10, 500_000),
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def shard_graphs(root, shard):
    from molgap.pcqm_topology import load_packed_topology_shard
    assert (shard + 1) * SHARD_ROWS <= TAIL_START, "refusing to read tail rows"
    graphs = list(load_packed_topology_shard(Path(root) / "store" / "topology" / "train" / f"train_shard_{shard:04d}.pt"))
    assert [int(g.source_idx) for g in graphs] == list(range(shard * SHARD_ROWS, (shard + 1) * SHARD_ROWS))
    return graphs


def device_batch(batch, device, fields):
    from copy import copy
    from molgap.gptrans_execution_backend import prepare_layout
    from molgap.gptrans_execution_hotpath import MODEL_FIELDS
    batch = prepare_layout(copy(batch))
    return batch.to(device, *MODEL_FIELDS, *fields)


def dev_scores(args, device):
    from pcqm_tail_holdout_eval import gptrans_scorer
    out, truth_by_shard = {}, {}
    for name, (folder, init, shard, rows) in ITEMS.items():
        recipe_path = HERE / folder / "recipe.json"
        load, predict, recipe = gptrans_scorer(recipe_path, HERE / "init" / init, "model")
        graphs = truth_by_shard.setdefault(shard, shard_graphs(args.fixed_root, shard))
        truth = torch.cat([g.y for g in graphs]).double()
        checkpoint = Path(args.retrieved) / "training" / name / "selected_model.pt"
        model = load(checkpoint, device)
        pred = predict(model, graphs, recipe.target_mean_eV, recipe.target_std_eV, device)
        mae = float((pred - truth).abs().mean())
        recorded = json.loads((checkpoint.parent / "selected_eval.json").read_text())["dev_mae_eV"]
        print(name, "dev MAE", mae, "recorded", recorded, flush=True)
        if abs(mae - recorded) > 1e-4:
            raise SystemExit(f"STOP: {name} reproduced dev MAE {mae:.6f} differs from recorded {recorded:.6f}")
        out[name] = {"dev_mae_eV": mae, "recorded_dev_mae_eV": recorded, "checkpoint_sha256": sha256_file(checkpoint),
                     "abs_error": (pred - truth).abs().numpy()}
        del model
        torch.cuda.empty_cache()
    return out


def bootstrap(delta_a, delta_b):
    """Paired bootstrap of mean(abs_err_a - abs_err_b) over development rows."""
    diff = delta_a - delta_b
    rng = np.random.default_rng(SEED)
    means = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(DRAWS)])
    return {"delta_mev": float(diff.mean() * 1000), "ci95_mev": [float(np.percentile(means, 2.5) * 1000),
            float(np.percentile(means, 97.5) * 1000)], "draws": DRAWS, "seed": SEED, "rows": len(diff)}


def sidecar_rows(root, shard):
    from molgap.gptrans_dft_distance_aux import SHARD_FORMAT
    path = Path(root) / "sidecars" / "pcqm-dft-train-geometry-500k-v3" / f"shard_{shard:04d}.pt"
    content = torch.load(path, map_location="cpu", weights_only=True)
    assert content["format"] == SHARD_FORMAT
    return content["pos"], content["atom_offsets"], content["valid"]


def distance_mae(model, root, shard, device, limit, batch_size=64):
    """Pair-count weighted L1 (Angstrom) of the head over real i<j heavy-atom pairs of valid rows (objective() definition)."""
    from torch_geometric.loader import DataLoader
    from molgap.gptrans_execution_backend import prepared_forward
    graphs = shard_graphs(root, shard)[:limit]
    pos, offsets, valid = sidecar_rows(root, shard)
    kept = []
    for i, g in enumerate(graphs):
        if bool(valid[i]):
            a, b = int(offsets[i]), int(offsets[i + 1])
            assert b - a == g.x.shape[0]
            g.dft_pos, g.dft_valid = pos[a:b], torch.ones(1, dtype=torch.bool)
            kept.append(g)
    addon = model._molgap_addons["dft_distance"]
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in DataLoader(kept, batch_size=batch_size, shuffle=False):
            batch = device_batch(batch, device, ("dft_pos", "dft_valid"))
            addon.capture, addon.pair = True, None
            try:
                prepared_forward(model, batch)
            finally:
                addon.capture = False
            pair, addon.pair = addon.pair, None
            layout = batch.execution_layout
            size, widest = layout["batch_size"], layout["max_nodes"]
            real = layout["mask"]
            pairs = real[:, :, None] & real[:, None, :] & torch.triu(
                torch.ones(widest, widest, dtype=torch.bool, device=real.device), diagonal=1)
            coords = batch.dft_pos.new_zeros(size * widest, 3)
            coords[layout["flat_index"]] = batch.dft_pos
            coords = coords.view(size, widest, 3)
            dist = (coords[:, :, None] - coords[:, None, :]).pow(2).sum(-1).sqrt()
            states = pair[:, :, 1:widest + 1, 1:widest + 1].permute(0, 2, 3, 1)
            predicted = addon.head(states).squeeze(-1)
            predicted = 0.5 * (predicted + predicted.transpose(1, 2))
            total += float(torch.where(pairs, (predicted - dist).abs(), 0.0).sum())
            count += int(pairs.sum())
    return {"mae_angstrom": total / count, "pairs": count, "rows": len(kept), "invalid_rows_skipped": len(graphs) - len(kept)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--retrieved", required=True)
    parser.add_argument("--fixed-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--distance-rows", type=int, default=SHARD_ROWS)
    parser.add_argument("--skip-distance", action="store_true")
    args = parser.parse_args(argv)
    device = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scores = dev_scores(args, device)
    for name, item in scores.items():
        np.save(out / f"{name}_dev_abs_error.npy", item["abs_error"])
    result = {"format": "molgap-gptdft-dev-analysis-v1", "dev_rows": {"100k": [100_000, 150_000], "500k": [500_000, 550_000]},
              "protected_roles_read": {"tail_holdout_v1": False, "official_validation": False, "test": False},
              "items": {n: {k: v for k, v in s.items() if k != "abs_error"} for n, s in scores.items()},
              "bootstrap": {
                  "d100_minus_c100": bootstrap(scores["d100-dft-100k-s43"]["abs_error"], scores["c100-gptt-100k-s43"]["abs_error"]),
                  "d500_minus_c500": bootstrap(scores["d500-dft-500k-s44"]["abs_error"], scores["c500-gptt-500k-s44"]["abs_error"])}}
    if not args.skip_distance:
        from pcqm_tail_holdout_eval import gptrans_scorer
        result["distance_head"] = {}
        for name in ("d100-dft-100k-s43", "d500-dft-500k-s44"):
            folder, init, _, rows = ITEMS[name]
            load, _, _ = gptrans_scorer(HERE / folder / "recipe.json", HERE / "init" / init, "model")
            model = load(Path(args.retrieved) / "training" / name / "selected_model.pt", device)
            entry = {"train_rows_0_to_50k": distance_mae(model, args.fixed_root, 0, device, args.distance_rows)}
            if rows == 100_000:  # sidecar rows 150,000..199,999 are unseen by d100
                entry["heldout_rows_150k_to_200k"] = distance_mae(model, args.fixed_root, 3, device, args.distance_rows)
            else:
                entry["heldout"] = "evidence missing: the sidecar ends at row 499,999, the whole d500 train role"
            result["distance_head"][name] = entry
            print(name, entry, flush=True)
    (out / "dev_analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["bootstrap"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
