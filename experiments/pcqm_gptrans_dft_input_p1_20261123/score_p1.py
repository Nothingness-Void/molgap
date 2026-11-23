"""Offline dev scoring of the six P1 chain items, gates (i)-(iv), paired bootstrap (run with -I from anywhere).

Reads only TRAIN rows: development rows 500,000..549,999 (labels from the stored topology shard, DFT coordinates from the
local dev sidecar for the 3D modes only). Never reads tail_holdout_v1, data.csv.gz, official valid or test. The 2D score uses
graphs that carry no coordinate field and never opens a sidecar. Truth is `g.y`; normalisation comes from the frozen recipe.

Usage: score_p1.py --retrieved DIR --fixed-root DIR --dev-sidecar DIR --out DIR [--limit N] [--items a,b]
  DIR/training/<item>/{selected_model.pt,train_log.jsonl}; --limit scores the first N dev rows (local checks only).
"""
import argparse
import json
import sys
from copy import copy
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
C1 = HERE.parent / "pcqm_gptrans_dft_distance_aux_20261121"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "platforms" / "local"))
sys.path.insert(0, str(C1))
from score_dev_and_distance import bootstrap, sha256_file, shard_graphs  # noqa: E402 (reused: shard check, paired bootstrap)

INITS = {42: C1 / "init" / "s44.pt", 43: C1 / "init" / "s43.pt"}
ITEMS = {f"{arm}-500k-s{seed}": (f"{arm}_s{seed}", seed, arm) for seed in (42, 43) for arm in ("ctl", "p1", "zer")}
DEV_SHARD, BATCH = 10, 256


def check_dev_sidecar(directory):
    manifest = json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
    rows, alignment = manifest["rows"], manifest["alignment"]
    match = (alignment["identity_order"] + alignment["isomorphism_mapped"]) / rows["rows"]
    if (manifest["status"] != "complete" or rows["source_idx_start"] != 500_000 or rows["source_idx_stop"] != 550_000
            or match < 0.999):
        raise SystemExit(f"STOP: dev sidecar is not complete rows 500000..549999 with topology match >= 99.9% ({match})")
    return match


def with_geometry(graphs, directory):
    """Shallow copies carrying DFT coordinates; the clean graphs stay coordinate-free for the 2D score."""
    from molgap.gptrans_dft_distance_aux import SHARD_FORMAT
    content = torch.load(Path(directory) / f"shard_{DEV_SHARD:04d}.pt", map_location="cpu", weights_only=True)
    assert content["format"] == SHARD_FORMAT and int(content["source_idx"][0]) == DEV_SHARD * 50_000
    offsets, result = content["atom_offsets"], []
    for i, graph in enumerate(graphs):
        a, b = int(offsets[i]), int(offsets[i + 1])
        assert b - a == graph.x.shape[0]
        graph = copy(graph)
        graph.dft_pos, graph.dft_valid = content["pos"][a:b], content["valid"][i:i + 1]
        result.append(graph)
    return result


def predict(model, graphs, mean, std, device, mode):
    """mode: 2d (no distance state), 3d (3D-only, clean coordinates) or 2d3d (clean coordinates, nothing masked)."""
    from torch_geometric.loader import DataLoader
    from molgap.gptrans_dft_distance_aux import EXTRA_FIELDS
    from molgap.gptrans_dft_input_p1 import clear_inputs, set_inputs
    from molgap.gptrans_execution_backend import prepare_layout, prepared_forward
    from molgap.gptrans_execution_hotpath import MODEL_FIELDS
    out = []
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=BATCH, shuffle=False):
            batch = prepare_layout(copy(batch))
            if mode == "2d":
                assert not hasattr(batch, "dft_pos")
                batch = batch.to(device, *MODEL_FIELDS)
            else:
                batch = batch.to(device, *MODEL_FIELDS, *EXTRA_FIELDS)
                valid = batch.dft_valid.view(-1).bool()
                set_inputs(model, batch, valid, valid if mode == "3d" else torch.zeros_like(valid))
            try:
                out.append(prepared_forward(model, batch).double().cpu())
            finally:
                if mode != "2d":
                    clear_inputs(model)
    return torch.cat(out) * std + mean


def step_time(path):
    """Median seconds per optimizer step over passes (consecutive train_log rows of one process; a restart row is skipped)."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    pairs = [(a, b) for a, b in zip(rows, rows[1:]) if b["elapsed_s"] > a["elapsed_s"] and b["step"] > a["step"]]
    per_step = [(b["elapsed_s"] - a["elapsed_s"]) / (b["step"] - a["step"]) for a, b in pairs]
    if not per_step:
        return {"median_s_per_step": None, "passes_logged": len(rows)}
    median = float(np.median(per_step))
    # A resume row spans a process restart (start-up plus a partial pass); a 3x outlier is dropped and counted, not hidden.
    kept = [(p, s) for p, s in zip(pairs, per_step) if s <= 3 * median]
    return {"median_s_per_step": median, "passes_logged": len(rows), "intervals_dropped_as_restart": len(pairs) - len(kept),
            "wall_s_sum_of_intervals": float(sum(b["elapsed_s"] - a["elapsed_s"] for (a, b), _ in kept))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--retrieved", required=True)
    parser.add_argument("--fixed-root", required=True)
    parser.add_argument("--dev-sidecar", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=50_000)
    parser.add_argument("--items")
    args = parser.parse_args(argv)
    from pcqm_tail_holdout_eval import gptrans_scorer
    match = check_dev_sidecar(args.dev_sidecar)
    device = torch.device("cuda")
    graphs = shard_graphs(args.fixed_root, DEV_SHARD)[:args.limit]
    graphs3d = with_geometry(graphs, args.dev_sidecar)
    truth = torch.cat([g.y for g in graphs]).double()
    result = {"format": "molgap-gptdft-input-p1-dev-v1", "dev_rows": [500_000, 500_000 + len(graphs)],
              "dev_sidecar": {"manifest_sha256": sha256_file(Path(args.dev_sidecar) / "manifest.json"), "topology_match": match},
              "protected_roles_read": {"tail_holdout_v1": False, "official_validation": False, "test": False}, "items": {}}
    errors = {}
    for name in args.items.split(",") if args.items else ITEMS:
        folder, seed, arm = ITEMS[name]
        load, _, recipe = gptrans_scorer(HERE / folder / "recipe.json", INITS[seed], "model")
        checkpoint = Path(args.retrieved) / "training" / name / "selected_model.pt"
        model = load(checkpoint, device)
        entry = {"checkpoint_sha256": sha256_file(checkpoint), "mae_eV": {}}
        errors[name] = {}
        for mode in ("2d",) if arm == "ctl" else ("2d", "3d", "2d3d"):
            pred = predict(model, graphs if mode == "2d" else graphs3d, recipe.target_mean_eV, recipe.target_std_eV, device, mode)
            errors[name][mode] = (pred - truth).abs().numpy()
            entry["mae_eV"][mode] = float(errors[name][mode].mean())
            print(name, mode, entry["mae_eV"][mode], flush=True)
        log = Path(args.retrieved) / "training" / name / "train_log.jsonl"
        entry["step_time"] = step_time(log) if log.is_file() else "evidence missing: no train_log.jsonl"
        summary = Path(args.retrieved) / "training" / "serial_summary.json"
        attempts = json.loads(summary.read_text(encoding="utf-8"))["items"][name]["attempts"] if summary.is_file() else None
        entry["wall_s_all_attempts"] = sum(x["wall_s"] for x in attempts) if attempts else "evidence missing: no serial_summary.json"
        result["items"][name] = entry
        del model
        torch.cuda.empty_cache()
    pairs = {}
    for seed in (42, 43):
        c, p, z = (errors.get(f"{arm}-500k-s{seed}") for arm in ("ctl", "p1", "zer"))
        if p is None:
            continue
        pair = {"p1_3d_minus_p1_2d": bootstrap(p["3d"], p["2d"]), "p1_2d3d_minus_p1_2d": bootstrap(p["2d3d"], p["2d"])}
        if c is not None:
            pair["p1_2d_minus_control_2d"] = bootstrap(p["2d"], c["2d"])
        if z is not None:
            pair["p1_2d_minus_zeros_2d"] = bootstrap(p["2d"], z["2d"])
        pairs[f"s{seed}"] = pair
    result["pairs"] = pairs
    gates = {}
    if len(pairs) == 2 and all("p1_2d_minus_control_2d" in v and "p1_2d_minus_zeros_2d" in v for v in pairs.values()):
        delta = lambda key: [pairs[s][key]["delta_mev"] for s in pairs]
        gates = {"i": all(x <= -10.0 for x in delta("p1_3d_minus_p1_2d")),
                 "ii": float(np.mean(delta("p1_2d_minus_control_2d"))) <= -2.0 and all(x < 0 for x in delta("p1_2d_minus_control_2d")),
                 "iii": all(x <= -1.0 for x in delta("p1_2d_minus_zeros_2d")),
                 "iv": "read step_time per item: P1 and zeros median <= 1.5 x control, wall sum <= 2.0 x control"}
    result["gates"] = gates
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dev_analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gates": gates, "pairs": pairs}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
