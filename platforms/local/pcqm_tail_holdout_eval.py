"""Score finished neural_atom_k1 or gptrans_t checkpoints on the tail_holdout_v1 role (TRAIN rows 3_250_000..3_378_605).

The tail is reserved: no run may train or select on it. A run with train_rows > TAIL_START is refused,
so the full K1/GPT-T runs (which trained on every row) cannot be scored here as evidence. Official validation,
test-dev and test-challenge are never read.
Graphs come from the stored topology shards (the training pipeline's own graphs); the csv supplies
labels only, for normalisation. Each run is normalised with the mean/std of its own train rows. For gptrans_t pass --recipe and --initial-state of the run (the frozen contract rebuilds the exact model, addons
included); --weights-key is model for selected_model.pt and ema for a last_checkpoint.pt snapshot. Passing --dev-range also scores that
run's development rows (they are TRAIN rows, a selection role of that run) so the normalisation and
weights can be checked against the MAE recorded in the run's own trace before the tail number is trusted.
"""
import argparse, hashlib, json, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pcqm_official_valid_eval as base

TAIL_START, TAIL_END, SHARD_ROWS = 3_250_000, 3_378_606, 50_000
ROLE = "tail_holdout_v1"


def gptrans_scorer(recipe_path, initial_state, key):
    """(load, predict) for gptrans_t: the frozen recipe rebuilds the model, a checkpoint supplies its weights."""
    from copy import copy
    from torch_geometric.loader import DataLoader
    from molgap.family_training_kernel import load_family_model, mode_policy
    from molgap.family_training_runner import read_recipe
    from molgap.gptrans_execution_backend import prepare_layout, prepared_forward
    from molgap.gptrans_execution_hotpath import MODEL_FIELDS
    _, recipe = read_recipe(recipe_path)
    prepare = mode_policy(recipe.family, recipe.mode).prepare_batch

    def load(path, device):
        model, _ = load_family_model(recipe, Path(initial_state))
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)[key], strict=True)
        return model.to(device).eval()

    def predict(model, graphs, mean, std, device, batch_size=1024):
        out = []
        with torch.no_grad():
            for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
                batch = prepare_layout(copy(batch))
                if prepare is not None:
                    batch = prepare(batch, training=False)
                out.append(prepared_forward(model, batch.to(device, *MODEL_FIELDS)).double().cpu())
        return torch.cat(out) * std + mean
    return load, predict, recipe


def parse_run(text):
    name, rest = text.split("=", 1)
    path, rows = rest.rsplit(":", 1)
    return name, Path(path), int(rows)


def parse_range(text):
    start, stop = (int(part) for part in text.split(":"))
    assert 0 <= start < stop <= TAIL_START, "development rows must be TRAIN rows below the tail"
    assert start % SHARD_ROWS == 0 and stop % SHARD_ROWS == 0, "development range must be shard aligned"
    return start, stop


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", required=True, help="folder holding the extracted pcqm4m-v2 directory")
    parser.add_argument("--data-root", required=True, help="pcqm_full_train_v1 root with store/topology/train shards")
    parser.add_argument("--run", action="append", required=True, metavar="NAME=CHECKPOINT:TRAIN_ROWS")
    parser.add_argument("--dev-range", type=parse_range, help="START:STOP of the runs' development role")
    parser.add_argument("--expect-dev-mae", type=float, help="recorded dev MAE (eV) of the first run, checked to 1e-4")
    parser.add_argument("--recipe", help="frozen training_contract.json of a gptrans_t run")
    parser.add_argument("--initial-state", help="the run's frozen initial state file (gptrans_t)")
    parser.add_argument("--weights-key", default="model", choices=["model", "ema"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--src", default=str(Path(__file__).resolve().parents[2] / "src"))
    args = parser.parse_args(argv)
    sys.path.insert(0, args.src)
    runs = [parse_run(text) for text in args.run]
    for name, _, rows in runs:
        if rows > TAIL_START:
            raise SystemExit(f"STOP: {name} trained on {rows} rows and overlaps {ROLE} (starts at {TAIL_START})")
    from molgap.pcqm_topology import load_packed_topology_shard
    from ogb.lsc import PCQM4Mv2Evaluator
    base.setup_determinism()
    device = torch.device(args.device)
    raw_csv = Path(args.raw_dir) / "pcqm4m-v2" / "raw" / "data.csv.gz"
    most = max(rows for _, _, rows in runs)
    labels = torch.empty(most, dtype=torch.float32)
    for i, row in enumerate(base.csv_rows(raw_csv)):
        if i >= most:
            break
        labels[i] = float(row[2])

    def rows_of(start, stop):
        graphs = []
        for shard in range(start // SHARD_ROWS, (stop - 1) // SHARD_ROWS + 1):
            path = Path(args.data_root) / "store" / "topology" / "train" / f"train_shard_{shard:04d}.pt"
            graphs += [g for g in load_packed_topology_shard(path)]
        ids = [int(g.source_idx) for g in graphs]
        assert ids == list(range(start, stop)), "stored shards do not cover the requested rows in order"
        return graphs, hashlib.sha256(json.dumps(ids).encode()).hexdigest()

    sets, digests = {}, {}
    sets[ROLE], digests[ROLE] = rows_of(TAIL_START, TAIL_END)
    sets["development"], digests["development"] = rows_of(*args.dev_range) if args.dev_range else ([], None)
    evaluator = PCQM4Mv2Evaluator()
    load_model, predict, recipe = base.load_model, base.predict_ev, None
    if args.recipe:
        assert args.initial_state, "--recipe needs --initial-state"
        load_model, predict, recipe = gptrans_scorer(args.recipe, args.initial_state, args.weights_key)
    results = {}
    for name, path, rows in runs:
        targets = labels[:rows].double()
        mean, std = float(targets.mean()), float(targets.std(correction=1))
        if recipe is not None:
            assert abs(recipe.target_mean_eV - mean) < 1e-6 and abs(recipe.target_std_eV - std) < 1e-6, "recipe normalisation differs from csv"
        model = load_model(path, device)
        results[name] = {"train_rows": rows, "checkpoint_sha256": base.sha256(path)}
        for key, graphs in sets.items():
            if not graphs:
                continue
            truth = torch.cat([g.y for g in graphs]).double()
            pred = predict(model, graphs, mean, std, device)
            results[name][key] = {"mae_ev": float(evaluator.eval({"y_pred": pred.float(), "y_true": truth.float()})["mae"]), "rows": len(graphs)}
    if args.expect_dev_mae is not None:
        observed = results[runs[0][0]]["development"]["mae_ev"]
        if abs(observed - args.expect_dev_mae) > 1e-4:
            raise SystemExit(f"STOP: dev MAE {observed:.6f} differs from recorded {args.expect_dev_mae:.6f}; do not trust the tail number")
    evidence = {"format": "molgap-tail-holdout-eval-v1", "role": ROLE, "rows": [TAIL_START, TAIL_END],
                "tail_source_idx_sha256": digests[ROLE], "dev_range": args.dev_range, "dev_source_idx_sha256": digests["development"],
                "protected_roles_read": {"official_validation": False, "test_dev": False, "test_challenge": False},
                "results": results, "script_sha256": base.sha256(__file__), "environment": base.environment(),
                "per_row_predictions_saved": False}
    Path(args.out).write_text(json.dumps(evidence, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
