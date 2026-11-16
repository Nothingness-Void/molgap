"""Login-node CPU check for one card of the GPT-T 500K capacity question. Trains nothing.

Run it with the job's own interpreter and PYTHONPATH. It imports the launcher and driver, checks the runner,
manifest and the first ten train shards, loads the runner module, then for each arm of the card sets the model
shape, builds the recipe exactly as the launcher does and loads the init strictly. The parameter count must match.
  python login_check.py --card s42 --root <root>            (the root has code/, inputs/, data/)
  python login_check.py --card s42 --root <dir> --no-shards  (local run without the shard store)
"""
import argparse, importlib.util, json, os, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--card", choices=["s42", "s43"], required=True)
ap.add_argument("--root", type=Path, required=True)
ap.add_argument("--no-shards", action="store_true")
a = ap.parse_args()
root = a.root.resolve()
sys.path.insert(0, str(root / "code" / "platforms" / "ims" / "gptrans_minus_tail"))
import run_gptrans_width_500k as L
import run_width_500k_serial as D  # noqa: F401  (import check only)
import torch
from ogb.graphproppred.mol_encoder import AtomEncoder  # noqa: F401
import molgap
print("torch", torch.__version__, "| molgap", Path(molgap.__file__).parent, flush=True)

runner = root / "code" / L.RUNNER_REL
assert L.sha256(runner) == L.RUNNER_SHA, "runner sha differs"
manifest = root / "data" / "ogb-train-full" / "manifest.json"
if not a.no_shards:
    assert L.sha256(manifest) == L.MANIFEST_SHA, "manifest sha differs"
    train = [r for r in json.loads(manifest.read_text(encoding="utf-8"))["assets"]["topology"] if r["role"] == "train"]
    L.check_shards(manifest.parent, train)
    print("manifest and ten train shards ok", flush=True)

spec = importlib.util.spec_from_file_location("train_full_b1024", runner)
mod = importlib.util.module_from_spec(spec)
sys.modules["train_full_b1024"] = mod
spec.loader.exec_module(mod)
from molgap.family_training_kernel import load_family_model
from molgap.gptrans_bond_local import gptrans_shape, make_bond_local_model

for name in [f"{x}-{a.card}" for x in ("R", "L20", "W320", "F3")]:
    arm = L.ARMS[name]
    init_name, init_file_sha, init_state_sha = arm["init"]
    for k, v in arm["shape"].items():
        os.environ[f"MOLGAP_GPTRANS_{k}"] = str(v)
    init = root / "inputs" / init_name
    assert L.sha256(init) == init_file_sha, f"{init_name} sha differs"
    assert gptrans_shape() == arm["shape"], f"{name}: shape differs"
    recipe, _ = mod.make_recipe("gptrans_pairnorm", total_rows=L.MAX_SHARDS * 50_000, gptrans_batch_size=L.BATCH, gptrans_lr_mult=L.LR_MULT,
                                gptrans_ema_decay=L.EMA, gptrans_init_sha=init_state_sha, gptrans_stats=L.STATS_500K)
    assert (recipe.target_mean_eV, recipe.target_std_eV) == L.STATS_500K
    model, _ = load_family_model(recipe, init)
    n = sum(p.numel() for p in model.parameters())
    assert n == arm["params"] == sum(p.numel() for p in make_bond_local_model().parameters()), f"{name}: {n} params"
    print(f"{name}: strict load ok | params {n:,} | lr {recipe.learning_rate:.4g} | steps {recipe.optimizer_steps:,}", flush=True)
print("all four arms ok")
