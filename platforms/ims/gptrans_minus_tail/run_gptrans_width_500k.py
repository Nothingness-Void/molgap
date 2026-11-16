"""IMS launcher for one arm of the GPT-T 500K capacity question (experiments/pcqm_gptrans_width_500k_20261117).

Checks identities and exits with code 3 before any GPU work if something is off, then runs the registered
B1024 runner unchanged on the first 10 train shards (rows [0, 500,000)). The model shape comes from the
MOLGAP_GPTRANS_{LAYERS,WIDTH,HEADS,FFN_RATIO} variables, which run_width_500k_serial.py sets from ARMS below;
the shape is recorded in arm_record.json beside the runner's run_config.json, which is where the tail scorer reads it.
As in the levers precedent, the runner stays byte-identical and only IMS's torch 2.7.1 pin is swapped into the recipe.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

RUNNER_SHA = "120d5571593c56647afe8f7c6f6060afeb7781ff8aea101fb9454fdac0ee5b37"
MANIFEST_SHA = "7d358a679d299bd5de6c5822e42dcef34b0e0b9fa9598c684709509c0b94021d"
MAX_SHARDS = 10  # rows [0, 500,000); shards 0065+ are the tail holdout and must not be on disk
TAIL_FIRST_SHARD = 65
STATS_500K = (5.413434769952119, 1.176728647311213)  # runner constant GPTRANS_500K_STATS, selected by the runner when the store holds 500,000 rows
BATCH, LR_MULT, EMA, PASSES = 1024, 8 ** 0.5, 0.999 ** 8, 60
SHAPES = {  # LAYERS, WIDTH, HEADS, FFN_RATIO, parameter count
    "R": ({"LAYERS": 12, "WIDTH": 256, "HEADS": 8, "FFN_RATIO": 1}, 5_871_201),
    "L20": ({"LAYERS": 20, "WIDTH": 256, "HEADS": 8, "FFN_RATIO": 1}, 9_529_761),
    "W320": ({"LAYERS": 12, "WIDTH": 320, "HEADS": 10, "FFN_RATIO": 1}, 8_827_321),
    "F3": ({"LAYERS": 12, "WIDTH": 256, "HEADS": 8, "FFN_RATIO": 3}, 9_023_073),
}
# arm -> init file, init file sha256, init tensor-state sha256 (record: experiments/pcqm_gptrans_width_500k_20261117/init_record.json)
INITS = {
    "R-s42": ("gptrans_init.pt", "27590c1a928cafa8b633998e2291916b8f171d345ed5cb3b0234f0e9a37c3320", "a65034dae2d01d82eb0074eba8f2ee697d93ca259188ae45a4213e9041d1ed1b"),
    "R-s43": ("gptrans_init_s43.pt", "2ccd1eb79de4b82fffb58522b9e2325026d36f04f398be0d49d3eea55ee05f3c", "c95da412764ea1b9a701ae9421c167e32a18f9dc423d364aa3d4fe8c0a8d5f5b"),
    "L20-s42": ("gptrans_init_l20_s42.pt", "8ee2d6aa69168b451285aed872217b6bd7149ed81f7c6ed7c98beca035611b5e", "15aa22a9a36606ef17ef74fb35d88347ed6bf3822fae297b0e260745bca085e9"),
    "L20-s43": ("gptrans_init_l20_s43.pt", "bb375fee040e239eaf37f116143834a23ebc1dab36f04d7d7081c86062ed3fe8", "21b7de3f1fdaa13331a5f05318c3bde888d0ccea5f81aaccb658479358886340"),
    "W320-s42": ("gptrans_init_w320_s42.pt", "a395cd836a9bc3671cbe09a440242e6fe603e98d4642a21a51f306c6f4fbcce2", "fda15782ca558f8d05f5ff3d9fe369238183ffa7eaf610826b70031586b56ef8"),
    "W320-s43": ("gptrans_init_w320_s43.pt", "a119dcc6b8ce7df43c26161e753c1cb5c9a787a716b0e2851d80f4315d4d601d", "5a383a0315f9d93c7467fb0212a4ad487c3131ebc8c9a126825dc6b7ccac6d68"),
    "F3-s42": ("gptrans_init_f3_s42.pt", "f65a446da51a2a98f8bbe2c223690b263c358f284973adcdeae1fa1b30b3ff38", "0f11b9aedbf7eab488bf2163c7838e177840335fa452361ac67b642b212f0cc6"),
    "F3-s43": ("gptrans_init_f3_s43.pt", "3d9756a38d4bb0c5ae0d205fad18c728b9b8e5badc637ced2e0132846d1af0d6", "7139f69633f1dc2d12ceb44329fa4be766d395b48fb88dd8ccac767230e9b1c2"),
}
ARMS = {a: {"shape": SHAPES[a.split("-")[0]][0], "params": SHAPES[a.split("-")[0]][1], "init": i, "run_dir": f"gptrans-b1024-{a.lower()}-500k-20261117"} for a, i in INITS.items()}
RUNNER_REL = "platforms/colab/pcqm_k1_b1024_production/train_full_b1024.py"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def fail(msg: str) -> None:
    print(f"PREFLIGHT FAIL: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(3)


def check_shards(base: Path, train: list[dict]) -> None:
    """The ten training shards must be intact; the tail shards must not exist; together they pin rows [0, 500,000)."""
    if sum(r["rows"] for r in train[:MAX_SHARDS]) != MAX_SHARDS * 50_000 or train[MAX_SHARDS - 1]["source_idx_max"] != 499_999:
        fail("manifest does not give rows [0, 500,000) in its first 10 train shards")
    for r in train[:MAX_SHARDS]:
        p = base / r["file"]
        if not p.is_file() or p.stat().st_size != r["bytes"] or sha256(p) != r["sha256"]:
            fail(f"shard missing or corrupt: {r['file']}")
    for r in train[TAIL_FIRST_SHARD:]:
        if (base / r["file"]).exists():
            fail(f"tail-holdout shard present on disk: {r['file']}")


def atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--wall-s", type=float, required=True, help="seconds left in the PBS job")
    ap.add_argument("--margin-s", type=float, default=1800.0, help="reserve for startup, VRAM load and the final checkpoint")
    ap.add_argument("--min-run-s", type=float, default=3600.0, help="refuse to start with a smaller run budget (local rehearsal lowers it)")
    a = ap.parse_args()
    arm = ARMS[a.arm]
    init_name, init_file_sha, init_state_sha = arm["init"]
    shape = arm["shape"]
    for k, v in shape.items():
        if os.environ.get(f"MOLGAP_GPTRANS_{k}") != str(v):
            fail(f"MOLGAP_GPTRANS_{k}={os.environ.get(f'MOLGAP_GPTRANS_{k}')} does not match arm {a.arm} ({v})")
    root = a.root.resolve()
    code, inputs, data = root / "code", root / "inputs", root / "data"
    runner = code / RUNNER_REL
    run_dir = root / "runs" / arm["run_dir"]

    if sha256(runner) != RUNNER_SHA:
        fail("runner differs from the registered runner")
    if not (inputs / init_name).is_file() or sha256(inputs / init_name) != init_file_sha:
        fail(f"{init_name} is not the registered initialisation of arm {a.arm}")
    manifest = data / "ogb-train-full" / "manifest.json"
    if sha256(manifest) != MANIFEST_SHA:
        fail("full_train_v1 manifest differs from the accepted one")
    train = [r for r in json.loads(manifest.read_text(encoding="utf-8"))["assets"]["topology"] if r["role"] == "train"]
    check_shards(manifest.parent, train)

    import torch
    ver = torch.__version__.split("+")[0]
    if ver not in {"2.7.1", "2.11.0"}:
        fail(f"torch {ver} is not an allowed recipe pin")
    if not torch.cuda.is_available():
        fail("CUDA is not available")
    print(f"preflight ok | torch {torch.__version__} | {torch.cuda.get_device_name(0)} | rows {MAX_SHARDS * 50000}", flush=True)

    from molgap.gptrans_bond_local import gptrans_shape, make_bond_local_model
    if gptrans_shape() != shape:
        fail("model shape differs from the arm")
    params = sum(p.numel() for p in make_bond_local_model().parameters())
    if params != arm["params"]:
        fail(f"model has {params} parameters, arm {a.arm} expects {arm['params']}")

    spec = importlib.util.spec_from_file_location("train_full_b1024", runner)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_full_b1024"] = mod
    spec.loader.exec_module(mod)
    original = mod.make_recipe

    def make_recipe(*args, **kwargs):
        recipe, name = original(*args, **kwargs)
        object.__setattr__(recipe, "torch_version", ver)
        if (recipe.target_mean_eV, recipe.target_std_eV) != STATS_500K:
            fail(f"recipe normalisation {recipe.target_mean_eV, recipe.target_std_eV} is not the 500K statistics")
        return recipe, name

    mod.make_recipe = make_recipe
    out_dir = run_dir / "gptrans_pairnorm"  # the runner's own output folder; arm_record.json sits beside its run_config.json
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {"arm": a.arm, "shape": shape, "params": params, "passes": PASSES, "batch": BATCH, "lr_mult": LR_MULT, "ema_decay": EMA,
              "max_shards": MAX_SHARDS, "train_rows": MAX_SHARDS * 50_000, "total_steps": (MAX_SHARDS * 50_000 // BATCH) * PASSES,
              "init_file": init_name, "init_file_sha256": init_file_sha, "init_state_sha256": init_state_sha, "runner_sha256": RUNNER_SHA}
    atomic_json(out_dir / "arm_record.json", record)
    print("[arm]", json.dumps(record), flush=True)
    max_wall = int(a.wall_s - a.margin_s)
    if max_wall < a.min_run_s:
        fail(f"wall budget {max_wall}s is too small")
    mod.main([
        "--arm", "gptrans_pairnorm", "--code-dir", str(inputs), "--data-root", str(data), "--run-root", str(run_dir),
        "--max-wall-seconds", str(max_wall), "--max-shards", str(MAX_SHARDS),
        "--init-file", init_name, "--init-sha", init_state_sha, "--gptrans-batch-size", str(BATCH), "--gptrans-lr-mult", repr(LR_MULT), "--gptrans-ema-decay", repr(EMA),
    ])


if __name__ == "__main__":
    main()
