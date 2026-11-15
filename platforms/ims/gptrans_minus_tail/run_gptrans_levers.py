# LEVER-ARM VARIANT of run_gptrans_s44.py (experiments/pcqm_gptrans_full_levers_20261115): arms b256-s42/-s43, c-l20-s42, d-p90-s42; one table row per arm, the runner stays byte-identical.
"""IMS launcher for the GPT-T minus-tail B1024 run (experiments/pcqm_full_minus_tail_fusion_20261113).

Verifies identities and exits before any GPU work if something is off, then runs the registered
Colab runner unchanged. The only runtime adaptation: the runner builds its recipe with the Colab
torch pin (2.11.0); IMS ships torch 2.7.1, which FamilyRecipe also allows, so the recipe's
torch_version is set to the installed build (recorded in sessions.jsonl).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import zipfile
from pathlib import Path

RUNNER_SHA = "120d5571593c56647afe8f7c6f6060afeb7781ff8aea101fb9454fdac0ee5b37"
MANIFEST_SHA = "7d358a679d299bd5de6c5822e42dcef34b0e0b9fa9598c684709509c0b94021d"
MAX_SHARDS = 65  # rows [0, 3,250,000); shards 0065+ are the tail holdout and must not exist here
S42_STATE = "a65034dae2d01d82eb0074eba8f2ee697d93ca259188ae45a4213e9041d1ed1b"
# arm -> init file, init file sha, init tensor-state sha, batch, lr multiplier (sqrt of batch/128), EMA (0.999 ** (batch/128)), layers, passes, run directory
ARMS = {
    "b256-s42": ("gptrans_init.pt", "27590c1a928cafa8b633998e2291916b8f171d345ed5cb3b0234f0e9a37c3320", S42_STATE, 256, 2 ** 0.5, 0.999 ** 2, 12, 60, "gptrans-b256-minus-tail-s42-20261115"),
    "b256-s43": ("gptrans_init_s43.pt", "2ccd1eb79de4b82fffb58522b9e2325026d36f04f398be0d49d3eea55ee05f3c", "c95da412764ea1b9a701ae9421c167e32a18f9dc423d364aa3d4fe8c0a8d5f5b", 256, 2 ** 0.5, 0.999 ** 2, 12, 60, "gptrans-b256-minus-tail-s43-20261115"),
    "c-l20-s42": ("gptrans_init_l20_s42.pt", "8ee2d6aa69168b451285aed872217b6bd7149ed81f7c6ed7c98beca035611b5e", "15aa22a9a36606ef17ef74fb35d88347ed6bf3822fae297b0e260745bca085e9", 1024, 8 ** 0.5, 0.999 ** 8, 20, 60, "gptrans-b1024-l20-minus-tail-s42-20261115"),
    "d-p90-s42": ("gptrans_init.pt", "27590c1a928cafa8b633998e2291916b8f171d345ed5cb3b0234f0e9a37c3320", S42_STATE, 1024, 8 ** 0.5, 0.999 ** 8, 12, 90, "gptrans-b1024-p90-minus-tail-s42-20261115"),
}
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


def stage_shards(root: Path, wanted: list[dict]) -> None:
    """Unpack the uploaded zips (shards 0000-0064 only) once and verify every shard against the manifest sha256."""
    base = root / "data" / "ogb-train-full"
    marker = base / ".verified_65_shards"
    if marker.is_file() and marker.read_text().strip() == MANIFEST_SHA:
        for r in wanted:
            p = base / r["file"]
            if not p.is_file() or p.stat().st_size != r["bytes"]:
                fail(f"verified shard missing or resized: {r['file']}")
        return
    names = {r["file"] for r in wanted}
    for z in sorted((root / "upload").glob("train_topology_*.zip")):
        with zipfile.ZipFile(z) as zf:
            for m in zf.namelist():
                if m not in names:
                    fail(f"{z.name} holds {m}, which is outside shards 0000-0064")
                if not (base / m).is_file():
                    zf.extract(m, base)
    for r in wanted:
        p = base / r["file"]
        if not p.is_file() or p.stat().st_size != r["bytes"] or sha256(p) != r["sha256"]:
            fail(f"shard missing or corrupt: {r['file']}")
    marker.write_text(MANIFEST_SHA)
    print(f"staged and verified {len(wanted)} shards", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--wall-s", type=float, required=True, help="PBS walltime in seconds")
    ap.add_argument("--margin-s", type=float, default=1800.0, help="reserve for startup, VRAM load and the final checkpoint")
    a = ap.parse_args()
    INIT_NAME, INIT_FILE_SHA, INIT_STATE_SHA, BATCH, LR_MULT, EMA, LAYERS, PASSES, RUN_DIR = ARMS[a.arm]
    if str(LAYERS) != os.environ.get("MOLGAP_GPTRANS_LAYERS", "12"):
        fail(f"MOLGAP_GPTRANS_LAYERS={os.environ.get('MOLGAP_GPTRANS_LAYERS')} does not match arm layers={LAYERS}")
    root = a.root.resolve()
    code, inputs, data = root / "code", root / "inputs", root / "data"
    runner = code / RUNNER_REL
    run_root = root / "runs" / RUN_DIR

    if sha256(runner) != RUNNER_SHA:
        fail("runner differs from the registered minus-tail runner")
    if sha256(inputs / INIT_NAME) != INIT_FILE_SHA:
        fail(f"{INIT_NAME} is not the registered initialisation of arm {a.arm}")
    manifest = data / "ogb-train-full" / "manifest.json"
    if sha256(manifest) != MANIFEST_SHA:
        fail("full_train_v1 manifest differs from the accepted one")
    train = [r for r in json.loads(manifest.read_text(encoding="utf-8"))["assets"]["topology"] if r["role"] == "train"]
    stage_shards(root, train[:MAX_SHARDS])
    for r in train[MAX_SHARDS:]:
        if (data / "ogb-train-full" / r["file"]).exists():
            fail(f"tail-holdout shard present on disk: {r['file']}")

    import torch
    ver = torch.__version__.split("+")[0]
    if ver not in {"2.7.1", "2.11.0"}:
        fail(f"torch {ver} is not an allowed recipe pin")
    if not torch.cuda.is_available():
        fail("CUDA is not available")
    print(f"preflight ok | torch {torch.__version__} | {torch.cuda.get_device_name(0)} | rows {MAX_SHARDS * 50000}", flush=True)

    spec = importlib.util.spec_from_file_location("train_full_b1024", runner)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_full_b1024"] = mod
    spec.loader.exec_module(mod)
    original = mod.make_recipe

    def make_recipe(*args, **kwargs):
        recipe, init_name = original(*args, **kwargs)
        object.__setattr__(recipe, "torch_version", ver)
        return recipe, init_name

    mod.make_recipe = make_recipe
    mod.TOTAL_EPOCHS = PASSES  # read at call time by make_recipe and run_arm; warmup stays 4 passes, cosine spans PASSES
    from molgap.gptrans_bond_local import gptrans_layers
    if gptrans_layers() != LAYERS:
        fail("model depth differs from the arm")
    run_root.mkdir(parents=True, exist_ok=True)
    record = {"arm": a.arm, "layers": LAYERS, "passes": PASSES, "batch": BATCH, "lr_mult": LR_MULT, "ema_decay": EMA,
              "init_file": INIT_NAME, "init_file_sha256": INIT_FILE_SHA, "init_state_sha256": INIT_STATE_SHA,
              "total_steps": (MAX_SHARDS * 50000 // BATCH) * PASSES, "runner_sha256": RUNNER_SHA}
    (run_root / "arm_record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("[arm]", json.dumps(record), flush=True)
    max_wall = int(a.wall_s - a.margin_s)
    if max_wall < 3600:
        fail(f"wall budget {max_wall}s is too small")
    mod.main([
        "--arm", "gptrans_pairnorm", "--code-dir", str(inputs), "--data-root", str(data), "--run-root", str(run_root),
        "--max-wall-seconds", str(max_wall), "--max-shards", str(MAX_SHARDS),
        "--init-file", INIT_NAME, "--init-sha", INIT_STATE_SHA, "--gptrans-batch-size", str(BATCH), "--gptrans-lr-mult", repr(LR_MULT), "--gptrans-ema-decay", repr(EMA),
    ])


if __name__ == "__main__":
    main()
