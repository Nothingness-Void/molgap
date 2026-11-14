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
import sys
import zipfile
from pathlib import Path

RUNNER_SHA = "120d5571593c56647afe8f7c6f6060afeb7781ff8aea101fb9454fdac0ee5b37"
INIT_FILE_SHA = "27590c1a928cafa8b633998e2291916b8f171d345ed5cb3b0234f0e9a37c3320"
MANIFEST_SHA = "7d358a679d299bd5de6c5822e42dcef34b0e0b9fa9598c684709509c0b94021d"
MAX_SHARDS = 65  # rows [0, 3,250,000); shards 0065+ are the tail holdout and must not exist here
BATCH, LR_MULT, EMA = 1024, (1024 / 128) ** 0.5, 0.999 ** (1024 // 128)
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
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--wall-s", type=float, required=True, help="PBS walltime in seconds")
    ap.add_argument("--margin-s", type=float, default=1800.0, help="reserve for startup, VRAM load and the final checkpoint")
    a = ap.parse_args()
    root = a.root.resolve()
    code, inputs, data = root / "code", root / "inputs", root / "data"
    runner = code / RUNNER_REL
    run_root = root / "runs" / "gptrans-b1024-minus-tail-20261114"

    if sha256(runner) != RUNNER_SHA:
        fail("runner differs from the registered minus-tail runner")
    if sha256(inputs / "gptrans_init.pt") != INIT_FILE_SHA:
        fail("gptrans_init.pt is not the seed-42 initialisation")
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
    max_wall = int(a.wall_s - a.margin_s)
    if max_wall < 3600:
        fail(f"wall budget {max_wall}s is too small")
    mod.main([
        "--arm", "gptrans_pairnorm", "--code-dir", str(inputs), "--data-root", str(data), "--run-root", str(run_root),
        "--max-wall-seconds", str(max_wall), "--max-shards", str(MAX_SHARDS),
        "--gptrans-batch-size", str(BATCH), "--gptrans-lr-mult", repr(LR_MULT), "--gptrans-ema-decay", repr(EMA),
    ])


if __name__ == "__main__":
    main()
