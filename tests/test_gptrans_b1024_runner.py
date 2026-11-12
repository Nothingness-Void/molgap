"""train_full_b1024.py: defaults keep the pre-change recipes; a real X GPT-T run with the new knobs."""
import dataclasses
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "platforms/colab/pcqm_k1_b1024_production/train_full_b1024.py"
SRC = str(Path(__file__).resolve().parents[1] / "src")


def _load(text, name):
    path = Path(__file__).with_name(f"_{name}.py")
    path.write_text(text, encoding="utf-8")
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, SRC)
        spec.loader.exec_module(module)
        return module
    finally:
        path.unlink()


def test_default_recipes_equal_pre_change():
    root = SCRIPT.parents[3]
    old_text = subprocess.run(["git", "show", "HEAD:" + SCRIPT.relative_to(root).as_posix()], cwd=root, check=True,
                              capture_output=True, text=True, encoding="utf-8").stdout
    old, new = _load(old_text, "gt_old"), _load(SCRIPT.read_text(encoding="utf-8"), "gt_new")
    for arm, kwargs in (("k1_muon", {}), ("k1_muon", {"batch_size": 128, "lr_mult": 1.0}), ("gptrans_pairnorm", {"batch_size": 128, "lr_mult": 1.0})):
        a, init_a = old.make_recipe(arm, total_rows=500_000, **kwargs)
        b, init_b = new.make_recipe(arm, total_rows=500_000, **kwargs)
        assert init_a == init_b and a.identity == b.identity
        assert dataclasses.asdict(a) == dataclasses.asdict(b), arm


@pytest.mark.skipif(not (os.environ.get("MOLGAP_B1024_SMOKE_DATA") and os.environ.get("MOLGAP_B1024_SMOKE_CODE")),
                    reason="needs a CUDA box with full_train_v1 shards (MOLGAP_B1024_SMOKE_DATA) and the code dir with gptrans_init*.pt (MOLGAP_B1024_SMOKE_CODE)")
def test_gptrans_one_shard_short_flow(tmp_path):
    code = Path(os.environ["MOLGAP_B1024_SMOKE_CODE"])
    driver = (  # shrink to 5 passes (warmup 4 needs a cosine tail) and accept the local torch build instead of the recipe's pin
        "import runpy,sys,torch; m = runpy.run_path(sys.argv[1], run_name='gt'); g = m['main'].__globals__; g['TOTAL_EPOCHS'] = 5\n"
        "make = g['make_recipe']\n"
        "def patched(*a, **k):\n    r, i = make(*a, **k); object.__setattr__(r, 'torch_version', torch.__version__.split('+')[0]); return r, i\n"
        "g['make_recipe'] = patched; m['main'](sys.argv[2:])")
    sha43 = os.environ.get("MOLGAP_B1024_S43_SHA")  # tensor-state sha of gptrans_init_s43.pt; unset -> seed-42 init
    args = ["--arm", "gptrans_pairnorm", "--code-dir", str(code), "--data-root", os.environ["MOLGAP_B1024_SMOKE_DATA"], "--run-root", str(tmp_path),
            "--max-shards", "1", "--gptrans-batch-size", "256", "--gptrans-lr-mult", repr((256 / 128) ** 0.5), "--gptrans-ema-decay", repr(0.999 ** 2)]
    if sha43:
        args += ["--init-file", "gptrans_init_s43.pt", "--init-sha", sha43]
    proc = subprocess.run([sys.executable, "-c", driver, str(SCRIPT), *args], env=dict(os.environ, PYTHONPATH=SRC), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    out = tmp_path / "gptrans_pairnorm"
    cfg = json.loads((out / "run_config.json").read_text())
    assert cfg["batch_size"] == 256 and cfg["max_shards"] == 1 and cfg["train_rows"] == 50_000 and cfg["ema_decay"] == 0.999 ** 2
    assert (out / "training_complete.json").is_file() and (out / "ema_model.pt").is_file()
    assert len((out / "train_log.jsonl").read_text().splitlines()) == 5
