"""Host-neutral compatibility retained when accepting the shared core snapshot."""
import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch

from molgap import training_reproducibility as runtime


def test_runtime_manifest_retains_vendored_ogb_observation(monkeypatch):
    monkeypatch.setitem(sys.modules, "ogb", SimpleNamespace(__version__="fixture"))
    monkeypatch.setenv("MOLGAP_OGB_SOURCE_SHA256", "a" * 64)
    monkeypatch.setattr(runtime, "_installed_distributions", lambda: ["fixture==1"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    record = runtime.build_runtime_manifest({"seed": 42})
    assert record["external_runtime_artifacts"]["ogb"] == {
        "version": "fixture", "source_archive_sha256": "a" * 64}
    identity = record.pop("runtime_fingerprint")
    assert runtime.canonical_fingerprint(record) == identity


def test_safe_checkpoint_api_restores_native_numpy_rng_without_unsafe_fallback(tmp_path):
    state = runtime.capture_rng_state()
    path = tmp_path / "state.pt"
    torch.save({"rng_state": state, "model": {"weight": torch.ones(2)}}, path)
    restored = runtime.safe_cpu_torch_load(path)
    assert torch.equal(restored["model"]["weight"], torch.ones(2))
    assert restored["rng_state"]["numpy"][1].tolist() == state["numpy"][1].tolist()


def test_ema_import_order_and_legacy_default_are_independent_of_trainers(tmp_path):
    import ast
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "src/molgap/foreach_ema.py").read_text())
    assert not any(isinstance(node, ast.ImportFrom) and node.module == "pcqm_gptrans_v4"
                   for node in ast.walk(tree))
    for first in ("gptrans_execution_hotpath", "pcqm_gptrans_v4"):
        subprocess.run([sys.executable, "-I", "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from importlib import import_module; import_module('molgap.' + sys.argv[2]); "
            "import torch; from molgap.gptrans_execution_hotpath import BoundForeachEMA; "
            "from molgap.pcqm_gptrans_v4 import ExponentialMovingAverage; "
            "assert BoundForeachEMA is ExponentialMovingAverage; "
            "assert ExponentialMovingAverage(torch.nn.Linear(1, 1)).decay == .9999",
            str(root / "src"), first], cwd=tmp_path, check=True, capture_output=True, text=True)
