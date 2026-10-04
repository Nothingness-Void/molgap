"""Isolate a pinned workload from Kaggle's moving system Python image."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

UV_VERSION = "0.8.22"
PYTHON_VERSION = "3.12"
REQUIREMENTS = (
    "numpy==1.26.4", "torch==2.4.1+cu121", "torch-geometric==2.6.1",
    "ogb==1.3.6", "rdkit==2025.9.5",
)


def prepare_python(directory: Path, *, deadline: float, source_root: Path) -> tuple[Path, dict]:
    """Install wheels in a new environment; qualify imports, never run a model."""
    directory = directory.resolve()
    if directory.exists():
        raise ValueError("Never reuse an unqualified environment directory")
    directory.mkdir(parents=True)
    bootstrap = directory / "bootstrap"
    environment = dict(os.environ, PYTHONPATH=str(bootstrap),
                       UV_PYTHON_INSTALL_DIR=str(directory / "python"),
                       UV_CACHE_DIR=str(directory / "cache"))

    def run(command, *, capture=False, env=environment):
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError("Environment qualification deadline")
        return subprocess.run(command, env=env, check=True, timeout=remaining,
                              text=True, capture_output=capture)

    run([sys.executable, "-m", "pip", "install", "--quiet", "--target", str(bootstrap),
         "uv==" + UV_VERSION])
    uv = [sys.executable, "-m", "uv"]
    run(uv + ["venv", "--python", PYTHON_VERSION, "--managed-python", str(directory / "venv")])
    executable = directory / "venv" / "bin" / "python"
    run(uv + ["pip", "install", "--python", str(executable), "--index-strategy", "unsafe-best-match",
              "--extra-index-url", "https://download.pytorch.org/whl/cu121", *REQUIREMENTS])
    # An isolated process tests binary/import compatibility without CUDA allocation,
    # deserialization, model construction, graph access or inference.
    probe = """import sys,json,importlib.metadata as m
import numpy,torch,torch_geometric,ogb,rdkit
import molgap.gptrans,molgap.pcqm_wedge,molgap.gptrans_portability
assert sys.version_info[:2] == (3,12)
assert numpy.__version__ == '1.26.4'
assert torch.__version__ == '2.4.1+cu121' and torch.version.cuda == '12.1'
assert torch_geometric.__version__ == '2.6.1'
assert m.version('ogb') == '1.3.6' and m.version('rdkit') == '2025.9.5'
print(json.dumps(dict(python=sys.version,executable=sys.executable,
    packages={n:m.version(n) for n in ('numpy','torch','torch-geometric','ogb','rdkit')},
    cuda_build=torch.version.cuda,model_constructed=False,model_inference_executed=False,
    training_executed=False,graph_role_read=False)))
"""
    probe_environment = dict(os.environ, PYTHONPATH=str(source_root / "src"), CUDA_VISIBLE_DEVICES="")
    result = json.loads(run([str(executable), "-c", probe], capture=True, env=probe_environment).stdout)
    result.update(bootstrap_uv=UV_VERSION, requested_python=PYTHON_VERSION,
                  requirements=list(REQUIREMENTS), system_python=sys.version,
                  status="IMPORTS_QUALIFIED_NOT_GPU_CALIBRATED")
    return executable, result
