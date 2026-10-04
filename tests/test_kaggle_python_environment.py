from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from molgap import kaggle_python_environment as environment


def test_prepare_python_uses_pinned_uv_environment_and_no_gpu_probe(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    source_root.mkdir()
    calls = []

    probe_output = json.dumps({
        "python": "3.12",
        "executable": "pinned-python",
        "packages": {
            "numpy": "1.26.4",
            "torch": "2.4.1+cu121",
            "torch-geometric": "2.6.1",
            "ogb": "1.3.6",
            "rdkit": "2025.9.5",
        },
        "cuda_build": "12.1",
        "model_constructed": False,
        "model_inference_executed": False,
        "training_executed": False,
        "graph_role_read": False,
    })

    def fake_run(command, *, env, check, timeout, text, capture_output=False):
        calls.append((command, env, capture_output))
        assert check is True
        assert text is True
        assert timeout > 0
        return SimpleNamespace(stdout=probe_output if capture_output else "")

    monkeypatch.setattr(environment.subprocess, "run", fake_run)
    directory = tmp_path / "audit_env"
    executable, qualification = environment.prepare_python(
        directory, deadline=time.time() + 60, source_root=source_root
    )

    bootstrap = directory / "bootstrap"
    assert calls[0][0] == [
        sys.executable, "-m", "pip", "install", "--quiet", "--target",
        str(bootstrap), "uv==0.8.22",
    ]
    assert calls[1][0][:3] == [sys.executable, "-m", "uv"]
    assert calls[1][0][3:6] == ["venv", "--python", "3.12"]
    assert calls[1][0][-1] == str(directory / "venv")
    assert calls[2][0][0:3] == [sys.executable, "-m", "uv"]
    assert calls[2][0][3:6] == ["pip", "install", "--python"]
    assert str(directory / "venv" / "bin" / "python") in calls[2][0]
    assert "torch==2.4.1+cu121" in calls[2][0]
    assert calls[3][0][0] == str(executable)
    assert calls[3][1]["PYTHONPATH"] == str(source_root / "src")
    assert calls[3][1]["CUDA_VISIBLE_DEVICES"] == ""
    assert qualification["status"] == "IMPORTS_QUALIFIED_NOT_GPU_CALIBRATED"
    assert qualification["model_inference_executed"] is False
    assert qualification["training_executed"] is False
