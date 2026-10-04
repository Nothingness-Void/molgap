from __future__ import annotations

import builtins
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from molgap import kaggle_python_environment as environment


RUNNER = Path(__file__).resolve().parents[1] / "experiments/pcqm_gptrans_ema_portability/run.py"


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


def _load_runner_stdlib_only(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "molgap" or name.startswith("molgap."):
            raise AssertionError(f"entry imported project module during load: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    spec = importlib.util.spec_from_file_location("gptrans_portability_entry", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_entry_loads_with_stdlib_only_and_rejects_conflicting_modes(monkeypatch):
    module = _load_runner_stdlib_only(monkeypatch)
    monkeypatch.setattr(sys, "argv", [str(RUNNER), "--arm", "ema999", "--environment-only"])
    monkeypatch.setattr(module, "one", lambda *_: pytest.fail("input access before argparse rejection"))
    monkeypatch.setattr(module, "source", lambda *_: pytest.fail("source access before argparse rejection"))

    with pytest.raises(SystemExit) as error:
        module.main()

    assert error.value.code == 2


def _environment_records(tmp_path, *, source_identity="release-id", model_inference=False):
    from molgap.gptrans_portability_records import accept_environment

    output = tmp_path / "environment"
    output.mkdir()
    requirements = {item.split("==")[0]: item.split("==")[1] for item in environment.REQUIREMENTS}
    (output / "environment_qualification.json").write_text(json.dumps({
        "source_identity": source_identity,
        "status": "IMPORTS_QUALIFIED_NOT_GPU_CALIBRATED",
        "requirements": list(environment.REQUIREMENTS),
        "bootstrap_uv": environment.UV_VERSION,
        "requested_python": environment.PYTHON_VERSION,
        "cuda_build": "12.1",
        "python": "3.12.4 (main)",
        "packages": requirements,
        "model_constructed": False,
        "model_inference_executed": model_inference,
        "training_executed": False,
        "graph_role_read": False,
    }))
    (output / "environment_cost.json").write_text(json.dumps({
        "status": "COMPLETE",
        "wall_seconds": 10,
        "allocated_devices": 0,
        "device_hours": 0,
        "training_executed": False,
        "model_inference_executed": False,
        "graph_role_read": False,
    }))
    return output, accept_environment, {"source_release": {"identity": "release-id"}}


def test_accept_environment_accepts_exact_pins_and_source(tmp_path):
    output, accept_environment, binding = _environment_records(tmp_path)

    result = accept_environment(output, binding)

    assert result["accepted"] is True
    assert result["model_inference_executed"] is False


@pytest.mark.parametrize("mutation", ["source", "model_inference"])
def test_accept_environment_rejects_changed_source_or_inference(tmp_path, mutation):
    source_identity = "changed" if mutation == "source" else "release-id"
    output, accept_environment, binding = _environment_records(
        tmp_path, source_identity=source_identity, model_inference=mutation == "model_inference"
    )

    with pytest.raises(ValueError):
        accept_environment(output, binding)
