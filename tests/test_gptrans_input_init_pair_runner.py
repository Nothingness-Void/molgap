"""The shared T4x2 runner must bind the initialization arm to its source Spec."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


RUNNER = (Path(__file__).resolve().parents[1] / "experiments"
          / "pcqm_gptrans_pair_norm_100k" / "run_candidates.py")


def _runner():
    spec = importlib.util.spec_from_file_location("gptrans_pair_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_input_init_profile_binds_candidate_hash_and_rejects_spec_drift(tmp_path, monkeypatch):
    runner = _runner()
    monkeypatch.setattr(runner, "PROFILE", "reference-input-init-kaggle1-v1")
    assert runner.PROFILES["reference-input-init-kaggle1-v1"][0] == (
        "reference", "input_embedding_normal002")
    candidate_hash = "a" * 64
    declaration = {
        "platform": {"accelerator": "NvidiaTeslaT4", "device_count": 2},
        "arms": [
            {"arm_id": "reference", "scientific_role": "reference"},
            {"arm_id": "input-embedding-normal002", "scientific_role": "candidate",
             "initialization": {"state_sha256": candidate_hash}},
        ],
        "prospective": {"same_run_replay": {
            "reference_arm_id": "reference",
            "candidate_arm_ids": ["input-embedding-normal002"],
        }},
    }
    spec_bytes = json.dumps(declaration, sort_keys=True).encode()
    spec_path = tmp_path / "experiment_spec.json"
    spec_path.write_bytes(spec_bytes)
    package_path = tmp_path / "package_manifest.json"
    package_path.write_text(json.dumps({"spec_sha256": hashlib.sha256(spec_bytes).hexdigest()}))
    monkeypatch.setattr(runner, "find_one", lambda name: tmp_path / name)

    assert runner.input_init_candidate_sha256("reference") is None
    assert runner.input_init_candidate_sha256("input_embedding_normal002") == candidate_hash

    package_path.write_text(json.dumps({"spec_sha256": "0" * 64}))
    with pytest.raises(RuntimeError, match="differs from source package"):
        runner.input_init_candidate_sha256("input_embedding_normal002")
