"""Actual full-run wrappers with synthetic CPU kernels and release-gate stubs.

No retained molecular input, checkpoint, protected role or GPU is consumed.
These tests qualify orchestration/persistence only, never a native contract.
"""
import json
from types import SimpleNamespace

import pytest
import torch

from molgap.family_training_core import normalized_gap_update
from molgap.training_reproducibility import atomic_json, canonical_fingerprint


class CPUModel(torch.nn.Linear):
    def to(self, *args, **kwargs):
        return super().to("cpu")


class CPUBatch(SimpleNamespace):
    def to(self, *args, **kwargs):
        return self


@pytest.fixture(params=["k1", "gptrans_t"])
def full_owner(request, monkeypatch, tmp_path):
    if request.param == "k1":
        from molgap import pcqm_k1_full_runner as owner
    else:
        from molgap import pcqm_gptrans_full_runner as owner
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    original_tensor = torch.tensor
    original_zeros = torch.zeros
    monkeypatch.setattr(torch, "tensor", lambda *args, **kwargs:
        original_tensor(*args, **{**kwargs, "device": "cpu"}) if kwargs.get("device") == "cuda"
        else original_tensor(*args, **kwargs))
    monkeypatch.setattr(torch, "zeros", lambda *args, **kwargs:
        original_zeros(*args, **{**kwargs, "device": "cpu"}) if kwargs.get("device") == "cuda"
        else original_zeros(*args, **kwargs))
    monkeypatch.setattr(owner, "configure_fp32_determinism", lambda seed: torch.manual_seed(seed))
    monkeypatch.setattr(owner, "build_runtime_manifest", lambda determinism: {"runtime_fingerprint": "1" * 64})
    monkeypatch.setattr(owner, "validate_source_archive", lambda *args: "2" * 64)
    monkeypatch.setattr(owner, "TRAIN_ROWS", 4)
    monkeypatch.setattr(owner, "PHYSICAL_BATCH", 2)
    monkeypatch.setattr(owner, "BATCHES_PER_PASS", 2)
    monkeypatch.setattr(owner, "CHECKPOINT_EVERY_STEPS", 2)
    manifest = {"aggregates": {"topology": "3" * 64}}
    monkeypatch.setattr(owner, "validate_full_manifest", lambda *args, **kwargs: (manifest, []))
    monkeypatch.setattr(owner, "_load_graphs", lambda paths: ("synthetic", []))
    monkeypatch.setattr(owner, "_target_stats", lambda shards: (0., 1.))
    batches = [CPUBatch(num_graphs=2, x=torch.tensor([[float(i)], [float(i + 1)]]), y=torch.zeros(2))
               for i in (0, 2)]
    monkeypatch.setattr(owner, "_loader", lambda graphs, *, pass_index, start_batch: batches[start_batch:])

    def model_factory():
        model = CPUModel(1, 1)
        with torch.no_grad():
            model.weight.fill_(.5)
            model.bias.fill_(.2)
        return model

    if request.param == "k1":
        monkeypatch.setattr(owner, "MAX_OPTIMIZER_STEPS", 4)
        def make_state():
            model = model_factory()
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4, eta_min=1e-6)
            return model, optimizer, scheduler
        monkeypatch.setattr(owner, "_make_model_and_optimizer", make_state)
        monkeypatch.setattr(owner, "_optimizer_step", lambda model, optimizer, batch, mean, std, *, check_finite:
            normalized_gap_update(model, optimizer, batch, mean, std,
                forward=lambda model, batch: model(batch.x).view(-1),
                clip_norm=1., check_finite=check_finite, cast_target_float=False))
    else:
        monkeypatch.setattr(owner, "TOTAL_STEPS", 4)
        monkeypatch.setattr(owner, "_runtime_gate", lambda: "CPU synthetic")
        monkeypatch.setattr(owner, "_make_model", model_factory)
        monkeypatch.setattr(owner, "_forward", lambda model, batch: model(batch.x).view(-1))
        monkeypatch.setattr(owner, "GPTRANS_SOURCE_SHA256",
            owner._architecture_source_sha256(owner._gptrans_source_path()))

    certificate = {"runtime_fingerprint": "1" * 64, "accelerator": "CPU synthetic"}
    preflight = {"accepted": True, "training_contract_sha256": owner.TRAINING_CONTRACT_SHA256,
        "target_stats": {"mean_eV": 0., "sample_std_eV": 1.}, "runtime_certificate": certificate,
        "runtime_certificate_id": canonical_fingerprint(certificate)}
    if request.param == "gptrans_t":
        preflight["initial_model_sha256"] = owner._state_sha256(model_factory())
        certificate["initial_model_sha256"] = preflight["initial_model_sha256"]
        preflight["runtime_certificate_id"] = canonical_fingerprint(certificate)
    path = tmp_path / "preflight.json"
    atomic_json(path, preflight)
    kwargs = dict(dataset_root=tmp_path, manifest_path=tmp_path / "unused.json",
        preflight_path=path, source_commit="a" * 40, source_archive=tmp_path / "unused.zip",
        resume=False, max_wall_seconds=None)
    return owner, kwargs


def _equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _equal(a, b)
    else:
        assert left == right


def test_full_wrapper_uses_shared_loop_and_preserves_exact_resume(full_owner, tmp_path):
    owner, kwargs = full_owner
    uninterrupted = tmp_path / "uninterrupted"
    owner.train_full(output=uninterrupted, **kwargs)
    report = json.loads((uninterrupted / "run_summary.json").read_text())
    assert report["complete"] is True
    assert report["global_step"] == 4 and report["sample_presentations"] == 8
    original = torch.load(uninterrupted / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    resumed = tmp_path / "resumed"
    paused = owner.train_full(output=resumed, **{**kwargs, "max_wall_seconds": 300})
    assert paused["status"] == "paused" and paused["global_step"] == 1
    owner.train_full(output=resumed, **{**kwargs, "resume": True})
    report = json.loads((resumed / "run_summary.json").read_text())
    assert report["complete"] is True and report["global_step"] == 4
    actual = torch.load(resumed / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    for key in ("model", "optimizer", "scheduler", "ema", "rng_state", "global_step", "pass_index", "next_batch_in_pass"):
        if key in original:
            if key == "rng_state":
                # NumPy's retained RNG array has separate equality semantics.
                assert actual[key]["numpy"][0] == original[key]["numpy"][0]
                assert (actual[key]["numpy"][1] == original[key]["numpy"][1]).all()
                _equal(actual[key]["numpy"][2:], original[key]["numpy"][2:])
                _equal({k: v for k, v in actual[key].items() if k != "numpy"},
                       {k: v for k, v in original[key].items() if k != "numpy"})
            else:
                _equal(actual[key], original[key])
    summary = json.loads((resumed / "run_summary.json").read_text())
    assert all(summary[role] is False for role in (
        "official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"))
