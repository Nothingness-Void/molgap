"""CPU-only contracts for the retained GPTrans-T execution observer."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Data

from molgap.family_execution_profile import ProfileConfig
from molgap import gptrans_legacy_execution_profile as legacy_profile
from molgap.gptrans_legacy_execution_profile import (
    LegacyGPTransExecutionProfiler,
    gptrans_state_fingerprint,
    gptrans_state_snapshot,
    verify_profile_state_proof,
)
from molgap import gptrans_screen_workflow as workflow


class EMA:
    def __init__(self, model):
        self.state = {name: value.detach().clone() for name, value in model.state_dict().items()}

    def update(self, model):
        self.state = {name: value.detach().clone() for name, value in model.state_dict().items()}

    def state_dict(self):
        return self.state


def fixture():
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(1, 2), torch.nn.ReLU(), torch.nn.Linear(2, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=.9)
    ema = EMA(model)
    owner = SimpleNamespace()
    owner.SEED = 42
    owner._forward = lambda model, batch: model(batch.x)
    owner._evaluate = lambda *args, **kwargs: None
    owner._save_checkpoint = lambda *args, **kwargs: None

    def make_state(*args, **kwargs):
        return model, optimizer, scheduler, ema

    owner._make_training_state = make_state

    def step(model, optimizer, ema, batch, mean, std, *, check_finite):
        optimizer.zero_grad(set_to_none=True)
        loss = owner._forward(model, batch).sub(batch.y).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        ema.update(model)
        return loss.detach()

    owner._optimizer_step = step
    batch = Data(x=torch.ones(4, 1), y=torch.zeros(4, 1))
    owner._training_loader = lambda graphs, epoch: [batch.clone(), batch.clone(), batch.clone()]
    return owner, model, optimizer, scheduler, ema


def native_proof_state():
    """Build a small CPU fixture with the native proof's required state shape."""
    owner, model, optimizer, scheduler, ema = fixture()
    scheduler.step(0)
    batch = Data(x=torch.ones(4, 1), y=torch.zeros(4, 1))
    for _ in range(2):
        owner._optimizer_step(model, optimizer, ema, batch, torch.tensor(0.), torch.tensor(1.),
                              check_finite=True)
    state = gptrans_state_snapshot(model, optimizer, scheduler, ema)
    # The retained V4 scheduler is FrozenEpochScheduler; the CPU fixture uses
    # StepLR only to produce optimizer moments and a serializable model.
    state["scheduler"] = {"epoch": 0}
    # A native proof is captured on one isolated CUDA device.  The validator
    # only needs the portable bytes and deliberately does not claim CUDA here.
    state["rng_state"]["cuda"] = [torch.zeros(16, dtype=torch.uint8)]
    return state


def write_state_proof(tmp_path, recipe, plain, profiled):
    from molgap.family_training_runner import normalized_file_sha
    canonical = __import__("molgap.screen_policy", fromlist=["canonical_fingerprint"]).canonical_fingerprint
    records = []
    for name, state in (("plain.pt", plain), ("profiled.pt", profiled)):
        path = tmp_path / name
        torch.save(state, path)
        records.append({"path": name, "sha256": normalized_file_sha(path),
                       "fingerprint": gptrans_state_fingerprint(state)})
    return {"format": "molgap-gptrans-profile-state-proof-v1", "status": "accepted",
            "recipe_identity": canonical(recipe), "updates": 2,
            "plain": records[0], "profiled": records[1], "equal": True}


def run(owner, model, optimizer, scheduler, ema, output, *, profile):
    config = ProfileConfig(warmup_steps=0, active_steps=2)
    context = LegacyGPTransExecutionProfiler(owner, {}, output, config, model=model)
    observed = None
    if profile:
        with context as observed:
            observed._observe_scheduler(scheduler)
            scheduler.step()
            for batch in owner._training_loader(None, 0):
                batch = batch.to("cpu", non_blocking=True)
                owner._optimizer_step(model, optimizer, ema, batch, torch.tensor(0.), torch.tensor(1.),
                                      check_finite=True)
    else:
        scheduler.step()
        for batch in owner._training_loader(None, 0):
            owner._optimizer_step(model, optimizer, ema, batch, torch.tensor(0.), torch.tensor(1.),
                                  check_finite=True)
    return observed


def test_gptrans_observer_preserves_state_and_restores_owner(tmp_path):
    owner, model, optimizer, scheduler, ema = fixture()
    run(owner, model, optimizer, scheduler, ema, tmp_path / "plain", profile=False)
    expected = gptrans_state_fingerprint(gptrans_state_snapshot(model, optimizer, scheduler, ema))

    owner, model, optimizer, scheduler, ema = fixture()
    originals = dict(vars(owner))
    observed = run(owner, model, optimizer, scheduler, ema, tmp_path / "profile", profile=True)
    actual = gptrans_state_fingerprint(gptrans_state_snapshot(model, optimizer, scheduler, ema))

    assert actual == expected
    assert dict(vars(owner)) == originals
    assert observed.report["status"] == "COMPLETE"
    assert observed.report["captured_updates"] == 2
    assert observed.report["observed_updates"] == 3
    phases = {row["name"] for row in observed.report["phases_by_host_self_time"]}
    assert {"forward", "transfer", "optimizer"} <= phases


def test_gptrans_profile_state_proof_is_hash_and_fingerprint_bound(tmp_path):
    state = native_proof_state()
    recipe = {"family": "gptrans_t", "mode": "reference", "seed": 42}
    proof = write_state_proof(tmp_path, recipe, state, state)
    assert verify_profile_state_proof(tmp_path, recipe, proof, required=True) == proof
    altered = dict(proof, equal=False)
    with pytest.raises(ValueError, match="binding"):
        verify_profile_state_proof(tmp_path, recipe, altered, required=True)


def test_native_profile_state_proof_rejects_rehashed_empty_state(tmp_path):
    recipe = {"family": "gptrans_t", "mode": "reference", "seed": 42}
    valid = native_proof_state()
    empty = dict(valid, model={})
    proof = write_state_proof(tmp_path, recipe, empty, valid)
    with pytest.raises(ValueError, match="model state is empty"):
        verify_profile_state_proof(tmp_path, recipe, proof, required=True)


def test_native_profile_state_proof_rejects_rehashed_incomplete_optimizer(tmp_path):
    recipe = {"family": "gptrans_t", "mode": "reference", "seed": 42}
    valid = native_proof_state()
    optimizer = dict(valid["optimizer"], state={})
    incomplete = dict(valid, optimizer=optimizer)
    proof = write_state_proof(tmp_path, recipe, incomplete, valid)
    with pytest.raises(ValueError, match="optimizer state is empty"):
        verify_profile_state_proof(tmp_path, recipe, proof, required=True)


def test_gptrans_observer_rejects_unknown_module_before_owner_mutation(tmp_path):
    owner, model, optimizer, scheduler, ema = fixture()
    config = ProfileConfig(warmup_steps=0, active_steps=1, module_paths=("missing",))
    with pytest.raises(AttributeError):
        with LegacyGPTransExecutionProfiler(owner, {}, tmp_path / "profile", config, model=model):
            pass
    assert owner._optimizer_step is not None


def test_gptrans_owner_kwargs_separate_preflight_calibration_from_training(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    from molgap.research_memory.trace import file_digest
    monkeypatch.setattr(workflow.owner, "MANIFEST_SHA256", file_digest(manifest))

    def preflight(*, dataset_root, manifest_path, source_archive, source_archive_sha256,
                  source_commit, output, platform_id, initial_state_path, variant,
                  runtime_calibration_fingerprint):
        return None

    monkeypatch.setattr(workflow.owner, "run_preflight", preflight)
    context = SimpleNamespace(
        source_archive_sha256="a" * 64, source_commit="b" * 40,
        platform="kaggle", to_dict=lambda: {"context": "synthetic"},
    )
    recipe = {"runtime_calibration_fingerprint": "c" * 64}
    common = dict(context=context, package_dir=tmp_path, mode="reference", recipe=recipe,
        initial_state_path=tmp_path / "initial.pt", input_root=tmp_path, output=tmp_path / "out",
        preflight_path=tmp_path / "preflight.json")
    preflight_kwargs = workflow._owner_kwargs(**common, target="preflight")
    training_kwargs = workflow._owner_kwargs(**common, target="training")
    assert preflight_kwargs["runtime_calibration_fingerprint"] == "c" * 64
    assert "runtime_calibration_fingerprint" not in training_kwargs
    assert training_kwargs["preflight_path"] == common["preflight_path"]


def test_gptrans_owner_kwargs_rejects_calibration_for_unextended_native_preflight(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    from molgap.research_memory.trace import file_digest
    monkeypatch.setattr(workflow.owner, "MANIFEST_SHA256", file_digest(manifest))
    context = SimpleNamespace(
        source_archive_sha256="a" * 64, source_commit="b" * 40,
        platform="kaggle", to_dict=lambda: {"context": "synthetic"},
    )
    with pytest.raises(TypeError, match="does not accept runtime calibration"):
        workflow._owner_kwargs(context=context, package_dir=tmp_path, mode="reference",
            recipe={"runtime_calibration_fingerprint": "c" * 64},
            initial_state_path=tmp_path / "initial.pt", input_root=tmp_path,
            output=tmp_path / "out", target="preflight")


def test_qualification_consumes_two_batches_in_each_branch_and_closes_iterators(tmp_path, monkeypatch):
    import json
    import molgap.gptrans_legacy_execution_profile as module

    torch.manual_seed(42)
    calls = []
    loaders = []

    class Iterator:
        def __init__(self):
            self.index = 0
            self.closed = False

        def __iter__(self):
            return self

        def __next__(self):
            if self.index == 3:
                raise StopIteration
            self.index += 1
            return Data(x=torch.ones(4, 1), y=torch.zeros(4, 1))

        def close(self):
            self.closed = True

    class Loader:
        def __iter__(self):
            iterator = Iterator()
            loaders.append(iterator)
            return iterator

    class FakeProfiler:
        def __init__(self, owner, recipe, output, config, *, model=None):
            self.output = Path(output)
            self.report = {"status": "COMPLETE", "captured_updates": 2,
                           "device_events_observed": True}

        def __enter__(self):
            self.output.mkdir(parents=True, exist_ok=False)
            (self.output / "profile.json").write_text(json.dumps(self.report), encoding="utf-8")
            return self

        def __exit__(self, *_):
            return False

        def _observe_scheduler(self, scheduler):
            return None

    class Owner:
        SEED = 42

        @staticmethod
        def _run_target_stats(shards, mode, target_transform_path):
            return 0., 1.

        @staticmethod
        def _make_training_state(initial_state_path, mode):
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, .9)
            return model, optimizer, scheduler, EMA(model)

        @staticmethod
        def _training_loader(graphs, epoch):
            return Loader()

        @staticmethod
        def _optimizer_step(model, optimizer, ema, batch, mean, std, *, check_finite):
            calls.append(1)
            optimizer.zero_grad(set_to_none=True)
            loss = model(batch.x).sub(batch.y).square().mean()
            loss.backward()
            optimizer.step()
            ema.update(model)
            return loss.detach()

    native_tensor = torch.tensor
    native_to = Data.to

    def cpu_tensor(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs = {**kwargs, "device": "cpu"}
        return native_tensor(*args, **kwargs)

    def cpu_to(self, device, *args, **kwargs):
        return native_to(self, "cpu" if device == "cuda" else device, *args, **kwargs)

    monkeypatch.setattr(torch, "tensor", cpu_tensor)
    monkeypatch.setattr(Data, "to", cpu_to)
    monkeypatch.setattr(module, "LegacyGPTransExecutionProfiler", FakeProfiler)
    recipe = {"mode": "reference",
              "profiling": {"warmup_steps": 120, "active_steps": 16,
                            "record_shapes": False, "profile_memory": False,
                            "with_stack": False, "row_limit": 80, "module_paths": []}}
    profile_proof, state_proof = module.qualify_profile_state(
        owner=Owner(), recipe=recipe, train_graphs=object(), train_shards=object(),
        initial_state_path=tmp_path / "initial.pt", output=tmp_path)
    assert len(calls) == 4
    assert len(loaders) == 2 and all(iterator.closed for iterator in loaders)
    assert profile_proof["report"] == "profile_qualification/profile.json"
    assert state_proof["equal"] is True

