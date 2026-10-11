"""Actual K1 loop/output hooks with tiny CPU fixtures, not model qualification.

Only GPU compute and external data/release gates are substituted. Initialization,
recipe checks, optimizer/scheduler/RNG persistence, timing arrays, trace and
complete family output inspection use their real owners. No molecular roles.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from molgap import k1_screen_training as owner
from molgap.experiment_family_workflow import RunContext, tensor_digest
from molgap.research_memory.trace import load_canonical_trace
from molgap.training_reproducibility import atomic_json, sha256_file
from molgap.v4_runtime import normalized_source_sha256, state_dict_sha256


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "Tesla T4 synthetic")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    tensor = torch.tensor
    monkeypatch.setattr(torch, "tensor", lambda *args, **kwargs: tensor(
        *args, **{**kwargs, "device": "cpu"}) if kwargs.get("device") == "cuda"
        else tensor(*args, **kwargs))
    monkeypatch.setattr(owner, "STEPS_PER_EPOCH", 2)
    monkeypatch.setattr(owner, "ROWS_PER_EPOCH", 4)
    monkeypatch.setattr(owner, "SAMPLE_EXPOSURE", 160)
    monkeypatch.setattr(owner, "DEVELOPMENT_ROWS", 3)
    monkeypatch.setattr(owner, "compute_row_order_fingerprint", lambda seed: "1" * 64)
    monkeypatch.setattr(owner, "configure_fp32_determinism", lambda seed: {})
    monkeypatch.setattr(owner, "build_runtime_manifest", lambda determinism:
                        {"runtime_fingerprint": "2" * 64})
    monkeypatch.setattr(owner, "_validate_prospective", lambda *args: None)
    monkeypatch.setattr(owner, "validate_runtime_preflight", lambda *args:
                        {"accelerator": "Tesla T4 synthetic"})
    monkeypatch.setattr(owner, "find_fixed_cache", lambda root: (root, {}))
    monkeypatch.setattr(owner, "load_roles", lambda *args:
                        {"train": "synthetic", "development": "synthetic"})
    monkeypatch.setattr(owner, "_target_stats", lambda *args: (0.0, 1.0))
    monkeypatch.setattr(owner, "_development_loader", lambda *args: ())
    started_epochs = []

    def loader(graphs, epoch, seed):
        started_epochs.append(epoch)
        return (None, None)

    monkeypatch.setattr(owner, "_train_loader", loader)

    def make_model(state, mode, seed):
        model = torch.nn.Linear(1, 1, bias=False)
        model.load_state_dict(state, strict=True)
        return model

    monkeypatch.setattr(owner, "_make_screen_model", make_model)
    # CPU fixtures exercise persistence, not CUDA fused arithmetic or K1 quality.
    monkeypatch.setattr(owner, "_screen_optimizer", lambda model, mode:
                        torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-5))

    def step(model, optimizer, batch, mean, std, mode):
        optimizer.zero_grad(set_to_none=True)
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        return None, 0.5, 2

    monkeypatch.setattr(owner, "_screen_step", step)
    idx = torch.tensor([101, 102, 103], dtype=torch.int64)
    target = torch.zeros(3, dtype=torch.float64)

    def evaluate(model, *args):
        prediction = model.weight.detach().double().reshape(1).repeat(3)
        return float(prediction.abs().mean()), target.clone(), prediction, idx.clone()

    monkeypatch.setattr(owner, "_evaluate", evaluate)
    state = {"weight": torch.ones(1, 1)}
    initial_sha = state_dict_sha256(state)
    monkeypatch.setattr(owner, "INITIAL_STATE_SHA256", initial_sha)
    initial = tmp_path / "initial.pt"
    torch.save({"format": "synthetic", "model_state": state,
                "state_sha256": initial_sha}, initial)

    def request(mode):
        recipe = owner.build_screen_recipe(mode,
            source_idx_sha256=tensor_digest(idx, role="source_idx"),
            target_sha256=tensor_digest(target, role="target"))
        recipe_path = tmp_path / (mode + "_recipe.json")
        atomic_json(recipe_path, recipe)
        context = RunContext("synthetic-k1", "synthetic-run", mode, "a" * 64,
            "b" * 64, "neural_atom_k1", "2", "c" * 40, "d" * 64, "e" * 64,
            sha256_file(recipe_path), "kaggle", "synthetic", "synthetic/kernel", None)
        monkeypatch.setattr(RunContext, "for_training", lambda *args, **kwargs: context)
        addons = [] if mode == "reference" else [{"name": "k1_fused_layout",
            "version": "1", "config": {},
            "source_sha256": normalized_source_sha256(Path(owner.__file__))}]
        arm = {"arm_id": mode, "initialization": {"kind": "frozen_state", "seed": 42,
                "state_sha256": initial_sha},
            "training": {"sampler": {"sha256": recipe["row_order_fingerprint"]},
                         "overrides": {}}, "addons": addons}
        return dict(spec=SimpleNamespace(to_dict=lambda: {"arms": [arm]}),
            package_dir=tmp_path / "package", expected_package_identity="e" * 64,
            arm_id=mode, mode=mode, recipe_path=recipe_path, initial_state_path=initial,
            input_root=tmp_path / "input", output=tmp_path / mode,
            account="synthetic", run_reference="synthetic/kernel",
            trajectory_id="synthetic-" + mode)

    return request, started_epochs


@pytest.mark.parametrize("mode", ["reference", "fused_layout"])
def test_actual_loop_writes_40_epoch_timing_and_verified_terminal(lifecycle, mode):
    request, started = lifecycle
    args = request(mode)
    report = owner.run_screen_arm(**args)
    assert report["status"] == "MECHANICALLY_VERIFIED", report
    root = args["output"]
    timing = json.loads((root / "execution_times.json").read_text())
    assert isinstance(timing, list) and len(timing) == 40
    assert [row["epoch"] for row in timing] == list(range(1, 41))
    assert started == list(range(40))
    assert timing[-1]["optimizer_steps"] == 80
    assert timing[-1]["sample_presentations"] == 160
    assert all(row["train_seconds"] >= 0 for row in timing)
    assert report["observed"]["selected_epoch"] == 40
    assert report["observed"]["progress"]["epochs"] == 40
    assert (root / "output_manifest.json").exists()
    assert not list(root.glob(".*.tmp"))


@pytest.mark.parametrize("mode", ["reference", "fused_layout"])
def test_complete_epoch_resume_keeps_timing_trace_and_schedule(lifecycle, monkeypatch, mode):
    request, started = lifecycle
    args = request(mode)
    evaluate = owner._evaluate
    evaluations = 0

    def interrupt(model, *values):
        nonlocal evaluations
        evaluations += 1
        if evaluations == 3:
            raise RuntimeError("synthetic interruption")
        return evaluate(model, *values)

    monkeypatch.setattr(owner, "_evaluate", interrupt)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        owner.run_screen_arm(**args)
    root = args["output"]
    checkpoint = torch.load(root / "last_checkpoint.pt", weights_only=True)
    assert checkpoint["cursor"]["epoch"] == 2
    assert len(json.loads((root / "execution_times.json").read_text())) == 2
    before_trace = load_canonical_trace(root / "canonical_trace.json")["observations"]
    assert len(before_trace) == 2
    assert not (root / "output_manifest.json").exists()
    started.clear()
    monkeypatch.setattr(owner, "_evaluate", evaluate)
    report = owner.run_screen_arm(**args)
    assert report["status"] == "MECHANICALLY_VERIFIED", report
    assert started == list(range(2, 40))
    timing = json.loads((root / "execution_times.json").read_text())
    assert [row["epoch"] for row in timing] == list(range(1, 41))
    trace = load_canonical_trace(root / "canonical_trace.json")["observations"]
    assert trace[:2] == before_trace
    assert trace[2]["learning_rate"] == checkpoint["optimizer"]["param_groups"][0]["lr"]
    assert trace[-1]["optimizer_step"] == 80
    resumed = torch.load(root / "last_checkpoint.pt", weights_only=True)
    uninterrupted_args = {**args, "output": root.parent / (mode + "_uninterrupted")}
    uninterrupted_report = owner.run_screen_arm(**uninterrupted_args)
    assert uninterrupted_report["status"] == "MECHANICALLY_VERIFIED", uninterrupted_report
    uninterrupted = torch.load(uninterrupted_args["output"] / "last_checkpoint.pt", weights_only=True)
    assert resumed["scheduler"] == uninterrupted["scheduler"]
    assert resumed["optimizer"]["param_groups"] == uninterrupted["optimizer"]["param_groups"]
    assert all(torch.equal(value, uninterrupted["model"][key])
               for key, value in resumed["model"].items())
    costs = json.loads((root / "allocation_cost.json").read_text())["costs"]
    assert costs[1]["status"] == "missing" and costs[1]["value"] is None


def test_resume_never_invents_missing_fused_timing(lifecycle, monkeypatch):
    request, started = lifecycle
    args = request("fused_layout")
    evaluate = owner._evaluate
    evaluations = 0

    def interrupt(model, *values):
        nonlocal evaluations
        evaluations += 1
        if evaluations == 2:
            raise RuntimeError("synthetic interruption")
        return evaluate(model, *values)

    monkeypatch.setattr(owner, "_evaluate", interrupt)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        owner.run_screen_arm(**args)
    (args["output"] / "execution_times.json").unlink()
    started.clear()
    monkeypatch.setattr(owner, "_evaluate", evaluate)
    with pytest.raises(ValueError, match="Execution timing and complete-epoch cursor disagree"):
        owner.run_screen_arm(**args)
    assert started == []
    assert not (args["output"] / "execution_times.json").exists()
