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
    def configure(seed):
        torch.random.default_generator.manual_seed(seed)
        return {}

    monkeypatch.setattr(owner, "configure_fp32_determinism", configure)
    monkeypatch.setattr(owner, "build_runtime_manifest", lambda determinism:
                        {"runtime_fingerprint": "2" * 64, "torch": "2.11.0+synthetic"})
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

    def step(model, optimizer, batch, mean, std, mode, **execution_options):
        optimizer.zero_grad(set_to_none=True)
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        return None, torch.tensor(0.5), 2

    monkeypatch.setattr(owner, "_screen_step", step)
    idx = torch.tensor([101, 102, 103], dtype=torch.int64)
    target = torch.zeros(3, dtype=torch.float64)

    def evaluate(model, *args, **execution_options):
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
        addon_names = {"mean2": "k1_two_pass_mean", "mean2_clean_second": "k1_mean2_clean_second"}
        addon_source = Path(owner.__file__).with_name("k1_clean_second.py") if mode == "mean2_clean_second" else Path(owner.__file__)
        addons = [] if mode == "reference" else [{"name": addon_names[mode],
            "version": "1", "config": {},
            "source_sha256": normalized_source_sha256(addon_source)}]
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


@pytest.mark.parametrize("mode", ["reference", "mean2", "mean2_clean_second"])
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


def test_retired_local_prefix_fails_before_role_access(lifecycle):
    request, started = lifecycle
    args = request("reference")
    recipe = json.loads(args["recipe_path"].read_text())
    recipe["local_execution_prefix"] = {"epochs": 10, "original_execution": True}
    atomic_json(args["recipe_path"], recipe)
    with pytest.raises(ValueError, match="Retired execution prefix"):
        owner.run_screen_arm(**args)
    assert started == []


@pytest.mark.parametrize("mode", ["reference", "mean2", "mean2_clean_second"])
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


@pytest.mark.parametrize("mode", ["reference", "mean2"])
def test_resume_never_invents_missing_timing(lifecycle, monkeypatch, mode):
    request, started = lifecycle
    args = request(mode)
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


@pytest.mark.parametrize("mode", ["reference", "mean2"])
def test_non_auxiliary_arm_rejects_label_cache(lifecycle, mode):
    request, started = lifecycle
    with pytest.raises(ValueError, match="does not consume chemical auxiliary labels"):
        owner.run_screen_arm(**request(mode), label_cache=object())
    assert started == []


@pytest.mark.parametrize("change", ["missing_selected", "missing_predictions", "wrong_context",
                                   "nonfinite_weights", "wrong_metric", "wrong_rows", "nonfinite_predictions"])
def test_resume_checks_existing_selection_before_any_new_step(lifecycle, monkeypatch, change):
    request, started = lifecycle
    args = request("reference")
    evaluate = owner._evaluate
    calls = 0

    def interrupt(model, *values):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic interruption")
        return evaluate(model, *values)

    monkeypatch.setattr(owner, "_evaluate", interrupt)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        owner.run_screen_arm(**args)
    root = args["output"]
    checkpoint_sha = sha256_file(root / "last_checkpoint.pt")
    trace = (root / "canonical_trace.json").read_bytes()
    if change.startswith("missing_"):
        (root / ("selected_model.pt" if change == "missing_selected" else "development_predictions.pt")).unlink()
    else:
        path = root / ("development_predictions.pt" if change in {"wrong_rows", "nonfinite_predictions"} else "selected_model.pt")
        payload = torch.load(path, weights_only=True)
        if change == "wrong_context":
            payload["context"]["arm_id"] = "foreign"
        elif change == "nonfinite_weights":
            payload["model"]["weight"].fill_(float("nan"))
        elif change == "wrong_metric":
            payload["development_mae_eV"] += 1
        elif change == "wrong_rows":
            payload["source_idx"][0] = 999
        else:
            payload["prediction_eV"][0] = float("nan")
        torch.save(payload, path)
    started.clear()
    monkeypatch.setattr(owner, "_evaluate", evaluate)
    with pytest.raises((ValueError, OSError, RuntimeError)):
        owner.run_screen_arm(**args)
    assert started == []
    assert sha256_file(root / "last_checkpoint.pt") == checkpoint_sha
    assert (root / "canonical_trace.json").read_bytes() == trace


@pytest.mark.parametrize("mode", ["reference", "mean2"])
def test_deferred_metrics_keep_weights_trace_rng_and_scheduler(lifecycle, monkeypatch, mode):
    request, _ = lifecycle
    args = request(mode)
    assert owner.run_screen_arm(**args)["status"] == "MECHANICALLY_VERIFIED"
    deferred = torch.load(args["output"] / "last_checkpoint.pt", weights_only=True)
    deferred_trace = load_canonical_trace(args["output"] / "canonical_trace.json")

    def historical_sum(values):
        total = 0.0
        for value in values:
            total += float(value)
        return total

    monkeypatch.setattr(owner, "sum_scalar_metrics", historical_sum)
    original_args = {**args, "output": args["output"].parent / (mode + "_original_sum")}
    assert owner.run_screen_arm(**original_args)["status"] == "MECHANICALLY_VERIFIED"
    original = torch.load(original_args["output"] / "last_checkpoint.pt", weights_only=True)
    original_trace = load_canonical_trace(original_args["output"] / "canonical_trace.json")
    assert all(torch.equal(value, original["model"][name]) for name, value in deferred["model"].items())
    assert deferred["scheduler"] == original["scheduler"]
    assert torch.equal(deferred["rng_state"]["torch"], original["rng_state"]["torch"])
    assert [r["live_train_metric"] for r in deferred_trace["observations"]] == [
        r["live_train_metric"] for r in original_trace["observations"]]


def test_completion_forwards_binding_and_inspects_only_once(lifecycle, monkeypatch):
    import molgap.experiment_family_workflow as family
    request, _ = lifecycle
    args = request("reference")
    inspect = family.inspect_output
    bindings = []

    def checked_output(*args, **kwargs):
        bindings.append(kwargs.get("target_identity"))
        return inspect(*args, **kwargs)

    monkeypatch.setattr(family, "inspect_output", checked_output)
    # An unverified binding must block, not disappear at the trainer boundary.
    invalid = {"encoding": "unverified"}
    report = owner.run_screen_arm(**args, target_identity=invalid)
    assert report["status"] == "BLOCKED"
    assert bindings == [invalid]
    assert any("explicit pinned target identity" in b for b in report["blockers"])


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1., True])
def test_invalid_development_never_overwrites_acknowledged_checkpoint(lifecycle, monkeypatch, invalid):
    request, _ = lifecycle
    args = request("reference")
    evaluate = owner._evaluate
    calls, previous = 0, {}

    def unhealthy(model, *values):
        nonlocal calls
        calls += 1
        result = evaluate(model, *values)
        if calls == 2:
            previous["checkpoint"] = sha256_file(args["output"] / "last_checkpoint.pt")
            previous["selected"] = sha256_file(args["output"] / "selected_model.pt")
            return (invalid, *result[1:])
        return result

    monkeypatch.setattr(owner, "_evaluate", unhealthy)
    with pytest.raises(RuntimeError, match="preserving last acknowledged checkpoint"):
        owner.run_screen_arm(**args)
    assert sha256_file(args["output"] / "last_checkpoint.pt") == previous["checkpoint"]
    assert sha256_file(args["output"] / "selected_model.pt") == previous["selected"]
    checkpoint = torch.load(args["output"] / "last_checkpoint.pt", weights_only=True)
    assert checkpoint["cursor"]["epoch"] == 1
    assert len(json.loads((args["output"] / "execution_times.json").read_text())) == 1
    assert len(load_canonical_trace(args["output"] / "canonical_trace.json")["observations"]) == 1


@pytest.mark.parametrize("change", ["missing", "model", "optimizer"])
def test_resume_requires_checkpoint_bytes_bound_by_last_trace(lifecycle, monkeypatch, change):
    request, started = lifecycle
    args = request("reference")
    evaluate = owner._evaluate
    calls = 0

    def interrupt(model, *values):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic interruption")
        return evaluate(model, *values)

    monkeypatch.setattr(owner, "_evaluate", interrupt)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        owner.run_screen_arm(**args)
    checkpoint = args["output"] / "last_checkpoint.pt"
    trace_path = args["output"] / "canonical_trace.json"
    before = trace_path.read_bytes()
    if change == "missing":
        checkpoint.unlink()
    else:
        saved = torch.load(checkpoint, weights_only=True)
        if change == "model": saved["model"]["weight"].add_(1.)
        else: saved["optimizer"]["param_groups"][0]["lr"] = 0.1
        torch.save(saved, checkpoint)
    started.clear()
    monkeypatch.setattr(owner, "_evaluate", evaluate)
    with pytest.raises(ValueError, match="hash mismatch|no resume checkpoint"):
        owner.run_screen_arm(**args)
    assert started == []
    assert trace_path.read_bytes() == before


@pytest.mark.parametrize("change", ["epoch", "steps", "samples", "negative", "nan", "scope", "pipeline", "wrapped", "legacy"])
def test_resume_rejects_corrupt_timing_before_any_new_step(lifecycle, monkeypatch, change):
    request, started = lifecycle
    args = request("mean2")
    evaluate = owner._evaluate
    calls = 0

    def interrupt(model, *values):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic interruption")
        return evaluate(model, *values)

    monkeypatch.setattr(owner, "_evaluate", interrupt)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        owner.run_screen_arm(**args)
    path = args["output"] / "execution_times.json"
    timing = json.loads(path.read_text())
    row = timing[0]
    if change == "epoch": row["epoch"] = 2
    elif change == "steps": row["optimizer_steps"] += 1
    elif change == "samples": row["sample_presentations"] = True
    elif change == "negative": row["train_seconds"] = -1
    elif change == "nan": row["development_seconds"] = float("nan")
    elif change == "scope": row["scope"] = ""
    elif change == "pipeline": row["epoch_pipeline_seconds"] = 0
    elif change == "wrapped": timing = {"rows": timing}
    elif change == "legacy": timing = [{"epoch": 1, "timing": None, "reason": "unverified legacy"}]
    atomic_json(path, timing)
    before = path.read_bytes()
    started.clear()
    monkeypatch.setattr(owner, "_evaluate", evaluate)
    with pytest.raises(ValueError, match="[Ee]xecution|[Tt]iming"):
        owner.run_screen_arm(**args)
    assert started == []
    assert path.read_bytes() == before
