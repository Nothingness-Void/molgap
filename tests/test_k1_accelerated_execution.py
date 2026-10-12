"""Execution-policy acceptance with scratch CPU graphs; no molecular roles."""
import copy
import json
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch, Data

from molgap import k1_screen_training as owner
from molgap.experiment_execution import build_family_recipe, training_adapter
from molgap.experiment_spec import ExperimentSpec
from molgap.training_reproducibility import (
    configure_fp32_determinism, deterministic_memory_fill_context,
)
from molgap.v4_runtime import normalized_source_sha256


@pytest.fixture
def deterministic_cpu():
    settings = (torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.utils.deterministic.fill_uninitialized_memory, torch.get_num_threads(),
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
        torch.get_rng_state().clone())
    configure_fp32_determinism(42)
    torch.set_num_threads(1)
    try:
        yield
    finally:
        enabled, warn, fill, threads, matmul, cudnn, benchmark, deterministic, rng = settings
        torch.use_deterministic_algorithms(enabled, warn_only=warn)
        torch.utils.deterministic.fill_uninitialized_memory = fill
        torch.set_num_threads(threads)
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = deterministic
        torch.set_rng_state(rng)


def fixture_batch(*, empty_edges=False):
    graphs = []
    for n in (1, 3, 7):
        edges = torch.stack((torch.arange(n - 1), torch.arange(1, n)))
        edges = torch.cat((edges, edges.flip(0)), dim=1)
        if empty_edges:
            edges = torch.empty(2, 0, dtype=torch.long)
        graphs.append(Data(x=torch.zeros(n, 9, dtype=torch.long), edge_index=edges,
            edge_attr=torch.zeros(edges.shape[1], 3, dtype=torch.long),
            random_walk_pe=torch.rand(n, 16), y=torch.randn(1)))
    return Batch.from_data_list(graphs)


def test_memory_fill_scope_restores_nested_and_exception_state(deterministic_cpu):
    before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="scratch"):
        with deterministic_memory_fill_context(enabled=False):
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.utils.deterministic.fill_uninitialized_memory
            assert not torch.backends.cuda.matmul.allow_tf32
            with deterministic_memory_fill_context(enabled=True):
                assert torch.utils.deterministic.fill_uninitialized_memory
            assert not torch.utils.deterministic.fill_uninitialized_memory
            raise RuntimeError("scratch")
    assert torch.utils.deterministic.fill_uninitialized_memory
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("enabled", [0, 1, "false", None])
def test_memory_fill_rejects_non_boolean(enabled):
    with pytest.raises(ValueError, match="exact bool"):
        with deterministic_memory_fill_context(enabled=enabled):
            pytest.fail("Invalid policy entered")


@pytest.mark.parametrize("enabled,warn", [(False, False), (True, True)])
def test_memory_fill_requires_fail_closed_determinism(deterministic_cpu, enabled, warn):
    torch.use_deterministic_algorithms(enabled, warn_only=warn)
    with pytest.raises(ValueError, match="Fail-closed"):
        with deterministic_memory_fill_context(enabled=False):
            pytest.fail("Invalid runtime entered")
    assert torch.utils.deterministic.fill_uninitialized_memory


def recipe(mode="reference"):
    return build_family_recipe(("neural_atom_k1", "2"),
        addon=None if mode == "reference" else "k1_two_pass_mean",
        source_idx_sha256="1" * 64, target_sha256="2" * 64)


@pytest.mark.parametrize("mode", ["reference", "mean2"])
def test_default_policy_keeps_native_optimizer_and_exposure(monkeypatch, mode):
    value = recipe(mode)
    owner.validate_recipe(value, mode=mode)
    assert value["execution_policy"] == owner.ACCELERATED_EXECUTION_POLICY
    assert value["training_recipe"]["scheduler_t_max"] == 40
    assert value["acceptance_requirements"]["optimizer_steps"] == 31240
    calls = []
    monkeypatch.setattr(torch.optim, "AdamW", lambda params, **kwargs: calls.append(kwargs))
    owner._screen_optimizer(torch.nn.Linear(1, 1), mode)
    assert calls == [{"lr": 4e-4, "weight_decay": 1e-5}]


@pytest.mark.parametrize("key,value", [("adamw_fused", True), ("adamw_fused", 0),
    ("adamw_foreach", False), ("cpu_layout", 1), ("fill_uninitialized_memory", True),
    ("torch_version", "2.8.0"), ("loader", "persistent")])
def test_default_policy_rejects_drift_and_boolean_aliases(key, value):
    altered = recipe()
    altered["execution_policy"][key] = value
    with pytest.raises(ValueError, match="execution policy"):
        owner.validate_recipe(altered, mode="reference")


@pytest.mark.parametrize("addon", ["k1_fused_layout", "k1_initialized_layout"])
def test_retired_execution_addons_cannot_launch(addon):
    arm = {"family": {"name": "neural_atom_k1", "version": "2"},
           "addons": [{"name": addon, "version": "1"}]}
    with pytest.raises(ValueError, match="executable"):
        training_adapter(arm)
    with pytest.raises(ValueError, match="executable"):
        build_family_recipe(("neural_atom_k1", "2"), addon=addon,
            source_idx_sha256="1" * 64, target_sha256="2" * 64)


def test_historical_recipe_requires_frozen_source_not_silent_migration():
    value = recipe()
    del value["execution_policy"]
    with pytest.raises(ValueError, match="frozen source"):
        owner.validate_recipe(value, mode="reference")


@pytest.mark.parametrize("mode", ["reference", "mean2", "mean2_clean_second"])
def test_default_cpu_step_matches_test_only_unoptimized_oracle(deterministic_cpu, mode):
    from molgap.training_reproducibility import capture_rng_state, restore_rng_state
    fixtures = [fixture_batch(), fixture_batch(empty_edges=True)]
    model = owner.make_encoder("neural_atom_k1")
    for mixer in model.neural_atom_mixers.values():
        torch.nn.init.normal_(mixer.return_projection.weight, std=0.01)
    state, rng = copy.deepcopy(model.state_dict()), capture_rng_state()
    observations = []
    for accelerated in (False, True):
        model.load_state_dict(state)
        optimizer = owner._screen_optimizer(model, mode)
        restore_rng_state(rng)
        rows = []
        for fixture in fixtures:
            if accelerated:
                result = owner._screen_step(model, optimizer, fixture.clone(), 0., 1., mode, device="cpu")
            else:
                # Only the test oracle retains the old assembly; no live recipe can select it.
                result = owner._optimizer_step(model, optimizer, fixture.clone(), 0., 1., mode)
            rows.append(owner._step_identity(model, optimizer, *result))
        observations.append(rows)
    assert observations[0] == observations[1]
    assert torch.utils.deterministic.fill_uninitialized_memory


def test_step_exception_restores_fill_and_layout(deterministic_cpu, monkeypatch):
    model = owner.make_encoder("neural_atom_k1")
    def fail(*args):
        assert not torch.utils.deterministic.fill_uninitialized_memory
        assert hasattr(model, "_execution_graph_count")
        raise RuntimeError("scratch")
    monkeypatch.setattr(owner, "_optimizer_step", fail)
    with pytest.raises(RuntimeError, match="scratch"):
        owner._screen_step(model, None, fixture_batch(), 0., 1., "reference", device="cpu")
    assert torch.utils.deterministic.fill_uninitialized_memory
    assert not hasattr(model, "_execution_graph_count")
    assert all(not hasattr(mixer, "_execution_dense_layout") for mixer in model.neural_atom_mixers.values())


def test_unqualified_runtime_fails_before_transfer(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.8.0")
    with pytest.raises(ValueError, match="qualified PyTorch"):
        owner._screen_step(None, None, None, 0., 1., "reference", device="cpu")


def test_default_mode_reaches_native_preflight_recipe_boundary(tmp_path):
    with pytest.raises(FileNotFoundError):
        owner.run_screen_preflight(spec=None, package_dir=tmp_path,
            expected_package_identity="1" * 64, arm_id="reference", mode="reference",
            recipe_path=tmp_path / "absent.json", initial_state_path=tmp_path / "initial.pt",
            input_root=tmp_path, output=tmp_path / "out", account="synthetic",
            run_reference="synthetic/kernel", trajectory_id="synthetic")


@pytest.mark.parametrize("observed,accepted", [("2.11.0+cu128", True), ("2.7.1", False)])
def test_runtime_must_match_frozen_recipe(tmp_path, observed, accepted):
    from types import SimpleNamespace
    from molgap.training_reproducibility import atomic_json, sha256_file
    path, initial = tmp_path / "recipe.json", tmp_path / "initial.pt"
    atomic_json(path, recipe())
    initial.write_bytes(b"synthetic-file-hash-only")
    context = SimpleNamespace(training_recipe_sha256=sha256_file(path), to_dict=lambda: {})
    runtime = {"torch": observed, "runtime_fingerprint": "3" * 64}
    if accepted:
        assert owner._runtime_provenance(context, path, initial, runtime, "reference")["mode"] == "reference"
    else:
        with pytest.raises(ValueError, match="observed runtime"):
            owner._runtime_provenance(context, path, initial, runtime, "reference")


def test_resume_preserves_native_moments_scheduler_rng_and_step(deterministic_cpu):
    from molgap.training_reproducibility import capture_rng_state, restore_rng_state
    model = owner.make_encoder("neural_atom_k1").train()
    optimizer = owner._screen_optimizer(model, "reference")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-6)
    batch = fixture_batch()
    owner._screen_step(model, optimizer, batch.clone(), 0., 1., "reference", device="cpu")
    scheduler.step()
    saved = copy.deepcopy({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "scheduler": scheduler.state_dict(), "rng": capture_rng_state()})
    result = owner._screen_step(model, optimizer, batch.clone(), 0., 1., "reference", device="cpu")
    uninterrupted = owner._step_identity(model, optimizer, *result)
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    restore_rng_state(saved["rng"])
    result = owner._screen_step(model, optimizer, batch.clone(), 0., 1., "reference", device="cpu")
    assert uninterrupted == owner._step_identity(model, optimizer, *result)
    assert scheduler.state_dict() == saved["scheduler"]



@pytest.mark.parametrize("corruption", [None, "missing_hashes", "resume_drift", "loss_drift"])
def test_preflight_requires_exact_resume_not_only_summary(tmp_path, corruption):
    from molgap.training_reproducibility import atomic_json, canonical_fingerprint
    provenance = {"mode": "reference", "runtime_fingerprint": "1" * 64}
    architecture = {"accepted": True, "repeatability": {"accepted": True},
        "resume_roundtrip": {"accepted": True, "loss_delta": 0., "state_sha256": ["2" * 64] * 2},
        "zero_initialization_delta": 0., "maximum_overhead_fraction": 0.25,
        "synchronized_step_overhead_fraction": 0.}
    if corruption == "missing_hashes":
        del architecture["resume_roundtrip"]["state_sha256"]
    elif corruption == "resume_drift":
        architecture["resume_roundtrip"]["state_sha256"][1] = "3" * 64
    elif corruption == "loss_drift":
        architecture["resume_roundtrip"]["loss_delta"] = .01
    certificate = {"status": "accepted", "calibration_checks_passed": True,
        "runtime_fingerprint": provenance["runtime_fingerprint"],
        "provenance_sha256": canonical_fingerprint(provenance),
        "architecture_sha256": canonical_fingerprint(architecture)}
    for name, value in (("runtime_provenance", provenance), ("architecture_preflight", architecture),
            ("runtime_certificate", certificate), ("runtime_manifest", {"runtime_fingerprint": "1" * 64})):
        atomic_json(tmp_path / (name + ".json"), value)
    if corruption:
        with pytest.raises(ValueError, match="exact resume"):
            owner.validate_runtime_preflight(tmp_path, provenance)
    else:
        assert owner.validate_runtime_preflight(tmp_path, provenance) == certificate
