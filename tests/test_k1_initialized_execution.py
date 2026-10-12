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


def recipe():
    return build_family_recipe(("neural_atom_k1", "2"), addon="k1_initialized_layout",
                               source_idx_sha256="1" * 64, target_sha256="2" * 64)


def test_new_policy_keeps_native_optimizer_and_entire_training_contract(monkeypatch):
    value = recipe()
    owner.validate_recipe(value, mode="initialized_layout")
    original = owner.build_screen_recipe("reference", source_idx_sha256="1" * 64,
                                        target_sha256="2" * 64)
    assert value["training_recipe"] == original["training_recipe"]
    assert value["acceptance_requirements"] == original["acceptance_requirements"]
    assert value["row_order_fingerprint"] == original["row_order_fingerprint"]
    assert value["execution_policy"] == owner.INITIALIZED_LAYOUT_POLICY
    calls = []
    monkeypatch.setattr(torch.optim, "AdamW", lambda params, **kwargs: calls.append(kwargs))
    owner._screen_optimizer(torch.nn.Linear(1, 1), "initialized_layout")
    assert calls == [{"lr": 4e-4, "weight_decay": 1e-5}]


def test_recommended_execution_is_explicit_and_preserves_reference():
    from molgap.experiment_execution import (
        RECOMMENDED_K1_EXECUTION_ADDON, RECOMMENDED_K1_EXECUTION_TORCH_VERSION,
    )
    options = {"source_idx_sha256": "1" * 64, "target_sha256": "2" * 64}
    reference = build_family_recipe(("neural_atom_k1", "2"), **options)
    recommended = build_family_recipe(("neural_atom_k1", "2"),
        addon=RECOMMENDED_K1_EXECUTION_ADDON,
        execution_torch_version=RECOMMENDED_K1_EXECUTION_TORCH_VERSION, **options)
    assert "execution_policy" not in reference
    assert recommended["training_recipe"] == reference["training_recipe"]
    assert recommended["acceptance_requirements"] == reference["acceptance_requirements"]
    assert recommended["execution_policy"]["torch_version"] == "2.11.0"


@pytest.mark.parametrize("key,value", [("adamw_fused", True), ("adamw_fused", 0),
    ("adamw_foreach", False), ("cpu_layout", 1), ("fill_uninitialized_memory", True),
    ("torch_version", "2.8.0"), ("loader", "persistent")])
def test_new_policy_rejects_drift_and_boolean_aliases(key, value):
    changed = recipe()
    changed["execution_policy"][key] = value
    with pytest.raises(ValueError, match="execution policy"):
        owner.validate_recipe(changed, mode="initialized_layout")


def test_addon_spec_and_owner_validation_are_wired_without_new_records():
    from test_experiment_spec import payload as synthetic_payload
    root = Path(__file__).resolve().parents[1]
    payload = synthetic_payload.__wrapped__()
    arm = json.loads((root / "experiments/pcqm_k1_slot_width96/reference_binding/original_reference_arm.json").read_text())
    arm.update(arm_id="initialized-layout", scientific_role="candidate", addon_semantics="ordered")
    payload["arms"][1] = arm
    arm["addons"] = [{"name": "k1_initialized_layout", "version": "1", "config": {},
                      "source_sha256": normalized_source_sha256(Path(owner.__file__))}]
    spec = ExperimentSpec(payload)
    assert training_adapter(arm).mode(arm) == "initialized_layout"
    frozen = recipe()
    # Only row/target pins differ between this scratch recipe and the retained template.
    assert owner.validate_screen_recipe(spec, arm["arm_id"], frozen)["mode"] == "initialized_layout"
    payload["arms"][1]["addons"][0]["config"] = {"skip_reference_check": True}
    with pytest.raises(ValueError):
        ExperimentSpec(payload)


def test_cpu_multistep_parity_and_rng_restore(deterministic_cpu):
    fixtures = [fixture_batch(), fixture_batch(empty_edges=True)] * 2
    model = owner.make_encoder("neural_atom_k1")
    for mixer in model.neural_atom_mixers.values():
        torch.nn.init.normal_(mixer.return_projection.weight, std=0.01)
    rng = torch.get_rng_state().clone()
    result = owner.qualify_initialized_layout(model.state_dict(), fixtures, 0., 1., device="cpu")
    assert result["accepted"] and result["optimizer_steps"] == 4
    assert result["observations"]["reference"] == result["observations"]["initialized_layout"]
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.utils.deterministic.fill_uninitialized_memory
    assert all(batch.x.device.type == batch.batch.device.type == "cpu" for batch in fixtures)


def test_step_exception_restores_fill_and_layout(deterministic_cpu, monkeypatch):
    model = owner.make_encoder("neural_atom_k1")
    def fail(*args):
        assert not torch.utils.deterministic.fill_uninitialized_memory
        assert hasattr(model, "_execution_graph_count")
        raise RuntimeError("scratch")
    monkeypatch.setattr(owner, "_optimizer_step", fail)
    with pytest.raises(RuntimeError, match="scratch"):
        owner._screen_step(model, None, fixture_batch(), 0., 1., "initialized_layout", device="cpu")
    assert torch.utils.deterministic.fill_uninitialized_memory
    assert not hasattr(model, "_execution_graph_count")
    assert all(not hasattr(mixer, "_execution_dense_layout") for mixer in model.neural_atom_mixers.values())


def test_unqualified_runtime_fails_before_transfer(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.8.0")
    with pytest.raises(ValueError, match="qualified PyTorch"):
        owner._screen_step(None, None, None, 0., 1., "initialized_layout", device="cpu")


def test_registered_mode_reaches_native_preflight_recipe_boundary(tmp_path):
    # CPU-only sentry: distinguish admission to the owner from a missing recipe.
    with pytest.raises(FileNotFoundError):
        owner.run_screen_preflight(spec=None, package_dir=tmp_path,
            expected_package_identity="1" * 64, arm_id="candidate",
            mode="initialized_layout", recipe_path=tmp_path / "absent.json",
            initial_state_path=tmp_path / "absent.pt", input_root=tmp_path,
            output=tmp_path / "not-created", account="synthetic",
            run_reference="synthetic/not-submitted", trajectory_id="synthetic")
    assert not (tmp_path / "not-created").exists()


@pytest.mark.parametrize("version", ["2.7.1", "2.11.0"])
def test_explicit_native_runtime_recipe_pin(version):
    frozen = build_family_recipe(("neural_atom_k1", "2"), addon="k1_initialized_layout",
        source_idx_sha256="1" * 64, target_sha256="2" * 64,
        execution_torch_version=version)
    owner.validate_recipe(frozen, mode="initialized_layout")
    assert frozen["execution_policy"]["torch_version"] == version
    assert frozen["training_recipe"] == recipe()["training_recipe"]
    with pytest.raises(ValueError, match="runtime pin"):
        build_family_recipe(("neural_atom_k1", "2"), addon=None,
            source_idx_sha256="1" * 64, target_sha256="2" * 64,
            execution_torch_version=version)


@pytest.mark.parametrize("observed,accepted", [("2.11.0+cu128", True),
    ("2.7.1+cu128", False), ("", False)])
def test_observed_runtime_must_match_frozen_recipe(tmp_path, observed, accepted):
    from types import SimpleNamespace
    from molgap.training_reproducibility import atomic_json, sha256_file
    frozen = build_family_recipe(("neural_atom_k1", "2"), addon="k1_initialized_layout",
        source_idx_sha256="1" * 64, target_sha256="2" * 64,
        execution_torch_version="2.11.0")
    path, initial = tmp_path / "recipe.json", tmp_path / "initial.pt"
    atomic_json(path, frozen)
    initial.write_bytes(b"synthetic-file-hash-only")
    context = SimpleNamespace(training_recipe_sha256=sha256_file(path), to_dict=lambda: {})
    runtime = {"torch": observed, "runtime_fingerprint": "3" * 64}
    if accepted:
        assert owner._runtime_provenance(context, path, initial, runtime,
                                        "initialized_layout")["mode"] == "initialized_layout"
    else:
        with pytest.raises(ValueError, match="observed runtime"):
            owner._runtime_provenance(context, path, initial, runtime, "initialized_layout")


def test_resume_preserves_native_moments_scheduler_rng_and_step(deterministic_cpu):
    from molgap.training_reproducibility import capture_rng_state, restore_rng_state
    model = owner.make_encoder("neural_atom_k1").train()
    optimizer = owner._screen_optimizer(model, "initialized_layout")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-6)
    batch = fixture_batch()
    owner._screen_step(model, optimizer, batch.clone(), 0., 1., "initialized_layout", device="cpu")
    scheduler.step()
    saved = copy.deepcopy({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "scheduler": scheduler.state_dict(), "rng": capture_rng_state()})
    result = owner._screen_step(model, optimizer, batch.clone(), 0., 1., "initialized_layout", device="cpu")
    uninterrupted = owner._step_identity(model, optimizer, *result)
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    restore_rng_state(saved["rng"])
    result = owner._screen_step(model, optimizer, batch.clone(), 0., 1., "initialized_layout", device="cpu")
    assert uninterrupted == owner._step_identity(model, optimizer, *result)
    assert scheduler.state_dict() == saved["scheduler"]


def test_summary_booleans_cannot_replace_reference_observations():
    for evidence in (None, {}, {"accepted": True, "optimizer_steps": 2, "reference_mode": "reference"},
            {"accepted": True, "optimizer_steps": 2, "reference_mode": "reference",
             "observations": {"reference": [{}, {}], "initialized_layout": [{}, {}]}}):
        with pytest.raises(ValueError):
            owner._validate_initialized_reference_evidence(evidence)


def test_qualification_detects_changed_metric_and_restores_rng(deterministic_cpu, monkeypatch):
    fixture = fixture_batch()
    model = owner.make_encoder("neural_atom_k1")
    original_step = owner._screen_step
    def changed_step(*args, **kwargs):
        loss, absolute, count = original_step(*args, **kwargs)
        return (loss + 1 if args[5] == "initialized_layout" else loss), absolute, count
    monkeypatch.setattr(owner, "_screen_step", changed_step)
    rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="reference step identity"):
        owner.qualify_initialized_layout(model.state_dict(), [fixture, fixture], 0., 1., device="cpu")
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.utils.deterministic.fill_uninitialized_memory


@pytest.mark.parametrize("corruption", [None, "summary_only", "step_drift", "resume_drift"])
def test_downstream_preflight_requires_real_step_and_exact_resume_fields(
        deterministic_cpu, tmp_path, monkeypatch, corruption):
    from molgap.training_reproducibility import atomic_json, canonical_fingerprint
    fixture = fixture_batch()
    model = owner.make_encoder("neural_atom_k1")
    evidence = owner.qualify_initialized_layout(model.state_dict(), [fixture, fixture], 0., 1., device="cpu")
    monkeypatch.setattr(owner, "BATCH_SIZE", 3)
    provenance = {"mode": "initialized_layout", "runtime_fingerprint": "1" * 64}
    architecture = {"accepted": True, "repeatability": {"accepted": True},
        "resume_roundtrip": {"accepted": True, "loss_delta": 0., "state_sha256": ["2" * 64] * 2},
        "zero_initialization_delta": 0., "maximum_overhead_fraction": 0.25,
        "synchronized_step_overhead_fraction": 0., "reference_equivalence": evidence}
    if corruption == "summary_only":
        architecture["reference_equivalence"] = {"accepted": True, "optimizer_steps": 2}
    elif corruption == "step_drift":
        evidence["observations"]["initialized_layout"][0]["loss"] += 1
    elif corruption == "resume_drift":
        architecture["resume_roundtrip"]["state_sha256"][1] = "3" * 64
    certificate = {"status": "accepted", "calibration_checks_passed": True,
        "runtime_fingerprint": provenance["runtime_fingerprint"],
        "provenance_sha256": canonical_fingerprint(provenance),
        "architecture_sha256": canonical_fingerprint(architecture)}
    for name, value in (("runtime_provenance", provenance), ("architecture_preflight", architecture),
            ("runtime_certificate", certificate), ("runtime_manifest", {"runtime_fingerprint": "1" * 64})):
        atomic_json(tmp_path / (name + ".json"), value)
    if corruption:
        with pytest.raises(ValueError, match="Initialized-layout"):
            owner.validate_runtime_preflight(tmp_path, provenance)
    else:
        assert owner.validate_runtime_preflight(tmp_path, provenance) == certificate
