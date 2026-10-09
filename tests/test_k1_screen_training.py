"""CPU initialization and frozen-contract checks; no molecular role access."""
import copy
import json
from types import SimpleNamespace

import pytest

from molgap.k1_screen_training import (
    EPOCHS, INITIAL_STATE_SHA256, ROW_ORDER_FINGERPRINT, ROWS_PER_EPOCH,
    SAMPLE_EXPOSURE, STEPS_PER_EPOCH, compute_row_order_fingerprint,
    epoch_order, validate_recipe,
    validate_runtime_preflight, _allocation_costs,
    build_initial_state, build_screen_recipe, _validate_arm_binding,
    _runtime_provenance, _maximum_overhead,
    _load_initial_state,
)
from molgap.v4_runtime import state_dict_sha256, normalized_source_sha256
from molgap.training_reproducibility import sha256_file
from molgap.screen_policy import canonical_fingerprint


def recipe(mode="ssma"):
    return {"mode": mode, "row_order_fingerprint": ROW_ORDER_FINGERPRINT,
            "initialization_sha256": INITIAL_STATE_SHA256,
            "training_recipe": {"seed": 42, "batch_size": 128, "drop_last": True,
                "optimizer": "AdamW", "learning_rate": 4e-4, "weight_decay": 1e-5,
                "clip_grad_norm": 1.0, "scheduler": "CosineAnnealingLR",
                "scheduler_t_max": 40, "scheduler_eta_min": 1e-6,
                "ema": False, "selection": "best-development-live",
                "auxiliary_weight": 0.1 if mode == "clean_fingerprint" else 0.0},
            "acceptance_requirements": {"epochs": 40, "optimizer_steps": 31240,
                "sample_presentations": 3998720, "development_rows": 50000,
                "precision": "fp32"}}


def test_historical_sampler_and_exposure():
    assert compute_row_order_fingerprint() == ROW_ORDER_FINGERPRINT
    assert len(epoch_order(0)) == ROWS_PER_EPOCH == 99968
    assert len(set(epoch_order(0))) == ROWS_PER_EPOCH
    assert epoch_order(0) != epoch_order(1)
    assert EPOCHS * STEPS_PER_EPOCH == 31240
    assert SAMPLE_EXPOSURE == 3998720


@pytest.mark.parametrize("mode", ["reference", "ssma", "clean_fingerprint"])
def test_historical_recipe(mode):
    validate_recipe(recipe(mode), mode=mode)


@pytest.mark.parametrize("field,value", [
    ("epochs", 41), ("optimizer_steps", 31280),
    ("sample_presentations", 4000000), ("precision", "fp16")])
def test_reject_recipe_drift(field, value):
    altered = copy.deepcopy(recipe())
    altered["acceptance_requirements"][field] = value
    with pytest.raises(ValueError, match="exposure"):
        validate_recipe(altered, mode="ssma")


def test_reject_continuous_sampler_substitution():
    altered = recipe()
    altered["row_order_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="sampler"):
        validate_recipe(altered, mode="ssma")


def preflight_files(root, *, accepted=True):
    provenance = {"runtime_fingerprint": "1" * 64, "context": {"arm_id": "reference"}}
    architecture = {"accepted": accepted, "repeatability": {"accepted": True},
        "resume_roundtrip": {"accepted": True}, "zero_initialization_delta": 0.0,
        "maximum_overhead_fraction": 0.25, "synchronized_step_overhead_fraction": 0.20}
    certificate = {"status": "accepted", "calibration_checks_passed": True,
        "runtime_fingerprint": provenance["runtime_fingerprint"],
        "provenance_sha256": canonical_fingerprint(provenance),
        "architecture_sha256": canonical_fingerprint(architecture)}
    for name, value in [("runtime_provenance", provenance), ("runtime_certificate", certificate),
                         ("architecture_preflight", architecture),
                         ("runtime_manifest", {"runtime_fingerprint": provenance["runtime_fingerprint"]})]:
        (root / (name + ".json")).write_text(json.dumps(value), encoding="utf-8")
    return provenance


def test_training_requires_matching_arm_preflight(tmp_path):
    provenance = preflight_files(tmp_path)
    assert validate_runtime_preflight(tmp_path, provenance)["status"] == "accepted"
    changed = copy.deepcopy(provenance)
    changed["context"]["arm_id"] = "ssma"
    with pytest.raises(ValueError, match="identity"):
        validate_runtime_preflight(tmp_path, changed)


def test_preflight_requires_accepted_architecture(tmp_path):
    provenance = preflight_files(tmp_path, accepted=False)
    with pytest.raises(ValueError, match="qualified"):
        validate_runtime_preflight(tmp_path, provenance)


def test_preflight_does_not_trust_mutated_timing(tmp_path):
    provenance = preflight_files(tmp_path)
    path = tmp_path / "architecture_preflight.json"
    altered = json.loads(path.read_text())
    altered["synchronized_step_overhead_fraction"] = 0.40
    path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="qualified"):
        validate_runtime_preflight(tmp_path, provenance)


def test_allocation_cost_window_keeps_wall_and_allocated_device_separate():
    costs = _allocation_costs(12.5, "Tesla T4", scope="invocation_excludes_queue_bootstrap")
    assert costs[0]["value"] == costs[1]["value"] == 12.5
    assert costs[0]["semantics"] == "process_wall"
    assert costs[1]["semantics"] == "allocated_device"
    assert len(costs) == 2


def test_default_builder_preserves_historical_mapping():
    expected = recipe("reference")
    expected["development_role_identity"] = "pcqm4mv2-ogb-fixed-100k-v1:development-100000-150000"
    expected["acceptance_requirements"].update(source_idx_sha256="1" * 64, target_sha256="2" * 64)
    actual = build_screen_recipe("reference", source_idx_sha256="1" * 64, target_sha256="2" * 64)
    assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)


@pytest.fixture(scope="module")
def seed43_sha():
    import torch
    before = torch.random.get_rng_state().clone()
    historical = build_initial_state()
    assert state_dict_sha256(historical) == INITIAL_STATE_SHA256
    fresh = build_initial_state(43)
    digest = state_dict_sha256(fresh)
    assert digest != INITIAL_STATE_SHA256
    assert digest == state_dict_sha256(build_initial_state(43))
    assert historical.keys() == fresh.keys()
    assert all(tensor.device.type == "cpu" for tensor in fresh.values())
    assert torch.equal(before, torch.random.get_rng_state())
    return digest


def seed43_recipe(digest, mode="reference"):
    return build_screen_recipe(mode, source_idx_sha256="1" * 64, target_sha256="2" * 64,
                               seed=43, initialization_sha256=digest)


@pytest.mark.parametrize("mode", ["reference", "ssma", "mean2", "mean2_clean_second"])
def test_seed43_recipe(seed43_sha, mode):
    fresh = seed43_recipe(seed43_sha, mode)
    validate_recipe(fresh, mode=mode)
    assert fresh["row_order_fingerprint"] == compute_row_order_fingerprint(43)
    assert fresh["row_order_fingerprint"] != ROW_ORDER_FINGERPRINT
    assert epoch_order(0, 43) != epoch_order(0)
    assert len(epoch_order(0, 43)) == ROWS_PER_EPOCH


@pytest.mark.parametrize("seed", [-1, True, 43.0, "43", None, 2**32])
def test_invalid_seed(seed):
    with pytest.raises(ValueError, match="seed"):
        build_screen_recipe("reference", source_idx_sha256="1" * 64, target_sha256="2" * 64, seed=seed)
    altered = recipe()
    altered["training_recipe"]["seed"] = seed
    with pytest.raises(ValueError, match="seed"):
        validate_recipe(altered, mode="ssma")
    with pytest.raises(ValueError, match="seed"):
        compute_row_order_fingerprint(seed=seed)
    with pytest.raises(ValueError, match="seed"):
        epoch_order(0, seed)


def test_seed43_requires_explicit_initialization(seed43_sha):
    with pytest.raises(ValueError, match="requires"):
        build_screen_recipe("reference", source_idx_sha256="1" * 64, target_sha256="2" * 64, seed=43)
    fresh = seed43_recipe(seed43_sha)
    fresh["initialization_sha256"] = "not-a-sha256"
    with pytest.raises(ValueError, match="initialization"):
        validate_recipe(fresh, mode="reference")


def test_nonhistorical_runtime_validation_never_reconstructs(monkeypatch):
    import molgap.k1_screen_training as trainer
    def forbidden(*args, **kwargs):
        pytest.fail("Runtime validation must not regenerate transported initialization")
    monkeypatch.setattr(trainer, "build_initial_state", forbidden)
    monkeypatch.setattr(trainer, "make_encoder", forbidden)
    fresh = seed43_recipe("3" * 64)
    validate_recipe(fresh, mode="reference")
    spec, _ = binding_spec(fresh)
    _validate_arm_binding(spec, SimpleNamespace(arm_id="single"), "reference", fresh)


def test_transported_initial_tensor_hash_is_verified_without_reconstruction(tmp_path, monkeypatch):
    import torch
    import molgap.k1_screen_training as trainer
    state = {"toy_tensor": torch.arange(4, dtype=torch.float32)}
    initial = tmp_path / "initial.pt"
    torch.save(state, initial)
    fresh = seed43_recipe(state_dict_sha256(state))
    def forbidden(*args, **kwargs):
        pytest.fail("Loading pinned tensors must not regenerate initialization")
    monkeypatch.setattr(trainer, "make_encoder", forbidden)
    loaded = _load_initial_state(initial, fresh)
    assert torch.equal(loaded["toy_tensor"], state["toy_tensor"])
    torch.save({"toy_tensor": state["toy_tensor"] + 1}, initial)
    with pytest.raises(ValueError, match="tensor identity"):
        _load_initial_state(initial, fresh)


def test_seed42_rejects_initialization_drift():
    historical = recipe()
    historical["initialization_sha256"] = "3" * 64
    with pytest.raises(ValueError, match="historical initialization"):
        validate_recipe(historical, mode="ssma")


def binding_spec(fresh, mode="reference"):
    from pathlib import Path
    import molgap.k1_screen_training as trainer
    addons = []
    if mode == "mean2_clean_second":
        addons = [{"name": "k1_mean2_clean_second", "version": "1", "config": {},
                   "source_sha256": normalized_source_sha256(Path(trainer.__file__).with_name("k1_clean_second.py"))}]
    arm = {"arm_id": "single", "initialization": {"kind": "frozen_state",
           "seed": fresh["training_recipe"]["seed"], "state_sha256": fresh["initialization_sha256"]},
           "training": {"sampler": {"sha256": fresh["row_order_fingerprint"]}, "overrides": {}},
           "addons": addons}
    return SimpleNamespace(to_dict=lambda: {"arms": [arm]}), arm


@pytest.mark.parametrize("field,value", [("seed", 42), ("state_sha256", INITIAL_STATE_SHA256)])
def test_spec_recipe_seed_and_tensor_binding(seed43_sha, field, value):
    fresh = seed43_recipe(seed43_sha)
    spec, arm = binding_spec(fresh)
    context = SimpleNamespace(arm_id="single")
    _validate_arm_binding(spec, context, "reference", fresh)
    arm["initialization"][field] = value
    with pytest.raises(ValueError, match="initialization/sampler"):
        _validate_arm_binding(spec, context, "reference", fresh)


def test_clean_second_recipe_and_addon_binding():
    fresh = build_screen_recipe("mean2_clean_second", source_idx_sha256="1" * 64, target_sha256="2" * 64)
    validate_recipe(fresh, mode="mean2_clean_second")
    spec, arm = binding_spec(fresh, "mean2_clean_second")
    _validate_arm_binding(spec, SimpleNamespace(arm_id="single"), "mean2_clean_second", fresh)
    assert _maximum_overhead("mean2_clean_second") == _maximum_overhead("mean2") == 1.0
    arm["addons"][0]["config"] = {"eval": True}
    with pytest.raises(ValueError, match="Clean-second"):
        _validate_arm_binding(spec, SimpleNamespace(arm_id="single"), "mean2_clean_second", fresh)


def test_seed43_spec_sampler_must_match(seed43_sha):
    fresh = seed43_recipe(seed43_sha)
    spec, arm = binding_spec(fresh)
    arm["training"]["sampler"]["sha256"] = ROW_ORDER_FINGERPRINT
    with pytest.raises(ValueError, match="initialization/sampler"):
        _validate_arm_binding(spec, SimpleNamespace(arm_id="single"), "reference", fresh)


def test_seed43_provenance_and_preflight_identity(seed43_sha, tmp_path):
    fresh = seed43_recipe(seed43_sha)
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(fresh), encoding="utf-8")
    initial = tmp_path / "initial.pt"
    initial.write_bytes(b"file identity only; no checkpoint loaded")
    context = SimpleNamespace(training_recipe_sha256=sha256_file(recipe_path), to_dict=lambda: {"arm_id": "single"})
    provenance = _runtime_provenance(context, recipe_path, initial, {"runtime_fingerprint": "1" * 64}, "reference")
    assert provenance["initialization_sha256"] == seed43_sha
    assert provenance["row_order_fingerprint"] == fresh["row_order_fingerprint"]
    assert provenance["initial_state_file_sha256"] == sha256_file(initial)
    saved = preflight_files(tmp_path)
    saved.update(provenance)
    (tmp_path / "runtime_provenance.json").write_text(json.dumps(saved))
    certificate_path = tmp_path / "runtime_certificate.json"
    certificate = json.loads(certificate_path.read_text())
    certificate["provenance_sha256"] = canonical_fingerprint(saved)
    certificate_path.write_text(json.dumps(certificate))
    validate_runtime_preflight(tmp_path, provenance)
    stale = copy.deepcopy(provenance)
    stale["initialization_sha256"] = INITIAL_STATE_SHA256
    with pytest.raises(ValueError, match="identity"):
        validate_runtime_preflight(tmp_path, stale)
