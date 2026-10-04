"""CPU-only tests for the frozen K1 train-role teacher cache and objective."""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from molgap.experiment_execution import build_family_recipe
from molgap.experiment_spec import (
    ExperimentSpec,
    FAMILIES,
    SCHEMA_VERSION_V2,
    TERMINAL_PROTOCOL,
)
from molgap.k1_teacher_cache import (
    DISTILL_WEIGHTS,
    FORMAT,
    TRAIN_ROWS,
    K1TeacherCache,
    load_teacher_cache,
    validate_teacher_config,
)
from molgap import k1_screen_training as k1_training
from molgap.training_reproducibility import sha256_file
from molgap.v4_runtime import normalized_source_sha256


ROOT = Path(__file__).resolve().parents[1]
TEACHER_IDENTITY = "a" * 64
MANIFEST_IDENTITY = "b" * 64
SOURCE_IDX_SHA256 = "c" * 64
TARGET_SHA256 = "d" * 64


def _write_cache(
    root: Path,
    *,
    source_idx: torch.Tensor | None = None,
    prediction_eV: torch.Tensor | None = None,
    manifest_changes: dict | None = None,
) -> tuple[dict, str]:
    root.mkdir(parents=True, exist_ok=True)
    if source_idx is None:
        source_idx = torch.arange(TRAIN_ROWS, dtype=torch.int64)
    if prediction_eV is None:
        prediction_eV = torch.arange(TRAIN_ROWS, dtype=torch.float32) / 100.0
    payload_path = root / "teacher_predictions.pt"
    torch.save({"source_idx": source_idx, "prediction_eV": prediction_eV}, payload_path)
    manifest = {
        "format": FORMAT,
        "teacher_identity": TEACHER_IDENTITY,
        "role": "internal_training",
        "source_idx_range": [0, TRAIN_ROWS],
        "dataset_manifest_sha256": k1_training.FIXED_MANIFEST_SHA256,
        "path": "teacher_predictions.pt",
        "sha256": sha256_file(payload_path),
    }
    if manifest_changes:
        manifest.update(manifest_changes)
    manifest_bytes = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    manifest_path = root / "manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    return manifest, hashlib.sha256(manifest_bytes).hexdigest()


def _config(manifest_sha256: str, *, weight: float = 0.1) -> dict:
    return {
        "weight": weight,
        "teacher_identity": TEACHER_IDENTITY,
        "cache_manifest_sha256": manifest_sha256,
    }


def test_cache_manifest_hash_and_attach_join_preserve_requested_order(tmp_path):
    manifest, manifest_sha256 = _write_cache(tmp_path / "cache")
    config = _config(manifest_sha256)
    cache = load_teacher_cache(tmp_path, config, mode="distill_weak")

    assert cache.identity == manifest_sha256
    assert cache.manifest == manifest
    batch = SimpleNamespace(source_idx=torch.tensor([17, 2, 9], dtype=torch.int64))
    cache.attach(batch)
    assert torch.equal(batch.teacher_eV, torch.tensor([0.17, 0.02, 0.09]))


def test_cache_rejects_a_manifest_changed_after_its_hash_was_pinned(tmp_path):
    _, manifest_sha256 = _write_cache(tmp_path / "cache")
    manifest_path = tmp_path / "cache" / "manifest.json"
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="exactly one pinned"):
        load_teacher_cache(tmp_path, _config(manifest_sha256), mode="distill_weak")


def test_cache_rejects_payload_hash_mismatch(tmp_path):
    _, manifest_sha256 = _write_cache(tmp_path / "cache")
    payload_path = tmp_path / "cache" / "teacher_predictions.pt"
    payload_path.write_bytes(payload_path.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="payload hash mismatch"):
        load_teacher_cache(tmp_path, _config(manifest_sha256), mode="distill_weak")


@pytest.mark.parametrize(
    ("invalidity", "error"),
    [
        ("out_of_order_source_idx", "ascending train indices"),
        ("wrong_source_idx_dtype", "ascending train indices"),
        ("wrong_prediction_dtype", "ascending train indices"),
        ("nonfinite_prediction", "ascending train indices"),
    ],
)
def test_cache_requires_exact_100k_order_and_finite_float32_predictions(
    tmp_path, invalidity, error
):
    source_idx = torch.arange(TRAIN_ROWS, dtype=torch.int64)
    prediction = torch.arange(TRAIN_ROWS, dtype=torch.float32) / 100.0
    if invalidity == "out_of_order_source_idx":
        source_idx[0], source_idx[1] = source_idx[1].clone(), source_idx[0].clone()
    elif invalidity == "wrong_source_idx_dtype":
        source_idx = source_idx.to(torch.int32)
    elif invalidity == "wrong_prediction_dtype":
        prediction = prediction.to(torch.float64)
    elif invalidity == "nonfinite_prediction":
        prediction[123] = float("nan")

    _, manifest_sha256 = _write_cache(
        tmp_path / "cache", source_idx=source_idx, prediction_eV=prediction
    )
    with pytest.raises(ValueError, match=error):
        load_teacher_cache(tmp_path, _config(manifest_sha256), mode="distill_weak")


@pytest.mark.parametrize("source_idx", [[], [-1], [TRAIN_ROWS], [3, 3]])
def test_attach_rejects_empty_duplicate_and_nontraining_indices(tmp_path, source_idx):
    _, manifest_sha256 = _write_cache(tmp_path / "cache")
    cache = load_teacher_cache(tmp_path, _config(manifest_sha256), mode="distill_weak")
    batch = SimpleNamespace(source_idx=torch.tensor(source_idx, dtype=torch.int64))

    with pytest.raises(ValueError, match="duplicate or non-training indices"):
        cache.attach(batch)


@pytest.mark.parametrize(
    ("mode", "weight"),
    [("distill_weak", 0.1), ("distill_strong", 1.0)],
)
def test_registered_teacher_weights_are_exact(mode, weight):
    config = _config(MANIFEST_IDENTITY, weight=weight)
    assert DISTILL_WEIGHTS[mode] == weight
    assert validate_teacher_config(config, mode) == config


@pytest.mark.parametrize(
    ("mode", "weight"),
    [
        ("distill_weak", 0.1000001),
        ("distill_weak", 1.0),
        ("distill_strong", 0.1),
        ("distill_strong", True),
    ],
)
def test_teacher_config_rejects_weights_outside_frozen_mode(mode, weight):
    with pytest.raises(ValueError, match="differs from frozen mode"):
        validate_teacher_config(_config(MANIFEST_IDENTITY, weight=weight), mode)


def test_teacher_config_rejects_unpinned_or_extra_fields():
    extra = _config(MANIFEST_IDENTITY)
    extra["temperature"] = 1.0
    missing = _config(MANIFEST_IDENTITY)
    del missing["teacher_identity"]
    for config in (extra, missing):
        with pytest.raises(ValueError, match="exact K1 teacher cache configuration"):
            validate_teacher_config(config, "distill_weak")


def _spec_for_distillation(arm_id: str, addon_name: str, config: dict) -> ExperimentSpec:
    family = ("neural_atom_k1", "2")
    contract = FAMILIES[family]
    addon_source = normalized_source_sha256(
        Path(k1_training.__file__).with_name("k1_teacher_cache.py")
    )
    arm = {
        "arm_id": arm_id,
        "scientific_role": "candidate",
        "family": {"name": family[0], "version": family[1]},
        "base": {"name": "synthetic-base", "version": "1", "sha256": "1" * 64},
        "initialization": {
            "kind": "frozen_state",
            "seed": 42,
            "state_sha256": k1_training.INITIAL_STATE_SHA256,
        },
        "data": {
            "dataset": {"name": "pcqm4mv2", "version": "1", "sha256": "2" * 64},
            "split": {"name": "synthetic-split", "version": "1", "sha256": "3" * 64},
            "roles": [
                {
                    "role": role,
                    "membership_sha256": "4" * 64,
                    "row_order_sha256": "5" * 64,
                    "usage_sha256": "6" * 64,
                }
                for role in contract.roles
            ],
            "feature_schema": contract.feature_schema,
            "feature_sha256": "7" * 64,
            "target": "pcqm4mv2-gap-eV-direct",
        },
        "training": {
            "recipe": {"name": contract.recipe, "version": "1", "sha256": "8" * 64},
            "overrides": {},
            "objective": {
                "name": "normalized-gap-l1-plus-frozen-teacher-mse",
                "version": "1",
                "sha256": "9" * 64,
            },
            "sampler": {
                "name": contract.sampler,
                "version": "1",
                "sha256": k1_training.ROW_ORDER_FINGERPRINT,
            },
            "transform": {
                "name": contract.transform,
                "version": "1",
                "sha256": "a" * 64,
            },
        },
        "addons": [{
            "name": addon_name,
            "version": "1",
            "config": copy.deepcopy(config),
            "source_sha256": addon_source,
        }],
        "addon_semantics": "ordered",
    }
    return ExperimentSpec({
        "schema_version": SCHEMA_VERSION_V2,
        "experiment_id": "synthetic-k1-distillation-test",
        "logical_run_id": "synthetic-k1-distillation-run",
        "arms": [arm],
        "platform": {
            "name": "local",
            "accelerator": "synthetic-cpu",
            "device_count": 1,
            "cpu_cores": 2,
            "memory_gib": 4,
            "atomic_checkpoints": True,
            "retrievable_chunks": True,
        },
        "prospective": {"arms": [{
            "arm_id": arm_id,
            "trajectory_id": "trajectory-" + arm_id,
            "plan_spec_ref": "plans/" + arm_id + ".json",
            "plan_spec_sha256": "b" * 64,
            "output": "experiments/" + arm_id,
        }]},
        "evidence": {
            "policy": {"name": "molgap-v5", "version": "1", "sha256": "c" * 64},
            "required_artifacts": [
                "v5_evidence", "costs", "roles", "trace_manifest", "terminal_artifact",
            ],
        },
        "terminal_protocol": TERMINAL_PROTOCOL,
    })


@pytest.mark.parametrize(
    ("mode", "addon_name", "weight"),
    [
        ("distill_weak", "k1_fusion_distill_weak", 0.1),
        ("distill_strong", "k1_fusion_distill_strong", 1.0),
    ],
)
def test_family_recipe_and_spec_bind_exact_objective_and_teacher_config(
    mode, addon_name, weight
):
    config = _config(MANIFEST_IDENTITY, weight=weight)
    recipe = build_family_recipe(
        ("neural_atom_k1", "2"),
        addon=addon_name,
        source_idx_sha256=SOURCE_IDX_SHA256,
        target_sha256=TARGET_SHA256,
        addon_config=config,
    )
    spec = _spec_for_distillation(mode, addon_name, config)
    declared_arm = spec.to_dict()["arms"][0]

    assert recipe["teacher_cache"] == config
    assert declared_arm["training"]["objective"]["name"] == (
        "normalized-gap-l1-plus-frozen-teacher-mse"
    )
    assert declared_arm["addons"][0]["config"] == config
    assert k1_training.validate_screen_recipe(spec, mode, recipe)["mode"] == mode

    wrong_objective = spec.to_dict()
    wrong_objective["arms"][0]["training"]["objective"]["name"] = "normalized-gap-l1"
    with pytest.raises(ValueError):
        ExperimentSpec(wrong_objective)

    mismatched_recipe = copy.deepcopy(recipe)
    mismatched_recipe["teacher_cache"]["teacher_identity"] = "e" * 64
    with pytest.raises(ValueError, match="Teacher recipe differs from Spec addon configuration"):
        k1_training.validate_screen_recipe(spec, mode, mismatched_recipe)


@pytest.mark.parametrize(("mode", "weight"), [("distill_weak", 0.1), ("distill_strong", 1.0)])
def test_optimizer_step_uses_weighted_l1_plus_mse_but_reports_label_mae(
    monkeypatch, mode, weight
):
    class TinyStudent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.4, dtype=torch.float32))

    model = TinyStudent()
    features = torch.tensor([0.1, 0.2], dtype=torch.float32)
    teacher = torch.tensor([11.2, 12.0], dtype=torch.float32, requires_grad=True)
    batch = SimpleNamespace(
        features=features,
        y=torch.tensor([10.0, 9.0], dtype=torch.float32),
        teacher_eV=teacher,
    )
    monkeypatch.setattr(k1_training, "_forward", lambda student, item: student.scale * item.features)

    probe = torch.tensor(0.4, dtype=torch.float32, requires_grad=True)
    prediction = probe * features
    target = (batch.y - 10.0) / 2.0
    teacher_target = (teacher.detach() - 10.0) / 2.0
    expected_loss = torch.nn.functional.l1_loss(prediction, target) + weight * (
        torch.nn.functional.mse_loss(prediction, teacher_target)
    )
    expected_gradient = torch.autograd.grad(expected_loss, probe)[0]
    assert abs(float(expected_gradient)) < 1.0

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss, label_absolute_sum, rows = k1_training._optimizer_step(
        model, optimizer, batch, mean=10.0, std=2.0, mode=mode
    )

    assert loss.item() == pytest.approx(expected_loss.item(), rel=1e-6)
    assert model.scale.grad.item() == pytest.approx(expected_gradient.item(), rel=1e-6)
    assert teacher.grad is None
    assert rows == 2
    assert label_absolute_sum.item() == pytest.approx((prediction.detach() - target).abs().sum().item())
    assert label_absolute_sum.item() * 2.0 / rows == pytest.approx(0.62, abs=1e-6)


def test_kaggle_credential_session_binds_basic_without_using_global_oauth(tmp_path, monkeypatch):
    class FakeSession:
        def __init__(self):
            self.auth = ("global-oauth-user", "global-oauth-token")

    class FakeHttpClient:
        def __init__(self):
            self._session = FakeSession()
            self._signed_in = False

            def global_oauth_fill():
                self._session.auth = ("global-oauth-user", "global-oauth-token")
                self._signed_in = True

            self._try_fill_auth = global_oauth_fill

    class FakeKaggleApi:
        def _load_config(self, config):
            self.loaded_config = config

    class FakeKaggleClient:
        def __init__(self, *, username, password):
            self.init_credentials = (username, password)
            self._http_client = FakeHttpClient()

    kaggle = ModuleType("kaggle")
    kaggle.__path__ = []
    kaggle_api = ModuleType("kaggle.api")
    kaggle_api.__path__ = []
    kaggle_extended = ModuleType("kaggle.api.kaggle_api_extended")
    kaggle_extended.KaggleApi = FakeKaggleApi
    kagglesdk = ModuleType("kagglesdk")
    kagglesdk.KaggleClient = FakeKaggleClient
    for name, module in (
        ("kaggle", kaggle),
        ("kaggle.api", kaggle_api),
        ("kaggle.api.kaggle_api_extended", kaggle_extended),
        ("kagglesdk", kagglesdk),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    credentials_path = tmp_path / "mock-credentials.json"
    credentials_path.write_text(
        json.dumps({"username": "fixture-account", "key": "fixture-key"}),
        encoding="utf-8",
    )
    credential_api = importlib.import_module("platforms.kaggle.credential_api")
    api = credential_api.api_for_credentials(credentials_path)
    client = api.build_kaggle_client()

    assert api.loaded_config == {"username": "fixture-account", "key": "fixture-key"}
    assert client.init_credentials == ("fixture-account", "fixture-key")
    assert client._http_client._session.auth == (
        "global-oauth-user", "global-oauth-token"
    )
    client._http_client._try_fill_auth()
    assert client._http_client._session.auth == ("fixture-account", "fixture-key")
    assert client._http_client._signed_in is True


def test_objective_comparison_prelaunch_allows_only_loss_identity():
    from molgap.comparison_readiness import (
        CAUSAL_REQUIRED_ROLE_KINDS,
        ROLE_EVENT_KINDS,
        TRACE_FIELD_DECLARATIONS,
        assess_comparison_prelaunch,
    )

    arguments = {
        "candidate_id": "synthetic-distillation",
        "candidate_plan": {
            "comparison_identity": {"loss_identity": "normalized-gap-l1-plus-teacher-mse"},
            "source_config_status": "frozen",
            "source_commit_or_archive": "f" * 40,
        },
        "reference_id": None,
        "reference_bundle": None,
        "experiment_purpose": "training_objective_comparison",
        "intervention_group_id": "distillation-loss",
        "declared_intervention_fields": ["loss_identity"],
        "role_applicability_plan": {
            kind: ("applicable" if kind in CAUSAL_REQUIRED_ROLE_KINDS else "not_applicable")
            for kind in ROLE_EVENT_KINDS
        },
        "trace_plan": {
            field: field in {
                "optimizer_step",
                "sample_presentations",
                "epoch_or_pass",
                "learning_rate",
                "live_train_metric",
                "live_dev_metric",
                "checkpoint_identity",
            }
            for field in TRACE_FIELD_DECLARATIONS
        },
        "runtime_qualification_plan": {
            "status": "declared",
            "runtime_certificate_required": True,
            "qualification_scope": "candidate runtime tuple",
        },
    }

    record = assess_comparison_prelaunch(**arguments)
    assert record["experiment_purpose"] == "training_objective_comparison"
    assert record["declared_intervention_fields"] == ["loss_identity"]
    assert "REFERENCE_BUNDLE_UNAVAILABLE" in record["blocker_codes"]

    wrong_scope = dict(arguments, declared_intervention_fields=["architecture_config_identity"])
    with pytest.raises(ValueError, match="outside one logical group"):
        assess_comparison_prelaunch(**wrong_scope)


def test_generated_teacher_cache_output_is_cpu_finite_and_matches_source_row_hash():
    cache_dir = ROOT / "experiments" / "pcqm_k1_fusion_distillation" / "teacher_cache"
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        pytest.skip("Generated teacher cache is not present in this checkout")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    generation = json.loads(
        (cache_dir.parent / "teacher_generation.json").read_text(encoding="utf-8")
    )
    manifest_sha256 = sha256_file(manifest_path)
    assert manifest_sha256 == generation["manifest_sha256"]
    assert manifest["sha256"] == generation["payload_sha256"]

    config = {
        "weight": 0.1,
        "teacher_identity": manifest["teacher_identity"],
        "cache_manifest_sha256": manifest_sha256,
    }
    cache = K1TeacherCache(
        cache_dir,
        manifest_sha256=manifest_sha256,
        teacher_identity=manifest["teacher_identity"],
    )
    assert validate_teacher_config(config, "distill_weak") == config

    payload = torch.load(
        cache_dir / manifest["path"], map_location="cpu", weights_only=True
    )
    source_idx = payload["source_idx"]
    prediction = payload["prediction_eV"]
    assert source_idx.dtype == torch.int64
    assert source_idx.shape == (TRAIN_ROWS,)
    assert torch.equal(source_idx, torch.arange(TRAIN_ROWS, dtype=torch.int64))
    assert hashlib.sha256(source_idx.numpy().astype("<i8", copy=False).tobytes()).hexdigest() == (
        generation["source_idx_sha256"]
    )
    assert prediction.dtype == torch.float32
    assert prediction.shape == (TRAIN_ROWS,)
    assert torch.isfinite(prediction).all()
    assert torch.equal(cache.prediction_eV, prediction)
