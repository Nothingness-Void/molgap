"""Synthetic GPTrans adapter checks; never read benchmark roles or run a GPU."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import molgap.gptrans_screen_workflow as workflow
from molgap.experiment_family_workflow import RunContext, tensor_digest
from molgap.research_memory.trace import file_digest, load_canonical_trace
from molgap.training_reproducibility import capture_rng_state


@pytest.fixture
def context():
    return RunContext("screen", "run", "candidate", "a" * 64, "b" * 64,
        "gptrans_t", "1", "c" * 40, "d" * 64, "e" * 64, "f" * 64,
        "local", "account", "account/screen", None)


def _request(tmp_path, context, arm=None):
    if arm is None:
        arm = {"arm_id": "candidate", "initialization": {"kind": "frozen_state",
            "seed": workflow.owner.SEED, "state_sha256": workflow.owner.EXPECTED_INITIAL_MODEL_SHA256},
            "training": {"overrides": {}}, "addons": []}
    spec = SimpleNamespace(to_dict=lambda: {"arms": [arm], "prospective": {"arms": []}})
    return dict(spec=spec, package_dir=tmp_path / "package", expected_package_identity="e" * 64,
        arm_id="candidate", mode="reference", recipe_path=tmp_path / "recipe.json",
        initial_state_path=tmp_path / "initial.pt", input_root=tmp_path / "input",
        account="account", run_reference="account/screen", trajectory_id="trajectory")


@pytest.mark.parametrize("change", ["family", "initialization", "overrides", "addons", "mode"])
def test_identity_rejected_before_recipe_or_data(tmp_path, monkeypatch, context, change):
    request = _request(tmp_path, context)
    if change == "family":
        context = replace(context, family_name="neural_atom_k1")
    elif change == "mode":
        request["mode"] = "unsupported"
    else:
        arm = request["spec"].to_dict()["arms"][0]
        if change == "initialization":
            arm["initialization"]["kind"] = "random"
        elif change == "overrides":
            arm["training"]["overrides"] = {"epochs": 1}
        else:
            arm["addons"] = [{"name": "centered_logits"}]
    monkeypatch.setattr(workflow.RunContext, "for_training", lambda *a, **k: context)
    with pytest.raises(ValueError):
        workflow._validate_request(**request)


def test_recipe_byte_identity_rejected_before_package(tmp_path, monkeypatch, context):
    request = _request(tmp_path, context)
    request["recipe_path"].write_text("{}", encoding="utf-8")
    monkeypatch.setattr(workflow.RunContext, "for_training", lambda *a, **k: context)
    with pytest.raises(ValueError, match="recipe bytes"):
        workflow._validate_request(**request)


@pytest.fixture
def legacy(tmp_path, monkeypatch, context):
    monkeypatch.setattr(workflow.owner, "BATCHES_PER_EPOCH", 2)
    monkeypatch.setattr(workflow.owner, "PHYSICAL_BATCH", 2)
    root = tmp_path / "legacy"
    root.mkdir()
    target = torch.tensor([1., 2.], dtype=torch.float64)
    idx = torch.tensor([100001, 100002], dtype=torch.int64)
    recipe = {"acceptance_requirements": {"epochs": 2, "optimizer_steps": 4,
        "sample_presentations": 8, "development_rows": 2, "precision": "fp32",
        "source_idx_sha256": tensor_digest(idx, role="source_idx"),
        "target_sha256": tensor_digest(target, role="target")},
        "development_role_identity": workflow.DEV_ROLE, "row_order_fingerprint": "1" * 64,
        "metric_semantics": workflow._screen_metric_semantics()}
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
    context = replace(context, training_recipe_sha256=file_digest(recipe_path))
    rows = [{"epoch": i, "optimizer_steps": 2, "sample_presentations": 4,
             "train_mae_eV": 0.5 - i * 0.1, "development_mae_eV": 0.25 - i * 0.125,
             "learning_rate": 0.001, "elapsed_seconds": 1.} for i in range(2)]
    (root / "trace.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")
    state = {"weight": torch.tensor([2.])}
    torch.save({"model": state, "epoch": 1, "source_archive_sha256": context.source_archive_sha256},
               root / "best_model.pt")
    flags = {k: False for k in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read")}
    torch.save({"prediction_eV": target + 0.125, "target_eV": target, "source_idx": idx, **flags},
               root / "development_predictions.pt")
    torch.save({"format": workflow.owner.CHECKPOINT_FORMAT, "source_archive_sha256": context.source_archive_sha256,
        "model": state, "ema": state, "optimizer": {"state": {}, "param_groups": [{"lr": .001}]},
        "scheduler": {"last_epoch": 2}, "rng_state": capture_rng_state(), "epoch": 1, "trace": rows},
        root / "last_checkpoint.pt")
    completion = {"complete": True, "best_epoch": 1,
        "source_archive_sha256": context.source_archive_sha256, "source_commit": context.source_commit,
        "checkpoint_sha256": file_digest(root / "last_checkpoint.pt"),
        "best_model_sha256": file_digest(root / "best_model.pt"),
        "development_predictions_sha256": file_digest(root / "development_predictions.pt"), **flags}
    return root, context, recipe_path, completion


def test_normalization_retains_ema_semantics_and_original_files(legacy):
    root, context, recipe_path, completion = legacy
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    report = workflow._normalize(root, context, recipe_path, "trajectory", completion,
        process_wall_seconds=3., certificate={"accelerator": "synthetic CPU fixture"})
    assert report["status"] == "MECHANICALLY_VERIFIED", report
    assert all((root / name).read_bytes() == payload for name, payload in before.items())
    trace = load_canonical_trace(root / "normalized" / "canonical_trace.json")
    assert trace["metric_semantics"]["live_dev_metric"] is None
    assert all(row["live_dev_metric"] is None for row in trace["observations"])
    assert [r["optimizer_step"] for r in trace["observations"]] == [2, 4]
    assert [r["epoch_or_pass"] for r in trace["observations"]] == [1, 2]
    assert report["observed"]["selected_weights"] == "ema"
    assert report["observed"]["costs"][1]["status"] == "missing"
    checkpoint = torch.load(root / "normalized" / "last_checkpoint.pt", weights_only=True)
    assert checkpoint["cursor"] == {"epoch": 2, "next_batch": 0, "sampler_order_sha256": "1" * 64}


def test_raw_output_tampering_blocks_normalization(legacy):
    root, context, recipe, completion = legacy
    with (root / "best_model.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="output hash"):
        workflow._normalize(root, context, recipe, "trajectory", completion,
            process_wall_seconds=1., certificate={"accelerator": "synthetic"})


def test_protected_role_flag_blocks_translation(legacy):
    root, context, recipe, completion = legacy
    completion["official_validation_role_read"] = True
    with pytest.raises(ValueError, match="protected roles"):
        workflow._normalize(root, context, recipe, "trajectory", completion,
            process_wall_seconds=1., certificate={"accelerator": "synthetic"})


def test_preflight_delegates_output_directory_unchanged(tmp_path, monkeypatch, context):
    request = _request(tmp_path, context)
    monkeypatch.setattr(workflow, "_validate_request", lambda **k: (context, {}))
    monkeypatch.setattr(workflow, "_owner_kwargs", lambda *a, **k: {"output": k["output"]})
    seen = {}
    def preflight(**kwargs):
        seen.update(kwargs)
        return {"accepted": True}
    monkeypatch.setattr(workflow.owner, "run_preflight", preflight)
    output = tmp_path / "preflight"
    assert workflow.run_screen_preflight(**request, output=output) == {"accepted": True}
    assert seen == {"output": output}


def test_owner_training_receives_real_preflight_json_path(tmp_path, monkeypatch, context):
    request = _request(tmp_path, context)
    recipe = {"acceptance_requirements": {}}
    monkeypatch.setattr(workflow, "_validate_request", lambda **k: (context, recipe))
    monkeypatch.setattr(workflow, "_owner_kwargs", lambda *a, **k: {"variant": k["mode"], "output": k["output"]})
    preflight = tmp_path / "preflight"
    preflight.mkdir()
    (preflight / "workflow_context.json").write_text(json.dumps(context.to_dict()), encoding="utf-8")
    (preflight / "runtime_certificate.json").write_text('{"accelerator":"fixture"}', encoding="utf-8")
    seen = {}
    def training(**kwargs):
        seen.update(kwargs)
        return {"complete": True}
    monkeypatch.setattr(workflow.owner, "run_training", training)
    monkeypatch.setattr(workflow, "_normalize", lambda *a, **k: {"status": "synthetic"})
    result = workflow.run_screen_arm(**request, output=tmp_path / "outputs", preflight_dir=preflight)
    assert seen["preflight_path"] == preflight / "preflight.json"
    assert result == {"status": "synthetic"}
