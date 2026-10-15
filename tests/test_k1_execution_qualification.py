"""CPU-only checks for the K1 candidate/native execution qualification gate."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest
import torch

from molgap.k1_execution_qualification import (
    _execution_state_fingerprint,
    _execution_compare_state,
    _verify_execution_proof,
    _write_execution_proof,
)
from molgap.experiment_family_workflow import tensor_digest, tensor_safe_rng_state
from molgap.family_training_data import PassBatchSampler
from molgap.k1_execution_backend import K1ExecutionConfig
from molgap.family_training_recipe import ScheduleSpec
from molgap.family_training_runner import normalized_file_sha
from molgap.training_reproducibility import capture_rng_state
from test_family_training_recipe import recipe


def _state(recipe_value, identity, value=1.0, *, cuda=True):
    tensor = torch.tensor([value], dtype=torch.float32)
    step = 2
    plan = recipe_value.loop_plan
    _, offset = divmod(step, plan.steps_per_pass)
    rng = tensor_safe_rng_state(capture_rng_state())
    rng["cuda"] = [torch.zeros(8, dtype=torch.uint8)] if cuda else []
    return {
        "model": {"weight": tensor},
        "optimizer": {"state": {0: {"step": torch.tensor(2.),
                                      "exp_avg": tensor.clone(),
                                      "exp_avg_sq": tensor.clone()}},
                       "param_groups": [{"params": [0], "betas": (.9, .999),
                                          "eps": 1e-8, "weight_decay": recipe_value.weight_decay,
                                          "amsgrad": False, "maximize": False,
                                          "capturable": False, "differentiable": False,
                                          "foreach": None, "fused": None,
                                          "lr": 1e-6, "initial_lr": recipe_value.learning_rate}]},
        "scheduler": {"completed_units": 2,
                       "native": {"T_max": 2, "eta_min": 1e-6,
                                   "base_lrs": [recipe_value.learning_rate],
                                   "last_epoch": 2, "_step_count": 3,
                                   "_last_lr": [1e-6]}},
        "ema": None,
        "rng_state": rng,
        "cursor": {"recipe_identity": identity, "pass_index": 0,
                    "next_batch": offset, "status": "complete",
                    "pending_absolute": [torch.tensor(1.), torch.tensor(1.)],
                    "pending_rows": offset * recipe_value.batch_size,
                    "best_development_mae_eV": None, "selected_step": step,
                    "optimizer_state_ids": [0]},
        "optimizer_step": 2,
        "sample_presentations": offset * recipe_value.batch_size,
    }


def test_shadow_comparison_removes_only_cursor_recipe_identity():
    value = replace(recipe(development=False), optimizer_steps=2,
                    checkpoint_every=1, schedule=ScheduleSpec("cosine_step", 2))
    left = _state(value, "candidate")
    right = _state(value, "native")
    assert _execution_state_fingerprint(left) == _execution_state_fingerprint(right)
    compared = _execution_compare_state(left)
    assert "recipe_identity" not in compared["cursor"]
    assert torch.equal(compared["model"]["weight"], left["model"]["weight"])
    changed = _state(value, "native", value=2.0)
    assert _execution_state_fingerprint(left) != _execution_state_fingerprint(changed)


def _proof_fixture(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    probe = replace(recipe(development=False), optimizer_steps=2,
        checkpoint_every=1, schedule=ScheduleSpec("cosine_step", 2),
        k1_execution=K1ExecutionConfig(
        aggregation="segmented", pooling="segmented", projection="edge"))
    candidate = _state(probe, probe.identity, value=1.0)
    native = _state(probe, replace(probe, k1_execution=None).identity, value=1.0)
    candidate_path = tmp_path / "repeat-b.pt"
    native_path = tmp_path / "execution-native-state.pt"
    torch.save(candidate, candidate_path)
    torch.save(native, native_path)
    candidate_eval = torch.arange(128, dtype=torch.float32)
    native_eval = candidate_eval.clone()
    sampler = PassBatchSampler(probe.train_rows, probe.batch_size, probe.seed,
        algorithm=probe.sampler, pass_index=0, start_batch=0, drop_last=probe.drop_last)
    source = torch.tensor(next(iter(sampler)), dtype=torch.long)
    proof = _write_execution_proof(
        output=tmp_path,
        probe=probe,
        candidate_state=candidate,
        candidate_path=candidate_path,
        native_state=native,
        native_path=native_path,
        candidate_eval=candidate_eval,
        candidate_eval_path=tmp_path / "execution-candidate-evaluation.pt",
        native_eval=native_eval,
        native_eval_path=tmp_path / "execution-native-evaluation.pt",
        source_idx=source,
        source_idx_path=tmp_path / "execution-evaluation-source-idx.pt",
    )
    return probe, proof


def test_execution_proof_rechecks_cpu_artifacts_and_bindings(tmp_path):
    probe, proof = _proof_fixture(tmp_path)
    assert _verify_execution_proof(tmp_path, probe, proof, required=True) == proof
    with pytest.raises(ValueError, match="retained repeat-b proof"):
        _verify_execution_proof(tmp_path, probe, proof, required=True,
                                expected_candidate_sha256="0" * 64)

    wrong_binding = deepcopy(proof)
    wrong_binding["config"]["projection"] = "node"
    with pytest.raises(ValueError, match="binding"):
        _verify_execution_proof(tmp_path, probe, wrong_binding, required=True)

    wrong_state = deepcopy(proof)
    wrong_state["candidate_state"]["fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint"):
        _verify_execution_proof(tmp_path, probe, wrong_state, required=True)

    cursor_root = tmp_path / "cursor"
    cursor_probe, cursor_proof = _proof_fixture(cursor_root)
    wrong_cursor = deepcopy(cursor_proof)
    candidate_path = cursor_root / "repeat-b.pt"
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=True)
    candidate["cursor"]["recipe_identity"] = "f" * 64
    torch.save(candidate, candidate_path)
    wrong_cursor["candidate_state"]["sha256"] = normalized_file_sha(candidate_path)
    wrong_cursor["candidate_state"]["fingerprint"] = _execution_state_fingerprint(candidate)
    with pytest.raises(ValueError, match="cursor recipe identity"):
        _verify_execution_proof(cursor_root, cursor_probe, wrong_cursor, required=True)

    rng_root = tmp_path / "rng"
    rng_probe, rng_proof = _proof_fixture(rng_root)
    wrong_rng = deepcopy(rng_proof)
    native_path = rng_root / "execution-native-state.pt"
    native = torch.load(native_path, map_location="cpu", weights_only=True)
    native["rng_state"]["cuda"] = []
    torch.save(native, native_path)
    wrong_rng["native_state"]["sha256"] = normalized_file_sha(native_path)
    wrong_rng["native_state"]["fingerprint"] = _execution_state_fingerprint(native)
    with pytest.raises(ValueError, match="CUDA state"):
        _verify_execution_proof(rng_root, rng_probe, wrong_rng, required=True)

    eval_root = tmp_path / "eval"
    eval_probe, eval_proof = _proof_fixture(eval_root)
    wrong_eval = deepcopy(eval_proof)
    eval_path = eval_root / "execution-candidate-evaluation.pt"
    torch.save(torch.tensor([0.25] * 128), eval_path)
    wrong_eval["evaluation"]["candidate"]["sha256"] = normalized_file_sha(eval_path)
    # The tensor fingerprint is intentionally the generic tensor fingerprint;
    # the output mismatch is the gate under test.
    from molgap.family_training_runner import _state_fingerprint
    wrong_eval["evaluation"]["candidate"]["fingerprint"] = _state_fingerprint(torch.tensor([0.25] * 128))
    with pytest.raises(ValueError, match="evaluation candidate/native"):
        _verify_execution_proof(eval_root, eval_probe, wrong_eval, required=True)

    source_root = tmp_path / "source"
    source_probe, source_proof = _proof_fixture(source_root)
    wrong_source = deepcopy(source_proof)
    source_path = source_root / "execution-evaluation-source-idx.pt"
    bad_source = torch.arange(128, dtype=torch.long)
    torch.save(bad_source, source_path)
    wrong_source["evaluation"]["source_idx"]["sha256"] = normalized_file_sha(source_path)
    wrong_source["evaluation"]["source_idx"]["fingerprint"] = _state_fingerprint(bad_source)
    wrong_source["evaluation"]["source_idx_sha256"] = tensor_digest(bad_source, role="source_idx")
    with pytest.raises(ValueError, match="frozen first training batch"):
        _verify_execution_proof(source_root, source_probe, wrong_source, required=True)


def test_unconfigured_recipe_rejects_stale_execution_proof(tmp_path):
    probe = recipe()
    with pytest.raises(ValueError, match="Unconfigured"):
        _verify_execution_proof(tmp_path, probe, {"stale": True}, required=False)
    assert _verify_execution_proof(tmp_path, probe, None, required=True) is None


def test_configured_recipe_requires_execution_proof(tmp_path):
    probe, _ = _proof_fixture(tmp_path)
    with pytest.raises(ValueError, match="lacks native execution proof"):
        _verify_execution_proof(tmp_path, probe, None, required=True)


@pytest.mark.parametrize("status", [None, "rejected", "running"])
def test_preflight_rejects_nonaccepted_certificate_before_artifact_loading(tmp_path, status):
    from molgap.family_training_runner import _verify_preflight
    from molgap.screen_policy import canonical_fingerprint
    provenance = {"fixture": "cpu-retained-proof"}
    certificate = {"provenance_sha256": canonical_fingerprint(provenance)}
    if status is not None:
        certificate["status"] = status
    (tmp_path / "runtime_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    (tmp_path / "runtime_certificate.json").write_text(json.dumps(certificate), encoding="utf-8")
    with pytest.raises(ValueError, match="certificate is not accepted"):
        _verify_preflight(tmp_path, recipe(), provenance)
