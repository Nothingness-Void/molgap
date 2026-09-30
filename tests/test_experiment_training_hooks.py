"""Saved synthetic states only: no model construction, training or inference."""
import ast
import copy
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from molgap.experiment_family_workflow import FamilyOutputSession, close_family_replay, family_replay_status
from molgap.experiment_training_hooks import TrainingOutputHooks, acknowledged_order_digest, bind_training_outputs
from molgap.research_memory.trace import json_bytes
from test_experiment_family_workflow import (
    launch_contexts, repo, payload, _recipe, SOURCE_IDX, TARGET, METRIC_SEMANTICS,
    _closure_descriptor,
)


@pytest.mark.parametrize("family,adapter,trainer,variant", [
    ("gptrans_t", "gptrans-v1", "pcqm_gptrans_v4", "reference"),
    ("neural_atom_k1", "k1-v1", "pcqm_k1_variants_runner", "neural_atom_k1_v4"),
])
def test_native_events_resume_and_completion(tmp_path, launch_contexts, family, adapter, trainer, variant):
    context = launch_contexts[4][family]
    contract = tmp_path / "contract.json"
    contract.write_bytes(json_bytes(_recipe(family)))
    session = FamilyOutputSession(tmp_path / "family_outputs", context, adapter=adapter,
        contract=contract, trajectory_id="T-hook", metric_semantics=METRIC_SEMANTICS)
    hook = TrainingOutputHooks(session, trainer=trainer, variant=variant)
    start = dict(trainer=trainer, variant=variant, source_commit=context.source_commit,
                 source_archive_sha256=context.source_archive_sha256, epochs=2, steps=4,
                 samples=8, development_rows=3, trajectory_id="T-hook", native_output=tmp_path)
    hook.validate_start(**start)
    with pytest.raises(ValueError, match="exposure"):
        hook.validate_start(**{**start, "steps": 5})
    with pytest.raises(ValueError, match="cannot be overwritten"):
        hook.validate_start(**{**start, "native_output": session.root})
    hook.validate_resume(None, completed_epochs=0, optimizer_step=0, sample_presentations=0)
    native = {
        "model": {"w": torch.tensor([0.5])},
        "optimizer": {"state": {}, "param_groups": [{"params": [0], "lr": 0.001}]},
        "scheduler": {"last_epoch": 1},
        "rng_state": {"python": random.getstate(), "numpy": np.random.get_state(),
                      "torch": torch.get_rng_state(), "cuda": []},
        "family_output_binding": hook.binding,
    }
    if family == "gptrans_t":
        native["ema"] = copy.deepcopy(native["model"])
    for epoch in range(2):
        native["scheduler"]["last_epoch"] = epoch + 1
        hook.completed_epoch(epoch=epoch, optimizer_step=(epoch + 1) * 2,
            sample_presentations=(epoch + 1) * 4, train_mae_eV=1.1,
            live_dev_mae_eV=1.0, ema_dev_mae_eV=1.0 if family == "gptrans_t" else None,
            checkpoint=native, sampler_order_sha256=acknowledged_order_digest([2, 1, 0]),
            selected={"model_state": native["model"],
                      "weights": "ema" if family == "gptrans_t" else "live",
                      "prediction_eV": torch.ones(3, dtype=torch.float64),
                      "target_eV": TARGET, "source_idx": SOURCE_IDX})
        hook.validate_resume(native, completed_epochs=epoch + 1,
            optimizer_step=(epoch + 1) * 2, sample_presentations=(epoch + 1) * 4)
    with pytest.raises(ValueError, match="binding"):
        hook.validate_resume({}, completed_epochs=2, optimizer_step=4, sample_presentations=8)
    with pytest.raises(ValueError, match="prefix disagree"):
        hook.validate_resume(native, completed_epochs=1, optimizer_step=2, sample_presentations=4)
    altered = copy.deepcopy(native)
    altered["model"]["w"] += 1
    with pytest.raises(ValueError, match="state disagrees"):
        hook.validate_resume(altered, completed_epochs=2, optimizer_step=4, sample_presentations=8)
    for key in ("optimizer", "scheduler", "rng_state"):
        altered = copy.deepcopy(native)
        if key == "optimizer":
            altered[key]["param_groups"][0]["lr"] += 1
        elif key == "scheduler":
            altered[key]["last_epoch"] += 1
        else:
            altered[key]["torch"][0] ^= 1
        with pytest.raises(ValueError, match=key + " state disagrees"):
            hook.validate_resume(altered, completed_epochs=2, optimizer_step=4, sample_presentations=8)
    hook.archive_prefix(epoch=1)
    assert (tmp_path / "family_recovery_epoch_02.tar").is_file()
    report = hook.complete(hardware="synthetic-cpu")
    assert report["status"] == "MECHANICALLY_VERIFIED", report
    assert report["replay_readiness"] == "NOT_EVALUATED"
    observations = session.stage.recorder.record["observations"]
    assert [r["event"] for r in observations] == ["observation", "checkpoint"] * 2


def test_factory_rejects_native_directory_before_any_write(tmp_path):
    path = tmp_path / "last_checkpoint.pt"
    path.write_bytes(b"native-checkpoint-unchanged")
    with pytest.raises(ValueError, match="cannot be overwritten"):
        bind_training_outputs(None, package_dir=tmp_path, expected_package_identity="invalid",
            arm_id="unused", account="unused", run_reference="unused",
            output=tmp_path, native_output=tmp_path, contract=tmp_path / "recipe.json")
    assert path.read_bytes() == b"native-checkpoint-unchanged"
    assert not (tmp_path / "canonical_trace.json").exists()


def test_actual_trainers_have_opt_in_hooks_and_preserve_default():
    root = Path(__file__).resolve().parents[1] / "src/molgap"
    for filename, function in (("pcqm_gptrans_v4.py", "run_training"),
                               ("pcqm_k1_variants_runner.py", "train_arm")):
        tree = ast.parse((root / filename).read_text(encoding="utf-8"))
        trainer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
        index = next(i for i, n in enumerate(trainer.args.kwonlyargs) if n.arg == "family_outputs")
        assert isinstance(trainer.args.kw_defaults[index], ast.Constant)
        assert trainer.args.kw_defaults[index].value is None
        calls = {n.func.attr for n in ast.walk(trainer) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                 and n.func.value.id == "family_outputs"}
        assert {"validate_start", "validate_resume", "completed_epoch", "complete", "archive_prefix"} <= calls


def test_edge_state_profile_requires_live_scheduler_and_correct_family():
    from molgap.experiment_family_artifacts import artifact_adapter
    profile = artifact_adapter("edge-state-v1", ("edge_state_gps", "1"))
    assert profile.ema_required is False
    assert "scheduler" in profile.resume_keys
    with pytest.raises(ValueError, match="family mismatch"):
        artifact_adapter("edge-state-v1", ("neural_atom_k1", "1"))


def test_terminal_complete_is_not_automatically_replay_ready(tmp_path, repo, launch_contexts):
    spec, descriptor, outputs = _closure_descriptor(tmp_path, repo, launch_contexts)
    result = close_family_replay(repo, spec, descriptor, outputs=outputs, execute=True)
    assert result["status"] == "COMPLETE", result
    assert result["replay_readiness"] == "BLOCKED"
    assert len(result["replay_blockers"]) == 2


def test_dry_run_never_claims_replay_or_writes_terminal(tmp_path, repo, launch_contexts):
    spec, descriptor, outputs = _closure_descriptor(tmp_path, repo, launch_contexts)
    result = close_family_replay(repo, spec, descriptor, outputs=outputs)
    assert result["executed"] is False
    assert result["replay_readiness"] == "NOT_EVALUATED"
    assert not list((repo / "experiments").glob("*/rml_finalized"))


def test_compiled_qualified_pair_is_ready_only_after_both_acceptances(tmp_path):
    from test_same_run_replay import _prepare_arm, _strict_comparison, _put
    from test_terminal_trace_closure import setup_mock_repo, create_candidate_arm
    from molgap.research_memory.terminal_wiring import close_terminal_arm
    import json
    setup_mock_repo(tmp_path)
    reference = create_candidate_arm(tmp_path, "pair_ref", "T-pair-ref", "run-ref", "ev-pair-ref")
    candidate = create_candidate_arm(tmp_path, "pair_candidate", "T-pair-candidate", "run-candidate", "ev-pair-candidate")
    ref_trace = _prepare_arm(tmp_path, reference, reference, "reference")
    candidate_trace = _prepare_arm(tmp_path, candidate, reference, "candidate")
    ids = [reference["traj_id"], candidate["traj_id"]]
    assert family_replay_status(tmp_path, ids)["replay_readiness"] == "BLOCKED"
    close_terminal_arm(tmp_path, reference["traj_path"], reference["terminal_path"], trace=ref_trace)
    assert family_replay_status(tmp_path, ids)["replay_readiness"] == "BLOCKED"
    comparison_ref, digest = _strict_comparison(tmp_path, reference, candidate)
    terminal = json.loads(candidate["terminal_path"].read_bytes())
    terminal["comparison_readiness_ref"] = comparison_ref
    terminal["artifact_hashes"][comparison_ref] = digest
    _put(candidate["terminal_path"], terminal)
    close_terminal_arm(tmp_path, candidate["traj_path"], candidate["terminal_path"], trace=candidate_trace)
    assert family_replay_status(tmp_path, ids)["replay_readiness"] == "REPLAY_READY"
