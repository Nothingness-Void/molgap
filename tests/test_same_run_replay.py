"""A real same-family pair reaches replay only after its reference is accepted."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from molgap import modern2d_training as formal
from molgap.comparison_readiness import CAUSAL_REQUIRED_ROLE_KINDS, ROLE_EVENT_KINDS
from molgap.common.wire import canonical_fingerprint
from molgap.experiment_execution import MODERN2D_FORMAL_PROFILE, build_addon_declaration
from molgap.experiment_family_workflow import RunContext, family_replay_status, tensor_digest
from molgap.experiment_spec import ExperimentSpec, TERMINAL_PROTOCOL
from molgap.experiment_terminal_staging import stage_verified_terminal_packages
from molgap.family_pair_acceptance import register_session_reference
from molgap.family_training_outputs import SessionOutputs
from molgap.family_training_recipe import ScheduleSpec
from molgap.family_training_runner import build_screen_recipe, _probe_recipe
from molgap.modern2d_training_release import (
    comparison_identity,
    validate_retained_comparison_plan,
    validate_same_run_comparison_plan,
)
from molgap.research_memory.compiler import compile_research_memory
from molgap.research_memory.paired import PAIR_SCHEMA
from molgap.research_memory.paths import bound_artifact_path
from molgap.research_memory.terminal_wiring import close_terminal_arm
from molgap.research_memory.trace import file_digest, json_bytes

from test_experiment_spec import payload as spec_payload
from test_modern2d_formal_training import actual_session
from test_modern2d_integration_session import make_preview_session, preview_spec
from test_terminal_trace_closure import create_candidate_arm, setup_mock_repo


PROFILE = formal.Modern2DFormalTrainingProfile(
    "configured-screen", horizon_passes=2, horizon_steps=4, execution_mode="batched-v1"
)
SOURCE_COMMIT = "c" * 40
SOURCE_ARCHIVE = "d" * 64
PACKAGE_IDENTITY = "e" * 64


def _put(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def _index_digest(indices) -> str:
    return tensor_digest(torch.tensor(indices, dtype=torch.int64), role="source_idx")


def _make_recipe(fixture_root: Path):
    preview, _, data, initial = make_preview_session(fixture_root, "ppgt_small", "2", [])
    preview.close()
    base = replace(
        preview.recipe,
        family_version="3",
        formal_training=PROFILE,
        schedule=ScheduleSpec("cosine_step", 4, reference_epochs=2),
    )
    addon = build_addon_declaration(
        base.interface, "ppgt_research_mean_readout", repo_root=Path(__file__).resolve().parents[1]
    )
    candidate = replace(base, mode="composed", addons=(addon,))
    return base, candidate, data, initial


def _make_contract(root: Path, name: str, recipe, data):
    record = build_screen_recipe(
        recipe.mode,
        family=recipe.interface,
        configuration=recipe,
        train_role_identity="3" * 64,
        development_role_identity="4" * 64,
        source_idx_sha256=_index_digest(data.development_source_indices),
        target_sha256=tensor_digest(
            torch.cat([item.y for item in data.development]), role="target"
        ),
    )
    path = root / "experiments" / name / "contract.json"
    _put(path, record)
    return path, record


def _declaration(arm_id: str, role: str, recipe):
    declaration = preview_spec(spec_payload.__wrapped__(), recipe).to_dict()["arms"][0]
    declaration.update(arm_id=arm_id, scientific_role=role)
    declaration["addons"] = list(recipe.addons)
    declaration["addon_semantics"] = "ordered" if recipe.addons else "baseline"
    return declaration


def _make_arm(root: Path, name: str, arm_id: str, trajectory_id: str, run_id: str,
              role: str, recipe, data):
    scaffold = create_candidate_arm(root, name, trajectory_id, run_id, "ev-" + arm_id)
    contract, record = _make_contract(root, name, recipe, data)
    plan = root / "experiments" / name / "plan.json"
    _put(plan, {
        "format": "synthetic-formal-pair-plan-v1", "arm_id": arm_id,
        "trajectory_id": trajectory_id, "recipe_identity": recipe.identity,
    })
    declaration = _declaration(arm_id, role, recipe)
    declaration["training"]["recipe"]["sha256"] = file_digest(contract)
    return {
        "scaffold": scaffold, "name": name, "arm_id": arm_id,
        "trajectory_id": trajectory_id, "run_id": run_id, "role": role,
        "recipe": recipe, "data": data, "contract": contract, "record": record,
        "plan": plan, "declaration": declaration,
        "output_dir": scaffold["exp_dir"] / "output",
    }


def _spec(root: Path, *, experiment_id: str, logical_run_id: str, arms: list[dict], details,
          same_run: bool):
    payload = {
        "schema_version": "molgap-experiment-spec-v2",
        "experiment_id": experiment_id,
        "logical_run_id": logical_run_id,
        "arms": arms,
        "platform": {
            "name": "local", "accelerator": "CPU substitute", "device_count": 1,
            "cpu_cores": 1, "memory_gib": 2, "atomic_checkpoints": True,
            "retrievable_chunks": True,
        },
        "prospective": {
            "arms": [
                {
                    "arm_id": item["arm_id"],
                    "trajectory_id": item["trajectory_id"],
                    "plan_spec_ref": item["plan"].relative_to(root).as_posix(),
                    "plan_spec_sha256": file_digest(item["plan"]),
                    "output": item["output_dir"].relative_to(root).as_posix(),
                }
                for item in details
            ],
            **({"same_run_replay": {"reference_arm_id": "reference", "candidate_arm_ids": ["candidate"]}}
               if same_run else {}),
        },
        "evidence": {
            "policy": {"name": "molgap-v5", "version": "1", "sha256": "a" * 64},
            "required_artifacts": ["v5_evidence", "costs", "roles", "trace_manifest", "terminal_artifact"],
        },
        "terminal_protocol": TERMINAL_PROTOCOL,
    }
    return ExperimentSpec(payload)


def _context(spec: ExperimentSpec, item, *, run_reference: str):
    producer = RunContext(
        spec.to_dict()["experiment_id"], spec.to_dict()["logical_run_id"], item["arm_id"],
        canonical_fingerprint(item["declaration"]), spec.identity,
        item["recipe"].family, item["recipe"].family_version,
        SOURCE_COMMIT, SOURCE_ARCHIVE, PACKAGE_IDENTITY,
        file_digest(item["contract"]), "local", "fixture", run_reference, None,
    )
    return producer, replace(producer, platform_version="cpu-substitute-v1")


def _write_trajectory(root: Path, item, spec: ExperimentSpec, action_id: str, attempt_id: str,
                      *, reference_id: str | None = None):
    path = item["scaffold"]["traj_path"]
    trajectory = json.loads(path.read_bytes())
    trajectory["trajectory_id"] = item["trajectory_id"]
    state = trajectory["state_at_start"]
    state.update(
        source_commit=SOURCE_COMMIT,
        source_config_identity=canonical_fingerprint({"recipe_identity": item["recipe"].identity}),
        contract_refs=[item["contract"].relative_to(root).as_posix()],
        reference_ids=([reference_id] if reference_id is not None else []),
        prior_evidence_ids=([reference_id] if reference_id is not None else ["ev-ref-1"]),
    )
    if item["role"] in {"reference", "candidate"} and reference_id is None:
        state["same_run_replay"] = {
            "schema": PAIR_SCHEMA,
            "spec_identity": spec.identity,
            "logical_run_id": spec.to_dict()["logical_run_id"],
            "arm_id": item["arm_id"],
            "comparison_role": item["role"],
            "reference_arm_id": "reference",
            "reference_trajectory_id": "T-pair-reference",
            "reference_trajectory_ref": "experiments/pair_formal_reference/trajectory.json",
        }
    else:
        state.pop("same_run_replay", None)
    trajectory["actions"] = [{
        "action_id": action_id, "type": "TRAIN", "source_commit": SOURCE_COMMIT,
        "run_ids": [item["run_id"]], "attempt_ids": [attempt_id],
        "evidence_refs": [item["scaffold"]["exp_dir"].relative_to(root).as_posix() + "/acceptance.json"],
        "cost_event_ids": [],
    }]
    trajectory["decision"].update(
        outcome="ACTIVE", final=False,
        decision_ref=item["scaffold"]["exp_dir"].relative_to(root).as_posix() + "/decision.md",
        next_allowed_actions=["CLOSE"], reopen_conditions=[],
    )
    _put(path, trajectory)


def _run_session(item, producer_context: RunContext, *, action_id: str):
    case = SimpleNamespace(recipe=item["recipe"], data=item["data"], initial=item["initial"])
    session = actual_session(case, [])
    output = SessionOutputs(
        item["output_dir"], producer_context, contract=item["contract"],
        adapter=MODERN2D_FORMAL_PROFILE, trajectory_id=item["trajectory_id"],
        train_role_identity=item["record"]["train_role_identity"], action_id=action_id,
    )
    session.loaders.attach_role_audit(output.role_audit)
    session.checkpoint_builder = output.checkpoint_from_builder
    session.selected = output.selected
    session.pass_finished = output.pass_finished
    session.batch_finished = output.batch_finished
    output.logger.start(session.progress)
    endpoint = session.run()
    provenance = {
        "format": "molgap-runtime-provenance-v1",
        "context": producer_context.to_dict(),
        "recipe_identity": item["recipe"].identity,
        "runtime_fingerprint": canonical_fingerprint({
            "family": item["recipe"].family, "scope": "CPU_SUBSTITUTE_ONLY"
        }),
        "scope": "CPU_SUBSTITUTE_ONLY", "accelerator": "cpu",
        "native_execution_released": False,
    }
    certificate = {
        "format": "molgap-runtime-certificate-v1", "status": "accepted",
        "provenance_sha256": canonical_fingerprint(provenance),
        "probe_recipe_identity": _probe_recipe(item["recipe"]).identity,
        "formal_training_exposure": 0,
        "scope": "CPU_SUBSTITUTE_ONLY", "native_execution_released": False,
        "proofs": [],
    }
    (output.root / "runtime_provenance.json").write_bytes(json_bytes(provenance))
    (output.root / "runtime_certificate.json").write_bytes(json_bytes(certificate))
    report = output.complete(
        endpoint,
        runtime={
            "platform": producer_context.platform, "account": producer_context.account,
            "precision": "fp32", "source_commit": producer_context.source_commit,
            "source_archive_sha256": producer_context.source_archive_sha256,
            "runtime_certificate_id": canonical_fingerprint(certificate),
        },
        costs=[
            {"metric": "wall_seconds", "unit": "seconds", "value": 1.0,
             "status": "measured", "semantics": "process_wall", "hardware": "synthetic-cpu"},
            {"metric": "device_seconds", "unit": "seconds", "value": None,
             "status": "missing", "semantics": "allocated_device", "hardware": "synthetic-cpu",
             "reason": "CPU substitute has no allocated accelerator device time"},
        ],
    )
    assert report["status"] == "MECHANICALLY_VERIFIED", report
    assert report["observed"]["role_audit"]["path"] == "role_audit.json"
    assert json.loads((output.root / "role_audit.json").read_bytes())["role_use"]["official_validation"] == "untouched"
    session.close()
    return output, report


def _comparison(reference, candidate, baseline_recipe):
    baseline_identity, transform, contract_identity = comparison_identity(
        arm=reference["declaration"], recipe=reference["recipe"], baseline=baseline_recipe,
        record=reference["record"], metadata=None, target=None,
    )
    candidate_identity, candidate_transform, candidate_contract = comparison_identity(
        arm=candidate["declaration"], recipe=candidate["recipe"], baseline=baseline_recipe,
        record=candidate["record"], metadata=None, target=None,
    )
    assert candidate_transform == transform
    assert candidate_contract == contract_identity
    assert candidate_identity["architecture_config_identity"] != baseline_identity["architecture_config_identity"]
    role_plan = {
        kind: ("applicable" if kind in CAUSAL_REQUIRED_ROLE_KINDS else "not_applicable")
        for kind in ROLE_EVENT_KINDS
    }
    trace_plan = {
        "optimizer_step": True, "sample_presentations": True, "epoch_or_pass": True,
        "learning_rate": True, "live_train_metric": True, "live_dev_metric": True,
        "ema_dev_metric": False, "checkpoint_identity": True,
    }
    return {
        "experiment_purpose": "architecture_comparison",
        "intervention_group_id": "ppgt-readout-addon",
        "declared_intervention_fields": ["architecture_config_identity"],
        "contract_identity": contract_identity,
        "role_applicability_plan": role_plan,
        "trace_plan": trace_plan,
        "runtime_qualification_plan": {
            "status": "declared", "runtime_certificate_required": True,
            "qualification_scope": "CPU_SUBSTITUTE_ONLY",
        },
        "target_transform_asset": transform,
        "identities": {"reference": baseline_identity, "candidate": candidate_identity},
    }, baseline_identity, candidate_identity


def _stage_and_complete(root, spec, details, contexts, locations, comparison, *, run_id, attempt_id):
    outputs = {
        item["arm_id"]: {
            "output_dir": str(item["output_dir"]),
            "expected": item["record"]["acceptance_requirements"],
            "context": contexts[item["arm_id"]],
        }
        for item in details
    }
    staged = stage_verified_terminal_packages(
        None, root, run_id, attempt_id,
        outcome={item["arm_id"]: "NEGATIVE_UNDER_CONTRACT" for item in details},
        decision_ref={
            item["arm_id"]: item["scaffold"]["exp_dir"].relative_to(root).as_posix() + "/decision.md"
            for item in details
        },
        resolved={
            "spec": spec, "outputs": outputs, "locations": locations,
            "acceptance_plan": {"comparison": comparison},
        },
    )
    assert staged["status"] == "STAGED_CLOSURE_READY", staged
    return staged, outputs


def _locations(root: Path, item):
    output_trace = item["output_dir"] / "canonical_trace.json"
    return {
        "trajectory_id": item["trajectory_id"], "run_id": item["run_id"],
        "trajectory": item["scaffold"]["traj_path"].relative_to(root).as_posix(),
        "terminal": item["scaffold"]["terminal_path"].relative_to(root).as_posix(),
        "trace": output_trace.relative_to(root).as_posix(),
    }


def _snapshot(paths):
    return {str(path): path.read_bytes() for path in paths}


def test_same_run_pair_reaches_replay_only_after_reference_acceptance(tmp_path):
    setup_mock_repo(tmp_path)
    fixture_root = tmp_path / "formal_fixtures"
    fixture_root.mkdir()
    base_recipe, candidate_recipe, data, initial = _make_recipe(fixture_root)

    reference = _make_arm(
        tmp_path, "pair_formal_reference", "reference", "T-pair-reference",
        "formal-pair-run:reference:downstream", "reference", base_recipe, data,
    )
    candidate = _make_arm(
        tmp_path, "pair_formal_candidate", "candidate", "T-pair-candidate",
        "formal-pair-run:candidate:downstream", "candidate", candidate_recipe, data,
    )
    for item in (reference, candidate):
        item["initial"] = initial
    spec = _spec(
        tmp_path, experiment_id="formal-pair-question", logical_run_id="formal-pair-run",
        arms=[reference["declaration"], candidate["declaration"]],
        details=[reference, candidate], same_run=True,
    )
    reference["declaration"] = spec.to_dict()["arms"][0]
    candidate["declaration"] = spec.to_dict()["arms"][1]
    _write_trajectory(tmp_path, reference, spec, "action-formal-pair", "attempt-formal-pair-v1")
    _write_trajectory(tmp_path, candidate, spec, "action-formal-pair", "attempt-formal-pair-v1")

    reference_context, reference_physical = _context(
        spec, reference, run_reference="fixture/formal-pair"
    )
    candidate_context, candidate_physical = _context(
        spec, candidate, run_reference="fixture/formal-pair"
    )
    reference["output"], reference["report"] = _run_session(
        reference, reference_context, action_id="action-formal-pair"
    )
    candidate["output"], candidate["report"] = _run_session(
        candidate, candidate_context, action_id="action-formal-pair"
    )
    contexts = {"reference": reference_physical, "candidate": candidate_physical}
    locations = {item["arm_id"]: _locations(tmp_path, item) for item in (reference, candidate)}
    comparison, baseline_identity, candidate_identity = _comparison(
        reference, candidate, base_recipe
    )
    frozen_plan = {
        "comparison": comparison,
        "arms": [
            {"arm_id": item["arm_id"], "expected": item["record"]["acceptance_requirements"]}
            for item in (reference, candidate)
        ],
    }
    assert validate_same_run_comparison_plan(spec, frozen_plan) == comparison
    staged, _ = _stage_and_complete(
        tmp_path, spec, [reference, candidate], contexts, locations, comparison,
        run_id="formal-pair-run", attempt_id="attempt-formal-pair-v1",
    )
    readiness_path = Path(staged["paths"]["candidate"]["terminal"]).parent / "comparison_readiness.json"
    readiness = json.loads(readiness_path.read_bytes())
    assert readiness["comparison_class"] == "STRICT_CAUSAL"
    assert readiness["strict_ready"] is True
    assert json.loads((reference["output"].root / "runtime_certificate.json").read_bytes())["scope"] == "CPU_SUBSTITUTE_ONLY"

    reference_where, candidate_where = staged["locations"]["reference"], staged["locations"]["candidate"]
    with pytest.raises(ValueError, match="accepted"):
        close_terminal_arm(
            tmp_path, candidate_where["trajectory"], candidate_where["terminal"],
            trace=candidate_where["trace"],
        )
    assert close_terminal_arm(
        tmp_path, reference_where["trajectory"], reference_where["terminal"],
        trace=reference_where["trace"],
    )["pipeline_status"] == "COMPLETE"
    # Publish the accepted control before the candidate's finalization.  This
    # is the reference-first boundary that makes retained comparison reusable.
    registered = register_session_reference(tmp_path, spec, staged)
    assert registered and (tmp_path / registered["path"]).is_file()
    assert close_terminal_arm(
        tmp_path, candidate_where["trajectory"], candidate_where["terminal"],
        trace=candidate_where["trace"],
    )["pipeline_status"] == "COMPLETE"

    reference_bundle_path = tmp_path / registered["path"]
    reference_bundle_before = reference_bundle_path.read_bytes()
    reference_bundle = json.loads(reference_bundle_before)
    reference_paths = [
        reference["output"].root / name
        for name in (
            "output_manifest.json", "training_contract.json", "selected_model.pt",
            "last_checkpoint.pt", "development_predictions.pt", "canonical_trace.json",
            "canonical_trace.json.context.json", "role_audit.json",
            "runtime_certificate.json", "runtime_provenance.json",
        )
    ]
    reference_paths += [
        reference["scaffold"]["traj_path"],
        reference["scaffold"]["terminal_path"],
        bound_artifact_path(tmp_path, reference_bundle["checkpoint_ref"]),
    ]
    finalized = reference["scaffold"]["exp_dir"] / "rml_finalized"
    reference_paths.extend(path for path in finalized.iterdir() if path.is_file())
    reference_bytes_before = _snapshot(reference_paths)

    replay = json.loads(compile_research_memory(tmp_path)["replay_pool.json"])
    first_entries = [
        entry for entry in replay["entries"]
        if entry["trajectory_id"] in {reference["trajectory_id"], candidate["trajectory_id"]}
    ]
    assert len(first_entries) == 2
    assert {entry["comparison_role"] for entry in first_entries} == {"reference", "candidate"}
    assert {entry["terminal_outcome"] for entry in first_entries} == {"NEGATIVE_UNDER_CONTRACT"}
    assert family_replay_status(
        tmp_path, [reference["trajectory_id"], candidate["trajectory_id"]]
    )["replay_readiness"] == "REPLAY_READY"

    candidate2 = _make_arm(
        tmp_path, "pair_formal_candidate_2", "candidate2", "T-pair-candidate-2",
        "formal-pair-candidate-2:candidate2:downstream", "candidate", candidate_recipe, data,
    )
    candidate2["initial"] = initial
    candidate2["contract"].write_bytes(candidate["contract"].read_bytes())
    candidate2["record"] = copy.deepcopy(candidate["record"])
    candidate2["declaration"] = copy.deepcopy(candidate["declaration"])
    candidate2["declaration"]["arm_id"] = "candidate2"
    candidate2_plan = candidate2["plan"]
    _put(candidate2_plan, {
        "format": "synthetic-formal-retained-candidate-plan-v1",
        "arm_id": "candidate2", "trajectory_id": candidate2["trajectory_id"],
        "recipe_identity": candidate_recipe.identity,
    })
    spec2 = _spec(
        tmp_path, experiment_id="formal-retained-candidate-question",
        logical_run_id="formal-pair-candidate-2", arms=[candidate2["declaration"]],
        details=[candidate2], same_run=False,
    )
    candidate2["declaration"] = spec2.to_dict()["arms"][0]
    _write_trajectory(
        tmp_path, candidate2, spec2, "action-formal-retained-candidate-2",
        "attempt-formal-retained-candidate-2", reference_id=registered["reference_id"],
    )
    candidate2_context, candidate2_physical = _context(
        spec2, candidate2, run_reference="fixture/formal-pair-candidate-2"
    )
    candidate2["output"], candidate2["report"] = _run_session(
        candidate2, candidate2_context, action_id="action-formal-retained-candidate-2"
    )
    comparison2 = {
        **{key: comparison[key] for key in (
            "experiment_purpose", "intervention_group_id",
            "declared_intervention_fields", "contract_identity",
            "role_applicability_plan", "trace_plan", "runtime_qualification_plan",
            "target_transform_asset",
        )},
        "reference_id": registered["reference_id"],
        "reference_bundle": {"path": registered["path"], "sha256": registered["sha256"]},
        "reference_identity": baseline_identity,
        "identities": {"candidate2": candidate_identity},
    }
    retained_plan = {
        "comparison": comparison2,
        "arms": [{"arm_id": "candidate2", "expected": candidate2["record"]["acceptance_requirements"]}],
    }
    assert validate_retained_comparison_plan(spec2, retained_plan, tmp_path) == comparison2
    candidate2_locations = {"candidate2": _locations(tmp_path, candidate2)}
    staged2, _ = _stage_and_complete(
        tmp_path, spec2, [candidate2], {"candidate2": candidate2_physical},
        candidate2_locations, comparison2,
        run_id="formal-pair-candidate-2", attempt_id="attempt-formal-retained-candidate-2",
    )
    candidate2_where = staged2["locations"]["candidate2"]
    assert close_terminal_arm(
        tmp_path, candidate2_where["trajectory"], candidate2_where["terminal"],
        trace=candidate2_where["trace"],
    )["pipeline_status"] == "COMPLETE"

    status = family_replay_status(
        tmp_path, [reference["trajectory_id"], candidate["trajectory_id"], candidate2["trajectory_id"]]
    )
    assert status["replay_readiness"] == "REPLAY_READY"
    replay = json.loads(compile_research_memory(tmp_path)["replay_pool.json"])
    entries = [
        entry for entry in replay["entries"]
        if entry["trajectory_id"] in {
            reference["trajectory_id"], candidate["trajectory_id"], candidate2["trajectory_id"]
        }
    ]
    assert len(entries) == 3
    assert sum(entry["comparison_role"] == "reference" for entry in entries) == 1
    assert sum(entry["comparison_role"] == "candidate" for entry in entries) == 2
    assert all(entry["terminal_outcome"] == "NEGATIVE_UNDER_CONTRACT" for entry in entries)
    assert all(entry["reference_id"] == registered["reference_id"] for entry in entries)
    assert reference_bundle_path.read_bytes() == reference_bundle_before
    assert _snapshot(reference_paths) == reference_bytes_before
