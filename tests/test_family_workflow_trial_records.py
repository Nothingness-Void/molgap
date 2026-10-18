"""Host metadata adapter tests; no model/data loading, execution or platform API.

The synthetic owner snapshot fixture tests translation and byte revalidation,
not native qualification or checkpoint correctness. Those remain the existing
output inspector and the independently pinned host native-verification report.
"""
import copy
import json
from pathlib import Path

import pytest

from molgap import family_workflow_trial_records as records
from molgap.common.wire import canonical_fingerprint
from molgap.experiment_family_workflow import RunContext
from molgap.experiment_inspection import InspectionSnapshot
from molgap.experiment_spec import ExperimentSpec
from molgap.research_memory.finalize import finalize
from molgap.research_memory.trace import canonicalize_trace, file_digest, json_bytes
from molgap.v5_common import validate_v5_evidence_envelope

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else json_bytes(value))
    return file_digest(path)


@pytest.fixture
def trial(tmp_path, monkeypatch):
    from test_terminal_trace_closure import setup_mock_repo

    root, run, inspection = (tmp_path / p for p in ("repo", "run", "inspection"))
    root.mkdir()
    setup_mock_repo(root)
    for ref in ("experiment_spec.json", "acceptance_plan.json", "authority.json", "role_plan.json",
                "k1/contract.json", "gptrans/contract.json"):
        save(root / records.BASE / ref, (ROOT / records.BASE / ref).read_bytes())
    declaration = json.loads((root / records.BASE / "experiment_spec.json").read_bytes())
    spec = ExperimentSpec(declaration)
    save(run / "inputs/experiment_spec.json", declaration)
    save(run / "operations/launch_receipt.json", {})
    report = dict(format=records.REPORT_FORMAT, verified=True, source_commit=records.SOURCE,
        spec_identity=records.SPEC, package_identity=records.PACKAGE,
        source_archive_sha256=records.ARCHIVE, arms={})
    snapshots, contexts = {}, {}
    reference = json.loads((root / "experiments/ref_exp/trajectory.json").read_bytes())
    for arm in declaration["arms"]:
        name = arm["arm_id"]
        context = RunContext(declaration["experiment_id"], declaration["logical_run_id"], name,
            canonical_fingerprint(arm), spec.identity, arm["family"]["name"], arm["family"]["version"],
            records.SOURCE, records.ARCHIVE, records.PACKAGE, arm["training"]["recipe"]["sha256"],
            "kaggle", "kaseichou", "kaseichou/molgap-family-workflow-trial-100k-v1", "1")
        contexts[name] = context
        producer = {**context.to_dict(), "platform_version": None}
        tid = f"TC-family-workflow-trial-{name}-100k-s42-v1"
        run_id = f"{context.logical_run_id}:{name}:downstream"
        frozen = copy.deepcopy(reference)
        frozen.update(owner="server", record_mode="prospective", track="C", trajectory_id=tid)
        frozen["hypothesis"].update(hypothesis_id="H-" + tid, supporting_evidence_ids=["ev-ref-1"],
            alternative_explanations=["startup"], related_closed_family_ids=["F-ref"], historical_unknowns=[],
            observed_deficiency="Unverified execution", changed_mechanism="front door",
            cheapest_falsifier="32 updates", expected_native_cost_ref="cost-test-1",
            decision_changed_if_positive="accept execution", decision_changed_if_negative="do not retry")
        frozen["state_at_start"].update(source_commit=records.SOURCE,
            source_config_identity=canonical_fingerprint(arm), contract_refs=[f"{records.BASE}/{name}/contract.json"],
            budget_snapshot_ref=f"{records.BASE}/authority.json", prior_evidence_ids=["ev-ref-1"],
            role_snapshot_refs=[f"{records.BASE}/role_plan.json"])
        frozen["actions"] = [dict(action_id="A001", type="execution_diagnostic", source_commit=records.SOURCE,
            run_ids=[run_id], attempt_ids=[name + "-v1"], cost_event_ids=["cost-test-1"],
            evidence_refs=[f"{records.BASE}/{name}/contract.json"])]
        frozen["decision"].update(outcome="ACTIVE", final=False)
        frozen["result"] = dict(evidence_ids=[], evidence_refs=[])
        save(root / records.BASE / name / "rml/trajectory.json", frozen)
        output = inspection / name
        provenance = dict(context=producer, accelerator="Tesla T4", recipe_identity="a" * 64,
            dataset_manifest_sha256=arm["data"]["dataset"]["sha256"])
        native = dict(status="VERIFIED", proof_artifacts=[])
        certificate = dict(format="molgap-runtime-certificate-v1", status="accepted",
            provenance_sha256=canonical_fingerprint(provenance), formal_training_exposure=0, proofs=[])
        for filename in ("repeat-a.pt", "repeat-b.pt", "resumed.pt", "resumed-first.pt"):
            digest = save(output / filename, (name + filename).encode())
            binding = dict(path=filename, sha256=digest)
            if filename == "resumed-first.pt":
                certificate["resume_origin"] = binding
            else:
                certificate["proofs"].append(binding)
            native["proof_artifacts"].append(dict(path=f"{name}/{filename}", sha256=digest))
        native["certificate_sha256"] = save(output / "runtime_certificate.json", certificate)
        native["provenance_sha256"] = save(output / "runtime_provenance.json", provenance)
        contract = json.loads((root / records.BASE / name / "contract.json").read_bytes())
        trace = canonicalize_trace(dict(trajectory_id=tid, run_id=run_id,
            observations=[dict(optimizer_step=32, sample_presentations=4096, epoch_or_pass=32 / 781)]))
        values = dict(contract=(root / records.BASE / name / "contract.json").read_bytes(), trace=trace,
            trace_context=dict(context=producer, stage_id="downstream"), selected_model=b"selected-state",
            resume=b"resume-state")
        artifacts = {role: dict(path=role + (".pt" if role in {"selected_model", "resume"} else ".json"),
            sha256=save(output / (role + (".pt" if role in {"selected_model", "resume"} else ".json")), value))
            for role, value in values.items()}
        progress = dict(optimizer_steps=32, sample_presentations=4096, completed_passes=0,
            next_batch=32, status="complete")
        manifest = dict(format="molgap-family-output-v1", context=producer, artifacts=artifacts,
            adapter="k1-session-v1" if name == "k1" else "gptrans-session-v1", progress=progress,
            runtime=dict(runtime_certificate_id=canonical_fingerprint(certificate)), costs=[])
        manifest_digest = save(output / "output_manifest.json", manifest)
        observed = dict(status="MECHANICALLY_VERIFIED", blockers=[], context=context.to_dict(), observed=dict(
            artifacts=artifacts, progress=progress, selected_optimizer_step=32,
            selected_weights="live" if name == "k1" else "ema", producer_platform_version=None))
        snapshot = InspectionSnapshot._from_owner(output_dir=output.absolute(), manifest_sha256=manifest_digest,
            context=context.to_dict(), expected=contract["acceptance_requirements"], report=observed,
            manifest_context=producer, artifact_bindings=artifacts)
        snapshots[name] = snapshot
        report["arms"][name] = dict(inspection_snapshot_sha256=snapshot.digest,
            role_use=records.ROLE_USE, native_qualification=native)
    monkeypatch.setattr(records.RunContext, "from_launch", classmethod(
        lambda cls, spec, receipt_path, package_dir, *, expected_package_identity, arm_id: contexts[arm_id]))
    missing = dict(value=None, status="measurement_missing")
    ledger = dict(format="molgap-allocation-ledger-v1", spec_identity=records.SPEC,
        started_at="2026-10-18T11:45:00+00:00", observed_at="2026-10-18T11:46:40+00:00", status="complete",
        allocation_device_count=2, wall_seconds=100., allocated_device_seconds=200., unassigned_device_seconds=0.,
        devices=[dict(device=i, hardware="Tesla T4", arm_id=name, allocated_seconds=100.) for i, name in enumerate(records.ARMS)],
        scope="Python bootstrap/runtime observation window; includes idle allocated devices",
        provisioning_before_python_seconds=missing, queue_seconds=missing,
        prior_segments=[], prior_unobserved_intervals=False)
    ledger_sha = save(inspection / "experiment/allocation_ledger.json", ledger)
    for name in records.ARMS:
        save(inspection / f"experiment/{name}/allocation_ledger.json", ledger)
    execution = dict(spec_identity=records.SPEC, status="complete", arms={name: dict(device=i,
        training_started=True, resumed_from_checkpoint=False, terminal_status="complete", exit_reason="train_worker_exit_0")
        for i, name in enumerate(records.ARMS)}, phases=[dict(phase=phase, status="complete", workers=[
            dict(arm_id=name, device=i, returncode=0) for i, name in enumerate(records.ARMS)]) for phase in ("preflight", "train")])
    execution_sha = save(inspection / "experiment/execution_report.json", execution)
    save(inspection / "experiment/execution_retention.json", dict(format="molgap-execution-retention-v1", status="SEALED",
        spec_identity=records.SPEC, package_identity=records.PACKAGE, source_commit=records.SOURCE,
        source_archive_sha256=records.ARCHIVE, execution_report=dict(path="execution_report.json", sha256=execution_sha),
        allocation_ledger=dict(root=dict(path="allocation_ledger.json", sha256=ledger_sha), arms={name:
            dict(path=f"{name}/allocation_ledger.json", sha256=ledger_sha) for name in records.ARMS})))
    save(inspection / "bootstrap_allocation.json", dict(status="complete", allocation_device_count=2,
        hardware=["Tesla T4"] * 2, wall_seconds=120., allocated_device_seconds=240.))
    digest = save(inspection / "acceptance.json", report)
    return dict(repo_root=root, run_directory=run, inspection_root=inspection,
        output_relative=f"{records.BASE}/results/terminal_trial", finalized_at="2026-10-18T12:00:00+00:00",
        acceptance_sha256=digest, inspection_snapshots=snapshots)


def test_prepare_idempotent_finalizable_context_only_training_not_no_train(trial):
    prepared = records.prepare_trial_terminal(**trial)
    assert prepared["status"] == "PREPARED"
    assert records.prepare_trial_terminal(**trial)["status"] == "ALREADY_PREPARED"
    root = trial["repo_root"]
    assert not list(root.rglob("*.pt"))
    for item in prepared["arms"].values():
        terminal = json.loads((root / item["terminal"]).read_bytes())
        evidence = terminal["evidence"]
        validate_v5_evidence_envelope(evidence, repo_root=root)
        assert terminal["decision"]["outcome"] == "INCONCLUSIVE"
        assert evidence["outcome"]["execution_status"] == "complete"
        assert evidence["outcome"]["artifact_status"] == "accepted"
        assert evidence["outcome"]["comparison_status"] == "context_only"
        assert evidence["execution_observation"]["training_executed"] is True
        assert evidence["migration"]["training_executed"] is False
        assert evidence["execution_observation"]["noReplay"] is True
        assert terminal["decision"]["next_allowed_actions"] == []
        assert "trace_manifest" not in terminal and "action_replay" not in terminal
        assert {v["status"] for e in terminal["costs"] for v in e["measurement"].values()} == {"measurement_missing"}
        allocation = evidence["execution_observation"]["shared_allocation_observation"]
        assert allocation["bootstrap_device_seconds"] == 240
        assert allocation["ledger_device_seconds"] == 200
        assert not allocation["additive_with_other_arm"] and not allocation["bootstrap_additive_with_ledger"]
        receipt = finalize(root, item["trajectory"], item["terminal"])
        assert receipt["outcome"] == "INCONCLUSIVE"
        assert finalize(root, item["trajectory"], item["terminal"])["status"] == "ALREADY_FINALIZED"


@pytest.mark.parametrize("ref", ["k1/selected_model.pt", "gptrans/resume.pt", "k1/trace_context.json",
    "gptrans/repeat-a.pt", "k1/runtime_certificate.json", "acceptance.json", "experiment/allocation_ledger.json"])
def test_changed_artifact_fails_before_any_terminal_publication(trial, ref):
    path = trial["inspection_root"] / ref
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        records.prepare_trial_terminal(**trial)
    assert not (trial["repo_root"] / trial["output_relative"]).exists()


@pytest.mark.parametrize("change", ["role", "steps", "package", "verified", "native", "snapshot"])
def test_rehashed_host_report_cannot_override_owner_evidence(trial, change):
    report_path = trial["inspection_root"] / "acceptance.json"
    report = json.loads(report_path.read_bytes())
    if change == "role":
        report["arms"]["k1"]["role_use"] = {**records.ROLE_USE, "development": "consumed"}
    elif change == "package":
        report["package_identity"] = "f" * 64
    elif change == "verified":
        report["verified"] = 1
    elif change == "native":
        report["arms"]["k1"]["native_qualification"]["proof_artifacts"] = []
    elif change == "snapshot":
        trial["inspection_snapshots"]["k1"] = {"status": "MECHANICALLY_VERIFIED"}
    else:
        manifest = trial["inspection_root"] / "k1/output_manifest.json"
        value = json.loads(manifest.read_bytes())
        value["progress"]["optimizer_steps"] = 31
        save(manifest, value)
    trial["acceptance_sha256"] = save(report_path, report)
    with pytest.raises((ValueError, TypeError)):
        records.prepare_trial_terminal(**trial)
    assert not (trial["repo_root"] / trial["output_relative"]).exists()


def test_changed_finalized_time_does_not_overwrite_existing_package(trial):
    records.prepare_trial_terminal(**trial)
    path = trial["repo_root"] / trial["output_relative"] / "k1/terminal.json"
    before = path.read_bytes()
    trial["finalized_at"] = "2026-10-18T12:01:00+00:00"
    with pytest.raises(ValueError, match="PREPARATION_CONFLICT"):
        records.prepare_trial_terminal(**trial)
    assert path.read_bytes() == before


def test_interrupted_metadata_publication_resumes_identical_bytes(trial, monkeypatch):
    publish = records.publish_immutable_bytes
    calls = []
    def interrupted(path, payload):
        calls.append(path)
        if len(calls) == 3:
            raise OSError("simulated host interruption")
        return publish(path, payload)
    with monkeypatch.context() as patch:
        patch.setattr(records, "publish_immutable_bytes", interrupted)
        with pytest.raises(OSError, match="interruption"):
            records.prepare_trial_terminal(**trial)
    assert not (trial["repo_root"] / trial["output_relative"] / "preparation.json").exists()
    assert records.prepare_trial_terminal(**trial)["status"] == "PREPARED"
    assert records.prepare_trial_terminal(**trial)["status"] == "ALREADY_PREPARED"


@pytest.mark.parametrize("destination", ["../outside", records.BASE + "/results", records.BASE + "/k1/rml"])
def test_destination_is_confined_to_dedicated_results(trial, destination):
    trial["output_relative"] = destination
    with pytest.raises(ValueError):
        records.prepare_trial_terminal(**trial)
