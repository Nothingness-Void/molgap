"""Prospective, checkpoint-only recovery; never repeat the completed timing probe."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time

from molgap.research_memory.finalize import finalize
from molgap.research_memory.plan import plan
from molgap.training_reproducibility import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RECOVERY = HERE / "checkpoint_recovery"
REL = RECOVERY.relative_to(ROOT).as_posix()
TID = "TB-k1-execution-checkpoint-recovery-20261010"
RUN = "k1-execution-checkpoint-recovery-local-20261010"
EID = "pcqm-k1-execution-checkpoint-recovery-20261010"


def main():
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    inputs = read(HERE / "inputs.json")
    if (RECOVERY / "rml").exists():
        raise FileExistsError("Recovery prospective already exists; reconcile, do not repeat")
    spec = deepcopy(read(HERE / "plan_input.json"))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    executed = (HERE / "run.py", Path(__file__), ROOT / "src/molgap/k1_execution_layout.py",
                ROOT / "src/molgap/k1_execution_efficiency.py")
    binding = {"source_commit": head, "source_files": {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in executed},
               "original_inputs_sha256": sha256_file(HERE / "inputs.json"),
               "normalization_runtime_sha256": sha256_file(HERE / "results/cpu-attempt-001/runtime.json"),
               "bindings": inputs["bindings"], "sample_source_idx": inputs["source_idx"], "worker_ceiling_seconds": 120}
    atomic_json(RECOVERY / "inputs.json", binding)
    atomic_json(RECOVERY / "role_plan.json", {"train_decoded_rows": 4096, "train_executed_rows": 256,
        "development": "untouched", "protected_roles": "untouched", "selection_used": False})
    trajectory = spec["trajectory"]
    trajectory.update(trajectory_id=TID, question="Does corrected atomic checkpoint resume preserve CPU K1 state and RNG?",
                      family_id="k1-execution-checkpoint-recovery", comparison_readiness_ref=f"{REL}/README.md")
    cost_id = "cost-k1-execution-checkpoint-recovery-estimate"
    trajectory["hypothesis"].update(hypothesis_id="H-k1-execution-checkpoint-recovery-20261010",
        observed_deficiency="Completed CPU probe failed at st_size metadata before checkpoint continuation.",
        changed_mechanism="Correct metadata/property usage, CPU AdamW scalar placement and budget checks; checkpoint-only retry",
        cheapest_falsifier="Three discarded updates on two frozen train-only batches, atomic save and exact resumed comparison",
        supporting_evidence_ids=["pcqm-k1-execution-efficiency-20261010"], expected_native_cost_ref=cost_id,
        decision_changed_if_positive="Mechanical CPU checkpoint path usable, GPU/speed qualification still absent",
        decision_changed_if_negative="Keep checkpoint qualification blocked; no timing rerun")
    state = trajectory["state_at_start"]
    state.update(source_commit=head, source_config_identity=sha256_file(RECOVERY / "inputs.json"),
        contract_refs=[f"{REL}/README.md", f"{REL}/inputs.json"],
        parent_trajectory_ids=["TB-k1-execution-efficiency-20261010"],
        prior_evidence_ids=["pcqm-k1-execution-efficiency-20261010"],
        role_snapshot_refs=[f"{REL}/role_plan.json"], budget_snapshot_ref=f"{REL}/README.md")
    trajectory["decision"].update(decision_ref=f"{REL}/decision.md", next_allowed_actions=["CPU checkpoint-only recovery"])
    trajectory["actions"] = [dict(action_id="A001", type="execution_profile", run_ids=[RUN], attempt_ids=["attempt-001"],
        source_commit=head, evidence_refs=[f"{REL}/inputs.json"], cost_event_ids=[cost_id])]
    spec["decision_state"].update(source_commit=head, state_timestamp=datetime.now(timezone.utc).isoformat(),
        budget_snapshot_ref=f"{REL}/README.md", role_snapshot_refs=[f"{REL}/role_plan.json"])
    cost = spec["costs"][0]
    cost.update(cost_event_id=cost_id, trajectory_id=TID, run_id=RUN, hardware="CPU; no accelerator work",
                evidence_ref=f"{REL}/README.md")
    cost["measurement"] = {k: {"value": 120 / 3600 if k == "wall_hours" else None,
        "status": "estimated" if k == "wall_hours" else "not_applicable" if k in ("device_hours", "queue_hours")
        else "measurement_missing"} for k in ("wall_hours", "cpu_hours", "device_hours", "queue_hours")}
    atomic_json(RECOVERY / "plan_input.json", spec)
    atomic_json(RECOVERY / "plan_receipt.json", plan(ROOT, spec, f"{REL}/rml"))

    # Execution begins only after repository-backed prospective publication.
    import torch
    from torch_geometric.loader import DataLoader
    from molgap.k1_execution_efficiency import compare_states
    from molgap.k1_execution_layout import cpu_layout_context
    from molgap.pcqm_500k_v4_evidence import make_model, optimizer_for, step
    from molgap.training_reproducibility import configure_fp32_determinism
    from molgap.v4_runtime import state_dict_sha256
    from molgap.k1_screen_training import FORBIDDEN_MODEL_FIELDS
    from run import _checkpoint_resume
    started, cpu_started = time.perf_counter(), time.process_time()
    for name, digest in binding["source_files"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError("Recovery source changed")
    for entry in binding["bindings"].values():
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("Recovery input changed")
    configure_fp32_determinism(42)
    torch.set_num_threads(4)
    initial = torch.load(inputs["bindings"]["initial"]["path"], map_location="cpu", weights_only=True)["model_state"]
    if state_dict_sha256(initial) != inputs["bindings"]["initial"]["tensor_sha256"]:
        raise ValueError("Recovery initialization tensor identity differs")
    if sha256_file(HERE / "results/cpu-attempt-001/runtime.json") != binding["normalization_runtime_sha256"]:
        raise ValueError("Recovery normalization metadata changed")
    normalization = read(HERE / "results/cpu-attempt-001/runtime.json")["normalization"]
    graphs = torch.load(inputs["bindings"]["train_probe"]["path"], map_location="cpu", weights_only=False)
    if [int(g.source_idx) for g in graphs] != inputs["source_idx"]:
        raise ValueError("Recovery training members differ")
    if any(not 0 <= int(g.source_idx) < 500000 or any(k in g for k in FORBIDDEN_MODEL_FIELDS) for g in graphs):
        raise ValueError("Nontraining or geometry recovery input")
    batches = list(DataLoader(graphs[:256], batch_size=128, shuffle=False, num_workers=0))

    def factory():
        model = make_model("neural_atom_k1")
        model.load_state_dict(initial, strict=True)
        return model

    result = _checkpoint_resume(torch=torch, factory=factory, optimizer_for=optimizer_for, step=step,
        batches=batches, mean=normalization["mean"], std=normalization["std"], device="cpu", deadline=started + 120,
        checkpoint_path=ROOT / "platforms/_records/local/staging/k1-execution-efficiency/cpu-scratch-recovery.pt",
        cpu_layout_context=cpu_layout_context, compare_states=compare_states)
    result.update(source_commit=head, scientific_training=False, training_replay_ready=False,
        worker_wall_seconds=time.perf_counter() - started, process_cpu_seconds=time.process_time() - cpu_started,
        train_decoded_source_idx=inputs["source_idx"], train_executed_source_idx=inputs["source_idx"][:256])
    atomic_json(RECOVERY / "result.json", result)
    if not result["comparison"]["exact"] or not result["state_keys_preserved"]:
        raise ValueError("Recovery exact continuation failed; keep evidence, no false acceptance")
    outcome = dict(execution_status="complete_checkpoint_diagnostic", artifact_status="local_hash_verified",
        comparison_status="exact_cpu_checkpoint_continuation", scientific_status="NO_TRAIN", transfer_status="not_evaluated",
        budget_decision="bounded_cpu_checkpoint_recovery_complete", full_handoff_status="not_applicable")
    decision = dict(outcome="NO_TRAIN", decision_ref=f"{REL}/decision.md", next_allowed_actions=[],
                    reopen_conditions=["Separately prospective idle-GPU or native T4 execution qualification"])
    cost = deepcopy(cost)
    cost.update(cost_event_id="cost-k1-checkpoint-recovery-observed", category="other", evidence_ref=f"{REL}/acceptance.json")
    cost["measurement"].update(wall_hours=dict(value=result["worker_wall_seconds"] / 3600, status="measured"),
                               cpu_hours=dict(value=result["process_cpu_seconds"] / 3600, status="measured"))
    roles = []
    for role, rows, kinds in (("train_decoded", inputs["source_idx"], ("labels_read",)),
        ("train_executed", inputs["source_idx"][:256], ("prediction_input", "metric_computed"))):
        digest = hashlib.sha256(b"".join(struct.pack("<q", row) for row in rows)).hexdigest()
        for kind in kinds:
            roles.append(dict(schema="molgap-role-event-v1", role_event_id=f"role-k1-checkpoint-recovery-{role}-{kind}",
                trajectory_id=TID, action_id="A001", run_id=RUN, dataset_identity="pcqm4mv2-ogb-fixed-500k-scnet-v1",
                row_manifest_hash=digest, role_name=role, access_kind=kind, selection_used=False, evidence_ref=f"{REL}/acceptance.json"))
    role_use = dict(train_decoded="labels_read", train_executed="prediction_input", development="untouched",
                    official_validation="untouched", test_dev="untouched", test_challenge="untouched")
    atomic_json(RECOVERY / "acceptance.json", dict(evidence_id=EID, run_id=RUN, outcome=outcome,
        trajectory_decision=decision, costs=[cost], roles=roles, role_use=role_use, checkpoint_resume_exact=True,
        gpu_qualification=False, training_replay_ready=False))
    (RECOVERY / "attribution.md").write_text(
        "# Checkpoint recovery attribution\n\n"
        "The original tail failed while calling st_size as a method; it did not\n"
        "invalidate the retained scratch comparison or serialize a failed checkpoint.\n"
        "This bounded CPU-only recovery reuses the corrected atomic-save/resume helper.\n"
        "Its resumed update matches uninterrupted loss, parameters, BN buffers,\n"
        "optimizer state and RNG exactly on the two frozen training-only batches.\n"
        "The new checkpoint preserves the failed attempt's saved bytes.\n\n"
        "This qualifies only this short CPU continuation. Native CUDA performance,\n"
        "full training replay and model accuracy remain unqualified; no new training\n"
        "or reference is justified merely to complete those fields. See result.json\n"
        "and acceptance.json for the observed identities, counts and measured cost.\n",
        encoding="utf-8",
    )
    files = [p for p in RECOVERY.iterdir() if p.is_file() and p.name != "terminal.json"]
    hashes = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in files}
    evidence = dict(format="molgap-v5-evidence-envelope-v1", contract="MOLGAP-COMMON-V5-FINAL", evidence_id=EID,
        track="B", scope="desktop_k1_checkpoint_recovery", legacy_contract="pcqm-k1-execution-checkpoint-recovery-v1",
        outcome=outcome, role_use=role_use, authority=dict(pointers=[f"{REL}/README.md", f"{REL}/decision.md", f"{REL}/acceptance.json"]),
        migration=dict(migrated_at=datetime.now(timezone.utc).date().isoformat(), training_executed=False,
            inference_executed=False, scientific_reinterpretation=False, verification_scope="Metadata acceptance of prospective CPU checkpoint diagnostic"),
        observed_execution=dict(training_executed=False, inference_executed=True, training_replay_ready=False,
                                execution_ref=f"{REL}/result.json"),
        artifacts=[dict(name=p.removeprefix(REL + "/"), locator=p, sha256=h, availability="locally_retained_hash_verified")
                   for p, h in hashes.items()])
    atomic_json(RECOVERY / "terminal.json", dict(format="molgap-rml-terminal-package-v1", trajectory_id=TID,
        run_id=RUN, action_id="A001", finalized_at=datetime.now(timezone.utc).isoformat(), acceptance_ref=f"{REL}/acceptance.json",
        artifact_hashes=hashes, evidence=evidence, decision=decision, costs=[cost], roles=roles))
    print(json.dumps(finalize(ROOT, f"{REL}/rml", f"{REL}/terminal.json")))


if __name__ == "__main__":
    main()
