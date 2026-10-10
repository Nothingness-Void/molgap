"""Thin precision diagnostic custody adapter; family/probe owners execute."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time

from molgap.research_memory.plan import plan
from molgap.research_memory.finalize import finalize
from molgap.training_reproducibility import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ATTEMPT = HERE / "tf32_attempt_001"
REL = ATTEMPT.relative_to(ROOT).as_posix()
TID = "TB-k1-tf32-execution-20261010"
RUN = "k1-tf32-rtx5060-20261010"
EID = "pcqm-k1-tf32-execution-20261010"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare():
    if (ATTEMPT / "rml").exists():
        raise FileExistsError("Already planned; reconcile, do not overwrite")
    spec, inputs = deepcopy(read(HERE / "gpu_attempt_001/plan_input.json")), deepcopy(read(HERE / "gpu_attempt_001/inputs.json"))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    files = [*inputs["files"], Path(__file__).relative_to(ROOT).as_posix(),
             "src/molgap/k1_precision_diagnostic.py", "src/molgap/k1_frozen_inference.py", f"{REL}/README.md", f"{REL}/decision.md"]
    inputs.update(source_commit=head, files={name: sha256_file(ROOT / name) for name in files},
        selected_checkpoint_source_sha256="0d53f2ae5a46d51cff9730e00767fa16b80ffb27002fa50b52a67c12ac8ab65a",
        selected_checkpoint_epoch=48, worker_wall_limit_seconds=600)
    for binding in inputs["bindings"].values():
        if sha256_file(Path(binding["path"])) != binding["sha256"]:
            raise ValueError("Retained artifact changed")
    atomic_json(ATTEMPT / "inputs.json", inputs)
    atomic_json(ATTEMPT / "role_plan.json", dict(train_decoded=4096, train_prediction_and_metrics=4096,
        development="untouched", protected_roles="untouched", selection_used=False))
    cost_id = "cost-k1-tf32-estimate"
    prior = ["pcqm-k1-execution-efficiency-gpu-20261010"]
    trajectory = spec["trajectory"]
    trajectory.update(trajectory_id=TID, family_id="k1-precision-execution", comparison_readiness_ref=f"{REL}/README.md",
                      question="How much do TF32 matmul permission and equivalent layout change K1 cost and frozen prediction/update numerics?")
    trajectory["hypothesis"].update(hypothesis_id="H-k1-tf32-execution-20261010", supporting_evidence_ids=prior,
        expected_native_cost_ref=cost_id, observed_deficiency="Exact layout saves about7% locally; remaining compute and optimizer costs are unseparated.",
        changed_mechanism="Explicit matmul-only TF32 vs original/layout, frozen trained prediction perturbation and phase/operator attribution",
        cheapest_falsifier="Bounded scratch timings and numerical comparison; train-only frozen prediction deltas and profiler",
        decision_changed_if_positive="Nominate separately contracted precision/native qualification if useful; no production adoption",
        decision_changed_if_negative="Keep strict FP32 and prioritize remaining measured bottlenecks")
    state = trajectory["state_at_start"]
    state.update(source_commit=head, source_config_identity=sha256_file(ATTEMPT / "inputs.json"),
        contract_refs=[f"{REL}/README.md", f"{REL}/inputs.json"], parent_trajectory_ids=["TB-k1-execution-efficiency-gpu-20261010"],
        prior_evidence_ids=prior, role_snapshot_refs=[f"{REL}/role_plan.json"], budget_snapshot_ref=f"{REL}/README.md")
    trajectory["actions"] = [dict(action_id="A001", type="execution_profile", run_ids=[RUN], attempt_ids=["tf32-attempt-001"],
        source_commit=head, evidence_refs=[f"{REL}/inputs.json"], cost_event_ids=[cost_id])]
    trajectory["decision"].update(decision_ref=f"{REL}/decision.md", next_allowed_actions=["Bounded idle-only precision diagnostic"])
    spec["decision_state"].update(source_commit=head, state_timestamp=datetime.now(timezone.utc).isoformat(),
        budget_snapshot_ref=f"{REL}/README.md", role_snapshot_refs=[f"{REL}/role_plan.json"])
    cost = spec["costs"][0]
    cost.update(cost_event_id=cost_id, trajectory_id=TID, run_id=RUN, attempt_id="tf32-attempt-001", evidence_ref=f"{REL}/README.md")
    atomic_json(ATTEMPT / "plan_input.json", spec)
    atomic_json(ATTEMPT / "plan_receipt.json", plan(ROOT, spec, f"{REL}/rml"))
    print(TID + " prospective frozen")


def execute():
    started, cpu_started = time.perf_counter(), time.process_time()
    from run import idle_observation
    import torch
    from torch_geometric.loader import DataLoader
    from molgap.k1_screen_training import FORBIDDEN_MODEL_FIELDS
    from molgap.k1_frozen_inference import load_native500k_k1, predict_clean
    from molgap.k1_precision_diagnostic import (matmul_tf32, prediction_drift, probe_k1_precision, profile_k1_step)
    from molgap.pcqm_500k_v4_evidence import make_model
    from molgap.training_reproducibility import configure_fp32_determinism, build_runtime_manifest
    from molgap.v4_runtime import state_dict_sha256

    output = ATTEMPT / "results"
    if output.exists():
        raise FileExistsError("Attempt exists; reconcile, never repeat blindly")
    inputs, trajectory = read(ATTEMPT / "inputs.json"), read(ATTEMPT / "rml/trajectory.json")
    if trajectory["record_mode"] != "prospective" or trajectory["decision"]["outcome"] != "ACTIVE":
        raise ValueError("Active prospective required")
    deadline = started + inputs["worker_wall_limit_seconds"]
    for name,digest in inputs["files"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError("Frozen source changed")
    for binding in inputs["bindings"].values():
        if sha256_file(Path(binding["path"])) != binding["sha256"]:
            raise ValueError("Frozen input changed")
    observations = [idle_observation()]
    time.sleep(2)
    observations.append(idle_observation())
    if not all(o["idle"] for o in observations):
        raise RuntimeError("GPU not idle; no work launched, do not monitor")
    output.mkdir(parents=True)
    settings = configure_fp32_determinism(42)
    torch.set_num_threads(4)
    graphs = torch.load(inputs["bindings"]["train_probe"]["path"], map_location="cpu", weights_only=False)
    if len(graphs) != 4096 or [int(g.source_idx) for g in graphs] != inputs["source_idx"] or any(
            not 0 <= int(g.source_idx) < 500000 or any(k in g for k in FORBIDDEN_MODEL_FIELDS) for g in graphs):
        raise ValueError("Frozen train-only pure2D members differ")
    initial = torch.load(inputs["bindings"]["initial"]["path"], map_location="cpu", weights_only=True)["model_state"]
    if state_dict_sha256(initial) != inputs["bindings"]["initial"]["tensor_sha256"]:
        raise ValueError("Frozen initialization differs")
    selected = inputs["bindings"]["normalization_checkpoint"]
    model, meta = load_native500k_k1(Path(selected["path"]), expected_sha256=selected["sha256"],
        expected_source_sha256=inputs["selected_checkpoint_source_sha256"], expected_epoch=inputs["selected_checkpoint_epoch"],
        checkpoint_kind="selected")
    runtime = build_runtime_manifest(settings)
    runtime.update(source_commit=inputs["source_commit"], input_sha256=sha256_file(ATTEMPT / "inputs.json"),
        prospective_sha256=sha256_file(ATTEMPT / "rml/trajectory.json"), idle_observations=observations,
        allowed_tf32_cases=["tf32", "tf32_layout"], normalization=dict(mean=meta["mean"], std=meta["std"]))
    atomic_json(output / "runtime.json", runtime)
    def factory():
        scratch = make_model("neural_atom_k1")
        scratch.load_state_dict(initial, strict=True)
        return scratch
    batches = list(DataLoader(graphs[:1920], batch_size=128, shuffle=False, num_workers=0))
    probe = probe_k1_precision(factory, batches, meta["mean"], meta["std"], deadline=deadline)
    atomic_json(output / "scratch_probe.json", probe)
    if probe["status"] != "complete":
        raise TimeoutError("Scratch diagnostic budget exhausted; do not continue")
    # Trained-state perturbations use the existing clean-inference owner, not a new loop.
    before = state_dict_sha256(model.state_dict())
    predictions, prediction_costs = {}, {}
    for name, enabled in (("fp32", False), ("tf32", True), ("fp32_repeat", False)):
        with matmul_tf32(enabled):
            predictions[name], prediction_costs[name] = predict_clean(model,
                DataLoader(graphs, batch_size=128, shuffle=False, num_workers=0),
                mean=meta["mean"], std=meta["std"], device="cuda", deadline=deadline)
        if state_dict_sha256(model.state_dict()) != before:
            raise ValueError("Frozen trained inference mutated state")
    drift = prediction_drift(predictions["fp32"], predictions["tf32"])
    repeat = prediction_drift(predictions["fp32"], predictions["fp32_repeat"])
    if not repeat["exact"]:
        raise ValueError("FP32 repeated prediction changed")
    atomic_json(output / "predictions.json", {name:{k:v.tolist() for k,v in artifact.items()} for name,artifact in predictions.items()})
    atomic_json(output / "prediction_drift.json", dict(tf32=drift, repeated_fp32=repeat,
        timings=prediction_costs, speed_qualification="One inference pass each, not repeated balanced speed evidence"))
    del model
    for name, enabled, layout in (("fp32",False,False), ("layout",False,True), ("tf32_layout",True,True)):
        atomic_json(output / ("profile_"+name+".json"), profile_k1_step(factory, batches[0], meta["mean"], meta["std"],
            tf32=enabled, layout=layout, deadline=deadline))
    wall = time.perf_counter()-started
    result = dict(status="complete", worker_wall_seconds=wall, process_cpu_seconds=time.process_time()-cpu_started,
        inputs_sha256=sha256_file(ATTEMPT / "inputs.json"), prospective_sha256=sha256_file(ATTEMPT / "rml/trajectory.json"),
        train_decoded_source_idx=inputs["source_idx"], train_executed_source_idx=inputs["source_idx"],
        frozen_trained_state_unchanged=True, fp32_repeat_exact=True, training_replay_ready=False,
        scientific_training=False, generalization_accuracy_qualified=False, observed_overrun_seconds=max(0.,wall-inputs["worker_wall_limit_seconds"]))
    atomic_json(output / "result.json", result)
    atomic_json(output / "completion.json", dict(status="complete", artifacts={p.name:sha256_file(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(dict(status=result["status"], wall_seconds=wall, prediction_drift=drift,
        timings={k:v.get("median_seconds_per_step") for k,v in probe["cases"].items()})))


def close():
    output = ATTEMPT / "results"
    inputs, result, completion = read(ATTEMPT / "inputs.json"), read(output / "result.json"), read(output / "completion.json")
    if (result["status"] != "complete" or completion["status"] != "complete" or result["observed_overrun_seconds"] > 0
            or result["inputs_sha256"] != sha256_file(ATTEMPT / "inputs.json")
            or result["prospective_sha256"] != sha256_file(ATTEMPT / "rml/trajectory.json")
            or result["train_executed_source_idx"] != inputs["source_idx"]):
        raise ValueError("Incomplete or mismatched precision diagnostic")
    for name,digest in completion["artifacts"].items():
        if sha256_file(output / name) != digest:
            raise ValueError("Output bytes differ")
    for name,digest in inputs["files"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError("Executed source changed")
    for name in ("terminal_decision.md", "attribution.md"):
        if not (ATTEMPT / name).exists():
            raise ValueError("Parent interpretation required")
    decision = dict(outcome="NO_TRAIN", decision_ref=f"{REL}/terminal_decision.md", next_allowed_actions=[],
                    reopen_conditions=["Separately authorized native precision/quality qualification"])
    outcome = dict(execution_status="complete_precision_diagnostic", artifact_status="local_hash_verified",
        comparison_status="local_train_only_numerical_perturbation", scientific_status="NO_TRAIN", transfer_status="not_evaluated",
        budget_decision="bounded_local_diagnostic_complete", full_handoff_status="not_applicable")
    cost = deepcopy(read(ATTEMPT / "plan_input.json")["costs"][0])
    cost.update(cost_event_id="cost-k1-tf32-observed", category="other", evidence_ref=f"{REL}/acceptance.json")
    cost["measurement"] = dict(wall_hours=dict(value=result["worker_wall_seconds"]/3600,status="measured"),
        cpu_hours=dict(value=result["process_cpu_seconds"]/3600,status="measured"),
        device_hours=dict(value=None,status="measurement_missing"), queue_hours=dict(value=None,status="not_applicable"))
    digest = hashlib.sha256(b"".join(struct.pack("<q",i) for i in inputs["source_idx"])).hexdigest()
    roles = [dict(schema="molgap-role-event-v1",role_event_id="role-k1-tf32-"+access,trajectory_id=TID,action_id="A001",run_id=RUN,
        dataset_identity="pcqm4mv2-ogb-fixed-500k-scnet-v1",row_manifest_hash=digest,role_name="train_probe",
        access_kind=access,selection_used=False,evidence_ref=f"{REL}/acceptance.json") for access in ("labels_read","prediction_input","metric_computed")]
    role_use = dict(train_probe="consumed",development="untouched",official_validation="untouched",test_dev="untouched",test_challenge="untouched")
    atomic_json(ATTEMPT / "acceptance.json",dict(evidence_id=EID,outcome=outcome,trajectory_decision=decision,costs=[cost],roles=roles,
        role_use=role_use,training_replay_ready=False,generalization_accuracy_qualified=False,precision_contract_change=True))
    files = [p for p in ATTEMPT.rglob("*") if p.is_file() and "rml" not in p.relative_to(ATTEMPT).parts and p.name not in ("terminal.json","closure_receipt.json")]
    hashes = {p.relative_to(ROOT).as_posix():sha256_file(p) for p in files}
    evidence = dict(format="molgap-v5-evidence-envelope-v1",contract="MOLGAP-COMMON-V5-FINAL",evidence_id=EID,track="B",scope="k1_tf32_execution",
        legacy_contract="pcqm-k1-tf32-execution-v1",outcome=outcome,role_use=role_use,
        authority=dict(pointers=[f"{REL}/{n}" for n in ("README.md","terminal_decision.md","attribution.md","acceptance.json")]),
        migration=dict(migrated_at=datetime.now(timezone.utc).date().isoformat(),training_executed=False,inference_executed=False,
            scientific_reinterpretation=False,verification_scope="Metadata closure of separately executed precision diagnostic"),
        observed_execution=dict(training_executed=False,inference_executed=True,training_replay_ready=False,execution_ref=f"{REL}/results/result.json"),
        artifacts=[dict(name=p.removeprefix(REL+"/"),locator=p,sha256=h,availability="locally_retained_hash_verified") for p,h in hashes.items()])
    atomic_json(ATTEMPT / "terminal.json",dict(format="molgap-rml-terminal-package-v1",trajectory_id=TID,run_id=RUN,action_id="A001",
        finalized_at=datetime.now(timezone.utc).isoformat(),acceptance_ref=f"{REL}/acceptance.json",artifact_hashes=hashes,evidence=evidence,
        decision=decision,costs=[cost],roles=roles))
    atomic_json(ATTEMPT / "closure_receipt.json",finalize(ROOT,f"{REL}/rml",f"{REL}/terminal.json"))
    print(EID+" finalized NO_TRAIN")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("operation",choices=("prepare","execute","close"))
    {"prepare":prepare,"execute":execute,"close":close}[parser.parse_args().operation]()
