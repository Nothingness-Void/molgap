"""Prospective projection placement adapter; reuse family step and scratch probe."""
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
ATTEMPT = HERE / "projection_attempt_001"
REL = ATTEMPT.relative_to(ROOT).as_posix()
TID = "TB-k1-projection-execution-20261010"
RUN = "k1-projection-rtx5060-20261010"
EID = "pcqm-k1-projection-execution-20261010"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare():
    if (ATTEMPT / "rml").exists():
        raise FileExistsError("Already planned; reconcile")
    inputs, spec = deepcopy(read(HERE / "gpu_attempt_001/inputs.json")), deepcopy(read(HERE / "gpu_attempt_001/plan_input.json"))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    files = [*inputs["files"], "src/molgap/k1_execution_projection.py", Path(__file__).relative_to(ROOT).as_posix(),
             f"{REL}/README.md", f"{REL}/decision.md"]
    inputs.update(source_commit=head, files={name:sha256_file(ROOT / name) for name in files})
    for entry in inputs["bindings"].values():
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("Frozen train/initial identity differs")
    atomic_json(ATTEMPT / "inputs.json", inputs)
    atomic_json(ATTEMPT / "role_plan.json", read(HERE / "role_plan.json"))
    cost_id = "cost-k1-projection-estimate"
    prior = ["pcqm-k1-tf32-execution-20261010", "pcqm-k1-execution-efficiency-gpu-20261010"]
    t = spec["trajectory"]
    t.update(trajectory_id=TID, family_id="k1-projection-execution", comparison_readiness_ref=f"{REL}/README.md",
             question="Can node projection before edge gathering or dense padding reduce K1 step cost without changing state/RNG?")
    t["hypothesis"].update(hypothesis_id="H-k1-projection-execution-20261010", supporting_evidence_ids=prior,
        expected_native_cost_ref=cost_id, observed_deficiency="Repeated edge-row/padded-node projection is untested; local operators remain material after layout changes.",
        changed_mechanism="Separate project-before-gather and project-before-pack opt-in hooks",
        alternative_explanations=["Changed GEMM/backward reduction order loses bitwise equality", "Index/padding costs offset saved projection", "Small kernels dominate"],
        cheapest_falsifier="Synthetic gradient/RNG controls then three exact real scratch updates and short matched timings",
        decision_changed_if_positive="Nominate exact locally faster hook; native qualification still absent",
        decision_changed_if_negative="Reject strict-FP32 replacement or costly reorder and keep defaults")
    t["state_at_start"].update(source_commit=head,source_config_identity=sha256_file(ATTEMPT / "inputs.json"),
        contract_refs=[f"{REL}/README.md",f"{REL}/inputs.json"],
        parent_trajectory_ids=["TB-k1-execution-efficiency-gpu-20261010"],prior_evidence_ids=prior,
        role_snapshot_refs=[f"{REL}/role_plan.json"],budget_snapshot_ref=f"{REL}/README.md")
    t["actions"] = [dict(action_id="A001",type="execution_profile",run_ids=[RUN],attempt_ids=["projection-attempt-001"],
        source_commit=head,evidence_refs=[f"{REL}/inputs.json"],cost_event_ids=[cost_id])]
    t["decision"].update(decision_ref=f"{REL}/decision.md", next_allowed_actions=["Bounded idle-only projection diagnostic"])
    spec["decision_state"].update(source_commit=head,state_timestamp=datetime.now(timezone.utc).isoformat(),
        budget_snapshot_ref=f"{REL}/README.md",role_snapshot_refs=[f"{REL}/role_plan.json"])
    spec["costs"][0].update(cost_event_id=cost_id,trajectory_id=TID,run_id=RUN,attempt_id="projection-attempt-001",evidence_ref=f"{REL}/README.md")
    atomic_json(ATTEMPT / "plan_input.json",spec)
    atomic_json(ATTEMPT / "plan_receipt.json",plan(ROOT,spec,f"{REL}/rml"))
    print(TID+" prospective frozen")


def execute():
    started, cpu_started = time.perf_counter(), time.process_time()
    import torch
    from contextlib import contextmanager
    from torch_geometric.loader import DataLoader
    from molgap.k1_execution_efficiency import probe_k1_execution_efficiency
    from molgap.k1_execution_layout import cpu_layout_context
    from molgap.k1_execution_projection import project_before_gather_context, project_before_pack_context
    from molgap.k1_screen_training import FORBIDDEN_MODEL_FIELDS
    from molgap.pcqm_500k_v4_evidence import make_model
    from molgap.training_reproducibility import configure_fp32_determinism, build_runtime_manifest
    from molgap.v4_runtime import state_dict_sha256
    from run import idle_observation

    output = ATTEMPT / "results"
    if output.exists():
        raise FileExistsError("Attempt exists; reconcile, do not repeat")
    inputs, t = read(ATTEMPT / "inputs.json"),read(ATTEMPT / "rml/trajectory.json")
    if t["record_mode"] != "prospective" or t["decision"]["outcome"] != "ACTIVE":
        raise ValueError("Active prospective required")
    for name,digest in inputs["files"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError("Source changed")
    for entry in inputs["bindings"].values():
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("Input changed")
    observations = [idle_observation()]
    time.sleep(2)
    observations.append(idle_observation())
    if not all(o["idle"] for o in observations):
        raise RuntimeError("GPU not idle; no work launched")
    settings = configure_fp32_determinism(42)
    torch.set_num_threads(4)
    graphs = torch.load(inputs["bindings"]["train_probe"]["path"],map_location="cpu",weights_only=False)
    if len(graphs) != 4096 or [int(g.source_idx) for g in graphs] != inputs["source_idx"] or any(
            not 0 <= int(g.source_idx) < 500000 or any(k in g for k in FORBIDDEN_MODEL_FIELDS) for g in graphs):
        raise ValueError("Frozen pure2D train-only members differ")
    initial = torch.load(inputs["bindings"]["initial"]["path"],map_location="cpu",weights_only=True)["model_state"]
    if state_dict_sha256(initial) != inputs["bindings"]["initial"]["tensor_sha256"]:
        raise ValueError("Initialization changed")
    norm = torch.load(inputs["bindings"]["normalization_checkpoint"]["path"],map_location="cpu",weights_only=True)
    mean,std = float(norm["mean"]),float(norm["std"])
    del norm
    batches = list(DataLoader(graphs[:1920],batch_size=128,shuffle=False,num_workers=0))
    output.mkdir(parents=True)
    atomic_json(output / "runtime.json",dict(runtime=build_runtime_manifest(settings),source_commit=inputs["source_commit"],
        inputs_sha256=sha256_file(ATTEMPT / "inputs.json"),prospective_sha256=sha256_file(ATTEMPT / "rml/trajectory.json"),
        idle_observations=observations,train_decoded_source_idx=inputs["source_idx"],train_executed_source_idx=inputs["source_idx"][:1920],
        normalization=dict(mean=mean,std=std),scientific_training=False))
    def factory():
        model = make_model("neural_atom_k1")
        model.load_state_dict(initial,strict=True)
        return model
    @contextmanager
    def matched_gather(model,batch,device):
        with project_before_gather_context(model,batch,device):
            yield
            # Extra probe cases omit the baseline barrier; match it explicitly.
            torch.cuda.synchronize()
    @contextmanager
    def gather_control(model,batch,device):
        yield
        torch.cuda.synchronize()
    report = probe_k1_execution_efficiency(factory,batches,mean,std,device="cuda",**inputs["timing"],
        equivalence_steps=3,wall_seconds=started+inputs["worker_wall_limit_seconds"]-time.perf_counter(),atol=1e-6,rtol=1e-5,
        extra_cases={"cpu_layout":cpu_layout_context,"gather_control":gather_control,
                     "project_before_gather":matched_gather,"project_before_pack":project_before_pack_context})
    expected = {"baseline","non_blocking","no_redundant_sync","transfer_and_sync","cpu_layout","gather_control","project_before_gather","project_before_pack"}
    complete = (report["status"] == "complete" and set(report["cases"]) == expected and all(
        v["status"] in {"qualified","numerical_mismatch"} and len(v["timings"]) == 4 for v in report["cases"].values()))
    atomic_json(output / "result.json",report)
    atomic_json(output / "cost_observation.json",dict(worker_wall_seconds=time.perf_counter()-started,process_cpu_seconds=time.process_time()-cpu_started,
        gpu_busy_seconds=None,cost_scope="Whole execute body except module import/parent/tests; retained pre-collated batches"))
    atomic_json(output / "completion.json",dict(status="complete" if complete else "incomplete_case_diagnostic",
        artifacts={p.name:sha256_file(p) for p in output.iterdir() if p.is_file()}))
    if not complete:
        raise ValueError("Cases failed or incomplete; retain evidence, do not false-accept")
    print(json.dumps({k:dict(status=v["status"],exact=v["equivalence"]["exact"],seconds=v["median_seconds_per_step"],
        max_error=v["equivalence"]["max_abs_error"]) for k,v in report["cases"].items()}))


def close():
    output = ATTEMPT / "results"
    inputs,result,runtime,completion = read(ATTEMPT / "inputs.json"),read(output / "result.json"),read(output / "runtime.json"),read(output / "completion.json")
    if (completion["status"] != "complete" or result["status"] != "complete"
            or runtime["source_commit"] != inputs["source_commit"] or runtime["inputs_sha256"] != sha256_file(ATTEMPT / "inputs.json")
            or runtime["prospective_sha256"] != sha256_file(ATTEMPT / "rml/trajectory.json")
            or not all(o["idle"] for o in runtime["idle_observations"])):
        raise ValueError("Incomplete or mismatched diagnostic")
    for name,digest in completion["artifacts"].items():
        if sha256_file(output / name) != digest:
            raise ValueError("Output changed")
    for name,digest in inputs["files"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError("Executed source changed")
    for name in ("terminal_decision.md","attribution.md"):
        if not (ATTEMPT / name).exists():
            raise ValueError("Parent interpretation required")
    observation = read(output / "cost_observation.json")
    if observation["worker_wall_seconds"] > inputs["worker_wall_limit_seconds"]:
        raise ValueError("Worker bound exceeded")
    for name in ("baseline","cpu_layout","non_blocking","no_redundant_sync","transfer_and_sync","gather_control"):
        if not result["cases"][name]["equivalence"]["exact"]:
            raise ValueError("Unchanged control failed replay")
    decision = dict(outcome="NO_TRAIN",decision_ref=f"{REL}/terminal_decision.md",next_allowed_actions=[],
                    reopen_conditions=["Separately authorized numerical/native/quality qualification"])
    outcome = dict(execution_status="complete_projection_diagnostic",artifact_status="local_hash_verified",comparison_status="matched_cuda_scratch_only",
        scientific_status="NO_TRAIN",transfer_status="not_evaluated",budget_decision="bounded_local_diagnostic_complete",full_handoff_status="not_applicable")
    cost = deepcopy(read(ATTEMPT / "plan_input.json")["costs"][0])
    cost.update(cost_event_id="cost-k1-projection-observed",category="other",evidence_ref=f"{REL}/acceptance.json")
    cost["measurement"] = dict(wall_hours=dict(value=observation["worker_wall_seconds"]/3600,status="measured"),
        cpu_hours=dict(value=observation["process_cpu_seconds"]/3600,status="measured"),device_hours=dict(value=None,status="measurement_missing"),
        queue_hours=dict(value=None,status="not_applicable"))
    roles = []
    for role,rows,accesses in (("train_decoded",inputs["source_idx"],("labels_read",)),
        ("train_executed",inputs["source_idx"][:1920],("prediction_input","metric_computed"))):
        digest = hashlib.sha256(b"".join(struct.pack("<q",i) for i in rows)).hexdigest()
        for access in accesses:
            roles.append(dict(schema="molgap-role-event-v1",role_event_id=f"role-k1-projection-{role}-{access}",trajectory_id=TID,action_id="A001",run_id=RUN,
                dataset_identity="pcqm4mv2-ogb-fixed-500k-scnet-v1",row_manifest_hash=digest,role_name=role,access_kind=access,
                selection_used=False,evidence_ref=f"{REL}/acceptance.json"))
    role_use = dict(train_decoded="labels_read",train_executed="prediction_input",development="untouched",official_validation="untouched",test_dev="untouched",test_challenge="untouched")
    atomic_json(ATTEMPT / "acceptance.json",dict(evidence_id=EID,run_id=RUN,outcome=outcome,trajectory_decision=decision,role_use=role_use,costs=[cost],roles=roles,
        training_replay_ready=False,generalization_accuracy_qualified=False))
    files = [p for p in ATTEMPT.rglob("*") if p.is_file() and "rml" not in p.relative_to(ATTEMPT).parts and p.name not in ("terminal.json","closure_receipt.json")]
    hashes = {p.relative_to(ROOT).as_posix():sha256_file(p) for p in files}
    evidence = dict(format="molgap-v5-evidence-envelope-v1",contract="MOLGAP-COMMON-V5-FINAL",evidence_id=EID,track="B",scope="k1_projection_execution",
        legacy_contract="pcqm-k1-projection-execution-v1",outcome=outcome,role_use=role_use,
        authority=dict(pointers=[f"{REL}/{n}" for n in ("README.md","terminal_decision.md","attribution.md","acceptance.json")]),
        migration=dict(migrated_at=datetime.now(timezone.utc).date().isoformat(),training_executed=False,inference_executed=False,scientific_reinterpretation=False,
            verification_scope="Metadata closure of separately executed prospective projection diagnostic"),
        observed_execution=dict(training_executed=False,inference_executed=True,training_replay_ready=False,execution_ref=f"{REL}/results/result.json"),
        artifacts=[dict(name=p.removeprefix(REL+"/"),locator=p,sha256=h,availability="locally_retained_hash_verified") for p,h in hashes.items()])
    atomic_json(ATTEMPT / "terminal.json",dict(format="molgap-rml-terminal-package-v1",trajectory_id=TID,run_id=RUN,action_id="A001",
        finalized_at=datetime.now(timezone.utc).isoformat(),acceptance_ref=f"{REL}/acceptance.json",artifact_hashes=hashes,evidence=evidence,decision=decision,costs=[cost],roles=roles))
    atomic_json(ATTEMPT / "closure_receipt.json",finalize(ROOT,f"{REL}/rml",f"{REL}/terminal.json"))
    print(EID+" finalized NO_TRAIN")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("operation",choices=("prepare","execute","close"))
    {"prepare":prepare,"execute":execute,"close":close}[parser.parse_args().operation]()
