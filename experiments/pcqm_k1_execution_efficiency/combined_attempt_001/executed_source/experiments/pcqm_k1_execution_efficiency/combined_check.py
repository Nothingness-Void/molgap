"""Joint execution custody; reuse owner factory/step, probe and RML closure."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

from molgap.research_memory.plan import plan
from molgap.training_reproducibility import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ATTEMPT = HERE / "combined_attempt_001"
REL = ATTEMPT.relative_to(ROOT).as_posix()
TID = "TB-k1-joint-execution-20261010"
RUN = "k1-joint-execution-rtx5060-20261010"
EID = "pcqm-k1-joint-execution-20261010"
CONTROLS = ("baseline", "non_blocking", "no_redundant_sync", "transfer_and_sync", "cpu_layout")
VARIANTS = ("fused_layout", "fused_tf32_layout")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare():
    if (ATTEMPT / "rml").exists():
        raise FileExistsError("Already planned; reconcile, never overwrite")
    inputs = deepcopy(read(HERE / "gpu_attempt_001/inputs.json"))
    spec = deepcopy(read(HERE / "gpu_attempt_001/plan_input.json"))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    names = list(dict.fromkeys([*inputs["files"], "src/molgap/k1_precision_diagnostic.py",
        "src/molgap/cuda_execution_closure.py", Path(__file__).relative_to(ROOT).as_posix(),
        f"{REL}/README.md", f"{REL}/decision.md"]))
    hashes = {n:hashlib.sha256(subprocess.check_output(["git","show",f"{head}:{n}"],cwd=ROOT)).hexdigest() for n in names}
    if any(sha256_file(ROOT/n) != digest for n,digest in hashes.items()):
        raise ValueError("Committed and executed source bytes differ")
    inputs.update(source_commit=head,files=hashes,worker_wall_limit_seconds=600,
        optimizer_case_flags={n:dict(foreach=False,fused=True) for n in VARIANTS})
    for binding in inputs["bindings"].values():
        if sha256_file(Path(binding["path"])) != binding["sha256"]:
            raise ValueError("Retained input changed")
    atomic_json(ATTEMPT/"inputs.json",inputs)
    atomic_json(ATTEMPT/"role_plan.json",dict(train_decoded=4096,train_executed=1920,
        development="untouched",protected_roles="untouched",selection_used=False))
    prior = ["pcqm-k1-execution-efficiency-gpu-20261010","pcqm-k1-tf32-execution-20261010","pcqm-k1-kernel-trace-20261010"]
    cost_id = "cost-k1-joint-execution-estimate"
    t = spec["trajectory"]
    t.update(trajectory_id=TID,family_id="k1-joint-execution",comparison_readiness_ref=f"{REL}/README.md",
        question="Do CPU layout, fused AdamW and TF32 retain their local gains when combined, and how does scratch update drift change?")
    t["hypothesis"].update(hypothesis_id="H-k1-joint-execution-20261010",supporting_evidence_ids=prior,
        observed_deficiency="Isolated gains and scratch drift do not establish combined cost or numerical interaction",
        changed_mechanism="Joint layout/fused and joint layout/fused/matmul TF32; unchanged factory and owner step",
        alternative_explanations=["Gains overlap", "Fused optimizer exposes graph/launch bottlenecks", "Combined numeric drift grows"],
        cheapest_falsifier="Three scratch comparisons and four matched short timing windows",
        decision_changed_if_positive="Nominate a jointly measured separate execution contract; no quality/default admission",
        decision_changed_if_negative="Prefer simpler isolated candidate; no additive gain claim",
        expected_native_cost_ref=cost_id)
    t["state_at_start"].update(source_commit=head,source_config_identity=sha256_file(ATTEMPT/"inputs.json"),
        contract_refs=[f"{REL}/README.md",f"{REL}/inputs.json"],
        parent_trajectory_ids=["TB-k1-tf32-execution-20261010"],prior_evidence_ids=prior,
        role_snapshot_refs=[f"{REL}/role_plan.json"],budget_snapshot_ref=f"{REL}/README.md")
    t["actions"] = [dict(action_id="A001",type="execution_profile",run_ids=[RUN],attempt_ids=["combined-attempt-001"],
        source_commit=head,evidence_refs=[f"{REL}/inputs.json"],cost_event_ids=[cost_id])]
    t["decision"].update(decision_ref=f"{REL}/decision.md",next_allowed_actions=["Bounded idle-only joint execution diagnostic"])
    spec["decision_state"].update(source_commit=head,state_timestamp=datetime.now(timezone.utc).isoformat(),
        role_snapshot_refs=[f"{REL}/role_plan.json"],budget_snapshot_ref=f"{REL}/README.md")
    spec["costs"][0].update(cost_event_id=cost_id,trajectory_id=TID,run_id=RUN,attempt_id="combined-attempt-001",evidence_ref=f"{REL}/README.md")
    atomic_json(ATTEMPT/"plan_input.json",spec)
    atomic_json(ATTEMPT/"plan_receipt.json",plan(ROOT,spec,f"{REL}/rml"))
    print(TID+" prospective frozen")


def execute():
    started,cpu_started = time.perf_counter(),time.process_time()
    import torch
    from torch_geometric.loader import DataLoader
    from molgap.k1_execution_efficiency import probe_k1_execution_efficiency
    from molgap.k1_precision_diagnostic import precision_case, qualify_combined_report
    from molgap.k1_screen_training import FORBIDDEN_MODEL_FIELDS
    from molgap.pcqm_500k_v4_evidence import make_model
    from molgap.training_reproducibility import build_runtime_manifest,configure_fp32_determinism
    from molgap.v4_runtime import state_dict_sha256
    from run import idle_observation

    output = ATTEMPT/"results"
    if output.exists():
        raise FileExistsError("Attempt exists; reconcile")
    inputs,t = read(ATTEMPT/"inputs.json"),read(ATTEMPT/"rml/trajectory.json")
    if t["record_mode"] != "prospective" or t["decision"]["outcome"] != "ACTIVE":
        raise ValueError("Active prospective required")
    for n,h in inputs["files"].items():
        if sha256_file(ROOT/n) != h:
            raise ValueError("Frozen source changed")
    for binding in inputs["bindings"].values():
        if sha256_file(Path(binding["path"])) != binding["sha256"]:
            raise ValueError("Frozen input changed")
    observations = [idle_observation()]
    time.sleep(2)
    observations.append(idle_observation())
    if not all(o["idle"] for o in observations):
        raise RuntimeError("GPU not idle; no work launched; no monitor")
    settings = configure_fp32_determinism(42)
    if torch.cuda.get_device_capability()[0] < 8:
        raise ValueError("TF32 unsupported on this device")
    torch.set_num_threads(4)
    graphs = torch.load(inputs["bindings"]["train_probe"]["path"],map_location="cpu",weights_only=False)
    if len(graphs) != 4096 or [int(g.source_idx) for g in graphs] != inputs["source_idx"] or any(
            not 0 <= int(g.source_idx) < 500000 or any(k in g for k in FORBIDDEN_MODEL_FIELDS) for g in graphs):
        raise ValueError("TRAIN-only pure2D membership differs")
    initial = torch.load(inputs["bindings"]["initial"]["path"],map_location="cpu",weights_only=True)["model_state"]
    if state_dict_sha256(initial) != inputs["bindings"]["initial"]["tensor_sha256"]:
        raise ValueError("Initialization differs")
    norm = torch.load(inputs["bindings"]["normalization_checkpoint"]["path"],map_location="cpu",weights_only=True)
    mean,std = float(norm["mean"]),float(norm["std"])
    del norm
    batches = list(DataLoader(graphs[:1920],batch_size=128,shuffle=False,num_workers=0))
    output.mkdir(parents=True)
    atomic_json(output/"runtime.json",dict(runtime=build_runtime_manifest(settings),source_commit=inputs["source_commit"],
        inputs_sha256=sha256_file(ATTEMPT/"inputs.json"),prospective_sha256=sha256_file(ATTEMPT/"rml/trajectory.json"),
        idle_observations=observations,train_decoded_source_idx=inputs["source_idx"],train_executed_source_idx=inputs["source_idx"][:1920],
        normalization=dict(mean=mean,std=std),scientific_training=False))
    def factory():
        model = make_model("neural_atom_k1")
        model.load_state_dict(initial,strict=True)
        return model
    report = probe_k1_execution_efficiency(factory,batches,mean,std,device="cuda",**inputs["timing"],
        equivalence_steps=3,wall_seconds=started+inputs["worker_wall_limit_seconds"]-time.perf_counter(),atol=1e-6,rtol=1e-5,
        extra_cases=dict(cpu_layout=precision_case(layout=True),fused_layout=precision_case(layout=True),
                         fused_tf32_layout=precision_case(layout=True,tf32=True)),optimizer_case_flags=inputs["optimizer_case_flags"])
    for name,case in report["cases"].items():
        case["precision_contract_change"] = name == "fused_tf32_layout"
    complete = report["status"] == "complete" and set(report["cases"]) == set((*CONTROLS,*VARIANTS)) and all(
        c["status"] in {"qualified","numerical_mismatch"} and len(c["timings"]) == 4 for c in report["cases"].values())
    atomic_json(output/"result.json",report)
    if complete:
        atomic_json(output/"contrasts.json",qualify_combined_report(report))
    atomic_json(output/"cost_observation.json",dict(worker_wall_seconds=time.perf_counter()-started,process_cpu_seconds=time.process_time()-cpu_started,
        gpu_busy_seconds=None,cost_scope="Whole execute body; excludes parent/tests/closure; no loader throughput measurement"))
    atomic_json(output/"completion.json",dict(status="complete" if complete else "incomplete_case_diagnostic",
        artifacts={p.name:sha256_file(p) for p in output.iterdir() if p.is_file()}))
    if not complete:
        raise ValueError("Failed/incomplete cases; retain artifacts, never false-accept")
    print(json.dumps({n:dict(status=c["status"],ms=1000*c["median_seconds_per_step"],exact=c["equivalence"]["exact"],
        tensor_error=c["equivalence"]["max_abs_error"]) for n,c in report["cases"].items()}))


def close():
    from molgap.cuda_execution_closure import close_cuda_execution_diagnostic
    close_cuda_execution_diagnostic(ROOT,ATTEMPT,evidence_id=EID,trajectory_id=TID,run_id=RUN,
        scope="k1_joint_execution",legacy_contract="pcqm-k1-joint-execution-v1",exact_controls=CONTROLS,
        expected_cases=(*CONTROLS,*VARIANTS),executed_rows=1920)
    print(EID+" finalized NO_TRAIN")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("operation",choices=("prepare","execute","close"))
    {"prepare":prepare,"execute":execute,"close":close}[parser.parse_args().operation]()
