"""CPU loader adapter using existing identity, loader, IO and RML owners."""
import os
# The dedicated process must hide CUDA before any family/Torch imports.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

from molgap.research_memory.plan import plan
from molgap.saved_diagnostic_closure import close_saved_diagnostic
from molgap.training_reproducibility import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ATTEMPT = HERE / "loader_attempt_001"
REL = ATTEMPT.relative_to(ROOT).as_posix()
TID = "TB-k1-loader-cpu-20261010"
RUN = "k1-loader-cpu-20261010"
COST_ID = "cost-k1-loader-cpu-estimate"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare():
    if (ATTEMPT / "rml").exists():
        raise FileExistsError("Prospective already exists; reconcile")
    parent = read(HERE / "gpu_attempt_001/inputs.json")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    files = [*parent["files"], Path(__file__).relative_to(ROOT).as_posix(),
             "src/molgap/k1_loader_diagnostic.py", "src/molgap/saved_diagnostic_closure.py",
             f"{REL}/README.md", f"{REL}/protocol.md", f"{REL}/decision.md"]
    hashes = {n: hashlib.sha256(subprocess.check_output(["git", "show", f"{head}:{n}"], cwd=ROOT)).hexdigest() for n in files}
    if any(sha256_file(ROOT / n) != h for n, h in hashes.items()):
        raise ValueError("Executable Git bytes differ")
    binding = parent["bindings"]["train_probe"]
    if sha256_file(Path(binding["path"])) != binding["sha256"]:
        raise ValueError("TRAIN probe changed")
    inputs = dict(kind="loader_metadata", cpu_threads=4, ceiling_seconds=300,
                  executed_source_files=hashes, snapshot_files={}, source_commit=head,
                  train_probe=binding, source_idx=parent["source_idx"], development_bounds=[0, 0])
    atomic_json(ATTEMPT / "inputs.json", inputs)
    atomic_json(ATTEMPT / "role_plan.json", dict(train_rows_decoded=4096, labels_read=True,
                prediction_executed=False, model_execution=False, protected_roles="untouched"))
    spec = deepcopy(read(HERE / "checkpoint_io_attempt_001/plan_input.json"))
    prior = ["pcqm-k1-execution-efficiency-gpu-20261010", "pcqm-k1-joint-execution-20261010"]
    t = spec["trajectory"]
    t.update(trajectory_id=TID, family_id="k1-loader-attribution", comparison_readiness_ref=f"{REL}/README.md",
             question="Does actual CPU graph collation or worker startup dominate the previously excluded input phase?")
    t["hypothesis"].update(hypothesis_id="H-" + TID.removeprefix("TB-"), supporting_evidence_ids=prior,
        observed_deficiency="Accepted scratch step timing excluded actual loader startup/fetch; CPU contribution remains unknown.",
        changed_mechanism="Compare original loader worker counts0/2 without GPU/model execution",
        alternative_explanations=["Windows spawn affects startup only", "Graph collation is small", "Pinning and GPU-consumer overlap change native ranking"],
        cheapest_falsifier="Four balanced actual train-only CPU loader windows with exact batch hashes",
        decision_changed_if_positive="Nominate native loader phase qualification; no default worker change",
        decision_changed_if_negative="Deprioritize loader rewrite; retain default and investigate launch/sync",
        expected_native_cost_ref=COST_ID)
    t["state_at_start"].update(source_commit=head, source_config_identity=sha256_file(ATTEMPT / "inputs.json"),
        contract_refs=[f"{REL}/README.md", f"{REL}/inputs.json"], parent_trajectory_ids=["TB-k1-joint-execution-20261010"],
        prior_evidence_ids=prior, role_snapshot_refs=[f"{REL}/role_plan.json"], budget_snapshot_ref=f"{REL}/README.md")
    if ATTEMPT.name == "loader_attempt_002":
        t["state_at_start"]["parent_trajectory_ids"].append("TB-k1-loader-cpu-20261010")
        t["hypothesis"]["observed_deficiency"] += " Attempt001 lost check details at final rejection; preserve its failure attribution."
    elif ATTEMPT.name == "loader_attempt_003":
        t["state_at_start"]["parent_trajectory_ids"].extend([
            "TB-k1-loader-cpu-20261010", "TB-k1-loader-cpu-20261010-attempt002"])
        t["hypothesis"]["observed_deficiency"] += " Attempt002 retained exact batches but observed unexpected CUDA initialization under empty visibility."
    t["actions"] = [dict(action_id="A001", type="execution_profile", run_ids=[RUN], attempt_ids=[ATTEMPT.name.replace("_", "-")],
                         source_commit=head, evidence_refs=[f"{REL}/inputs.json"], cost_event_ids=[COST_ID])]
    t["decision"].update(outcome="ACTIVE", decision_ref=f"{REL}/decision.md", next_allowed_actions=["Bounded TRAIN-only CPU collation"])
    spec["decision_state"].update(source_commit=head, state_timestamp=datetime.now(timezone.utc).isoformat(),
        role_snapshot_refs=[f"{REL}/role_plan.json"], budget_snapshot_ref=f"{REL}/README.md")
    cost = spec["costs"][0]
    cost.update(cost_event_id=COST_ID, trajectory_id=TID, run_id=RUN, attempt_id=ATTEMPT.name.replace("_", "-"),
                hardware="CPU four intra-op threads and optional owned workers; no accelerator", evidence_ref=f"{REL}/README.md")
    cost["measurement"]["wall_hours"] = dict(value=300 / 3600, status="estimated")
    spec["costs"] = [cost]
    atomic_json(ATTEMPT / "plan_input.json", spec)
    atomic_json(ATTEMPT / "plan_receipt.json", plan(ROOT, spec, f"{REL}/rml"))
    print(TID + " prospective frozen")


def execute():
    started, cpu_started = time.perf_counter(), time.process_time()
    import torch
    from molgap.k1_loader_diagnostic import profile_loader_fetches
    from molgap.pcqm_500k_v4_evidence import loader
    from molgap.k1_screen_training import FORBIDDEN_MODEL_FIELDS, _batch_sha256
    inputs, trajectory = read(ATTEMPT / "inputs.json"), read(ATTEMPT / "rml/trajectory.json")
    output = ATTEMPT / "results"
    if output.exists() or trajectory["record_mode"] != "prospective" or trajectory["decision"]["outcome"] != "ACTIVE":
        raise ValueError("Fresh active prospective required")
    for n, h in inputs["executed_source_files"].items():
        if sha256_file(ROOT / n) != h:
            raise ValueError("Source changed")
    binding = inputs["train_probe"]
    if sha256_file(Path(binding["path"])) != binding["sha256"]:
        raise ValueError("TRAIN artifact changed")
    torch.set_num_threads(4)
    graphs = torch.load(binding["path"], map_location="cpu", weights_only=False)
    observed = [int(g.source_idx) for g in graphs]
    if observed != inputs["source_idx"] or len(graphs) != 4096 or any(
            not 0 <= int(g.source_idx) < 500000 or any(k in g for k in FORBIDDEN_MODEL_FIELDS) for g in graphs):
        raise ValueError("Training-only membership/schema changed")
    if torch.cuda.is_initialized() or torch.cuda.is_available() or torch.cuda.device_count() != 0:
        raise ValueError("CPU-only process did not actually hide CUDA")

    def factory(workers):
        def create():
            os.environ["MOLGAP_V4_LOADER_WORKERS"] = str(workers)
            return loader(graphs)
        return create

    previous = os.environ.get("MOLGAP_V4_LOADER_WORKERS")
    try:
        report, batches = profile_loader_fetches({"workers0": factory(0), "workers2": factory(2)}, wall_seconds=250)
    finally:
        if previous is None:
            os.environ.pop("MOLGAP_V4_LOADER_WORKERS", None)
        else:
            os.environ["MOLGAP_V4_LOADER_WORKERS"] = previous
    checks = dict(batch_membership_exact=True, batch_tensor_bytes_exact=True, owned_workers_stopped=True,
                  cuda_uninitialized=not torch.cuda.is_initialized(),
                  cuda_unavailable=not torch.cuda.is_available() and torch.cuda.device_count() == 0)
    reference = None
    for windows in batches.values():
        for values in windows:
            rows = [int(i) for b in values for i in b.source_idx.view(-1).tolist()]
            checks["batch_membership_exact"] &= rows == observed and len(values) == 32
            hashes = [_batch_sha256(b) for b in values]
            if reference is None:
                reference = hashes
            checks["batch_tensor_bytes_exact"] &= hashes == reference
    checks["owned_workers_stopped"] = all(c["workers_stopped"] for w in report["windows"] for c in w.values())
    parent_cpu = time.process_time() - cpu_started
    child_cpu = report["worker_process_cpu_seconds"]
    wall = time.perf_counter() - started
    accepted = all(checks.values()) and wall <= 300
    result = dict(format="molgap-k1-saved-analysis-v1", kind=inputs["kind"],
        status="complete" if accepted else "mechanically_rejected", checks=checks,
        inputs_sha256=sha256_file(ATTEMPT / "inputs.json"), prospective_sha256=sha256_file(ATTEMPT / "rml/trajectory.json"),
        wall_seconds=wall, process_cpu_seconds=parent_cpu, worker_process_cpu_seconds=child_cpu,
        total_process_cpu_seconds=None,
        observed_parent_plus_worker_cpu_seconds=None if child_cpu is None else parent_cpu + child_cpu,
        runtime=dict(device="cpu", cpu_threads=4, torch=torch.__version__, cuda_initialized=torch.cuda.is_initialized(),
                     cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"], cuda_available=torch.cuda.is_available(),
                     cuda_device_count=torch.cuda.device_count(), effective_pin_memory=False), observed_source_idx=observed,
        training_executed=False, inference_executed=False, optimizer_created=False, gradients_computed=False,
        loader_profile=report, batch_hashes=reference,
        limits=["Windows CPU fetch without GPU pinning/consumer; not native T4/A100 training throughput",
                "Worker CPU sampled before shutdown; teardown CPU not included in observed sum"])
    atomic_json(output / "result.json", result)
    atomic_json(output / "completion.json", dict(complete=accepted, inputs_sha256=result["inputs_sha256"],
                files={"result.json": sha256_file(output / "result.json")}))
    if not accepted:
        raise ValueError(f"CPU loader rejection retained: checks={checks}, wall={wall:.3f}s")
    print(json.dumps(report["medians"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("operation", choices=("prepare", "execute", "close"))
    p.add_argument("--attempt", choices=("loader_attempt_001", "loader_attempt_002", "loader_attempt_003"), default="loader_attempt_001")
    args = p.parse_args()
    if args.attempt != "loader_attempt_001":
        ATTEMPT = HERE / args.attempt
        REL = ATTEMPT.relative_to(ROOT).as_posix()
        suffix = "-attempt" + args.attempt.rsplit("_", 1)[1]
        TID += suffix
        RUN += suffix
        COST_ID += suffix
    {"prepare": prepare, "execute": execute,
     "close": lambda: print(close_saved_diagnostic(ROOT, ATTEMPT))}[args.operation]()
