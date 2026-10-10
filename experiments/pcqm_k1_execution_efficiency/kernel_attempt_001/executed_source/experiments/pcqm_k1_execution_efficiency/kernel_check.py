"""Isolated CUPTI attribution; existing K1 execution and RML owners do the work."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from molgap.research_memory.plan import plan
from molgap.training_reproducibility import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ATTEMPT = HERE / "kernel_attempt_001"
REL = ATTEMPT.relative_to(ROOT).as_posix()
TID = "TB-k1-kernel-trace-20261010"
RUN = "k1-kernel-rtx5060-20261010"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare():
    if (ATTEMPT / "rml").exists():
        raise FileExistsError("Already planned; reconcile")
    inputs = deepcopy(read(HERE / "tf32_attempt_001/inputs.json"))
    spec = deepcopy(read(HERE / "tf32_attempt_001/plan_input.json"))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    files = [*inputs["files"], "src/molgap/cupti_activity.py",
             Path(__file__).relative_to(ROOT).as_posix(), f"{REL}/README.md", f"{REL}/decision.md"]
    inputs.update(source_commit=head, files={n: sha256_file(ROOT / n) for n in files},
                  worker_wall_limit_seconds=300, train_executed_source_idx=inputs["source_idx"][:256])
    paths = {"bundled": Path(sys.executable).parents[1] / "Lib/site-packages/torch/lib/cupti64_2025.1.0.dll",
             "isolated": ROOT / ".tmp/profiler_tools/cuda_cupti-windows-x86_64-12.9.79-archive/lib/cupti64_2025.2.1.dll"}
    inputs["profilers"] = {k: dict(path=str(p), sha256=sha256_file(p)) for k, p in paths.items()}
    inputs["sdk"] = dict(url="https://developer.download.nvidia.com/compute/cuda/redist/cuda_cupti/windows-x86_64/cuda_cupti-windows-x86_64-12.9.79-archive.zip",
        sha256="c96018456dc4db6af6d69d6db0170a2ccc656ddde8c8ce6ee05682c2c5569daa")
    atomic_json(ATTEMPT / "inputs.json", inputs)
    atomic_json(ATTEMPT / "role_plan.json", dict(train_decoded=4096, train_execution=256,
        selection_used=False, development="untouched", protected="untouched"))
    prior = ["pcqm-k1-tf32-execution-20261010", "pcqm-k1-projection-execution-20261010"]
    t = spec["trajectory"]
    t.update(trajectory_id=TID, family_id="k1-kernel-attribution", comparison_readiness_ref=f"{REL}/README.md",
             question="Can isolated CUPTI retain actual K1 FP32/TF32 GPU kernel attribution after bundled profiler failure?")
    t["hypothesis"].update(hypothesis_id="H-k1-kernel-trace-20261010", supporting_evidence_ids=prior,
        observed_deficiency="TF32 timings/output perturbation exist; actual CUDA dispatch remains unobserved due to INVALID_DEVICE.",
        changed_mechanism="Only isolated concurrent-kernel tracer/library; unchanged model and owner step",
        alternative_explanations=["Bundled tracer compatibility", "Driver/device tracing unsupported", "TF32 permission falls back on small GEMMs"],
        cheapest_falsifier="Four bounded fresh-process trace cases with explicit profiler hashes",
        decision_changed_if_positive="Attribute raw dispatch; no precision/native/quality adoption",
        decision_changed_if_negative="Record specific tracing limitation; do not treat CPU traces as CUDA metrics",
        expected_native_cost_ref="cost-k1-kernel-estimate")
    t["state_at_start"].update(source_commit=head, source_config_identity=sha256_file(ATTEMPT / "inputs.json"),
        contract_refs=[f"{REL}/README.md", f"{REL}/inputs.json"], parent_trajectory_ids=["TB-k1-tf32-execution-20261010"],
        prior_evidence_ids=prior, role_snapshot_refs=[f"{REL}/role_plan.json"], budget_snapshot_ref=f"{REL}/README.md")
    t["actions"] = [dict(action_id="A001", type="execution_profile", run_ids=[RUN], attempt_ids=["kernel-attempt-001"],
        source_commit=head, evidence_refs=[f"{REL}/inputs.json"], cost_event_ids=["cost-k1-kernel-estimate"])]
    t["decision"].update(decision_ref=f"{REL}/decision.md", next_allowed_actions=["Bounded isolated idle-only kernel tracing"])
    spec["decision_state"].update(source_commit=head, state_timestamp=datetime.now(timezone.utc).isoformat(),
        budget_snapshot_ref=f"{REL}/README.md", role_snapshot_refs=[f"{REL}/role_plan.json"])
    cost = spec["costs"][0]
    cost.update(cost_event_id="cost-k1-kernel-estimate", trajectory_id=TID, run_id=RUN,
                attempt_id="kernel-attempt-001", evidence_ref=f"{REL}/README.md")
    for unit in ("device_hours", "wall_hours"):
        cost["measurement"][unit] = dict(value=300/3600, status="estimated")
    atomic_json(ATTEMPT / "plan_input.json", spec)
    atomic_json(ATTEMPT / "plan_receipt.json", plan(ROOT, spec, f"{REL}/rml"))
    print(TID + " prospective frozen")


def worker(case):
    started, cpu_started = time.perf_counter(), time.process_time()
    inputs = read(ATTEMPT / "inputs.json")
    trajectory = read(ATTEMPT / "rml/trajectory.json")
    if trajectory["record_mode"] != "prospective" or trajectory["decision"]["outcome"] != "ACTIVE":
        raise ValueError("Active prospective required even for a child worker")
    for name, digest in inputs["files"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError("Worker source changed")
    for binding in (*inputs["bindings"].values(), *inputs["profilers"].values()):
        if sha256_file(Path(binding["path"])) != binding["sha256"]:
            raise ValueError("Worker input/profiler bytes changed")
    import torch
    from torch_geometric.loader import DataLoader
    from molgap.cupti_activity import CuptiActivityCollector
    from molgap.k1_precision_diagnostic import precision_case
    from molgap.k1_screen_training import FORBIDDEN_MODEL_FIELDS
    from molgap.pcqm_500k_v4_evidence import make_model, optimizer_for, step
    from molgap.training_reproducibility import configure_fp32_determinism, build_runtime_manifest
    from molgap.v4_runtime import state_dict_sha256
    from run import idle_observation
    profiler, precision = case.split("_")
    tf32 = precision == "tf32"
    result = dict(case=case, status="unavailable", phase="idle_check", source_commit=inputs["source_commit"],
        inputs_sha256=sha256_file(ATTEMPT / "inputs.json"), prospective_sha256=sha256_file(ATTEMPT / "rml/trajectory.json"),
        train_decoded_source_idx=[], train_executed_source_idx=[], warmup_steps=0, trace_steps=0,
        profiler=inputs["profilers"][profiler], scientific_training=False, throughput_measurement=False)
    try:
        idle = idle_observation()
        result["idle_observation"] = idle
        if not idle["idle"]:
            raise RuntimeError("GPU occupied; no accelerator work")
        settings = configure_fp32_determinism(42)
        torch.set_num_threads(4)
        result["runtime"] = build_runtime_manifest(settings)
        result["tf32_allowed"] = tf32
        result["phase"] = "decode"
        graphs = torch.load(inputs["bindings"]["train_probe"]["path"], map_location="cpu", weights_only=False)
        result["train_decoded_source_idx"] = [int(g.source_idx) for g in graphs]
        if result["train_decoded_source_idx"] != inputs["source_idx"] or any(
            not 0 <= int(g.source_idx) < 500000 or any(k in g for k in FORBIDDEN_MODEL_FIELDS) for g in graphs):
            raise ValueError("Frozen train-only graphs differ")
        initial = torch.load(inputs["bindings"]["initial"]["path"], map_location="cpu", weights_only=True)["model_state"]
        if state_dict_sha256(initial) != inputs["bindings"]["initial"]["tensor_sha256"]:
            raise ValueError("Initialization differs")
        norm = torch.load(inputs["bindings"]["normalization_checkpoint"]["path"], map_location="cpu", weights_only=True)
        mean, std = float(norm["mean"]), float(norm["std"])
        del norm
        batches = list(DataLoader(graphs[:256], batch_size=128, shuffle=False, num_workers=0))
        model = make_model("neural_atom_k1")
        model.load_state_dict(initial, strict=True)
        model = model.cuda().train()
        optimizer = optimizer_for(model)
        result["phase"] = "warmup"
        def update(index):
            if time.perf_counter()-started > 80:
                raise TimeoutError("Worker cooperative ceiling exhausted")
            data = batches[index % 2].clone()
            observed = data.source_idx.tolist()
            result["train_executed_source_idx"] = sorted(set(result["train_executed_source_idx"]) | set(observed))
            with precision_case(tf32=tf32, layout=True)(model, data, "cuda"):
                step(model, optimizer, data.cuda(), mean, std)
            torch.cuda.synchronize()
        for i in range(3):
            update(i)
            result["warmup_steps"] += 1
        trace = CuptiActivityCollector(Path(result["profiler"]["path"]),
            synchronize=torch.cuda.synchronize, isolated_process=True)
        result["phase"] = "cupti_trace"
        with trace:
            for i in range(2):
                update(i)
                result["trace_steps"] += 1
        result["trace"] = trace.report()
        result["status"] = "trace_available" if result["trace"].get("records") else "unavailable"
        result["phase"] = "complete"
    except Exception as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        if "trace" in locals() and hasattr(trace, "diagnostics"):
            result["partial_trace"] = trace.diagnostics()
    finally:
        result.update(worker_wall_seconds=time.perf_counter()-started, process_cpu_seconds=time.process_time()-cpu_started)
        atomic_json(ATTEMPT / "results" / (case + ".json"), result)
    print(json.dumps(dict(case=case, status=result["status"], error=result.get("error"))))


def execute():
    started = time.perf_counter()
    inputs, trajectory = read(ATTEMPT / "inputs.json"), read(ATTEMPT / "rml/trajectory.json")
    if (ATTEMPT / "results").exists():
        raise FileExistsError("Attempt exists; reconcile, never repeat")
    if trajectory["record_mode"] != "prospective" or trajectory["decision"]["outcome"] != "ACTIVE":
        raise ValueError("Active prospective required")
    for name, digest in inputs["files"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError("Executed source changed")
    for binding in (*inputs["bindings"].values(), *inputs["profilers"].values()):
        if sha256_file(Path(binding["path"])) != binding["sha256"]:
            raise ValueError("Input/profiler bytes changed")
    cases = [p + "_" + f for p in ("bundled", "isolated") for f in ("fp32", "tf32")]
    for case in cases:
        remaining = inputs["worker_wall_limit_seconds"] - (time.perf_counter()-started)
        if remaining < 5:
            raise TimeoutError("Bounded diagnostic exhausted")
        child = subprocess.run([sys.executable, str(Path(__file__)), "worker", "--case", case],
            cwd=ROOT, capture_output=True, text=True, timeout=min(90, remaining))
        atomic_json(ATTEMPT / "results" / (case + "_process.json"), dict(returncode=child.returncode,
            stdout=child.stdout, stderr=child.stderr))
        if child.returncode or not (ATTEMPT / "results" / (case + ".json")).exists():
            raise RuntimeError("Child failed; retain partial evidence")
        row = read(ATTEMPT / "results" / (case + ".json"))
        if (row["inputs_sha256"] != sha256_file(ATTEMPT / "inputs.json")
                or row["prospective_sha256"] != sha256_file(ATTEMPT / "rml/trajectory.json")
                or row["source_commit"] != inputs["source_commit"]
                or row.get("idle_observation", {}).get("idle") is not True
                or row["train_decoded_source_idx"] != inputs["source_idx"]
                or row["train_executed_source_idx"] != sorted(inputs["train_executed_source_idx"])
                or row["warmup_steps"] != 3):
            raise ValueError("Case scope/runtime incomplete; no accepted trace result")
        expected_trace_error = (row["phase"] == "cupti_trace"
            and "CUPTI" in row.get("error", {}).get("message", "").upper())
        empty_completed_trace = (row["phase"] == "complete" and row["trace_steps"] == 2
            and not row.get("trace", {}).get("records") and "error" not in row)
        if row["status"] != "trace_available" and not (expected_trace_error or empty_completed_trace):
            raise ValueError("Unexpected infrastructure/workload failure, not profiler unavailability")
        if row["status"] == "trace_available" and row["trace_steps"] != 2:
            raise ValueError("Incomplete traced workload")
        print(child.stdout.strip(), flush=True)
    result = dict(status="complete_diagnostic", cases=cases, worker_wall_seconds=time.perf_counter()-started,
        process_cpu_seconds=sum(read(ATTEMPT / "results" / (c + ".json"))["process_cpu_seconds"] for c in cases),
        cpu_cost_scope="Measured child CPU only; parent CPU unavailable", inputs_sha256=sha256_file(ATTEMPT / "inputs.json"),
        prospective_sha256=sha256_file(ATTEMPT / "rml/trajectory.json"), scientific_training=False)
    atomic_json(ATTEMPT / "results/result.json", result)
    atomic_json(ATTEMPT / "results/completion.json", dict(complete=True,
        files={p.name: sha256_file(p) for p in (ATTEMPT / "results").iterdir() if p.is_file()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("prepare", "execute", "worker"))
    parser.add_argument("--case", choices=("bundled_fp32", "bundled_tf32", "isolated_fp32", "isolated_tf32"))
    args = parser.parse_args()
    if args.operation == "worker":
        if args.case is None:
            parser.error("Worker requires case")
        worker(args.case)
    else:
        {"prepare": prepare, "execute": execute}[args.operation]()
