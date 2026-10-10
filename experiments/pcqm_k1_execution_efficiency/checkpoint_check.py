"""CPU checkpoint-copy custody; no family trainer or remote launch is imported."""
import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import time

from molgap.research_memory.plan import plan
from molgap.saved_diagnostic_closure import close_saved_diagnostic
from molgap.training_reproducibility import atomic_json, atomic_torch_save, sha256_file

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ATTEMPT = HERE / "checkpoint_io_attempt_001"
REL = ATTEMPT.relative_to(ROOT).as_posix()
TID = "TB-k1-checkpoint-io-20261010"
RUN = "k1-checkpoint-io-cpu-20261010"
OWNER = Path("D:/w/k1-colab-500k-efficient/src/molgap/colab_k1_screen.py")
OWNER_SHA = "ac13918013015c27b1f46020b0eabf2b4f7ad0eb18bb091a9e6033b94948dc34"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare():
    if (ATTEMPT / "rml").exists():
        raise FileExistsError("Already planned; reconcile")
    if sha256_file(OWNER) != OWNER_SHA:
        raise ValueError("Inspected sender source differs")
    source = ATTEMPT / "source_snapshot/colab_k1_screen.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OWNER, source)
    checkpoint = ROOT / "platforms/_records/local/staging/k1-execution-efficiency/cuda-scratch.pt"
    receipt = read(HERE / "gpu_attempt_001/results/checkpoint_resume.json")["publication"]
    if sha256_file(checkpoint) != receipt["sha256"]:
        raise ValueError("Retained checkpoint differs")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    files = [Path(__file__).relative_to(ROOT).as_posix(), "src/molgap/checkpoint_snapshot.py",
             "src/molgap/training_reproducibility.py", "src/molgap/k1_execution_efficiency.py",
             "src/molgap/saved_diagnostic_closure.py", f"{REL}/README.md", f"{REL}/decision.md"]
    hashes = {n:hashlib.sha256(subprocess.check_output(["git", "show", f"{head}:{n}"], cwd=ROOT)).hexdigest() for n in files}
    if any(sha256_file(ROOT/n) != digest for n,digest in hashes.items()):
        raise ValueError("Commit bytes and executable checkout differ")
    inputs = dict(kind="checkpoint_io", cpu_threads=4, ceiling_seconds=180, repeats=3,
        sizes=[0,128,8192,32768,65536], source_commit=head, executed_source_files=hashes,
        snapshot_files={"colab_k1_screen.py":OWNER_SHA}, development_bounds=[0,0],
        checkpoint=dict(path=str(checkpoint),sha256=receipt["sha256"],receipt_ref="experiments/pcqm_k1_execution_efficiency/gpu_attempt_001/results/checkpoint_resume.json"),
        synthetic_telemetry=True, molecular_data_read=False)
    atomic_json(ATTEMPT / "inputs.json", inputs)
    atomic_json(ATTEMPT / "role_plan.json", dict(molecular_data_read=False, protected_roles="untouched", model_execution=False,
        checkpoint_tensor_arithmetic_only=True, synthetic_telemetry_not_training_evidence=True))
    spec = deepcopy(read(HERE / "tf32_attempt_001/plan_input.json"))
    t = spec["trajectory"]
    prior = ["pcqm-k1-kernel-trace-20261010", "pcqm-k1-execution-efficiency-gpu-20261010"]
    t.update(trajectory_id=TID, family_id="k1-checkpoint-io", comparison_readiness_ref=f"{REL}/README.md",
        question="Can immutable scalar dispatch reduce exact checkpoint snapshot cost as full metadata history grows?")
    t["hypothesis"].update(hypothesis_id="H-k1-checkpoint-io-20261010", supporting_evidence_ids=prior,
        observed_deficiency="Sender recursively copies full step history every128 updates; native growing-history cost unknown.",
        changed_mechanism="Immutable scalar fast path only; same tensor clone/container/atomic publication behavior",
        alternative_explanations=["Filesystem/JSON writing dominates", "Copied history has little native impact", "Container/alias behavior differs"],
        cheapest_falsifier="CPU exact snapshot/roundtrip controls, then synthetic metadata-size timing grid",
        decision_changed_if_positive="Nominate additive CPU helper; native sender qualification still separate",
        decision_changed_if_negative="Keep existing copier; no log deletion/frequency change",
        expected_native_cost_ref="cost-k1-checkpoint-io-estimate")
    t["state_at_start"].update(source_commit=head, source_config_identity=sha256_file(ATTEMPT / "inputs.json"),
        contract_refs=[f"{REL}/README.md", f"{REL}/inputs.json"], parent_trajectory_ids=["TB-k1-kernel-trace-20261010"], prior_evidence_ids=prior,
        role_snapshot_refs=[f"{REL}/role_plan.json"], budget_snapshot_ref=f"{REL}/README.md")
    t["actions"] = [dict(action_id="A001", type="execution_profile", run_ids=[RUN], attempt_ids=["checkpoint-io-attempt-001"], source_commit=head,
        evidence_refs=[f"{REL}/inputs.json"],cost_event_ids=["cost-k1-checkpoint-io-estimate"])]
    t["decision"].update(decision_ref=f"{REL}/decision.md",next_allowed_actions=["Bounded CPU-only snapshot diagnostic"])
    spec["decision_state"].update(source_commit=head, state_timestamp=datetime.now(timezone.utc).isoformat(),
        budget_snapshot_ref=f"{REL}/README.md",role_snapshot_refs=[f"{REL}/role_plan.json"])
    cost = spec["costs"][0]
    cost.update(cost_event_id="cost-k1-checkpoint-io-estimate",trajectory_id=TID,run_id=RUN,attempt_id="checkpoint-io-attempt-001",
        hardware="CPU four intra-op threads; no accelerator",evidence_ref=f"{REL}/README.md")
    cost["measurement"] = dict(wall_hours=dict(value=180/3600,status="estimated"),cpu_hours=dict(value=None,status="measurement_missing"),
        device_hours=dict(value=None,status="not_applicable"),queue_hours=dict(value=None,status="not_applicable"))
    atomic_json(ATTEMPT / "plan_input.json", spec)
    atomic_json(ATTEMPT / "plan_receipt.json", plan(ROOT,spec,f"{REL}/rml"))
    print(TID+" prospective frozen")


def execute():
    started, cpu_started = time.perf_counter(), time.process_time()
    import copy
    import torch
    from molgap.checkpoint_snapshot import checkpoint_to_cpu
    from molgap.k1_execution_efficiency import compare_states
    inputs = read(ATTEMPT / "inputs.json")
    trajectory = read(ATTEMPT / "rml/trajectory.json")
    if (ATTEMPT / "results").exists():
        raise FileExistsError("Attempt exists; reconcile, do not repeat")
    if trajectory["record_mode"] != "prospective" or trajectory["decision"]["outcome"] != "ACTIVE":
        raise ValueError("Active prospective required")
    for name,digest in inputs["executed_source_files"].items():
        if sha256_file(ROOT/name) != digest:
            raise ValueError("Source changed")
    source = ATTEMPT / "source_snapshot/colab_k1_screen.py"
    if sha256_file(source) != OWNER_SHA or sha256_file(Path(inputs["checkpoint"]["path"])) != inputs["checkpoint"]["sha256"]:
        raise ValueError("Snapshot/checkpoint changed")
    # Reuse only the inspected copier, without importing a remote runner or its models.
    nodes = [n for n in ast.parse(source.read_text(encoding="utf-8")).body if isinstance(n,ast.FunctionDef) and n.name == "_cpu"]
    if len(nodes) != 1:
        raise ValueError("Frozen copier not uniquely resolved")
    namespace = {"copy":copy}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),"exec"),namespace)
    baseline = namespace["_cpu"]
    torch.set_num_threads(4)
    if torch.cuda.is_initialized():
        raise ValueError("CPU-only diagnostic requires no CUDA initialization")
    checkpoint = torch.load(inputs["checkpoint"]["path"],map_location="cpu",weights_only=False)
    if set(checkpoint) != {"model","optimizer","rng"}:
        raise ValueError("Unexpected accepted scratch checkpoint fields")
    rng_before = torch.get_rng_state().clone()
    output = ATTEMPT / "results"
    output.mkdir(parents=True)
    scratch = ROOT / ".tmp/k1-checkpoint-io-probe"
    if scratch.exists():
        raise FileExistsError("Scratch publication exists; do not overwrite unknown work")
    scratch.mkdir(parents=True)
    def guard():
        if time.perf_counter()-started > inputs["ceiling_seconds"]:
            raise TimeoutError("CPU snapshot worker budget exhausted")
    checks = dict(exact_snapshots=True, exact_roundtrip=True, input_unchanged=True,
                  cpu_step_states_preserved=True, rng_unchanged=True, cuda_uninitialized=True)
    timings = []
    for size in inputs["sizes"]:
        guard()
        # Fixture values are invented for IO only, never a canonical learning trace.
        telemetry = [dict(step=i+1,epoch=i//3906,next_batch=i%3906+1,samples=(i+1)*128,
            loss=0.5,seconds=0.125,allocation_seconds=(i+1)*0.125) for i in range(size)]
        payload = dict(retained_scratch_checkpoint=checkpoint,synthetic_telemetry=dict(arm_a=telemetry,arm_b=telemetry))
        expected = baseline(payload)
        candidate = checkpoint_to_cpu(payload)
        checks["exact_snapshots"] &= compare_states(expected,candidate)["exact"]
        steps = checkpoint["optimizer"]["state"]
        checks["cpu_step_states_preserved"] &= (len(steps) == 313 and all(
            state["step"].device.type == "cpu" and torch.equal(state["step"],candidate["retained_scratch_checkpoint"]["optimizer"]["state"][key]["step"])
            for key,state in steps.items()))
        del expected,candidate
        samples = []
        for repeat in range(inputs["repeats"]):
            pair = {}
            order = ("baseline","candidate") if repeat%2 == 0 else ("candidate","baseline")
            for name in order:
                guard()
                tick = time.perf_counter()
                value = (baseline if name == "baseline" else checkpoint_to_cpu)(payload)
                copied = time.perf_counter()-tick
                tick = time.perf_counter()
                atomic_torch_save(scratch / "checkpoint.pt",value)
                atomic_json(scratch / "telemetry.json",value["synthetic_telemetry"])
                hashes = {p.name:sha256_file(p) for p in scratch.iterdir()}
                published = time.perf_counter()-tick
                loaded = torch.load(scratch / "checkpoint.pt",map_location="cpu",weights_only=False)
                checks["exact_roundtrip"] &= compare_states(payload,loaded)["exact"]
                checks["exact_roundtrip"] &= read(scratch / "telemetry.json") == payload["synthetic_telemetry"]
                pair[name] = dict(copy_seconds=copied,publication_hash_seconds=published,
                    copy_publication_seconds=copied+published,pt_bytes=(scratch/"checkpoint.pt").stat().st_size,
                    json_bytes=(scratch/"telemetry.json").stat().st_size,hashes=hashes)
                del value,loaded
            samples.append(pair)
        checks["input_unchanged"] &= compare_states(checkpoint,torch.load(inputs["checkpoint"]["path"],map_location="cpu",weights_only=False))["exact"]
        timings.append(dict(synthetic_rows_per_arm=size,repeat_samples=samples,
            median_copy_seconds={n:statistics.median(r[n]["copy_seconds"] for r in samples) for n in ("baseline","candidate")},
            median_copy_publication_seconds={n:statistics.median(r[n]["copy_publication_seconds"] for r in samples) for n in ("baseline","candidate")}))
        atomic_json(output/"progress.json", dict(status="partial",checks=checks,timings=timings,
            synthetic_fixture_not_training_evidence=True,inputs_sha256=sha256_file(ATTEMPT/"inputs.json")))
        print(json.dumps(timings[-1]["median_copy_seconds"] | {"synthetic_rows_per_arm":size}),flush=True)
    checks["rng_unchanged"] = torch.equal(rng_before,torch.get_rng_state())
    checks["cuda_uninitialized"] = not torch.cuda.is_initialized()
    wall = time.perf_counter()-started
    if not all(checks.values()) or wall > inputs["ceiling_seconds"]:
        raise ValueError("Exactness/CPU scope/budget check failed; retain partial scratch")
    result = dict(format="molgap-k1-saved-analysis-v1",kind=inputs["kind"],status="complete",checks=checks,
        inputs_sha256=sha256_file(ATTEMPT/"inputs.json"),prospective_sha256=sha256_file(ATTEMPT/"rml/trajectory.json"),
        wall_seconds=wall,process_cpu_seconds=time.process_time()-cpu_started,runtime=dict(device="cpu",cpu_threads=4,torch=torch.__version__),
        observed_source_idx=[],training_executed=False,inference_executed=False,optimizer_created=False,gradients_computed=False,
        synthetic_fixture_not_training_evidence=True,timings=timings,
        limits=["CPU checkpoint plus synthetic telemetry; not complete two-arm native worker publication",
                "Local file writes/page cache; no fsync power-loss or remote storage durability/throughput qualification",
                "No training end-to-end saving, role access, checkpoint-frequency change or log deletion"])
    atomic_json(output/"result.json",result)
    atomic_json(output/"completion.json",dict(complete=True,inputs_sha256=result["inputs_sha256"],
        files={p.name:sha256_file(p) for p in output.iterdir() if p.is_file()}))
    # Only these two owned temporary publication files are removed; evidence remains.
    for name in ("checkpoint.pt","telemetry.json"):
        (scratch/name).unlink()
    scratch.rmdir()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("operation",choices=("prepare","execute","close"))
    {"prepare":prepare,"execute":execute,"close":lambda: print(close_saved_diagnostic(ROOT,ATTEMPT))}[parser.parse_args().operation]()
