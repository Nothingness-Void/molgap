"""Bounded retained-state adapter; preparation does not import a model factory."""
from __future__ import annotations

import argparse
import ast
import base64
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
import tarfile
import zipfile

from molgap.constants import REPO_ROOT as ROOT
from molgap.research_memory.plan import plan as publish_plan
from molgap.research_memory.schemas import validate_cost_event, validate_trajectory
from molgap.research_memory.trace import atomic_write, file_digest, json_bytes

DIRECTORY = ROOT / "experiments/pcqm_scale_fit_retained"
COHORTS = {"train": (0, 5000), "development": (500000, 505000)}
STATES = {"reference_100k": {"live": "model", "ema": "ema"},
          "joint_100k": {"live": "model_state", "ema": "ema_state"},
          "reference_500k": {"live": "model"}, "joint_500k": {"live": "model"}}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_hash(path, expected):
    actual = file_digest(path)
    if actual != expected:
        raise ValueError(f"Hash mismatch: {path}: {actual} != {expected}")
    return actual


def bindings(value):
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            yield value["path"], value["sha256"]
        for child in value.values():
            yield from bindings(child)
    elif isinstance(value, list):
        for child in value:
            yield from bindings(child)


def validate_inputs(data):
    if data["format"] != "molgap-scale-fit-retained-inputs-v1":
        raise ValueError("Unsupported inputs")
    if set(data["cohorts"]) != set(COHORTS):
        raise ValueError("Unknown cohort")
    for role, pair in COHORTS.items():
        if (data["cohorts"][role]["start"], data["cohorts"][role]["stop"]) != pair:
            raise ValueError("Unauthorized cohort")
    if set(data["checkpoints"]) != set(STATES):
        raise ValueError("Checkpoint arms changed")
    for arm, states in STATES.items():
        entry = data["checkpoints"][arm]
        if entry["states"] != states:
            raise ValueError("State inventory changed; 500K EMA is absent")
        joint = arm.startswith("joint")
        if entry["factory"] != ("gptrans_noisy_pair_norm" if joint else "gptrans"):
            raise ValueError("Wrong factory")
        if entry["parameters"] != (5277400 if joint else 5246817):
            raise ValueError("Wrong parameter count")
    if data["missing_states"] != ["reference_500k.ema", "joint_500k.ema"]:
        raise ValueError("Missing-state truth changed")
    auth = data["authorization"]
    if (auth["official_roles_allowed"] or auth["sealed_roles_allowed"]
            or auth["optimizer_updates_allowed"] or not auth["run_review_required"]
            or auth["allowed_roles"] != ["train_prefix", "reused_internal_development"]):
        raise ValueError("Role/operation authority changed")
    if data["execution"] != {"precision": "fp32", "tf32_enabled": False,
            "device_count": 1, "batch_size": 128, "max_inference_seconds": 900,
            "max_molecule_forwards": 60000}:
        raise ValueError("Execution contract changed")


def verify_inputs(data):
    validate_inputs(data)
    for path, digest in bindings(data):
        require_hash(path, digest)
    manifest = read(data["cache_manifest"]["path"])
    for role, cohort in data["cohorts"].items():
        entries = [e for e in manifest["geometry_shards"]
                   if e["role"] == role and e["source_idx_min"] == cohort["start"]]
        if len(entries) != 1 or entries[0]["sha256"] != cohort["shard"]["sha256"]:
            raise ValueError("Cohort shard not bound by cache manifest")
        entry = entries[0]
        path = Path(data["cache_manifest"]["path"]).parent / entry["file"]
        if path.resolve() != Path(cohort["shard"]["path"]).resolve():
            raise ValueError("Manifest shard path changed")
        if entry["rows"] != 50000 or entry["source_idx_max"] != cohort["start"] + 49999:
            raise ValueError("Containing shard membership changed")


def source_equivalence(data):
    """Compare immutable producer definitions without importing their code."""
    with tarfile.open(data["source_100k"]["path"]) as archive:
        old100 = {name: archive.extractfile(name).read() for name in (
            "src/molgap/gptrans.py", "src/molgap/gptrans_variants.py", "src/molgap/noisy_nodes.py")}
    for name, old in old100.items():
        if old.replace(b"\r\n", b"\n") != (ROOT / name).read_bytes().replace(b"\r\n", b"\n"):
            raise ValueError(f"100K frozen model source differs: {name}")
    source = data["sources_500k"]
    with zipfile.ZipFile(source["reference"]["path"]) as archive:
        baseline = {n: archive.read(n) for n in ("src/molgap/gptrans.py",
                    "src/molgap/pcqm_gptrans_v4.py", "src/molgap/gptrans_variants.py")}
    tree = ast.parse(Path(source["joint_runner"]["path"]).read_bytes())
    encoded = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
        and getattr(n.targets[0], "id", None) == "SOURCE_PAYLOAD_B64")
    payload = base64.b64decode(encoded, validate=True)
    import hashlib
    if hashlib.sha256(payload).hexdigest() != source["joint_payload_sha256"]:
        raise ValueError("500K joint immutable payload differs")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name, old in old100.items():
            if archive.read(name).replace(b"\r\n", b"\n") != old.replace(b"\r\n", b"\n"):
                # Training-only changes in noisy_nodes do not change the frozen factory/classes.
                names = ("NoisyNodesConfig", "GPTransNoisyNodes", "make_noisy_nodes_pair_norm_model")
                def definitions(value):
                    return ast.dump(ast.Module(body=[n for n in ast.parse(value).body
                        if getattr(n, "name", None) in names], type_ignores=[]))
                if name != "src/molgap/noisy_nodes.py" or definitions(archive.read(name)) != definitions(old):
                    raise ValueError(f"500K joint model definitions differ: {name}")
    if baseline["src/molgap/gptrans.py"].replace(b"\r\n", b"\n") != old100["src/molgap/gptrans.py"].replace(b"\r\n", b"\n"):
        raise ValueError("500K reference model code differs")
    # Initialization loading changed; inference calls the factory with no initial state.
    def prefix(value, function, condition):
        node = next(n for n in ast.parse(value).body if getattr(n, "name", None) == function)
        body = []
        for n in node.body:
            body.append(n)
            if isinstance(n, ast.If) and ast.unparse(n.test) == condition:
                return ast.dump(ast.Module(body=body, type_ignores=[]))
        raise ValueError("Factory branch absent")
    for name, function, condition in (("src/molgap/pcqm_gptrans_v4.py", "_make_model", "initial_state_path is None"),
                                      ("src/molgap/gptrans_variants.py", "apply_variant", "variant == 'reference'")):
        if prefix(baseline[name], function, condition) != prefix((ROOT / name).read_bytes(), function, condition):
            raise ValueError(f"500K reference factory path differs: {function}")
    return {"100k_model_modules": "exact-LF-bytes", "500k_reference": "same-core-and-no-init-reference-factory-AST",
            "500k_joint": "same-core-variants-and-noisy-model-definitions", "parameter_counts_alone": False}


def freeze(directory=DIRECTORY):
    directory = Path(directory)
    if (directory / "frozen_plan.json").exists() or (directory / "rml").exists():
        raise FileExistsError("Immutable preparation already exists")
    data = read(directory / "inputs.json")
    verify_inputs(data)
    equivalence = source_equivalence(data)
    trajectory = read(directory / "trajectory_template.json")
    validate_trajectory(trajectory)
    cost = read(directory / "expected_cost.json")
    validate_cost_event(cost)
    # Pin the entire local Python source tree, including transitive factory imports.
    paths = sorted((ROOT / "src/molgap").rglob("*.py"))
    paths += sorted(p for p in directory.rglob("*") if p.is_file()
                    and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"}
                    and not {"results", "rml"}.intersection(p.relative_to(directory).parts)
                    and p.name not in {"frozen_plan.json", "run_review.json"})
    for pointer in (*trajectory["state_at_start"]["role_snapshot_refs"],
                    "experiments/pcqm_gptrans_100k_transfer_control/paired_release_v2.json",
                    "experiments/pcqm_500k_v4_evidence/stage5_gptrans_acceptance.md",
                    "experiments/pcqm_gptrans_noisy_pair_norm_500k/results/stage_manifest.json"):
        paths.append(ROOT / pointer)
    paths += [ROOT / "tests/test_scale_fit_retained.py",
              ROOT / "experiments/pcqm_gptrans_100k_transfer_control/analyze_pair.py"]
    paths += list((ROOT / "experiments/pcqm_gptrans_100k_transfer_control/selection_probe").glob("*.json"))
    paths += list((ROOT / "experiments/pcqm_gptrans_100k_transfer_control/selection_probe").glob("*.md"))
    source_hashes = {p.relative_to(ROOT).as_posix(): file_digest(p) for p in paths}
    plan = {"format": "molgap-scale-fit-retained-frozen-plan-v1", "inputs": data,
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                cwd=ROOT, text=True).strip(), "source_hashes": source_hashes,
            "planning_mode": "public-RML-plan-diagnostic-only-not-promotion-or-replay-ready",
            "source_equivalence": equivalence,
            "inference_executed": False, "frozen_at_unix": time.time()}
    if plan["source_commit"] != trajectory["state_at_start"]["source_commit"]:
        raise ValueError("Owning branch source tip changed")
    action_inputs = {"trajectory_id": trajectory["trajectory_id"], "state_timestamp": "2026-10-01",
        "evidence_ids": trajectory["state_at_start"]["prior_evidence_ids"],
        "user_requested_scope_reviewed": 1,
        "authority": data["authorization"]["authority"], "execution_review_pending": True}
    atomic_write(directory / "action_inputs.json", json_bytes(action_inputs))
    spec = {"trajectory": trajectory, "costs": [cost],
        "action_inputs_ref": "experiments/pcqm_scale_fit_retained/action_inputs.json", "decision_state": {
        "known_trajectory_ids": [], "known_evidence_ids": [], "active_reference_ids": [],
        "available_actions": ["RUN_DIAGNOSTIC", "NO_TRAIN"], "chosen_action": "RUN_DIAGNOSTIC",
        "policy_id": "pcqm-scale-fit-retained-diagnostic", "policy_version": "1",
        "role_snapshot_refs": trajectory["state_at_start"]["role_snapshot_refs"],
        "budget_snapshot_ref": trajectory["state_at_start"]["budget_snapshot_ref"],
        "state_timestamp": "2026-10-01", "source_commit": plan["source_commit"]}}
    atomic_write(directory / "plan_input.json", json_bytes(spec))
    plan["rml_plan_receipt"] = publish_plan(ROOT, spec, directory / "rml")
    for name in ("action_inputs.json", "plan_input.json", "rml/decision_state.json", "rml/policy_snapshot.json"):
        path = directory / name
        plan["source_hashes"][path.relative_to(ROOT).as_posix()] = file_digest(path)
    plan["trajectory_sha256"] = file_digest(directory / "rml/trajectory.json")
    plan["source_hashes"]["research_memory/policies/pcqm-scale-fit-retained-diagnostic.1.json"] = file_digest(
        ROOT / "research_memory/policies/pcqm-scale-fit-retained-diagnostic.1.json")
    atomic_write(directory / "frozen_plan.json", json_bytes(plan))
    return {"plan_sha256": file_digest(directory / "frozen_plan.json"),
            "trajectory_sha256": plan["trajectory_sha256"], "status": "PREPARED_NOT_EXECUTED"}


def check(directory=DIRECTORY):
    directory = Path(directory)
    plan = read(directory / "frozen_plan.json")
    if plan["format"] != "molgap-scale-fit-retained-frozen-plan-v1":
        raise ValueError("Unsupported frozen plan")
    require_hash(directory / "rml/trajectory.json", plan["trajectory_sha256"])
    validate_trajectory(read(directory / "rml/trajectory.json"))
    for pointer, digest in plan["source_hashes"].items():
        path = (ROOT / pointer).resolve()
        path.relative_to(ROOT)
        require_hash(path, digest)
    verify_inputs(plan["inputs"])
    source_equivalence(plan["inputs"])
    return plan


def check_review(review, plan_sha, trajectory_sha):
    if (review.get("approved") is not True or not review.get("authority")
            or not review.get("approved_at") or review.get("plan_sha256") != plan_sha
            or review.get("trajectory_sha256") != trajectory_sha
            or review.get("prospective_freeze_scope_approved") is not True):
        raise ValueError("Explicit hash-bound parent run review required")


def checkpoint_states(data):
    import torch
    from molgap.v4_runtime import torch_load_compat

    payloads = {}
    for arm, entry in data["checkpoints"].items():
        require_hash(entry["path"], entry["sha256"])
        payloads[arm] = torch_load_compat(Path(entry["path"]), map_location="cpu", weights_only=False)
    reference = payloads["reference_100k"]
    mean = float(reference["target_stats"]["mean_eV"])
    std = float(reference["target_stats"]["sample_std_eV"])
    ref_authority = read(data["checkpoints"]["reference_100k"]["authority"]["path"])
    joint_authority = read(data["checkpoints"]["joint_100k"]["authority"]["path"])
    joint_identity = payloads["joint_100k"]["run_identity"]
    manifest_pin = "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d"
    if ref_authority["manifest_sha256"] != manifest_pin or joint_identity["manifest_sha256"] != manifest_pin:
        raise ValueError("100K training-statistics input identity differs")
    if joint_authority["source_archive_sha256"] != ref_authority["source_archive_sha256"]:
        raise ValueError("100K transform producer source differs")
    with tarfile.open(data["source_100k"]["path"]) as archive:
        for file, function in (("src/molgap/pcqm_gptrans_v4.py", "run_training"),
                               ("src/molgap/noisy_nodes.py", "run_training_noisy_nodes")):
            node = next(n for n in ast.parse(archive.extractfile(file).read()).body
                        if getattr(n, "name", None) == function)
            calls = {ast.unparse(n) for n in ast.walk(node) if isinstance(n, ast.Call)}
            if "_target_stats(train_shards)" not in calls:
                raise ValueError("100K shared target-transform derivation changed")
    for arm, payload in payloads.items():
        entry = data["checkpoints"][arm]
        if arm.endswith("100k"):
            authority = read(entry["authority"]["path"])
            if (payload["epoch"] != 59 or authority["best_epoch"] != 59
                    or authority["checkpoint_sha256"] != entry["sha256"]
                    or authority["source_archive_sha256"] != data["source_100k"]["sha256"]
                    or authority["source_commit"] != "12f930f42adc10d3a8ae6dbb53ac874f96cdb052"):
                raise ValueError("100K endpoint/producer identity mismatch")
            producer = payload if arm.startswith("reference") else payload["run_identity"]
            if producer["source_archive_sha256"] != data["source_100k"]["sha256"]:
                raise ValueError("100K checkpoint source mismatch")
            if arm.startswith("joint") and (producer["noise_std"], producer["loss_weight"],
                    producer["pair_update_norm"]) != (0.15, 0.1, True):
                raise ValueError("Joint mechanism changed")
            transform = {"mean": mean, "std": std}
        else:
            if payload["next_epoch"] != 60 or payload["arm"] != entry["factory"]:
                raise ValueError("500K terminal arm mismatch")
            if payload["source_sha256"] != entry["source_sha256"]:
                raise ValueError("500K producer source mismatch")
            if any("ema" in k.lower() for k in payload):
                raise ValueError("500K state inventory changed")
            pointer = entry["transform_artifact"]
            require_hash(pointer["path"], pointer["sha256"])
            transform = torch_load_compat(Path(pointer["path"]), map_location="cpu", weights_only=False)
            if (transform["arm"] != payload["arm"] or transform["contract"] != payload["contract"]
                    or transform["source_sha256"] != payload["source_sha256"]):
                raise ValueError("Transform provenance mismatch")
            if (payload["contract"]["data_role_fingerprint"] != data["cache_manifest"]["sha256"]
                    or payload["contract"]["precision"] != "fp32"
                    or payload["contract"]["sample_exposure"] != 29998080):
                raise ValueError("500K contract mismatch")
        if not math.isfinite(transform["mean"]) or not math.isfinite(transform["std"]) or transform["std"] <= 0:
            raise ValueError("Invalid retained target transform")
        for semantics, key in entry["states"].items():
            state = payload[key]
            if not state or not all(isinstance(t, torch.Tensor) and bool(torch.isfinite(t).all())
                                    for t in state.values()):
                raise ValueError("Non-finite checkpoint state")
            yield arm, semantics, state, float(transform["mean"]), float(transform["std"])


def validate_prediction(record, role):
    import torch
    start, stop = COHORTS[role]
    if not torch.equal(record["source_idx"], torch.arange(start, stop)):
        raise ValueError("Prediction row alignment changed")
    for key in ("prediction_eV", "target_eV"):
        if record[key].shape != (stop - start,) or not bool(torch.isfinite(record[key]).all()):
            raise ValueError("Prediction shape/finite check failed")


def validate_graph(graph, expected_index):
    import torch
    if (graph.source_idx.numel() != 1 or int(graph.source_idx.item()) != expected_index
            or graph.x.ndim != 2 or graph.x.shape[1] != 9
            or graph.edge_attr.ndim != 2 or graph.edge_attr.shape[1] != 3
            or graph.edge_index.shape != (2, graph.edge_attr.shape[0])
            or graph.y.numel() != 1 or not bool(torch.isfinite(graph.y).all())):
        raise ValueError("Graph row/feature/target identity invalid")
    if graph.edge_index.numel() and (int(graph.edge_index.min()) < 0 or int(graph.edge_index.max()) >= len(graph.x)):
        raise ValueError("Graph edge endpoint invalid")


def paired_analysis(records):
    spec = importlib.util.spec_from_file_location("scale_fit_pair_analysis",
        ROOT / "experiments/pcqm_gptrans_100k_transfer_control/analyze_pair.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = {}
    for scale, semantics in (("100k", "live"), ("100k", "ema"), ("500k", "live")):
        for role, (start, stop) in COHORTS.items():
            reference = records[f"reference_{scale}.{semantics}.{role}"]
            candidate = records[f"joint_{scale}.{semantics}.{role}"]
            metrics = module.paired_metrics(reference, candidate, start, stop)
            metrics.pop("material_3mev_point_gain")
            result[f"{scale}.{semantics}.{role}"] = metrics
    return result


def infer(review_path, directory=DIRECTORY):
    directory = Path(directory)
    check_review(read(review_path), file_digest(directory / "frozen_plan.json"),
                 file_digest(directory / "rml/trajectory.json"))
    started = time.perf_counter()
    plan = check(directory)
    data = plan["inputs"]
    output = directory / "results"
    output.mkdir(exist_ok=False)
    atomic_write(output / "run_review.json", json_bytes(read(review_path)))
    # The watchdog is only created inside the separately authorized inference path.
    import torch
    from torch.utils.data import Subset
    from molgap import pcqm_gptrans_v4 as v4
    from molgap.pcqm_500k_v4_evidence import make_model
    from molgap.training_reproducibility import atomic_torch_save, configure_fp32_determinism
    from molgap.v4_runtime import state_dict_sha256

    settings = configure_fp32_determinism(42)
    v4.LOADER_WORKERS = 0
    if not torch.cuda.is_available():
        raise RuntimeError("Authorized local CUDA device unavailable")
    cohorts = {}
    for role, cohort in data["cohorts"].items():
        graphs, _ = v4._load_datasets((Path(cohort["shard"]["path"]),))
        if len(graphs) != 50000:
            raise ValueError("Containing shard row count changed")
        cohorts[role] = Subset(graphs, range(5000))
        for offset in range(5000):
            validate_graph(cohorts[role][offset], COHORTS[role][0] + offset)
    states = list(checkpoint_states(data))
    report = {"status": "RUNNING", "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "settings": settings, "plan_sha256": file_digest(directory / "frozen_plan.json"),
        "loader_workers": 0, "loader_choice": "execution-only; no batch/precision change",
        "joint_100k_transform_authority": "same SHA-bound producer, training manifest and _target_stats(train_shards); joint numerical stats not independently serialized",
        "missing_states": data["missing_states"], "state_reports": {},
        "limitations": data["limitations"], "cpu_hours": None, "queue_hours": None}
    records = {}
    torch.cuda.synchronize()
    gpu_start = time.perf_counter()
    forward_seconds = 0.0
    forwards = 0
    deadline = gpu_start + 900

    def watchdog():
        atomic_write(output / "watchdog_timeout.json", json_bytes({"status": "STOP_FOR_COST",
            "ceiling_seconds": 900, "elapsed_script_seconds": time.perf_counter() - started,
            "note": "Process terminated; partial outputs unaccepted; no terminal closure."}))
        os._exit(124)

    timer = threading.Timer(900, watchdog)
    timer.daemon = True
    timer.start()
    try:
        for arm, semantics, state, mean, std in states:
            if time.perf_counter() >= deadline:
                raise TimeoutError("Inference ceiling")
            entry = data["checkpoints"][arm]
            model = make_model(entry["factory"]).float().to("cuda")
            if sum(p.numel() for p in model.parameters()) != entry["parameters"]:
                raise ValueError("Factory parameter identity changed")
            model.load_state_dict(state, strict=True)
            model.eval()
            state_hash = state_dict_sha256(model.state_dict())
            for role, graphs in cohorts.items():
                rows = {k: [] for k in ("source_idx", "target_eV", "prediction_eV")}
                torch.cuda.synchronize()
                cohort_start = time.perf_counter()
                with torch.inference_mode():
                    for batch in v4._development_loader(graphs):
                        if time.perf_counter() >= deadline or forwards + batch.num_graphs > 60000:
                            raise TimeoutError("Inference time/row ceiling")
                        batch = batch.to("cuda")
                        before = torch.cuda.Event(enable_timing=True)
                        after = torch.cuda.Event(enable_timing=True)
                        before.record()
                        prediction = v4._forward(model, batch) * std + mean
                        after.record()
                        after.synchronize()
                        forward_seconds += before.elapsed_time(after) / 1000
                        forwards += batch.num_graphs
                        rows["prediction_eV"].append(prediction.cpu().float().view(-1))
                        rows["target_eV"].append(batch.y.cpu().float().view(-1))
                        rows["source_idx"].append(batch.source_idx.cpu().long().view(-1))
                record = {key: torch.cat(values) for key, values in rows.items()}
                validate_prediction(record, role)
                if records:
                    prior = next((v for k, v in records.items() if k.endswith("." + role)), None)
                    if prior is not None and not torch.equal(prior["target_eV"], record["target_eV"]):
                        raise ValueError("Cross-state target alignment changed")
                key = f"{arm}.{semantics}.{role}"
                records[key] = record
                path = output / f"{key}.pt"
                atomic_torch_save(path, record)
                report["state_reports"][key] = {"artifact_sha256": file_digest(path),
                    "state_sha256": state_hash, "checkpoint_sha256": entry["sha256"],
                    "mean_eV": mean, "std_eV": std,
                    "source_idx_sha256": state_dict_sha256({"source_idx": record["source_idx"]}),
                    "target_sha256": state_dict_sha256({"target_eV": record["target_eV"]}),
                    "mae_eV": float((record["prediction_eV"] - record["target_eV"]).abs().double().mean()),
                    "synchronized_wall_seconds": time.perf_counter() - cohort_start}
                atomic_write(output / "progress.json", json_bytes(report))
            if state_dict_sha256(model.state_dict()) != state_hash:
                raise ValueError("Inference mutated state")
            del model
            torch.cuda.empty_cache()
        if forwards != 60000 or len(records) != 12:
            raise ValueError("Incomplete state/cohort coverage")
        report["status"] = "EXECUTED_PENDING_ACCEPTANCE"
    except Exception as exc:
        report["status"] = "INCOMPLETE_UNACCEPTED"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        torch.cuda.synchronize()
        report.update(synchronized_gpu_resident_seconds=time.perf_counter() - gpu_start,
                      cuda_event_forward_seconds=forward_seconds, molecule_forwards=forwards,
                      total_script_wall_seconds=time.perf_counter() - started)
        atomic_write(output / "execution.json", json_bytes(report))
        timer.cancel()
    report["paired_metrics"] = paired_analysis(records)
    atomic_write(output / "analysis.json", json_bytes(report))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("freeze", "check", "infer"))
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    if args.operation == "infer" and args.review is None:
        parser.error("infer requires --review with separate parent authorization")
    result = freeze() if args.operation == "freeze" else check() if args.operation == "check" else infer(args.review)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
