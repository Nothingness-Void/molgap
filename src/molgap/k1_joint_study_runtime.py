"""Single authorized Kaggle objective study, using existing training/audit wheels."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

RECIPES = ("k1_corrupt_gap", "k1_corrupt_gap_atom_aux")
RUN_ID = "kaseichou/molgap-k1-joint-atom-s42:v1"
TRAJECTORIES = {recipe: "TC-" + recipe.replace("_", "-") + "-100k-s42" for recipe in RECIPES}
AUDIT_TRAJECTORY = "TC-k1-joint-atom-post100k-audit-s42"
ROOT = Path("/kaggle/working/pcqm_k1_joint_atom_reconstruction")
MAX_SECONDS = 6 * 3600


def verify_source(source, archive):
    """Bind all executable bytes and both prelaunch identities before imports."""
    from .k1_joint_objective import objective_fingerprint
    source, archive = Path(source), Path(archive)
    payload = archive.parent
    commit = (payload / "SOURCE_COMMIT.txt").read_text().strip()
    digest = (payload / "SOURCE_ARCHIVE_SHA256.txt").read_text().strip()
    release = json.loads((payload / "JOINT_RELEASE.json").read_text())
    if len(commit) != 40 or hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Immutable source identity failed")
    if release["source_commit"] != commit or release["run_id"] != RUN_ID:
        raise RuntimeError("Prospective release source/run mismatch")
    for row in json.loads((payload / "SOURCE_FILES.json").read_text())["files"]:
        path = (source / row["path"]).resolve()
        if not path.is_relative_to(source.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError("Source inventory hash/path mismatch")
    if set(release["arms"]) != set(RECIPES):
        raise RuntimeError("Incorrect dual-arm release")
    for recipe in RECIPES:
        arm = release["arms"][recipe]
        if (arm["prelaunch_ready"] is not True or arm["trajectory_id"] != TRAJECTORIES[recipe]
                or arm["loss_identity"] != objective_fingerprint(recipe)):
            raise RuntimeError("Objective/trajectory not prospectively released")
    transform = payload / "target_transform.json"
    if hashlib.sha256(transform.read_bytes()).hexdigest() != release["target_transform_file_sha256"]:
        raise RuntimeError("Released target transform asset changed")
    return {"python_root": str(source / "src"), "source_commit": commit,
            "source_archive_sha256": digest, "transform_asset": str(transform)}


def cache_by_hash(digest):
    matches = [p.parent for p in Path("/kaggle/input").rglob("manifest.json")
               if hashlib.sha256(p.read_bytes()).hexdigest() == digest]
    if len(matches) != 1:
        raise RuntimeError("Missing or ambiguous fixed cache manifest")
    return matches[0]


def audit(recipe, output, record, context):
    """Reproduce a frozen checkpoint, then read ONLY fixed500K development."""
    import torch
    from .k1_portability_audit import _graphs, _model, _payload, _infer, _joined
    from .pcqm_k1_cross_scale_diagnostic import MANIFESTS, TRANSFORM_SHA256
    from .training_reproducibility import atomic_json, sha256_file

    started = time.monotonic()
    checkpoint_sha = record["training"]["best_model_sha256"]
    completion = json.loads((output / "completion_manifest.json").read_text())
    for name, digest in completion["artifact_sha256"].items():
        if sha256_file(output / name) != digest:
            raise RuntimeError("Training artifact mutated before terminal inference")
    if (record.get("complete") is not True or record["source_commit"] != context["source_commit"]
            or record["training"]["optimizer_steps"] != 31240
            or record["training"]["sample_presentations"] != 3998720):
        raise RuntimeError("Incomplete training cannot open portability role")
    asset = Path(context["transform_asset"])
    if sha256_file(asset) != TRANSFORM_SHA256:
        raise RuntimeError("Portable transform bytes changed")
    transform = json.loads(asset.read_text())
    model = _model("neural_atom_k1_v4", output / "best_model.pt", checkpoint_sha)
    saved = _payload(output / "best_development_payload.pt", record["training"]["payload_sha256"], recipe)
    audit_root = output / "post100k_audit"
    original_root = audit_root / "original_100k"
    graphs = _graphs(cache_by_hash(MANIFESTS["original_100k"]), "original_100k")
    original = _infer(graphs, model, start=100000, output=original_root,
        role="original_100k", mode=recipe, mean=transform["mean"], std=transform["std"])
    joined = _joined(original_root, original)
    difference = float((joined["prediction_eV"] - saved["prediction_eV"].view(-1).float()).abs().max())
    if not torch.equal(joined["target_eV"], saved["target_eV"].view(-1).float()) or difference > 0.001:
        raise RuntimeError("Frozen clean checkpoint reproduction failed")
    del graphs, joined
    # This line is the only entry to the 500K role, after reproduction and hashes.
    graphs = _graphs(cache_by_hash(MANIFESTS["unseen_500k"]), "unseen_500k")
    target = audit_root / "unseen_500k"
    unseen = _infer(graphs, model, start=500000, output=target,
        role="unseen_500k", mode=recipe, mean=transform["mean"], std=transform["std"])
    joined = _joined(target, unseen)
    if sha256_file(output / "best_model.pt") != checkpoint_sha:
        raise RuntimeError("Frozen checkpoint changed during inference")
    result = {"format": "molgap-k1-joint-portability-audit-v1", "complete": True,
        "trajectory_id": AUDIT_TRAJECTORY, "run_id": RUN_ID, "recipe": recipe,
        "experiment_purpose": "NO_TRAIN", "training_executed_in_audit_stage": False,
        "optimizer_steps_in_audit_stage": 0, "model_inference_executed": True,
        "source_commit": context["source_commit"], "source_archive_sha256": context["source_archive_sha256"],
        "checkpoint_sha256": checkpoint_sha, "manifest_sha256": MANIFESTS,
        "target_transform_sha256": TRANSFORM_SHA256,
        "original_reproduction_max_abs_eV": difference,
        "original_100k": {"chunks": original, "rows": 50000},
        "unseen_500k": {"chunks": unseen, "rows": 50000,
            "mae_eV": float((joined["prediction_eV"].double() - joined["target_eV"].double()).abs().mean())},
        "roles_observed": {"internal_development_100000_150000": ["prediction_input", "labels_read", "metric_computed"],
            "internal_development_500000_550000": ["prediction_input", "labels_read", "metric_computed"]},
        "selection_used_in_audit_stage": False, "training_membership_in_audit_stage": False,
        "official_validation_role_read": False, "test_dev_role_read": False, "test_challenge_role_read": False,
        "wall_seconds": time.monotonic() - started}
    atomic_json(audit_root / "terminal.json", result)
    return result


def child(context, recipe):
    from .training_reproducibility import atomic_json, sha256_file
    if recipe not in RECIPES:
        raise ValueError("Unauthorized recipe")
    output = ROOT / recipe
    output.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.monotonic(), time.process_time()
    stage, training_seconds, audit_seconds = "preflight_training", None, None
    completed, hardware = False, None
    try:
        import torch
        if torch.cuda.device_count() != 1:
            raise RuntimeError("An independent arm must see exactly one GPU")
        hardware = torch.cuda.get_device_name(0)
        from .pcqm_k1_variants_runner import train_arm
        record = train_arm("neural_atom_k1_v4", output, source_commit=context["source_commit"],
            source_archive_sha256=context["source_archive_sha256"], objective_recipe=recipe,
            trajectory_id=TRAJECTORIES[recipe], physical_run_id=RUN_ID,
            target_transform_asset=Path(context["transform_asset"]))
        training_seconds = time.monotonic() - started
        stage = "frozen_portability_inference"
        audit_result = audit(recipe, output, record, context)
        audit_seconds = audit_result["wall_seconds"]
        completed = True
    except BaseException:
        atomic_json(output / "failure.json", {"run_id": RUN_ID, "recipe": recipe,
            "stage": stage, "traceback": traceback.format_exc(), "automatic_retry": False})
        raise
    finally:
        elapsed = time.monotonic() - started
        cost = {"format": "molgap-joint-study-native-cost-v1", "recipe": recipe,
            "trajectory_id": TRAJECTORIES[recipe], "run_id": RUN_ID,
            "hardware": hardware, "device_count": 1, "wall_seconds": elapsed,
            "allocated_device_seconds": elapsed, "cpu_process_seconds": time.process_time() - cpu_started,
            "cpu_scope": "worker process only; dataloader subprocess CPU unmeasured",
            "training_wall_seconds": training_seconds, "audit_wall_seconds": audit_seconds,
            "measurement": "monotonic-wall-times-one-device", "complete": completed,
            "training_completed": training_seconds is not None}
        atomic_json(output / "native_cost.json", cost)
        manifest_path = output / "completion_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            manifest["artifact_sha256"]["native_cost.json"] = sha256_file(output / "native_cost.json")
            atomic_json(manifest_path, manifest)


def run(context):
    from .k1_edge_kaggle_runtime import _pin_runtime
    from .k1_relation_study_runtime import worker
    from .training_reproducibility import atomic_json
    ROOT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    _pin_runtime(required_devices=2, required_name="T4")
    import torch
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    atomic_json(ROOT / "launch_identity.json", {**context, "run_id": RUN_ID,
        "recipes": RECIPES, "allocated_devices": names, "used_device_count": 2})
    print(json.dumps({"run_id": RUN_ID, "gpu_names": names, "recipes": RECIPES}), flush=True)
    deadline = time.monotonic() + MAX_SECONDS
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda pair: worker(context, pair[1], pair[0], deadline,
            output_root=ROOT, child_module="molgap.k1_joint_study_runtime", run_id=RUN_ID), enumerate(RECIPES)))
    elapsed = time.monotonic() - started
    complete = all(row["complete"] for row in results)
    atomic_json(ROOT / "execution_summary.json", {"run_id": RUN_ID, "workers": results,
        "complete": complete, "allocated_device_names": names, "used_device_count": 2,
        "allocated_device_count": len(names), "total_job_wall_seconds": elapsed,
        "total_allocated_device_seconds": elapsed * len(names), "automatic_successor_submitted": False})
    if not complete:
        raise RuntimeError("Preserved worker failure; no automatic restart")


if __name__ == "__main__":
    child(json.loads(os.environ["MOLGAP_RELATION_CONTEXT"]), os.environ["MOLGAP_RELATION_MODE"])
