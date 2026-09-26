"""Accepted-checkpoint NO_TRAIN audit, with hash-bound reference prediction reuse."""
from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

MODES = ("neural_atom_k1_receiver_pair", "neural_atom_k1_triplet_aggregate", "neural_atom_k1_rrwp_pair")
REFERENCE = "neural_atom_k1_v4"


def package(output):
    from molgap.constants import REPO_ROOT
    from molgap.k1_relation_study_records import REL, ROOT, load, save, accept_training
    from molgap.research_memory.trace import file_digest
    if subprocess.check_output(["git", "status", "--porcelain", "--", "src/molgap/k1_relation_audit.py",
            f"{REL}/kaggle_audit", f"{REL}/package_audit.py"], cwd=REPO_ROOT, text=True).strip():
        raise ValueError("Commit the audit helper and launcher before packaging")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new immutable audit package")
    reference = REPO_ROOT / "platforms/_records/kaggle/training/pcqm_k1_v4_reference_s42_v2/pcqm_k1_v4_reference"
    frozen = load(ROOT / "source_config.json")
    source_commit = frozen["source_commit"]
    source_sha = load(ROOT / "submission_receipt_v1.json")["jobs"][0]["source_archive_sha256"]
    roots = {}
    accepted = {}
    for slot in ("dual", "rrwp"):
        record = REPO_ROOT / f"platforms/_records/kaggle/training/k1_relation_resolution_{slot}_v1"
        accepted[slot] = accept_training(reference, record / "pcqm_k1_relation_resolution", source_commit, source_sha, slot)
        for mode in accepted[slot]["candidates"]:
            roots[mode] = record / "pcqm_k1_relation_resolution" / mode
            short = mode.removeprefix("neural_atom_k1_")
            receipt = load(ROOT / f"arms/{short}/rml_plan/rml_finalized/finalization.json")
            if receipt.get("trace_status") != "available":
                raise ValueError("Training terminal closure required before audit")
    retained = REPO_ROOT / "platforms/_records/kaggle/training/pcqm_k1_linear_attention_audit_s42_v1"
    old_acceptance = load(retained / "controller_acceptance.json")
    old_summary = load(REPO_ROOT / "experiments/pcqm_k1_linear_attention_100k/results/audit_acceptance_summary.json")
    if (not old_acceptance["accepted"] or old_acceptance["model_inference_executed_by_acceptance"]
        or file_digest(retained / "controller_acceptance.json") != old_summary["source_acceptance_sha256"]):
        raise ValueError("Reference predictions lack accepted retained provenance")
    old_audit = retained / "pcqm_k1_linear_attention_audit/post100k_audit"
    terminal = load(old_audit / "terminal.json")
    from molgap.k1_portability_audit import TRANSFORM_SHA256
    from molgap.pcqm_k1_cross_scale_diagnostic import MANIFESTS, ARMS
    if (terminal["manifest_sha256"] != MANIFESTS or terminal["target_transform_sha256"] != TRANSFORM_SHA256
        or terminal["checkpoint_sha256"][REFERENCE] != ARMS["k1"]["model_sha256"]):
        raise ValueError("Reference reuse identity differs from frozen benchmark")
    paths = {}
    for mode, root in roots.items():
        for name in ("best_model.pt", "best_development_payload.pt", "arm_record.json"):
            paths[f"candidates/{mode}/{name}"] = root / name
    paths["reference_development.pt"] = reference / REFERENCE / "best_development_payload.pt"
    paths["reference_terminal.json"] = old_audit / "terminal.json"
    paths["target_transform.json"] = REPO_ROOT / "experiments/v5_legacy_evidence_migration/k1_v4_100k_reference/target_transform.json"
    for role, key in (("original_100k", "reproduction"), ("unseen_500k", "unseen_500k")):
        for chunk in terminal[key][REFERENCE]["chunks"]:
            path = old_audit / role / REFERENCE / chunk["file"]
            if file_digest(path) != chunk["sha256"]:
                raise ValueError("Reference chunk changed")
            paths[f"reference/{role}/{REFERENCE}/{chunk['file']}"] = path
    output.mkdir(parents=True)
    with tarfile.open(output / "audit_inputs.bin", "w:gz") as archive:
        for name, path in paths.items():
            archive.add(path, arcname=name, recursive=False)
    shutil.copyfile(REPO_ROOT / "src/molgap/k1_relation_audit.py", output / "audit_impl.py")
    spec = {"format": "molgap-relation-audit-release-v1", "run_id": "kaseichou/molgap-k1-relation-audit-s42:v1",
        "source_commit": source_commit, "source_archive_sha256": source_sha,
        "audit_helper_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "audit_helper_sha256": file_digest(output / "audit_impl.py"),
        "input_archive_sha256": file_digest(output / "audit_inputs.bin"),
        "input_files": {name: file_digest(path) for name, path in paths.items()},
        "model_source_dataset": "kaseichou/molgap-k1-relation-resolution-source",
        "reference_inference_reused": True, "reference_acceptance_sha256": file_digest(retained / "controller_acceptance.json"),
        "modes": list(MODES), "training_authorized": False, "max_allocated_device_seconds": 5400,
        "training_acceptance_sha256": {slot: file_digest(REPO_ROOT / f"platforms/_records/kaggle/training/k1_relation_resolution_{slot}_v1/acceptance.json") for slot in accepted}}
    save(output / "AUDIT_RELEASE.json", spec)
    save(ROOT / "audit_release.json", spec)
    save(output / "dataset-metadata.json", {"id": "kaseichou/molgap-k1-relation-audit-inputs",
        "title": "MolGap K1 Relation Audit Inputs", "isPrivate": True, "licenses": [{"name": "other"}]})
    return spec


def run(inputs, cache_100k, cache_500k, output, spec):
    """Remote CUDA execution only. Training is not imported or called here."""
    import torch
    from molgap.k1_portability_audit import _graphs, _model, _payload, _infer, _joined, TRANSFORM_SHA256
    from molgap.pcqm_k1_cross_scale_diagnostic import MANIFESTS
    from molgap.training_reproducibility import atomic_json, sha256_file
    inputs, output = Path(inputs), Path(output)
    for name, digest in spec["input_files"].items():
        if sha256_file(inputs / name) != digest:
            raise ValueError(f"Frozen audit input changed: {name}")
    if spec["training_authorized"] is not False or tuple(spec["modes"]) != MODES:
        raise ValueError("NO_TRAIN release changed")
    if sha256_file(inputs / "target_transform.json") != TRANSFORM_SHA256:
        raise ValueError("Target transform changed")
    transform = json.loads((inputs / "target_transform.json").read_text())
    mean, std = float(transform["mean"]), float(transform["std"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if torch.cuda.device_count() != 1:
        raise ValueError("One visible inference device required")
    started = time.monotonic()
    old = json.loads((inputs / "reference_terminal.json").read_text())
    records = {mode: json.loads((inputs / "candidates" / mode / "arm_record.json").read_text()) for mode in MODES}
    for mode, record in records.items():
        if (not record["complete"] or record["source_commit"] != spec["source_commit"]
            or record["contract"]["source_archive_sha256"] != spec["source_archive_sha256"]):
            raise ValueError(f"Candidate source or completion changed: {mode}")
    for role in ("original_100k", "unseen_500k"):
        shutil.copytree(inputs / "reference" / role / REFERENCE, output / role / REFERENCE)
    reproduction = {REFERENCE: old["reproduction"][REFERENCE]}
    audit = {REFERENCE: old["unseen_500k"][REFERENCE]}
    original = _graphs(cache_100k, "original_100k")
    for mode, record in records.items():
        root = inputs / "candidates" / mode
        model = _model(mode, root / "best_model.pt", record["training"]["best_model_sha256"])
        saved = _payload(root / "best_development_payload.pt", record["training"]["payload_sha256"], mode)
        target = output / "original_100k" / mode
        chunks = _infer(original, model, start=100000, output=target, role="original_100k", mode=mode, mean=mean, std=std)
        joined = _joined(target, chunks)
        diff = float((joined["prediction_eV"] - saved["prediction_eV"].view(-1).float()).abs().max())
        if diff > .001 or not torch.equal(joined["target_eV"], saved["target_eV"].view(-1).float()):
            raise ValueError(f"Original-role reproduction failed: {mode}")
        reproduction[mode] = {"max_abs_eV": diff, "chunks": chunks}
        del model
    del original
    unseen = _graphs(cache_500k, "unseen_500k")
    for mode, record in records.items():
        model = _model(mode, inputs / "candidates" / mode / "best_model.pt", record["training"]["best_model_sha256"])
        target = output / "unseen_500k" / mode
        chunks = _infer(unseen, model, start=500000, output=target, role="unseen_500k", mode=mode, mean=mean, std=std)
        joined = _joined(target, chunks)
        audit[mode] = {"mae_eV": float((joined["prediction_eV"] - joined["target_eV"]).abs().mean()), "rows": 50000, "chunks": chunks}
        del model
    terminal = {"format": "molgap-k1-post100k-portability-audit-v1", "complete": True,
        "experiment_purpose": "NO_TRAIN", "training_executed_in_audit_stage": False,
        "model_inference_executed": True, "reference_model_inference_executed": False,
        "reference_inference_reused": True, "reference_terminal_sha256": sha256_file(inputs / "reference_terminal.json"),
        "source_commit": spec["source_commit"], "source_archive_sha256": spec["source_archive_sha256"],
        "manifest_sha256": MANIFESTS, "target_transform_sha256": TRANSFORM_SHA256,
        "reference_bundle_id": "reference-k1-v4-100k-s42-v5-recovered",
        "checkpoint_sha256": {REFERENCE: old["checkpoint_sha256"][REFERENCE],
            **{mode: record["training"]["best_model_sha256"] for mode, record in records.items()}},
        "reproduction": reproduction, "unseen_500k": audit,
        "official_validation_role_read": False, "test_dev_role_read": False, "test_challenge_role_read": False,
        "elapsed_seconds": time.monotonic() - started}
    atomic_json(output / "terminal.json", terminal)
    return terminal


def accept_audit(audit_root):
    """Recompute retained chunks only; does not instantiate or execute a model."""
    import torch
    from molgap.constants import REPO_ROOT
    from molgap.k1_relation_study_records import ROOT, load
    from molgap.k1_relation_study_runtime import SLOTS
    from molgap.research_memory.trace import file_digest
    from molgap.pcqm_k1_cross_scale_diagnostic import MANIFESTS
    root = Path(audit_root)
    terminal = load(root / "post100k_audit/terminal.json")
    execution = load(root / "execution.json")
    spec = load(ROOT / "audit_release.json")
    if (execution["spec"] != spec or not execution["complete"] or execution["training_executed"] is not False
        or execution["optimizer_steps"] != 0 or execution["cublas_workspace_config"] != ":4096:8"
        or not terminal["complete"] or terminal["training_executed_in_audit_stage"] is not False
        or terminal["source_commit"] != spec["source_commit"]
        or terminal["source_archive_sha256"] != spec["source_archive_sha256"]
        or terminal["manifest_sha256"] != MANIFESTS or terminal["reference_inference_reused"] is not True
        or terminal["reference_model_inference_executed"] is not False
        or terminal["reference_terminal_sha256"] != spec["input_files"]["reference_terminal.json"]
        or any(terminal[flag] is not False for flag in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"))):
        raise ValueError("Audit identity, execution or role truth invalid")
    path = REPO_ROOT / "experiments/pcqm_k1_linear_attention_100k/accept.py"
    adapter_spec = importlib.util.spec_from_file_location("relation_audit_saved_chunk_acceptance", path)
    adapter = importlib.util.module_from_spec(adapter_spec)
    adapter_spec.loader.exec_module(adapter)
    reference_root = REPO_ROOT / "platforms/_records/kaggle/training/pcqm_k1_v4_reference_s42_v2/pcqm_k1_v4_reference" / REFERENCE
    names = (REFERENCE, *MODES)
    if set(terminal["reproduction"]) != set(names) or set(terminal["unseen_500k"]) != set(names):
        raise ValueError("Audit model inventory differs from the release")
    results, aligned_target = {}, None
    for mode in names:
        if mode == REFERENCE:
            arm = reference_root
        else:
            slot = next(slot for slot, modes in SLOTS.items() if mode in modes)
            arm = REPO_ROOT / f"platforms/_records/kaggle/training/k1_relation_resolution_{slot}_v1/pcqm_k1_relation_resolution/{mode}"
        if terminal["checkpoint_sha256"][mode] != file_digest(arm / "best_model.pt"):
            raise ValueError("Audit checkpoint differs from accepted best model")
        saved = torch.load(arm / "best_development_payload.pt", map_location="cpu", weights_only=False)
        reproduced = adapter._joined(root / "post100k_audit/original_100k" / mode, terminal["reproduction"][mode]["chunks"], 100000)
        diff = float((reproduced["prediction_eV"] - saved["prediction_eV"].view(-1)).abs().max())
        if (not torch.equal(reproduced["target_eV"], saved["target_eV"].view(-1)) or diff > .001
            or abs(diff - terminal["reproduction"][mode]["max_abs_eV"]) > 1e-7):
            raise ValueError("Original development reproduction failed")
        unseen = adapter._joined(root / "post100k_audit/unseen_500k" / mode, terminal["unseen_500k"][mode]["chunks"], 500000)
        if aligned_target is not None and not torch.equal(aligned_target, unseen["target_eV"]):
            raise ValueError("Audit target alignment failed")
        aligned_target = unseen["target_eV"]
        mae = float((unseen["prediction_eV"] - unseen["target_eV"]).abs().mean())
        if abs(mae - terminal["unseen_500k"][mode]["mae_eV"]) > 1e-7:
            raise ValueError("Audit metric differs from retained chunks")
        if mode == REFERENCE:
            for role, key in (("original_100k", "reproduction"), ("unseen_500k", "unseen_500k")):
                for row in terminal[key][mode]["chunks"]:
                    if row["sha256"] != spec["input_files"][f"reference/{role}/{mode}/{row['file']}"]:
                        raise ValueError("Reused reference bytes were replaced")
        results[mode] = {"mae_eV": mae, "reproduction_max_abs_eV": diff}
    return {"accepted": True, "model_inference_executed_by_acceptance": False,
        "reference_inference_reused": True, "training_executed": False, "run_id": spec["run_id"],
        "audit": results, "official_validation_role_read": False, "test_dev_role_read": False,
        "test_challenge_role_read": False, "full_training_authorized": False}
