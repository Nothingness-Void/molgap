"""Saved-output acceptance for short GPTrans probes, without model execution."""
import json
from pathlib import Path

from .experiment_family_workflow import _artifact_path, _check_context, _finite_tree
from .research_memory.trace import file_digest


def inspect_variant_state(state, updates):
    """Check portable AdamW/EMA/RNG endpoints, including each parameter's cursor."""
    import torch
    from .experiment_family_workflow import _validate_rng_state
    if not {"model", "ema", "optimizer", "scheduler", "rng_state"} <= state.keys():
        raise ValueError("Incomplete disposable state")
    _finite_tree(state, "probe state")
    model, ema, optimizer = (state[key] for key in ("model", "ema", "optimizer"))
    if not model or model.keys() != ema.keys():
        raise ValueError("Probe model/EMA identities differ")
    for name, tensor in model.items():
        if (not torch.is_tensor(tensor) or tensor.device.type != "cpu"
                or tensor.shape != ema[name].shape or tensor.dtype != ema[name].dtype):
            raise ValueError("Nonportable or incompatible model/EMA tensor")
    slots = optimizer["state"]
    ids = [item for group in optimizer["param_groups"] for item in group["params"]]
    if len(ids) != len(set(ids)) or set(ids) != slots.keys() or len(ids) != len(model):
        raise ValueError("Incomplete optimizer parameter ownership")
    for slot, tensor in zip((slots[i] for i in ids), model.values()):
        if (set(slot) != {"step", "exp_avg", "exp_avg_sq"}
                or float(slot["step"]) != updates
                or any(slot[key].shape != tensor.shape for key in ("exp_avg", "exp_avg_sq"))):
            raise ValueError("Incomplete optimizer update endpoint")
    if state["scheduler"] != {"epoch": 0}:
        raise ValueError("Disposable probe changed the frozen learning-rate epoch")
    _validate_rng_state(state["rng_state"])
    if len(state["rng_state"]["cuda"]) != 1:
        raise ValueError("Probe needs one isolated CUDA RNG stream")


def inspect_saved_probes(*, package, receipt, output_root, verify_all_witnesses=False):
    """Default to small metadata; an explicit full audit also checks tensors.

    This callable never downloads. Metadata verification is a different status
    from an independent tensor audit and never requires the entire output set.
    """
    import torch
    from .experiment_spec import ExperimentSpec
    from .experiment_package import verify_experiment_source_package, read_packaged_text
    from .experiment_family_workflow import RunContext
    from .experiment_retention import validate_execution_retention
    from .family_training_runner import _recipe_configuration, _state_fingerprint
    from .gptrans_execution_analysis import _native_validator
    from .gptrans_infrastructure_comparison import comparison_base
    from .gptrans_execution_diagnostic import configuration, phases, variants_for_arm, summarize_probes
    from .screen_policy import canonical_fingerprint
    torch.set_num_threads(1)
    package, root = Path(package).absolute(), Path(output_root).absolute()
    load = lambda p: json.loads(Path(p).read_bytes())
    manifest = verify_experiment_source_package(package)
    spec = ExperimentSpec.from_json((package / "experiment_spec.json").read_text())
    base = comparison_base(spec)
    diagnostic = json.loads(read_packaged_text(package, base + "/diagnostic.json"))
    if diagnostic != configuration(diagnostic["kind"]):
        raise ValueError("Diagnostic dimensions differ from the accepted short protocol")
    validate_execution_retention(root, spec, package_identity=manifest["package_identity"],
        source_commit=manifest["source_commit"], source_archive_sha256=manifest["archive_sha256"])
    report = load(root / "execution_report.json")
    expected_phases = phases(diagnostic)
    if (report["status"] != "complete" or report["spec_identity"] != spec.identity
            or report["diagnostic"] != diagnostic or report["scientific_training_executed"] is not False
            or [row["phase"] for row in report["phases"]] != list(expected_phases)
            or report["hardware"] != ["Tesla T4", "Tesla T4"]):
        raise ValueError("Incomplete or differently bound short execution")
    for row in report["phases"]:
        workers = row["workers"]
        if (row["status"] != "complete" or len(workers) != 2
                or {w["arm_id"] for w in workers} != {"original", "optimized"}
                or {w["device"] for w in workers} != {0, 1}
                or any(w["returncode"] != 0 for w in workers)):
            raise ValueError("Missing successful independent phase workers")
    bootstrap = load(root.parent / "bootstrap_allocation.json")
    if (bootstrap["status"] != "complete" or bootstrap["allocation_device_count"] != 2
            or not 0 < bootstrap["wall_seconds"] <= diagnostic["wall_ceiling_seconds"]
            or bootstrap["allocated_device_seconds"] != 2 * bootstrap["wall_seconds"]):
        raise ValueError("Incomplete allocation or exceeded budget")
    checked, endpoint_fingerprints = [], {}
    for arm in ("original", "optimized"):
        context = RunContext.from_launch(spec, Path(receipt), package,
            expected_package_identity=manifest["package_identity"], arm_id=arm)
        arm_root = root / arm
        recipe = json.loads(read_packaged_text(package, base + "/" + arm + "/contract.json"))
        identity = canonical_fingerprint(recipe) if arm == "original" else _recipe_configuration(recipe).identity
        for group in ("profile_artifacts", "diagnostic_artifacts"):
            refs = report[group][arm]
            if len({ref["path"] for ref in refs}) != len(refs):
                raise ValueError("Aliased retained artifacts")
            for ref in refs:
                if not verify_all_witnesses and not ref["path"].endswith(".json"):
                    continue
                path = _artifact_path(arm_root, ref["path"])
                if file_digest(path) != ref["sha256"]:
                    raise ValueError("Sealed probe artifact changed: " + ref["path"])
                checked.append(arm + "/" + ref["path"])
        certificate = load(arm_root / "runtime_certificate.json")
        if verify_all_witnesses:
            _native_validator(arm_id=arm, root=arm_root, context=context, record=recipe,
                certificate=certificate, identity=identity)
        elif certificate.get("status", "accepted" if certificate.get("accepted") else None) != "accepted":
            raise ValueError("Native certificate is not accepted")
        endpoint_fingerprints[arm] = {}
        for phase in expected_phases[1:]:
            directory = arm_root / phase
            record = load(directory / "output_manifest.json")
            _check_context(record["context"], context)
            worker = next(w for p in report["phases"] if p["phase"] == phase
                for w in p["workers"] if w["arm_id"] == arm)
            expected_device = (0 if arm == "original" else 1) ^ (phase == "diagnostic-1")
            if (record["status"] != "COMPLETE" or record["diagnostic"] != diagnostic
                    or record["cpu_threads"] != 1 or record["phase"] != phase
                    or record["physical_device"] != str(expected_device) or worker["device"] != expected_device
                    or record["runtime"]["torch"] != "2.7.1+cu126" or record["runtime"]["cuda"] != "12.6"
                    or record["runtime"]["device"] != "Tesla T4"
                    or record["runtime"]["determinism"]["tf32_enabled"] is not False
                    or record["runtime"]["determinism"]["deterministic_algorithms"] is not True):
                raise ValueError("Probe phase/runtime/device controls differ")
            expected_variants = variants_for_arm(arm) if diagnostic["kind"] == "attribution" else (
                "native" if arm == "original" else "optimized",)
            if tuple(v["variant"] for v in record["variants"]) != expected_variants:
                raise ValueError("Missing frozen factor variant")
            for ref in record["files"]:
                if not verify_all_witnesses and not ref["path"].endswith(".json"):
                    continue
                if file_digest(_artifact_path(directory, ref["path"])) != ref["sha256"]:
                    raise ValueError("Probe manifest artifact changed")
            for variant in record["variants"]:
                attribution = diagnostic["kind"] == "attribution"
                warmup = 0 if attribution else diagnostic["warmup_steps"]
                measured = diagnostic["attribution_steps"] if attribution else diagnostic[
                    "profile_steps" if phase == "diagnostic-profile" else "measured_steps"]
                if (variant["updates"] != warmup + measured or variant["warmup_updates"] != warmup
                        or variant["measured_updates"] != measured):
                    raise ValueError("Probe exposure differs")
                endpoint_fingerprints[arm][phase + "/" + variant["variant"]] = variant["last_state_fingerprint"]
                if not verify_all_witnesses:
                    if len(variant["steps"]) != (measured if attribution else 0):
                        raise ValueError("Missing per-step attribution receipts")
                    continue
                state = torch.load(directory / (variant["variant"] + "-last.pt"), map_location="cpu", weights_only=True)
                inspect_variant_state(state, warmup + measured)
                fingerprint = _state_fingerprint(state)
                if fingerprint != variant["last_state_fingerprint"]:
                    raise ValueError("Disposable endpoint tensor identity differs")
                endpoint_fingerprints[arm][phase + "/" + variant["variant"]] = fingerprint
                if warmup:
                    state = torch.load(directory / "warmup_checkpoint.pt", map_location="cpu", weights_only=True)
                    inspect_variant_state(state, warmup)
                if len(variant["steps"]) != (measured if attribution else 0):
                    raise ValueError("Missing per-step attribution witnesses")
                for index, ref in enumerate(variant["steps"], 1):
                    path = _artifact_path(directory, ref["path"])
                    if file_digest(path) != ref["sha256"] or ref["step"] != index:
                        raise ValueError("Attribution step integrity differs")
                    state = torch.load(path, map_location="cpu", weights_only=True)
                    inspect_variant_state(state, index)
                    if (state["step"] != index or state["variant"] != variant["variant"]
                            or state["batch_sha256"] != ref["batch_sha256"]
                            or state["prediction"].numel() != diagnostic["physical_batch"]
                            or _state_fingerprint(state) != ref["state_fingerprint"]):
                        raise ValueError("Attribution state/input identity differs")
    summary_path = _artifact_path(root, report["diagnostic_summary"]["path"])
    if file_digest(summary_path) != report["diagnostic_summary"]["sha256"]:
        raise ValueError("Sealed CPU diagnostic summary changed")
    summary = (summarize_probes(root, diagnostic, publish=False)
        if verify_all_witnesses or diagnostic["kind"] == "timing" else load(summary_path))
    if (summary != load(summary_path) or summary["status"] != "COMPLETE"
            or summary["kind"] != diagnostic["kind"]):
        raise ValueError("Independent saved-tensor summary differs from remote output")
    return {"status": "SHORT_PROBE_VERIFIED" if verify_all_witnesses else "SHORT_PROBE_METADATA_VERIFIED",
        "verification_scope": "all pinned tensors" if verify_all_witnesses else "small sealed metadata only",
        "kind": diagnostic["kind"],
        "spec_identity": spec.identity, "source_commit": manifest["source_commit"],
        "package_identity": manifest["package_identity"], "source_archive_sha256": manifest["archive_sha256"],
        "checked_artifact_count": len(checked), "native_preflight": {
            a: "ACCEPTED" if verify_all_witnesses else "CERTIFICATE_BOUND" for a in endpoint_fingerprints},
        "endpoint_fingerprints": endpoint_fingerprints, "summary": summary,
        "allocated_t4_hours": bootstrap["allocated_device_seconds"] / 3600,
        "scientific_acceptance": "NOT_EVALUATED", "replay_readiness": "NOT_APPLICABLE_EXECUTION_DIAGNOSTIC"}
