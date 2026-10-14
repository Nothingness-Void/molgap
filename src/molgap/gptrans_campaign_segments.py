"""Read-only acceptance of durable campaign prefixes before portable recovery.

A ten-pass prefix remains incomplete scientific work. This owner checks the
existing pause, session, qualification and allocation artifacts, then delegates
recovery to the original transport without changing its frozen source package.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .experiment_family_workflow import (
    RunContext, _artifact_path, _check_context, _digest, _finite_tree,
    _inspect_predictions, _json,
)
from .experiment_package import SIDECARS, read_packaged_text, verify_experiment_source_package
from .experiment_retention import RETENTION_FILENAME, validate_execution_retention
from .experiment_allocation import _prior_segments
from .family_pair_release import RETENTION_PASSES, WALL_CEILING_SECONDS
from .family_training_outputs import _assert_model_schema
from .family_training_runner import (
    _state_fingerprint, _verify_preflight, read_recipe, validate_screen_recipe,
)
from .family_training_session import validate_session_resume
from .gptrans_campaign_release import ARMS, BASE, SLOTS, retired_bootstrap_cost
from .gptrans_cycle_inputs import FIXED_100K_SHA
from .research_memory.trace import file_digest, validate_canonical_trace


def _scope(recipe):
    if (recipe.family, recipe.train_rows, recipe.development_rows, recipe.batch_size,
            recipe.seed, recipe.optimizer_steps, recipe.ema_decay, recipe.selection,
            recipe.schedule.total_units, recipe.schedule.warmup_units,
            recipe.dataset.manifest_sha256, recipe.torch_version) != (
            "gptrans_t", 100000, 50000, 128, 42, 46860, .999,
            "best-development-ema", 60, 4, FIXED_100K_SHA, "2.7.1"):
        raise ValueError("Segment changed the frozen fixed100K/60pass campaign scope")


def _snapshot(paths):
    return {str(path): file_digest(path) for path in paths}


def _verify_actual_resume(native, recipe, observations, *, completed_step):
    """Verify the two retained scratch next-steps when this is a resumed segment."""
    import torch

    path = native / "actual_resume_certificate.json"
    if not path.exists():
        return []
    certificate = _json(_artifact_path(native, path.name))
    step = certificate.get("resume_step")
    if (certificate.get("status") != "accepted" or certificate.get("formal_training_exposure") != 0
            or type(step) is not int or step <= 0
            or step != completed_step - recipe.loop_plan.steps_per_pass * RETENTION_PASSES):
        raise ValueError("Invalid actual retained-state resume qualification")
    origin = next((row for row in observations if row["optimizer_step"] == step), None)
    if origin is None or origin["checkpoint_identity"] != certificate.get("checkpoint_sha256"):
        raise ValueError("Actual resume proof is not bound to a retained segment checkpoint")
    proofs = certificate.get("proofs")
    if type(proofs) is not list or [p.get("path") for p in proofs] != ["actual-resume-a.pt", "actual-resume-b.pt"]:
        raise ValueError("Missing actual resume proof payloads")
    paths, fingerprints = [path], []
    for proof in proofs:
        artifact = _artifact_path(native, proof["path"])
        if file_digest(artifact) != proof["sha256"]:
            raise ValueError("Actual resume proof bytes changed")
        state = torch.load(artifact, map_location="cpu", weights_only=True)
        progress = validate_session_resume(recipe, state)
        if (progress.optimizer_step != step + 1 or progress.status != "paused"
                or len(state["rng_state"]["cuda"]) != 1):
            raise ValueError("Actual resume proof is not one isolated-GPU next-step")
        fingerprint = _state_fingerprint(state)
        if fingerprint != proof["state_fingerprint"]:
            raise ValueError("Actual resume proof tensor identity changed")
        fingerprints.append(fingerprint)
        paths.append(artifact)
        del state
    if len(set(fingerprints)) != 1:
        raise ValueError("Actual resume proof next-steps disagree")
    return paths


def _inspect_prefix(root, context, record, recipe, *, trajectory_id):
    """Check the producer's incomplete session without manufacturing a manifest."""
    import torch

    names = ("last_checkpoint.pt", "selected_model.pt", "development_predictions.pt",
        "canonical_trace.json", "canonical_trace.json.context.json", "training_contract.json",
        "pause_receipt.json", "allocation_costs.json", "invocation_cost_state.json")
    paths = {name: _artifact_path(root, name) for name in names}
    before = _snapshot(paths.values())
    if (root / "output_manifest.json").exists() or (root / "session_commit.json").exists():
        raise ValueError("Completed or unacknowledged producer transaction is not a durable prefix")
    if file_digest(paths["training_contract.json"]) != context.training_recipe_sha256:
        raise ValueError("Retained prefix contract bytes differ from the frozen arm")
    state = torch.load(paths["last_checkpoint.pt"], map_location="cpu", weights_only=True)
    _check_context(state.pop("context"), context)
    progress = validate_session_resume(recipe, state)
    if (progress.status != "paused" or progress.next_batch != 0
            or progress.pass_index not in range(RETENTION_PASSES, 60, RETENTION_PASSES)):
        raise ValueError("Only acknowledged incomplete ten-pass boundaries can be continued")
    if len(state["rng_state"]["cuda"]) != 1:
        raise ValueError("CPU-only or unisolated checkpoint cannot qualify native continuation")
    for weights in (state["model"], state["ema"]):
        if any(value.is_floating_point() and value.dtype != torch.float32 for value in weights.values()):
            raise ValueError("Prefix model/EMA changed FP32 precision")
    if any(value.is_floating_point() and value.dtype != torch.float32
            for moments in state["optimizer"]["state"].values() for value in moments.values()):
        raise ValueError("Prefix optimizer changed FP32 precision")
    checkpoint_sha = before[str(paths["last_checkpoint.pt"])]
    pause = _json(paths["pause_receipt.json"])
    _check_context(pause.get("context"), context)
    if (pause.get("status") != "PAUSED" or pause.get("optimizer_steps") != progress.optimizer_step
            or pause.get("sample_presentations") != progress.sample_presentations
            or pause.get("checkpoint_sha256") != checkpoint_sha):
        raise ValueError("Pause acknowledgement differs from the retained checkpoint")
    costs = _json(paths["allocation_costs.json"])
    invocation = _json(paths["invocation_cost_state.json"])
    for item in (costs, invocation):
        _check_context(item.get("context"), context)
    seconds = costs.get("wall_seconds")
    if (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0
            or costs.get("checkpoint_sha256") != checkpoint_sha or invocation.get("status") != "closed"):
        raise ValueError("Prefix invocation cost is unknown or is not closed against this checkpoint")
    prior, elapsed = invocation.get("prior_wall_seconds"), invocation.get("current_invocation_seconds")
    if (any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in (prior, elapsed))
            or costs.get("current_invocation_seconds") != elapsed
            or not math.isclose(seconds, prior + elapsed, rel_tol=1e-9, abs_tol=1e-6)):
        raise ValueError("Prefix cumulative process cost differs from closed observed invocations")
    trace = validate_canonical_trace(_json(paths["canonical_trace.json"]))
    companion = _json(paths["canonical_trace.json.context.json"])
    _check_context(companion.get("context"), context)
    if (companion.get("stage_id") != "downstream" or trace["trajectory_id"] != trajectory_id
            or trace["run_id"] != context.logical_run_id + ":" + context.arm_id + ":downstream"
            or trace.get("device_time_semantics") != "sum_over_devices"):
        raise ValueError("Prefix trace stage/trajectory/role identity changed")
    rows = trace["observations"]
    steps = list(range(recipe.loop_plan.steps_per_pass, progress.optimizer_step + 1,
                       recipe.loop_plan.steps_per_pass))
    if [r["optimizer_step"] for r in rows] != steps or any(r["event"] != "observation" for r in rows):
        raise ValueError("Prefix trace is missing exact complete-pass exposure")
    for name in ("live_train_metric", "live_dev_metric", "ema_dev_metric"):
        role = record["train_role_identity" if name == "live_train_metric" else "development_role_identity"]
        semantics = {"metric": "MAE", "unit": "eV", "target": "Gap", "role_identity": role,
            "weights": "ema" if name == "ema_dev_metric" else "live", "direction": "minimize"}
        if trace["metric_semantics"][name] != semantics or any(
                type(r[name]) not in (int, float) or not math.isfinite(r[name]) or r[name] < 0 for r in rows):
            raise ValueError("Prefix live/EMA metrics changed their frozen role/weight semantics")
    for row in rows:
        _digest(row["checkpoint_identity"], "prefix.checkpoint_identity")
        if (row["sample_presentations"] != row["optimizer_step"] * recipe.batch_size
                or row["epoch_or_pass"] != row["optimizer_step"] / recipe.loop_plan.steps_per_pass):
            raise ValueError("Prefix trace sample/pass exposure differs from acknowledged updates")
    if rows[-1]["checkpoint_identity"] != checkpoint_sha:
        raise ValueError("Last trace observation is not bound to the paused checkpoint bytes")
    selected = torch.load(paths["selected_model.pt"], map_location="cpu", weights_only=True)
    _check_context(selected.get("context"), context)
    _finite_tree(selected, "prefix.selected")
    if (selected.get("optimizer_step") != state["cursor"]["selected_step"]
            or selected.get("selection") != recipe.selection or selected.get("weights") != "ema"
            or selected.get("epoch_or_pass") != selected["optimizer_step"] / recipe.loop_plan.steps_per_pass):
        raise ValueError("Prefix selected checkpoint changed policy or acknowledged endpoint")
    _assert_model_schema(selected["model"], state["model"])
    predictions = torch.load(paths["development_predictions.pt"], map_location="cpu", weights_only=True)
    if any(predictions[name].dtype != torch.float32 for name in ("target_eV", "prediction_eV")):
        raise ValueError("Prefix retained target/prediction precision differs from FP32")
    del predictions
    observed = _inspect_predictions(paths["development_predictions.pt"], context=context,
        expected=record["acceptance_requirements"], target_identity=None)
    at = next((r for r in rows if r["optimizer_step"] == selected["optimizer_step"]), None)
    if at is None:
        raise ValueError("Selected endpoint is absent from the retained trajectory")
    if selected["optimizer_step"] != min(rows, key=lambda row: row["ema_dev_metric"])["optimizer_step"]:
        raise ValueError("Selected checkpoint is not the first strict best-development EMA endpoint")
    for value in (selected.get("development_mae_eV"), state["cursor"]["best_development_mae_eV"],
                  at["ema_dev_metric"], min(r["ema_dev_metric"] for r in rows)):
        if type(value) not in (int, float) or not math.isclose(
                value, observed["development_mae_eV"], rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError("Prefix selected predictions/trace/resume metric disagree")
    if before != _snapshot(paths.values()):
        raise ValueError("Prefix artifacts changed during acceptance")
    return {"status": "MECHANICALLY_VERIFIED", "progress": {
        "optimizer_steps": progress.optimizer_step, "sample_presentations": progress.sample_presentations,
        "completed_passes": progress.pass_index, "next_batch": 0, "status": progress.status},
        "observed": observed, "artifacts": before}, rows


def accept_campaign_segment(spec, *, package_dir, receipt_path, output_root, contracts):
    """Accept both assigned prefixes and full observed allocation before recovery."""
    package_dir, root = Path(package_dir), Path(output_root).absolute()
    declaration = spec.to_dict()
    number = {"gptrans-research-slot1": 1, "gptrans-research-slot2": 2}.get(declaration["experiment_id"])
    order = [a["arm_id"] for a in declaration["arms"]]
    if number is None or tuple(order) != SLOTS[number - 1] or set(contracts) != set(order):
        raise ValueError("Segment requires one exact authorized two-arm campaign slot")
    # Snapshot before parsing: the subsequent transport must consume these same
    # identities and observations, not a mixture from another producer revision.
    tracked = {root / RETENTION_FILENAME, root / "allocation_ledger.json", root / "execution_report.json",
               *(root / arm_id / "allocation_ledger.json" for arm_id in order),
               *(package_dir / name for name in SIDECARS), Path(receipt_path),
               *(Path(path) for path in contracts.values())}
    snapshot = _snapshot(tracked)
    package = verify_experiment_source_package(package_dir)
    if package["spec_identity"] != spec.identity:
        raise ValueError("Segment Spec differs from the frozen source package")
    campaign = json.loads(read_packaged_text(package_dir, BASE + "/campaign.json"))
    retention = validate_execution_retention(root, spec, package_identity=package["package_identity"],
        source_commit=package["source_commit"], source_archive_sha256=package["archive_sha256"])
    ledger = retention["ledgers"]["root"]
    segments = _prior_segments([ledger], spec.identity)
    reserve = retired_bootstrap_cost(None, number, source_text=read_packaged_text(
        package_dir, f"{BASE}/infrastructure_costs/slot{number}.json"))
    wall = reserve["wall_seconds"] + sum(s["wall_seconds"] for s in segments)
    device = reserve["allocated_device_seconds"] + sum(s["allocated_device_seconds"] for s in segments)
    if (ledger["prior_unobserved_intervals"] or any(s["status"] != "complete" for s in segments)
            or wall >= WALL_CEILING_SECONDS or device >= 16 * 3600):
        raise ValueError("Incomplete, unknown or exhausted full-allocation campaign segment budget")
    execution = retention["ledgers"]["execution_report"]
    if (execution is None or execution.get("status") != "complete" or execution.get("spec_identity") != spec.identity
            or set(execution.get("arms", {})) != set(order) or any(
                a.get("training_started") is not True or a.get("terminal_status") != "interrupted"
                for a in execution["arms"].values())):
        raise ValueError("Physical segment did not retain both incomplete trained arms")
    reports = {}
    for arm_id in order:
        context = RunContext.from_launch(spec, Path(receipt_path), package_dir,
            expected_package_identity=package["package_identity"], arm_id=arm_id)
        record, recipe = read_recipe(Path(contracts[arm_id]))
        validate_screen_recipe(spec, arm_id, record)
        _scope(recipe)
        frozen = campaign["arms"][arm_id]
        if (recipe.mode != ARMS[arm_id] or frozen["contract"]["sha256"] != context.training_recipe_sha256
                or file_digest(Path(contracts[arm_id])) != context.training_recipe_sha256):
            raise ValueError("Campaign segment recipe identity changed")
        native = root / "_resume_preflight" / arm_id
        resumed = native.exists()
        native = native if resumed else root / arm_id
        cert_path = _artifact_path(native, "runtime_certificate.json")
        certificate_snapshot = _snapshot([cert_path])
        published = _json(_artifact_path(native, "runtime_certificate.json"))
        if published.get("status") != "accepted":
            raise ValueError("Prefix lacks accepted native repeat/resume qualification")
        proof_paths = [_artifact_path(native, name) for name in (
            "runtime_provenance.json", "runtime_certificate.json", "resource_qualification.json", "resumed-first.pt")]
        proof_paths += [_artifact_path(native, p["path"]) for p in published["proofs"]]
        if resumed:
            actual_path = _artifact_path(native, "actual_resume_certificate.json")
            certificate_snapshot.update(_snapshot([actual_path]))
            actual = _json(actual_path)
            proof_paths += [_artifact_path(native, "actual_resume_certificate.json")]
            proof_paths += [_artifact_path(native, p["path"]) for p in actual["proofs"]]
        proof_snapshot = _snapshot(proof_paths)
        provenance = _json(_artifact_path(native, "runtime_provenance.json"))
        _check_context(provenance.get("context"), context)
        if (provenance.get("recipe_identity") != recipe.identity
                or provenance.get("dataset_manifest_sha256") != FIXED_100K_SHA
                or "T4" not in provenance.get("accelerator", "")
                or provenance.get("initialization", {}).get("state_sha256") != recipe.initialization_sha256
                or provenance["initialization"].get("file_sha256") != frozen["initial_state"]["sha256"]):
            raise ValueError("Prefix native fixed-data/runtime/initialization identity differs")
        _verify_preflight(native, recipe, provenance)
        resource = _json(_artifact_path(native, "resource_qualification.json"))
        if resource.get("status") != "accepted" or resource.get("zero_start", {}).get("accepted") is not True:
            raise ValueError("Prefix lacks native unchanged-parent qualification")
        binding = next(b for b in declaration["prospective"]["arms"] if b["arm_id"] == arm_id)
        report, observations = _inspect_prefix(root / arm_id, context, record, recipe,
            trajectory_id=binding["trajectory_id"])
        completed = report["progress"]["completed_passes"]
        if (completed > RETENTION_PASSES) != resumed:
            raise ValueError("Retained segment lacks its exact resume-preflight generation")
        if resumed and not (native / "actual_resume_certificate.json").is_file():
            raise ValueError("Resumed prefix lacks actual retained-state qualification")
        actual_paths = (_verify_actual_resume(native, recipe, observations,
            completed_step=report["progress"]["optimizer_steps"]) if resumed else [])
        if (proof_snapshot != _snapshot(proof_paths)
                or certificate_snapshot != _snapshot(map(Path, certificate_snapshot))):
            raise ValueError("Native proof changed or resumed prefix failed to advance a segment")
        report["artifacts"].update(proof_snapshot)
        report["artifacts"].update(_snapshot(actual_paths))
        reports[arm_id] = report
    if len({r["progress"]["completed_passes"] for r in reports.values()}) != 1:
        raise ValueError("Assigned campaign arms ended at different segment boundaries")
    for report in reports.values():
        if report["artifacts"] != _snapshot(map(Path, report["artifacts"])):
            raise ValueError("Accepted arm evidence changed while checking the other assigned arm")
    if snapshot != _snapshot(tracked):
        raise ValueError("Full-allocation retention changed during segment acceptance")
    return {"status": "MECHANICALLY_VERIFIED", "arms": reports,
        "full_allocation": {"wall_seconds": wall, "allocated_device_seconds": device},
        "retired_bootstrap_cost": reserve, "artifacts": snapshot,
        "scientific_acceptance": "NOT_EVALUATED", "replay_readiness": "NOT_EVALUATED",
        "submitted": False, "resume_published": False}


def build_campaign_segment_resume(spec, repo_root, prepared, package_dir, expected_package_identity,
                                  receipt_path, output, *, arm_id, output_root, contracts):
    """Gate the whole slot, then export one arm through the existing resume owner."""
    from .experiment_workflow_resume import build_workflow_resume

    report = accept_campaign_segment(spec, package_dir=package_dir, receipt_path=receipt_path,
        output_root=output_root, contracts=contracts)
    snapshots = dict(report["artifacts"])
    for arm in report["arms"].values():
        snapshots.update(arm["artifacts"])
    if arm_id not in report["arms"] or snapshots != _snapshot(map(Path, snapshots)):
        raise ValueError("Accepted segment bytes changed before recovery export")
    result = build_workflow_resume(spec, repo_root, prepared, package_dir, expected_package_identity,
        receipt_path, output, arm_id=arm_id, source_output=Path(output_root) / arm_id)
    if snapshots != _snapshot(map(Path, snapshots)):
        raise ValueError("Accepted segment changed during recovery export; preserve bundle for reconciliation")
    return {**result, "segment_acceptance": report,
        "scientific_acceptance": "NOT_EVALUATED", "replay_readiness": "NOT_EVALUATED"}
