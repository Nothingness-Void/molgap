"""Question-owned CPU evidence translation; reuse the existing RML finalizer."""
from pathlib import Path
import json

from .research_memory.finalize import finalize
from .training_reproducibility import atomic_json, sha256_file


def accept_cpu_sidecar(*, output, preparation, receipt, fixed_manifests,
                       local_entry, pulled_entry, prior_wall_seconds=0):
    """Stream saved tensors one chunk at a time; no model or original graph load."""
    import torch
    from .gptrans_cycle_inputs import FORMAT, FIXED_100K_SHA, FIXED_500K_SHA, validate_chunk, _confined
    from .pcqm_fixed_records import graph_records
    from .v4_runtime import normalized_source_sha256
    output = Path(output).resolve()
    prepared, returned = (json.loads(Path(p).read_bytes()) for p in (preparation, receipt))
    path = output / "manifest.json"
    manifest = json.loads(path.read_bytes())
    native = json.loads((output / "cost.json").read_bytes())
    if (returned.get("status") != "submitted" or returned.get("accelerator") != "CPU-only" or
            returned.get("kernel") != prepared["run"] or returned.get("reconciliation_required") is not False or
            normalized_source_sha256(Path(local_entry)) != normalized_source_sha256(Path(pulled_entry))):
        raise ValueError("CPU source/receipt reconciliation failed")
    if (manifest.get("format") != FORMAT or manifest.get("status") != "complete" or
            manifest.get("maximum_cycle") != 18 or manifest.get("source_commit") != prepared["source_commit"] or
            manifest.get("source_archive_sha256") != prepared["archive_sha256"] or
            manifest.get("independent_algorithm") != "networkx.chordless_cycles" or
            any(manifest.get(k) is not False for k in ("labels_used", "model_executed", "geometry_used", "gpu_used"))):
        raise ValueError("CPU derivation status/source/role attestation changed")
    expected = []
    for fixed in fixed_manifests:
        fixed = Path(fixed)
        digest = sha256_file(fixed)
        if digest not in (FIXED_100K_SHA, FIXED_500K_SHA):
            raise ValueError("Unaccepted fixed data metadata")
        original = json.loads(fixed.read_bytes())
        roles = ("train", "development") if digest == FIXED_100K_SHA else ("development",)
        for record in graph_records(original, digest):
            if record["role"] in roles:
                expected.append((digest, record))
    if len(manifest["chunks"]) != len(expected):
        raise ValueError("Missing or extra cycle chunks")
    rows = pairs = 0
    for (digest, original), record in zip(expected, manifest["chunks"]):
        if (record["source_manifest_sha256"] != digest or record["role"] != original["role"] or
                record["source_payload_sha256"] != original["sha256"] or record["rows"] != original["rows"] or
                record["source_start"] != original["source_idx_min"] or
                record["source_stop"] != original["source_idx_max"] + 1 or record.get("independently_rederived") is not True):
            raise ValueError("Cycle role, row or source bytes differ from accepted metadata")
        chunk = _confined(output, record["file"])
        if sha256_file(chunk) != record["sha256"]:
            raise ValueError("Retrieved cycle bytes changed")
        payload = torch.load(chunk, map_location="cpu", weights_only=True, mmap=True)
        validate_chunk(payload, record)
        rows += record["rows"]
        pairs += record["pairs"]
        del payload
    if rows != manifest["rows"] or pairs != manifest["pairs"]:
        raise ValueError("Cycle total coverage changed")
    if (native["source_commit"] != prepared["source_commit"] or
            native["archive_sha256"] != prepared["archive_sha256"] or native["gpu_device_hours"] != 0 or
            not 0 < native["wall_seconds"] + prior_wall_seconds <= 7200 or
            native["allocated_cpu_hours_status"] != "measurement_missing" or native["allocated_cpu_hours"] is not None):
        raise ValueError("CPU cost/source binding changed or budget exceeded")
    return {"format": "molgap-cycle-sidecar-acceptance-v1", "accepted": True,
        "kernel": returned["kernel"], "kernel_id": returned["kernel_id"], "version": returned["version_number"],
        "source_commit": prepared["source_commit"], "archive_sha256": prepared["archive_sha256"],
        "manifest_sha256": sha256_file(path), "rows": rows, "pairs": pairs, "chunks": len(expected),
        "checks": ["source_pull_normalized_equal", "exact_fixed_manifests", "source_shard_pins",
            "complete_ordered_role_coverage", "saved_chunk_sha256", "decoded_tensor_schema_bounds_symmetry",
            "frozen_remote_two_algorithm_every_row_rederivation", "no_model_or_geometry_feature_use", "native_budget"],
        "independent_rederivation_scope": "two algorithms executed by frozen CPU source; local acceptance checks saved bytes, not a third graph rederivation",
        "wall_seconds_this_attempt": native["wall_seconds"], "wall_seconds_all_attempts": native["wall_seconds"] + prior_wall_seconds,
        "gpu_device_hours": 0, "allocated_cpu_hours": None, "allocated_cpu_hours_status": "measurement_missing",
        "scientific_gain": "NOT_EVALUATED", "gpu_training_released": False}


def close_cpu_failure(root, *, plan_path, receipt_path, observation, failure_path,
                      decision_path, result, finalized_at):
    root = Path(root).resolve()
    paths = [Path(p).resolve() for p in (plan_path, receipt_path, observation, failure_path, decision_path, result)]
    for path in paths:
        path.relative_to(root)
    plan_path, receipt_path, observation, failure_path, decision_path, result = paths
    if (plan_path.parent / "rml_finalized").exists():
        return finalize(root, plan_path, result / "terminal.json")
    frozen, receipt, failure = (json.loads(p.read_bytes()) for p in (plan_path, receipt_path, failure_path))
    logs = list(observation.glob("*.log"))
    if len(logs) != 1:
        raise ValueError("Expected one retained native kernel log")
    # Kaggle CLI may save progress glyphs using the Windows console encoding.
    # Raw bytes stay hash-bound; ASCII traceback fields do not depend on glyphs.
    log = json.loads(logs[0].read_text(encoding="utf-8", errors="replace"))
    error = "".join(row["data"] for row in log if row["stream_name"] == "stderr")
    native_path = observation / "cycle_inputs/cost.json"
    native = json.loads(native_path.read_bytes())
    if ((observation / "cycle_inputs/manifest.json").exists() or
            (observation / "cycle_inputs/progress.json").exists() or
            "Traceback (most recent call last)" not in error or
            ("KeyError: 'assets'" not in error and "ModuleNotFoundError: No module named 'molgap.pcqm_wedge'" not in error)):
        raise ValueError("Not an independently qualified pre-row CPU infrastructure failure")
    action = frozen["actions"][0]
    run, tid = action["run_ids"][0], frozen["trajectory_id"]
    if (receipt["status"] != "submitted" or receipt["accelerator"] != "CPU-only" or
            receipt["kernel"] != run or receipt["kernel_id"] != failure["kernel_id"] or
            receipt["version_number"] != failure["version"] or native["gpu_device_hours"] != 0 or
            native["source_commit"] != action["source_commit"] or
            native["wall_seconds"] != failure["wall_seconds"] or failure["processed_rows"] != 0):
        raise ValueError("CPU failure identity/cost differs from its prospective attempt")
    def rel(path):
        return path.relative_to(root).as_posix()
    decision = {"decision_ref": rel(decision_path), "outcome": "INFRASTRUCTURE_ONLY",
        "next_allowed_actions": [], "reopen_conditions": ["Separately prospective tested infrastructure recovery"]}
    outcome = {"execution_status": "failed_before_first_graph_completed",
        "artifact_status": "failure_metadata_verified", "comparison_status": "not_evaluated",
        "scientific_status": "not_evaluated", "transfer_status": "not_evaluated",
        "budget_decision": "measured_cpu_wall_retained_no_gpu", "full_handoff_status": "not_authorized"}
    eid = "pcqm-" + tid.removeprefix("TC-") + "-infrastructure"
    acceptance_path = result / "acceptance.json"
    role_use = {"official_validation": "untouched", "test_dev": "untouched", "test_challenge": "untouched",
        "train": "unknown", "internal_development_100000_150000": "untouched",
        "internal_development_500000_550000": "untouched"}
    cost = {"schema": "molgap-cost-event-v1", "cost_event_id": action["cost_event_ids"][0],
        "trajectory_id": tid, "action_id": action["action_id"], "run_id": run,
        "attempt_id": action["attempt_ids"][0], "category": "infrastructure_failure",
        "platform": "kaggle2", "hardware": "CPU-only", "evidence_ref": rel(acceptance_path),
        "measurement": {"device_hours": {"status": "not_applicable", "value": None},
            "cpu_hours": {"status": "measurement_missing", "value": None},
            "wall_hours": {"status": "measured", "value": native["wall_seconds"] / 3600},
            "queue_hours": {"status": "measurement_missing", "value": None}}}
    atomic_json(acceptance_path, {"evidence_id": eid, "run_id": run, "outcome": outcome,
        "trajectory_decision": decision, "role_use": role_use, "costs": [cost], "roles": [],
        "native_process_cpu_seconds": native["process_cpu_seconds"], "processed_graph_rows": 0,
        "training_executed": False, "inference_executed": False,
        "deserialization_completion": "not_completed; no label-use claim inferred"})
    pointers = [plan_path, receipt_path, failure_path, decision_path, acceptance_path, native_path, logs[0]]
    hashes = {rel(p): sha256_file(p) for p in pointers}
    evidence = {"format": "molgap-v5-evidence-envelope-v1", "contract": "MOLGAP-COMMON-V5-FINAL",
        "evidence_id": eid, "track": "C", "scope": "cpu_input_preparation_failure",
        "legacy_contract": "none-v5-prospective", "outcome": outcome, "role_use": role_use,
        "authority": {"pointers": list(hashes)},
        "artifacts": [{"name": p.name, "locator": rel(p), "sha256": hashes[rel(p)],
            "availability": "retained_metadata"} for p in (native_path, logs[0])],
        "migration": {"migrated_at": finalized_at, "training_executed": False,
            "inference_executed": False, "scientific_reinterpretation": False,
            "verification_scope": "Exact CPU receipt, source commit, retained traceback and native cost; no completed graph or scientific score"}}
    terminal = result / "terminal.json"
    atomic_json(terminal, {"format": "molgap-rml-terminal-package-v1", "trajectory_id": tid,
        "run_id": run, "action_id": action["action_id"], "finalized_at": finalized_at,
        "acceptance_ref": rel(acceptance_path), "artifact_hashes": hashes, "evidence": evidence,
        "decision": decision, "costs": [cost], "roles": [], "role_use": role_use})
    return finalize(root, plan_path, terminal)


def close_cpu_success(root, *, plan_path, acceptance_path, output, decision_path, result, finalized_at):
    """Translate accepted cache/incidentally decoded roles, never training evidence."""
    root = Path(root).resolve()
    plan_path, acceptance_path, output, decision_path, result = (
        Path(p).resolve() for p in (plan_path, acceptance_path, output, decision_path, result))
    for path in (plan_path, acceptance_path, output, decision_path, result):
        path.relative_to(root)
    if (plan_path.parent / "rml_finalized").exists():
        return finalize(root, plan_path, result / "terminal.json")
    frozen = json.loads(plan_path.read_bytes())
    accepted = json.loads(acceptance_path.read_bytes())
    manifest = json.loads((output / "manifest.json").read_bytes())
    native = json.loads((output / "cost.json").read_bytes())
    action = frozen["actions"][0]
    tid, run = frozen["trajectory_id"], action["run_ids"][0]
    if (accepted.get("accepted") is not True or accepted["kernel"] != run or
            accepted["source_commit"] != action["source_commit"] or
            accepted["manifest_sha256"] != sha256_file(output / "manifest.json") or
            accepted["rows"] != 200000 or accepted["gpu_training_released"] is not False):
        raise ValueError("CPU terminal is not its accepted source-bound complete sidecar")
    def rel(path):
        return path.relative_to(root).as_posix()
    terminal_acceptance = result / "acceptance.json"
    decision = {"decision_ref": rel(decision_path), "outcome": "NO_TRAIN", "next_allowed_actions": [],
        "reopen_conditions": ["Separately validated shared-contract GPU release after user decision"]}
    outcome = {"execution_status": "complete", "artifact_status": "cycle_inputs_independently_accepted",
        "comparison_status": "not_evaluated", "scientific_status": "not_evaluated",
        "transfer_status": "not_evaluated", "budget_decision": "cpu_budget_passed_zero_gpu",
        "full_handoff_status": "not_authorized"}
    from .gptrans_cycle_inputs import FIXED_100K_SHA
    role_use = {"official_validation": "untouched", "test_dev": "untouched", "test_challenge": "untouched"}
    roles, seen = [], set()
    for chunk in manifest["chunks"]:
        digest = chunk["source_manifest_sha256"]
        role = "train" if chunk["role"] == "train" else (
            "internal_development_100000_150000" if digest == FIXED_100K_SHA else "internal_development_500000_550000")
        if role in seen:
            continue
        seen.add(role)
        role_use[role] = "labels_read"
        roles.append({"schema": "molgap-role-event-v1", "role_event_id": f"role-{tid}-{role}-incidental-deserialization",
            "trajectory_id": tid, "action_id": action["action_id"], "run_id": run,
            "dataset_identity": "accepted-fixed-pcqm@" + digest, "row_manifest_hash": digest,
            "role_name": role, "access_kind": "labels_read", "selection_used": False,
            "evidence_ref": rel(terminal_acceptance),
            "semantics": "incidental trusted packed-file deserialization; labels never inspected, scored or used to select cycles"})
    cost = {"schema": "molgap-cost-event-v1", "cost_event_id": action["cost_event_ids"][0],
        "trajectory_id": tid, "action_id": action["action_id"], "run_id": run,
        "attempt_id": action["attempt_ids"][0], "category": "cache_build", "platform": "kaggle2",
        "hardware": "CPU-only", "evidence_ref": rel(terminal_acceptance), "measurement": {
            "device_hours": {"status": "not_applicable", "value": None},
            "cpu_hours": {"status": "measurement_missing", "value": None},
            "wall_hours": {"status": "measured", "value": native["wall_seconds"] / 3600},
            "queue_hours": {"status": "measurement_missing", "value": None}}}
    eid = "pcqm-gptrans-common-cycle-inputs-accepted-v1"
    atomic_json(terminal_acceptance, {"evidence_id": eid, "run_id": run, "outcome": outcome,
        "trajectory_decision": decision, "role_use": role_use, "roles": roles, "costs": [cost],
        "sidecar_acceptance_ref": rel(acceptance_path), "training_executed": False,
        "inference_executed": False, "label_scoring_executed": False})
    pointers = [plan_path, acceptance_path, decision_path, terminal_acceptance,
        output / "manifest.json", output / "cost.json", *[output / c["file"] for c in manifest["chunks"]]]
    hashes = {rel(p): sha256_file(p) for p in pointers}
    evidence = {"format": "molgap-v5-evidence-envelope-v1", "contract": "MOLGAP-COMMON-V5-FINAL",
        "evidence_id": eid, "track": "C", "scope": "accepted_cpu_common_cycle_inputs",
        "legacy_contract": "none-v5-prospective", "outcome": outcome, "role_use": role_use,
        "authority": {"pointers": list(hashes)},
        "artifacts": [{"name": p.name, "locator": rel(p), "sha256": hashes[rel(p)],
            "availability": "retained_metadata" if p.suffix == ".json" else "retained_local_payload"} for p in pointers[4:]],
        "migration": {"migrated_at": finalized_at, "training_executed": False,
            "inference_executed": False, "scientific_reinterpretation": False,
            "verification_scope": "Accepted frozen-source cycle cache only; no model quality or replay-training claim"}}
    terminal = result / "terminal.json"
    atomic_json(terminal, {"format": "molgap-rml-terminal-package-v1", "trajectory_id": tid,
        "run_id": run, "action_id": action["action_id"], "finalized_at": finalized_at,
        "acceptance_ref": rel(terminal_acceptance), "artifact_hashes": hashes,
        "evidence": evidence, "decision": decision, "costs": [cost], "roles": roles, "role_use": role_use})
    return finalize(root, plan_path, terminal)
