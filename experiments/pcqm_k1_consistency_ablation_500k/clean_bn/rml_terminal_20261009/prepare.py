"""Prepare retained-only custody inputs and test closure in a disposable copy.

No owner finalization entry point is exposed. Shared RML owns validation,
recovery, publication and replay. This adapter only translates accepted facts.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile

from molgap.evidence_pointers import load_json_object
from molgap.research_memory.compiler import compile_research_memory, frozen_differences
from molgap.research_memory.discovery import discover_records
from molgap.research_memory.finalize import verified_receipt
from molgap.research_memory.paths import repo_local_path, verify_bound_artifact
from molgap.research_memory.portability import _evidence_closure_paths
from molgap.research_memory.recovery import recover_trace
from molgap.research_memory.schemas import validate_cost_event, validate_role_event, validate_trace_manifest
from molgap.research_memory.terminal_wiring import build_default_trace_manifest, close_terminal_multi_arm
from molgap.research_memory.trace import atomic_write, file_digest, json_bytes, load_canonical_trace
from molgap.research_memory.validate import validate_repository_records
from molgap.v5_common import validate_v5_evidence_envelope


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
QUESTION = HERE.parent.parent
Q = QUESTION.relative_to(ROOT).as_posix()
P = HERE.relative_to(ROOT).as_posix()
ARMS = ("k1_pretrained_mean2", "k1_pretrained_consistency")
RESULTS = Q + "/clean_bn/results_attempt3_20261009"
INSPECTION = Q + "/submission_kaggle3_v1/terminal_inspection_20261009"
ATTEMPTS = (
    ("kaggle1-consistency-ablation-001", Q + "/submission_v1/terminal_inspection"),
    ("kaggle1-consistency-ablation-002", Q + "/submission_v2/terminal_inspection_20261009"),
    ("kaggle3-consistency-ablation-001", INSPECTION),
)
CPU_REPORTS = (
    Q + "/clean_bn/results_20261009/pair_report.json",
    Q + "/clean_bn/results_attempt2_20261009/pair_report.json",
    RESULTS + "/pair_report.json",
)


def read(relative):
    return load_json_object(repo_local_path(ROOT, relative))


def write(relative, value):
    path = repo_local_path(ROOT, relative)
    path.relative_to(HERE)
    content = json_bytes(value)
    if path.exists() and path.read_bytes() != content and (HERE / "dry_run_report.json").exists():
        raise ValueError("Refusing changed preparation input: " + relative)
    atomic_write(path, content)


def measurement(value=None, status="measurement_missing"):
    return {"value": value, "status": status}


def copied_event(trajectory, event_id, observed_ref, *, attempt, category, hardware,
                 device=None, cpu=None, local_cpu=False):
    event = {
        "schema": "molgap-cost-event-v1", "cost_event_id": event_id,
        "trajectory_id": trajectory["trajectory_id"], "action_id": "A001",
        "run_id": trajectory["actions"][0]["run_ids"][0], "attempt_id": attempt,
        "platform": "local" if local_cpu else "kaggle", "hardware": hardware,
        "category": category, "evidence_ref": observed_ref,
        "measurement": {
            "device_hours": measurement(None, "not_applicable") if local_cpu else measurement(device, "measured"),
            "cpu_hours": measurement(cpu, "measured") if cpu is not None else measurement(),
            "wall_hours": measurement(),
            "queue_hours": measurement(None, "not_applicable") if local_cpu else measurement(),
        },
    }
    return validate_cost_event(event)


def copy_bound(source, destination):
    if not source.is_file():
        raise FileNotFoundError("Required authoritative file not retained: " + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if file_digest(source) != file_digest(destination):
        raise ValueError("Temporary copy SHA mismatch: " + str(source))


def temp_test(entries, packages, baseline):
    # Copy only canonical metadata, its direct closure, and required retained
    # bindings. No models are deserialized or data/graph caches opened.
    copies = _evidence_closure_paths(ROOT, baseline["records"])
    discovered = discover_records(ROOT)
    for paths in vars(discovered).values():
        copies.update(paths)
    for _, record in baseline["records"]["comparison_readiness"]:
        for field in ("candidate_artifact_bindings", "reference_artifact_bindings"):
            copies.update(repo_local_path(ROOT, binding["ref"])
                          for binding in (record.get(field) or {}).values())
    for directory in (ROOT / "experiments").glob("**/rml_finalized"):
        copies.update(path for path in directory.rglob("*") if path.is_file())
    for _, package in packages:
        copies.update(repo_local_path(ROOT, ref) for ref in package["artifact_hashes"])
    for entry in entries:
        copies.update(repo_local_path(ROOT, entry[key]) for key in ("trajectory", "terminal", "trace"))
    copied = {}
    with tempfile.TemporaryDirectory(prefix="molgap-actual500k-custody-") as directory:
        temp = Path(directory).resolve()
        def include(source):
            relative = source.relative_to(ROOT).as_posix()
            copy_bound(source, repo_local_path(temp, relative))
            copied[relative] = file_digest(source)
        for source in sorted(copies):
            include(source)
        runtime_root = Path(importlib.import_module("molgap.research_memory.schemas").__file__).resolve().parents[3]
        for folder in ("research_memory/schemas", "research_memory/policies"):
            source_root = runtime_root if folder.endswith("schemas") else ROOT
            for source in (source_root / folder).glob("*.json"):
                copy_bound(source, repo_local_path(temp, folder + "/" + source.name))

        # Let the existing validator identify additional hash-bound files, not
        # a second hand-written reference validator. Missing owner files stop.
        for _ in range(100):
            try:
                validate_repository_records(temp)
                compile_research_memory(temp)
                break
            except ValueError as exc:
                message = str(exc)
                marker = "required repository pointer is missing: "
                if marker not in message:
                    raise
                relative = message.split(marker, 1)[1]
                source = repo_local_path(ROOT, relative)
                if relative in copied:
                    raise
                include(source)
        else:
            raise ValueError("Temporary metadata closure did not converge")

        for entry in entries:
            canonical = load_canonical_trace(repo_local_path(temp, entry["trace"]))
            package = load_json_object(repo_local_path(temp, entry["terminal"]))
            raw = next(item["locator"] for item in package["evidence"]["artifacts"] if item["name"] == "training_trace")
            recovered = recover_trace([repo_local_path(temp, raw)], {
                "trajectory_id": canonical["trajectory_id"], "run_id": canonical["run_id"],
                "rows_key": "epochs", "field_mapping": {
                    "epoch_or_pass": "epoch", "optimizer_step": "global_step",
                    "sample_presentations": "sample_presentations", "learning_rate": "lr",
                    "live_train_metric": "train_mae_eV", "live_dev_metric": "live_development_mae_eV",
                    "ema_dev_metric": "ema_development_mae_eV", "wall_time_seconds": "seconds",
                }, "metric_semantics": canonical["metric_semantics"],
                "device_time_semantics": canonical["device_time_semantics"],
            }, temp / "research_memory/.staging" / (canonical["trajectory_id"] + ".recovery-check.json"), repo_root=temp)
            assert recovered["observations"] == canonical["observations"]
        results = close_terminal_multi_arm(temp, entries)
        validated = validate_repository_records(temp)
        outputs = {name: json.loads(value) for name, value in compile_research_memory(temp).items()
                   if name.endswith(".json")}
        target_ids = {entry["trajectory_id"] for entry in report_entries(entries)}
        assert not target_ids.intersection(item["trajectory_id"] for item in outputs["replay_pool.json"]["entries"])
        assert not target_ids.intersection(item["trajectory_id"] for item in outputs["replay_pool.json"]["action_entries"])
        assert not target_ids.intersection(outputs["screening_backtest.json"]["included_trajectory_ids"])
        assert not target_ids.intersection(item["trajectory_id"] for item in outputs["ready_for_desktop_index.json"]["valid"])
        assert frozen_differences(temp) == []
        receipts = []
        for entry, result in zip(entries, results):
            assert result["pipeline_status"] == "COMPLETE" and result["validation_status"] == "VALID"
            finalized = repo_local_path(temp, Path(entry["trajectory"]).parent / "rml_finalized")
            receipt = verified_receipt(finalized)
            manifest = load_json_object(finalized / "trace_manifest.json")
            assert receipt["trace_status"] == "available"
            assert manifest["reference_id"] is None and manifest["backtest_eligibility"]["eligible"] is False
            assert (finalized / "trace.json").read_bytes() == repo_local_path(ROOT, entry["trace"]).read_bytes()
            assert (finalized / "prospective_snapshot.json").read_bytes() == repo_local_path(ROOT, entry["trajectory"]).read_bytes()
            assert load_json_object(finalized / "trajectory.json")["state_at_start"]["reference_ids"] == []
            receipts.append({"trajectory_id": receipt["trajectory_id"], "trace_status": receipt["trace_status"],
                             "published_hashes": receipt["published_hashes"], "pipeline": result})
        retry = close_terminal_multi_arm(temp, entries)
        assert all(result["finalization_status"] == "ALREADY_FINALIZED" for result in retry)
        # The compiler's discovered events must count each training allocation
        # once, including replacement of V1 events rather than duplication.
        costs = [event for _, event in validated["records"]["costs"]
                 if event["trajectory_id"] in target_ids and event["category"] == "training"
                 and event["measurement"]["device_hours"]["status"] == "measured"]
        device_hours = sum(event["measurement"]["device_hours"]["value"] for event in costs)
        assert abs(device_hours - read(INSPECTION + "/inspection_report.json")["budget"]["cumulative_training_T4_device_hours"]) < 1e-9
        return {"status": "PASSED_TEMPORARY_COPY_ONLY", "official_finalization_executed": False,
                "published_owner_receipts": [], "temporary_receipt_checks": receipts,
                "training_T4_device_hours": device_hours, "measured_training_event_count": len(costs),
                "validation": "VALID", "derived_frozen": True, "idempotent_retry": True,
                "shared_recovery_reproduced_60_observations": True,
                "strict_grouping_excluded": True, "replay_excluded": True, "ready_excluded": True,
                "copied_sha256": copied}


def report_entries(entries):
    return [{"trajectory_id": read(entry["trajectory"])["trajectory_id"]} for entry in entries]


def compact_report(report, entries):
    """Summarize the completed test, never persist corpus or policy outputs."""
    digest = lambda value: hashlib.sha256(json_bytes(value)).hexdigest()
    checks = report["temporary_receipt_checks"]
    targets = {item["trajectory_id"] for item in checks}
    arms = {}
    for entry, check in zip(entries, checks):
        package = read(entry["terminal"])
        trace = load_canonical_trace(ROOT / entry["trace"])
        manifest = package["trace_manifest"]
        delta = check["pipeline"]["replay_pool_delta"]
        arms[entry["arm_identifier"]] = {
            "status": check["pipeline"]["pipeline_status"],
            "validation": check["pipeline"]["validation_status"],
            "trace_status": check["trace_status"],
            "observations": len(trace["observations"]),
            "optimizer_steps": manifest["exposure"]["optimizer_steps"],
            "sample_presentations": manifest["exposure"]["sample_presentations"],
            "reference_id": manifest["reference_id"],
            "replay_eligible": manifest["backtest_eligibility"]["eligible"],
            "terminal_input_sha256": file_digest(ROOT / entry["terminal"]),
            "bound_inventory_sha256": digest(package["artifact_hashes"]),
            "temporary_publication_inventory_sha256": digest(check["published_hashes"]),
            "replay_delta": {
                "corpus_added_count": len(delta["added"]),
                "corpus_removed_count": len(delta["removed"]),
                "target_added_count": sum(row[0] in targets for row in delta["added"]),
                "target_removed_count": sum(row[0] in targets for row in delta["removed"]),
            },
        }
    result = {
        "status": report["status"], "official_finalization_executed": False,
        "arms_input_ref": "arms.input.json", "arms_input_sha256": file_digest(HERE / "arms.input.json"),
        "arms": arms,
        "checks": {key: report[key] for key in (
            "validation", "derived_frozen", "idempotent_retry", "shared_recovery_reproduced_60_observations",
            "strict_grouping_excluded", "replay_excluded", "ready_excluded")},
        "replay_delta_scope": "Initial temporary rebuild adds existing corpus entries; target arms add/remove none",
        "inventories": {
            "copied_file_count": len(report["copied_sha256"]),
            "copied_inventory_sha256": digest(report["copied_sha256"]),
            "original_input_inventory_sha256": digest(report["original_input_sha256"]),
            "original_inputs_unchanged": all(file_digest(ROOT / ref) == sha for ref, sha in report["original_input_sha256"].items()),
            "runtime_inventory_sha256": digest(report["runtime_modules"]),
            "runtime_root": "D:/w/rml-unreferenced-trace/src",
        },
        "actual_cost_summary": {
            "training_T4_device_hours": report["training_T4_device_hours"],
            "measured_training_event_count": report["measured_training_event_count"],
            "cpu_worker_process_seconds_all_three_attempts": report["cpu_worker_process_seconds_all_three_attempts"],
            "pair_wall_seconds_all_three_cpu_attempts": report["pair_wall_seconds_all_three_cpu_attempts"],
            "scope": "Disjoint training allocations; CPU/wall separate; unknown queue, native/parent CPU and preparation cost remain unknown",
        },
    }
    assert result["inventories"]["original_inputs_unchanged"]
    assert len(json_bytes(result)) <= 4096
    return result


def main():
    baseline = validate_repository_records(ROOT)
    analysis = read(RESULTS + "/analysis_final.json")
    inputs = read(Q + "/clean_bn/inputs_attempt3.json")
    inspection = read(INSPECTION + "/inspection_report.json")
    assert analysis["mechanical_status"] == "passed"
    assert analysis["original_cpu_pair"]["original_material_nomination_gate_passed"] is False
    assert analysis["both_calibrated_pair"]["primary_promotion_gate_applicable"] is False
    # Reuse the owner's pure exposure check without importing its model/data
    # acceptance dependencies, which are intentionally absent from the RML tree.
    acceptance_source = (ROOT / Q / "clean_bn/accept.py").read_text(encoding="utf-8")
    exposure_function = next(node for node in ast.parse(acceptance_source).body
                             if isinstance(node, ast.FunctionDef) and node.name == "validate_exposure")
    exposure_namespace = {}
    exec(compile(ast.Module(body=[exposure_function], type_ignores=[]), "clean_bn/accept.py", "exec"), exposure_namespace)
    stamp_path = HERE / "preparation_identity.json"
    previous_terminal = HERE / ARMS[0] / "terminal.input.json"
    stamp = (load_json_object(stamp_path)["prepared_at"] if stamp_path.exists() else
             load_json_object(previous_terminal)["finalized_at"] if previous_terminal.exists() else
             datetime.now(timezone.utc).isoformat())
    write(P + "/preparation_identity.json", {"prepared_at": stamp, "official_finalization_executed": False})
    original_hashes = {}
    entries, packages = [], []
    for arm in ARMS:
        arm_dir = P + "/" + arm
        trajectory_ref = Q + "/kaggle1_v1/" + arm + "/trajectory.json"
        trajectory = read(trajectory_ref)
        verify_bound_artifact(ROOT, trajectory_ref, inputs["arms"][arm]["trajectory"]["sha256"])
        assert trajectory["state_at_start"]["reference_ids"] == []
        assert trajectory["decision_state"]["active_reference_ids"] == []
        assert trajectory["decision"]["outcome"] == "ACTIVE"
        assert not (ROOT / trajectory_ref).parent.joinpath("rml_finalized").exists()
        original_hashes[trajectory_ref] = file_digest(ROOT / trajectory_ref)
        run_id = trajectory["actions"][0]["run_ids"][0]
        stage = INSPECTION + "/evidence/stages/" + arm
        stage_manifest = read(stage + "/stage_manifest.json")
        recipe_ref = Q + "/training_recipe_" + arm + ".json"
        recipe = read(recipe_ref)
        original_hashes[recipe_ref] = file_digest(ROOT / recipe_ref)
        raw_ref = stage + "/trace.json"
        verify_bound_artifact(ROOT, raw_ref, stage_manifest["artifacts"]["trace.json"])
        retained_ref = RESULTS + "/rml_audit/" + arm + "/canonical_trace.json"
        canonical = load_canonical_trace(ROOT / retained_ref)
        exposure_namespace["validate_exposure"](canonical)
        assert canonical["trajectory_id"] == trajectory["trajectory_id"] and canonical["run_id"] == run_id
        assert any(item["sha256"] == file_digest(ROOT / raw_ref) for item in canonical["provenance"])
        assert all(row[key] is None for row in canonical["observations"] for key in (
            "ema_dev_metric", "checkpoint_identity", "device_time_seconds",
            "cumulative_device_time_seconds", "cumulative_wall_time_seconds"))
        original_hashes[raw_ref] = file_digest(ROOT / raw_ref)
        original_hashes[retained_ref] = file_digest(ROOT / retained_ref)
        observed_ref = arm_dir + "/observed_records.json"
        costs, roles, translations = [], [], []
        for index, (attempt, location) in enumerate(ATTEMPTS):
            receipt_ref = location + "/evidence/invocation_cost.json"
            receipt = read(receipt_ref)
            assert receipt["allocation_verified"] is True and receipt["hardware"] == "T4x2"
            windows = receipt["phase_windows"]
            seconds = {key: windows[key]["allocated_T4_seconds"] for key in ("training", "preflight")}
            seconds["bootstrap-overhead"] = receipt["allocated_T4_seconds"] - sum(seconds.values())
            for phase, allocated in seconds.items():
                assert allocated >= 0
                if index == 0:
                    old_ref = Q + "/kaggle1_v1/" + arm + "/costs/cost-" + trajectory["trajectory_id"] + "-measured-v1-" + phase + ".json"
                    event = read(old_ref)
                    assert abs(event["measurement"]["device_hours"]["value"] - allocated / 7200) < 1e-9
                    event["evidence_ref"] = observed_ref
                    translations.append({"event_id": event["cost_event_id"], "source_event_ref": old_ref,
                                         "source_event_sha256": file_digest(ROOT / old_ref), "receipt_ref": receipt_ref})
                    validate_cost_event(event)
                else:
                    event = copied_event(trajectory, "cost-" + trajectory["trajectory_id"] + "-" + attempt + "-" + phase,
                                         observed_ref, attempt=attempt, category=phase if phase in {"training", "preflight"} else "other",
                                         hardware="Tesla T4; one of two allocated devices; includes assigned idle capacity",
                                         device=allocated / 7200)
                    translations.append({"event_id": event["cost_event_id"], "receipt_ref": receipt_ref,
                                         "receipt_sha256": file_digest(ROOT / receipt_ref), "formula": "allocated_T4_seconds / 2 / 3600"})
                costs.append(event)
        for index, report_ref in enumerate(CPU_REPORTS):
            report = read(report_ref)
            if arm not in report["arms"]:
                # Both failed bootstraps stopped after the mean2 worker. Do not
                # duplicate its cost or invent a zero-cost consistency worker.
                continue
            cpu_seconds = report["arms"][arm]["process_cpu_seconds"]
            event = copied_event(trajectory, "cost-" + trajectory["trajectory_id"] + "-clean-bn-attempt" + str(index + 1),
                                 observed_ref, attempt="local-clean-bn-attempt" + str(index + 1),
                                 category="acceptance" if index == 2 else "infrastructure_failure",
                                 hardware="CPU worker process; parent CPU unmeasured", cpu=cpu_seconds / 3600, local_cpu=True)
            costs.append(event)
            translations.append({"event_id": event["cost_event_id"], "report_ref": report_ref,
                                 "report_sha256": file_digest(ROOT / report_ref), "worker_process_cpu_seconds": cpu_seconds})
        data_hash = file_digest(ROOT / stage / "data_manifest.json")
        for access, role, selected, authority in (
            ("training_membership", "train", False, INSPECTION + "/raw_acceptance.md"),
            ("metric_computed", "internal_development", True, INSPECTION + "/raw_acceptance.md"),
            ("selection_used", "internal_development", True, stage + "/stage_manifest.json"),
            ("labels_read", "train", False, RESULTS + "/analysis_final.json"),
            ("metric_computed", "internal_development", False, RESULTS + "/analysis_final.json"),
        ):
            event = {"schema": "molgap-role-event-v1", "role_event_id": "role-" + trajectory["trajectory_id"] + "-" + role + "-" + access,
                     "trajectory_id": trajectory["trajectory_id"], "action_id": "A001", "run_id": run_id,
                     "dataset_identity": recipe["data_role_fingerprint"], "row_manifest_hash": data_hash,
                     "role_name": role, "access_kind": access, "selection_used": selected, "evidence_ref": observed_ref}
            # Distinguish training selection from the secondary saved diagnostic.
            if authority.endswith("analysis_final.json"):
                event["role_event_id"] += "-clean-bn"
            roles.append(validate_role_event(event))
            translations.append({"event_id": event["role_event_id"], "authority_ref": authority,
                                 "authority_sha256": file_digest(ROOT / authority),
                                 "scope": "observed membership/selection/diagnostic only; no new data access"})
        decision = {"outcome": "NEGATIVE_UNDER_CONTRACT", "final": True,
                    "decision_ref": Q + "/clean_bn/DECISION.md", "next_allowed_actions": [], "reopen_conditions": []}
        outcome = {"execution_status": "complete", "artifact_status": "durable_local_verified",
                   "comparison_status": "context_only", "scientific_status": "NEGATIVE_UNDER_CONTRACT",
                   "transfer_status": "stop", "budget_decision": "closed", "full_handoff_status": "not_applicable"}
        role_use = {"official_validation": "untouched", "test_dev": "untouched", "test_challenge": "untouched",
                    "train": "consumed", "internal_development": "consumed"}
        observed = {"format": "molgap-rml-retained-observation-translation-v1", "costs": costs, "roles": roles,
                    "translations": translations, "training_inspection_ref": INSPECTION + "/inspection_report.json",
                    "clean_bn_acceptance_ref": RESULTS + "/analysis_final.json",
                    "limits": ["No strict reference was frozen", "Physical jobs differ; strict replay unqualified",
                               "CPU/wall/T4 units are separate", "Pair wall windows overlap across arms; no additive per-arm wall inferred",
                               "Queue and native CPU costs unknown", "Parent CPU and preparation/testing/analysis cost unknown"]}
        write(observed_ref, observed)
        acceptance_ref = arm_dir + "/acceptance.json"
        write(acceptance_ref, {"format": "molgap-rml-retained-acceptance-binding-v1", "evidence_id": "pcqm-k1-consistency-500k-" + arm + "-s42-v1-terminal",
                              "run_id": run_id, "outcome": outcome, "trajectory_decision": decision, "role_use": role_use,
                              "mechanical_acceptance_ref": INSPECTION + "/inspection_report.json",
                              "scientific_decision_ref": decision["decision_ref"], "secondary_acceptance_ref": RESULTS + "/analysis_final.json",
                              "strict_comparison_qualified": False, "replay_ready": False, "ready_for_desktop": False})
        refs = {raw_ref, retained_ref, observed_ref, acceptance_ref, trajectory_ref, recipe_ref,
                Q + "/clean_bn/ATTRIBUTION.md", Q + "/clean_bn/RML_BLOCKERS.md", decision["decision_ref"],
                RESULTS + "/analysis_final.json", Q + "/clean_bn/inputs_attempt3.json", INSPECTION + "/inspection_report.json",
                INSPECTION + "/raw_acceptance.md", INSPECTION + "/retrieval_manifest.json", INSPECTION + "/scheduler_snapshot.json"}
        refs.update(ref for item in translations for key, ref in item.items() if key.endswith("_ref"))
        refs.update(trajectory["state_at_start"]["contract_refs"])
        for _, location in ATTEMPTS:
            refs.add(location + "/evidence/invocation_cost.json")
        refs.update(CPU_REPORTS)
        for submission in ("submission_v1", "submission_v2", "submission_kaggle3_v1"):
            refs.add(Q + "/" + submission + "/package_manifest.json")
        for filename in ("best_model.pt", "best_predictions.pt", "last_checkpoint.pt", "scientific_contract.json",
                         "runtime_certificate.json", "runtime.json", "data_manifest.json", "calibration.json"):
            relative = stage + "/" + filename
            verify_bound_artifact(ROOT, relative, stage_manifest["artifacts"][filename])
            refs.add(relative)
        refs.add(stage + "/stage_manifest.json")
        refs.update(analysis["artifact_hashes"])
        for relative, expected in analysis["artifact_hashes"].items():
            verify_bound_artifact(ROOT, relative, expected)
        bindings = {relative: file_digest(repo_local_path(ROOT, relative)) for relative in sorted(refs)}
        artifacts = [{"name": "training_trace" if relative == raw_ref else relative.rsplit("/", 1)[-1],
                      "locator": relative, "sha256": digest, "availability": "durable_local_verified"}
                     for relative, digest in bindings.items() if relative != retained_ref]
        evidence = {"format": "molgap-v5-evidence-envelope-v1", "contract": "MOLGAP-COMMON-V5-FINAL",
                    "evidence_id": "pcqm-k1-consistency-500k-" + arm + "-s42-v1-terminal", "track": "B", "scope": "500k",
                    "legacy_contract": recipe["benchmark_id"], "outcome": outcome, "role_use": role_use,
                    "authority": {"pointers": [decision["decision_ref"], Q + "/clean_bn/ATTRIBUTION.md", INSPECTION + "/inspection_report.json", RESULTS + "/analysis_final.json"]},
                    "artifacts": artifacts, "migration": {"migrated_at": stamp,
                    "verification_scope": "Retained accepted results translated to context-only terminal custody; no recomputation or runtime qualification",
                    "training_executed": False, "inference_executed": False, "scientific_reinterpretation": False}}
        validate_v5_evidence_envelope(evidence, repo_root=ROOT)
        manifest = build_default_trace_manifest(ROOT, trajectory, {"run_id": run_id}, canonical)
        manifest.update(trace_artifact_ref=retained_ref, trace_artifact_sha256=file_digest(ROOT / retained_ref),
                        terminal_evidence_ref=arm_dir + "/evidence.input.json",
                        metric_semantics="Development Gap MAE/eV; train scaled objective/eV-equivalent, not clean inference MAE",
                        evaluation_role_identity="consumed-internal-development50K",
                        selection_semantics=recipe["selection_fingerprint"], model_identity="frozen-recipe-sha256:" + bindings[recipe_ref])
        manifest["comparability_identity"].update(
            scientific_contract=recipe["benchmark_id"], dataset_identity=recipe["data_role_fingerprint"],
            row_split_identity=recipe["row_order_fingerprint"], architecture_identity="retained-K1-3658817-parameters",
            optimizer_identity=recipe["optimizer_fingerprint"], lr_schedule_identity=recipe["schedule_fingerprint"],
            target_transform_identity=stage_manifest["contract"]["target_transform_fingerprint"], precision_identity=recipe["precision"],
            evaluation_role_identity="consumed-internal-development50K", selection_role_identity=recipe["selection_fingerprint"])
        manifest["backtest_eligibility"]["exclusion_reasons"].extend([
            "multiple_physical_jobs_unqualified_for_strict_replay", "unobserved_checkpoint_identities",
            "missing_per_observation_cumulative_native_cost", "one_seed_consumed_selection_role_not_independent_transfer"])
        validate_trace_manifest(manifest)
        write(arm_dir + "/evidence.input.json", evidence)
        write(arm_dir + "/trace_manifest.input.json", manifest)
        terminal = {"format": "molgap-rml-terminal-package-v1", "trajectory_id": trajectory["trajectory_id"],
                    "run_id": run_id, "action_id": "A001", "finalized_at": stamp, "acceptance_ref": acceptance_ref,
                    "artifact_hashes": bindings, "evidence": evidence, "decision": decision,
                    "costs": costs, "roles": roles, "role_use": role_use, "trace_manifest": manifest}
        terminal_ref = arm_dir + "/terminal.input.json"
        write(terminal_ref, terminal)
        entries.append({"trajectory": trajectory_ref, "terminal": terminal_ref, "trace": retained_ref, "arm_identifier": arm})
        packages.append((arm, terminal))
    write(P + "/preparation_identity.json", {"prepared_at": stamp, "official_finalization_executed": False})
    write(P + "/arms.input.json", entries)
    report = temp_test(entries, packages, baseline)
    assert all(file_digest(ROOT / relative) == digest for relative, digest in original_hashes.items())
    report["original_input_sha256"] = original_hashes
    report["cpu_worker_process_seconds_all_three_attempts"] = sum(
        event["measurement"]["cpu_hours"]["value"] * 3600 for _, package in packages for event in package["costs"]
        if event["platform"] == "local")
    assert abs(report["cpu_worker_process_seconds_all_three_attempts"] - analysis["cost_observation"]["all_diagnostic_worker_process_cpu_seconds"]) < 1e-9
    report["pair_wall_seconds_all_three_cpu_attempts"] = analysis["cost_observation"]["all_diagnostic_attempt_wall_seconds"]
    report["runtime_modules"] = {name: {"path": str(importlib.import_module(name).__file__),
                                       "sha256": file_digest(importlib.import_module(name).__file__)} for name in (
        "molgap.research_memory.terminal_wiring", "molgap.research_memory.finalize", "molgap.research_memory.schemas",
        "molgap.research_memory.validate", "molgap.research_memory.backtest", "molgap.research_memory.replay")}
    report = compact_report(report, entries)
    write(P + "/dry_run_report.json", report)
    print(json.dumps({"status": report["status"], "official_finalization_executed": False,
                      "report_bytes": len(json_bytes(report)), "checks": report["checks"]}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        atomic_write(HERE / "BLOCKER.json", json_bytes({"status": "BLOCKED", "error_type": type(exc).__name__,
                     "error": str(exc), "official_finalization_executed": False,
                     "instruction": "Do not finalize owner, invent evidence, fetch remotely or rerun inference. Parent review required."}))
        raise
