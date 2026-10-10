"""Saved kernel attribution and NO_TRAIN custody through the existing finalizer."""
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import shutil
import struct

from molgap.research_memory.finalize import finalize
from molgap.training_reproducibility import atomic_json, sha256_file
from kernel_check import ROOT, ATTEMPT, REL, TID, RUN, read


def summarize(records):
    groups = defaultdict(lambda: dict(calls=0, summed_kernel_ns=0))
    names = defaultdict(lambda: dict(calls=0, summed_kernel_ns=0))
    intervals = []
    for row in records:
        name, start, end = row["name"], row["start"], row["end"]
        if not name or type(start) is not int or type(end) is not int or not 0 < start <= end < 2**64-1:
            raise ValueError("Invalid retained kernel record")
        intervals.append((start, end))
        categories = ["all"] + [key for key in ("gemm", "tensorop", "simt", "indexing_backward", "radixsort", "elementwise") if key in name.lower()]
        for key in categories:
            groups[key]["calls"] += 1
            groups[key]["summed_kernel_ns"] += end-start
        names[name]["calls"] += 1
        names[name]["summed_kernel_ns"] += end-start
    if not intervals:
        raise ValueError("Cannot summarize unavailable trace as zero GPU cost")
    intervals.sort()
    low, high = intervals[0]
    union = 0
    for start, end in intervals[1:]:
        if start <= high:
            high = max(high, end)
        else:
            union += high-low
            low, high = start, end
    union += high-low
    return dict(kernel_records=len(records), unique_names=len(names), groups=dict(groups),
        traced_kernel_union_ns=union, trace_span_ns=max(e for _, e in intervals)-intervals[0][0],
        device_ids=sorted({r["deviceId"] for r in records}), stream_ids=sorted({r["streamId"] for r in records}),
        top_kernels=[dict(name=n, **v) for n, v in sorted(names.items(), key=lambda x:x[1]["summed_kernel_ns"], reverse=True)[:20]],
        limits=["Two instrumented owner steps, not uninstrumented throughput or long-run GPU utilization",
                "Name groups overlap; sums are not category shares or hardware instruction counts"])


def close():
    inputs, result, completion = (read(ATTEMPT / n) for n in ("inputs.json", "results/result.json", "results/completion.json"))
    if result["status"] != "complete_diagnostic" or completion["complete"] is not True:
        raise ValueError("Incomplete diagnostic")
    if result["inputs_sha256"] != sha256_file(ATTEMPT / "inputs.json") or result["prospective_sha256"] != sha256_file(ATTEMPT / "rml/trajectory.json"):
        raise ValueError("Identity changed")
    for name, digest in completion["files"].items():
        if sha256_file(ATTEMPT / "results" / name) != digest:
            raise ValueError("Output changed")
    cases = [p+"_"+f for p in ("bundled", "isolated") for f in ("fp32", "tf32")]
    if result["cases"] != cases:
        raise ValueError("Case inventory differs")
    rows, summaries = [], {}
    for case in cases:
        row = read(ATTEMPT / "results" / (case+".json"))
        if (row["inputs_sha256"] != result["inputs_sha256"] or row["prospective_sha256"] != result["prospective_sha256"]
                or row["source_commit"] != inputs["source_commit"] or row["case"] != case
                or row["tf32_allowed"] != case.endswith("tf32") or row["warmup_steps"] != 3
                or row["train_decoded_source_idx"] != inputs["source_idx"]
                or row["train_executed_source_idx"] != sorted(inputs["train_executed_source_idx"])
                or row["idle_observation"]["idle"] is not True):
            raise ValueError("Case binding/observed scope differs")
        if row["status"] == "trace_available":
            if row["phase"] != "complete" or row["trace_steps"] != 2 or row["trace"]["dropped_records"] != 0:
                raise ValueError("Traced work is incomplete")
            summaries[case] = summarize(row["trace"]["records"])
        else:
            summaries[case] = dict(status="unavailable", error=row.get("error"), diagnostics=row.get("partial_trace"),
                                   gpu_time=None)
        rows.append(row)
    if not math.isfinite(result["worker_wall_seconds"]) or not 0 <= result["worker_wall_seconds"] <= inputs["worker_wall_limit_seconds"]:
        raise ValueError("Wall budget exceeded")
    if not math.isfinite(result["process_cpu_seconds"]) or result["process_cpu_seconds"] != sum(r["process_cpu_seconds"] for r in rows):
        raise ValueError("Child CPU cost differs")
    atomic_json(ATTEMPT / "kernel_summary.json", dict(cases=summaries, closure_source_sha256=sha256_file(Path(__file__))))
    for name, digest in inputs["files"].items():
        source, snapshot = ROOT / name, ATTEMPT / "executed_source" / name
        if snapshot.exists():
            if sha256_file(snapshot) != digest:
                raise ValueError("Source snapshot changed")
        else:
            if sha256_file(source) != digest:
                raise ValueError("Executed source changed")
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, snapshot)
    for name in ("terminal_decision.md", "attribution.md"):
        if not (ATTEMPT / name).is_file():
            raise ValueError("Parent interpretation required")
    eid = "pcqm-k1-kernel-trace-20261010"
    decision = dict(outcome="NO_TRAIN", decision_ref=f"{REL}/terminal_decision.md", next_allowed_actions=[],
        reopen_conditions=["Separately authorized native/precision training-quality qualification"])
    outcome = dict(execution_status="complete_kernel_diagnostic", artifact_status="local_hash_verified", comparison_status="instrumented_scratch_only",
        scientific_status="NO_TRAIN", transfer_status="not_evaluated", budget_decision="bounded_diagnostic_complete", full_handoff_status="not_applicable")
    cost = deepcopy(read(ATTEMPT / "plan_input.json")["costs"][0])
    cost.update(cost_event_id="cost-k1-kernel-observed", category="other", evidence_ref=f"{REL}/acceptance.json")
    cost["measurement"] = dict(wall_hours=dict(value=result["worker_wall_seconds"]/3600, status="measured"),
        cpu_hours=dict(value=result["process_cpu_seconds"]/3600, status="measured"),
        device_hours=dict(value=None, status="measurement_missing"), queue_hours=dict(value=None, status="not_applicable"))
    roles = []
    for role, ids, accesses in (("train_decoded", inputs["source_idx"], ("labels_read",)),
        ("train_executed", sorted(inputs["train_executed_source_idx"]), ("prediction_input", "metric_computed"))):
        digest = hashlib.sha256(b"".join(struct.pack("<q", i) for i in ids)).hexdigest()
        for access in accesses:
            roles.append(dict(schema="molgap-role-event-v1", role_event_id=f"role-k1-kernel-{role}-{access}", trajectory_id=TID, action_id="A001", run_id=RUN,
                dataset_identity="pcqm4mv2-ogb-fixed-500k-scnet-v1", role_name=role, row_manifest_hash=digest, access_kind=access,
                selection_used=False, evidence_ref=f"{REL}/acceptance.json"))
    role_use = dict(train_decoded="labels_read", train_executed="prediction_input", development="untouched", official_validation="untouched",
                    test_dev="untouched", test_challenge="untouched")
    atomic_json(ATTEMPT / "acceptance.json", dict(evidence_id=eid, run_id=RUN, outcome=outcome, trajectory_decision=decision,
        role_use=role_use, costs=[cost], roles=roles, training_replay_ready=False, generalization_accuracy_qualified=False,
        cost_scope="Parent execute wall and sum of measured child CPU; excludes setup/tests/closure, parent CPU and GPU busy time unavailable",
        closure_source_sha256=sha256_file(Path(__file__))))
    files = [p for p in ATTEMPT.rglob("*") if p.is_file() and "rml" not in p.relative_to(ATTEMPT).parts
        and p.name not in ("terminal.json", "closure_receipt.json")]
    hashes = {p.relative_to(ROOT).as_posix():sha256_file(p) for p in files}
    evidence = dict(format="molgap-v5-evidence-envelope-v1", contract="MOLGAP-COMMON-V5-FINAL", evidence_id=eid, track="B", scope="k1_kernel_attribution",
        legacy_contract="pcqm-k1-kernel-trace-v1", outcome=outcome, role_use=role_use,
        authority=dict(pointers=[f"{REL}/{n}" for n in ("README.md", "terminal_decision.md", "attribution.md", "acceptance.json")]),
        migration=dict(migrated_at=datetime.now(timezone.utc).date().isoformat(), training_executed=False, inference_executed=False,
            scientific_reinterpretation=False, verification_scope="Metadata closure of separately executed prospective kernel diagnostic"),
        observed_execution=dict(training_executed=False, inference_executed=True, training_replay_ready=False, execution_ref=f"{REL}/results/result.json"),
        artifacts=[dict(name=p.removeprefix(REL+"/"), locator=p, sha256=h, availability="locally_retained_hash_verified") for p,h in hashes.items()])
    atomic_json(ATTEMPT / "terminal.json", dict(format="molgap-rml-terminal-package-v1", trajectory_id=TID, run_id=RUN, action_id="A001",
        finalized_at=datetime.now(timezone.utc).isoformat(), acceptance_ref=f"{REL}/acceptance.json", artifact_hashes=hashes,
        evidence=evidence, decision=decision, costs=[cost], roles=roles))
    atomic_json(ATTEMPT / "closure_receipt.json", finalize(ROOT, f"{REL}/rml", f"{REL}/terminal.json"))
    print(eid + " finalized NO_TRAIN")


if __name__ == "__main__":
    close()
