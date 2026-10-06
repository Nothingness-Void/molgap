"""Retained-tensor acceptance and NO_TRAIN finalization for the frozen audit."""
from datetime import datetime, timezone
import json
from pathlib import Path

from .gptrans_portability import verify_file, check_reproduction, check_rows
from .gptrans_bottleneck import portability_indices
from .training_reproducibility import atomic_json, sha256_file

BASE = "experiments/pcqm_gptrans_bottleneck_audit"


def accept_audit(output, inputs):
    import numpy as np
    import torch
    from .k1_terminal_analysis import paired_saved_errors
    manifest = json.loads((output/"output_manifest.json").read_text())
    release = json.loads((inputs/"audit_release.json").read_text())
    if (manifest["release"] != release or manifest["status"] != "complete"
            or manifest["training_executed"] is not False or manifest["optimizer_steps"] != 0
            or any(manifest[k] is not False for k in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"))):
        raise ValueError("Frozen audit execution/release scope differs")
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file() and p.name != "output_manifest.json"}
    if actual != set(manifest["files"]):
        raise ValueError("Frozen diagnostic output inventory differs")
    for name,digest in manifest["files"].items():
        path = (output/name).resolve()
        if not path.is_relative_to(output.resolve()):
            raise ValueError("Unsafe diagnostic output pointer")
        verify_file(path, digest)
    for name,digest in release["files"].items():
        verify_file(inputs/name, digest)
    ledger = json.loads((output/"allocation_ledger.json").read_text())
    if (ledger["status"] != "complete" or ledger["spec_identity"] != release["identity"]
            or ledger["allocation_device_count"] != 2 or not 0 < ledger["wall_seconds"] <= 1800
            or ledger["allocated_device_seconds"] != 2*ledger["wall_seconds"]):
        raise ValueError("Frozen audit allocation/cost differs")
    for task in ("mechanism", "portability"):
        terminal = json.loads((output/task/"terminal.json").read_text())
        runtime = json.loads((output/task/"runtime.json").read_text())
        if (terminal["complete"] is not True or terminal["training_executed"] is not False
                or terminal["optimizer_steps"] != 0 or terminal["unchanged_state"] is not True
                or any(terminal[k] is not False for k in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"))
                or runtime["accelerator"]["device_count_visible"] != 1 or "T4" not in runtime["accelerator"]["name"]
                or runtime["determinism"]["precision"] != "fp32" or runtime["determinism"]["tf32_enabled"] is not False):
            raise ValueError("Frozen diagnostic worker evidence differs")
    events = json.loads((output/"portability/role_events.json").read_text())
    if [(r["role"],r["event"]) for r in events] != [(role,event) for role in ("original_100k","unseen_500k")
            for event in ("prediction_input_and_labels_read", "metric_computed")]:
        raise ValueError("Observed portability role history differs")
    def joined(role):
        directory = output/"portability"/role
        chunks = json.loads((directory/"progress.json").read_text())["chunks"]
        if len(chunks) != (10 if role == "original_100k" else 2):
            raise ValueError("Missing prediction chunks")
        parts = []
        for index,item in enumerate(chunks):
            if item["file"] != f"chunk_{index:02d}.pt" or item["rows"] != 5000:
                raise ValueError("Prediction chunk scope differs")
            verify_file(directory/item["file"], item["sha256"])
            parts.append(torch.load(directory/item["file"], map_location="cpu", weights_only=False))
        payload = {k:torch.cat([part[k] for part in parts]) for k in ("source_idx","target_eV","prediction_eV")}
        if role == "original_100k":
            check_rows(payload, 100000, 50000)
        elif not torch.equal(payload["source_idx"], torch.tensor(portability_indices()+500000)):
            raise ValueError("Later inference cohort differs")
        return payload
    original, later = joined("original_100k"), joined("unseen_500k")
    reproduction = check_reproduction(original, torch.load(inputs/"local_predictions.pt", map_location="cpu", weights_only=False))
    comparisons = {"original_100k":paired_saved_errors(torch.load(inputs/"reference_predictions.pt", map_location="cpu", weights_only=False),original),
        "unseen_500k":paired_saved_errors(torch.load(inputs/"reference_later.pt", map_location="cpu", weights_only=False),later)}
    contract = json.loads((inputs/"contract.json").read_text())
    chunks = json.loads((output/"mechanism/terminal.json").read_text())["chunks"]
    if len(chunks) != 20:
        raise ValueError("Incomplete checkpoint-panel derivative chunks")
    by_asset = {}
    for item in chunks:
        part = torch.load(output/"mechanism"/item["file"], map_location="cpu", weights_only=False)
        asset = part["model_asset"]
        if (part["source_checkpoint_sha256"] != contract["model_assets"][asset]["source_checkpoint_sha256"]
                or part["state_sha256"] != contract["model_assets"][asset]["state_sha256"] or len(part["layers"]) != 12):
            raise ValueError("Derivative checkpoint/state identity differs")
        by_asset.setdefault(asset,[]).append(part)
    summaries = {}
    for asset,parts in by_asset.items():
        ids = torch.cat([p["source_idx"] for p in parts])
        if not torch.equal(ids,torch.arange(100000,100512)):
            raise ValueError("Derivative panel differs")
        degree = torch.cat([p["max_degree"] for p in parts]).numpy()
        layers = []
        for index in range(12):
            rows = [p["layers"][index] for p in parts]
            fields = {key:torch.cat([row[key] for row in rows]).numpy() for key in
                ("input_rms","output_rms","update_rms","pair_input_rms","pair_update_rms")}
            if any(not np.isfinite(v).all() for v in fields.values()):
                raise ValueError("Nonfinite derivative activation metrics")
            ratio = fields["update_rms"] / np.maximum(fields["input_rms"],1e-12)
            pair_ratio = fields["pair_update_rms"] / np.maximum(fields["pair_input_rms"],1e-12)
            layers.append(dict(block=index+1, mean_input_rms=float(fields["input_rms"].mean()),
                mean_output_rms=float(fields["output_rms"].mean()), mean_update_ratio=float(ratio.mean()),
                update_ratio_p90=float(np.quantile(ratio,.9)), mean_pair_update_ratio=float(pair_ratio.mean()),
                degree_groups={label:dict(rows=int(mask.sum()),mean_update_ratio=float(ratio[mask].mean()) if mask.any() else None)
                    for label,mask in (("maxdegree_le2",degree<=2),("maxdegree_eq3",degree==3),("maxdegree_ge4",degree>=4))},
                connected_batches=sum(row["loss_gradient_connected"] for row in rows),
                branch_parameter_gradient_l2=[row["branch_parameter_gradient_l2"] for row in rows]))
        summaries[asset] = layers
    return dict(accepted=True, training_executed=False, local_model_inference_executed=False,
        comparison_class="PAIRED_ENDPOINT", strict_ready=False, training_replay_ready=False,
        mechanism_comparison_class="CONTEXT_ONLY", reproduction=reproduction, comparisons=comparisons,
        layer_summaries=summaries, allocation=ledger, release_identity=release["identity"],
        caveats=["eval EMA gradients are not historical training gradients", "reused internal development, not sealed confirmation",
                 "frozen portability is not500K optimization; amplitude correlation is not causal proof"])


def close_audit(root, output, inputs):
    from .k1_relation_audit_records import prepare_no_train_terminal
    from .research_memory.finalize import finalize
    frozen = json.loads((root/BASE/"rml_plan/trajectory.json").read_text())
    if (root/BASE/"rml_plan/rml_finalized").exists():
        return finalize(root,root/BASE/"rml_plan/trajectory.json",root/BASE/"results/terminal.json")
    accepted = accept_audit(output,inputs)
    if not (root/BASE/"decision_terminal.md").is_file():
        raise ValueError("Controller must write a dated diagnostic interpretation before terminal closure")
    # The shared terminal owner expects this exact decision pointer.
    decision = root/BASE/"decision.md"
    if "decision_terminal.md" not in decision.read_text():
        raise ValueError("Decision entry must route to the controller interpretation")
    results = root/BASE/"results"
    atomic_json(results/"acceptance_summary.json",accepted)
    ledger = accepted["allocation"]
    atomic_json(results/"execution.json",dict(allocation=ledger, training_executed=False, model_inference_executed=False))
    manifest = json.loads((output/"output_manifest.json").read_text())
    role_manifests = {"original_100k":dict(rows=50000,source_idx_start=100000,source_idx_stop=150000),
        "unseen_500k":dict(rows=10000,source_idx_start=500000,source_idx_stop=550000,
            row_selection="frozen-RandomState42-without-replacement-sorted", reference_payload_sha256=sha256_file(inputs/"reference_later.pt"))}
    atomic_json(results/"role_row_manifests.json",role_manifests)
    refs = [str((output/name).relative_to(root)).replace("\\","/") for name in ("output_manifest.json",*manifest["files"]) if name.endswith(".json")]
    refs += [BASE+"/results/"+name+".json" for name in ("acceptance_summary","execution","role_row_manifests","role_history","cost_records")]
    missing = dict(status="measurement_missing",value=None)
    scope = dict(execution_status="complete",artifact_status="accepted",comparison_status="paired_endpoint_diagnostic",
        scientific_status="no_train_bottleneck_observations",transfer_status="frozen_cohort_diagnostic_only",
        budget_decision="within_frozen_cap",full_handoff_status="not_authorized")
    paths = prepare_no_train_terminal(prefix=BASE,frozen=frozen,run_id=frozen["actions"][0]["run_ids"][0],
        evidence_id="pcqm-gptrans-bottleneck-frozen-v1",outcome=scope,scope="eval_derivatives_and_frozen_cohort_NO_TRAIN",
        finalized_at=datetime.now(timezone.utc).isoformat(),artifact_refs=refs,
        authority=[BASE+"/"+name for name in ("protocol.md","contract.json","role_plan.json","budget.json","release_binding.json","submission_receipt.json","decision.md","decision_terminal.md")],
        acceptance_name="acceptance_summary",repo_root=root,cost_measurement=dict(
            device_hours=dict(status="measured",value=ledger["allocated_device_seconds"]/3600),
            wall_hours=dict(status="measured",value=ledger["wall_seconds"]/3600),cpu_hours=missing,queue_hours=missing),
        cost_semantics=ledger["scope"])
    return finalize(root,root/paths["trajectory"],root/paths["terminal"])
