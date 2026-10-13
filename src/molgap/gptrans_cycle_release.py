"""Freeze the authorized cycle pair using shared recipes, Spec and RML owners."""
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess

from .experiment_spec import ExperimentSpec, FAMILIES, TERMINAL_PROTOCOL
from .family_training_recipe import FamilyRecipe, ScheduleSpec
from .family_training_data import DatasetSelection
from .family_training_runner import build_screen_recipe, family_arm_bindings
from .experiment_family_workflow import tensor_digest
from .gptrans_cycle_pair import PARENT, CYCLE, PARENT_STATE_SHA256, freeze_initial
from .gptrans_cycle_inputs import FIXED_100K_SHA
from .screen_policy import canonical_fingerprint
from .training_reproducibility import atomic_json, sha256_file
from .v4_runtime import normalized_source_sha256

BASE = "experiments/pcqm_gptrans_cycle_pair_100k"
RUN = "kaseichou/molgap-gptrans-cycle-dual-100k-v1"
CONTEXT = "pcqm-gptrans-g1-degree-bond-local-ema999-100k-s42"
CACHE = "pcqm-gptrans-common-cycle-inputs-accepted-v1"


def freeze_pair(root):
    import torch
    root = Path(root).resolve()
    gpu = root / BASE / "gpu"
    if (gpu / "experiment_spec.json").exists():
        raise ValueError("Pair already frozen; reconcile rather than generating another plan")
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    cache = json.loads((root / BASE / "cpu/acceptance.json").read_bytes())
    if cache.get("accepted") is not True or cache["rows"] != 200000:
        raise ValueError("Accepted full cycle input cache is required")
    parent = root / "platforms/_records/kaggle/initializations/degree_bond_local_ema999_initial.pt"
    candidate = root / "platforms/_records/kaggle/initializations/degree_local_cycle_common_ema999_initial.pt"
    initial = freeze_initial(parent, candidate)
    reference_bundle = json.loads((root / "experiments/pcqm_gptrans_local_control_100k/reference/reference_bundle.json").read_bytes())
    original = reference_bundle["prediction_manifest"]
    original_path = root / original["artifact_locator"]
    if sha256_file(original_path) != original["artifact_sha256"]:
        raise ValueError("Pinned source of target-only identity changed")
    payload = torch.load(original_path, map_location="cpu", weights_only=True)
    idx_sha = tensor_digest(payload["source_idx"].reshape(-1), role="source_idx")
    target_sha = tensor_digest(payload["target_eV"].reshape(-1), role="target")
    atomic_json(gpu / "target_manifest.json", {"target_encoding": "little-endian-float64-contiguous-raw-bytes",
        "development_target_sha256": target_sha, "development_source_idx_sha256": idx_sha,
        "source_payload_sha256": original["artifact_sha256"], "rows": 50000,
        "scope": "Target/row identity only; historical predictions and MAE are not the new-contract comparator"})
    def pin(relative):
        return {"path": relative, "sha256": sha256_file(root / relative)}
    def ref(name, sha):
        return {"name": name, "version": "1", "sha256": sha}
    arms, bindings, acceptance, jobs = [], [], [], []
    for device, (arm_id, mode, role, state_sha, init_path) in enumerate((
            ("local_reference", PARENT, "reference", PARENT_STATE_SHA256, parent),
            ("cycle_common", CYCLE, "candidate", initial["state_sha256"], candidate))):
        tid = "TC-gptrans-shared-" + arm_id.replace("_", "-") + "-100k-s42"
        role_ref, budget_ref = f"{BASE}/gpu/role_plan.json", f"{BASE}/gpu/budget.json"
        usage = canonical_fingerprint({"role": "train", "plan": sha256_file(root / role_ref)})
        dev_usage = canonical_fingerprint({"role": "development", "plan": sha256_file(root / role_ref)})
        recipe = FamilyRecipe("gptrans_t", mode, 42, state_sha,
            DatasetSelection("fixed100k/manifest.json", FIXED_100K_SHA, True),
            100000, 50000, 46860, .001, .05, ScheduleSpec("warmup_cosine_epoch", 60, warmup_units=4),
            "best-development-ema", "torch-pass-randperm", 781,
            5.338300266634226, 1.2750902063230574, ema_decay=.999, torch_version="2.7.1",
            input_sidecar=DatasetSelection("cycles/manifest.json", cache["manifest_sha256"], True) if mode == CYCLE else None)
        record = build_screen_recipe(mode, configuration=recipe, family=("gptrans_t", "2"),
            source_idx_sha256=idx_sha, target_sha256=target_sha, train_role_identity=usage,
            development_role_identity=dev_usage)
        recipe_ref = f"{BASE}/gpu/{arm_id}/contract.json"
        atomic_json(root / recipe_ref, record)
        bound = family_arm_bindings(recipe)
        train_idx_sha = tensor_digest(torch.arange(100000), role="source_idx")
        arm = {"arm_id": arm_id, "scientific_role": role, "family": {"name": "gptrans_t", "version": "2"},
            "base": bound["base"], "initialization": {"kind": "frozen_state", "seed": 42, "state_sha256": state_sha},
            "data": {"dataset": ref("pcqm4mv2", FIXED_100K_SHA), "split": ref("fixed100k-internal50k", FIXED_100K_SHA),
                "roles": [{"role": name, "membership_sha256": digest, "row_order_sha256": digest, "usage_sha256": use}
                    for name, digest, use in (("train", train_idx_sha, usage), ("development", idx_sha, dev_usage))],
                "feature_schema": FAMILIES[("gptrans_t", "2")].feature_schema,
                "feature_sha256": canonical_fingerprint({"schema": "ogb9-bond3-spd20", "fixed_manifest": FIXED_100K_SHA}),
                "target": "pcqm4mv2-gap-eV-direct"},
            "training": {"recipe": ref("shared-family-training", normalized_source_sha256(root / recipe_ref)),
                "overrides": {}, **{k: bound[k] for k in ("sampler", "objective", "transform")}},
            "addons": [{"name": mode, "version": "1", "config": {}, "source_sha256": normalized_source_sha256(
                Path(__file__).with_name("gptrans_capacity.py" if mode == PARENT else "gptrans_cycle_pair.py"))}],
            "addon_semantics": "ordered"}
        arms.append(arm)
        cost = "cost-" + tid
        plan_ref = f"{BASE}/gpu/{arm_id}/plan_input.json"
        proposal = {"trajectory": {"trajectory_id": tid, "record_mode": "prospective", "owner": "server", "track": "C",
            "family_id": "gptrans-shared-cycle-pair", "question": "Does zero-start common-cycle pair encoding improve the same-contract local-bond reference?",
            "hypothesis": {"hypothesis_id": "H-" + tid, "observed_deficiency": "Shared-contract reference absent; ring pair relation is not explicit in the local-bond core",
                "supporting_evidence_ids": [CACHE, CONTEXT], "alternative_explanations": ["Redundant topology", "Extra fitting freedom", "New execution contract changes historical background"],
                "changed_mechanism": "unchanged shared-contract reference" if role == "reference" else "zero-start64active common-cycle pair embedding entries",
                "cheapest_falsifier": "one matched seed42 fixed100K pair", "related_closed_family_ids": ["gptrans-local-relation-flow", "gptrans-triplet-routing"],
                "expected_native_cost_ref": cost, "decision_changed_if_positive": "nominate only after actual reference and candidate acceptance",
                "decision_changed_if_negative": "close exact encoding without a grid or extra seed", "historical_unknowns": []},
            "state_at_start": {"source_commit": source_commit, "source_config_identity": canonical_fingerprint(arm),
                "contract_refs": [recipe_ref, f"{BASE}/gpu/authority.json", f"{BASE}/gpu/audit_action.json"],
                "reference_ids": [CONTEXT], "parent_trajectory_ids": [], "prior_trajectory_ids": [], "prior_evidence_ids": [CACHE, CONTEXT],
                "role_snapshot_refs": [role_ref], "budget_snapshot_ref": budget_ref},
            "actions": [{"action_id": "A001", "type": "shared_pair_training", "source_commit": source_commit,
                "run_ids": [RUN], "attempt_ids": ["v1"], "evidence_refs": [recipe_ref], "cost_event_ids": [cost]}],
            "result": {"evidence_ids": [], "evidence_refs": []},
            "decision": {"decision_ref": f"{BASE}/protocol.md", "outcome": "ACTIVE",
                "next_allowed_actions": ["one same-run reference/candidate screen after all-arm preflight"], "reopen_conditions": ["terminal evidence"]}},
            "decision_state": {"known_trajectory_ids": [], "known_evidence_ids": [], "active_reference_ids": [CONTEXT],
                "available_actions": ["RUN_SHARED_CYCLE_PAIR", "DEFER"], "chosen_action": "RUN_SHARED_CYCLE_PAIR",
                "policy_id": "gptrans-shared-cycle-pair", "policy_version": "v1", "budget_snapshot_ref": budget_ref,
                "role_snapshot_refs": [role_ref], "state_timestamp": datetime.now(timezone.utc).isoformat(), "source_commit": source_commit},
            "costs": [{"schema": "molgap-cost-event-v1", "cost_event_id": cost, "trajectory_id": tid,
                "action_id": "A001", "run_id": RUN, "attempt_id": "v1", "category": "training", "platform": "kaggle2",
                "hardware": "Tesla_T4x2_shared_allocation", "evidence_ref": budget_ref, "measurement": {
                    "device_hours": {"status": "estimated", "value": 8}, "wall_hours": {"status": "estimated", "value": 4},
                    "cpu_hours": {"status": "measurement_missing", "value": None}, "queue_hours": {"status": "measurement_missing", "value": None}}}]}
        atomic_json(root / plan_ref, proposal)
        bindings.append({"arm_id": arm_id, "trajectory_id": tid, "plan_spec_ref": plan_ref,
            "plan_spec_sha256": sha256_file(root / plan_ref), "output": f"{BASE}/gpu/{arm_id}/rml_plan"})
        acceptance.append({"arm_id": arm_id, "adapter": "gptrans-session-v1", "contract": pin(recipe_ref),
            "expected": record["acceptance_requirements"], "target_manifest": pin(f"{BASE}/gpu/target_manifest.json"),
            "initial_state": pin(init_path.relative_to(root).as_posix())})
        jobs.append({"arm_id": arm_id, "device": device, "recipe": recipe_ref, "initial_state": str(init_path)})
    spec = ExperimentSpec({"schema_version": "molgap-experiment-spec-v2", "experiment_id": "gptrans-common-cycle-100k",
        "logical_run_id": "gptrans-shared-cycle-dual-100k-s42-v1", "arms": arms,
        "platform": {"name": "kaggle", "accelerator": "NvidiaTeslaT4", "device_count": 2,
            "cpu_cores": 4, "memory_gib": 29, "atomic_checkpoints": True, "retrievable_chunks": True},
        "prospective": {"arms": bindings, "same_run_replay": {"reference_arm_id": "local_reference", "candidate_arm_ids": ["cycle_common"]}},
        "evidence": {"policy": ref("molgap-v5", canonical_fingerprint({"contract": "MOLGAP-COMMON-V5-FINAL"})),
            "required_artifacts": ["v5_evidence", "costs", "roles", "trace_manifest", "terminal_artifact"]},
        "terminal_protocol": TERMINAL_PROTOCOL})
    atomic_json(gpu / "experiment_spec.json", spec.to_dict())
    atomic_json(gpu / "acceptance_plan.json", {"format": "molgap-family-same-run-acceptance-plan-v1", "spec_identity": spec.identity,
        "authority": pin(f"{BASE}/gpu/authority.json"), "arms": acceptance, "audit": pin(f"{BASE}/gpu/audit_action.json"),
        "cache_acceptance": pin(f"{BASE}/cpu/acceptance.json")})
    atomic_json(gpu / "workflow_plan.json", {"format": "molgap-experiment-workflow-v1", "spec_identity": spec.identity,
        "source_files": [f"{BASE}/gpu/{name}" for name in ("authority.json", "audit_action.json", "target_manifest.json", "role_plan.json", "budget.json", "acceptance_plan.json")]
            + [f"{BASE}/cpu/acceptance.json"],
        "arms": jobs, "acceptance_plan": f"{BASE}/gpu/acceptance_plan.json", "kaggle": {
            "account": "kaseichou", "kernel": RUN, "title": "MolGap GPTrans Cycle Dual 100K V1",
            "source_dataset": "kaseichou/molgap-gptrans-cycle-dual-source-v1", "accelerator": "NvidiaTeslaT4",
            "datasets": ["kaseichou/molgap-gptrans-cycle-dual-source-v1", "kaseichou/pcqm4mv2-ogb-fixed-100k-v1",
                "kaseichou/molgap-gptrans-common-cycle-inputs-v1"]}})
    return {"spec_identity": spec.identity, "initialization": initial, "kernel": RUN}


def stage_cycle_dataset(root, output):
    """Publish only already accepted sidecar bytes; no graph or label rebuilding."""
    import shutil
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Use a fresh immutable cache staging directory")
    accepted = json.loads((root / BASE / "cpu/acceptance.json").read_bytes())
    source = root / "platforms/_records/kaggle/cycle_pair_cpu_recovery_v2/output/cycle_inputs"
    if sha256_file(source / "manifest.json") != accepted["manifest_sha256"] or accepted.get("accepted") is not True:
        raise ValueError("Cycle acceptance or manifest changed")
    manifest = json.loads((source / "manifest.json").read_bytes())
    output.mkdir(parents=True)
    files = {"manifest.json": accepted["manifest_sha256"], **{c["file"]: c["sha256"] for c in manifest["chunks"]}}
    for name, digest in files.items():
        original = source / name
        if sha256_file(original) != digest:
            raise ValueError("Accepted cycle bytes changed before upload")
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        if sha256_file(target) != digest:
            raise ValueError("Staged cycle bytes changed")
    atomic_json(output / "dataset-metadata.json", {"id": "kaseichou/molgap-gptrans-common-cycle-inputs-v1",
        "title": "MolGap GPTrans Common Cycle Inputs V1", "licenses": [{"name": "CC-BY-4.0"}]})
    return {"files": len(files), "manifest_sha256": accepted["manifest_sha256"], "private": True}
