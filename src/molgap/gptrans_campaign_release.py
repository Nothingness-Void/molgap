"""Question-owned four-arm binding over existing shared release primitives."""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess

from .experiment_spec import ExperimentSpec
from .family_training_runner import read_recipe, build_screen_recipe, family_arm_bindings, validate_screen_recipe
from .family_pair_release import resource_estimate
from .gptrans_research_campaign import MODES, BASE as PARENT, CYCLE, AUX, freeze_initial, inherited_state, PARENT_STATE_SHA
from .training_reproducibility import atomic_json, sha256_file
from .screen_policy import canonical_fingerprint
from .v4_runtime import normalized_source_sha256, read_frozen_state_artifact, state_dict_sha256

BASE = "experiments/pcqm_gptrans_research_four_arm_100k"
FORMAT = "molgap-four-arm-campaign-plan-v1"
SLOTS = (("reference", "cycle"), ("carry", "correspondence"))
ARMS = dict(zip(("reference", "cycle", "carry", "correspondence"), MODES))


def retired_bootstrap_cost(root, slot, *, source_text=None):
    """Retain measured old Spec costs without forging a new-Spec native ledger."""
    import math
    if source_text is None:
        path = Path(root) / BASE / f"infrastructure_costs/slot{slot}.json"
        if not path.exists():
            return {"wall_seconds": 0.0, "allocated_device_seconds": 0.0}
        source_text = path.read_bytes()
    value = json.loads(source_text)
    if value.get("format") != "molgap-campaign-retired-bootstrap-cost-v1":
        raise ValueError("Unsupported retired bootstrap cost record")
    wall = device = 0.0
    seen = set()
    for item in value["attempts"]:
        identity = (item["kernel"], item["version_number"])
        native = item["bootstrap_allocation"]
        seconds = native["wall_seconds"]
        if (identity in seen or native["status"] != "failed" or native["cost_status"] != "measured"
                or native["allocation_device_count"] != 2 or len(native["hardware"]) != 2
                or any("T4" not in name for name in native["hardware"])
                or not math.isfinite(seconds) or seconds < 0
                or not math.isclose(native["allocated_device_seconds"], 2 * seconds)):
            raise ValueError("Unknown or duplicated retired allocation interval")
        seen.add(identity)
        wall += seconds
        device += native["allocated_device_seconds"]
    return {"wall_seconds": wall, "allocated_device_seconds": device}


def check_campaign(spec, root, plan, *, initial_states=None):
    """Prospective authority/input checks only; future reference remains pending."""
    root = Path(root).resolve()
    from .experiment_family_workflow import _artifact_path, _json
    if set(plan) != {"format", "spec_identity", "campaign"} or plan["format"] != FORMAT or plan["spec_identity"] != spec.identity:
        raise ValueError("Four-arm release plan mismatch")
    def load(pin):
        path = _artifact_path(root, pin["path"])
        if normalized_source_sha256(path) != pin["sha256"]:
            raise ValueError("Campaign input bytes changed")
        return _json(path)
    campaign = load(plan["campaign"])
    authority = load(campaign["authority"])
    if (authority.get("user_authorized") is not True or authority.get("scope") != "two-slots-one-reference-three-candidates"
            or authority.get("seeds") != [42] or authority.get("maximum_allocated_t4_hours") != 32
            or authority.get("maximum_slot_wall_seconds") != 28800 or set(campaign["arms"]) != set(ARMS)):
        raise ValueError("Campaign exceeds explicit four-arm authority")
    audit = load(campaign["audit"])
    if audit.get("mode") != "NO_TRAIN" or audit.get("roles") != {"100k": [100000, 150000], "500k": [500000, 550000]}:
        raise ValueError("Deferred full50K audit scope changed")
    cache = load(campaign["cpu_acceptance"])
    if cache.get("accepted") is not True or cache.get("rows") != 200000:
        raise ValueError("Accepted immutable cycle inputs are missing")
    physical = [ExperimentSpec(load(pin)) for pin in campaign["slots"]]
    selected = next((s for s in physical if s == spec), None)
    if selected is None or {tuple(a["arm_id"] for a in s.to_dict()["arms"]) for s in physical} != set(SLOTS):
        raise ValueError("Physical slots do not cover exactly the authorized four arms")
    recipes = {}
    for arm_id, declaration in campaign["arms"].items():
        _, recipe = read_recipe(_artifact_path(root, declaration["contract"]["path"]))
        if normalized_source_sha256(root / declaration["contract"]["path"]) != declaration["contract"]["sha256"] or recipe.mode != ARMS[arm_id]:
            raise ValueError("Campaign recipe mode/pin changed")
        recipes[arm_id] = recipe
    ref = recipes["reference"]
    for arm_id, recipe in recipes.items():
        if replace(recipe, mode=ref.mode, initialization_sha256=ref.initialization_sha256, input_sidecar=None) != ref:
            raise ValueError("Candidate changed an undeclared optimizer/data/exposure field")
    if (ref.train_rows, ref.development_rows, ref.optimizer_steps, ref.batch_size, ref.seed, ref.ema_decay,
            ref.selection, ref.schedule.total_units, ref.schedule.warmup_units, ref.torch_version) != (
            100000, 50000, 46860, 128, 42, .999, "best-development-ema", 60, 4, "2.7.1"):
        raise ValueError("Campaign scientific/resource scope changed")
    available = initial_states or {name: root / item["initial_state"]["path"] for name, item in campaign["arms"].items()}
    for arm in spec.to_dict()["arms"]:
        arm_id = arm["arm_id"]
        record, recipe = read_recipe(root / campaign["arms"][arm_id]["contract"]["path"])
        validate_screen_recipe(spec, arm_id, record)
        path = Path(available[arm_id])
        if sha256_file(path) != campaign["arms"][arm_id]["initial_state"]["sha256"]:
            raise ValueError("Campaign initialization bytes changed")
        state, _ = read_frozen_state_artifact(path, expected_state_sha256=recipe.initialization_sha256)
        if state_dict_sha256(inherited_state(state)) != PARENT_STATE_SHA:
            raise ValueError("Campaign changed frozen aggregate parent tensors")
        if arm_id in {"cycle", "carry"} and any(v.count_nonzero() for k, v in state.items() if k not in inherited_state(state)):
            raise ValueError("Cycle/carry intervention must start at zero")
    return {"status": "ACCEPTANCE_INPUTS_AVAILABLE", "reference_status": "PLANNED_NOT_ACCEPTED",
        "scientific_acceptance": "NOT_EVALUATED", "scope": "one-reference-three-candidates-two-physical-slots"}


def freeze(root):
    """Reuse authored target pins and typed recipe/RML/Spec constructors."""
    root = Path(root).resolve()
    destination = root / BASE
    if destination.exists():
        raise ValueError("Campaign already frozen; reconcile instead of duplicating")
    old = root / "experiments/pcqm_gptrans_cycle_pair_100k/gpu"
    load = lambda path: json.loads(Path(path).read_bytes())
    # Text pins must match the LF transport used by the shared source packager.
    # Initialization artifacts retain their exact binary hash.
    pin = lambda path: {"path": str(path), "sha256": (
        sha256_file(root / path) if Path(path).suffix == ".pt"
        else normalized_source_sha256(root / path))}
    original = load(old / "experiment_spec.json")
    target = load(old / "target_manifest.json")
    template = load(old / "local_reference/plan_input.json")
    _, recipe_template = read_recipe(old / "local_reference/contract.json")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    stamp = datetime.now(timezone.utc).isoformat()
    atomic_json(destination / "target_manifest.json", target)
    authority = {"user_authorized": True, "user_reply": "Submit two slots, four arms total: one baseline and three candidates",
        "scope": "two-slots-one-reference-three-candidates", "seeds": [42],
        "maximum_allocated_t4_hours": 32, "maximum_slot_wall_seconds": 28800,
        "physical_segment_passes": 10, "reference_status": "PLANNED_NOT_ACCEPTED",
        "forbidden": ["extra_seed", "fifth_arm", "500k_training", "full_training", "protected_roles"]}
    atomic_json(destination / "authority.json", authority)
    audit = load(old / "audit_action.json")
    audit["combined_training_and_audit_allocated_t4_hours_ceiling"] = 32
    audit["release_policy"] = "Separate idempotent NO_TRAIN action after actual four-arm training acceptance; no automatic audit launch"
    atomic_json(destination / "audit_action.json", audit)
    atomic_json(destination / "role_plan.json", load(old / "role_plan.json"))
    atomic_json(destination / "budget.json", {"physical_notebooks": 2, "models": 4, "seed": 42,
        "allocated_t4_hours_ceiling": 32, "wall_hours_per_slot_ceiling": 8,
        "estimated_slot_wall_hours": 5, "native_timing_required": True,
        "quota_observation": "Not API-certified; returned scheduler/admission decides availability"})
    arms, bindings, entries = {}, {}, {}
    parent = root / "platforms/_records/kaggle/initializations/degree_local_triplet_aggregate_ema999_initial.pt"
    for arm_id, mode in ARMS.items():
        initial = root / f"platforms/_records/kaggle/initializations/{mode}_initial.pt"
        facts = freeze_initial(mode, parent, initial)
        recipe = replace(recipe_template, mode=mode, initialization_sha256=facts["state_sha256"],
            input_sidecar=replace(recipe_template.dataset, manifest="cycles/manifest.json",
                manifest_sha256="1fd890bf91bc633c8323c17fe87d664f3106b26052f881e145d22b57f6606ac4") if mode == CYCLE else None)
        record = build_screen_recipe(mode, configuration=recipe, family=("gptrans_t", "2"),
            source_idx_sha256=target["development_source_idx_sha256"], target_sha256=target["development_target_sha256"],
            train_role_identity=load(old / "local_reference/contract.json")["train_role_identity"],
            development_role_identity=load(old / "local_reference/contract.json")["development_role_identity"])
        recipe_ref = f"{BASE}/{arm_id}/contract.json"
        atomic_json(root / recipe_ref, record)
        arm = deepcopy(original["arms"][0])
        bound = family_arm_bindings(recipe)
        arm.update(arm_id=arm_id, scientific_role="reference" if arm_id == "reference" else "candidate",
            base=bound["base"], initialization={"kind": "frozen_state", "seed": 42, "state_sha256": facts["state_sha256"]},
            training={"recipe": {"name": "shared-family-training", "version": "1", "sha256": normalized_source_sha256(root / recipe_ref)},
                "overrides": {}, **{k: bound[k] for k in ("sampler", "objective", "transform")}},
            addons=[{"name": mode, "version": "1", "config": {},
                "source_sha256": normalized_source_sha256(Path(__file__).with_name("gptrans_research_campaign.py"))}])
        arms[arm_id] = arm
        tid = "TC-gptrans-aggregate-" + arm_id + "-100k-s42"
        proposal = deepcopy(template)
        trajectory = proposal["trajectory"]
        trajectory.update(trajectory_id=tid, family_id="gptrans-aggregate-research-campaign",
            question="Does " + arm_id + " improve the matched shared aggregate reference?")
        trajectory["hypothesis"].update(hypothesis_id="H-" + tid,
            changed_mechanism=arm_id, supporting_evidence_ids=["pcqm-gptrans-g1-degree-local-triplet-aggregate-ema999-100k-s42"],
            expected_native_cost_ref="cost-" + tid, cheapest_falsifier="one shared-contract seed42 fixed100K screen",
            alternative_explanations=["Redundant information", "Optimization interference", "Single-seed development reuse"])
        trajectory["state_at_start"].update(source_commit=source, source_config_identity=canonical_fingerprint(arm),
            contract_refs=[recipe_ref, f"{BASE}/authority.json", f"{BASE}/audit_action.json"],
            reference_ids=["pcqm-gptrans-g1-degree-local-triplet-aggregate-ema999-100k-s42"],
            parent_trajectory_ids=["TC-gptrans-g1-local-triplet-aggregate-100k-s42"],
            role_snapshot_refs=[f"{BASE}/role_plan.json"], budget_snapshot_ref=f"{BASE}/budget.json")
        run = "kaseichou/molgap-gptrans-research-slot" + ("1" if arm_id in SLOTS[0] else "2") + "-100k-v1"
        trajectory["actions"][0].update(source_commit=source, run_ids=[run], evidence_refs=[recipe_ref], cost_event_ids=["cost-" + tid])
        trajectory["decision"].update(decision_ref=f"{BASE}/protocol.md", next_allowed_actions=["one four-arm authorized campaign; reference first acceptance"])
        proposal["decision_state"].update(source_commit=source,
            state_timestamp=stamp, role_snapshot_refs=[f"{BASE}/role_plan.json"], budget_snapshot_ref=f"{BASE}/budget.json")
        proposal["decision_state"].update(policy_id="gptrans-aggregate-four-arm", policy_version="v1",
            available_actions=["RUN_FOUR_ARM_CAMPAIGN", "DEFER"], chosen_action="RUN_FOUR_ARM_CAMPAIGN")
        proposal["costs"][0].update(cost_event_id="cost-" + tid, trajectory_id=tid, run_id=run, evidence_ref=f"{BASE}/budget.json")
        plan_ref = f"{BASE}/{arm_id}/plan_input.json"
        atomic_json(root / plan_ref, proposal)
        bindings[arm_id] = {"arm_id": arm_id, "trajectory_id": tid, "plan_spec_ref": plan_ref,
            "plan_spec_sha256": sha256_file(root / plan_ref), "output": f"{BASE}/{arm_id}/rml_plan"}
        entries[arm_id] = {"contract": pin(recipe_ref), "initial_state": pin(initial.relative_to(root).as_posix()),
            "parameters": facts["parameters"], "reference_arm_id": "reference", "mode": mode}
    specs = []
    for number, slot in enumerate(SLOTS, 1):
        payload = deepcopy(original)
        payload.update(experiment_id="gptrans-research-slot" + str(number),
            logical_run_id="gptrans-research-slot" + str(number) + "-100k-s42-v1",
            arms=[arms[name] for name in slot], prospective={"arms": [bindings[name] for name in slot]})
        spec = ExperimentSpec(payload)
        path = f"{BASE}/slot{number}/scientific_spec.json"
        atomic_json(root / path, spec.to_dict())
        specs.append((spec, path))
    atomic_json(destination / "campaign.json", {"format": FORMAT, "authority": pin(f"{BASE}/authority.json"),
        "audit": pin(f"{BASE}/audit_action.json"), "arms": entries, "slots": [pin(p) for _, p in specs],
        "cpu_acceptance": pin("experiments/pcqm_gptrans_cycle_pair_100k/cpu/acceptance.json"),
        "reference_status": "PLANNED_NOT_ACCEPTED", "terminal_order": ["reference", "cycle", "carry", "correspondence"]})
    for number, ((spec, _), slot) in enumerate(zip(specs, SLOTS), 1):
        plan = {"format": FORMAT, "spec_identity": spec.identity, "campaign": pin(f"{BASE}/campaign.json")}
        atomic_json(destination / f"slot{number}/acceptance_plan.json", plan)
        source_files = [f"{BASE}/{p}" for p in ("authority.json", "audit_action.json", "role_plan.json", "budget.json", "target_manifest.json", "campaign.json")]
        # prepare_workflow already contributes the physical slot's recipes.
        source_files += [entries[a]["contract"]["path"] for a in ARMS if a not in slot] + [p for _, p in specs]
        source_files += [f"{BASE}/slot{number}/acceptance_plan.json", "experiments/pcqm_gptrans_cycle_pair_100k/cpu/acceptance.json"]
        dataset = f"kaseichou/molgap-gptrans-research-slot{number}-source-v1"
        atomic_json(destination / f"slot{number}/workflow_plan.json", {"format": "molgap-experiment-workflow-v1",
            "spec_identity": spec.identity, "source_files": source_files,
            "acceptance_plan": f"{BASE}/slot{number}/acceptance_plan.json",
            "arms": [{"arm_id": a, "device": i, "recipe": entries[a]["contract"]["path"],
                "initial_state": str(root / entries[a]["initial_state"]["path"])} for i, a in enumerate(slot)],
            "kaggle": {"account": "kaseichou", "kernel": f"kaseichou/molgap-gptrans-research-slot{number}-100k-v1",
                "title": f"MolGap GPTrans Research Slot{number} 100K V1", "source_dataset": dataset,
                "accelerator": "NvidiaTeslaT4", "datasets": [dataset, "kaseichou/pcqm4mv2-ogb-fixed-100k-v1",
                    "kaseichou/molgap-gptrans-common-cycle-inputs-v1"]}})
        check_campaign(spec, root, plan)
    return {"arms": entries, "spec_identities": [s.identity for s, _ in specs]}
