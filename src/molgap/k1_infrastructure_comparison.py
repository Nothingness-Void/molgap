"""Release checks for the authorized original/shared K1 execution diagnostic."""
from pathlib import Path, PureWindowsPath
import re

from .experiment_family_workflow import _artifact_path, _json
from .training_reproducibility import sha256_file

FORMAT = "molgap-k1-infrastructure-comparison-plan-v1"
BASE = "experiments/k1_infrastructure_t4_100k"
WALL_SECONDS = 7 * 3600
_MISSING = object()


def _safe_base(value, *, require_under_base=True):
    """Normalize a comparison storage root without accepting path aliases."""
    if isinstance(value, Path):
        value = value.as_posix()
    if type(value) is not str or not value or "\\" in value or ":" in value:
        raise ValueError("Comparison base must be a repository-relative POSIX path")
    if value.startswith("/") or PureWindowsPath(value).drive or PureWindowsPath(value).root:
        raise ValueError("Comparison base must be relative")
    parts = value.split("/")
    if any(not part or part in {".", ".."} or part.endswith((" ", "."))
            or PureWindowsPath(part).is_reserved() for part in parts):
        raise ValueError("Comparison base contains an unsafe path component")
    if require_under_base and value != BASE and not value.startswith(BASE + "/"):
        raise ValueError("Comparison base must be under the frozen K1 comparison root")
    return value


def _run_version(value):
    if type(value) is not str or re.fullmatch(r"v[1-9][0-9]*", value) is None:
        raise ValueError("Comparison run_version must be a version such as v1 or v2")
    return value


def _arm_plan_base(path, arm_id):
    suffix = "/" + arm_id + "/plan_input.json"
    if type(path) is not str or not path.endswith(suffix):
        raise ValueError("Comparison plan reference has an unexpected arm suffix")
    return path[:-len(suffix)]


def comparison_base(spec):
    """Return the common prospective storage root for a comparison round.

    The plan references are part of the Spec identity, so deriving the root
    from them keeps runtime/analysis from silently falling back to the old v1
    directory when a later diagnostic is selected.
    """
    if not is_comparison(spec):
        raise ValueError("Not the frozen K1 execution comparison")
    declaration = spec.to_dict()
    arms = declaration.get("arms")
    prospective = declaration.get("prospective")
    if (type(arms) is not list or [arm.get("arm_id") for arm in arms] != ["original", "optimized"]
            or type(prospective) is not dict or type(prospective.get("arms")) is not list
            or [item.get("arm_id") for item in prospective["arms"]] != ["original", "optimized"]):
        raise ValueError("Comparison Spec must declare original and optimized prospective arms")
    refs = prospective["arms"]
    bases = []
    for item, arm_id in zip(refs, ("original", "optimized")):
        if type(item) is not dict or type(item.get("plan_spec_ref")) is not str:
            raise ValueError("Comparison prospective arm lacks a plan reference")
        path = item["plan_spec_ref"]
        # A reference is deliberately checked before suffix stripping. This
        # rejects absolute paths, backslashes and traversal aliases even when
        # the suffix itself looks correct.
        _safe_base(path, require_under_base=False)
        bases.append(_safe_base(_arm_plan_base(path, arm_id)))
    if bases[0] != bases[1]:
        raise ValueError("Comparison prospective arms do not share a storage root")
    return bases[0]


def freeze_comparison(root, *, account="nvoid912", base=BASE, run_version="v1",
                      execution_config=None):
    """Freeze declarations using existing recipe, identity and RML schemas."""
    import subprocess
    from datetime import datetime, timezone
    import torch
    from .experiment_spec import ExperimentSpec, FAMILIES, TERMINAL_PROTOCOL
    from .experiment_family_workflow import tensor_digest
    from .family_training_recipe import FamilyRecipe, ScheduleSpec
    from .family_training_data import DatasetSelection
    from .family_training_runner import build_screen_recipe, family_arm_bindings
    from .k1_screen_training import (build_screen_recipe as legacy_recipe, INITIAL_STATE_SHA256,
        FIXED_MANIFEST_SHA256, ROW_ORDER_FINGERPRINT, _screen_metric_semantics)
    from .qm9_neural_atom import make_encoder
    from .training_reproducibility import atomic_torch_save
    from .research_memory.trace import atomic_write, json_bytes
    from .screen_policy import canonical_fingerprint
    from .v4_runtime import state_dict_sha256, normalized_source_sha256
    root = Path(root).resolve()
    base = _safe_base(base)
    run_version = _run_version(run_version)
    if execution_config is not None:
        from .k1_execution_backend import K1ExecutionConfig
        if type(execution_config) is not K1ExecutionConfig:
            raise TypeError("Typed K1 execution configuration is required")
    if base != BASE and execution_config is None:
        raise ValueError("A new comparison round requires an explicit K1 execution configuration")

    def atomic_json(path, value):
        atomic_write(path, json_bytes(value))
    here = root / base
    dynamic_paths = ("experiment_spec.json", "acceptance_plan.json", "workflow_plan.json",
                     "original", "optimized")
    if any((here / name).exists() for name in dynamic_paths):
        raise ValueError("Comparison already frozen; reconcile before another attempt")
    # The original v1 directory is an immutable historical declaration. A
    # caller may request a new round only in a fresh descendant directory.
    if base == BASE and (run_version != "v1" or execution_config is not None):
        raise ValueError("The frozen v1 comparison cannot be overwritten")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    initial = root / "platforms/_records/kaggle/initializations/k1_reference_s42.pt"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = make_encoder("neural_atom_k1")
        state = model.state_dict()
    if state_dict_sha256(state) != INITIAL_STATE_SHA256 or sum(p.numel() for p in model.parameters()) != 3658817:
        raise ValueError("K1 frozen reference initialization changed")
    if initial.exists():
        from .v4_runtime import inspect_frozen_state_artifact
        inspect_frozen_state_artifact(initial, expected_state_sha256=INITIAL_STATE_SHA256)
    else:
        atomic_torch_save(initial, state)
    del model, state
    target = _json(root / "experiments/pcqm_gptrans_cycle_pair_100k/gpu/target_manifest.json")
    idx_sha, target_sha = target["development_source_idx_sha256"], target["development_target_sha256"]
    train_sha = tensor_digest(torch.arange(100000), role="source_idx")
    def reused_gate(name):
        candidate = base + "/" + name
        return candidate if (root / candidate).is_file() else BASE + "/" + name

    role_ref = reused_gate("role_plan.json")
    protocol_ref = reused_gate("protocol.md")
    decision_ref = reused_gate("decision.md")
    authority_ref = base + "/authority.json"
    if base == BASE:
        authority = _json(root / authority_ref)
    else:
        authority_path = root / authority_ref
        if authority_path.is_file():
            authority = _json(authority_path)
        else:
            authority = _json(root / (BASE + "/authority.json"))
            authority["optimized_execution"] = execution_config.to_dict()
            authority["source"] = ("User-authorized matched native profiler comparison of retained original K1 "
                                    "and shared execution providers at fixed 100K; execution diagnostic only.")
            atomic_json(authority_path, authority)
        if authority.get("optimized_execution") != execution_config.to_dict():
            raise ValueError("Existing round authority has a different K1 execution configuration")
    usage = {r: canonical_fingerprint({"role": r, "plan": sha256_file(root / role_ref)}) for r in ("train", "development")}
    config = FamilyRecipe("neural_atom_k1", "reference", 42, INITIAL_STATE_SHA256,
        DatasetSelection("manifest.json", FIXED_MANIFEST_SHA256, True), 100000, 50000, 31240,
        .0004, .00001, ScheduleSpec("cosine_epoch", 40), "best-development-live", "python-epoch-shuffle",
        781, 5.3383002281188965, 1.275090217590332, torch_version="2.7.1",
        k1_execution=execution_config)
    shared = build_screen_recipe("reference", configuration=config, family=("neural_atom_k1", "3"),
        source_idx_sha256=idx_sha, target_sha256=target_sha, train_role_identity=usage["train"],
        development_role_identity=usage["development"])
    records = [legacy_recipe("reference", source_idx_sha256=idx_sha, target_sha256=target_sha), shared]
    profiling = {"warmup_steps": 120, "active_steps": 16, "record_shapes": False,
        "profile_memory": False, "with_stack": False, "row_limit": 80, "module_paths": []}
    def ref(name, digest):
        return {"name": name, "version": "1", "sha256": digest}
    def pin(path):
        return {"path": path, "sha256": sha256_file(root / path)}
    arms, bindings, jobs, entries = [], [], [], []
    run = "k1-infrastructure-t4-100k-s42-" + run_version
    kernel = account + "/molgap-k1-infrastructure-dual-100k-" + run_version
    timestamp = datetime.now(timezone.utc).isoformat()
    for device, (name, version, record) in enumerate(zip(("original", "optimized"), ("2", "3"), records)):
        record["profiling"] = profiling
        recipe_ref = base + "/" + name + "/contract.json"
        atomic_json(root / recipe_ref, record)
        bound = family_arm_bindings(config)
        if version == "2":
            bound["sampler"] = ref("seed42-python-epoch-shuffle-v4", ROW_ORDER_FINGERPRINT)
            bound["transform"] = {**bound["transform"], "name": "fixed-train-100k-mean-sample-std"}
        arm = {"arm_id": name, "scientific_role": "reference" if name == "original" else "candidate",
            "family": {"name": "neural_atom_k1", "version": version}, "base": bound["base"],
            "initialization": {"kind": "frozen_state", "seed": 42, "state_sha256": INITIAL_STATE_SHA256},
            "data": {"dataset": ref("pcqm4mv2", FIXED_MANIFEST_SHA256),
                "split": ref("fixed100k-internal50k", FIXED_MANIFEST_SHA256),
                "roles": [{"role": r, "membership_sha256": d, "row_order_sha256": d, "usage_sha256": usage[r]}
                    for r, d in (("train", train_sha), ("development", idx_sha))],
                "feature_schema": FAMILIES[("neural_atom_k1", version)].feature_schema,
                "feature_sha256": canonical_fingerprint({"atom": 9, "bond": 3, "rwse": 16, "geometry": False}),
                "target": "pcqm4mv2-gap-eV-direct"},
            "training": {"recipe": ref(FAMILIES[("neural_atom_k1", version)].recipe,
                normalized_source_sha256(root / recipe_ref)), "overrides": {},
                **{k: bound[k] for k in ("sampler", "objective", "transform")}},
            "addons": [], "addon_semantics": "baseline"}
        arms.append(arm)
        tid = "TC-k1-infrastructure-" + name + "-100k-s42"
        if run_version != "v1":
            tid += "-" + run_version
        cost = "cost-" + tid + "-planned"
        plan_ref = base + "/" + name + "/plan_input.json"
        question = "How much does modular K1 infrastructure change T4 execution cost?"
        observed_deficiency = "Desktop cross-job speed observations lack a same-allocation original/shared comparison"
        changed_mechanism = "retained original execution" if name == "original" else "shared modular optimized execution"
        prior_trajectory_ids = []
        if run_version != "v1":
            question = "Does the qualified K1 execution-provider configuration reduce T4 execution cost at fixed 100K exposure?"
            observed_deficiency = "The qualified K1 execution-provider configuration lacks a matched native 100K profiler comparison"
            changed_mechanism = ("retained original K1 execution in the matched control"
                                 if name == "original" else "qualified cached K1 aggregation and pooling providers")
            prior_trajectory_ids = ["TC-k1-infrastructure-" + name + "-100k-s42"]
        proposal = {"trajectory": {"trajectory_id": tid, "record_mode": "prospective", "owner": "server", "track": "C",
            "family_id": "k1-infrastructure-execution", "question": question,
            "hypothesis": {"hypothesis_id": "H-" + tid, "observed_deficiency": observed_deficiency,
                "supporting_evidence_ids": ["pcqm-k1-v4-100k-reference-s42"],
                "alternative_explanations": ["CPU contention", "Device variability", "Instrumentation overhead"],
                "changed_mechanism": changed_mechanism,
                "cheapest_falsifier": "one matched seed42 T4x2 execution diagnostic with actual operator capture",
                "related_closed_family_ids": ["k1-v4-reference"], "expected_native_cost_ref": cost,
                "decision_changed_if_positive": "retain measured execution benefit without architecture promotion",
                "decision_changed_if_negative": "record no gain/regression without a retry or extra seed", "historical_unknowns": []},
            "state_at_start": {"source_commit": commit, "source_config_identity": canonical_fingerprint(arm),
                "contract_refs": [recipe_ref, protocol_ref, authority_ref], "reference_ids": [],
                "parent_trajectory_ids": [], "prior_trajectory_ids": prior_trajectory_ids,
                "prior_evidence_ids": ["pcqm-k1-v4-100k-reference-s42"],
                "role_snapshot_refs": [role_ref], "budget_snapshot_ref": authority_ref},
            "actions": [{"action_id": "A001", "type": "execution_diagnostic", "source_commit": commit,
                "run_ids": [run + ":" + name + ":downstream"], "attempt_ids": [name + "-" + run_version],
                "evidence_refs": [recipe_ref], "cost_event_ids": [cost]}],
            "result": {"evidence_ids": [], "evidence_refs": []}, "decision": {"decision_ref": decision_ref,
                "outcome": "ACTIVE", "next_allowed_actions": ["one authorized execution diagnostic"], "reopen_conditions": ["terminal evidence"]}},
            "decision_state": {"known_trajectory_ids": [], "known_evidence_ids": [], "active_reference_ids": [],
                "available_actions": ["RUN_K1_EXECUTION_DIAGNOSTIC", "DEFER"], "chosen_action": "RUN_K1_EXECUTION_DIAGNOSTIC",
                "policy_id": "k1-infrastructure-execution", "policy_version": "v1", "budget_snapshot_ref": authority_ref,
                "role_snapshot_refs": [role_ref], "state_timestamp": timestamp, "source_commit": commit},
            "costs": [{"schema": "molgap-cost-event-v1", "cost_event_id": cost, "trajectory_id": tid,
                "action_id": "A001", "run_id": run + ":" + name + ":downstream", "attempt_id": name + "-" + run_version,
                "category": "training", "platform": "kaggle3", "hardware": "Tesla_T4x2_shared_allocation", "evidence_ref": authority_ref,
                "measurement": {"device_hours": {"status": "estimated", "value": 7}, "wall_hours": {"status": "estimated", "value": 3.5},
                    "cpu_hours": {"status": "measurement_missing", "value": None}, "queue_hours": {"status": "measurement_missing", "value": None}}}]}
        atomic_json(root / plan_ref, proposal)
        bindings.append({"arm_id": name, "trajectory_id": tid, "plan_spec_ref": plan_ref,
            "plan_spec_sha256": sha256_file(root / plan_ref), "output": base + "/" + name + "/rml"})
        jobs.append({"arm_id": name, "device": device, "recipe": recipe_ref, "initial_state": str(initial)})
        entries.append({"arm_id": name, "contract": pin(recipe_ref), "expected": record["acceptance_requirements"],
            "initial_state": pin(initial.relative_to(root).as_posix())})
    spec = ExperimentSpec({"schema_version": "molgap-experiment-spec-v2", "experiment_id": "k1-infrastructure-t4-100k",
        "logical_run_id": run, "arms": arms, "platform": {"name": "kaggle", "accelerator": "NvidiaTeslaT4", "device_count": 2,
            "cpu_cores": 4, "memory_gib": 29, "atomic_checkpoints": True, "retrievable_chunks": True},
        "prospective": {"arms": bindings}, "evidence": {"policy": ref("molgap-v5", canonical_fingerprint({"contract": "MOLGAP-COMMON-V5-FINAL"})),
            "required_artifacts": ["v5_evidence", "costs", "roles", "trace_manifest", "terminal_artifact"]}, "terminal_protocol": TERMINAL_PROTOCOL})
    spec.write(here / "experiment_spec.json")
    atomic_json(here / "acceptance_plan.json", {"format": FORMAT, "spec_identity": spec.identity,
        "authority": pin(authority_ref), "arms": entries})
    from .experiment_source_inventory import registered_source_files
    sources = registered_source_files(spec, repo_root=root, auto_dependencies=True)
    source_dataset = account + "/molgap-k1-infrastructure-source-" + run_version
    title = "MolGap K1 Infrastructure Dual 100K " + run_version.upper()
    atomic_json(here / "workflow_plan.json", {"format": "molgap-experiment-workflow-v1", "spec_identity": spec.identity,
        "source_files": sources + [authority_ref, role_ref, protocol_ref, base + "/acceptance_plan.json"],
        "arms": jobs, "acceptance_plan": base + "/acceptance_plan.json", "kaggle": {"account": account, "kernel": kernel,
            "title": title, "source_dataset": source_dataset,
            "accelerator": "NvidiaTeslaT4", "datasets": [source_dataset, account + "/pcqm4mv2-ogb-fixed-100k-v1"]}})
    return {"spec_identity": spec.identity, "initial_state": str(initial), "kernel": kernel}


def is_comparison(spec):
    return spec.to_dict()["experiment_id"] == "k1-infrastructure-t4-100k"


def check_plan(spec, root, plan, *, initial_states=None):
    from .k1_screen_training import validate_screen_recipe as check_legacy
    from .family_training_runner import read_recipe, validate_screen_recipe as check_shared
    root = Path(root).absolute()
    declaration = spec.to_dict()
    if (not is_comparison(spec) or set(plan) != {"format", "spec_identity", "authority", "arms"}
            or plan["format"] != FORMAT or plan["spec_identity"] != spec.identity):
        raise ValueError("Unsupported K1 execution comparison")
    base = comparison_base(spec)
    authority_path = base + "/authority.json"
    if (type(plan.get("authority")) is not dict
            or plan["authority"].get("path") != authority_path):
        raise ValueError("Comparison authority path is not bound to the Spec round")
    def pinned(pointer):
        if type(pointer) is not dict or set(pointer) != {"path", "sha256"}:
            raise ValueError("Expected pinned execution input")
        path = _artifact_path(root, pointer["path"])
        if sha256_file(path) != pointer["sha256"]:
            raise ValueError("Execution comparison input changed")
        return path
    authority = _json(pinned(plan["authority"]))
    if (authority.get("user_authorized") is not True or authority.get("scope") != "execution-diagnostic-only"
            or authority.get("seeds") != [42] or authority.get("arms") != ["original", "optimized"]
            or authority.get("wall_ceiling_seconds") != WALL_SECONDS
            or authority.get("allocated_t4_hours_ceiling") != 14
            or authority.get("official_validation_allowed") is not False):
        raise ValueError("Execution diagnostic scope changed")
    arms = declaration["arms"]
    platform = declaration["platform"]
    if platform["name"] != "kaggle" or platform["accelerator"] != "NvidiaTeslaT4" or platform["device_count"] != 2:
        raise ValueError("K1 comparison requires its declared T4x2 allocation")
    if ([a["arm_id"] for a in arms] != ["original", "optimized"] or
            [a["family"] for a in arms] != [{"name": "neural_atom_k1", "version": v} for v in ("2", "3")]
            or any(a["addons"] or a["training"]["overrides"] for a in arms)
            or declaration["prospective"].get("same_run_replay") is not None):
        raise ValueError("Comparison requires original/shared K1 reference interfaces, no scientific addon or Replay claim")
    if arms[0]["data"] != arms[1]["data"] or arms[0]["initialization"] != arms[1]["initialization"]:
        raise ValueError("K1 comparison input/initialization differs")
    entries = plan["arms"]
    if type(entries) is not list or [e.get("arm_id") for e in entries] != ["original", "optimized"]:
        raise ValueError("Comparison plan must bind both arms")
    records = []
    for arm, entry in zip(arms, entries):
        if set(entry) != {"arm_id", "contract", "expected", "initial_state"}:
            raise ValueError("Unexpected execution comparison fields")
        expected_contract = base + "/" + arm["arm_id"] + "/contract.json"
        if (type(entry["contract"]) is not dict
                or entry["contract"].get("path") != expected_contract):
            raise ValueError("Comparison contract path is not bound to the Spec round")
        path = pinned(entry["contract"])
        record = _json(path)
        if record.get("acceptance_requirements") != entry["expected"]:
            raise ValueError("Comparison endpoint changed")
        (check_legacy if arm["arm_id"] == "original" else check_shared)(spec, arm["arm_id"], record)
        from .v4_runtime import inspect_frozen_state_artifact
        initial = pinned(entry["initial_state"]) if initial_states is None else Path(initial_states[arm["arm_id"]])
        if sha256_file(initial) != entry["initial_state"]["sha256"]:
            raise ValueError("Staged K1 initialization bytes changed")
        inspect_frozen_state_artifact(initial,
            expected_state_sha256=arm["initialization"]["state_sha256"])
        records.append(record)
    legacy, shared = records
    _, recipe = read_recipe(pinned(entries[1]["contract"]))
    if (recipe.family != "neural_atom_k1" or recipe.mode != "reference" or recipe.seed != 42
            or recipe.train_rows != 100000 or recipe.development_rows != 50000
            or recipe.optimizer_steps != 31240 or recipe.ema_decay is not None
            or recipe.learning_rate != .0004 or recipe.weight_decay != .00001
            or recipe.selection != "best-development-live" or recipe.sampler != "python-epoch-shuffle"
            or not recipe.drop_last or recipe.checkpoint_every != 781 or recipe.finite_check_every != 50
            or recipe.target_mean_eV != 5.3383002281188965 or recipe.target_std_eV != 1.275090217590332
            or recipe.torch_version != "2.7.1" or recipe.schedule.kind != "cosine_epoch"
            or recipe.schedule.total_units != 40 or recipe.schedule.minimum_lr != 1e-6):
        raise ValueError("Shared K1 differs from the frozen original scientific recipe")
    optimized_execution = authority.get("optimized_execution", _MISSING)
    if recipe.k1_execution is None:
        if optimized_execution is not _MISSING:
            raise ValueError("Authority contains an execution configuration without a bound recipe")
    else:
        from .k1_execution_backend import K1ExecutionConfig
        if optimized_execution is _MISSING:
            raise ValueError("Authority is missing the bound optimized execution configuration")
        try:
            bound_execution = K1ExecutionConfig.from_dict(optimized_execution)
        except (TypeError, ValueError) as exc:
            raise ValueError("Authority optimized execution configuration is invalid") from exc
        if bound_execution != recipe.k1_execution:
            raise ValueError("Authority optimized execution configuration differs from the recipe")
    comparable = ("optimizer_steps", "sample_presentations", "development_rows", "precision", "source_idx_sha256", "target_sha256")
    if any(legacy["acceptance_requirements"][key] != shared["acceptance_requirements"][key] for key in comparable):
        raise ValueError("K1 paired exposure/rows/target differs")
    if legacy.get("profiling") != shared.get("profiling") or not legacy.get("profiling"):
        raise ValueError("Both arms require the same bounded profile window")
    if legacy["profiling"] != {"warmup_steps": 120, "active_steps": 16, "record_shapes": False,
            "profile_memory": False, "with_stack": False, "row_limit": 80, "module_paths": []}:
        raise ValueError("K1 diagnostic profile window changed")
    return {"status": "ACCEPTANCE_INPUTS_AVAILABLE", "spec_identity": spec.identity,
        "scientific_acceptance": "NOT_EVALUATED", "replay_readiness": "NOT_APPLICABLE_EXECUTION_DIAGNOSTIC",
        "arms": [{"arm_id": a["arm_id"], "status": "AVAILABLE", "blockers": []} for a in arms]}
