"""Prospective shared-contract reference/candidate release, not future acceptance.

The explicitly authorized pair supplies its sole new-contract reference. Both
terminal outputs must later pass actual acceptance; the planned baseline is not
an accepted bundle and cannot be reused by another candidate before closure.
"""
from dataclasses import replace
from pathlib import Path

from .experiment_family_workflow import _artifact_path, _json
from .family_training_runner import read_recipe, validate_screen_recipe
from .training_reproducibility import sha256_file
from .v4_runtime import read_frozen_state_artifact

FORMAT = "molgap-family-same-run-acceptance-plan-v1"
MODES = {"degree_bond_local_ema999", "degree_local_cycle_common_ema999"}
WALL_CEILING_SECONDS = 8 * 3600
RETENTION_PASSES = 10


def resource_estimate(*, train_step_seconds, forward_seconds, reserved_bytes, total_bytes,
                      completed_passes=0):
    import math
    values = (train_step_seconds, forward_seconds, reserved_bytes, total_bytes)
    if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("Missing native timing or memory observations")
    if reserved_bytes > .85 * total_bytes:
        raise ValueError("Native profile lacks15percent GPU memory headroom")
    remaining = 60 - completed_passes
    if type(completed_passes) is not int or not 0 <= completed_passes < 60:
        raise ValueError("Unsupported resource exposure")
    # Both live and EMA are recorded each pass; all times include H2D/layout.
    estimate = 1.35 * remaining * (781 * train_step_seconds + 2 * 391 * forward_seconds + 5)
    if estimate >= WALL_CEILING_SECONDS - 120:
        raise ValueError("Native estimate exceeds frozen whole-screen wall budget")
    return {"status": "accepted", "estimated_remaining_seconds": estimate,
        "maximum_observed_train_step_seconds": train_step_seconds,
        "maximum_observed_forward_seconds": forward_seconds,
        "reserved_bytes": reserved_bytes, "total_bytes": total_bytes,
        "memory_headroom_fraction": 1 - reserved_bytes / total_bytes,
        "scope": "20 training batches; conservative maximum timings with35percent margin; not a speed claim"}


def bounded_cycle_pair(spec):
    arms = spec.to_dict()["arms"]
    return (len(arms) == 2 and all(a["family"] == {"name": "gptrans_t", "version": "2"}
        and len(a["addons"]) == 1 for a in arms)
        and {a["addons"][0]["name"] for a in arms} == MODES)


def check_same_run_plan(spec, root, plan, *, initial_states=None):
    """Validate exact recipes, role/target/cache pins and explicit pair authority."""
    declaration = spec.to_dict()
    if (not bounded_cycle_pair(spec) or set(plan) != {"format", "spec_identity", "authority", "arms", "audit", "cache_acceptance"}
            or plan["format"] != FORMAT or plan["spec_identity"] != spec.identity):
        raise ValueError("Unsupported same-run shared contract")
    root = Path(root).absolute()
    def pinned(pointer):
        if type(pointer) is not dict or set(pointer) != {"path", "sha256"}:
            raise ValueError("Expected pinned prospective input")
        path = _artifact_path(root, pointer["path"])
        if sha256_file(path) != pointer["sha256"]:
            raise ValueError("Same-run release input changed")
        return path
    authority = _json(pinned(plan["authority"]))
    cache = _json(pinned(plan["cache_acceptance"]))
    if cache.get("accepted") is not True or cache.get("rows") != 200000:
        raise ValueError("Accepted full CPU cycle inputs are required")
    if (authority.get("scope") != "one-new-contract-reference-and-one-candidate" or
            authority.get("user_authorized") is not True or authority.get("extra_arms") != 0 or
            authority.get("seeds") != [42] or authority.get("wall_ceiling_seconds") != WALL_CEILING_SECONDS or
            authority.get("allocated_t4_hours_ceiling") != 16):
        raise ValueError("New-contract pair authority or budget changed")
    paired = declaration["prospective"].get("same_run_replay")
    by_id = {a["arm_id"]: a for a in declaration["arms"]}
    if (not paired or len(paired["candidate_arm_ids"]) != 1 or
            by_id[paired["reference_arm_id"]]["addons"][0]["name"] != "degree_bond_local_ema999"):
        raise ValueError("A sole unchanged same-run reference is required")
    entries = plan["arms"]
    if [entry["arm_id"] for entry in entries] != list(by_id):
        raise ValueError("Same-run plan must cover all arms in order")
    recipes, records = {}, {}
    for entry in entries:
        if set(entry) != {"arm_id", "adapter", "contract", "expected", "target_manifest", "initial_state"}:
            raise ValueError("Unexpected shared same-run arm fields")
        arm_id = entry["arm_id"]
        if entry["adapter"] != "gptrans-session-v1":
            raise ValueError("Same-run shared artifact owner changed")
        record, recipe = read_recipe(pinned(entry["contract"]))
        validate_screen_recipe(spec, arm_id, record)
        if record["acceptance_requirements"] != entry["expected"]:
            raise ValueError("Same-run exposure/role expectations changed")
        target = _json(pinned(entry["target_manifest"]))
        if (target.get("development_target_sha256") != entry["expected"]["target_sha256"] or
                target.get("target_encoding") != "little-endian-float64-contiguous-raw-bytes"):
            raise ValueError("Target encoding/identity is not frozen")
        initial = Path(initial_states[arm_id]) if initial_states is not None else pinned(entry["initial_state"])
        if sha256_file(initial) != entry["initial_state"]["sha256"]:
            raise ValueError("Staged frozen initial bytes changed")
        state, _ = read_frozen_state_artifact(initial,
            expected_state_sha256=recipe.initialization_sha256)
        recipes[arm_id], records[arm_id] = recipe, (record, state)
    reference_id, candidate_id = paired["reference_arm_id"], paired["candidate_arm_ids"][0]
    ref, candidate = recipes[reference_id], recipes[candidate_id]
    from .gptrans_cycle_inputs import FIXED_100K_SHA
    if (ref.dataset.manifest_sha256 != FIXED_100K_SHA or
            candidate.input_sidecar.manifest_sha256 != cache["manifest_sha256"] or
            by_id[reference_id]["data"] != by_id[candidate_id]["data"]):
        raise ValueError("Pair fixed input/role/feature identity differs")
    matched = replace(candidate, mode=ref.mode, initialization_sha256=ref.initialization_sha256, input_sidecar=None)
    if matched != ref:
        raise ValueError("Reference/candidate differ outside the declared cycle mechanism")
    if (ref.seed != 42 or ref.train_rows != 100000 or ref.development_rows != 50000 or
            ref.optimizer_steps != 46860 or ref.ema_decay != .999 or ref.selection != "best-development-ema" or
            ref.learning_rate != .001 or ref.weight_decay != .05 or ref.gradient_clip != 1 or
            not ref.drop_last or ref.batch_size != 128 or ref.schedule.kind != "warmup_cosine_epoch" or
            ref.schedule.total_units != 60 or ref.schedule.warmup_units != 4 or ref.schedule.minimum_lr != 1e-6):
        raise ValueError("Pair exceeds the authorized scientific scope")
    for field in ("acceptance_requirements", "train_role_identity", "development_role_identity"):
        if records[reference_id][0][field] != records[candidate_id][0][field]:
            raise ValueError("Pair role/exposure/target identity differs")
    import torch
    reference_state, candidate_state = records[reference_id][1], records[candidate_id][1]
    if (set(candidate_state) != set(reference_state) | {"cycle_embedding.weight"} or
            candidate_state["cycle_embedding.weight"].shape != (3, 32) or
            candidate_state["cycle_embedding.weight"].count_nonzero() or
            any(not torch.equal(value, candidate_state[name]) for name, value in reference_state.items())):
        raise ValueError("Initial states are not exact parent plus zero cycle embedding")
    audit = _json(pinned(plan["audit"]))
    if (audit.get("roles") != {"100k": [100000, 150000], "500k": [500000, 550000]} or
            audit.get("mode") != "NO_TRAIN" or audit.get("batch_size") != 128 or
            audit.get("precision") != "fp32" or audit.get("tf32") is not False or
            audit.get("run_only_after_training_acceptance") is not True):
        raise ValueError("Full50K post-training audit is not prospectively reserved")
    return {"status": "ACCEPTANCE_INPUTS_AVAILABLE", "spec_identity": spec.identity,
        "reference_status": "PLANNED_NOT_ACCEPTED", "reference_arm_id": reference_id,
        "arms": [{"arm_id": name, "status": "AVAILABLE", "blockers": []} for name in by_id],
        "trainer_execution": "NOT_VERIFIED", "scientific_acceptance": "NOT_EVALUATED"}
