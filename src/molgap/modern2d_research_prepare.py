"""Reissue a model-only anchor using retained accepted input/target declarations.

This freezes fresh metadata, not a trained reference or new graph cache. The
normal scientific-input compiler still owns each actual run's envelopes.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path


def freeze_anchor(repo_root, *, base, family, model_config, initialization_sha256,
                  horizon_passes=20, execution_mode="original"):
    from .common.io import publish_immutable_bytes
    from .common.wire import canonical_fingerprint
    from .experiment_input import _read_scientific_recipe
    from .experiment_spec import ExperimentSpec
    from .family_training_runner import build_screen_recipe, family_arm_bindings
    from .modern2d_training import Modern2DFormalTrainingProfile
    from .modern2d_feature_binding import indexed_transform_identity, transform_configuration
    from .research_memory.paths import repo_local_path
    from .research_memory.trace import file_digest

    root = Path(repo_root).absolute()
    old = "experiments/modern2d_100k_formal_screen/retry_v2"
    old_id = "ppgt_author_v2" if family == "ppgt_small" else "esa_native_v3"
    interface = (family, "3" if family == "ppgt_small" else "4")
    load = lambda name: json.loads(repo_local_path(root, name).read_bytes())
    pins = {}
    def save(name, record=None, raw=None):
        target = repo_local_path(root, base + "/" + name)
        publish_immutable_bytes(target, raw if raw is not None else
            json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        pins[name] = dict(path=base + "/" + name, sha256=file_digest(target))
        return pins[name]
    old_record, baseline = _read_scientific_recipe(repo_local_path(root, old + "/" + old_id + "/contract.json"),
        interface, dict(purpose="CONFIGURED_SCREEN", scientific_training_authorized=True,
                        source_reissue_authorized=True))
    steps = horizon_passes * (baseline.train_rows // baseline.batch_size)
    recipe = replace(baseline, initialization_sha256=initialization_sha256,
        model_config=model_config, optimizer_steps=steps,
        schedule=replace(baseline.schedule, total_units=steps, reference_epochs=horizon_passes),
        checkpoint_every=128, formal_training=Modern2DFormalTrainingProfile(
            purpose="configured-screen", horizon_passes=horizon_passes,
            horizon_steps=steps, execution_mode=execution_mode))
    record = build_screen_recipe("reference", family=interface, configuration=recipe,
        source_idx_sha256=old_record["acceptance_requirements"]["source_idx_sha256"],
        target_sha256=old_record["acceptance_requirements"]["target_sha256"],
        train_role_identity=old_record["train_role_identity"],
        development_role_identity=old_record["development_role_identity"])
    save("contract.json", record)
    save("training_input.json", raw=repo_local_path(root, old + "/training_input.json").read_bytes())
    metadata = load(old + "/training_inputs.json")
    model = deepcopy(metadata["models"][old_id])
    model["model_config"] = record["model_config"]
    for role, feature in model["feature_bindings"].items():
        feature["transform_code_sha"] = indexed_transform_identity(interface, feature_only=True)
        feature["transform_config_sha"] = canonical_fingerprint(transform_configuration(interface, model_config))
        model["feature_binding_sha256"][role] = canonical_fingerprint(feature)
    metadata["models"] = {old_id: model}
    metadata["training_input"] = dict(path="training_input.json", sha256=pins["training_input.json"]["sha256"])
    save("training_inputs.json", metadata)
    prior = load(old + "/experiment_spec.json")
    arm = deepcopy(next(a for a in prior["arms"] if a["arm_id"] == old_id))
    arm["initialization"]["state_sha256"] = initialization_sha256
    arm["training"]["recipe"]["sha256"] = pins["contract.json"]["sha256"]
    bindings = family_arm_bindings(recipe)
    arm["base"] = bindings["base"]
    arm["training"].update({k: bindings[k] for k in ("sampler", "transform", "objective")})
    arm["data"]["feature_sha256"] = canonical_fingerprint(model["feature_binding_sha256"])
    prior["arms"] = [arm]
    prior["prospective"]["arms"] = [a for a in prior["prospective"]["arms"] if a["arm_id"] == old_id]
    spec = ExperimentSpec(prior)
    save("experiment_spec.json", spec.to_dict())
    old_plan = load(old + "/acceptance_plan.json")
    target = next(a for a in old_plan["arms"] if a["arm_id"] == old_id)["target_manifest"]
    save("acceptance_plan.json", dict(format=old_plan["format"], spec_identity=spec.identity,
        inputs=pins["training_inputs.json"], arms=[dict(arm_id=old_id,
            contract=pins["contract.json"], target_manifest=target)]))
    return dict(base=base, interface=list(interface), arm_id=old_id, pins=pins,
                scientific_reference_accepted=False, graph_payload_read=False)
