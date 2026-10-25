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


def freeze_scale_anchor(repo_root, *, base, source_spec_ref, source_contract_ref,
                        source_arm_id, accepted_metadata_ref, accepted_input_ref,
                        target_manifest):
    """Rebind an unchanged retained model to independently accepted500K inputs.

    This is input provenance, never a trained reference. The normal compiler
    still owns prospective planning, release, packaging and the actual run.
    """
    from .common.io import publish_immutable_bytes
    from .common.wire import canonical_fingerprint
    from .experiment_input import _read_scientific_recipe
    from .experiment_spec import ExperimentSpec
    from .family_training_data import DatasetSelection
    from .family_training_runner import build_screen_recipe, family_arm_bindings
    from .modern2d_training_release import CONFIGURED_PLAN_FORMAT
    from .modern2d_training import Modern2DFormalTrainingProfile
    from .modern2d_feature_binding import indexed_transform_identity, transform_configuration
    from .research_memory.paths import repo_local_path
    from .research_memory.trace import file_digest
    root = Path(repo_root).absolute()
    load = lambda ref: json.loads(repo_local_path(root, ref).read_bytes())
    prior = load(source_spec_ref)
    arm = deepcopy(next(a for a in prior["arms"] if a["arm_id"] == source_arm_id))
    interface = arm["family"]["name"], arm["family"]["version"]
    source, old = _read_scientific_recipe(repo_local_path(root, source_contract_ref), interface,
        dict(purpose="CONFIGURED_TRANSFER_STUDY", scientific_training_authorized=True,
             source_reissue_authorized=True))
    metadata = load(accepted_metadata_ref)
    binding = load(accepted_input_ref)
    if (old.train_rows != 100000 or metadata.get("status") != "INPUT_METADATA_PREPARED" or
            set(metadata.get("roles", {})) != {"train_500k", "common_dev"} or
            binding["train"]["permit"]["role"] != "train_500k" or
            metadata["training_input"]["sha256"] != file_digest(repo_local_path(root, accepted_input_ref)) or
            (metadata["target_mean_eV"], metadata["target_std_eV"]) != (old.target_mean_eV, old.target_std_eV)):
        raise ValueError("Scale anchor requires accepted500K metadata and unchanged source transform")
    target = load(target_manifest["path"])
    if (file_digest(repo_local_path(root, target_manifest["path"])) != target_manifest["sha256"] or
            target["development_target_sha256"] != source["acceptance_requirements"]["target_sha256"] or
            target["development_source_idx_sha256"] != metadata["roles"]["common_dev"]["source_idx_tensor_sha256"]):
        raise ValueError("Scale anchor target/row identity differs from its retained source")
    model_id = "ppgt_author_v2" if interface[0] == "ppgt_small" else "esa_native_v3"
    model = deepcopy(metadata["models"][model_id])
    if model["model_config"] != source["model_config"]:
        raise ValueError("Scale metadata must use the unchanged actual model configuration")
    steps_per_pass = 500000 // old.batch_size
    recipe = replace(old, train_rows=500000,
        dataset=DatasetSelection("training_input.json", file_digest(repo_local_path(root, accepted_input_ref)), True),
        formal_training=Modern2DFormalTrainingProfile("configured-screen",
            horizon_passes=(old.optimizer_steps + steps_per_pass - 1) // steps_per_pass,
            horizon_steps=old.optimizer_steps, evaluation_every_steps=old.loop_plan.steps_per_pass,
            execution_mode="prepared-v2"))
    train_identity = canonical_fingerprint(binding["train"])
    record = build_screen_recipe(recipe.mode, family=interface, configuration=recipe,
        source_idx_sha256=source["acceptance_requirements"]["source_idx_sha256"],
        target_sha256=source["acceptance_requirements"]["target_sha256"],
        train_role_identity=train_identity, development_role_identity=source["development_role_identity"])
    pins = {}
    def save(name, value=None, raw=None):
        path = repo_local_path(root, base + "/" + name)
        publish_immutable_bytes(path, raw if raw is not None else
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        pins[name] = dict(path=base + "/" + name, sha256=file_digest(path))
        return pins[name]
    save("training_input.json", raw=repo_local_path(root, accepted_input_ref).read_bytes())
    save("contract.json", record)
    model["family"] = list(interface)
    for role, feature in model["feature_bindings"].items():
        feature["transform_code_sha"] = indexed_transform_identity(interface, feature_only=True)
        feature["transform_config_sha"] = canonical_fingerprint(transform_configuration(interface, recipe.model_config))
        model["feature_binding_sha256"][role] = canonical_fingerprint(feature)
    save("training_inputs.json", dict(format="molgap-modern2d-formal-input-metadata-v1",
        scientific_training_authorized=True, target_mean_eV=recipe.target_mean_eV, target_std_eV=recipe.target_std_eV,
        training_input=dict(path="training_input.json", sha256=pins["training_input.json"]["sha256"]),
        models={source_arm_id: model}))
    values = family_arm_bindings(recipe)
    arm.update(base=values["base"], addons=list(recipe.addons), addon_semantics="ordered" if recipe.addons else "baseline")
    arm["training"].update({k: values[k] for k in ("sampler", "transform", "objective")})
    arm["training"]["recipe"]["sha256"] = pins["contract.json"]["sha256"]
    arm["data"]["dataset"]["sha256"] = recipe.dataset.manifest_sha256
    arm["data"]["split"] = dict(name="accepted-indexed-train500k-commondev-v2", version="1", sha256=recipe.dataset.manifest_sha256)
    arm["data"]["feature_sha256"] = canonical_fingerprint(model["feature_binding_sha256"])
    for item in arm["data"]["roles"]:
        role = "train_500k" if item["role"] == "train" else "common_dev"
        item.update(membership_sha256=metadata["roles"][role]["source_idx_tensor_sha256"],
            row_order_sha256=metadata["roles"][role]["source_idx_tensor_sha256"],
            usage_sha256=train_identity if role == "train_500k" else source["development_role_identity"])
    prior["arms"] = [arm]
    prior["prospective"]["arms"] = [a for a in prior["prospective"]["arms"] if a["arm_id"] == source_arm_id]
    prior["prospective"].pop("same_run_replay", None)
    spec = ExperimentSpec(prior)
    save("experiment_spec.json", spec.to_dict())
    save("acceptance_plan.json", dict(format=CONFIGURED_PLAN_FORMAT, spec_identity=spec.identity,
        inputs=pins["training_inputs.json"], arms=[dict(arm_id=source_arm_id,
        contract=pins["contract.json"], target_manifest=target_manifest)]))
    return dict(base=base, pins=pins, scientific_reference_accepted=False, graph_payload_read=False)
