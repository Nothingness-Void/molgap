"""Freeze CPU scratch states and metadata; no dataset payload or remote access."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

from molgap.common.io import publish_immutable_bytes
from molgap.experiment_execution import build_addon_declaration
from molgap.family_training_kernel import load_family_model
from molgap.family_training_runner import family_arm_bindings, read_recipe
from molgap.gptrans_bond_local import make_bond_local_model
from molgap.training_reproducibility import assert_finite_state_dict, atomic_torch_save
from molgap.v4_runtime import inspect_frozen_state_artifact, state_dict_sha256


def publish(path, value):
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    publish_immutable_bytes(path, payload)
    return hashlib.sha256(payload).hexdigest()


def main():
    root = Path(__file__).resolve().parents[2]
    here = Path(__file__).resolve().parent
    prior = root / "experiments/pcqm_gptrans_bondlocal_pairnorm_lr_500k"
    inputs = Path("D:/molgap-artifacts/runs/pcqm-single-model-total-budget-500k-s42/v1/inputs")
    inputs.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    template = json.loads((prior / "reference_recipe.json").read_bytes())
    declaration = json.loads((prior / "data_declaration.json").read_bytes())
    manifest = {"seed": 42, "kind": "scratch", "device": "cpu", "arms": {}}
    for arm_id, shape, layers, nodes, pairs, heads, count in (
            ("a_12x384", "12x384", 12, 384, 48, 12, 12_478_993),
            ("b_16x256", "16x256", 16, 256, 32, 8, 7_700_481)):
        model = make_bond_local_model(num_layers=layers, node_channels=nodes,
            pair_channels=pairs, num_heads=heads, pair_update_norm=True)
        assert sum(p.numel() for p in model.parameters()) == count
        state = model.state_dict()
        assert all(value.device.type == "cpu" for value in state.values())
        assert_finite_state_dict(state, label=arm_id)
        digest = state_dict_sha256(state)
        initial = inputs / (arm_id + ".pt")
        if not initial.exists():
            atomic_torch_save(initial, {"model_state": state, "state_sha256": digest})
        identity = inspect_frozen_state_artifact(initial, expected_state_sha256=digest)
        recipe = deepcopy(template)
        recipe.update(mode="degree_bond_local_pairnorm_" + shape + "_ema999",
                      initialization_sha256=digest)
        recipe_path = here / (arm_id + "_recipe.json")
        recipe_sha = publish(recipe_path, recipe)
        _, typed = read_recipe(recipe_path)
        loaded, _ = load_family_model(typed, initial)
        assert state_dict_sha256(loaded.state_dict()) == digest
        bindings = family_arm_bindings(typed)
        spec = deepcopy(declaration)
        spec.update(experiment_id="pcqm_single_model_total_budget",
                    logical_run_id="declaration-only-single-model-" + arm_id)
        arm = spec["arms"][0]
        arm.update(arm_id=arm_id, base=bindings["base"], initialization={
            "kind": "frozen_state", "seed": 42, "state_sha256": digest},
            addons=[build_addon_declaration(("gptrans_t", "2"),
                "gptrans_bondlocal_pairnorm_" + shape, repo_root=root)])
        arm["training"].update({key: bindings[key] for key in ("objective", "sampler", "transform")})
        arm["training"]["recipe"]["sha256"] = recipe_sha
        spec["prospective"]["arms"][0].update(arm_id=arm_id,
            output="experiments/pcqm_single_model_total_budget/declaration_only/" + arm_id,
            plan_spec_ref=recipe_path.relative_to(root).as_posix(),
            plan_spec_sha256=recipe_sha, trajectory_id="declaration-only-" + arm_id)
        publish(here / (arm_id + "_data_declaration.json"), spec)
        manifest["arms"][arm_id] = {**identity, "path": initial.as_posix(),
            "parameters": count, "recipe_sha256": recipe_sha}
    publish(here / "initialization_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
