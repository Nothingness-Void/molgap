"""One-off authoring of the seed-45/46 K1 post-BN initial states and their arm inputs.

Ports desktop's pcqm_k1_init_noise_100k construction: torch seed S, fresh
``make_encoder("neural_atom_k1")`` (as ``k1_screen_training.build_initial_state``),
then the frozen preset recipe and data declaration with only the initial-state
identity (and the recipe pin it implies) substituted. Before writing anything it
reproduces seed 42 (preset pin) and desktop's committed seed-43/44 states and
seed-43/44 recipe/data bytes.

Run from the worktree root:
  python platforms/local/run.py --profile <profile> --script experiments/k1_init_noise_server_100k/build_initial_states.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from molgap.qm9_neural_atom import make_encoder
from molgap.training_reproducibility import atomic_torch_save, sha256_file
from molgap.v4_runtime import inspect_frozen_state_artifact, state_dict_sha256

ROOT = Path.cwd()
HERE = "experiments/k1_init_noise_server_100k"
RECIPE = "experiments/pcqm_structural_bias_norm_100k/k1/training_recipe_post_bn.json"
DATA = "experiments/presets/k1-post-bn-100k-data.json"
PRESET = "experiments/presets/k1-post-bn-100k-v1.json"
STATES = "platforms/_records/kaggle/initializations"
PARAMETERS = 3658817
# Desktop's committed experiments/pcqm_k1_init_noise_100k/s43|s44 records.
DESKTOP = {43: {"state": "5b8b6b565b7e73a662091c4265df326d41833559b9a24c391e7ddd28f4b097fd",
                "recipe": "42fe1261e806f321f3d3867f2d257bfa88389bf9b06d9e79588cdf33212fcc43",
                "data": "ddf63b46347c93c7fb8b06d6bb8e36a9ac2c38c7539bf89318486210cbeb7a2c"},
           44: {"state": "7990bcf38530e56f596283f65a404d97dfa7efa0063481e2260d364df3d196c7",
                "recipe": "bde21e9e7761c9723d2a766cd0ba2085fe498ea4ce2954045b0bc4cc93abe518",
                "data": "fc31b094ee4f7d824509049f5595a39db1b34b6fc77823570bca0622f4f853e8"}}
ARMS = {45: "reference", 46: "candidate"}


def build_state(seed):
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(seed)
        model = make_encoder("neural_atom_k1")
    if sum(p.numel() for p in model.parameters()) != PARAMETERS:
        raise ValueError("K1 parameter count changed")
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def derive(seed, state_sha, role):
    """Desktop's byte substitution: initial state, recipe pin, arm ID and role only."""
    base_state = json.loads((ROOT / PRESET).read_text(encoding="utf-8"))["initialization_state_sha256"]
    recipe = (ROOT / RECIPE).read_bytes()
    recipe = recipe.replace(base_state.encode(), state_sha.encode())
    data = (ROOT / DATA).read_bytes().replace(base_state.encode(), state_sha.encode())
    data = data.replace(sha256_file(ROOT / RECIPE).encode(), hashlib.sha256(recipe).hexdigest().encode())
    data = data.replace(b'"post_bn"', f'"init_s{seed}"'.encode())
    if role != "reference":
        data = data.replace(b'"scientific_role":"reference"', f'"scientific_role":"{role}"'.encode())
    return recipe, data


def main():
    preset_state = json.loads((ROOT / PRESET).read_text(encoding="utf-8"))["initialization_state_sha256"]
    if state_dict_sha256(build_state(42)) != preset_state:
        raise ValueError("Seed 42 does not reproduce the preset initialization")
    for seed, expected in DESKTOP.items():
        sha = state_dict_sha256(build_state(seed))
        recipe, data = derive(seed, sha, "candidate")
        observed = {"state": sha, "recipe": hashlib.sha256(recipe).hexdigest(),
                    "data": hashlib.sha256(data).hexdigest()}
        if observed != expected:
            raise ValueError(f"Desktop seed {seed} not reproduced: {observed}")
        print(json.dumps({"seed": seed, "reproduced_desktop": True, **observed}))
    for seed, role in ARMS.items():
        state = build_state(seed)
        sha = state_dict_sha256(state)
        path = ROOT / STATES / f"k1_init_noise_s{seed}.pt"
        if not path.exists():
            atomic_torch_save(path, state)
        inspect_frozen_state_artifact(path, expected_state_sha256=sha)
        recipe, data = derive(seed, sha, role)
        arm = ROOT / HERE / f"init_s{seed}"
        arm.mkdir(parents=True, exist_ok=True)
        (arm / "recipe.json").write_bytes(recipe)
        (arm / "data_declaration.json").write_bytes(data)
        provenance = {"arm_id": f"init_s{seed}", "builder": "torch seed then molgap.qm9_neural_atom.make_encoder",
            "builder_source": HERE + "/build_initial_states.py", "format": "molgap-initialization-provenance-v1",
            "note": "spec seed label stays 42 because the compiler permits only seed 42; sampler/data order are seed 42",
            "reference_seed_42_state_sha256": preset_state, "state_file": f"{STATES}/k1_init_noise_s{seed}.pt",
            "state_file_sha256": sha256_file(path), "state_sha256": sha, "torch_seed": seed}
        (arm / "initialization_provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps({"seed": seed, "state_sha256": sha, "state_file_sha256": provenance["state_file_sha256"]}))


if __name__ == "__main__":
    main()
