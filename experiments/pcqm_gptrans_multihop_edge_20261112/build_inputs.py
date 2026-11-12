"""Deterministically build frozen initial states, arm recipes, data declarations and
experiment inputs for E1 (100K) and E2 (500K, frozen but not submitted).
Run from the worktree root with PYTHONPATH=src.  Re-run after any addon source edit."""
import json
from pathlib import Path

import torch
from torch import nn

from molgap import gptrans_bond_local as g
from molgap.training_reproducibility import sha256_file
from molgap.v4_runtime import normalized_source_sha256, read_frozen_state_artifact, state_dict_sha256

E = Path("experiments/pcqm_gptrans_multihop_edge_20261112")
INIT = Path("D:/molgap-artifacts/initializations/gptrans-multihop-edge-20261112")
OLD = Path("D:/molgap-artifacts/initializations/gptrans-pairnorm-init-noise-500k")
NORM = Path("experiments/pcqm_gptrans_norm_ema_100k_20261104")
NOISE = Path("experiments/pcqm_gptrans_pairnorm_init_noise_500k")
REF_MODE = "degree_bond_local_pair_update_norm_ema999"
CAND_MODE = "degree_bond_local_pair_update_norm_multihop_edge_ema999"
CAND_ADDON = "gptrans_degree_bond_local_pair_update_norm_multihop_edge"


def build(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = g.OGBGPTransTiny()
        with torch.no_grad():
            model.in_degree_encoder.weight.mul_(.0897)
            model.out_degree_encoder.weight.mul_(.0897)
        model.__class__ = g.BondLocalGPTrans
        model.blocks = nn.ModuleList(g.BondLocalBlock(b) for b in model.blocks)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save(state, path):
    sha = state_dict_sha256(state)
    if not path.exists():
        torch.save({"model_state": state, "state_sha256": sha}, path)
    return {"state_sha256": sha, "path": path.as_posix(),
            "file_sha256": read_frozen_state_artifact(path, expected_state_sha256=sha)[1]["file_sha256"]}


def states():
    INIT.mkdir(parents=True, exist_ok=True)
    assert state_dict_sha256(build(42)) == "a65034dae2d01d82eb0074eba8f2ee697d93ca259188ae45a4213e9041d1ed1b"
    out = {}
    for seed in (43, 44, 47, 48):
        ref = build(seed)
        out[f"ref_s{seed}"] = save(ref, INIT / f"ref_s{seed}.pt")
        out[f"mh_s{seed}"] = out[f"ref_s{seed}"]  # same initial state; the addon mounts after load
    for seed in (43, 44):  # must equal the retained 500K init-noise states
        assert out[f"ref_s{seed}"]["state_sha256"] == json.loads((OLD / "states.json").read_text())[str(seed)]["state_sha256"]
    (INIT / "states.json").write_text(json.dumps(out, indent=2))
    return out


def arm_files(folder, arm, template_recipe, template_decl, state, run, candidate):
    recipe = json.loads(Path(template_recipe).read_text(encoding="utf-8"))
    recipe.pop("auxiliary_ema_decays", None)
    recipe["execution_policy"].pop("auxiliary_ema", None)
    recipe["initialization_sha256"] = state["state_sha256"]
    recipe["mode"] = CAND_MODE if candidate else REF_MODE
    folder.mkdir(parents=True, exist_ok=True)
    recipe_path = folder / "recipe.json"
    recipe_path.write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    decl = json.loads(Path(template_decl).read_text(encoding="utf-8"))
    a = decl["arms"][0]
    a["arm_id"], a["scientific_role"] = arm, "candidate" if candidate else "reference"
    a["initialization"]["state_sha256"] = state["state_sha256"]
    a["training"]["recipe"]["sha256"] = normalized_source_sha256(recipe_path)
    if candidate:
        a["addons"] = [{"config": {}, "name": CAND_ADDON, "version": "1",
                        "source_sha256": normalized_source_sha256(Path("src/molgap/gptrans_multihop_edge.py"))}]
    decl["experiment_id"], decl["logical_run_id"] = E.name, run
    decl["prospective"]["arms"][0]["arm_id"] = arm
    (folder / "data_declaration.json").write_text(json.dumps(decl, separators=(",", ":"), sort_keys=True),
                                                  encoding="utf-8", newline="\n")


def main():
    st = states()
    for seed in (43, 44):
        run = f"pcqm-gptrans-multihop-edge-20261112-s{seed}"
        for role in ("ref", "mh"):
            arm = f"{role}_s{seed}"
            arm_files(E / "e1" / arm, arm, NORM / "recipe.json", NORM / "data_declaration.json", st[arm], run,
                      role == "mh")
    for arm, seed in (("ref_s47", 47), ("mh_s47", 47), ("mh_s48", 48)):
        arm_files(E / "e2" / arm, arm, NOISE / "init_s43" / "recipe.json", NOISE / "init_s43" / "data_declaration.json",
                  st[arm], f"pcqm-gptrans-multihop-edge-20261112-e2-s{seed}", arm.startswith("mh"))
    print(json.dumps({k: v["state_sha256"][:12] for k, v in st.items()}))


if __name__ == "__main__":
    main()
