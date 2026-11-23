"""Derive the six P1 recipes, the serial spec and the IMS run spec from the C1 c500 control recipe (run from the repo root).

Only these fields change: mode, seed, initialization hash, readout (final-ema, no in-job development role) and, for P1 and
zeros, the pinned C1 v3 train sidecar. Everything else is the c500 recipe byte for byte (batch 128, lr 1e-3, EMA 0.999,
weight decay 0.05, 60 passes, 234,360 steps, FP32). Writes <arm>_s<seed>/recipe.json, serial_spec.json and run_spec.json.

Budgets (s) are time-skip estimates, not limits. Control: 30,000 = the measured C1 c500 item wall (29,984 s on A100). P1 and
zeros: 30,000 x the local RTX 5060 step-time ratio (step_time_record.json: 1.031 and 1.009) x 1.05, rounded up to 100 s.
"""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
C1 = HERE.parent / "pcqm_gptrans_dft_distance_aux_20261121"
MODES = {"ctl": "degree_bond_local_pair_update_norm_ema999",
         "p1": "degree_bond_local_pair_update_norm_dft_input_p1_ema999",
         "zer": "degree_bond_local_pair_update_norm_dft_input_zeros_ema999"}
PAIRS = {42: ("s44.pt", "ff9496504ff111e0fb94b23a2d62930ebb1d0c027cdd06a0df9697b760d1728b"),
         43: ("s43.pt", "c95da412764ea1b9a701ae9421c167e32a18f9dc423d364aa3d4fe8c0a8d5f5b")}
BUDGET_S = {"ctl": 30_000, "p1": math.ceil(30_000 * 1.031 * 1.05 / 100) * 100, "zer": math.ceil(30_000 * 1.009 * 1.05 / 100) * 100}
RUN_ID = "gptdftp1-chain-20261123"
SIDECAR = json.loads((C1 / "d500_dft_s44" / "recipe.json").read_text())["input_sidecar"]


def build(arm, seed):
    recipe = json.loads((C1 / "c500_gptrans_pun_s44" / "recipe.json").read_text())
    recipe.update(mode=MODES[arm], seed=seed, initialization_sha256=PAIRS[seed][1], selection="final-ema",
                  development_rows=0, development_role_identity=None)
    recipe["dataset"]["development"] = False
    recipe["acceptance_requirements"].update(development_rows=0, source_idx_sha256=None, target_sha256=None)
    if arm != "ctl":
        recipe["input_sidecar"] = {**SIDECAR, "development": False}
    return recipe


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    items = []
    for seed in PAIRS:  # chain order: pair 42 control, P1, zeros, then pair 43
        for arm in MODES:
            path = HERE / f"{arm}_s{seed}" / "recipe.json"
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(build(arm, seed), indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
            init = C1 / "init" / PAIRS[seed][0]
            items.append({"name": f"{arm}-500k-s{seed}", "recipe": path.relative_to(HERE.parents[1]).as_posix(),
                          "recipe_sha256": sha(path), "budget_s": BUDGET_S[arm],
                          "initial_state": init.relative_to(HERE.parents[1]).as_posix(), "initial_state_sha256": sha(init)})
    spec = HERE / "serial_spec.json"
    spec.write_text(json.dumps({"format": "molgap-ims-family-serial-v1", "items": items}, indent=1) + "\n", encoding="utf-8", newline="\n")
    run = json.loads((C1 / "run_spec.json").read_text())
    run.update(run_id=RUN_ID, dataset_ids=["ogb-train-500k-scnet-v1"])
    run["contract"]["path"] = spec.relative_to(HERE.parents[1]).as_posix()
    run["resources"]["train_walltime"] = "24:00:00"
    run["stages"]["train"]["args"][1] = "{code_root}/" + run["contract"]["path"]
    # The scorer needs each item's final EMA and pass log. The launcher hashes these only when the whole chain ends with exit 0.
    run["stages"]["train"]["required_outputs"] = ["training/serial_summary.json"] + [
        f"training/{item['name']}/{name}" for item in items for name in ("selected_model.pt", "train_log.jsonl")]
    (HERE / "run_spec.json").write_text(json.dumps(run, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(items, indent=1))


if __name__ == "__main__":
    main()
