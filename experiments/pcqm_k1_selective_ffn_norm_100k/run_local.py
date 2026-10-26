"""Run frozen arms sequentially through the shared local adapter, then saved-row analysis."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from molgap.common.paths import MachinePaths
from molgap.experiment_family_workflow import RunContext, inspect_output
from molgap.router import paired_bootstrap_mean
from molgap.training_reproducibility import atomic_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", choices=("v1", "v2"), default="v1")
    args = parser.parse_args()
    config = read(HERE / "input.json")
    paths = MachinePaths.from_file(config["machine_profile"]).outside_checkout(ROOT)
    output = paths.artifact_root / "research" / config["experiment_id"]
    declarations = HERE
    if args.attempt == "v2":
        output = output / "attempt_2"
        declarations = HERE / "attempt_2"
    output.mkdir(parents=True, exist_ok=True)
    results, errors, truth, ids = {}, {}, None, None
    for item in config["arms"]:
        arm = item["arm_id"]
        spec = read(declarations / arm / "experiment_spec.json")
        run = spec["logical_run_id"]
        prepared = paths.run_directory(run, args.attempt) / "prepared"
        execution = output / arm
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1")
        command = [str(paths.python), str(ROOT / "platforms/local/train.py"),
            "--prepared", str(prepared), "--repo-root", str(ROOT),
            "--data-root", str(paths.data_root), "--output", str(execution)]
        print("START", arm, flush=True)
        with (output / (arm + ".log")).open("a", encoding="utf-8") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        folder = execution / arm
        manifest = read(folder / "output_manifest.json")
        contract = read(folder / "training_contract.json")
        inspection = inspect_output(folder, context=RunContext(**manifest["context"]),
                                    expected=contract["acceptance_requirements"])
        atomic_json(output / (arm + "_inspection.json"), inspection)
        if inspection.get("status") != "MECHANICALLY_VERIFIED":
            raise ValueError("Mechanical inspection failed: " + str(inspection))
        saved = torch.load(folder / "development_predictions.pt", map_location="cpu", weights_only=True)
        target = saved["target_eV"].reshape(-1)
        membership = saved["source_idx"].reshape(-1)
        if truth is not None and (not torch.equal(truth, target) or not torch.equal(ids, membership)):
            raise ValueError("Paired development rows/targets differ")
        truth, ids = target, membership
        error = (saved["prediction_eV"].reshape(-1).double() - target.double()).abs()
        errors[arm] = error.numpy()
        results[arm] = {"mae_eV": error.mean().item(), "mechanical": "ACCEPTED",
                        "output": str(folder), "native_cost": manifest["costs"]}
        print("COMPLETE", arm, results[arm]["mae_eV"], flush=True)
    comparisons = {}
    for arm in errors.keys() - {"bn_reference"}:
        delta = errors[arm] - errors["bn_reference"]
        row = paired_bootstrap_mean(delta, n_bootstrap=2000, seed=42)
        row["material_screen_signal"] = row["delta"] <= -0.003 and row["ci95"][1] < 0
        comparisons[arm] = row
    atomic_json(output / "summary.json", {"arms": results, "candidate_minus_bn": comparisons,
        "comparison_status": "LOCAL_SINGLE_SEED_EXPLORATORY",
        "limitations": ["Separate local jobs; not strict replay qualification",
            "Repeatedly consumed internal development role, not independent test",
            "Row bootstrap is not training-seed uncertainty; no automatic scale-up"]})
    print("SUMMARY", str(output / "summary.json"), flush=True)


if __name__ == "__main__":
    main()
