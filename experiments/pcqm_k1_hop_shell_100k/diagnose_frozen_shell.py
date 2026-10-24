"""Run the predeclared shell-mixture intervention through accepted K1 owners."""
from __future__ import annotations

import argparse
from copy import copy
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def run(inputs_path, output):
    started, cpu_started = time.perf_counter(), time.process_time()
    root = Path(__file__).resolve().parents[2]
    from molgap.training_reproducibility import (
        atomic_json, atomic_torch_save, configure_fp32_determinism,
        deterministic_memory_fill_context, sha256_file,
    )
    from molgap.family_training_runner import read_recipe, _assert_target_statistics
    from molgap.family_training_kernel import load_family_model
    from molgap.family_training_data import load_training_data
    from molgap.family_training_core import evaluate_development
    from molgap.family_training_state import FamilyTrainingState
    from molgap.family_model_ops import k1_batch_to_device, k1_forward
    from molgap.k1_execution_layout import cpu_layout_context
    from molgap.k1_execution_backend import execution_context
    from molgap.k1_hop_shell import HopShellResidual
    from molgap.v4_runtime import state_dict_sha256
    import torch
    from torch_geometric.loader import DataLoader

    inputs_path, output = Path(inputs_path).resolve(), Path(output).resolve()
    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    if (not 0 < inputs["ceiling_seconds"] <= 1800 or inputs["prediction_tolerance_eV"] != 1e-4
            or inputs["cpu_threads"] != 4 or inputs["device"] != "cuda:0"):
        raise ValueError("Frozen local diagnostic bounds differ")
    prospective = root / inputs["trajectory_ref"]
    trajectory = json.loads(prospective.read_text(encoding="utf-8"))
    if (trajectory["record_mode"] != "prospective" or trajectory["owner"] != "desktop"
            or trajectory["decision"]["outcome"] != "ACTIVE"
            or trajectory["state_at_start"]["source_config_identity"] != sha256_file(inputs_path)):
        raise ValueError("An active matching prospective must precede inference")
    for name, digest in inputs["source_sha256"].items():
        if sha256_file(root / name) != digest:
            raise ValueError("Diagnostic source differs: " + name)
    if inputs["source_sha256"].get(Path(__file__).relative_to(root).as_posix()) != sha256_file(Path(__file__)):
        raise ValueError("Diagnostic runner is not frozen")

    def bound(item):
        path = Path(item["path"]).resolve()
        if sha256_file(path) != item["sha256"]:
            raise ValueError("Frozen artifact differs: " + str(path))
        return path

    verified = json.loads(bound(inputs["strict_verification"]).read_text(encoding="utf-8"))
    if (verified["status"] != "COMPLETE" or verified["executed"] is not True
            or verified["strict_validation"]["status"] != "VERIFIED"):
        raise ValueError("Strict training acceptance must precede inference")
    arm = "hop_ordered"
    qualification = verified["strict_validation"]["arms"][arm]
    if any(qualification[key]["status"] != "VERIFIED" for key in ("native_qualification", "replay")):
        raise ValueError("Native/replay qualification is not verified")
    paths = {key: bound(inputs[key]) for key in ("recipe", "selected", "predictions", "initialization")}
    observed = verified["arms"][arm]["observed"]["artifacts"]
    for key, artifact in (("recipe", "contract"), ("selected", "selected_model"), ("predictions", "predictions")):
        if inputs[key]["sha256"] != observed[artifact]["sha256"]:
            raise ValueError("Diagnostic does not use the accepted artifact: " + key)
    contract, recipe = read_recipe(paths["recipe"])
    if (recipe.family != "neural_atom_k1" or recipe.selection_weights != "live"
            or recipe.train_rows != 100000 or recipe.development_rows != 50000
            or recipe.batch_size != 128 or recipe.clean_bn is not None):
        raise ValueError("Selected-LIVE recipe differs")
    selected = torch.load(paths["selected"], map_location="cpu", weights_only=True)
    retained = torch.load(paths["predictions"], map_location="cpu", weights_only=True)
    if (selected["context"] != retained["context"] or selected["weights"] != "live"
            or selected["context"]["arm_id"] != arm
            or not torch.equal(retained["source_idx"], torch.arange(100000, 150000))):
        raise ValueError("Selected checkpoint/predictions or rows differ")
    if inputs["roles"] != {"train_labels_decoded": [0, 100000], "development": [100000, 150000],
                           "selection_used": False, "protected_roles_consumed": False}:
        raise ValueError("Declared inference roles differ")
    gpu_started = time.perf_counter()
    torch.set_num_threads(4)
    configure_fp32_determinism(42)
    device = torch.device(inputs["device"])
    if not torch.cuda.is_available():
        raise ValueError("Existing desktop GPU is unavailable")
    runtime = {"device": str(device), "torch": str(torch.__version__), "cpu_threads": 4,
               "hardware": torch.cuda.get_device_name(device)}
    if runtime != inputs["local_inference_runtime"]:
        raise ValueError("Local inference runtime differs from prospective pin")
    output.mkdir(parents=True, exist_ok=False)
    report = {"format": "molgap-k1-frozen-shell-result-v1", "status": "running",
              "inputs_sha256": sha256_file(inputs_path), "prospective_sha256": sha256_file(prospective),
              "runtime": runtime, "training_recipe_torch_version": recipe.torch_version,
              "training_executed": False, "optimizer_created": False, "gradients_computed": False,
              "roles": inputs["roles"], "started_at_utc": datetime.now(timezone.utc).isoformat()}

    def guard():
        if time.perf_counter() - started >= inputs["ceiling_seconds"]:
            raise TimeoutError("Frozen diagnostic wall ceiling reached")

    def save():
        torch.cuda.synchronize(device)
        report.update(wall_seconds=time.perf_counter() - started,
                      process_cpu_seconds=time.process_time() - cpu_started,
                      device_seconds=time.perf_counter() - gpu_started,
                      device_cost_scope="one desktop GPU allocation window; excludes training and queue")
        atomic_json(output / "result.json", report)

    try:
        data = load_training_data(Path(inputs["data_root"]), recipe.dataset)
        _assert_target_statistics(recipe, data)
        model, _ = load_family_model(recipe, paths["initialization"])
        model.load_state_dict(selected["model"], strict=True)
        model.to(device).eval().requires_grad_(False)
        original_state = state_dict_sha256(model.state_dict())
        modules = [module for module in model.modules() if isinstance(module, HopShellResidual)]
        if len(modules) != 3 or not all(module.ordered for module in modules):
            raise ValueError("Expected the three accepted ordered-shell residuals")
        owner = FamilyTrainingState(model, None)

        def forward(current, cpu_batch):
            guard()
            borrowed = copy(cpu_batch)
            config = recipe.k1_execution
            fill = (nullcontext() if config is None else
                    deterministic_memory_fill_context(enabled=config.evaluation_memory_fill))
            with fill, \
                    cpu_layout_context(current, borrowed, device), \
                    execution_context(current, borrowed, device, recipe.k1_execution):
                value = k1_forward(current, k1_batch_to_device(borrowed, device, training=False))
            guard()
            return value

        def evaluate(current):
            loader = DataLoader(data.development, batch_size=128, shuffle=False, num_workers=0,
                                generator=torch.Generator().manual_seed(recipe.seed))
            return evaluate_development(current, loader, forward=forward, rows=50000,
                source_start=100000, mean=recipe.target_mean_eV, std=recipe.target_std_eV)

        original = owner.evaluate("live", evaluate)
        if not torch.equal(original["target_eV"], retained["target_eV"]):
            raise ValueError("Reproduced targets differ")
        error = float((original["prediction_eV"].double() - retained["prediction_eV"].double()).abs().max())
        report["reproduction_max_abs_eV"] = error
        report["original_mae_eV"] = original["mae_eV"]
        save()
        if error > inputs["prediction_tolerance_eV"]:
            raise ValueError("Original predictions exceed predeclared reproduction tolerance")
        atomic_torch_save(output / "original.pt", original)
        try:
            for module in modules:
                module.ordered = False
            mixed = owner.evaluate("live", evaluate)
        finally:
            for module in modules:
                module.ordered = True
        if not torch.equal(mixed["target_eV"], original["target_eV"]):
            raise ValueError("Intervention targets differ")
        if state_dict_sha256(model.state_dict()) != original_state:
            raise ValueError("Frozen weights or buffers changed")
        atomic_torch_save(output / "mixed.pt", mixed)
        delta = ((mixed["prediction_eV"].double() - mixed["target_eV"].double()).abs()
                 - (original["prediction_eV"].double() - original["target_eV"].double()).abs())
        from molgap.router import paired_bootstrap_mean
        report.update(status="complete", mixed_at_ordered_weights_mae_eV=mixed["mae_eV"],
            mixed_minus_original=paired_bootstrap_mean(delta.numpy(), n_bootstrap=2000, seed=42),
            selected_state_sha256=original_state,
            checks={"frozen_inputs": True, "original_reproduced": True, "exact_development_rows": True,
                    "weights_and_buffers_preserved": True, "ordered_flags_restored": True,
                    "no_training_or_selection": True, "protected_roles_untouched": True})
        guard()
        save()
        atomic_json(output / "completion.json", {"complete": True,
            "inputs_sha256": sha256_file(inputs_path),
            "files": {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}})
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        save()
        raise
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.inputs, args.output)
    print(json.dumps({"status": result["status"], "wall_seconds": result["wall_seconds"], "output": args.output}))
