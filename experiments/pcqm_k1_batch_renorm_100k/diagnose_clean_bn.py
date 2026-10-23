"""Bounded selected-LIVE Clean-BN attribution; requires a published prospective."""
from __future__ import annotations

import argparse
from copy import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time


ARMS = ("brn_scale", "brn_affine")


def run(args):
    started, cpu_started = time.perf_counter(), time.process_time()
    root = Path(__file__).resolve().parents[2]
    inputs_path = Path(args.inputs).resolve()
    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    if (inputs["format"] != "molgap-brn-clean-bn-input-v1" or
            inputs["ceiling_seconds"] != 1800 or inputs["cpu_threads"] != 4 or
            not 0 < inputs["prediction_tolerance_eV"] <= 1e-4):
        raise ValueError("Diagnostic contract/budget differs")
    device_name = inputs["device"]
    if device_name not in {"cpu", "cuda:0"}:
        raise ValueError("Only local CPU or one existing desktop GPU is authorized")
    if device_name == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import molgap
    Path(molgap.__file__).resolve().relative_to(root / "src")
    from molgap.training_reproducibility import (
        atomic_json, atomic_torch_save, configure_fp32_determinism, sha256_file,
        deterministic_memory_fill_context,
    )
    from molgap.v4_runtime import state_dict_sha256, normalized_source_sha256
    from molgap.family_training_runner import read_recipe, _assert_target_statistics
    from molgap.family_training_kernel import load_family_model
    from molgap.family_training_data import load_training_data
    from molgap.family_training_core import evaluate_development
    from molgap.family_training_state import FamilyTrainingState, CalibrationPolicy, source_order_sha256
    from molgap.gptrans_execution_hotpath import BoundForeachEMA
    from molgap.k1_execution_layout import cpu_layout_context
    from molgap.k1_execution_backend import execution_context
    from molgap.family_model_ops import k1_batch_to_device, k1_forward
    from molgap.router import paired_bootstrap_mean
    import numpy as np
    import torch
    from torch_geometric.loader import DataLoader

    deadline = started + 1800

    def guard():
        if time.perf_counter() >= deadline:
            raise TimeoutError("Total 1800-second diagnostic budget exhausted")

    trajectory_path = (root / inputs["trajectory_ref"]).resolve()
    trajectory_path.relative_to(root / "experiments")
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    if (trajectory["record_mode"] != "prospective" or trajectory["owner"] != "desktop" or
            trajectory["decision"]["outcome"] != "ACTIVE" or
            trajectory["state_at_start"]["source_config_identity"] != sha256_file(inputs_path)):
        raise ValueError("Matching active desktop prospective required before any forward")
    if not inputs["source_sha256"] or inputs["source_sha256"].get(
            Path(__file__).resolve().relative_to(root).as_posix()) != sha256_file(Path(__file__)):
        raise ValueError("Diagnostic implementation must be frozen in the prospective input")
    for name, digest in inputs["source_sha256"].items():
        path = (root / name).resolve()
        path.relative_to(root)
        if sha256_file(path) != digest:
            raise ValueError("Frozen diagnostic source differs: " + name)
    authority = inputs["role_authority"]
    expected_roles = {"train_source_bounds": [0, 100000],
                      "calibration_source_bounds": [0, 16384],
                      "development_source_bounds": [100000, 150000],
                      "protected_roles_consumed": False}
    if any(authority.get(key) != value for key, value in expected_roles.items()):
        raise ValueError("Diagnostic row authority differs")
    selected_roots = dict(item.split("=", 1) for item in args.selected_root)
    if set(selected_roots) != set(ARMS) or set(inputs["arms"]) != set(ARMS):
        raise ValueError("Both exact BRN arms required")

    def bound(item):
        path = Path(item["path"]).resolve()
        if sha256_file(path) != item["sha256"]:
            raise ValueError("Frozen artifact differs: " + str(path))
        return path

    initial = bound(inputs["initialization"])
    if initial != Path(args.initial_state).resolve():
        raise ValueError("Initialization argument differs from published input")
    strict = json.loads(bound(inputs["strict_verification"]).read_text(encoding="utf-8"))
    if (strict["status"] != "COMPLETE" or strict["executed"] is not True or
            strict["strict_validation"]["status"] != "VERIFIED"):
        raise ValueError("Frozen training strict acceptance is not VERIFIED")
    # All model/data identity checks precede any forward.
    bindings = {}
    for arm in ARMS:
        item = inputs["arms"][arm]
        paths = {name: bound(item[name]) for name in ("recipe", "selected", "predictions")}
        strict_arm = strict["strict_validation"]["arms"][arm]
        if strict_arm["native_qualification"]["status"] != "VERIFIED" or strict_arm["replay"]["status"] != "VERIFIED":
            raise ValueError("Training native qualification/replay not verified")
        observed_artifacts = strict["arms"][arm]["observed"]["artifacts"]
        for name, artifact in (("recipe", "contract"), ("selected", "selected_model"), ("predictions", "predictions")):
            if item[name]["sha256"] != observed_artifacts[artifact]["sha256"]:
                raise ValueError("Diagnostic input differs from accepted training artifact: " + name)
        selected_root = Path(selected_roots[arm]).resolve()
        for name in paths:
            paths[name].relative_to(selected_root)
        contract, recipe = read_recipe(paths["recipe"])
        context = item["context"]
        accepted = json.loads(bound(item["accepted_evidence"]).read_text(encoding="utf-8"))
        terminal = json.loads(bound(item["accepted_trajectory"]).read_text(encoding="utf-8"))
        accepted_context = strict["arms"][arm]["context"]
        if any(context[key] != value for key, value in accepted_context.items() if key != "platform_version"):
            raise ValueError("Producer context differs from accepted training context")
        if terminal["decision"]["outcome"] == "ACTIVE" or not accepted.get("evidence_id"):
            raise ValueError("Accepted training terminal custody missing")
        if (context["arm_id"] != arm or
                context["training_recipe_sha256"] != normalized_source_sha256(paths["recipe"]) or
                recipe.family != "neural_atom_k1" or recipe.selection_weights != "live" or
                recipe.train_rows != 100000 or recipe.development_rows != 50000 or
                recipe.batch_size != 128 or recipe.clean_bn is not None or
                contract["train_role_identity"] != authority["training_role_identity"]):
            raise ValueError("Frozen selected-LIVE recipe/context/role differs: " + arm)
        selected = torch.load(paths["selected"], map_location="cpu", weights_only=True)
        retained = torch.load(paths["predictions"], map_location="cpu", weights_only=True)
        if (selected["context"] != context or retained["context"] != context or
                selected["weights"] != "live" or selected["selection"] != recipe.selection):
            raise ValueError("Selected checkpoint/prediction context differs: " + arm)
        expected_ids = torch.arange(100000, 150000, dtype=torch.int64)
        if (retained["source_idx"].dtype != torch.int64 or
                not torch.equal(retained["source_idx"], expected_ids) or any(
                retained[key].shape != (50000,) or not torch.isfinite(retained[key]).all()
                for key in ("prediction_eV", "target_eV"))):
            raise ValueError("Retained predictions are not the finite ordered development role")
        bindings[arm] = recipe, selected, retained
    if bindings[ARMS[0]][0].dataset != bindings[ARMS[1]][0].dataset:
        raise ValueError("Arm dataset identities differ")
    torch.set_num_threads(4)
    gpu_started = time.perf_counter() if device_name == "cuda:0" else None
    configure_fp32_determinism(42)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested local desktop GPU unavailable")
    data = load_training_data(Path(args.data_root), bindings[ARMS[0]][0].dataset)
    if (data.train_source_start != 0 or data.development_source_start != 100000 or
            data.train_rows != 100000 or data.development_rows != 50000):
        raise ValueError("Accepted data escaped predeclared roles")
    guard()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"format": "molgap-brn-clean-bn-result-v1", "status": "running",
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "inputs_sha256": sha256_file(inputs_path), "prospective_sha256": sha256_file(trajectory_path),
              "training_executed": False, "optimizer_created": False, "gradients_computed": False,
              "runtime": {"device": device_name, "torch": str(torch.__version__), "cpu_threads": 4,
                          "hardware": "cpu" if device.type == "cpu" else torch.cuda.get_device_name(device)},
              "roles": {"training_labels_decoded": [0, 100000], "calibration_features": [0, 16384],
                        "calibration_labels_used": False, "development": [100000, 150000],
                        "development_selection_used_previously": True, "protected_roles_consumed": False},
              "arms": {}}

    def save_report():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        report["wall_seconds"] = time.perf_counter() - started
        report["process_cpu_seconds"] = time.process_time() - cpu_started
        report["device_seconds"] = time.perf_counter() - gpu_started if gpu_started is not None else 0.0
        report["device_cost_semantics"] = "one_local_gpu_process_allocation_window" if device.type == "cuda" else "not_applicable_cpu_only"
        atomic_json(output / "result.json", report)

    try:
        for arm in ARMS:
            guard()
            recipe, selected, retained = bindings[arm]
            _assert_target_statistics(recipe, data)
            if str(torch.__version__).split("+", 1)[0] != recipe.torch_version:
                raise ValueError("Local Torch differs from frozen recipe pin")
            model, _ = load_family_model(recipe, initial)
            model.load_state_dict(selected["model"], strict=True)
            model.to(device)
            original_identity = state_dict_sha256(model.state_dict())
            snapshot = BoundForeachEMA(model)
            owner = FamilyTrainingState(model, snapshot,
                clean_factory=lambda: load_family_model(recipe, initial)[0])
            policy = CalibrationPolicy(recipe.dataset.manifest_sha256, authority["training_role_identity"],
                source_order_sha256(torch.arange(16384, dtype=torch.int64)), 128, 128)

            def forward(current, cpu_batch):
                guard()
                borrowed = copy(cpu_batch)
                with deterministic_memory_fill_context(enabled=(recipe.k1_execution.evaluation_memory_fill
                        if recipe.k1_execution is not None else False)), \
                        cpu_layout_context(current, borrowed, device), \
                        execution_context(current, borrowed, device, recipe.k1_execution):
                    value = k1_forward(current, k1_batch_to_device(borrowed, device, training=False))
                guard()
                return value

            def loader(dataset):
                return DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0,
                                  generator=torch.Generator().manual_seed(recipe.seed))

            def evaluate(current):
                return evaluate_development(current, loader(data.development), forward=forward,
                    rows=50000, source_start=100000, mean=recipe.target_mean_eV, std=recipe.target_std_eV)

            original = owner.evaluate("live", evaluate)
            if not torch.equal(original["target_eV"], retained["target_eV"]):
                raise ValueError("Reproduced development target encoding differs")
            error = float((original["prediction_eV"].double() - retained["prediction_eV"].double()).abs().max())
            if error > inputs["prediction_tolerance_eV"]:
                raise ValueError("Original LIVE predictions exceed declared reproduction tolerance")
            atomic_torch_save(output / f"{arm}_original.pt", original)
            calibration = owner.calibrate(policy, loader(data.train), forward=forward,
                source_indices=lambda batch: batch.source_idx.reshape(-1))
            calibrated = owner.evaluate("clean-ema", evaluate, calibration_policy=policy)
            if not torch.equal(calibrated["target_eV"], original["target_eV"]):
                raise ValueError("Calibrated target identity differs")
            atomic_torch_save(output / f"{arm}_calibrated.pt", calibrated)
            atomic_torch_save(output / f"{arm}_calibrated_state.pt", owner.calibration_state_dict())
            if state_dict_sha256(model.state_dict()) != original_identity:
                raise ValueError("Original selected LIVE model changed")
            report["arms"][arm] = {"original_mae_eV": original["mae_eV"],
                "retained_mae_eV": float((retained["prediction_eV"].double() - retained["target_eV"].double()).abs().mean()),
                "calibrated_mae_eV": calibrated["mae_eV"], "reproduction_max_abs_eV": error,
                "selected_live_state_sha256": original_identity, "calibration": calibration,
                "calibration_source_weights": "selected-live snapshot; no EMA updates"}
            save_report()
            del owner, snapshot, model, original, calibrated
        contrasts = {}
        for state in ("original", "calibrated"):
            values = [torch.load(output / f"{arm}_{state}.pt", weights_only=True) for arm in ARMS]
            if not torch.equal(values[0]["target_eV"], values[1]["target_eV"]):
                raise ValueError("Arm targets differ")
            delta = ((values[1]["prediction_eV"].double() - values[1]["target_eV"].double()).abs() -
                     (values[0]["prediction_eV"].double() - values[0]["target_eV"].double()).abs()).numpy()
            contrasts[state] = paired_bootstrap_mean(np.asarray(delta), n_bootstrap=2000, seed=42)
            guard()
        report.update(status="complete", contrasts_affine_minus_scale=contrasts,
                      ended_at_utc=datetime.now(timezone.utc).isoformat(),
                      checks={"frozen_inputs": True, "exact_roles": True, "original_reproduced": True,
                              "finite_aligned_predictions": True, "selected_live_preserved": True,
                              "calibration_bn_buffers_only": True, "protected_roles_untouched": True})
        save_report()
        guard()
        atomic_json(output / "completion.json", {"complete": True, "inputs_sha256": sha256_file(inputs_path),
            "files": {path.name: sha256_file(path) for path in output.iterdir() if path.is_file()}})
        return report
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        save_report()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--selected-root", action="append", required=True)
    parser.add_argument("--initial-state", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "wall_seconds": result["wall_seconds"], "output": args.output}))


if __name__ == "__main__":
    main()
