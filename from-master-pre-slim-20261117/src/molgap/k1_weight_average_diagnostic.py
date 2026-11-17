"""Fixed endpoint parameter averaging with separately controlled BN buffers."""
from pathlib import Path
import hashlib
import json
import sys
import time


def mean_parameters(selected, final):
    """Average parameters only; selected buffers retain their exact bytes."""
    import torch
    a, b = dict(selected.named_parameters()), dict(final.named_parameters())
    if a.keys() != b.keys():
        raise ValueError("Parameter names differ")
    if any(a[n].shape != b[n].shape or a[n].dtype != b[n].dtype for n in a):
        raise ValueError("Parameter shape/dtype differs")
    with torch.no_grad():
        for name, value in a.items():
            value.copy_(value * .5 + b[name] * .5)


def run(root: Path, output: Path):
    started, cpu_started = time.perf_counter(), time.process_time()
    root, output = root.resolve(), output.resolve()
    inputs = json.loads((root / "inputs.json").read_text())
    trajectory = json.loads((root / "rml/trajectory.json").read_text())
    if trajectory["record_mode"] != "prospective" or trajectory["decision"]["outcome"] != "ACTIVE":
        raise ValueError("Active prospective record required")
    frozen = Path(inputs["frozen_source_root"]).resolve().parent
    for name, digest in inputs["source_files"].items():
        path = (frozen / name).resolve()
        if not path.is_relative_to(frozen) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen source differs: {name}")
    import molgap
    molgap.__path__.insert(0, str(frozen / "src/molgap"))
    if any(f"molgap.{n}" in sys.modules for n in ("qm9_neural_atom", "gps", "k1_frozen_inference")):
        raise ValueError("Model imported before frozen binding")
    import numpy as np
    import torch
    from torch_geometric.loader import DataLoader
    from .k1_frozen_inference import load_native500k_k1, predict_clean
    from .k1_bn_calibration import recalibrated_batch_norm
    from .k1_screen_training import FORBIDDEN_MODEL_FIELDS
    from .training_reproducibility import atomic_json, atomic_torch_save, configure_fp32_determinism, sha256_file
    from .v4_runtime import state_dict_sha256
    from .router import paired_bootstrap_mean
    if torch.__version__.split("+")[0] != "2.7.1":
        raise ValueError("Expected accepted CPU Torch2.7.1")
    settings = configure_fp32_determinism(42)
    torch.set_num_threads(inputs["cpu_threads"])
    deadline = started + inputs["ceiling_seconds"]
    for name, digest in inputs["executed_source_files"].items():
        if sha256_file(root.parents[1] / name) != digest:
            raise ValueError(f"Executed source differs: {name}")
    artifacts = inputs["artifacts"]
    def load(name, graphs=False):
        item = artifacts[name]
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Artifact differs: {name}")
        return torch.load(item["path"], map_location="cpu", weights_only=not graphs)
    train, dev = load("train_probe", True), load("development_probe", True)
    for values, ids in [(train, inputs["train_source_idx"]), (dev, inputs["development_source_idx"])]:
        if [int(g.source_idx) for g in values] != ids or len(set(ids)) != len(ids):
            raise ValueError("Graph membership differs")
        if any(any(f in g for f in FORBIDDEN_MODEL_FIELDS) or not torch.isfinite(g.y).all() for g in values):
            raise ValueError("Invalid pure2D graphs")
    if len(train) != 16384 or any(not 0 <= i < 500000 for i in inputs["train_source_idx"]) or inputs["development_source_idx"] != list(range(500000, 550000)):
        raise ValueError("Graph role bounds differ")
    def model(name, source, epoch, kind):
        item = artifacts[name]
        return load_native500k_k1(Path(item["path"]), expected_sha256=item["sha256"],
            expected_source_sha256=source, expected_epoch=epoch, checkpoint_kind=kind)
    selected, selected_meta = model("selected_checkpoint", inputs["source_sha256"], 48, "selected")
    final, final_meta = model("final_checkpoint", inputs["final_source_sha256"], 59, "final")
    if selected_meta["mean"] != final_meta["mean"] or selected_meta["std"] != final_meta["std"] or selected_meta["contract"] != final_meta["contract"]:
        raise ValueError("Endpoint contract/target transform differs")
    def normalize(value):
        return {"source_idx": value["source_idx"].long().reshape(-1),
                "target_eV": value.get("target_eV", value.get("target")).reshape(-1),
                "prediction_eV": value.get("prediction_eV", value.get("prediction")).reshape(-1)}
    original, clean, gp = (normalize(load(n)) for n in ("selected_original", "selected_clean_bn", "gptrans_predictions"))
    def aligned(value, target=None):
        if not torch.equal(value["source_idx"], torch.tensor(inputs["development_source_idx"])) or any(len(v) != 50000 or not torch.isfinite(v).all() for v in value.values()):
            raise ValueError("Prediction rows/finiteness differs")
        if target is not None and not torch.equal(value["target_eV"], target):
            raise ValueError("Prediction targets differ")
    for value in (original, clean, gp):
        aligned(value, original["target_eV"])
    def predict(m, values):
        return predict_clean(m, DataLoader(values, batch_size=128, shuffle=False, num_workers=0),
                             mean=selected_meta["mean"], std=selected_meta["std"], device="cpu", deadline=deadline)
    reconstructed, _ = predict(selected, dev[:128])
    reconstruction = float((reconstructed["prediction_eV"] - original["prediction_eV"][:128]).abs().max())
    if not torch.equal(reconstructed["target_eV"], original["target_eV"][:128]) or reconstruction > 1e-4:
        raise ValueError("Selected CPU reconstruction failed")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Do not overwrite a diagnostic attempt")
    output.mkdir(parents=True, exist_ok=True)
    predictions = {"selected_original": original, "selected_clean_bn": clean, "gptrans_ema": gp}
    cases = []
    def case(name, m, calibrated):
        before = state_dict_sha256(m.state_dict())
        if calibrated:
            with recalibrated_batch_norm(m, DataLoader(train, batch_size=128, shuffle=False, num_workers=0),
                    source_idx=inputs["train_source_idx"], source_bounds=(0, 500000), device="cpu", deadline=deadline) as calibration:
                value, timing = predict(m, dev)
                atomic_torch_save(output / f"{name}_buffers.pt", {n:v.detach().clone() for n,v in m.named_buffers()})
        else:
            calibration = None
            value, timing = predict(m, dev)
        aligned(value, original["target_eV"])
        if before != state_dict_sha256(m.state_dict()) or any(p.requires_grad or p.grad is not None for p in m.parameters()):
            raise ValueError("Frozen state/gradient restoration failed")
        error = (value["prediction_eV"].double() - value["target_eV"].double()).abs()
        row = dict(case=name, mae_eV=float(error.mean()), inference=timing, calibration=calibration, state_restored=True)
        predictions[name] = value
        cases.append(row)
        atomic_torch_save(output / f"{name}.pt", value)
        atomic_json(output / f"{name}.json", row)
        print(json.dumps(dict(case=name, mae_eV=row["mae_eV"])), flush=True)
    case("final_original", final, False)
    case("final_clean_bn", final, True)
    selected_buffers = state_dict_sha256(dict(selected.named_buffers()))
    mean_parameters(selected, final)
    if selected_buffers != state_dict_sha256(dict(selected.named_buffers())):
        raise ValueError("Averaging changed buffers")
    atomic_torch_save(output / "average_state.pt", dict(model=selected.state_dict(), mean=selected_meta["mean"], std=selected_meta["std"],
                                                       source_endpoints=artifacts, parameter_mean=[.5,.5], true_stepwise_ema=False))
    case("average_selected_bn", selected, False)
    case("average_clean_bn", selected, True)
    predictions["fixed_clean_prediction_blend"] = dict(clean, prediction_eV=(clean["prediction_eV"] + predictions["final_clean_bn"]["prediction_eV"]) * .5)
    atomic_torch_save(output / "fixed_clean_prediction_blend.pt", predictions["fixed_clean_prediction_blend"])
    errors = {n:(v["prediction_eV"].double() - v["target_eV"].double()).abs().numpy() for n,v in predictions.items()}
    specs = {"primary_average_clean_vs_selected_clean": ("selected_clean_bn", "average_clean_bn"),
             "final_clean_vs_selected_clean": ("selected_clean_bn", "final_clean_bn"),
             "average_raw_vs_selected_raw": ("selected_original", "average_selected_bn"),
             "final_bn_gain": ("final_original", "final_clean_bn"),
             "average_bn_gain": ("average_selected_bn", "average_clean_bn"),
             "fixed_prediction_blend_vs_selected_clean": ("selected_clean_bn", "fixed_clean_prediction_blend"),
             "average_clean_vs_gptrans_ema": ("gptrans_ema", "average_clean_bn")}
    contrasts = {}
    for name, (base, candidate) in specs.items():
        stats = paired_bootstrap_mean(errors[base] - errors[candidate], n_bootstrap=1000, seed=20261008)
        stats["probability_gain_nonnegative"] = 1 - stats.pop("probability_better")
        contrasts[name] = dict(baseline=base, candidate=candidate, gain=stats, sign="positive means lower candidate error")
    report = dict(format="molgap-k1-endpoint-average-diagnostic-v1", status="complete", cases=cases,
        metrics={n:dict(mae_eV=float(v.mean()),rows=len(v)) for n,v in errors.items()}, contrasts=contrasts,
        inputs_sha256=sha256_file(root / "inputs.json"), prospective_sha256=sha256_file(root / "rml/trajectory.json"),
        runtime=dict(torch=torch.__version__, device="cpu", cpu_threads=torch.get_num_threads(), determinism=settings),
        checks=dict(strict_state_loading=True, source_and_artifact_hashes=True, selected_reconstruction=True,
                    exact_rows_targets=True, finite_predictions=True, buffers_not_averaged=True, clean_training_only_bn=True,
                    state_restoration=True, no_training_or_gradients=True, protected_roles_untouched=True),
        reconstruction_max_abs_eV=reconstruction, true_stepwise_ema=False, training_executed=False, gradients_computed=False,
        optimizer_created=False, wall_seconds=time.perf_counter()-started, process_cpu_seconds=time.process_time()-cpu_started)
    if time.perf_counter() > deadline:
        raise TimeoutError("Diagnostic wall ceiling exhausted")
    atomic_json(output / "result.json", report)
    atomic_json(output / "completion.json", dict(complete=True, inputs_sha256=report["inputs_sha256"],
        files={p.name:sha256_file(p) for p in output.iterdir() if p.is_file()}))
    return report
