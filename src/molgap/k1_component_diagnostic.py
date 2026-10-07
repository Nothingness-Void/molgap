"""Bounded frozen K1 component reliance, never a retrained architecture comparison."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import json
from pathlib import Path
import sys
import time

CASES = (("atom_half", "atom", ()), ("bond_half", "bond", ()),
         ("rwse_half", "rwse", ()), ("edge_memory_reset", "edge_memory", ()),
         *((f"message_{name}_half", "message", layers) for name, layers in
           (("early", (0, 1, 2)), ("middle", (3, 4, 5)), ("late", (6, 7, 8)))),
         *((f"ffn_{name}_half", "ffn", layers) for name, layers in
           (("early", (0, 1, 2)), ("middle", (3, 4, 5)), ("late", (6, 7, 8)))),
         ("slot3_half", "slot", (3,)), ("slot6_half", "slot", (6,)),
         ("slot9_half", "slot", (9,)), ("slot_all_half", "slot", (3, 6, 9)))


@contextmanager
def component_intervention(model, kind, layers=()):
    """Restore every hook even when registration or the forward fails.

    Message hooks scale only neighbor messages; root projection and convolution
    bias remain intact. Memory reset retains each learned update and its norm.
    """
    if any(module.training for module in model.modules()):
        raise ValueError("Component intervention requires clean eval mode")
    handles, initial_edge = [], {}
    def halve(module, inputs, output):
        return output * 0.5
    def halve_message(module, inputs, output):
        return output * 0.5
    def capture_edge(module, inputs, output):
        initial_edge["value"] = output
    def reset_edge(module, inputs):
        if "value" not in initial_edge or inputs[2].shape != initial_edge["value"].shape:
            raise ValueError("Initial bond state missing or misaligned")
        return inputs[0], inputs[1], initial_edge["value"]
    try:
        with ExitStack() as stack:
            if kind in {"atom", "bond", "rwse"}:
                name = {"atom": "node_emb", "bond": "edge_emb", "rwse": "rwse_encoder"}[kind]
                handles.append(getattr(model, name).register_forward_hook(halve))
            elif kind == "edge_memory":
                handles.append(model.edge_emb.register_forward_hook(capture_edge))
                for module in list(model.edge_updates)[1:]:
                    handles.append(module.register_forward_pre_hook(reset_edge))
            elif kind in {"message", "ffn"}:
                if not layers or any(i not in range(9) for i in layers):
                    raise ValueError("Expected declared local block group")
                for i in layers:
                    block = model.local_blocks[i]
                    handles.append(block.conv.register_message_forward_hook(halve_message)
                                   if kind == "message" else block.mlp.register_forward_hook(halve))
            elif kind == "slot":
                from .k1_frozen_inference import scale_slot_return
                stack.enter_context(scale_slot_return(model, layers=tuple(layers), scale=0.5))
            else:
                raise ValueError("Unknown component case")
            yield
    finally:
        for handle in handles:
            handle.remove()


def hook_inventory(model):
    names = ("_forward_hooks", "_forward_pre_hooks", "_message_forward_hooks")
    return {name: {key: len(getattr(module, key, {})) for key in names}
            for name, module in model.named_modules()}


class PassiveObservations:
    """Pooled feature summaries and residual norms; no tensor replacement."""
    def __init__(self, model):
        self.model, self.handles, self.features, self.ratios, self.slot_rows = model, [], {}, {}, {}
        self.mixer_batches = {layer: 0 for layer in (3, 6, 9)}
        for i, block in enumerate(model.local_blocks):
            self.handles.append(block.register_forward_hook(self.node_capture(f"block{i + 1}")))
            self.handles.append(block.mlp.register_forward_hook(self.residual(f"ffn{i + 1}")))
            self.handles.append(block.conv.register_forward_hook(self.residual(f"conv{i + 1}")))
        self.handles.append(model.head.register_forward_pre_hook(self.head_input))
        for i in (0, 1, 3, 4):
            self.handles.append(model.head[i].register_forward_hook(self.feature_capture(f"head{i}")))
        for layer in (3, 6, 9):
            self.handles.append(model.neural_atom_mixers[str(layer)].register_forward_hook(self.slot_capture(layer)))

    def feature_capture(self, name):
        def observe(module, inputs, output):
            self.features.setdefault(name, []).append(output.detach().double().cpu())
        return observe

    def head_input(self, module, inputs):
        self.features.setdefault("head_input", []).append(inputs[0].detach().double().cpu())

    def node_capture(self, name):
        def observe(module, inputs, output):
            from torch_geometric.nn import global_mean_pool
            self.features.setdefault(name, []).append(global_mean_pool(output, inputs[2]).detach().double().cpu())
        return observe

    def residual(self, name):
        def observe(module, inputs, output):
            value = inputs[0]
            ratio = output.norm(dim=-1) / value.norm(dim=-1).clamp_min(1e-20)
            self.ratios.setdefault(name, []).append(ratio.detach().double().cpu())
        return observe

    def slot_capture(self, layer):
        def observe(module, inputs, output):
            import torch
            from torch_geometric.nn import global_mean_pool, global_add_pool
            hidden, batch = inputs
            update = output - hidden
            pooled = global_mean_pool(hidden, batch)
            returned = global_mean_pool(update, batch)
            count = torch.bincount(batch)
            values = {"nodes": count, "pooled_hidden_norm": pooled.norm(dim=-1),
                      "pooled_return_norm": returned.norm(dim=-1),
                      "sum_return_norm": global_add_pool(update, batch).norm(dim=-1),
                      "return_hidden_ratio": returned.norm(dim=-1) / pooled.norm(dim=-1).clamp_min(1e-20)}
            if self.mixer_batches[layer] < 8:
                # This extra diagnostic computation is included in baseline timers.
                repeated, diagnostics = module.compute_update(hidden, batch)
                if not torch.allclose(repeated, update, atol=3e-6, rtol=2e-5):
                    raise ValueError("Slot observation disagrees with frozen residual")
                assignment = diagnostics["assignment"][:, 0]
                entropy = -(assignment * assignment.clamp_min(1e-30).log()).sum(-1)
                values.update(entropy=entropy, effective_atoms=entropy.exp(),
                              normalized_entropy=entropy / count.double().log().clamp_min(1e-20))
            self.mixer_batches[layer] += 1
            for name, value in values.items():
                self.slot_rows.setdefault(f"slot{layer}_{name}", []).append(value.detach().double().cpu())
            self.features.setdefault(f"slot{layer}", []).append(global_mean_pool(output, batch).detach().double().cpu())
        return observe

    def remove(self):
        for handle in self.handles:
            handle.remove()

    def summary(self):
        import torch
        def distribution(value):
            value = torch.cat(value).view(-1)
            return {"rows": len(value), "mean": float(value.mean()),
                    "p10_median_p90": value.quantile(torch.tensor([.1, .5, .9], dtype=value.dtype)).tolist()}
        feature_summary = {}
        for name, parts in self.features.items():
            value = torch.cat(parts)
            centered = value - value.mean(0)
            covariance = centered.T @ centered / max(len(value) - 1, 1)
            eigen = torch.linalg.eigvalsh(covariance).clamp_min(0)
            total = eigen.sum()
            probabilities = eigen / total.clamp_min(1e-30)
            unit = value / value.norm(dim=-1, keepdim=True).clamp_min(1e-30)
            pair_cosine = ((unit.sum(0).square().sum() - unit.square().sum()) /
                           max(len(value) * (len(value) - 1), 1))
            feature_summary[name] = {"rows": len(value), "width": value.shape[1],
                "variance_sum": float(total), "participation_rank": float(total.square() / eigen.square().sum().clamp_min(1e-30)),
                "entropy_effective_rank": float((-(probabilities * probabilities.clamp_min(1e-30).log()).sum()).exp()) if total > 0 else 0.,
                "mean_pair_cosine": float(pair_cosine), "mean_squared_pair_distance": float(2 * total),
                "near_constant_features": int((covariance.diag() < 1e-12).sum()),
                "exact_zero_features": int((value == 0).all(0).sum()),
                "activation": "SiLU; exact-zero features are not dead-ReLU evidence"}
        return {"features": feature_summary, "residual_norm_ratios": {k: distribution(v) for k, v in self.ratios.items()},
                "slot": {k: distribution(v) for k, v in self.slot_rows.items()},
                "slot_entropy_scope": "first eight baseline batches per mixer; extra compute included in inference timing"}


def run(root: Path, output: Path):
    started, cpu_started = time.perf_counter(), time.process_time()
    root, output = root.resolve(), output.resolve()
    inputs = json.loads((root / "inputs.json").read_text(encoding="utf-8"))
    trajectory = json.loads((root / "rml/trajectory.json").read_text(encoding="utf-8"))
    if trajectory["record_mode"] != "prospective" or trajectory["decision"]["outcome"] != "ACTIVE":
        raise ValueError("Active prospective record required")
    frozen = Path(inputs["frozen_source_root"]).resolve().parent
    package = frozen / "src/molgap"
    for name, digest in inputs["source_files"].items():
        path = (frozen / name).resolve()
        if not path.is_relative_to(frozen) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen model source differs: {name}")
    import molgap
    sys.path.insert(0, str(frozen / "src"))
    molgap.__path__.insert(0, str(package))
    if any(f"molgap.{name}" in sys.modules for name in ("qm9_neural_atom", "gps", "pcqm_gap_architecture", "k1_frozen_inference")):
        raise ValueError("Frozen model modules must not be imported before source binding")
    import numpy as np
    import torch
    from torch_geometric.loader import DataLoader
    from .k1_frozen_inference import load_native500k_k1, predict_clean
    from .k1_screen_training import FORBIDDEN_MODEL_FIELDS
    from .training_reproducibility import atomic_json, atomic_torch_save, configure_fp32_determinism, sha256_file
    from .v4_runtime import state_dict_sha256
    from .router import paired_bootstrap_mean
    if torch.__version__.split("+")[0] != "2.7.1":
        raise ValueError("Frozen local Torch2.7.1 required")
    settings = configure_fp32_determinism(42)
    torch.set_num_threads(inputs.get("cpu_threads", 4))
    deadline = started + inputs.get("ceiling_seconds", 600)
    def guard():
        if time.perf_counter() >= deadline:
            raise TimeoutError("Component diagnostic wall ceiling exhausted")
    for name, digest in inputs["source_files"].items():
        path = (frozen / name).resolve()
        if not path.is_relative_to(frozen) or sha256_file(path) != digest:
            raise ValueError(f"Frozen model source differs: {name}")
    for name, digest in inputs.get("executed_source_files", {}).items():
        path = Path(name)
        if not path.is_absolute():
            path = root.parents[1] / path
        if sha256_file(path) != digest:
            raise ValueError(f"Diagnostic executable differs: {name}")
    artifacts = inputs["artifacts"]
    def load(name, *, graphs=False):
        guard()
        item = artifacts[name]
        path = Path(item["path"])
        if not path.is_absolute():
            path = root / path
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"Artifact bytes differ: {name}")
        return torch.load(path, map_location="cpu", weights_only=not graphs)
    def graph_ids(values, bounds):
        ids = [int(g.source_idx) for g in values]
        if len(set(ids)) != len(ids) or any(not bounds[0] <= i < bounds[1] for i in ids):
            raise ValueError("Graph role IDs differ")
        if any(any(field in g for field in FORBIDDEN_MODEL_FIELDS) or not torch.isfinite(g.y).all() for g in values):
            raise ValueError("Invalid pure2D graph/label payload")
        return ids
    train, development = load("train_probe", graphs=True), load("development_probe", graphs=True)
    train_ids, dev_ids = graph_ids(train, (0, 500000)), graph_ids(development, (500000, 550000))
    if len(train) != 16384 or train_ids != inputs["train_source_idx"] or dev_ids != inputs["development_source_idx"] or dev_ids != list(range(500000, 550000)):
        raise ValueError("Expected retained16K training and ordered50K development payload")
    rng = np.random.default_rng(inputs.get("dev_sample_seed", 20261009))
    dev_offsets = np.sort(rng.choice(len(development), inputs.get("dev_sample_rows", 4096), replace=False))
    dev = [development[int(i)] for i in dev_offsets]
    def training_part(lower, upper, count):
        eligible = [i for i, source in enumerate(train_ids) if lower <= source < upper]
        chosen = rng.choice(eligible, count, replace=False)
        return sorted((train[int(i)] for i in chosen), key=lambda g: int(g.source_idx))
    prefix = training_part(0, 100000, inputs.get("prefix_train_rows", 1024))
    extension = training_part(100000, 500000, inputs.get("extension_train_rows", 1024))
    checkpoint = artifacts["selected_checkpoint"]
    checkpoint_path = Path(checkpoint["path"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path
    model, meta = load_native500k_k1(checkpoint_path, expected_sha256=checkpoint["sha256"],
        expected_source_sha256=inputs["source_sha256"], expected_epoch=48, checkpoint_kind="selected")
    buffers = load("selected_clean_buffers")
    if "buffers" in buffers:
        buffers = buffers["buffers"]
    current = dict(model.named_buffers())
    bn_names = {name for name in current if name.endswith(("running_mean", "running_var", "num_batches_tracked"))}
    if set(buffers) != bn_names:
        raise ValueError("Expected exact accepted clean BN buffer keys")
    with torch.no_grad():
        for name, value in buffers.items():
            if value.shape != current[name].shape or not torch.isfinite(value).all():
                raise ValueError("Invalid clean BN buffer")
            current[name].copy_(value)
    state_hash, hooks = state_dict_sha256(model.state_dict()), hook_inventory(model)
    def predict(values):
        return predict_clean(model, DataLoader(values, batch_size=128, shuffle=False, num_workers=0),
                             mean=meta["mean"], std=meta["std"], device="cpu", deadline=deadline)
    def normalize(value):
        result = {"source_idx": value["source_idx"].long().view(-1),
                "target_eV": value.get("target_eV", value.get("target")).view(-1),
                "prediction_eV": value.get("prediction_eV", value.get("prediction")).view(-1)}
        if len({len(v) for v in result.values()}) != 1 or not all(torch.isfinite(v).all() for v in result.values()):
            raise ValueError("Reference prediction shape or finiteness differs")
        return result
    accepted, gp = normalize(load("selected_clean_predictions")), normalize(load("gptrans_predictions"))
    def align(reference, prediction):
        by_id = {int(source): i for i, source in enumerate(reference["source_idx"])}
        if len(by_id) != len(reference["source_idx"]):
            raise ValueError("Duplicate reference IDs")
        indices = torch.tensor([by_id[int(i)] for i in prediction["source_idx"]])
        target = reference["target_eV"][indices]
        if not torch.equal(target, prediction["target_eV"]):
            raise ValueError("Aligned target bytes differ")
        return reference["prediction_eV"][indices]
    if (output / "result.json").exists() or (output / "completion.json").exists():
        raise FileExistsError("Completed diagnostic must not be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    observations = PassiveObservations(model)
    try:
        baseline, baseline_timing = predict(dev + prefix + extension)
    finally:
        observations.remove()
    passive = observations.summary()
    ndev = len(dev)
    base_dev = {k: v[:ndev] for k, v in baseline.items()}
    accepted_prediction = align(accepted, base_dev)
    reconstruction = float((accepted_prediction - base_dev["prediction_eV"]).abs().max())
    if reconstruction > 1e-4:
        raise ValueError("Accepted clean prediction reconstruction failed")
    gp_prediction = align(gp, base_dev)
    atomic_torch_save(output / "baseline.pt", baseline)
    base_error = (base_dev["prediction_eV"].double() - base_dev["target_eV"].double()).abs()
    control, _ = predict(dev[:128])
    reports = []
    for name, kind, layers in CASES:
        guard()
        with component_intervention(model, kind, layers):
            prediction, timing = predict(dev)
        if not torch.equal(prediction["source_idx"], base_dev["source_idx"]) or not torch.equal(prediction["target_eV"], base_dev["target_eV"]):
            raise ValueError("Case alignment differs")
        restored, restoration_timing = predict(dev[:128])
        if state_dict_sha256(model.state_dict()) != state_hash or hook_inventory(model) != hooks or not torch.equal(restored["prediction_eV"], control["prediction_eV"]):
            raise ValueError("Frozen state/hooks/control not restored exactly")
        if any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise ValueError("Frozen parameter gradient boundary violated")
        case_error = (prediction["prediction_eV"].double() - prediction["target_eV"].double()).abs()
        improvement = base_error - case_error
        uncertainty = paired_bootstrap_mean(improvement.numpy(), n_bootstrap=1000, seed=20261009)
        # The shared helper's probability_better uses negative deltas; expose the
        # sign explicitly rather than assigning that field a reversed meaning.
        uncertainty["delta_sign"] = "baseline absolute error minus case absolute error; positive improves"
        uncertainty["probability_nonnegative_gain"] = 1 - uncertainty.pop("probability_better")
        row = {"case": name, "kind": kind, "layers": list(layers), "mae_eV": float(case_error.mean()),
               "paired_gain": uncertainty, "prediction_rms_shift_eV": float((prediction["prediction_eV"] - base_dev["prediction_eV"]).square().mean().sqrt()),
               "inference": timing, "restoration_inference": restoration_timing, "state_and_hooks_restored": True,
               "interpretation": "posthoc coadapted frozen sensitivity; not a training benefit"}
        reports.append(row)
        atomic_torch_save(output / f"{name}.pt", prediction)
        atomic_json(output / f"{name}.json", row)
        print(json.dumps({"case": name, "mae_eV": row["mae_eV"]}), flush=True)
    gp_error = (gp_prediction.double() - base_dev["target_eV"].double()).abs()
    nodes = np.array([int(g.num_nodes) for g in dev])
    bonds = np.array([int(g.edge_index.shape[1]) // 2 for g in dev])
    targets = base_dev["target_eV"].double().numpy()
    quartiles = np.quantile(targets, [.25, .5, .75])
    slices = {"atoms_le15": nodes <= 15, "atoms16_25": (nodes >= 16) & (nodes <= 25), "atoms26plus": nodes >= 26,
              "bonds_le15": bonds <= 15, "bonds16_25": (bonds >= 16) & (bonds <= 25), "bonds26plus": bonds >= 26}
    for q in range(4):
        slices[f"target_quartile{q + 1}"] = np.searchsorted(quartiles, targets, side="right") == q
    slice_reports = {name: {"rows": int(mask.sum()), "k1_mae_eV": float(base_error[mask].mean()) if mask.any() else None,
        "gptrans_mae_eV": float(gp_error[mask].mean()) if mask.any() else None} for name, mask in slices.items()}
    params = {name: sum(p.numel() for p in module.parameters()) for name, module in model.named_children()}
    module_params = {name: sum(p.numel() for p in module.parameters(recurse=False)) for name, module in model.named_modules()}
    report = {"format": "molgap-k1-component-diagnostic-v1", "status": "complete", "cases": reports,
        "inputs_sha256": sha256_file(root / "inputs.json"),
        "prospective_sha256": sha256_file(root / "rml/trajectory.json"),
        "checks": {"strict_state_loading": True, "source_and_artifact_hashes": True,
                   "selected_clean_reconstruction": True, "exact_rows_targets": True,
                   "finite_predictions": True, "state_and_hooks_restored": True,
                   "no_training_or_gradients": True, "protected_roles_untouched": True},
        "runtime": {"torch": torch.__version__, "device": "cpu", "cpu_threads": torch.get_num_threads(), "determinism": settings},
        "parameters": params, "direct_module_parameters": module_params, "state_sha256": state_hash,
        "baseline_inference": baseline_timing, "baseline_reconstruction_max_abs_eV": reconstruction,
        "baseline_development_mae_eV": float(base_error.mean()), "gptrans_same_sample_mae_eV": float(gp_error.mean()),
        "training_prefix_mae_eV": float((baseline["prediction_eV"][ndev:ndev + len(prefix)] - baseline["target_eV"][ndev:ndev + len(prefix)]).abs().mean()),
        "training_extension_mae_eV": float((baseline["prediction_eV"][ndev + len(prefix):] - baseline["target_eV"][ndev + len(prefix):]).abs().mean()),
        "sample_source_idx": baseline["source_idx"].tolist(), "passive": passive,
        "structure_slices": slice_reports, "target_quartile_boundaries_eV": quartiles.tolist(),
        "slice_interpretation": "posthoc descriptive slices; overlapping counts; no independent validation",
        "gradients_computed": False, "optimizer_created": False, "training_executed": False,
        "wall_seconds": time.perf_counter() - started, "process_cpu_seconds": time.process_time() - cpu_started}
    guard()
    atomic_json(output / "result.json", report)
    hashes = {path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file() and path.suffix in {".json", ".pt"} and path.name != "completion.json"}
    atomic_json(output / "completion.json", {"complete": True, "files": hashes, "inputs_sha256": sha256_file(root / "inputs.json"),
        "wall_seconds": time.perf_counter() - started})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)


if __name__ == "__main__":
    main()
