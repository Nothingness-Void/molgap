"""Frozen-model derivative diagnostics; no optimizer, model repair or selection."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import time

from .gptrans_portability import verify_file, check_reproduction, check_rows
from .training_reproducibility import atomic_json, atomic_torch_save, sha256_file
from .research_memory.trace import atomic_write, json_bytes

PROFILE = "gptrans-bottleneck-v1"
VARIANTS = {"reference": "degree_scale_ema999", "local": "degree_bond_local_ema999",
            "transition": "degree_pair_transition_ema999"}
COUNTS = {"reference": 5246817, "local": 5871201, "transition": 5263841}


def validate_release_contract(contract, release, inventory, metadata):
    """Narrow NO_TRAIN profile: arbitrary role/model inventories fail closed."""
    from .gptrans_portability import MODEL_SOURCE
    expected = {"reference_19.pt": ("reference", 19), "reference_59.pt": ("reference", 59),
        "local_19.pt": ("local", 19), "local_59.pt": ("local", 59),
        "local_best.pt": ("local", 41), "transition_best.pt": ("transition", 56)}
    if (contract.get("optimizer_steps") != 0 or contract.get("diagnostic_rows") != 512
            or contract.get("portability_rows") != 10000
            or contract.get("checkpoint_epochs") != [19, 59]
            or contract.get("workers") != ["mechanism", "portability"]
            or contract.get("portability_sampler") != "numpy-RandomState42-choice50000-without-replacement-sorted"
            or contract.get("weight_mode") != "eval-EMA-frozen-no-update"
            or contract.get("protected_roles_read") is not False
            or set(contract.get("source_identities", {})) != {"src/molgap/gptrans.py", "src/molgap/gptrans_capacity.py", "src/molgap/gptrans_pair_transition.py"}
            or set(contract.get("model_assets", {})) != set(expected)
            or set(contract.get("reference_payloads", {})) != {"local_predictions.pt", "reference_predictions.pt", "reference_later.pt"}):
        raise ValueError("Frozen bottleneck diagnostic scope changed")
    for name, (kind, epoch) in expected.items():
        spec = contract["model_assets"][name]
        if (spec["kind"] != kind or spec["epoch"] != epoch or spec["sha256"] != release["files"][name]
                or any(len(spec.get(k, "")) != 64 for k in ("state_sha256", "source_checkpoint_sha256"))):
            raise ValueError("Frozen diagnostic asset identity differs")
    for name, spec in contract["reference_payloads"].items():
        if spec["sha256"] != release["files"][name]:
            raise ValueError("Frozen diagnostic saved prediction identity differs")
    expected_mounts = ["kaseichou/molgap-gptrans-bottleneck-inputs",
        "kaseichou/pcqm4mv2-ogb-fixed-100k-v1", "kaseichou/pcqm4mv2-ogb-fixed-500k-scnet-v1"]
    if metadata["dataset_sources"] != expected_mounts or metadata["id"] != "kaseichou/molgap-gptrans-bottleneck-audit":
        raise ValueError("Frozen diagnostic mounts/owner differ")
    if inventory["src/molgap/gptrans.py"]["sha256"] != MODEL_SOURCE:
        raise ValueError("Frozen GPTrans core changed")
    for path, digest in contract["source_identities"].items():
        if inventory.get(path, {}).get("sha256") != digest:
            raise ValueError("Frozen model implementation differs")


def portability_indices():
    import numpy as np
    return np.sort(np.random.RandomState(42).choice(50000, 10000, replace=False))


def _rms(values, valid):
    import torch
    clean = values.detach().double().masked_fill(~valid[..., None], 0)
    return (clean.square().sum((1, 2)) /
            (valid.sum(1).clamp_min(1) * values.shape[-1])).sqrt().cpu()


def _pair_rms(pair, valid):
    mask = valid[:, :, None] & valid[:, None, :]
    return _rms(pair.permute(0, 2, 3, 1).flatten(1, 2), mask.flatten(1, 2))


@contextmanager
def capture_layers(model):
    """Observe existing tensors without replacing computation or consuming RNG."""
    import torch
    handles, rows = [], {}
    for index, block in enumerate(model.blocks, 1):
        row = rows[index] = {"block": index}
        core = getattr(block, "core", block)
        def before(_module, args, row=row):
            node, pair, padding = args[:3]
            valid = ~padding[:, 0, 0].clone()
            valid[:, 0] = False
            row.update(input=node, pair=pair, valid=valid, input_rms=_rms(node, valid),
                       pair_input_rms=_pair_rms(pair, valid))
            if node.requires_grad:
                node.retain_grad()
        handles.append(block.register_forward_pre_hook(before))
        if hasattr(block, "core"):
            def core_before(_module, args, row=row):
                row["update_rms"] = _rms(args[0] - row["input"], row["valid"])
                row["pair_update_rms"] = _pair_rms(args[1] - row["pair"], row["valid"])
            handles.append(core.register_forward_pre_hook(core_before))
            def returned(_module, _args, value, row=row):
                row["branch_return"] = value
                if value.requires_grad:
                    value.retain_grad()
            handles.append(block.output.register_forward_hook(returned))
            row["parameters"] = tuple(block.output.parameters())
        def after(_module, _args, output, row=row):
            row["output_rms"] = _rms(output[0], row["valid"])
        handles.append(block.register_forward_hook(after))
    try:
        yield rows
    finally:
        for handle in handles:
            handle.remove()


def derivative_probe(model, forward, target):
    """Gap-L1 derivatives in eval mode; weight and RNG identity must survive."""
    import torch
    from .v4_runtime import state_dict_sha256
    if model.training:
        raise ValueError("Diagnostic requires eval mode, never a training step")
    original = state_dict_sha256(model.state_dict())
    rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all() if target.is_cuda else []
    model.zero_grad(set_to_none=True)
    try:
        with torch.enable_grad(), capture_layers(model) as captured:
            prediction = forward().reshape(-1)
            if prediction.shape != target.shape or not torch.isfinite(prediction).all():
                raise ValueError("Derivative prediction identity/nonfinite failure")
            loss = (prediction - target).abs().mean()
            loss.backward()
            result = []
            for row in captured.values():
                branch = row.get("branch_return")
                gradients = [p.grad for p in row.get("parameters", ()) if p.grad is not None]
                norm = sum(float(g.detach().double().square().sum()) for g in gradients) ** .5
                input_grad = row["input"].grad
                result.append(dict(block=row["block"], input_rms=row["input_rms"],
                    output_rms=row["output_rms"],
                    update_rms=row.get("update_rms", torch.zeros_like(row["input_rms"])),
                    pair_input_rms=row["pair_input_rms"],
                    pair_update_rms=row.get("pair_update_rms", torch.zeros_like(row["input_rms"])),
                    input_gradient_rms=(None if input_grad is None else _rms(input_grad, row["valid"])),
                    has_branch=branch is not None, branch_parameter_gradient_l2=norm,
                    branch_return_gradient_l2=(None if branch is None or branch.grad is None
                        else float(branch.grad.detach().double().square().sum().sqrt())),
                    loss_gradient_connected=(bool(gradients) and norm > 0),
                    probe_semantics="eval normalized Gap L1; not historical training gradient or clipping"))
        if state_dict_sha256(model.state_dict()) != original or not torch.equal(torch.get_rng_state(), rng):
            raise ValueError("Derivative probe changed frozen weights/buffers or CPU RNG")
        if cuda_rng and any(not torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all())):
            raise ValueError("Derivative probe changed CUDA RNG")
        return result
    finally:
        model.zero_grad(set_to_none=True)


def model_from_asset(path, spec, transform):
    import torch
    from .gptrans import OGBGPTransTiny
    from .gptrans_capacity import construct as local_model
    from .gptrans_pair_transition import construct as pair_model
    from .v4_runtime import state_dict_sha256
    verify_file(path, spec["sha256"])
    saved = torch.load(path, map_location="cpu", weights_only=False)
    kind = spec["kind"]
    if (saved["variant"] != VARIANTS[kind] or saved["epoch"] != spec["epoch"]
            or saved["state_sha256"] != spec["state_sha256"]):
        raise ValueError("Frozen derived model identity differs")
    if saved["target_stats"] != {"mean_eV": transform["mean"], "sample_std_eV": transform["std"]}:
        raise ValueError("Frozen model target transform differs")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        base = OGBGPTransTiny()
        model = (base if kind == "reference" else local_model(VARIANTS[kind], base.state_dict())
                 if kind == "local" else pair_model(VARIANTS[kind], base.state_dict()))
    model.load_state_dict(saved["model"], strict=True)
    if state_dict_sha256(model.state_dict()) != spec["state_sha256"] or sum(p.numel() for p in model.parameters()) != COUNTS[kind]:
        raise ValueError("Frozen loader/state/parameter count mismatch")
    return model.to("cuda").eval()


def _forward(model, batch):
    # Move only inference-visible pure2D inputs, never cached geometry.
    return model(batch.x.to("cuda"), batch.edge_index.to("cuda"),
                 batch.edge_attr.to("cuda"), batch.batch.to("cuda")).reshape(-1)


def _deadline(deadline):
    if time.time() >= deadline:
        raise TimeoutError("Frozen diagnostic allocation budget exhausted")


def _events(output, events, role, event):
    events.append(dict(role=role, event=event, timestamp=time.time(), selection_used=False))
    atomic_write(output / "role_events.json", json_bytes(events))


def mechanism(inputs, cache, output, contract, deadline):
    import torch
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader
    from .pcqm_k1_cross_scale_diagnostic import _accepted_development
    from .v4_runtime import state_dict_sha256
    events = []
    _events(output, events, "original_100k", "prediction_input_and_labels_read")
    graphs = _accepted_development(cache, "original_100k")
    loader = DataLoader(Subset(graphs, range(512)), batch_size=128, shuffle=False, num_workers=0)
    transform = json.loads((inputs / "target_transform.json").read_text())
    results = []
    for name in ("reference_19.pt", "local_19.pt", "reference_59.pt", "local_59.pt", "transition_best.pt"):
        _deadline(deadline)
        spec = contract["model_assets"][name]
        model = model_from_asset(inputs / name, spec, transform)
        frozen = state_dict_sha256(model.state_dict())
        for number, batch in enumerate(loader):
            _deadline(deadline)
            target = (batch.y.reshape(-1).float().to("cuda") - transform["mean"]) / transform["std"]
            values = derivative_probe(model, lambda: _forward(model, batch), target)
            counts = torch.bincount(batch.batch, minlength=batch.num_graphs).float()
            degree = torch.bincount(batch.edge_index[1], minlength=batch.num_nodes).float()
            maximum = torch.stack([degree[batch.batch == i].max() for i in range(batch.num_graphs)])
            payload = dict(source_idx=batch.source_idx.reshape(-1).long(), atom_count=counts,
                mean_degree=torch.bincount(batch.batch[batch.edge_index[1]], minlength=batch.num_graphs).float()/counts,
                max_degree=maximum, layers=values, model_asset=name,
                source_checkpoint_sha256=spec["source_checkpoint_sha256"], state_sha256=frozen)
            filename = f"{name[:-3]}_batch_{number:02d}.pt"
            atomic_torch_save(output / filename, payload)
            results.append(dict(file=filename, sha256=sha256_file(output/filename), rows=batch.num_graphs))
        if state_dict_sha256(model.state_dict()) != frozen:
            raise ValueError("Derivative execution mutated state")
        del model
        torch.cuda.empty_cache()
        atomic_json(output / "progress.json", dict(chunks=results, completed_asset=name))
        print(f"Mechanism diagnostic accepted unchanged state: {name}", flush=True)
    _events(output, events, "original_100k", "derivative_computed")
    atomic_json(output / "terminal.json", dict(complete=True, chunks=results, training_executed=False,
        optimizer_steps=0, unchanged_state=True, official_validation_role_read=False,
        test_dev_role_read=False, test_challenge_role_read=False))


def portability(inputs, cache100k, cache500k, output, contract, deadline):
    import torch
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader
    from .pcqm_k1_cross_scale_diagnostic import _accepted_development
    from .v4_runtime import state_dict_sha256
    transform = json.loads((inputs / "target_transform.json").read_text())
    model = model_from_asset(inputs/"local_best.pt", contract["model_assets"]["local_best.pt"], transform)
    frozen = state_dict_sha256(model.state_dict())
    events, joined = [], {}
    for role in ("original_100k", "unseen_500k"):
        _deadline(deadline)
        _events(output, events, role, "prediction_input_and_labels_read")
        graphs = _accepted_development(cache100k if role == "original_100k" else cache500k, role)
        offsets = list(range(50000)) if role == "original_100k" else portability_indices().tolist()
        chunks, parts = [], []
        for position in range(0, len(offsets), 5000):
            _deadline(deadline)
            loader = DataLoader(Subset(graphs, offsets[position:position+5000]), batch_size=128,
                                shuffle=False, drop_last=False, num_workers=0, pin_memory=True)
            values, targets, ids = [], [], []
            with torch.inference_mode():
                for batch in loader:
                    _deadline(deadline)
                    values.append((_forward(model, batch).float()*transform["std"]+transform["mean"]).cpu())
                    targets.append(batch.y.reshape(-1).float().cpu())
                    ids.append(batch.source_idx.reshape(-1).long().cpu())
            payload = dict(source_idx=torch.cat(ids), target_eV=torch.cat(targets), prediction_eV=torch.cat(values))
            if not torch.isfinite(payload["prediction_eV"]).all():
                raise ValueError("Nonfinite frozen predictions")
            directory = output/role
            directory.mkdir(parents=True, exist_ok=True)
            filename = f"chunk_{position//5000:02d}.pt"
            atomic_torch_save(directory/filename, payload)
            chunks.append(dict(file=filename, rows=len(payload["source_idx"]), sha256=sha256_file(directory/filename)))
            parts.append(payload)
            atomic_json(directory/"progress.json", dict(chunks=chunks, model_sha256=contract["model_assets"]["local_best.pt"]["sha256"]))
            print(f"Frozen local {role}: {position+len(payload['source_idx'])}/{len(offsets)}", flush=True)
        joined[role] = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
        if role == "original_100k":
            saved = torch.load(inputs/"local_predictions.pt", map_location="cpu", weights_only=False)
            reproduction = check_reproduction(joined[role], saved)
            atomic_json(output/"reproduction.json", reproduction)
        else:
            expected = torch.tensor(portability_indices()+500000, dtype=torch.long)
            if not torch.equal(joined[role]["source_idx"], expected):
                raise ValueError("Frozen random cohort identity differs")
        _events(output, events, role, "metric_computed")
    if state_dict_sha256(model.state_dict()) != frozen:
        raise ValueError("Frozen inference mutated state")
    from .k1_terminal_analysis import paired_saved_errors
    comparisons = {}
    for role, name in (("original_100k", "reference_predictions.pt"), ("unseen_500k", "reference_later.pt")):
        comparisons[role] = paired_saved_errors(torch.load(inputs/name, map_location="cpu", weights_only=False), joined[role])
    atomic_json(output/"terminal.json", dict(complete=True, comparisons=comparisons, training_executed=False,
        optimizer_steps=0, unchanged_state=True, comparison_class="PAIRED_ENDPOINT", strict_ready=False,
        official_validation_role_read=False, test_dev_role_read=False, test_challenge_role_read=False))


def run_worker(task, inputs, cache100k, cache500k, output, deadline):
    import torch
    from .training_reproducibility import configure_fp32_determinism, build_runtime_manifest
    output.mkdir(parents=True, exist_ok=True)
    config = configure_fp32_determinism(42)
    if torch.cuda.device_count() != 1 or "T4" not in torch.cuda.get_device_name(0):
        raise ValueError("One isolated T4 is required per frozen diagnostic worker")
    atomic_json(output/"runtime.json", build_runtime_manifest(config))
    contract = json.loads((inputs/"contract.json").read_text())
    if task == "mechanism":
        mechanism(inputs, cache100k, output, contract, deadline)
    elif task == "portability":
        portability(inputs, cache100k, cache500k, output, contract, deadline)
    else:
        raise ValueError("Unfrozen diagnostic worker")
