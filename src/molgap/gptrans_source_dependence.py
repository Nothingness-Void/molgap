"""Read-only GPA/source and triplet-gate observations on fixed internal panels."""
from contextlib import contextmanager
import json
from pathlib import Path

from .training_reproducibility import atomic_json, atomic_torch_save, sha256_file
from .gptrans_portability import verify_file

BASE = "experiments/pcqm_gptrans_source_dependence"
PROFILE = "gptrans-source-dependence-v1"
KERNEL = "kaseichou/molgap-gptrans-source-dependence-s42"
DATASET = "kaseichou/molgap-gptrans-source-dependence-inputs"
MODELS = ("parent", "triplet")
ROLES = ("original_100k", "unseen_500k")


def panel_indices(role):
    import numpy as np
    from .gptrans_bottleneck import portability_indices
    # Later observations remain within an already scored panel for reproduction.
    population = np.arange(50000) if role == "original_100k" else portability_indices()
    if role not in ROLES:
        raise ValueError("Unknown diagnostic role")
    return np.sort(population[np.random.RandomState(42).choice(len(population), 512, replace=False)])


def validate_release_contract(contract, release, inventory, metadata):
    expected = json.loads((Path(__file__).resolve().parents[2] / BASE / "contract.json").read_text(encoding="utf-8"))
    if contract != expected or contract["workers"] != ["sources"] or contract["optimizer_steps"] != 0:
        raise ValueError("Source observation differs from frozen authority")
    for name, spec in {**contract["model_assets"], **contract["reference_payloads"]}.items():
        if release["files"].get(name) != spec["sha256"]:
            raise ValueError("Frozen source assay asset differs")
    for name, digest in contract["source_identities"].items():
        if inventory.get(name, {}).get("sha256") != digest:
            raise ValueError("Frozen model implementation differs")
    if metadata["id"] != KERNEL or metadata["dataset_sources"] != [DATASET,
            "kaseichou/pcqm4mv2-ogb-fixed-100k-v1", "kaseichou/pcqm4mv2-ogb-fixed-500k-scnet-v1"]:
        raise ValueError("Frozen diagnostic mounts differ")


def source_statistics(probabilities, valid):
    """B,H,Q,S distributions; separate virtual query and real-query averages."""
    import torch
    p = probabilities.detach().double()
    support = valid[:, None, None, :]
    p = p.masked_fill(~support, 0)
    real = valid.clone()
    real[:, 0] = False
    count = valid.sum(1).double()
    real_p = p.masked_fill(~real[:, None, None, :], 0)
    real_mass = real_p.sum(-1)
    conditional = real_p / real_mass.clamp_min(1e-30)[..., None]
    entropy = -(p * p.clamp_min(1e-300).log()).sum(-1)
    conditional_entropy = -(conditional * conditional.clamp_min(1e-300).log()).sum(-1)
    measures = dict(entropy=entropy, maximum_source_mass=p.max(-1).values,
        virtual_source_mass=p[..., 0], effective_source_fraction=entropy.exp()/count[:, None, None],
        real_conditional_entropy=conditional_entropy,
        real_conditional_maximum=conditional.max(-1).values,
        real_conditional_effective_fraction=conditional_entropy.exp()/(count-1).clamp_min(1)[:, None, None],
        real_mass_zero=(real_mass <= 1e-30).double())
    result = {}
    for group, query_mask in (("real_query", real), ("virtual_query", valid & ~real)):
        mask = query_mask[:, None, :]
        for key, value in measures.items():
            if key.startswith("real_conditional"):
                active = mask & (real_mass > 1e-30)
            else:
                active = mask
            result[group + "/" + key] = (value.masked_fill(~active, 0).sum(-1) / active.sum(-1).clamp_min(1)).cpu()
    return result


@contextmanager
def observe_sources(model):
    """Shadow deterministic projections, never replace forward arguments/results."""
    import torch
    from .gptrans import GraphPropagationAttention
    from .gptrans_triplet_communication import TripletLocalBlock, MODES
    from .gptrans_local_relation import valid_pairs
    from .gptrans_bottleneck import _pair_rms
    handles, rows = [], {}
    if model.training:
        raise ValueError("Only frozen eval models can be observed")
    for name, module in model.named_modules():
        if isinstance(module, GraphPropagationAttention):
            def attention_before(m, args, name=name):
                node, pair, padding = args
                b, n, _ = node.shape
                q, k, _ = m.qkv(node).reshape(b, n, 3, m.num_heads, m.head_channels).permute(2, 0, 3, 1, 4).unbind(0)
                logits = (q @ k.transpose(-2, -1)) * m.scale + m.pair_to_attention(pair)
                p = logits.masked_fill(padding, float("-inf")).softmax(-1)
                pair_update = m.attention_to_pair(p + logits)
                pair_p = pair_update.masked_fill(padding, float("-inf")).softmax(-1)
                valid = ~padding[:, 0, 0]
                rows[name] = dict(kind="GPA", node_sources=source_statistics(p, valid),
                                  pair_to_node_sources=source_statistics(pair_p, valid))
            handles.append(module.register_forward_pre_hook(attention_before))
        elif isinstance(module, TripletLocalBlock):
            if module.mode != MODES[0]:
                raise ValueError("Only accepted aggregation is in scope")
            def triplet_before(m, args, name=name):
                _, pair, padding = args[:3]
                valid = valid_pairs(padding)
                nodes = ~padding[:, 0, 0]
                values = m.norm(pair.permute(0, 2, 3, 1)).masked_fill(~valid[..., None], 0)
                row = rows[name] = dict(kind="triplet", input_pair_rms=_pair_rms(pair, nodes))
                for direction, dim in (("in", 2), ("out", 1)):
                    bias = getattr(m, direction + "_bias")(values)
                    gate = getattr(m, direction + "_gate")(values)
                    soft = bias.masked_fill(~valid[..., None], torch.finfo(bias.dtype).min).softmax(dim) * valid[..., None]
                    gated = soft * gate.sigmoid()
                    # Reverse outward axes so source is always the final dimension.
                    soft = soft.permute(0, 3, 1, 2) if direction == "in" else soft.permute(0, 3, 2, 1)
                    gated = gated.permute(0, 3, 1, 2) if direction == "in" else gated.permute(0, 3, 2, 1)
                    row[direction + "_softmax"] = source_statistics(soft, nodes)
                    for group, mask in (("real_query", nodes.clone()), ("virtual_query", nodes.clone())):
                        mask[:, 0] = group == "virtual_query"
                        if group == "virtual_query":
                            mask[:, 1:] = False
                        for metric, value in (("softmax_mass", soft.sum(-1)), ("gated_mass", gated.sum(-1)), ("gated_virtual_mass", gated[..., 0])):
                            row[direction + "/" + group + "/" + metric] = (value.masked_fill(~mask[:, None, :], 0).sum(-1)/mask.sum(1).clamp_min(1)[:, None]).cpu()
                row["_valid"] = nodes
            def output_after(_m, _args, delta, name=name):
                row = rows[name]
                row["return_pair_rms"] = _pair_rms(delta.permute(0, 3, 1, 2), row.pop("_valid"))
                row["return_input_ratio"] = row["return_pair_rms"] / row["input_pair_rms"].clamp_min(1e-30)
            handles.append(module.register_forward_pre_hook(triplet_before))
            handles.append(module.output.register_forward_hook(output_after))
    try:
        yield rows
    finally:
        for handle in handles:
            handle.remove()


def matched_prediction(payload, saved):
    """Panel reproduction against already accepted rows, not a new endpoint gate."""
    import torch
    idx = saved["source_idx"].reshape(-1).long()
    rows = torch.searchsorted(idx, payload["source_idx"])
    if (rows >= len(idx)).any() or not torch.equal(idx[rows], payload["source_idx"]):
        raise ValueError("Panel rows are missing or misaligned")
    target = saved["target_eV"].reshape(-1)[rows]
    expected = saved["prediction_eV"].reshape(-1)[rows]
    if not torch.equal(target, payload["target_eV"]) or not torch.isfinite(payload["prediction_eV"]).all():
        raise ValueError("Panel target/nonfinite mismatch")
    maximum = float((payload["prediction_eV"] - expected).abs().max())
    difference = abs(float((payload["prediction_eV"].double()-target.double()).abs().mean()) - float((expected.double()-target.double()).abs().mean()))
    if maximum > .001 or difference > .0001:
        raise ValueError("Frozen panel prediction did not reproduce")
    return dict(accepted=True, maximum_abs_eV=maximum, mae_difference_eV=difference)


def run_worker(inputs, cache100k, cache500k, output, deadline):
    import torch
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader
    from .pcqm_k1_cross_scale_diagnostic import _accepted_development
    from .gptrans_bottleneck import model_from_asset, _forward, _deadline, _events
    from .gptrans_triplet_portability import load_model
    from .v4_runtime import state_dict_sha256
    from .training_reproducibility import configure_fp32_determinism, build_runtime_manifest
    output.mkdir(parents=True, exist_ok=True)
    settings = configure_fp32_determinism(42)
    if torch.cuda.device_count() != 1 or "T4" not in torch.cuda.get_device_name(0):
        raise ValueError("One isolated T4 required")
    atomic_json(output / "runtime.json", build_runtime_manifest(settings))
    contract = json.loads((inputs / "contract.json").read_text())
    transform = json.loads((inputs / "target_transform.json").read_text())
    events, chunks = [], []
    for role, cache in zip(ROLES, (cache100k, cache500k)):
        _deadline(deadline)
        _events(output, events, role, "prediction_input_and_labels_read")
        graphs = _accepted_development(cache, role)
        loader = DataLoader(Subset(graphs, panel_indices(role).tolist()), batch_size=128, shuffle=False, num_workers=0)
        for arm in MODELS:
            spec = contract["model_assets"][arm + "_best.pt"]
            model = (model_from_asset if arm == "parent" else load_model)(inputs / (arm+"_best.pt"), spec, transform)
            frozen = state_dict_sha256(model.state_dict())
            for number, batch in enumerate(loader):
                _deadline(deadline)
                cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
                with torch.inference_mode():
                    unobserved = _forward(model, batch)
                    with observe_sources(model) as observations:
                        predicted = _forward(model, batch)
                    if not torch.equal(predicted, unobserved):
                        raise ValueError("Observers changed model predictions")
                if not torch.equal(cpu_rng, torch.get_rng_state()) or any(not torch.equal(a,b) for a,b in zip(gpu_rng, torch.cuda.get_rng_state_all())):
                    raise ValueError("Observers changed RNG")
                counts = torch.bincount(batch.batch, minlength=batch.num_graphs).float()
                degree = torch.bincount(batch.edge_index[1], minlength=batch.num_nodes).float()
                payload = dict(source_idx=batch.source_idx.reshape(-1).long().cpu(), target_eV=batch.y.reshape(-1).float().cpu(),
                    prediction_eV=(predicted.float()*transform["std"]+transform["mean"]).cpu(),
                    unobserved_prediction_eV=(unobserved.float()*transform["std"]+transform["mean"]).cpu(),
                    atom_count=counts, directed_bond_count=torch.bincount(batch.batch[batch.edge_index[1]], minlength=batch.num_graphs),
                    mean_degree=torch.bincount(batch.batch[batch.edge_index[1]], minlength=batch.num_graphs).float()/counts,
                    max_degree=torch.stack([degree[batch.batch==i].max() for i in range(batch.num_graphs)]),
                    observations=observations, arm=arm, role=role, observer_bitwise_equal=True,
                    rng_unchanged=True, state_sha256=frozen)
                saved = torch.load(inputs / (arm + ("_original.pt" if role == ROLES[0] else "_later.pt")), map_location="cpu", weights_only=False)
                payload["reproduction"] = matched_prediction(payload, saved)
                filename = f"{arm}_{role}_{number:02d}.pt"
                atomic_torch_save(output / filename, payload)
                chunks.append(dict(file=filename, rows=batch.num_graphs, sha256=sha256_file(output/filename)))
                atomic_json(output / "progress.json", dict(chunks=chunks, completed_panel_batch=filename))
            if state_dict_sha256(model.state_dict()) != frozen:
                raise ValueError("Frozen state changed")
            del model
            torch.cuda.empty_cache()
            print(f"Read-only source assay: {arm}/{role} 512 rows reproduced", flush=True)
        _events(output, events, role, "metric_computed")
        del graphs
    atomic_json(output / "terminal.json", dict(complete=True, training_executed=False, optimizer_steps=0,
        unchanged_state=True, rng_unchanged=True, observer_bitwise_equal=True, strict_ready=False,
        comparison_class="CONTEXT_ONLY", chunks=chunks, protected_roles_read=False,
        peak_memory_bytes=int(torch.cuda.max_memory_allocated())))


def accept(output, inputs):
    """Independent saved-tensor acceptance, without constructing a model."""
    import torch
    contract = json.loads((inputs / "contract.json").read_text())
    release = json.loads((inputs / "audit_release.json").read_text())
    manifest = json.loads((output / "output_manifest.json").read_text())
    if manifest["release"] != release or manifest["status"] != "complete" or manifest["training_executed"] is not False or manifest["optimizer_steps"] != 0:
        raise ValueError("Frozen observation run is not complete")
    for key in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"):
        if manifest[key] is not False:
            raise ValueError("Forbidden role accessed")
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file() and p.name != "output_manifest.json"}
    if actual != set(manifest["files"]):
        raise ValueError("Output inventory differs")
    for name, digest in manifest["files"].items():
        path = (output/name).resolve()
        if not path.is_relative_to(output.resolve()):
            raise ValueError("Unsafe output pointer")
        verify_file(path, digest)
    for name, digest in release["files"].items():
        verify_file(inputs/name, digest)
    ledger = json.loads((output / "allocation_ledger.json").read_text())
    if (ledger["status"] != "complete" or ledger["spec_identity"] != release["identity"] or ledger["allocation_device_count"] not in (1,2)
            or not 0 < ledger["wall_seconds"] <= 1800 or ledger["allocated_device_seconds"] != ledger["allocation_device_count"]*ledger["wall_seconds"]):
        raise ValueError("Native allocation cost invalid")
    directory = output / "sources"
    runtime = json.loads((directory/"runtime.json").read_text())
    if (runtime["accelerator"]["device_count_visible"] != 1 or "T4" not in runtime["accelerator"]["name"]
            or runtime["determinism"]["precision"] != "fp32" or runtime["determinism"]["tf32_enabled"] is not False
            or runtime["determinism"]["deterministic_algorithms"] is not True):
        raise ValueError("Runtime scope invalid")
    terminal = json.loads((directory/"terminal.json").read_text())
    if not all(terminal[k] is True for k in ("complete", "unchanged_state", "rng_unchanged", "observer_bitwise_equal")) or terminal["strict_ready"] is not False or terminal["training_executed"] is not False:
        raise ValueError("Observation terminal invalid")
    events = json.loads((directory/"role_events.json").read_text())
    if [(r["role"],r["event"]) for r in events] != [(role,event) for role in ROLES for event in ("prediction_input_and_labels_read", "metric_computed")]:
        raise ValueError("Role history incomplete")
    expected_files = {f"{arm}_{role}_{n:02d}.pt" for arm in MODELS for role in ROLES for n in range(4)}
    if {r["file"] for r in terminal["chunks"]} != expected_files or len(terminal["chunks"]) != 16:
        raise ValueError("Incomplete observed panels")
    panels = {}
    for arm in MODELS:
        panels[arm] = {}
        for role in ROLES:
            parts = []
            for n in range(4):
                name = f"{arm}_{role}_{n:02d}.pt"
                row = next(r for r in terminal["chunks"] if r["file"] == name)
                verify_file(directory/name, row["sha256"])
                payload = torch.load(directory/name, map_location="cpu", weights_only=False)
                if row["rows"] != 128 or payload["arm"] != arm or payload["role"] != role or payload["state_sha256"] != contract["model_assets"][arm+"_best.pt"]["state_sha256"]:
                    raise ValueError("Observation chunk identity differs")
                if not torch.equal(payload["unobserved_prediction_eV"], payload["prediction_eV"]):
                    raise ValueError("Saved observer/no-observer predictions differ")
                saved = torch.load(inputs/(arm+("_original.pt" if role == ROLES[0] else "_later.pt")), map_location="cpu", weights_only=False)
                matched_prediction(payload, saved)
                _validate_observations(payload, arm)
                parts.append(payload)
            ids = torch.cat([p["source_idx"] for p in parts])
            if not torch.equal(ids, torch.tensor(panel_indices(role)+(100000 if role==ROLES[0] else 500000))):
                raise ValueError("Input-only panel membership differs")
            panels[arm][role] = dict(rows=512, mean_atom_count=float(torch.cat([p["atom_count"] for p in parts]).mean()),
                observations=_summarize(parts), prediction_reproduction="accepted", row_selection="predeclared-input-only-reused-development")
    return dict(accepted=True, experiment_purpose="NO_TRAIN", comparison_class="CONTEXT_ONLY", strict_ready=False,
        training_replay_ready=False, local_model_inference_executed=False, automatic_training_released=False,
        panels=panels, native_cost=ledger)


def _flatten(value, prefix=""):
    import torch
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _flatten(item, prefix+"/"+key)


def _validate_observations(payload, arm):
    import torch
    observations = payload["observations"]
    if len([r for r in observations.values() if r["kind"] == "GPA"]) != 12 or len([r for r in observations.values() if r["kind"] == "triplet"]) != (4 if arm=="triplet" else 0):
        raise ValueError("Observer layer coverage incomplete")
    for name, tensor in _flatten(observations):
        if tensor.shape[0] != 128 or not torch.isfinite(tensor).all() or (tensor < -1e-8).any():
            raise ValueError("Nonfinite/negative observation: "+name)
        if any(k in name for k in ("fraction", "mass", "maximum", "real_mass_zero")) and (tensor > 1+1e-6).any():
            raise ValueError("Probability observation out of bounds: "+name)
    if payload["observer_bitwise_equal"] is not True or payload["rng_unchanged"] is not True:
        raise ValueError("Observer changed execution")


def _summarize(parts):
    import torch
    values = {}
    for part in parts:
        for name, tensor in _flatten(part["observations"]):
            values.setdefault(name, []).append(tensor)
    return {name: dict(mean=float(torch.cat(items).double().mean()), minimum=float(torch.cat(items).min()),
                      maximum=float(torch.cat(items).max())) for name, items in values.items()}
