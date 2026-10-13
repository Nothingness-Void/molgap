"""Full50K frozen-role inference for the shared local/cycle pair; no optimizer.

The deferred action is packaged before training. Accepted checkpoint identities
are bound only after actual training acceptance. It cannot launch itself.
"""
from pathlib import Path
import json
import time

from .family_training_data import DatasetSelection, resolve_pinned_manifest
from .gptrans_cycle_inputs import FIXED_100K_SHA, FIXED_500K_SHA, wrap_sidecar_role
from .pcqm_fixed_records import graph_records
from .training_reproducibility import atomic_json, atomic_torch_save, sha256_file, configure_fp32_determinism
from .experiment_family_workflow import _artifact_path, tensor_digest
from .v4_runtime import state_dict_sha256


def development_view(root, manifest_sha256):
    """Open only the complete internal-development shards; no training prefix."""
    import torch
    from torch.utils.data import ConcatDataset
    from .packed_graph_dataset import PackedGraphDataset
    if manifest_sha256 not in (FIXED_100K_SHA, FIXED_500K_SHA):
        raise ValueError("Audit dataset is not an accepted fixed identity")
    path = resolve_pinned_manifest(root, DatasetSelection("fixed/manifest.json", manifest_sha256, True))
    manifest = json.loads(path.read_bytes())
    role = manifest["roles"]["development"]
    start = 100000 if manifest_sha256 == FIXED_100K_SHA else 500000
    if role != {"source_idx_start": start, "source_idx_stop": start + 50000, "rows": 50000}:
        raise ValueError("Audit must use the complete fixed50K internal role")
    parts, cursor = [], start
    for record in graph_records(manifest, manifest_sha256):
        if record["role"] != "development":
            continue
        file = _artifact_path(path.parent, record["file"])
        if sha256_file(file) != record["sha256"] or record["source_idx_min"] != cursor:
            raise ValueError("Audit role source bytes/order changed")
        part = PackedGraphDataset(file, mmap=True)
        if (len(part) != record["rows"] or not torch.equal(part._data.source_idx.reshape(-1),
                torch.arange(cursor, cursor + len(part)))):
            raise ValueError("Audit decoded source rows changed")
        parts.append(part)
        cursor += len(part)
    if cursor != start + 50000:
        raise ValueError("Audit role is incomplete")
    return ConcatDataset(parts), start


def retain_predictions(model, dataset, *, start, mean, std, output, device, deadline=None):
    """Save all predictions before explicit target access; no scoring/selection."""
    import torch
    from copy import copy
    from torch_geometric.loader import DataLoader
    from .gptrans_execution_backend import prepare_layout, prepared_forward
    from .gptrans_execution_hotpath import MODEL_FIELDS
    output = Path(output)
    records, cursor = [], start
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0):
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("Deferred audit budget reached; retain completed prediction chunks")
            indices = batch.source_idx.reshape(-1).clone()
            if not torch.equal(indices, torch.arange(cursor, cursor + len(indices))):
                raise ValueError("Audit loader changed canonical rows")
            prepared = prepare_layout(copy(batch)).to(device, *MODEL_FIELDS)
            prediction = (prepared_forward(model, prepared).reshape(-1) * std + mean).float().cpu()
            if not torch.isfinite(prediction).all():
                raise ValueError("Nonfinite audit prediction")
            file = output / f"predictions_{cursor:06d}_{cursor + len(indices):06d}.pt"
            atomic_torch_save(file, {"source_idx": indices, "prediction_eV": prediction})
            records.append({"file": file.name, "sha256": sha256_file(file), "rows": len(indices)})
            cursor += len(indices)
            atomic_json(output / "progress.json", {"status": "partial", "start": start, "stop": cursor, "chunks": records})
    if cursor != start + 50000:
        raise ValueError("Audit predictions do not cover the full role")
    atomic_json(output / "prediction_manifest.json", {"status": "complete", "rows": 50000,
        "source_start": start, "source_stop": cursor, "chunks": records, "labels_scored": False})
    return records


def read_retained_predictions(output, records):
    import torch
    values = []
    for record in records:
        path = _artifact_path(Path(output), record["file"])
        if sha256_file(path) != record["sha256"]:
            raise ValueError("Retained audit prediction bytes changed")
        values.append(torch.load(path, map_location="cpu", weights_only=True)["prediction_eV"])
    return torch.cat(values)


def run_bound_arm(*, binding, input_root, output, device="cuda"):
    """Execute one accepted frozen arm; the platform/controller owns allocation."""
    import torch
    from .gptrans_cycle_pair import construct, MODES, CYCLE
    if binding.get("mode", "").startswith("degree_aggregate_"):
        from .gptrans_research_campaign import construct, MODES, CYCLE
    from .gptrans_execution_backend import bind_model_layout
    began = time.perf_counter()
    output = Path(output)
    if (binding.get("training_independently_accepted") is not True or binding.get("mode") not in MODES or
            binding.get("audit_kind") != "fixed500k-full50k-NO_TRAIN" or
            binding.get("batch_size") != 128 or binding.get("tf32") is not False or
            not 0 < binding.get("maximum_wall_seconds", 0) <= 1800):
        raise ValueError("Frozen audit lacks accepted parents or authorized role scope")
    configure_fp32_determinism(42)
    if str(device) != "cuda" or torch.cuda.device_count() != 1 or "T4" not in torch.cuda.get_device_name(0):
        raise ValueError("Audit requires one isolated native T4 per worker")
    checkpoint = Path(binding["checkpoint"])
    original = Path(binding["original_predictions"])
    for path, field in ((checkpoint, "checkpoint_sha256"), (original, "original_predictions_sha256")):
        if sha256_file(path) != binding[field]:
            raise ValueError("Accepted training parent bytes changed")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    original_payload = torch.load(original, map_location="cpu", weights_only=True)
    model = construct(binding["mode"], state)
    model.load_state_dict(state, strict=True)
    bind_model_layout(model)
    initial_digest = state_dict_sha256(model.state_dict())
    model.to(device).eval()
    sidecar = DatasetSelection(binding["sidecar_manifest"], binding["sidecar_manifest_sha256"], True)
    results = {}
    # The500Krole is not opened until the100Kreproduction gate passes.
    for name, digest in (("100k", FIXED_100K_SHA), ("500k", FIXED_500K_SHA)):
        dataset, start = development_view(input_root, digest)
        if binding["mode"] == CYCLE:
            dataset = wrap_sidecar_role(input_root, sidecar, dataset,
                source_manifest_sha256=digest, start=start, rows=50000)
        destination = output / name
        records = retain_predictions(model, dataset, start=start,
            mean=binding["target_mean_eV"], std=binding["target_std_eV"], output=destination, device=device,
            deadline=began + binding["maximum_wall_seconds"] - 60)
        prediction = read_retained_predictions(destination, records)
        if name == "100k":
            if (not torch.equal(original_payload["source_idx"], torch.arange(start, start + 50000)) or
                    not torch.allclose(prediction, original_payload["prediction_eV"].float(), rtol=0, atol=1e-6)):
                raise ValueError("Frozen100Kprediction reproduction failed;500Krole remains unopened")
        # Explicit label access occurs only after complete prediction retention.
        if time.perf_counter() >= began + binding["maximum_wall_seconds"] - 60:
            raise TimeoutError("Audit budget blocks explicit label scoring")
        target = torch.cat([dataset[i].y.reshape(-1).double() for i in range(len(dataset))])
        results[name] = {"rows": 50000, "source_start": start,
            "source_idx_sha256": tensor_digest(torch.arange(start, start + 50000), role="source_idx"),
            "target_sha256": tensor_digest(target, role="target"),
            "mae_eV": (prediction.double() - target).abs().mean().item(),
            "prediction_manifest_sha256": sha256_file(destination / "prediction_manifest.json")}
        atomic_torch_save(destination / "scored_predictions.pt", {"source_idx": torch.arange(start, start + 50000),
            "prediction_eV": prediction, "target_eV": target})
        del dataset
    if state_dict_sha256(model.state_dict()) != initial_digest:
        raise ValueError("Audit modified model tensors")
    atomic_json(output / "summary.json", {"status": "complete", "mode": "NO_TRAIN", "roles": results,
        "checkpoint_sha256": binding["checkpoint_sha256"], "model_state_sha256": initial_digest,
        "optimizer_executed": False, "backward_executed": False, "selection_used": False,
        "protected_roles_read": False, "wall_seconds": time.perf_counter() - began,
        "allocated_device_hours": None, "allocated_cost_owner": "platform_allocation_ledger"})
    return results
