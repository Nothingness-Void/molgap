"""Frozen-checkpoint relation interventions, never a trainer or new candidate.

The Kaggle entry imports this file beside the immutable training source. No
intervention changes checkpoint tensors. Destructive ablation measures a
trained network's dependence, not the counterfactual value of retraining.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import time
from types import MethodType

REFERENCE = "neural_atom_k1_v4"
VARIANTS = {
    "neural_atom_k1_receiver_pair": ("full", "half", "off", "common_return"),
    "neural_atom_k1_triplet_aggregate": ("full", "half", "off", "triplet_off"),
    "neural_atom_k1_rrwp_pair": ("full", "half", "off", "rrwp_off"),
}
ROLES = {"original_100k": 100000, "unseen_500k": 500000}
MAX_REPRODUCTION_EV = 0.0001
CHUNK_ROWS = 5000


def check_variant(mode, variant):
    if mode not in VARIANTS or variant not in VARIANTS[mode]:
        raise ValueError("Undeclared model/intervention")


def group_masks(conjugated_fraction):
    """Boundaries frozen from the previous descriptive analysis, not refitted."""
    import numpy as np
    values = np.asarray(conjugated_fraction)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Finite per-molecule conjugation fractions required")
    return {"all": np.ones(len(values), dtype=bool),
            "low_conjugation": values < 0.27272728085517883,
            "high_conjugation": values >= 0.7333333492279053,
            "middle_conjugation": (values >= 0.27272728085517883)
                                  & (values < 0.7333333492279053)}


def joined_payload(directory, chunks, start):
    """Hash-check exact frozen-role rows, including optional telemetry."""
    import hashlib
    import torch
    if len(chunks) != 10:
        raise ValueError("Exactly ten retained chunks required")
    parts = []
    for index, entry in enumerate(chunks):
        if entry["file"] != f"chunk_{index:02d}.pt" or entry["rows"] != CHUNK_ROWS:
            raise ValueError("Chunk identity changed")
        path = Path(directory) / entry["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Chunk SHA changed")
        part = torch.load(path, map_location="cpu", weights_only=False)
        expected = torch.arange(start + index*CHUNK_ROWS, start + (index+1)*CHUNK_ROWS)
        if not torch.equal(part["source_idx"], expected):
            raise ValueError("Missing, duplicate, or reordered source rows")
        for key in ("target_eV", "prediction_eV"):
            if part[key].shape != expected.shape or not torch.isfinite(part[key]).all():
                raise ValueError("Malformed prediction/target")
        for field in ("descriptors", "diagnostics"):
            for value in part.get(field, {}).values():
                if value.shape != expected.shape or not torch.isfinite(value).all():
                    raise ValueError("Malformed per-row diagnostic")
        parts.append(part)
    result = {key: torch.cat([p[key] for p in parts]) for key in
              ("source_idx", "target_eV", "prediction_eV")}
    for field in ("descriptors", "diagnostics"):
        if field in parts[0]:
            result[field] = {key: torch.cat([p[field][key] for p in parts])
                             for key in parts[0][field]}
    return result


def _graph_means(values, batch):
    from torch_geometric.utils import to_dense_batch
    dense, valid = to_dense_batch(values, batch)
    return dense.sum(dim=1) / valid.sum(dim=1).clamp_min(1).unsqueeze(-1)


@contextmanager
def intervention(addon, mode, variant, telemetry):
    """Temporarily intervene on computation; restore even on failure."""
    import torch
    check_variant(mode, variant)
    original_forward = addon.forward
    original_triplet = addon._triplet_update
    hook = None
    if variant == "rrwp_off":
        # Remove the entire learned RRWP contribution, including its bias.
        hook = addon.rrwp_projection.register_forward_hook(
            lambda _module, _args, output: torch.zeros_like(output))
    if variant == "triplet_off":
        # Preserve the output normalization; remove only the aggregate branch.
        def no_aggregate(self, pair, pair_valid, valid):
            output = self.triplet_norm(pair).masked_fill(
                ~pair_valid.unsqueeze(-1), 0.0)
            return output, None, None, None
        addon._triplet_update = MethodType(no_aggregate, addon)

    def forward(self, hidden, batch, edge_index=None):
        if variant == "off":
            return hidden
        update, details = self.compute_update(hidden, batch, edge_index)
        if not torch.isfinite(update).all():
            raise ValueError("Nonfinite diagnostic update")
        if variant == "full":
            mean_update = _graph_means(update, batch)
            energy = _graph_means(update.square(), batch).mean(-1).sqrt()
            hidden_energy = _graph_means(hidden.square(), batch).mean(-1).sqrt()
            dispersion = _graph_means(
                (update - mean_update[batch]).square(), batch).mean(-1).sqrt()
            assignment = details["assignment"]
            valid = details["valid"]
            entropy = -(assignment * assignment.clamp_min(1e-30).log()).sum(-1)
            entropy = (entropy * valid).sum(-1) / valid.sum(-1).clamp_min(1)
            telemetry.append({"update_rms": energy.detach().cpu(),
                "update_to_hidden_rms": (energy / hidden_energy.clamp_min(1e-12)).detach().cpu(),
                "receiver_dispersion_rms": dispersion.detach().cpu(),
                "assignment_entropy": entropy.detach().cpu()})
        if variant == "half":
            update = update * 0.5
        elif variant == "common_return":
            # Hold graph-mean update fixed while deleting receiver differences.
            update = _graph_means(update, batch)[batch]
        return hidden + update

    addon.forward = MethodType(forward, addon)
    try:
        yield
    finally:
        addon.forward = original_forward
        addon._triplet_update = original_triplet
        if hook is not None:
            hook.remove()


def _infer(graphs, model, mode, variant, role, output, mean, std):
    import torch
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader
    from molgap.pcqm_k1_causal_audit import _descriptors
    from molgap.training_reproducibility import atomic_json, atomic_torch_save, sha256_file
    output.mkdir(parents=True, exist_ok=False)
    chunks = []
    for offset in range(0, len(graphs), CHUNK_ROWS):
        loader = DataLoader(Subset(graphs, range(offset, offset + CHUNK_ROWS)),
            batch_size=128, shuffle=False, drop_last=False, num_workers=0, pin_memory=True)
        source, target, prediction, descriptors, diagnostics = [], [], [], [], []
        with torch.inference_mode(), intervention(model.relation_token, mode, variant, diagnostics):
            for batch in loader:
                if variant == "full":
                    descriptors.append(_descriptors(batch))
                batch = batch.to("cuda", non_blocking=True)
                value = model(batch.x, batch.edge_index, batch.edge_attr,
                              batch.batch, batch.random_walk_pe).view(-1)
                prediction.append((value.float() * std + mean).cpu())
                target.append(batch.y.view(-1).float().cpu())
                source.append(batch.source_idx.view(-1).long().cpu())
        row = {"source_idx": torch.cat(source), "target_eV": torch.cat(target),
               "prediction_eV": torch.cat(prediction)}
        if variant == "full":
            if len(descriptors) != len(diagnostics):
                raise ValueError("One layer-6 telemetry event per batch required")
            row["descriptors"] = {k: torch.cat([x[k] for x in descriptors]) for k in descriptors[0]}
            row["diagnostics"] = {k: torch.cat([x[k] for x in diagnostics]) for k in diagnostics[0]}
        expected = torch.arange(ROLES[role] + offset, ROLES[role] + offset + CHUNK_ROWS)
        if not torch.equal(row["source_idx"], expected):
            raise ValueError("Role/order changed")
        for key in ("target_eV", "prediction_eV"):
            if row[key].shape != expected.shape or not torch.isfinite(row[key]).all():
                raise ValueError("Invalid diagnostic payload")
        path = output / f"chunk_{offset // CHUNK_ROWS:02d}.pt"
        atomic_torch_save(path, row)
        chunks.append({"file": path.name, "rows": CHUNK_ROWS, "sha256": sha256_file(path)})
        atomic_json(output / "progress.json", {"mode": mode, "variant": variant,
            "role": role, "rows_completed": offset + CHUNK_ROWS, "chunks": chunks})
        print(f"diagnostic {mode}/{variant}/{role} {offset + CHUNK_ROWS}/50000", flush=True)
    return chunks


def run_worker(inputs, caches, output, release, mode):
    """Remote-only inference with the original, hash-verified model factory."""
    import torch
    from molgap.k1_portability_audit import _graphs, _model, TRANSFORM_SHA256
    from molgap.training_reproducibility import atomic_json, sha256_file
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Exactly one visible remote GPU required")
    if release["training_authorized"] is not False or release["variants"] != {k: list(v) for k,v in VARIANTS.items()}:
        raise ValueError("Frozen NO_TRAIN contract changed")
    inputs, output = Path(inputs), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, digest in release["input_files"].items():
        if sha256_file(inputs / name) != digest:
            raise ValueError(f"Input hash changed: {name}")
    if sha256_file(inputs / "target_transform.json") != TRANSFORM_SHA256:
        raise ValueError("Target transform changed")
    transform = json.loads((inputs / "target_transform.json").read_text())
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    checkpoint = inputs / f"candidates/{mode}/best_model.pt"
    digest = release["input_files"][f"candidates/{mode}/best_model.pt"]
    model = _model(mode, checkpoint, digest)
    model.requires_grad_(False)
    before = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    prior = json.loads((inputs / "prior_terminal.json").read_text())
    started = time.monotonic()
    cpu_started = time.process_time()
    records, role_events = {}, []
    for role in ROLES:
        graphs = _graphs(Path(caches[role]), role)
        if len(graphs) != 50000:
            raise ValueError("Frozen role size changed")
        role_events.extend({"role": role, "access_kind": access} for access in
                           ("prediction_input", "labels_read", "metric_computed"))
        records[role] = {}
        for variant in VARIANTS[mode]:
            dest = output / role / variant
            chunks = _infer(graphs, model, mode, variant, role, dest,
                            float(transform["mean"]), float(transform["std"]))
            payload = joined_payload(dest, chunks, ROLES[role])
            row = {"chunks": chunks, "rows": 50000,
                "mae_eV": float((payload["prediction_eV"]-payload["target_eV"]).double().abs().mean())}
            if variant == "full":
                key = "reproduction" if role == "original_100k" else "unseen_500k"
                saved = joined_payload(inputs / f"prior/{role}/{mode}", prior[key][mode]["chunks"], ROLES[role])
                if (not torch.equal(saved["source_idx"], payload["source_idx"])
                    or not torch.equal(saved["target_eV"], payload["target_eV"])):
                    raise ValueError("Reproduction row/target mismatch")
                diff = float((saved["prediction_eV"]-payload["prediction_eV"]).abs().max())
                row["reproduction_max_abs_eV"] = diff
                if diff > MAX_REPRODUCTION_EV:
                    raise ValueError("Unmodified model failed reproduction before intervention")
            records[role][variant] = row
            atomic_json(output / "progress.json", {"records": records, "complete": False})
        del graphs
    if any(not torch.equal(before[k], v.detach().cpu()) for k,v in model.state_dict().items()):
        raise ValueError("Frozen checkpoint tensors were mutated")
    terminal = {"format": "molgap-relation-intervention-worker-v1", "complete": True,
        "run_id": release["run_id"], "mode": mode, "checkpoint_sha256": digest,
        "records": records, "role_events": role_events, "state_unchanged": True,
        "training_executed": False, "optimizer_steps": 0, "model_inference_executed": True,
        "physical_batch": 128, "precision": "FP32", "tf32": False,
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0)},
        "official_validation_role_read": False, "test_dev_role_read": False,
        "test_challenge_role_read": False, "wall_seconds": time.monotonic()-started,
        "worker_process_cpu_seconds": time.process_time()-cpu_started,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    atomic_json(output / "terminal.json", terminal)
    return terminal
