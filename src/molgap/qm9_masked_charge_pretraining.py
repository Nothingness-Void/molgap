"""Equal-exposure masked pretraining screen for the QM9 charge adapter."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .qm9_charge_adapter import (
    BATCH_SIZE,
    CHARGE_ALGORITHM,
    MODEL_SEED,
    atomic_json,
    atomic_torch_save,
    forward_encoder,
    load_cache,
    make_charge_encoder,
    set_seed,
    sha256_file,
    train_arm,
    _make_loader,
    _restore_rng,
    _rng_state,
    _shared_initialization_exact,
)


PRETRAIN_EPOCHS = 20
FINETUNE_EPOCHS = 40
CONTROL_EPOCHS = PRETRAIN_EPOCHS + FINETUNE_EPOCHS
MASK_RATE = 0.15
LEARNING_RATE = 4e-4
WEIGHT_DECAY = 1e-5
MIN_GAIN_EV = 0.001
FORMAT = "molgap-qm9-masked-charge-pretraining-screen-v1"


class MaskedReconstructionHeads:
    """Training-only learned masks and local reconstruction heads."""

    def __init__(self, model):
        import torch
        import torch.nn as nn
        from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

        self.node_mask = None
        self.edge_mask = None
        self.node_state = None
        self.edge_state = None
        self.node_token = nn.Parameter(torch.zeros(192))
        self.edge_token = nn.Parameter(torch.zeros(64))
        nn.init.normal_(self.node_token, std=192**-0.5)
        nn.init.normal_(self.edge_token, std=64**-0.5)
        self.atom_heads = nn.ModuleList(
            nn.Linear(192, classes) for classes in get_atom_feature_dims()
        )
        self.bond_heads = nn.ModuleList(
            nn.Linear(64, classes) for classes in get_bond_feature_dims()
        )
        self.charge_head = nn.Linear(192, 2)
        self.module = nn.ModuleDict(
            {
                "atom_heads": self.atom_heads,
                "bond_heads": self.bond_heads,
                "charge_head": self.charge_head,
                "tokens": nn.ParameterDict(
                    {"node": self.node_token, "edge": self.edge_token}
                ),
            }
        )
        self.handles = [
            model.node_emb.register_forward_hook(self._mask_nodes),
            model.edge_emb.register_forward_hook(self._mask_edges),
            model.convs[-1].register_forward_hook(self._capture_nodes),
            model.edge_updates[-1].register_forward_hook(self._capture_edges),
        ]

    def set_masks(self, node_mask, edge_mask):
        self.node_mask = node_mask
        self.edge_mask = edge_mask

    def _mask_nodes(self, _module, _inputs, output):
        if self.node_mask is None:
            return output
        masked = output.clone()
        masked[self.node_mask] = self.node_token
        return masked

    def _mask_edges(self, _module, _inputs, output):
        if self.edge_mask is None:
            return output
        masked = output.clone()
        masked[self.edge_mask] = self.edge_token
        return masked

    def _capture_nodes(self, _module, _inputs, output):
        self.node_state = output

    def _capture_edges(self, _module, _inputs, output):
        self.edge_state = output

    def close(self):
        for handle in self.handles:
            handle.remove()


def _paired_edge_mask(edge_index, rate: float, generator):
    import torch

    source, target = edge_index.cpu()
    node_count = int(max(source.max(), target.max())) + 1
    low = torch.minimum(source, target)
    high = torch.maximum(source, target)
    keys = low * node_count + high
    _, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    undirected = torch.rand(
        int(inverse.max()) + 1, generator=generator
    ) < rate
    if not bool(undirected.any()):
        undirected[0] = True
    return undirected[inverse]


def _node_mask(node_count: int, rate: float, generator):
    import torch

    mask = torch.rand(node_count, generator=generator) < rate
    if not bool(mask.any()):
        mask[0] = True
    return mask


def _pretrain_loss(model, heads, batch, charge_mean, charge_std, generator):
    import torch
    import torch.nn.functional as functional

    original_x = batch.x.clone()
    original_edge = batch.edge_attr.clone()
    original_charge = batch.gasteiger_features.clone()
    node_mask = _node_mask(batch.x.shape[0], MASK_RATE, generator).to(batch.x.device)
    edge_mask = _paired_edge_mask(batch.edge_index, MASK_RATE, generator).to(
        batch.x.device
    )
    corrupted_charge = original_charge.clone()
    corrupted_charge[node_mask] = charge_mean
    batch.gasteiger_features = corrupted_charge
    heads.set_masks(node_mask, edge_mask)
    forward_encoder(model, batch, candidate=True)

    atom_loss = sum(
        functional.cross_entropy(
            head(heads.node_state[node_mask]), original_x[node_mask, column]
        )
        for column, head in enumerate(heads.atom_heads)
    ) / len(heads.atom_heads)
    bond_loss = sum(
        functional.cross_entropy(
            head(heads.edge_state[edge_mask]),
            original_edge[edge_mask, column],
        )
        for column, head in enumerate(heads.bond_heads)
    ) / len(heads.bond_heads)
    normalized_charge = (original_charge - charge_mean) / charge_std
    charge_loss = functional.smooth_l1_loss(
        heads.charge_head(heads.node_state[node_mask]),
        normalized_charge[node_mask],
    )
    loss = atom_loss + bond_loss + 0.5 * charge_loss
    return loss, {
        "atom": atom_loss,
        "bond": bond_loss,
        "charge": charge_loss,
    }


def pretrain(
    model,
    graphs,
    output_dir: Path,
    *,
    charge_mean,
    charge_std,
    source_commit: str,
    cache_sha256: str,
):
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model = model.to(device)
    heads = MaskedReconstructionHeads(model)
    heads.module = heads.module.to(device)
    parameters = list(model.parameters()) + list(heads.module.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=PRETRAIN_EPOCHS, eta_min=1e-6
    )
    loader = _make_loader(graphs, shuffle=True, seed=MODEL_SEED)
    mask_generator = torch.Generator().manual_seed(MODEL_SEED + 29)
    mean = torch.as_tensor(charge_mean, dtype=torch.float32, device=device).view(1, -1)
    std = torch.as_tensor(charge_std, dtype=torch.float32, device=device).view(1, -1)
    trace = []
    start_epoch = 0
    checkpoint_path = output_dir / "last_checkpoint.pt"
    if checkpoint_path.is_file():
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        required = {
            "source_commit": source_commit,
            "cache_sha256": cache_sha256,
            "seed": MODEL_SEED,
            "batch_size": BATCH_SIZE,
            "max_epochs": PRETRAIN_EPOCHS,
        }
        if any(checkpoint.get(key) != value for key, value in required.items()):
            raise RuntimeError("Masked-pretraining checkpoint contract changed")
        model.load_state_dict(checkpoint["model"])
        heads.module.load_state_dict(checkpoint["heads"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        trace = checkpoint["trace"]
        start_epoch = int(checkpoint["epoch"]) + 1
        _restore_rng(checkpoint["rng"], loader)
        mask_generator.set_state(checkpoint["mask_rng"])

    for epoch in range(start_epoch, PRETRAIN_EPOCHS):
        model.train()
        heads.module.train()
        totals = {"loss": 0.0, "atom": 0.0, "bond": 0.0, "charge": 0.0}
        batches = 0
        started = time.perf_counter()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, pieces = _pretrain_loss(
                model, heads, batch, mean, std, mask_generator
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            totals["loss"] += float(loss.detach())
            for name, value in pieces.items():
                totals[name] += float(value.detach())
            batches += 1
        scheduler.step()
        row = {
            "epoch": epoch,
            **{name: value / batches for name, value in totals.items()},
            "seconds": time.perf_counter() - started,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        trace.append(row)
        atomic_torch_save(
            checkpoint_path,
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "heads": heads.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "trace": trace,
                "source_commit": source_commit,
                "cache_sha256": cache_sha256,
                "seed": MODEL_SEED,
                "batch_size": BATCH_SIZE,
                "max_epochs": PRETRAIN_EPOCHS,
                "rng": _rng_state(loader),
                "mask_rng": mask_generator.get_state(),
            },
        )
        atomic_json(output_dir / "trace.json", {"epochs": trace})
        print(
            f"masked ep{epoch:02d} loss={row['loss']:.6f} "
            f"atom={row['atom']:.6f} bond={row['bond']:.6f} "
            f"charge={row['charge']:.6f} {row['seconds']:.1f}s",
            flush=True,
        )
    auxiliary_parameters = sum(p.numel() for p in heads.module.parameters())
    heads.close()
    del heads
    return model, {
        "epochs_completed": len(trace),
        "mean_epoch_seconds": float(np.mean([row["seconds"] for row in trace])),
        "auxiliary_parameters": auxiliary_parameters,
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def _load_parent(parent_metrics: Path) -> dict:
    parent = json.loads(parent_metrics.read_text(encoding="utf-8"))
    required = {
        "format": "molgap-qm9-charge-adapter-screen-v2",
        "complete": True,
        "charge_algorithm": CHARGE_ALGORITHM,
        "official_pcqm_roles_read": False,
        "test_role_read": False,
    }
    for key, value in required.items():
        if parent.get(key) != value:
            raise RuntimeError(f"Parent charge screen changed for {key}")
    return parent


def run_preflight(
    cache_root: Path,
    output_root: Path,
    *,
    parent_metrics: Path,
    source_commit: str,
    cache_sha256: str,
):
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Preflight requires exactly one visible DCU")
    parent = _load_parent(parent_metrics)
    roles, manifest = load_cache(cache_root, cache_sha256)
    stats = manifest["train_charge_statistics"]
    batch = next(
        iter(_make_loader(roles["train"][:BATCH_SIZE], shuffle=False, seed=MODEL_SEED))
    ).to("cuda")
    if int(batch.num_graphs) != BATCH_SIZE:
        raise RuntimeError("Preflight did not use physical batch 128")
    set_seed(MODEL_SEED)
    model = make_charge_encoder(stats["mean"], stats["std"]).to("cuda")
    heads = MaskedReconstructionHeads(model)
    heads.module = heads.module.to("cuda")
    generator = torch.Generator().manual_seed(MODEL_SEED + 29)
    mean = torch.tensor(stats["mean"], device="cuda").view(1, -1)
    std = torch.tensor(stats["std"], device="cuda").view(1, -1)
    loss, pieces = _pretrain_loss(model, heads, batch, mean, std, generator)
    loss.backward()
    parameters = list(model.parameters()) + list(heads.module.parameters())
    finite = bool(torch.isfinite(loss)) and all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters
    )
    result = {
        "format": "molgap-qm9-masked-charge-pretraining-preflight-v1",
        "accepted": finite,
        "source_commit": source_commit,
        "cache_aggregate_sha256": cache_sha256,
        "parent_metrics_sha256": sha256_file(parent_metrics),
        "parent_candidate_gain_eV": parent["candidate_gain_eV"],
        "physical_batch_per_device": BATCH_SIZE,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "auxiliary_parameters": sum(p.numel() for p in heads.module.parameters()),
        "finite_forward_backward": finite,
        "loss": float(loss.detach()),
        "loss_parts": {name: float(value.detach()) for name, value in pieces.items()},
        "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "gpu": torch.cuda.get_device_name(0),
        "official_pcqm_roles_read": False,
        "test_role_read": False,
    }
    heads.close()
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "preflight.json", result)
    if not finite:
        raise RuntimeError("Masked pretraining produced non-finite values")
    return result


def run_screen(
    cache_root: Path,
    output_root: Path,
    *,
    parent_metrics: Path,
    preflight_path: Path,
    source_commit: str,
    cache_sha256: str,
):
    import torch

    from .screen_policy import validate_paired_screen_contract

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Training requires exactly one visible DCU")
    parent = _load_parent(parent_metrics)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    required = {
        "format": "molgap-qm9-masked-charge-pretraining-preflight-v1",
        "accepted": True,
        "source_commit": source_commit,
        "cache_aggregate_sha256": cache_sha256,
        "parent_metrics_sha256": sha256_file(parent_metrics),
        "physical_batch_per_device": BATCH_SIZE,
        "finite_forward_backward": True,
        "official_pcqm_roles_read": False,
        "test_role_read": False,
    }
    for key, value in required.items():
        if preflight.get(key) != value:
            raise RuntimeError(f"Preflight contract changed for {key}")
    roles, manifest = load_cache(cache_root, cache_sha256)
    stats = manifest["train_charge_statistics"]
    output_root.mkdir(parents=True, exist_ok=True)

    set_seed(MODEL_SEED)
    control = make_charge_encoder(stats["mean"], stats["std"])
    set_seed(MODEL_SEED)
    candidate = make_charge_encoder(stats["mean"], stats["std"])
    if not _shared_initialization_exact(control, candidate):
        raise RuntimeError("Initial model states differ")

    set_seed(MODEL_SEED)
    control_result = train_arm(
        control,
        roles,
        output_root / "scratch60",
        candidate=True,
        source_commit=source_commit,
        cache_sha256=cache_sha256,
        epochs=CONTROL_EPOCHS,
    )
    del control
    torch.cuda.empty_cache()

    set_seed(MODEL_SEED)
    candidate, pretrain_result = pretrain(
        candidate,
        roles["train"],
        output_root / "masked_pretrain20",
        charge_mean=stats["mean"],
        charge_std=stats["std"],
        source_commit=source_commit,
        cache_sha256=cache_sha256,
    )
    set_seed(MODEL_SEED)
    finetune_result = train_arm(
        candidate,
        roles,
        output_root / "masked20_gap40",
        candidate=True,
        source_commit=source_commit,
        cache_sha256=cache_sha256,
        epochs=FINETUNE_EPOCHS,
    )

    accelerator = torch.cuda.get_device_name(0)
    shared = {
        "task_id": "qm9-masked-charge-pretraining-s42-v1",
        "platform_id": "scnet-kunshan",
        "accelerator": accelerator,
        "data_role_fingerprint": manifest["split_fingerprint"],
        "seed": MODEL_SEED,
        "precision": "fp32",
        "optimizer_fingerprint": "adamw-lr4e-4-wd1e-5-clip1",
        "schedule_fingerprint": "equal-encoder-exposure60",
        "sample_exposure": "qm9-train30000-encoder60",
        "physical_batch_per_device": BATCH_SIZE,
        "device_count": 1,
        "gradient_accumulation_steps": 1,
    }
    contracts = [{"arm": arm, **shared} for arm in ("scratch60", "masked20_gap40")]
    comparability = validate_paired_screen_contract(contracts)
    gain = (
        control_result["validation_gap_mae_eV"]
        - finetune_result["validation_gap_mae_eV"]
    )
    summary = {
        "format": FORMAT,
        "complete": True,
        "source_commit": source_commit,
        "cache_aggregate_sha256": cache_sha256,
        "parent_charge_screen": {
            "sha256": sha256_file(parent_metrics),
            "candidate_gain_eV": parent["candidate_gain_eV"],
            "nominated": parent["pcqm_transfer_nominated"],
        },
        "split_fingerprint": manifest["split_fingerprint"],
        "seed": MODEL_SEED,
        "gpu": accelerator,
        "training_contract": {
            "control_gap_epochs": CONTROL_EPOCHS,
            "candidate_pretrain_epochs": PRETRAIN_EPOCHS,
            "candidate_gap_epochs": FINETUNE_EPOCHS,
            "mask_rate": MASK_RATE,
            "masked_targets": ["atom_categories", "bond_categories", "gasteiger_charge"],
            "physical_batch_per_device": BATCH_SIZE,
            "total_encoder_exposure_epochs_per_arm": CONTROL_EPOCHS,
        },
        "comparability": comparability,
        "arm_contracts": contracts,
        "results": {
            "scratch60": control_result,
            "masked20_gap40": {
                "pretraining": pretrain_result,
                "finetuning": finetune_result,
            },
        },
        "candidate_gain_eV": gain,
        "required_gain_eV": MIN_GAIN_EV,
        "pcqm_transfer_nominated": gain >= MIN_GAIN_EV,
        "official_pcqm_roles_read": False,
        "test_role_read": False,
    }
    atomic_json(output_root / "metrics.json", summary)
    atomic_json(
        output_root / "completion_manifest.json",
        {
            **summary,
            "artifact_sha256": {
                str(path.relative_to(output_root)): sha256_file(path)
                for path in sorted(output_root.rglob("*"))
                if path.is_file() and path.name != "completion_manifest.json"
            },
        },
    )
    return summary
