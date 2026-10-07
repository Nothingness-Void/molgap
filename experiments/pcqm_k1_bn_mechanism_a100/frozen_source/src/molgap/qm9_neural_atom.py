"""QM9 Track-C screen for a low-rank Neural-Atom global mixer."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from .screen_policy import validate_paired_screen_contract, validate_screen_arm


TRAIN_ROWS = 30_000
VALIDATION_ROWS = 3_000
SEED = 42
BATCH_SIZE = 128
RWSE_DIM = 16
GAP_EPOCHS = 40
LEARNING_RATE = 4e-4
WEIGHT_DECAY = 1e-5
MIXER_LAYERS = (3, 6, 9)
MAX_SLOTS = 4
LATENT_CHANNELS = 64
MIN_GAIN_VS_BASELINE_EV = 0.003
MIN_GAIN_VS_ONE_SLOT_EV = 0.001
MAX_PARAMETER_COUNT = 4_200_000
MAX_EPOCH_TIME_RATIO = 1.15
TASK_ID = "qm9-neural-atom-mixer-s42-v1"
PLATFORM_ID = "kaggle2"


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _state_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _module_digest(digest, prefix: str, module) -> None:
    for name, value in sorted(module.state_dict().items()):
        digest.update(f"{prefix}.{name}".encode("utf-8") + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())


def _shared_backbone_sha256(model, *, full_gps: bool) -> str:
    """Hash only parameters shared by full-GPS and mixer architectures."""
    digest = hashlib.sha256()
    for name in ("node_emb", "edge_emb", "rwse_encoder", "edge_updates", "head"):
        _module_digest(digest, name, getattr(model, name))
    blocks = model.convs if full_gps else model.local_blocks
    for index, block in enumerate(blocks):
        for name in ("conv", "mlp", "norm1", "norm3"):
            _module_digest(digest, f"block{index}.{name}", getattr(block, name))
    return digest.hexdigest()


class _LocalGPSBlockFactory:
    @staticmethod
    def make(base):
        import torch.nn as nn
        import torch.nn.functional as functional

        class LocalGPSBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout = float(base.dropout)
                self.conv = base.conv
                self.mlp = base.mlp
                self.norm1 = base.norm1
                self.norm3 = base.norm3
                self.norm_with_batch = bool(base.norm_with_batch)

            def _normalize(self, normalizer, value, batch):
                if self.norm_with_batch:
                    return normalizer(value, batch=batch)
                return normalizer(value)

            def forward(self, x, edge_index, batch, *, edge_attr):
                local = self.conv(x, edge_index, edge_attr=edge_attr)
                local = functional.dropout(
                    local, p=self.dropout, training=self.training
                )
                output = self._normalize(self.norm1, local + x, batch)
                output = output + self.mlp(output)
                return self._normalize(self.norm3, output, batch)

        return LocalGPSBlock()


class _NeuralAtomMixerFactory:
    @staticmethod
    def make(
        hidden_channels: int,
        latent_channels: int,
        num_heads: int,
        max_slots: int,
        active_slots: int,
        dropout: float,
    ):
        import torch
        import torch.nn as nn
        from torch_geometric.utils import to_dense_batch

        if not 1 <= active_slots <= max_slots:
            raise ValueError("active_slots must be inside the allocated slot set")
        if latent_channels % num_heads:
            raise ValueError("latent_channels must be divisible by num_heads")

        class NeuralAtomMixer(nn.Module):
            def __init__(self):
                super().__init__()
                self.max_slots = int(max_slots)
                self.active_slots = int(active_slots)
                self.latent_channels = int(latent_channels)
                self.slot_seed = nn.Parameter(
                    torch.empty(self.max_slots, self.latent_channels)
                )
                nn.init.normal_(
                    self.slot_seed, std=self.latent_channels ** -0.5
                )
                self.node_norm = nn.LayerNorm(hidden_channels)
                self.node_key = nn.Linear(
                    hidden_channels, self.latent_channels, bias=False
                )
                self.node_value = nn.Linear(
                    hidden_channels, self.latent_channels, bias=False
                )
                self.slot_query = nn.Linear(
                    self.latent_channels, self.latent_channels, bias=False
                )
                self.slot_norm1 = nn.LayerNorm(self.latent_channels)
                self.slot_attention = nn.MultiheadAttention(
                    self.latent_channels,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                self.slot_norm2 = nn.LayerNorm(self.latent_channels)
                self.slot_ffn = nn.Sequential(
                    nn.Linear(self.latent_channels, 2 * self.latent_channels),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(2 * self.latent_channels, self.latent_channels),
                )
                self.return_projection = nn.Linear(
                    self.latent_channels, hidden_channels, bias=False
                )
                self.dropout = nn.Dropout(dropout)
                nn.init.zeros_(self.return_projection.weight)

            def compute_update(self, hidden, batch):
                dense, valid = to_dense_batch(self.node_norm(hidden), batch)
                keys = self.node_key(dense)
                values = self.node_value(dense)
                seeds = self.slot_seed[: self.active_slots]
                queries = self.slot_query(seeds)
                logits = torch.einsum("kd,bnd->bkn", queries, keys)
                logits = logits / math.sqrt(self.latent_channels)
                logits = logits.masked_fill(~valid.unsqueeze(1), float("-inf"))
                assignment = torch.softmax(logits, dim=-1)
                slots = seeds.unsqueeze(0) + torch.einsum(
                    "bkn,bnd->bkd", assignment, values
                )
                attended, _ = self.slot_attention(
                    slots, slots, slots, need_weights=False
                )
                slots = self.slot_norm1(slots + attended)
                slots = self.slot_norm2(slots + self.slot_ffn(slots))
                returned = torch.einsum("bkn,bkd->bnd", assignment, slots)
                update = self.dropout(self.return_projection(returned[valid]))
                diagnostics = {
                    "assignment": assignment,
                    "valid": valid,
                    "active_slots": self.active_slots,
                }
                return update, diagnostics

            def forward(self, hidden, batch):
                update, _ = self.compute_update(hidden, batch)
                return hidden + update

        return NeuralAtomMixer()


def make_encoder(mode: str):
    """Build the full-GPS baseline or a parameter-matched mixer arm."""
    if mode == "full_gps":
        from .qm9_local_hierarchy import make_encoder as make_baseline

        return make_baseline()
    if mode not in {"neural_atom_k1", "neural_atom_k4"}:
        raise ValueError(f"Unknown Neural-Atom mode: {mode}")

    import torch.nn as nn

    from .pcqm_gap_architecture import OGBEdgeStateStructuralGPSWrapper

    active_slots = 1 if mode == "neural_atom_k1" else MAX_SLOTS

    class NeuralAtomEdgeStateGPS(OGBEdgeStateStructuralGPSWrapper):
        def __init__(self):
            super().__init__(
                in_channels=9,
                edge_dim=3,
                hidden_channels=192,
                num_layers=9,
                num_heads=4,
                dropout=0.05,
                n_targets=1,
                pooling="mean",
                rwse_dim=RWSE_DIM,
                edge_state_channels=64,
            )
            self.local_blocks = nn.ModuleList(
                [_LocalGPSBlockFactory.make(base) for base in self.convs]
            )
            del self.convs
            self.neural_atom_mixers = nn.ModuleDict(
                {
                    str(layer): _NeuralAtomMixerFactory.make(
                        hidden_channels=192,
                        latent_channels=LATENT_CHANNELS,
                        num_heads=4,
                        max_slots=MAX_SLOTS,
                        active_slots=active_slots,
                        dropout=0.05,
                    )
                    for layer in MIXER_LAYERS
                }
            )

        def encode(self, x, edge_index, edge_attr, batch, random_walk_pe):
            expected = (x.shape[0], self.rwse_dim)
            if tuple(random_walk_pe.shape) != expected:
                raise ValueError(
                    f"random_walk_pe must have shape {expected}, "
                    f"got {tuple(random_walk_pe.shape)}"
                )
            h = self._embed_nodes(x)
            h = h + self.rwse_encoder(random_walk_pe.float())
            edge_state = self._embed_edges(edge_attr)
            for layer, (edge_update, block) in enumerate(
                zip(self.edge_updates, self.local_blocks), start=1
            ):
                edge_state = edge_update(h, edge_index, edge_state)
                h = block(h, edge_index, batch, edge_attr=edge_state)
                if layer in MIXER_LAYERS:
                    h = self.neural_atom_mixers[str(layer)](h, batch)
            return self._pool(h, batch)

    return NeuralAtomEdgeStateGPS()


def _arm_contract(arm: str, *, accelerator: str, split_fingerprint: str) -> dict:
    return {
        "arm": arm,
        "task_id": TASK_ID,
        "platform_id": PLATFORM_ID,
        "accelerator": accelerator,
        "data_role_fingerprint": split_fingerprint,
        "seed": SEED,
        "precision": "fp32",
        "optimizer_fingerprint": "adamw-lr4e-4-wd1e-5-clip1",
        "schedule_fingerprint": "cosine40-eta1e-6",
        "sample_exposure": "qm9-train30000-gap40",
        "physical_batch_per_device": BATCH_SIZE,
        "device_count": 1,
        "gradient_accumulation_steps": 1,
    }


def _preflight(roles, output_root: Path, *, source_commit: str) -> dict:
    import torch
    from torch_geometric.loader import DataLoader

    from .qm9_gape import forward_gap, set_seed

    batch = next(iter(DataLoader(roles["train"][:8], batch_size=8))).to("cuda")
    set_seed(SEED)
    baseline = make_encoder("full_gps").to("cuda").eval()
    baseline_backbone_sha = _shared_backbone_sha256(baseline, full_gps=True)
    baseline_parameter_count = sum(p.numel() for p in baseline.parameters())
    arms = {}
    candidate_initial_sha = None
    candidate_parameter_count = None
    for mode in ("neural_atom_k1", "neural_atom_k4"):
        set_seed(SEED)
        model = make_encoder(mode).to("cuda").eval()
        backbone_sha = _shared_backbone_sha256(model, full_gps=False)
        initial_sha = _state_sha256(model)
        parameter_count = sum(p.numel() for p in model.parameters())
        if backbone_sha != baseline_backbone_sha:
            raise RuntimeError(f"Shared backbone initialization changed for {mode}")
        if candidate_initial_sha is None:
            candidate_initial_sha = initial_sha
            candidate_parameter_count = parameter_count
        elif initial_sha != candidate_initial_sha:
            raise RuntimeError("K1/K4 initialization or parameterization changed")
        elif parameter_count != candidate_parameter_count:
            raise RuntimeError("K1/K4 parameter count changed")
        mixer_checks = []
        for layer in MIXER_LAYERS:
            mixer = model.neural_atom_mixers[str(layer)]
            probe = torch.linspace(
                -1.0,
                1.0,
                steps=int(batch.num_nodes) * 192,
                device="cuda",
            ).reshape(int(batch.num_nodes), 192)
            update, diagnostics = mixer.compute_update(probe, batch.batch)
            assignment = diagnostics["assignment"]
            valid = diagnostics["valid"]
            sums = assignment.sum(dim=-1)
            padded_mass = assignment.masked_select(~valid.unsqueeze(1)).sum()
            check = {
                "layer": layer,
                "active_slots": diagnostics["active_slots"],
                "assignment_mass_exact": bool(
                    torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=0)
                ),
                "padding_mass_zero": bool(padded_mass.item() == 0.0),
                "zero_return_exact": bool(torch.count_nonzero(update).item() == 0),
                "output_projection_zero": bool(
                    torch.count_nonzero(mixer.return_projection.weight).item() == 0
                ),
            }
            if not all(value for key, value in check.items() if key != "layer" and key != "active_slots"):
                raise RuntimeError(f"Mixer invariant failed for {mode}: {check}")
            mixer_checks.append(check)
        model.train()
        loss = forward_gap(model, batch, augmented=False).square().mean()
        loss.backward()
        gradients = [
            model.neural_atom_mixers[str(layer)].return_projection.weight.grad
            for layer in MIXER_LAYERS
        ]
        finite_gradient = bool(torch.isfinite(loss).item()) and all(
            gradient is not None
            and bool(torch.isfinite(gradient).all().item())
            and float(gradient.abs().sum()) > 0
            for gradient in gradients
        )
        if not finite_gradient or parameter_count > MAX_PARAMETER_COUNT:
            raise RuntimeError(f"Candidate preflight failed for {mode}")
        arms[mode] = {
            "parameter_count": parameter_count,
            "shared_backbone_sha256": backbone_sha,
            "initial_state_sha256": initial_sha,
            "mixer_checks": mixer_checks,
            "finite_return_projection_gradients": finite_gradient,
        }
        del model
    result = {
        "format": "molgap-qm9-neural-atom-preflight-v1",
        "accepted": True,
        "source_commit": source_commit,
        "gpu": torch.cuda.get_device_name(0),
        "baseline_parameter_count": baseline_parameter_count,
        "baseline_shared_backbone_sha256": baseline_backbone_sha,
        "candidate_arms": arms,
        "mixer_layers": list(MIXER_LAYERS),
        "max_slots": MAX_SLOTS,
        "latent_channels": LATENT_CHANNELS,
        "model_inference_executed": True,
        "official_pcqm_roles_read": False,
        "test_role_read": False,
    }
    _atomic_json(output_root / "preflight.json", result)
    return result


def run_worker(
    role: str,
    cache_root: Path,
    output_root: Path,
    *,
    source_commit: str,
    cache_sha256: str,
):
    import torch

    from .qm9_gape import set_seed, train_gap
    from .qm9_local_hierarchy import load_cache

    if role not in {"baseline", "mixers"}:
        raise ValueError(f"Unknown worker role: {role}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each worker requires exactly one visible GPU")
    validate_screen_arm(physical_batch_per_device=BATCH_SIZE)
    torch.backends.cuda.matmul.allow_tf32 = False
    roles, manifest = load_cache(cache_root, cache_sha256)
    accelerator = torch.cuda.get_device_name(0)

    def train_arm(mode: str):
        set_seed(SEED)
        torch.cuda.reset_peak_memory_stats()
        training = train_gap(
            make_encoder(mode),
            roles,
            output_root / mode,
            augmented=False,
            source_commit=source_commit,
            cache_sha256=cache_sha256,
        )
        peak_bytes = int(torch.cuda.max_memory_allocated())
        total_bytes = int(torch.cuda.get_device_properties(0).total_memory)
        training["peak_memory_mib"] = peak_bytes / 1024**2
        training["total_memory_mib"] = total_bytes / 1024**2
        training["memory_reserve_fraction"] = 1.0 - peak_bytes / total_bytes
        return training

    if role == "baseline":
        result = {
            "full_gps": {
                "training": train_arm("full_gps"),
                "contract": _arm_contract(
                    "full_gps",
                    accelerator=accelerator,
                    split_fingerprint=manifest["split_fingerprint"],
                ),
            }
        }
        _atomic_json(output_root / "baseline_worker.json", result)
        return result

    preflight = _preflight(roles, output_root, source_commit=source_commit)
    results = {}
    for mode in ("neural_atom_k1", "neural_atom_k4"):
        results[mode] = {
            "training": train_arm(mode),
            "contract": _arm_contract(
                mode,
                accelerator=accelerator,
                split_fingerprint=manifest["split_fingerprint"],
            ),
        }
        torch.cuda.empty_cache()
    result = {"preflight": preflight, **results}
    _atomic_json(output_root / "mixer_worker.json", result)
    return result


def aggregate(output_root: Path, *, source_commit: str, cache_sha256: str) -> dict:
    baseline_worker = json.loads(
        (output_root / "baseline_worker.json").read_text(encoding="utf-8")
    )
    mixer_worker = json.loads(
        (output_root / "mixer_worker.json").read_text(encoding="utf-8")
    )
    results = {
        "full_gps": baseline_worker["full_gps"],
        "neural_atom_k1": mixer_worker["neural_atom_k1"],
        "neural_atom_k4": mixer_worker["neural_atom_k4"],
    }
    comparability = validate_paired_screen_contract(
        [results[name]["contract"] for name in results]
    )
    mae = {
        name: float(result["training"]["validation_gap_mae_eV"])
        for name, result in results.items()
    }
    gain_baseline = mae["full_gps"] - mae["neural_atom_k4"]
    gain_one_slot = mae["neural_atom_k1"] - mae["neural_atom_k4"]
    epoch_ratio = (
        results["neural_atom_k4"]["training"]["mean_epoch_seconds"]
        / results["full_gps"]["training"]["mean_epoch_seconds"]
    )
    nominated = (
        gain_baseline >= MIN_GAIN_VS_BASELINE_EV
        and gain_one_slot >= MIN_GAIN_VS_ONE_SLOT_EV
        and epoch_ratio <= MAX_EPOCH_TIME_RATIO
        and results["neural_atom_k4"]["training"]["memory_reserve_fraction"]
        >= 0.15
    )
    summary = {
        "format": "molgap-qm9-neural-atom-screen-v1",
        "complete": True,
        "source_commit": source_commit,
        "cache_aggregate_sha256": cache_sha256,
        "comparability": comparability,
        "preflight": mixer_worker["preflight"],
        "results": results,
        "validation_gap_mae_eV": mae,
        "candidate_gain_vs_baseline_eV": gain_baseline,
        "candidate_gain_vs_one_slot_eV": gain_one_slot,
        "required_gain_vs_baseline_eV": MIN_GAIN_VS_BASELINE_EV,
        "required_gain_vs_one_slot_eV": MIN_GAIN_VS_ONE_SLOT_EV,
        "candidate_epoch_time_ratio": epoch_ratio,
        "max_epoch_time_ratio": MAX_EPOCH_TIME_RATIO,
        "pcqm_transfer_nominated": nominated,
        "model_inference_executed_by_acceptance": False,
        "official_pcqm_roles_read": False,
        "test_role_read": False,
    }
    _atomic_json(output_root / "metrics.json", summary)
    from .qm9_gape import sha256_file

    _atomic_json(
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
