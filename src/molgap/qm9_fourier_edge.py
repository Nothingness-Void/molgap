"""QM9 Track-C screen for Fourier-KAN persistent-edge dynamics."""
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
HARMONICS = 1
MIN_GAIN_VS_FULL_GPS_EV = 0.003
MIN_GAIN_VS_K1_EV = 0.001
MAX_PARAMETER_COUNT = 4_200_000
MAX_EPOCH_TIME_RATIO = 1.25
TASK_ID = "qm9-fourier-edge-k1-s42-v1"
PLATFORM_ID = "kaggle2"


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _state_sha256(model, *, omit_edge_proposal: bool = False) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if (
            omit_edge_proposal
            and name.startswith("edge_updates.")
            and ".update." in name
        ):
            continue
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class _FourierKANFactory:
    @staticmethod
    def make(input_channels: int, output_channels: int, harmonics: int):
        import torch
        import torch.nn as nn

        if input_channels <= 0 or output_channels <= 0 or harmonics <= 0:
            raise ValueError("Fourier-KAN dimensions must be positive")

        class FourierKAN(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_channels = int(input_channels)
                self.output_channels = int(output_channels)
                self.harmonics = int(harmonics)
                self.fourier_coefficients = nn.Parameter(
                    torch.randn(
                        2,
                        self.output_channels,
                        self.input_channels,
                        self.harmonics,
                    )
                    / math.sqrt(self.input_channels * self.harmonics)
                )
                self.bias = nn.Parameter(torch.zeros(self.output_channels))

            def forward(self, value):
                if value.shape[-1] != self.input_channels:
                    raise ValueError("Fourier-KAN input width changed")
                frequencies = torch.arange(
                    1,
                    self.harmonics + 1,
                    device=value.device,
                    dtype=value.dtype,
                )
                phase = value.unsqueeze(-1) * frequencies
                basis = torch.stack((torch.cos(phase), torch.sin(phase)), dim=0)
                result = torch.einsum(
                    "d...ik,doik->...o", basis, self.fourier_coefficients
                )
                return result + self.bias

        return FourierKAN()


def make_encoder(mode: str):
    """Build the absolute anchor, K1 control, or Fourier-edge candidate."""
    from .qm9_neural_atom import make_encoder as make_neural_atom_encoder

    if mode == "full_gps":
        return make_neural_atom_encoder("full_gps")
    if mode == "neural_atom_k1":
        return make_neural_atom_encoder("neural_atom_k1")
    if mode != "fourier_edge_k1":
        raise ValueError(f"Unknown Fourier-edge mode: {mode}")

    import torch.nn as nn

    model = make_neural_atom_encoder("neural_atom_k1")
    for edge_update in model.edge_updates:
        edge_update.update = nn.Sequential(
            nn.LayerNorm(model.edge_state_channels),
            _FourierKANFactory.make(
                model.edge_state_channels,
                model.edge_state_channels,
                HARMONICS,
            ),
            nn.Dropout(0.05),
        )
    return model


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
    import torch.nn.functional as functional
    from torch_geometric.loader import DataLoader

    from .qm9_gape import forward_gap, set_seed

    batch = next(iter(DataLoader(roles["train"][:8], batch_size=8))).to("cuda")
    reports = {}
    shared_k1_sha = None
    for mode in ("full_gps", "neural_atom_k1", "fourier_edge_k1"):
        set_seed(SEED)
        model = make_encoder(mode).to("cuda")
        parameter_count = sum(p.numel() for p in model.parameters())
        initial_sha = _state_sha256(model)
        shared_sha = None
        if mode != "full_gps":
            shared_sha = _state_sha256(model, omit_edge_proposal=True)
            if shared_k1_sha is None:
                shared_k1_sha = shared_sha
            elif shared_sha != shared_k1_sha:
                raise RuntimeError("K1/Fourier shared initialization changed")
        model.train()
        loss = forward_gap(model, batch, augmented=False).square().mean()
        loss.backward()
        finite = bool(torch.isfinite(loss).item()) and all(
            parameter.grad is None
            or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in model.parameters()
        )
        if not finite:
            raise RuntimeError(f"Non-finite remote preflight for {mode}")
        reports[mode] = {
            "parameter_count": parameter_count,
            "initial_state_sha256": initial_sha,
            "shared_without_edge_proposal_sha256": shared_sha,
            "finite_forward_backward": finite,
        }
        if mode == "fourier_edge_k1":
            layer = model.edge_updates[0].update[1]
            probe = torch.linspace(-1.0, 1.0, steps=5 * 64, device="cuda").view(5, 64)
            observed = layer(probe)
            frequencies = torch.arange(
                1, HARMONICS + 1, device="cuda", dtype=probe.dtype
            )
            phase = probe.unsqueeze(-1) * frequencies
            flat_basis = torch.cat(
                [torch.cos(phase).flatten(-2), torch.sin(phase).flatten(-2)],
                dim=-1,
            )
            weights = layer.fourier_coefficients.permute(1, 0, 2, 3).reshape(
                64, -1
            )
            manual = functional.linear(flat_basis, weights, layer.bias)
            equation_exact = bool(
                torch.allclose(observed, manual, atol=1e-6, rtol=1e-6)
            )
            gradients_nonzero = all(
                update.update[1].fourier_coefficients.grad is not None
                and float(update.update[1].fourier_coefficients.grad.abs().sum()) > 0
                for update in model.edge_updates
            )
            reports[mode]["manual_equation_match"] = equation_exact
            reports[mode]["all_fourier_gradients_nonzero"] = gradients_nonzero
            reports[mode]["harmonics"] = HARMONICS
            if (
                not equation_exact
                or not gradients_nonzero
                or parameter_count > MAX_PARAMETER_COUNT
            ):
                raise RuntimeError("Fourier-edge invariant failed")
        del model
    result = {
        "format": "molgap-qm9-fourier-edge-preflight-v1",
        "accepted": True,
        "source_commit": source_commit,
        "gpu": torch.cuda.get_device_name(0),
        "arms": reports,
        "shared_k1_fourier_sha256": shared_k1_sha,
        "harmonics": HARMONICS,
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

    if role not in {"anchor", "edge"}:
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

    if role == "anchor":
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
        _atomic_json(output_root / "anchor_worker.json", result)
        return result

    preflight = _preflight(roles, output_root, source_commit=source_commit)
    results = {}
    for mode in ("neural_atom_k1", "fourier_edge_k1"):
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
    _atomic_json(output_root / "edge_worker.json", result)
    return result


def aggregate(output_root: Path, *, source_commit: str, cache_sha256: str) -> dict:
    anchor = json.loads((output_root / "anchor_worker.json").read_text(encoding="utf-8"))
    edge = json.loads((output_root / "edge_worker.json").read_text(encoding="utf-8"))
    results = {
        "full_gps": anchor["full_gps"],
        "neural_atom_k1": edge["neural_atom_k1"],
        "fourier_edge_k1": edge["fourier_edge_k1"],
    }
    comparability = validate_paired_screen_contract(
        [results[name]["contract"] for name in results]
    )
    mae = {
        name: float(result["training"]["validation_gap_mae_eV"])
        for name, result in results.items()
    }
    gain_full = mae["full_gps"] - mae["fourier_edge_k1"]
    gain_k1 = mae["neural_atom_k1"] - mae["fourier_edge_k1"]
    epoch_ratio = (
        results["fourier_edge_k1"]["training"]["mean_epoch_seconds"]
        / results["neural_atom_k1"]["training"]["mean_epoch_seconds"]
    )
    nominated = (
        gain_full >= MIN_GAIN_VS_FULL_GPS_EV
        and gain_k1 >= MIN_GAIN_VS_K1_EV
        and epoch_ratio <= MAX_EPOCH_TIME_RATIO
        and results["fourier_edge_k1"]["training"]["memory_reserve_fraction"] >= 0.15
    )
    summary = {
        "format": "molgap-qm9-fourier-edge-screen-v1",
        "complete": True,
        "source_commit": source_commit,
        "cache_aggregate_sha256": cache_sha256,
        "comparability": comparability,
        "preflight": edge["preflight"],
        "results": results,
        "validation_gap_mae_eV": mae,
        "candidate_gain_vs_full_gps_eV": gain_full,
        "candidate_gain_vs_k1_eV": gain_k1,
        "required_gain_vs_full_gps_eV": MIN_GAIN_VS_FULL_GPS_EV,
        "required_gain_vs_k1_eV": MIN_GAIN_VS_K1_EV,
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
