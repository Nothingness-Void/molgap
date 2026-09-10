"""Paired PCQM-100K transfer of the accepted QM9 Fourier EdgeState model."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .screen_policy import validate_paired_screen_contract, validate_screen_arm


TRAIN_ROWS = 100_000
VALIDATION_ROWS = 10_000
SEED = 42
BATCH_SIZE = 128
GAP_EPOCHS = 40
LEARNING_RATE = 4e-4
WEIGHT_DECAY = 1e-5
GEOMETRY_CACHE_SHA256 = (
    "3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22"
)
PARENT_GRAPH_CACHE_SHA256 = (
    "eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21"
)
MIN_GAIN_VS_FULL_GPS_EV = 0.003
MIN_GAIN_VS_K1_EV = 0.001
MAX_EPOCH_TIME_RATIO = 1.25
TASK_ID = "pcqm-gap100k-fourier-edge-s42-v1"
PLATFORM_ID = "kaggle2"
EXPECTED_PARAMETERS = {
    "full_gps": 4_771_073,
    "neural_atom_k1": 3_658_817,
    "fourier_edge_k1": 3_658_241,
}


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


def _strip_geometry(roles: dict[str, list]) -> None:
    """Make the model-input boundary physically pure 2D after cache verification."""
    forbidden = (
        "pos",
        "edge_distance",
        "wedge_angle_cos",
        "wedge_edge_ids",
        "geometry_valid",
    )
    for graphs in roles.values():
        for graph in graphs:
            for name in forbidden:
                if hasattr(graph, name):
                    delattr(graph, name)


def load_roles() -> tuple[dict[str, list], dict]:
    from .pcqm_local_global_runner import find_geometry_cache, load_graphs

    root, manifest = find_geometry_cache()
    if manifest.get("aggregate_sha256") != GEOMETRY_CACHE_SHA256:
        raise RuntimeError("PCQM geometry cache identity changed")
    if manifest.get("parent_graph_cache_aggregate_sha256") != PARENT_GRAPH_CACHE_SHA256:
        raise RuntimeError("PCQM parent graph cache identity changed")
    if manifest.get("official_validation_role_read") is not False:
        raise RuntimeError("Official PCQM validation role boundary changed")
    if manifest.get("test_dev_role_read") is not False:
        raise RuntimeError("PCQM test-dev role boundary changed")
    roles = load_graphs(root, manifest)
    _strip_geometry(roles)
    if len(roles["train"]) != TRAIN_ROWS or len(roles["validation"]) != VALIDATION_ROWS:
        raise RuntimeError("PCQM 100K/10K role counts changed")
    return roles, manifest


def _arm_contract(arm: str, *, accelerator: str) -> dict:
    return {
        "arm": arm,
        "task_id": TASK_ID,
        "platform_id": PLATFORM_ID,
        "accelerator": accelerator,
        "data_role_fingerprint": PARENT_GRAPH_CACHE_SHA256,
        "seed": SEED,
        "precision": "fp32",
        "optimizer_fingerprint": "adamw-lr4e-4-wd1e-5-clip1",
        "schedule_fingerprint": "cosine40-eta1e-6",
        "sample_exposure": "pcqm-official-train-derived100000-gap40",
        "physical_batch_per_device": BATCH_SIZE,
        "device_count": 1,
        "gradient_accumulation_steps": 1,
    }


def _preflight(roles, output_root: Path, *, source_commit: str) -> dict:
    import torch
    import torch.nn.functional as functional
    from torch_geometric.loader import DataLoader

    from .qm9_fourier_edge import HARMONICS, make_encoder
    from .qm9_gape import forward_gap, set_seed

    batch = next(iter(DataLoader(roles["train"][:8], batch_size=8))).to("cuda")
    reports = {}
    shared_sha = None
    for mode in EXPECTED_PARAMETERS:
        set_seed(SEED)
        model = make_encoder(mode).to("cuda")
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != EXPECTED_PARAMETERS[mode]:
            raise RuntimeError(f"{mode} parameter count changed: {parameter_count}")
        shared_without_edge = None
        if mode != "full_gps":
            shared_without_edge = _state_sha256(model, omit_edge_proposal=True)
            if shared_sha is None:
                shared_sha = shared_without_edge
            elif shared_without_edge != shared_sha:
                raise RuntimeError("K1/Fourier shared initialization changed")
        model.train()
        loss = forward_gap(model, batch, augmented=False).square().mean()
        loss.backward()
        finite = bool(torch.isfinite(loss)) and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        if not finite:
            raise RuntimeError(f"Non-finite remote preflight for {mode}")
        reports[mode] = {
            "parameter_count": parameter_count,
            "initial_state_sha256": _state_sha256(model),
            "shared_without_edge_proposal_sha256": shared_without_edge,
            "finite_forward_backward": finite,
        }
        if mode == "fourier_edge_k1":
            layer = model.edge_updates[0].update[1]
            probe = torch.linspace(-1.0, 1.0, steps=5 * 64, device="cuda").view(5, 64)
            observed = layer(probe)
            frequencies = torch.arange(1, HARMONICS + 1, device="cuda", dtype=probe.dtype)
            phase = probe.unsqueeze(-1) * frequencies
            flat_basis = torch.cat(
                [torch.cos(phase).flatten(-2), torch.sin(phase).flatten(-2)], dim=-1
            )
            weights = layer.fourier_coefficients.permute(1, 0, 2, 3).reshape(64, -1)
            manual = functional.linear(flat_basis, weights, layer.bias)
            reports[mode]["manual_equation_match"] = bool(
                torch.allclose(observed, manual, atol=1e-6, rtol=1e-6)
            )
            reports[mode]["all_fourier_gradients_nonzero"] = all(
                update.update[1].fourier_coefficients.grad is not None
                and float(update.update[1].fourier_coefficients.grad.abs().sum()) > 0
                for update in model.edge_updates
            )
            reports[mode]["harmonics"] = HARMONICS
            if not reports[mode]["manual_equation_match"] or not reports[mode][
                "all_fourier_gradients_nonzero"
            ]:
                raise RuntimeError("Fourier equation/gradient invariant failed")
        del model
    result = {
        "format": "molgap-pcqm-gap100k-fourier-edge-preflight-v1",
        "accepted": True,
        "source_commit": source_commit,
        "gpu": torch.cuda.get_device_name(0),
        "arms": reports,
        "shared_k1_fourier_sha256": shared_sha,
        "pure_2d_model_inputs": ["ogb_atom", "ogb_real_bond", "rwse16"],
        "geometry_attributes_removed_before_batching": True,
        "model_inference_executed": True,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    _atomic_json(output_root / "preflight.json", result)
    return result


def run_worker(role: str, output_root: Path, *, source_commit: str) -> dict:
    import torch

    from .qm9_fourier_edge import make_encoder
    from .qm9_gape import set_seed, train_gap

    if role not in {"anchor", "edge"}:
        raise ValueError(f"Unknown worker role: {role}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each worker requires exactly one visible GPU")
    validate_screen_arm(physical_batch_per_device=BATCH_SIZE)
    torch.backends.cuda.matmul.allow_tf32 = False
    roles, _ = load_roles()
    accelerator = torch.cuda.get_device_name(0)

    def train_arm(mode: str) -> dict:
        set_seed(SEED)
        torch.cuda.reset_peak_memory_stats()
        training = train_gap(
            make_encoder(mode),
            roles,
            output_root / mode,
            augmented=False,
            source_commit=source_commit,
            cache_sha256=GEOMETRY_CACHE_SHA256,
        )
        peak_bytes = int(torch.cuda.max_memory_allocated())
        total_bytes = int(torch.cuda.get_device_properties(0).total_memory)
        training.update(
            {
                "peak_memory_mib": peak_bytes / 1024**2,
                "total_memory_mib": total_bytes / 1024**2,
                "memory_reserve_fraction": 1.0 - peak_bytes / total_bytes,
            }
        )
        return {
            "training": training,
            "contract": _arm_contract(mode, accelerator=accelerator),
        }

    if role == "anchor":
        result = {"full_gps": train_arm("full_gps")}
        _atomic_json(output_root / "anchor_worker.json", result)
        return result

    preflight = _preflight(roles, output_root, source_commit=source_commit)
    result = {"preflight": preflight}
    for mode in ("neural_atom_k1", "fourier_edge_k1"):
        result[mode] = train_arm(mode)
        torch.cuda.empty_cache()
    _atomic_json(output_root / "edge_worker.json", result)
    return result


def aggregate(output_root: Path, *, source_commit: str) -> dict:
    from .qm9_gape import sha256_file

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
        name: float(value["training"]["validation_gap_mae_eV"])
        for name, value in results.items()
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
        "format": "molgap-pcqm-gap100k-fourier-edge-screen-v1",
        "complete": True,
        "source_commit": source_commit,
        "geometry_cache_aggregate_sha256": GEOMETRY_CACHE_SHA256,
        "parent_graph_cache_aggregate_sha256": PARENT_GRAPH_CACHE_SHA256,
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
        "shadow_audit_authorized": nominated,
        "pure_2d": True,
        "model_inference_executed_by_acceptance": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "shadow_audit_read": False,
    }
    _atomic_json(output_root / "metrics.json", summary)
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
