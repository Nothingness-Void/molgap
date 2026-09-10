"""QM9 Track-C screen for a cardinality-preserving EdgeState GPS channel."""
from __future__ import annotations

import hashlib
import json
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
MAX_HOPS = 3
CHANNEL_LAYERS = (3, 6, 9)
MIN_GAIN_VS_BASELINE_EV = 0.003
MIN_GAIN_VS_SIZE_CONTROL_EV = 0.001
MAX_PARAMETER_COUNT = 5_100_000
MAX_EPOCH_TIME_RATIO = 1.35
TASK_ID = "qm9-cardinality-channel-s42-v1"
PLATFORM_ID = "kaggle2"


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _state_sha256(model, *, exclude_prefix: str | None = None) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if exclude_prefix and name.startswith(exclude_prefix):
            continue
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class _CardinalityChannelFactory:
    """Keep torch imports remote-only while exposing one tested mechanism."""

    @staticmethod
    def make(hidden_channels: int, num_heads: int, dropout: float, mode: str):
        import torch
        import torch.nn as nn
        from torch_geometric.utils import to_dense_batch

        if mode not in {"size_control", "cpa"}:
            raise ValueError(f"Unknown cardinality mode: {mode}")
        if hidden_channels % num_heads:
            raise ValueError("hidden_channels must be divisible by num_heads")

        class CardinalityChannel(nn.Module):
            def __init__(self):
                super().__init__()
                self.mode = mode
                self.hidden_channels = hidden_channels
                self.num_heads = num_heads
                self.head_channels = hidden_channels // num_heads
                self.norm = nn.LayerNorm(hidden_channels)
                self.query_gate = nn.Linear(hidden_channels, num_heads)
                self.value = nn.Linear(hidden_channels, hidden_channels, bias=False)
                self.output = nn.Linear(hidden_channels, hidden_channels, bias=False)
                self.dropout = nn.Dropout(dropout)
                nn.init.zeros_(self.output.weight)

            @staticmethod
            def support_mask(edge_index, batch, node_count):
                if batch.ndim != 1 or batch.numel() != node_count:
                    raise ValueError("batch does not align with node rows")
                graph_count = int(batch.max().item()) + 1
                counts = torch.bincount(batch, minlength=graph_count)
                max_nodes = int(counts.max().item())
                starts = torch.cumsum(counts, dim=0) - counts
                local = torch.arange(node_count, device=batch.device) - starts[batch]
                adjacency = torch.zeros(
                    (graph_count, max_nodes, max_nodes),
                    dtype=torch.bool,
                    device=batch.device,
                )
                source, target = edge_index
                if source.numel():
                    if not torch.equal(batch[source], batch[target]):
                        raise ValueError("edge_index contains a cross-graph edge")
                    adjacency[batch[target], local[target], local[source]] = True
                valid = (
                    torch.arange(max_nodes, device=batch.device).unsqueeze(0)
                    < counts.unsqueeze(1)
                )
                pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
                identity = torch.eye(
                    max_nodes, dtype=torch.bool, device=batch.device
                ).unsqueeze(0)
                support = (identity | adjacency) & pair_valid
                frontier = adjacency
                adjacency_float = adjacency.float()
                for _ in range(2, MAX_HOPS + 1):
                    frontier = torch.bmm(frontier.float(), adjacency_float) > 0
                    support = support | (frontier & pair_valid)
                return support, valid, local

            def forward(self, hidden, edge_index, batch):
                support, valid, local = self.support_mask(
                    edge_index, batch, int(hidden.shape[0])
                )
                normalized = self.norm(hidden)
                values = self.value(normalized).view(
                    hidden.shape[0], self.num_heads, self.head_channels
                )
                dense_values, dense_mask = to_dense_batch(values, batch)
                if not torch.equal(dense_mask, valid):
                    raise RuntimeError("dense node mask changed")
                support_count = support.sum(dim=-1, keepdim=True).to(hidden.dtype)
                if self.mode == "cpa":
                    aggregate = torch.einsum(
                        "bij,bjhd->bihd", support.to(hidden.dtype), dense_values
                    )
                else:
                    aggregate = dense_values * torch.log1p(
                        support_count
                    ).unsqueeze(-1)
                gate = torch.sigmoid(self.query_gate(normalized)).unsqueeze(-1)
                dense_gate, gate_mask = to_dense_batch(gate, batch)
                if not torch.equal(gate_mask, valid):
                    raise RuntimeError("dense gate mask changed")
                aggregate = aggregate * dense_gate
                flat = aggregate[batch, local].reshape(hidden.shape[0], hidden_channels)
                return hidden + self.dropout(self.output(flat))

        return CardinalityChannel()


def make_encoder(mode: str):
    """Build the frozen EdgeState GPS9 or one cardinality-channel arm."""
    if mode == "baseline":
        from .qm9_local_hierarchy import make_encoder as make_baseline

        return make_baseline()

    from .pcqm_gap_architecture import OGBEdgeStateStructuralGPSWrapper

    class CardinalityEdgeStateGPS(OGBEdgeStateStructuralGPSWrapper):
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
            self.cardinality_channel = _CardinalityChannelFactory.make(
                hidden_channels=192,
                num_heads=4,
                dropout=0.05,
                mode=mode,
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
            for layer, (edge_update, conv) in enumerate(
                zip(self.edge_updates, self.convs), start=1
            ):
                edge_state = edge_update(h, edge_index, edge_state)
                h = conv(h, edge_index, batch, edge_attr=edge_state)
                if layer in CHANNEL_LAYERS:
                    h = self.cardinality_channel(h, edge_index, batch)
            return self._pool(h, batch)

    return CardinalityEdgeStateGPS()


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
    baseline = make_encoder("baseline").to("cuda").eval()
    baseline_shared_sha = _state_sha256(baseline)
    baseline_output = forward_gap(baseline, batch, augmented=False)
    arms = {}
    candidate_initial_sha = None
    for mode in ("size_control", "cpa"):
        set_seed(SEED)
        model = make_encoder(mode).to("cuda").eval()
        shared_sha = _state_sha256(model, exclude_prefix="cardinality_channel.")
        initial_sha = _state_sha256(model)
        if shared_sha != baseline_shared_sha:
            raise RuntimeError(f"Shared initialization changed for {mode}")
        if candidate_initial_sha is None:
            candidate_initial_sha = initial_sha
        elif initial_sha != candidate_initial_sha:
            raise RuntimeError("Candidate initialization changed between controls")
        output = forward_gap(model, batch, augmented=False)
        # Separate CUDA forwards can differ by roundoff even when the added
        # branch returns an exact zero.  Bitwise equality therefore tests the
        # execution schedule, not the intended zero-return model invariant.
        zero_return_atol = 1e-7
        zero_return_rtol = 1e-6
        max_abs_difference = float((output - baseline_output).abs().max().item())
        if not torch.allclose(
            output,
            baseline_output,
            atol=zero_return_atol,
            rtol=zero_return_rtol,
        ):
            raise RuntimeError(
                f"Zero-return identity failed for {mode}: "
                f"max_abs_difference={max_abs_difference}"
            )
        model.train()
        loss = forward_gap(model, batch, augmented=False).square().mean()
        loss.backward()
        output_grad = model.cardinality_channel.output.weight.grad
        finite = bool(torch.isfinite(loss).item()) and output_grad is not None and bool(
            torch.isfinite(output_grad).all().item()
        ) and float(output_grad.abs().sum()) > 0
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if not finite or parameter_count > MAX_PARAMETER_COUNT:
            raise RuntimeError(f"Candidate preflight failed for {mode}")
        arms[mode] = {
            "parameter_count": parameter_count,
            "shared_state_sha256": shared_sha,
            "initial_state_sha256": initial_sha,
            "zero_return_identity": True,
            "zero_return_max_abs_difference": max_abs_difference,
            "zero_return_atol": zero_return_atol,
            "zero_return_rtol": zero_return_rtol,
            "finite_output_gradient": finite,
        }
        del model
    chain_edges = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]],
        dtype=torch.long,
        device="cuda",
    )
    chain_batch = torch.zeros(5, dtype=torch.long, device="cuda")
    support, _, _ = _CardinalityChannelFactory.make(192, 4, 0.0, "cpa").to(
        "cuda"
    ).support_mask(chain_edges, chain_batch, 5)
    support_check = bool(support[0, 0, :4].all()) and not bool(support[0, 0, 4])
    support_check = support_check and bool(support[0, 2].all())
    if not support_check:
        raise RuntimeError("Exact K<=3 support preflight failed")
    result = {
        "format": "molgap-qm9-cardinality-channel-preflight-v1",
        "accepted": True,
        "source_commit": source_commit,
        "gpu": torch.cuda.get_device_name(0),
        "baseline_parameter_count": sum(p.numel() for p in baseline.parameters()),
        "baseline_state_sha256": baseline_shared_sha,
        "candidate_arms": arms,
        "max_hops": MAX_HOPS,
        "channel_layers": list(CHANNEL_LAYERS),
        "chain_support_exact": support_check,
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

    if role not in {"baseline", "candidates"}:
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
            "baseline": {
                "training": train_arm("baseline"),
                "contract": _arm_contract(
                    "baseline",
                    accelerator=accelerator,
                    split_fingerprint=manifest["split_fingerprint"],
                ),
            }
        }
        _atomic_json(output_root / "baseline_worker.json", result)
        return result

    preflight = _preflight(roles, output_root, source_commit=source_commit)
    results = {}
    for mode in ("size_control", "cpa"):
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
    _atomic_json(output_root / "candidate_worker.json", result)
    return result


def aggregate(output_root: Path, *, source_commit: str, cache_sha256: str) -> dict:
    baseline_worker = json.loads(
        (output_root / "baseline_worker.json").read_text(encoding="utf-8")
    )
    candidate_worker = json.loads(
        (output_root / "candidate_worker.json").read_text(encoding="utf-8")
    )
    results = {
        "baseline": baseline_worker["baseline"],
        "size_control": candidate_worker["size_control"],
        "cpa": candidate_worker["cpa"],
    }
    comparability = validate_paired_screen_contract(
        [results[name]["contract"] for name in results]
    )
    mae = {
        name: float(result["training"]["validation_gap_mae_eV"])
        for name, result in results.items()
    }
    gain_baseline = mae["baseline"] - mae["cpa"]
    gain_size = mae["size_control"] - mae["cpa"]
    epoch_ratio = (
        results["cpa"]["training"]["mean_epoch_seconds"]
        / results["baseline"]["training"]["mean_epoch_seconds"]
    )
    nominated = (
        gain_baseline >= MIN_GAIN_VS_BASELINE_EV
        and gain_size >= MIN_GAIN_VS_SIZE_CONTROL_EV
        and epoch_ratio <= MAX_EPOCH_TIME_RATIO
        and results["cpa"]["training"]["memory_reserve_fraction"] >= 0.15
    )
    summary = {
        "format": "molgap-qm9-cardinality-channel-screen-v1",
        "complete": True,
        "source_commit": source_commit,
        "cache_aggregate_sha256": cache_sha256,
        "comparability": comparability,
        "preflight": candidate_worker["preflight"],
        "results": results,
        "validation_gap_mae_eV": mae,
        "cpa_gain_vs_baseline_eV": gain_baseline,
        "cpa_gain_vs_size_control_eV": gain_size,
        "required_gain_vs_baseline_eV": MIN_GAIN_VS_BASELINE_EV,
        "required_gain_vs_size_control_eV": MIN_GAIN_VS_SIZE_CONTROL_EV,
        "cpa_epoch_time_ratio": epoch_ratio,
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
