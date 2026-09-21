"""Noisy Nodes regularisation module (Godwin et al., ICLR 2022 / GPS++).

Applies auxiliary node-level denoising regularization by:
1. In training mode: injecting zero-mean Gaussian perturbation into input atom embeddings
2. Reconstructing original atom identities (atomic numbers) via an auxiliary projection head
3. In evaluation mode: zero noise, no auxiliary head forward pass, byte-identical inference graph.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from .gptrans import OGBGPTransTiny
from .pcqm_gptrans_v4 import (
    DEVELOPMENT_ROWS,
    EMA_DECAY,
    EPOCHS,
    EXPECTED_INITIAL_MODEL_SHA256,
    EXPECTED_INITIAL_STATE_ARTIFACT_SHA256,
    GRADIENT_CLIP,
    LEARNING_RATE,
    MANIFEST_SHA256,
    MIN_LEARNING_RATE,
    PHYSICAL_BATCH,
    SAMPLE_PRESENTATIONS,
    SEED,
    TAIL_ROWS_PER_EPOCH,
    TRAIN_ROWS,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
    DeterministicEpochBatchSampler,
    ExponentialMovingAverage,
    FrozenEpochScheduler,
    _evaluate,
    _load_datasets,
    _scientific_fields,
    _target_stats,
    _training_loader,
    validate_fixed_assets,
    validate_source_archive,
)
from .screen_policy import canonical_fingerprint, validate_runtime_certificate
from .training_reproducibility import (
    assert_finite_state_dict,
    atomic_json,
    atomic_torch_save,
    build_runtime_manifest,
    capture_rng_state,
    configure_fp32_determinism,
    restore_rng_state,
    sha256_file,
)
from .v4_runtime import (
    certify_numerical_repeatability,
    load_frozen_initial_state,
    make_adamw_compat,
    model_state_sha256,
    torch_load_compat,
)


EXPECTED_NOISY_NODES_PARAMETERS = 5_277_400


@dataclass(frozen=True)
class NoisyNodesConfig:
    noise_std: float = 0.15
    loss_weight: float = 0.1
    num_classes: int = 119  # OGB atomic numbers (0..118)


class GPTransNoisyNodes(OGBGPTransTiny):
    """OGBGPTransTiny with auxiliary Noisy Nodes denoising loss."""

    def __init__(
        self,
        *args,
        noise_std: float = 0.15,
        loss_weight: float = 0.1,
        num_classes: int = 119,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.noise_std = float(noise_std)
        self.loss_weight = float(loss_weight)
        self.num_classes = int(num_classes)
        self.denoise_head = nn.Linear(self.node_channels, self.num_classes)
        nn.init.normal_(self.denoise_head.weight, std=0.02)
        nn.init.zeros_(self.denoise_head.bias)

    def _dense_inputs_with_noise(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dense_x, node_mask = to_dense_batch(x.long(), batch)
        batch_size, max_nodes = dense_x.shape[:2]
        embedded_nodes, embedded_mask = to_dense_batch(self.atom_encoder(x.long()), batch)
        if not torch.equal(node_mask, embedded_mask):
            raise RuntimeError("categorical and embedded node masks differ")

        # Noise injection (only during training and when noise_std > 0)
        if self.training and self.noise_std > 0.0:
            noise = torch.randn_like(embedded_nodes) * self.noise_std
            noise = noise.masked_fill(~node_mask.unsqueeze(-1), 0.0)
            embedded_nodes = embedded_nodes + noise

        edge_batch, edge_src, edge_dst = self._local_edges(
            edge_index, batch, int(x.shape[0])
        )
        adjacency = torch.zeros(
            (batch_size, max_nodes, max_nodes),
            dtype=torch.bool,
            device=x.device,
        )
        adjacency[edge_batch, edge_src, edge_dst] = True
        degree = adjacency.sum(-1).clamp_max(511)
        embedded_nodes = (
            embedded_nodes
            + self.in_degree_encoder(degree)
            + self.out_degree_encoder(degree)
        )
        token = self.graph_token.expand(batch_size, -1, -1)
        node = self.input_dropout(self.node_norm(torch.cat((token, embedded_nodes), dim=1)))

        pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        spatial = self._shortest_path(adjacency, pair_mask)
        pair = self.spatial_encoder(spatial).permute(0, 3, 1, 2)
        bond = self.bond_encoder(edge_attr.long())
        pair[edge_batch, :, edge_src, edge_dst] += bond
        full_pair = pair.new_zeros(
            batch_size, self.pair_channels, max_nodes + 1, max_nodes + 1
        )
        full_pair[:, :, 1:, 1:] = pair
        full_pair[:, :, 0:1, :] += self.virtual_pair
        full_pair[:, :, 1:, 0:1] += self.virtual_pair
        full_node_mask = torch.cat(
            (
                torch.ones(batch_size, 1, dtype=torch.bool, device=x.device),
                node_mask,
            ),
            dim=1,
        )
        key_padding_mask = ~full_node_mask[:, None, None, :]
        return node, full_pair, key_padding_mask, node_mask

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        random_walk_pe: torch.Tensor | None = None,
        return_aux_loss: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del random_walk_pe
        node, pair, key_padding_mask, node_mask = self._dense_inputs_with_noise(
            x, edge_index, edge_attr, batch
        )
        for block in self.blocks:
            node, pair = block(node, pair, key_padding_mask)
        graph_state = self._pool_graph_state(node, pair, node_mask)
        prediction = self.readout(graph_state)

        if not return_aux_loss or not self.training or self.loss_weight <= 0.0:
            return prediction

        atom_nodes = node[:, 1:]  # [B, N, C]
        aux_logits = self.denoise_head(atom_nodes)  # [B, N, num_classes]

        dense_targets, _ = to_dense_batch(x[:, 0].long(), batch)
        valid_logits = aux_logits[node_mask]
        valid_targets = dense_targets[node_mask]
        aux_loss = F.cross_entropy(valid_logits, valid_targets)

        return prediction, aux_loss


def _make_noisy_nodes_model(
    initial_state_path: Path | None = None,
    noise_std: float = 0.15,
    loss_weight: float = 0.1,
    readout_mode: str = "virtual",
) -> GPTransNoisyNodes:
    model = GPTransNoisyNodes(
        node_channels=256,
        pair_channels=32,
        num_layers=12,
        num_heads=8,
        shortest_path_cap=20,
        dropout=0.1,
        drop_path=0.1,
        layer_scale=1.0,
        n_targets=1,
        readout_mode=readout_mode,
        noise_std=noise_std,
        loss_weight=loss_weight,
    )
    if initial_state_path is not None:
        load_frozen_initial_state(
            model,
            initial_state_path,
            expected_file_sha256=EXPECTED_INITIAL_STATE_ARTIFACT_SHA256,
            expected_state_sha256=EXPECTED_INITIAL_MODEL_SHA256,
            expected_format="molgap-gptrans-t-seed42-initial-state-v1",
        )
    return model


def make_noisy_nodes_pair_norm_model(
    initial_state_path: Path | None = None,
    noise_std: float = 0.15,
    loss_weight: float = 0.1,
    readout_mode: str = "virtual",
) -> GPTransNoisyNodes:
    model = _make_noisy_nodes_model(
        initial_state_path=initial_state_path,
        noise_std=noise_std,
        loss_weight=loss_weight,
        readout_mode=readout_mode,
    )
    from .gptrans_variants import apply_variant
    return apply_variant(model, "pair_update_norm")


def _optimizer_step_noisy_nodes(
    model: nn.Module,
    optimizer,
    ema,
    batch,
    mean,
    std,
    *,
    check_finite: bool,
) -> tuple[float, float]:
    optimizer.zero_grad(set_to_none=True)
    if hasattr(model, "loss_weight") and getattr(model, "loss_weight", 0.0) > 0.0:
        prediction, aux_loss = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            return_aux_loss=True,
        )
        aux_val = float(aux_loss.item())
        target = (batch.y.view(-1).float() - mean) / std
        gap_loss = F.l1_loss(prediction.view(-1), target)
        total_loss = gap_loss + model.loss_weight * aux_loss
    else:
        prediction = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
        )
        aux_val = 0.0
        target = (batch.y.view(-1).float() - mean) / std
        gap_loss = F.l1_loss(prediction.view(-1), target)
        total_loss = gap_loss

    if check_finite and (not bool(torch.isfinite(gap_loss)) or not bool(torch.isfinite(total_loss))):
        raise RuntimeError("Training loss became non-finite")
    total_loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
    if check_finite and not bool(torch.isfinite(gradient_norm)):
        raise RuntimeError("Gradient norm became non-finite")
    optimizer.step()
    ema.update(model)
    return float(gap_loss.item()), aux_val


def run_training_noisy_nodes(
    *,
    dataset_root: Path,
    manifest_path: Path,
    output: Path,
    platform_id: str = "kaggle-t4",
    initial_state_path: Path | None = None,
    preflight_path: Path | None = None,
    source_archive: Path | None = None,
    source_archive_sha256: str | None = None,
    source_commit: str | None = None,
    noise_std: float = 0.15,
    loss_weight: float = 0.1,
    pair_update_norm: bool = False,
    readout_mode: str = "virtual",
) -> dict:
    determinism = configure_fp32_determinism(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("V4 training requires a visible CUDA accelerator")
    if torch.cuda.device_count() > 1:
        print(f"Notice: {torch.cuda.device_count()} GPUs visible; pinning V4 execution to device 0", flush=True)
        torch.cuda.set_device(0)
    output.mkdir(parents=True, exist_ok=True)
    completion_path = output / "completion_manifest.json"
    if completion_path.is_file():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("complete") is True:
            return completion
        raise RuntimeError("Existing completion manifest is incompatible")

    if source_archive is not None and source_archive_sha256 is not None and source_commit is not None:
        validate_source_archive(source_archive, source_archive_sha256, source_commit)
    assets = validate_fixed_assets(dataset_root, manifest_path, verify_content=True)
    if preflight_path is not None and preflight_path.is_file():
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        if preflight.get("accepted") is not True:
            raise RuntimeError("V4 preflight was not accepted")

    train_graphs, train_shards = _load_datasets(assets.train_paths)
    development_graphs, _ = _load_datasets(assets.development_paths)
    if len(train_graphs) != TRAIN_ROWS or len(development_graphs) != DEVELOPMENT_ROWS:
        raise RuntimeError("Loaded role count changed")

    mean_value, std_value = _target_stats(train_shards)
    mean = torch.tensor(mean_value, device="cuda")
    std = torch.tensor(std_value, device="cuda")

    if loss_weight > 0.0 or noise_std > 0.0:
        if pair_update_norm:
            model = make_noisy_nodes_pair_norm_model(
                initial_state_path,
                noise_std=noise_std,
                loss_weight=loss_weight,
                readout_mode=readout_mode,
            ).to("cuda")
        else:
            model = _make_noisy_nodes_model(
                initial_state_path,
                noise_std=noise_std,
                loss_weight=loss_weight,
                readout_mode=readout_mode,
            ).to("cuda")
    else:
        from .gptrans import OGBGPTransTiny
        model = OGBGPTransTiny(
            node_channels=256,
            pair_channels=32,
            num_layers=12,
            num_heads=8,
            shortest_path_cap=20,
            dropout=0.1,
            drop_path=0.1,
            layer_scale=1.0,
            n_targets=1,
            readout_mode=readout_mode,
        ).to("cuda")
        if pair_update_norm:
            from .gptrans_variants import apply_variant
            model = apply_variant(model, "pair_update_norm")

    optimizer = make_adamw_compat(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fused=False,
        foreach=False,
    )
    scheduler = FrozenEpochScheduler(optimizer)
    ema = ExponentialMovingAverage(model)

    trace: list[dict] = []
    best = float("inf")
    best_epoch = -1
    steps_per_epoch = TRAIN_ROWS // PHYSICAL_BATCH

    for epoch in range(EPOCHS):
        model.train()
        epoch_start = time.perf_counter()
        scheduler.step(epoch)
        gap_losses = []
        aux_losses = []
        for step, batch in enumerate(_training_loader(train_graphs, epoch)):
            if int(batch.num_graphs) != PHYSICAL_BATCH:
                raise RuntimeError("Optimizer received a non-128 batch")
            batch = batch.to("cuda", non_blocking=True)
            check_finite = (step % 50 == 0)
            gl, al = _optimizer_step_noisy_nodes(
                model, optimizer, ema, batch, mean, std, check_finite=check_finite
            )
            gap_losses.append(gl)
            aux_losses.append(al)

        epoch_seconds = time.perf_counter() - epoch_start
        dev_result = _evaluate(model, ema, development_graphs, mean, std)
        dev_mae = float(dev_result["mae_eV"])
        improved = dev_mae < best
        if improved:
            best = dev_mae
            best_epoch = epoch
            atomic_torch_save(output / "best_model.pt", ema.state_dict())

        trace_row = {
            "epoch": epoch,
            "optimizer_step": (epoch + 1) * steps_per_epoch,
            "sample_presentations": (epoch + 1) * TRAIN_ROWS,
            "train_normalized_gap_mae": float(np.mean(gap_losses)),
            "train_aux_ce_loss": float(np.mean(aux_losses)),
            "development_gap_mae_eV": dev_mae,
            "epoch_seconds": epoch_seconds,
            "best_so_far_eV": best,
        }
        trace.append(trace_row)
        atomic_json(output / "trace.json", {"epochs": trace})

        last_checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "ema_state": ema.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_development_mae_eV": best,
            "best_epoch": best_epoch,
        }
        atomic_torch_save(output / "last_checkpoint.pt", last_checkpoint)
        print(
            f"NoisyNodes ep{epoch:02d} train_gap={trace_row['train_normalized_gap_mae']:.6f} "
            f"aux={trace_row['train_aux_ce_loss']:.4f} dev={dev_mae:.6f}eV {epoch_seconds:.1f}s"
            f"{' *' if improved else ''}",
            flush=True,
        )

    runtime_certificate = {
        "format": "molgap-runtime-certificate-v1",
        "status": "completed",
        "total_seconds": sum(r["epoch_seconds"] for r in trace),
        "total_optimizer_steps": EPOCHS * steps_per_epoch,
        "sample_presentations": EPOCHS * TRAIN_ROWS,
        "best_development_gap_mae_eV": best,
        "best_epoch": best_epoch,
    }
    atomic_json(output / "runtime_certificate.json", runtime_certificate)

    completion = {
        "format": "molgap-pcqm-gptrans-noisy-nodes-result-v1",
        "complete": True,
        "best_development_gap_mae_eV": best,
        "best_epoch": best_epoch,
        "epochs_completed": EPOCHS,
        "total_optimizer_steps": EPOCHS * steps_per_epoch,
        "sample_presentations": EPOCHS * TRAIN_ROWS,
        "trace": trace,
        "parameters": sum(p.numel() for p in model.parameters()),
        "platform_id": platform_id,
        "source_commit": source_commit,
        "source_archive_sha256": source_archive_sha256,
        "best_model_sha256": sha256_file(output / "best_model.pt"),
    }
    atomic_json(output / "completion_manifest.json", completion)
    return completion


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="GPTrans training")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform-id", default="kaggle-t4")
    parser.add_argument("--noise-std", type=float, default=0.15)
    parser.add_argument("--loss-weight", type=float, default=0.1)
    parser.add_argument("--pair-update-norm", action="store_true")
    parser.add_argument(
        "--readout-mode",
        default="virtual",
        choices=["virtual", "dual_stream_mean", "dual_stream_attentive"],
    )
    args = parser.parse_args()
    res = run_training_noisy_nodes(
        dataset_root=args.dataset_root,
        manifest_path=args.manifest_path,
        output=args.output,
        platform_id=args.platform_id,
        noise_std=args.noise_std,
        loss_weight=args.loss_weight,
        pair_update_norm=args.pair_update_norm,
        readout_mode=args.readout_mode,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
